"""Voices with Kokoro-82M (runs locally) + per-character pitch, mixed with sound effects."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from .ffmpeg_utils import FFMPEG
from .script_parser import Scene, split_sentences

QWEN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts_venv", "Scripts", "python.exe")
CB_PY = os.path.join(os.path.dirname(QWEN_PY), "..", "..", "cb_venv", "Scripts", "python.exe")
KOKORO_RATE = 24000
SAMPLE_RATE = 48000  # mix rate

VOICES = {
    "George (British male, deep)": "bm_george",
    "Fable (British male, storyteller)": "bm_fable",
    "Lewis (British male)": "bm_lewis",
    "Daniel (British male)": "bm_daniel",
    "Emma (British female)": "bf_emma",
    "Isabella (British female)": "bf_isabella",
    "Michael (American male)": "am_michael",
    "Fenrir (American male, intense)": "am_fenrir",
    "Puck (American male)": "am_puck",
    "Heart (American female, warm)": "af_heart",
    "Bella (American female)": "af_bella",
    "Nicole (American female, soft)": "af_nicole",
}

# Used for speakers not defined with [voice Name: ...]: e.g. "Ambulance:", "Car:".
DEFAULT_CHARACTER = {"voice": "am_puck", "pitch": 1.08, "speed": 0.95}


def load_audio(path: str, sr: int = SAMPLE_RATE, filters: str = "") -> np.ndarray:
    """Decode any audio file to mono float32 at `sr`, optionally through an ffmpeg filter chain."""
    af = ",".join(x for x in [filters, f"aresample={sr}"] if x)
    proc = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-af", af, "-ac", "1", "-f", "f32le", "-"],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore")[-1500:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


class Narrator:
    def __init__(self, device: str | None = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipelines = {}
        self._designed = {}  # (voice name, sentence) -> wav from Qwen3-TTS

    def _pipeline(self, voice: str):
        from kokoro import KPipeline
        lang = voice[0]  # 'a' American, 'b' British
        if lang not in self._pipelines:
            self._pipelines[lang] = KPipeline(lang_code=lang, device=self.device, repo_id="hexgrad/Kokoro-82M")
        return self._pipelines[lang]

    def _kokoro(self, text: str, voice: str, speed: float) -> np.ndarray:
        pipe = self._pipeline(voice)
        chunks = [audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
                  for _, _, audio in pipe(text, voice=voice, speed=speed, split_pattern=None)
                  if audio is not None]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

    def speak(self, text: str, voice: str, speed: float = 1.0, pitch: float = 1.0) -> np.ndarray:
        """Speech at SAMPLE_RATE. pitch > 1 raises the voice (1.3 ~ toddler, 1.45 ~ baby)."""
        raw = self._kokoro(text, voice, speed)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp = fh.name
        try:
            sf.write(tmp, raw, KOKORO_RATE)
            filt = f"rubberband=pitch={pitch:.3f}:formant=shifted:transients=smooth" if abs(pitch - 1) > 0.01 else ""
            return load_audio(tmp, SAMPLE_RATE, filt)
        finally:
            os.remove(tmp)

    def _character(self, speaker: str, cast: dict, narrator: dict) -> list[dict]:
        if speaker == "Narrator":
            return [{**narrator, "name": "narrator"}]
        names = [n.strip().lower() for n in speaker.replace(" and ", "&").split("&")]
        return [{**cast.get(n, cast.get("default", DEFAULT_CHARACTER)),
                 "name": n if n in cast else "default"} for n in names]

    def _say(self, sentence: str, c: dict, emotion: str = "") -> np.ndarray:
        if c.get("design") or c.get("ref"):
            path = self._designed[(c["name"], sentence, emotion)]
            speed = c.get("speed", 1.0)
            return load_audio(path, SAMPLE_RATE, f"atempo={speed:.3f}" if abs(speed - 1) > 0.01 else "")
        return self.speak(sentence, c["voice"], c.get("speed", 1.0), c.get("pitch", 1.0))

    def prerender_designed(self, scenes: list, cast: dict, narrator: dict, cache_dir: str, log=print,
                           engine: str = "qwen") -> None:
        """Render every line spoken by a designed voice in one batch (models load once).

        engine "qwen": Qwen3-TTS designs the voice and speaks the lines.
        engine "chatterbox": Qwen3-TTS designs a reference clip per voice; Chatterbox speaks the lines
        by cloning it, more expressively when the line has an emotion, e.g. Nisa (giggling): "..."."""
        voices, lines = {}, []
        for sc in scenes:
            for ev in sc.events:
                if ev.kind != "speech":
                    continue
                for c in self._character(ev.speaker, cast, narrator):
                    if not (c.get("design") or c.get("ref")):
                        continue
                    voices[c["name"]] = {"design": c.get("design", ""), "ref": c.get("ref", ""),
                                         "seed": int(c.get("seed", 0))}
                    for sentence in split_sentences(ev.text):
                        # each line gets its own seed so repeated lines ("No Baba!" twice) aren't identical clones
                        seed = int(hashlib.sha1(f"{sc.index}|{sentence}|{ev.emotion}".encode()).hexdigest()[:8], 16)
                        tag = [c.get("ref") or c["design"], sentence] + ([engine, ev.emotion, seed] if engine != "qwen" else [])
                        key = hashlib.sha1(json.dumps(tag).encode()).hexdigest()[:16]
                        out = os.path.join(cache_dir, f"line_{key}.wav")
                        self._designed[(c["name"], sentence, ev.emotion)] = out
                        expressive = bool(ev.emotion)
                        lines.append({"voice": c["name"], "text": sentence, "out": out, "seed": seed,
                                      "exaggeration": 0.8 if expressive else 0.55,
                                      "cfg_weight": 0.35 if expressive else 0.5})
        if not lines or all(os.path.isfile(ln["out"]) for ln in lines):
            return
        os.makedirs(cache_dir, exist_ok=True)
        job = os.path.join(cache_dir, "job.json")
        qwen_lines = lines if engine == "qwen" else []  # chatterbox: only design the reference clips
        designed = {n: v for n, v in voices.items() if not v["ref"]}
        refs = {n: v["ref"] for n, v in voices.items() if v["ref"]}
        if engine == "qwen":  # Qwen speaks everything: designed voices and cloned recordings (with transcripts)
            qv = dict(designed)
            for n, path in refs.items():
                txt = os.path.splitext(path)[0] + ".txt"
                qv[n] = {"ref": path, "ref_text": open(txt, encoding="utf-8").read().strip()
                         if os.path.isfile(txt) else ""}
            with open(job, "w", encoding="utf-8") as fh:
                json.dump({"cache_dir": cache_dir, "voices": qv, "lines": lines}, fh)
            log(f"  Qwen3-TTS: {len(qv)} voice(s), {len(lines)} line(s)")
            self._run_voice_tool(QWEN_PY, "qwen_voices.py", job, "QWEN_DONE")
            return
        if designed:
            with open(job, "w", encoding="utf-8") as fh:
                json.dump({"cache_dir": cache_dir, "voices": designed, "lines": qwen_lines}, fh)
            log(f"  Qwen3-TTS: {len(designed)} designed voice(s), {len(qwen_lines)} line(s)")
            self._run_voice_tool(QWEN_PY, "qwen_voices.py", job, "QWEN_DONE")
            refs.update(json.load(open(job + ".refs.json", encoding="utf-8")))
        if engine == "chatterbox":
            with open(job, "w", encoding="utf-8") as fh:
                json.dump({"voices": {n: {"ref": p} for n, p in refs.items()}, "lines": lines}, fh)
            log(f"  Chatterbox: {len(lines)} line(s)")
            self._run_voice_tool(CB_PY, "cb_voices.py", job, "CB_DONE")

    @staticmethod
    def _run_voice_tool(python: str, script: str, job: str, done: str) -> None:
        proc = subprocess.run([python, os.path.join(os.path.dirname(__file__), script), job],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or done not in proc.stdout:
            raise RuntimeError(f"{script} failed:\n" + (proc.stderr or proc.stdout)[-3000:])

    def narrate_scene(self, scene: Scene, out_path: str, cast: dict, narrator: dict,
                      sfx_files: dict, lead_in: float = 0.4, gap: float = 0.35, tail: float = 0.8,
                      sfx_gain: float = 0.55, sfx_lead: float = 1.1) -> None:
        """Lay out the scene's speech and sound effects on one timeline.

        Each line is rendered separately so subtitles get exact timings. A sound effect starts
        where its tag appears; the next line starts `sfx_lead` seconds into it (spoken over the tail).
        """
        segments = []  # (start_seconds, samples, gain)
        t = lead_in
        end = t
        scene.sentence_times = []
        for ev in scene.events:
            if ev.kind == "sfx":
                path = sfx_files.get((ev.text, ev.seconds))
                if not path:
                    continue
                audio = load_audio(path)
                segments.append((t, audio, sfx_gain))
                end = max(end, t + len(audio) / SAMPLE_RATE)
                t += min(sfx_lead, len(audio) / SAMPLE_RATE)
                continue
            for sentence in split_sentences(ev.text):
                voices = [self._say(sentence, c, ev.emotion) for c in self._character(ev.speaker, cast, narrator)]
                n = max(len(v) for v in voices)
                audio = sum(np.pad(v, (0, n - len(v))) for v in voices) / (len(voices) ** 0.5)
                dur = n / SAMPLE_RATE
                segments.append((t, audio, 1.0))
                scene.sentence_times.append((t, t + dur, sentence))
                t += dur + gap
                end = max(end, t)

        scene.duration = max(end + tail, scene.min_duration, 3.0)
        wav = np.zeros(int(scene.duration * SAMPLE_RATE) + 1, dtype=np.float32)
        for start, audio, gain in segments:
            i = int(start * SAMPLE_RATE)
            seg = audio[: len(wav) - i] * gain
            wav[i:i + len(seg)] += seg
        peak = float(np.abs(wav).max() or 1)
        if peak > 0.98:
            wav *= 0.98 / peak
        sf.write(out_path, wav, SAMPLE_RATE)
        scene.audio_path = out_path

    def release(self):
        self._pipelines.clear()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
