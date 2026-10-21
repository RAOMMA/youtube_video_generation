"""End-to-end: script + images -> narrated, graded, letterboxed cinematic mp4."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf

from . import kenburns
from .comfy_client import ComfyClient, missing_ltx_models, missing_models, missing_sfx_models
from .ffmpeg_utils import probe_duration, run as ffmpeg
from .script_parser import Scene, parse_cast, parse_script
from .tts import SAMPLE_RATE, Narrator

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(APP_DIR, "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "_ai_cache")
SFX_CACHE_DIR = os.path.join(OUTPUT_DIR, "_sfx_cache")
VOICE_CACHE_DIR = os.path.join(OUTPUT_DIR, "_voice_cache")

FPS = 24
SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920)}

GRADES = {
    "Teal & Orange": "colorbalance=rs=-0.05:bs=0.06:rh=0.06:gh=0.01:bh=-0.06,eq=contrast=1.08:saturation=1.05",
    "Warm Golden": "colorbalance=rm=0.05:gm=0.01:bm=-0.05,eq=contrast=1.05:saturation=1.1",
    "Cold Blue": "colorbalance=rs=-0.02:bs=0.05:rm=-0.04:bm=0.06,eq=contrast=1.06:saturation=0.85",
    "Noir (B&W)": "hue=s=0,eq=contrast=1.25:brightness=-0.02",
    "Natural": "",
}


@dataclass
class Settings:
    voice: str = "bm_george"
    voice_design: str = ""              # Qwen3-TTS description for the narrator (empty = Kokoro voice)
    tts_engine: str = "qwen"            # designed voices: "qwen" | "chatterbox" (Qwen-designed ref, Chatterbox performs)
    speed: float = 0.95
    motion_mode: str = "kenburns"       # kenburns | ai
    style_prompt: str = "cinematic, film grain, dramatic lighting, shallow depth of field"
    ai_width: int = 832
    ai_height: int = 480
    ai_frames: int = 49
    ai_steps: int = 20
    ai_cfg: float = 5.0
    ai_max_segments: int = 2
    ltx_width: int = 1280
    ltx_height: int = 704
    aspect: str = "16:9"                # "16:9" landscape | "9:16" vertical reel
    seed: int = -1
    kb_strength: float = 1.0
    crossfade: float = 0.8
    grade: str = "Teal & Orange"
    letterbox: bool = True
    vignette: bool = True
    grain: bool = True
    subtitles: bool = True
    music_path: str = ""
    music_volume: float = 0.25


class Pipeline:
    def __init__(self, log=print, progress=None):
        self.log = log
        self.progress = progress or (lambda frac, desc="": None)
        self.comfy: ComfyClient | None = None
        self.W, self.H = SIZES["16:9"]

    # ------------------------------------------------------------------ main
    def run(self, script: str, images: list[str], s: Settings) -> str:
        self.W, self.H = SIZES.get(s.aspect, SIZES["16:9"])
        job = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(job, exist_ok=True)
        scenes = parse_script(script, len(images))
        cast = parse_cast(script)
        self.log(f"{len(scenes)} scenes, {len(images)} images, cast: {', '.join(cast) or 'narrator only'} -> {job}")

        # 1. sound effects (AI, cached)
        sfx_files = self._sound_effects(scenes, s)

        # 2. voices + sfx timeline (sets scene lengths)
        narrator = Narrator()
        voice_cfg = {"voice": s.voice, "speed": s.speed, "pitch": 1.0, "design": s.voice_design}
        narrator.prerender_designed(scenes, cast, voice_cfg, VOICE_CACHE_DIR, self.log, engine=s.tts_engine)
        for sc in scenes:
            self.progress(0.08 + 0.07 * sc.index / len(scenes), f"Voicing scene {sc.index + 1}")
            narrator.narrate_scene(sc, os.path.join(job, f"voice_{sc.index:03d}.wav"), cast, voice_cfg, sfx_files)
            self.log(f"  scene {sc.index + 1}: {sc.duration:.1f}s")
        narrator.release()

        # 2. visuals
        clips = []
        for sc in scenes:
            last = sc.index == len(scenes) - 1
            length = sc.duration + (0 if last else s.crossfade)
            mode = sc.mode or s.motion_mode
            base = 0.15 + 0.7 * sc.index / len(scenes)
            span = 0.7 / len(scenes)
            out = os.path.join(job, f"clip_{sc.index:03d}.mp4")
            img = images[sc.image_index]
            if mode == "ltx":
                try:
                    self._ltx_clip(sc, img, out, length, s, base)
                    clips.append(out)
                    continue
                except Exception as e:
                    self.log(f"  LTX failed for scene {sc.index + 1} ({e}); using Ken Burns")
            if mode == "ai":
                try:
                    self._ai_clip(sc, img, out, length, s, base, span)
                    clips.append(out)
                    continue
                except Exception as e:  # keep the film going even if one AI shot fails
                    self.log(f"  AI motion failed for scene {sc.index + 1} ({e}); using Ken Burns")
            self.progress(base, f"Scene {sc.index + 1}: Ken Burns ({sc.camera})")
            kenburns.render(img, out, length, sc.camera, self.W, self.H, FPS, s.kb_strength)
            clips.append(out)

        if self.comfy:
            self.comfy = None  # leave server running for the next job (faster), it's cheap when idle

        # 3. audio + subtitles + final cut
        self.progress(0.88, "Mixing narration")
        narration = self._build_narration(scenes, job)
        srt = self._build_srt(scenes, job)
        self.progress(0.92, "Final cinematic render")
        final = os.path.join(job, "final.mp4")
        self._assemble(clips, scenes, narration, srt, final, s)
        with open(os.path.join(job, "scenes.json"), "w", encoding="utf-8") as fh:
            json.dump({"settings": asdict(s), "scenes": [asdict(x) for x in scenes]}, fh, indent=2)
        self.progress(1.0, "Done")
        self.log(f"Done: {final}")
        return final

    # --------------------------------------------------------- sound effects
    def _sound_effects(self, scenes, s: Settings) -> dict:
        wanted = list(dict.fromkeys((e.text, e.seconds) for sc in scenes for e in sc.events if e.kind == "sfx"))
        if not wanted:
            return {}
        missing = missing_sfx_models()
        if missing:
            self.log("  sound effects skipped, models missing: " + ", ".join(missing))
            return {}
        os.makedirs(SFX_CACHE_DIR, exist_ok=True)
        files = {}
        for i, (prompt, secs) in enumerate(wanted):
            key = hashlib.sha1(json.dumps([prompt, secs, s.seed]).encode()).hexdigest()[:16]
            path = os.path.join(SFX_CACHE_DIR, key + ".wav")
            if not os.path.isfile(path):
                if self.comfy is None:
                    self.comfy = ComfyClient(self.log)
                self.log(f"  sfx {i + 1}/{len(wanted)}: {prompt} ({secs:g}s)")
                self.progress(0.01 + 0.07 * i / len(wanted), f"Sound effect: {prompt}")
                try:
                    self.comfy.sound_effect(prompt, secs, path, seed=None if s.seed < 0 else s.seed + i)
                except Exception as e:
                    self.log(f"  sfx failed ({e}); continuing without it")
                    continue
            files[(prompt, secs)] = path
        return files

    # ------------------------------------------------------- LTX talking shot
    def _ltx_clip(self, sc: Scene, image: str, out: str, length: float, s: Settings, base: float):
        missing = missing_ltx_models()
        if missing:
            raise RuntimeError("LTX models missing: " + ", ".join(missing))
        if self.comfy is None:
            self.comfy = ComfyClient(self.log)
        # Only the speaking order, never the quoted words, and never mention text/subtitles at all
        # (even negated): either makes LTX draw garbled on-screen captions. Lip-sync comes from the audio.
        speakers = [e.speaker for e in sc.events if e.kind == "speech" and e.speaker != "Narrator"]
        lines = ("Speaking order: " + ", then ".join(speakers) + "."
                 if speakers else "")
        prompt = " ".join(x for x in [s.style_prompt + ".", sc.motion + "." if sc.motion else "", lines] if x)
        with open(image, "rb") as fh:
            img_hash = hashlib.sha1(fh.read()).hexdigest()
        with open(sc.audio_path, "rb") as fh:
            aud_hash = hashlib.sha1(fh.read()).hexdigest()
        key = "ltx_" + hashlib.sha1(json.dumps([img_hash, aud_hash, prompt, s.ltx_width, s.ltx_height,
                                                round(length, 2), s.seed]).encode()).hexdigest()[:16]
        cached = os.path.join(CACHE_DIR, key + ".mp4")
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.isfile(cached):
            self.log(f"  scene {sc.index + 1}: LTX shot from cache")
        else:
            self.log(f"  scene {sc.index + 1}: LTX talking shot {s.ltx_width}x{s.ltx_height}, {length:.1f}s")
            self.log(f"    prompt: {prompt}")

            def prog(v, m):
                self.progress(base, f"Scene {sc.index + 1}: LTX step {v}/{m}")
            self.comfy.ltx_talking(image, sc.audio_path, cached, prompt, seconds=length,
                                   width=s.ltx_width, height=s.ltx_height, fps=FPS,
                                   seed=None if s.seed < 0 else s.seed + sc.index, progress=prog)
        keep = 1 - min(sc.crop_bottom, 30) / 100
        crop = f"crop=iw*{keep:.4f}:ih*{keep:.4f}:(iw-ow)/2:0," if keep < 1 else ""
        ffmpeg(["-i", cached, "-vf", f"{crop}scale={self.W}:{self.H}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={self.W}:{self.H},fps={FPS},tpad=stop_mode=clone:stop_duration={length:.3f},"
                f"trim=duration={length:.3f},setsar=1,format=yuv420p",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", out])

    # ------------------------------------------------------------- AI motion
    def _ai_clip(self, sc: Scene, image: str, out: str, length: float, s: Settings, base, span):
        missing = missing_models()
        if missing:
            raise RuntimeError("models still downloading: " + ", ".join(missing))
        if self.comfy is None:
            self.comfy = ComfyClient(self.log)
        motion = sc.motion or "slow cinematic camera movement, subtle natural motion"
        prompt = f"{motion}. {s.style_prompt}".strip()
        seg_seconds = s.ai_frames / FPS
        # Generate enough segments to cover the shot (each continues from the previous last frame),
        # then slow-mo / hold to fill whatever remains.
        n_seg = max(1, min(s.ai_max_segments, int(np.ceil(length / (seg_seconds * 1.35)))))
        segs, src = [], image
        for k in range(n_seg):
            with open(src, "rb") as fh:
                img_hash = hashlib.sha1(fh.read()).hexdigest()
            key = hashlib.sha1(json.dumps([img_hash, prompt, s.ai_width, s.ai_height, s.ai_frames,
                                           s.ai_steps, s.ai_cfg, s.seed]).encode()).hexdigest()[:16]
            cached = os.path.join(CACHE_DIR, key + ".mp4")
            os.makedirs(CACHE_DIR, exist_ok=True)
            if not os.path.isfile(cached):
                self.log(f"  scene {sc.index + 1}: AI segment {k + 1}/{n_seg} ({s.ai_width}x{s.ai_height}, {s.ai_frames} frames)")

                def prog(v, m, k=k):
                    self.progress(base + span * (k + v / m) / n_seg,
                                  f"Scene {sc.index + 1}: AI motion {k + 1}/{n_seg} - step {v}/{m}")
                seed = None if s.seed < 0 else s.seed + sc.index * 17 + k
                self.comfy.animate(src, cached, prompt, width=s.ai_width, height=s.ai_height,
                                   frames=s.ai_frames, steps=s.ai_steps, cfg=s.ai_cfg, seed=seed,
                                   fps=FPS, progress=prog)
                frames = sorted(os.listdir(cached + "_frames"))
                shutil.copy(os.path.join(cached + "_frames", frames[-1]), cached + "_last.png")
                shutil.rmtree(cached + "_frames", ignore_errors=True)
            else:
                self.log(f"  scene {sc.index + 1}: AI segment {k + 1} from cache")
            segs.append(cached)
            src = cached + "_last.png"

        raw = out + "_raw.mp4"
        if len(segs) == 1:
            shutil.copy(segs[0], raw)
        else:
            inputs = sum([["-i", p] for p in segs], [])
            # drop the duplicated first frame of each continuation segment
            parts = "".join(f"[{i}:v]{'trim=start_frame=1,setpts=PTS-STARTPTS,' if i else ''}null[s{i}];"
                            for i in range(len(segs)))
            cat = "".join(f"[s{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[v]"
            ffmpeg([*inputs, "-filter_complex", parts + cat, "-map", "[v]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", raw])

        have = probe_duration(raw)
        factor = min(max(length / have, 1.0), 1.5)
        vf = []
        if factor > 1.02:  # smooth slow-motion via motion interpolation
            vf += [f"minterpolate=fps={FPS * factor:.3f}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                   f"setpts={factor:.4f}*PTS"]
        vf += [f"scale={self.W}:{self.H}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={self.W}:{self.H}",
               f"fps={FPS}", f"tpad=stop_mode=clone:stop_duration={length:.3f}",
               f"trim=duration={length:.3f}", "setsar=1", "format=yuv420p"]
        ffmpeg(["-i", raw, "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", out])
        os.remove(raw)

    # ------------------------------------------------------------ audio/subs
    @staticmethod
    def _starts(scenes):
        t, starts = 0.0, []
        for sc in scenes:
            starts.append(t)
            t += sc.duration
        return starts, t

    def _build_narration(self, scenes, job) -> str:
        starts, total = self._starts(scenes)
        track = np.zeros(int(total * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)
        for sc, st in zip(scenes, starts):
            if sc.audio_path:
                a, _ = sf.read(sc.audio_path, dtype="float32")
                i = int(st * SAMPLE_RATE)
                track[i:i + len(a)] += a[: len(track) - i]
        path = os.path.join(job, "narration.wav")
        sf.write(path, track, SAMPLE_RATE)
        return path

    def _build_srt(self, scenes, job) -> str:
        def ts(x):
            ms = int(round(x * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        starts, _ = self._starts(scenes)
        lines, n = [], 1
        for sc, st in zip(scenes, starts):
            for a, b, text in sc.sentence_times:
                lines += [str(n), f"{ts(st + a)} --> {ts(st + b + 0.15)}", text, ""]
                n += 1
        path = os.path.join(job, "subtitles.srt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return path

    # ----------------------------------------------------------------- final
    def _assemble(self, clips, scenes, narration, srt, final, s: Settings):
        job = os.path.dirname(final)
        _, total = self._starts(scenes)
        n = len(clips)
        args, fc = [], []
        for c in clips:
            args += ["-i", os.path.basename(c)]
        for i in range(n):
            fc.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={FPS},format=yuv420p[v{i}]")
        last, offset = "v0", 0.0
        for i in range(1, n):
            offset += scenes[i - 1].duration
            fc.append(f"[{last}][v{i}]xfade=transition=fade:duration={s.crossfade:.3f}:offset={offset:.3f}[x{i}]")
            last = f"x{i}"

        look = []
        if GRADES.get(s.grade):
            look.append(GRADES[s.grade])
        if s.vignette:
            look.append("vignette=angle=PI/5")
        if s.grain:
            look.append("noise=alls=5:allf=t+u")
        bar = round((self.H - self.W / 2.39) / 2) if s.letterbox and self.W > self.H else 0
        if bar:
            look.append(f"drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill,"
                        f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill")
        if s.subtitles and os.path.getsize(srt) > 0:
            if self.H > self.W:  # vertical reel: bold captions above the app UI safe zone
                style = ("FontName=Arial Black,FontSize=10,Bold=1,PrimaryColour=&H00FFFFFF,"
                         "OutlineColour=&H00000000,Outline=2,Shadow=0,MarginV=70,MarginL=20,MarginR=20")
            elif bar:  # elegant serif sitting inside the bottom bar
                style = ("FontName=Georgia,FontSize=13,PrimaryColour=&H00F0F0F0,OutlineColour=&H80000000,"
                         "Outline=1,Shadow=0,MarginV=16")
            else:    # bold, outlined, friendly over the picture
                style = ("FontName=Arial Rounded MT Bold,FontSize=17,Bold=1,PrimaryColour=&H00FFFFFF,"
                         "OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=22")
            look.append(f"subtitles={os.path.basename(srt)}:force_style='{style}'")
        look.append(f"fade=t=in:st=0:d=1.0,fade=t=out:st={max(0, total - 1.5):.3f}:d=1.5")
        fc.append(f"[{last}]{','.join(look)}[vout]")

        args += ["-i", os.path.basename(narration)]
        na = n
        if s.music_path and os.path.isfile(s.music_path):
            args += ["-stream_loop", "-1", "-i", s.music_path]
            fc.append(f"[{na}:a]aresample=48000,asplit=2[n1][n2]")
            fc.append(f"[{na + 1}:a]aresample=48000,atrim=duration={total:.3f},volume={s.music_volume:.3f},"
                      f"afade=t=in:d=2,afade=t=out:st={max(0, total - 3):.3f}:d=3[m]")
            fc.append("[m][n1]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=600[duck]")
            fc.append("[n2][duck]amix=inputs=2:duration=first:normalize=0,"
                      "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")
        else:
            fc.append(f"[{na}:a]aresample=48000,atrim=duration={total:.3f},"
                      f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")

        ffmpeg([*args, "-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
                "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                os.path.basename(final)], cwd=job)
