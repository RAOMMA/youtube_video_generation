"""Runs inside tts_venv (qwen-tts). Designs each character's voice once from a text description,
then speaks every line by cloning that designed voice, so a character sounds the same everywhere.

Usage: tts_venv\\Scripts\\python cineforge\\qwen_voices.py job.json
job.json: {"cache_dir": "...", "language": "English",
           "voices": {"nisa": {"design": "a cheerful two-year-old girl ...", "ref_text": "..."}},
           "lines": [{"voice": "nisa", "text": "This is a taxi!", "out": "C:/.../x.wav"}]}
"""
import gc
import glob
import hashlib
import json
import os
import sys

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from asr_check import check, score

DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REF_TEXT = ("Hello! It is so nice to meet you. Today I want to tell you a little story, "
                    "so come and sit with me, and let's begin.")


def load(name):
    return Qwen3TTSModel.from_pretrained(name, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                                         dtype=torch.bfloat16, attn_implementation="sdpa")


def main(job_path):
    job = json.load(open(job_path, encoding="utf-8"))
    cache, lang = job["cache_dir"], job.get("language", "English")
    os.makedirs(cache, exist_ok=True)

    # 1. design a reference clip per voice (cached by description)
    refs, todo = {}, []
    for name, v in job["voices"].items():
        if v.get("ref"):  # a real recording: clone it directly (with its transcript when available)
            refs[name] = (v["ref"], v.get("ref_text") or None)
            continue
        ref_text = v.get("ref_text") or DEFAULT_REF_TEXT
        key = hashlib.sha1(json.dumps([v["design"], ref_text, v.get("seed", 0)]).encode()).hexdigest()[:16]
        path = os.path.join(cache, f"ref_{key}.wav")
        refs[name] = (path, ref_text)
        if not os.path.isfile(path):
            todo.append((name, v, ref_text, path))
    if todo:
        model = load(DESIGN_MODEL)
        for name, v, ref_text, path in todo:
            torch.manual_seed(v.get("seed", 0))
            wavs, sr = model.generate_voice_design(text=ref_text, instruct=v["design"], language=lang)
            sf.write(path, wavs[0], sr)
            print(f"designed voice '{name}' -> {path}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with open(job_path + ".refs.json", "w", encoding="utf-8") as fh:
        json.dump({name: p for name, (p, _) in refs.items()}, fh)

    # 2. speak every line with the cloned designed voice
    lines = [ln for ln in job["lines"] if not os.path.isfile(ln["out"])]
    if lines:
        model = load(CLONE_MODEL)
        prompts = {name: model.create_voice_clone_prompt(ref_audio=p, ref_text=t, x_vector_only_mode=t is None)
                   for name, (p, t) in refs.items()}
        for ln in lines:
            # a sentence lasts ~0.4s per word; cap the 12 Hz codec tokens so a runaway (no end token) can't hang.
            # Whisper listens to each take; a take that is too long or doesn't say the line is redone with a new
            # seed, keeping the best of up to 4 takes. Lines rejected by voice_qa (.bad files) start on new seeds.
            limit_s = len(ln["text"].split()) * 0.6 + 4
            base = int(ln.get("seed", job.get("seed", 0))) + 1000 * len(glob.glob(ln["out"] + "*.bad"))
            best = None
            for attempt in range(4):
                torch.manual_seed((base + attempt) % (2 ** 31))
                wavs, sr = model.generate_voice_clone(text=ln["text"], language=lang,
                                                      voice_clone_prompt=prompts[ln["voice"]],
                                                      max_new_tokens=int(12 * limit_s * 1.5))
                secs = len(wavs[0]) / sr
                said, heard = check(ln["text"], wavs[0], sr)
                ratio, _ = score(ln["text"], heard)
                good = secs <= limit_s and said
                if best is None or (good, ratio) > best[2:]:
                    best = (wavs[0], sr, good, ratio)
                if good:
                    break
                print(f"retake ({secs:.0f}s, heard '{heard}'): {ln['text']}", flush=True)
            sf.write(ln["out"], best[0], best[1])
            print(f"spoke [{ln['voice']}] {ln['text']}", flush=True)
    print("QWEN_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
