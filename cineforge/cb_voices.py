"""Runs inside cb_venv (chatterbox-tts). Speaks lines by cloning a reference clip per voice,
with per-line expressiveness.

Usage: cb_venv\\Scripts\\python cineforge\\cb_voices.py job.json
job.json: {"voices": {"nisa": {"ref": "C:/.../ref.wav"}},
           "lines": [{"voice": "nisa", "text": "...", "out": "C:/.../x.wav",
                      "exaggeration": 0.5, "cfg_weight": 0.5}]}
exaggeration: 0.25 calm ... 0.5 natural ... 0.8+ very expressive.
cfg_weight: lower (~0.3) = slower, more deliberate delivery with stronger emotion.
"""
import json
import os
import sys

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def main(job_path):
    job = json.load(open(job_path, encoding="utf-8"))
    lines = [ln for ln in job["lines"] if not os.path.isfile(ln["out"])]
    if lines:
        model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
        for ln in lines:
            torch.manual_seed(int(ln.get("seed", 0)))
            wav = model.generate(ln["text"], audio_prompt_path=job["voices"][ln["voice"]]["ref"],
                                 exaggeration=float(ln.get("exaggeration", 0.5)),
                                 cfg_weight=float(ln.get("cfg_weight", 0.5)))
            ta.save(ln["out"], wav, model.sr)
            print(f"spoke [{ln['voice']}] {ln['text']}", flush=True)
    print("CB_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
