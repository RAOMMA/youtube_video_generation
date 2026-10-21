"""Transcribe voice samples with Whisper (runs in tts_venv). Writes <sample>.txt next to each file.
tts_venv\\Scripts\\python voices\\transcribe.py voices\\recordings\\me.wav [more.wav ...]"""
import sys

import soundfile as sf
import torch
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en",
               device=0 if torch.cuda.is_available() else -1)
for path in sys.argv[1:]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    text = asr({"raw": audio, "sampling_rate": sr}, chunk_length_s=30)["text"].strip()
    with open(path.rsplit(".", 1)[0] + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"{path}: {text}", flush=True)
