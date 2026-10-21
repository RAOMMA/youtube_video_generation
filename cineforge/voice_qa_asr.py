"""Runs inside tts_venv: Whisper-transcribes voice lines and scores them against their script text."""
import json
import sys

import soundfile as sf

from asr_check import check, score, words

job = sys.argv[1]
items = json.load(open(job, encoding="utf-8"))
out = []
for it in items:
    audio, sr = sf.read(it["path"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ok, heard = check(it["text"], audio, sr)
    match, wrong = score(it["text"], heard)
    secs = len(audio) / sr
    expect = len(words(it["text"])) * 0.4 + 0.6
    flag = not ok or secs > expect * 1.8 + 1.5
    out.append({**it, "heard": heard, "match": match, "wrong": wrong, "secs": secs, "expect": expect,
                "flag": bool(flag)})
json.dump(out, open(job + ".out.json", "w", encoding="utf-8"), indent=1)
print("QA_DONE")
