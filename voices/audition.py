"""Chatterbox auditions: each candidate reference voice says the same test lines.
venv\\Scripts\\python voices\\audition.py  ->  voices/auditions/*.mp3"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
from cineforge.ffmpeg_utils import run as ffmpeg  # noqa: E402
from cineforge.tts import CB_PY  # noqa: E402

MALE_LINES = [("Nisa, Nisa! Eating candy?", 0.7), ("Telling lies? Open your mouth!", 0.7),
              ("Assalamu alaikum! Let me show you the app I use every single day.", 0.5)]
FEMALE_LINES = [("Okay, real talk. My coffee used to go cold before I even got to work.", 0.55),
                ("Then I found Brewvo. Hot for twelve hours, and totally leak-proof.", 0.5)]

# every candidates/*.wav with its .json metadata; optional name filter: audition.py Pakistani
cdir = os.path.join(HERE, "candidates")
only = sys.argv[1] if len(sys.argv) > 1 else ""
cands = [{"name": f[:-4], "file": os.path.join(cdir, f), **json.load(open(os.path.join(cdir, f[:-4] + ".json")))}
         for f in sorted(os.listdir(cdir)) if f.endswith(".wav") and only in f]
work = os.path.join(HERE, "auditions", "_parts")
os.makedirs(work, exist_ok=True)
job = {"voices": {c["name"]: {"ref": c["file"]} for c in cands}, "lines": []}
for c in cands:
    for k, (text, exag) in enumerate(MALE_LINES if c["gender"] == "M" else FEMALE_LINES):
        job["lines"].append({"voice": c["name"], "text": text, "exaggeration": exag,
                             "cfg_weight": 0.4 if exag > 0.6 else 0.5, "seed": 7 + k,
                             "out": os.path.join(work, f"{c['name']}_{k}.wav")})
job_path = os.path.join(work, "job.json")
json.dump(job, open(job_path, "w"), indent=1)
subprocess.run([CB_PY, os.path.join(APP, "cineforge", "cb_voices.py"), job_path], check=True)

for c in cands:
    parts = sorted(ln["out"] for ln in job["lines"] if ln["voice"] == c["name"])
    inputs = sum([["-i", p] for p in parts], [])
    chain = "".join(f"[{i}:a]aresample=24000,apad=pad_dur=0.6[a{i}];" for i in range(len(parts)))
    chain += "".join(f"[a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1,loudnorm=I=-16[o]"
    out = os.path.join(HERE, "auditions", f"{c['name']}.mp3")
    ffmpeg([*inputs, "-filter_complex", chain, "-map", "[o]", "-b:a", "160k", out])
    print("audition", out, flush=True)
