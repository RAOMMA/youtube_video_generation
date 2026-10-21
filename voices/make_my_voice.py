"""Turn the creator's own recording (e.g. their YouTube intro video) into a clean voice-clone reference.

Finds the steadiest ~15 s stretch of continuous speech, removes hum/noise lightly, normalises loudness,
and writes voices/recordings/me.wav (the file episode scripts point to with [ref Narrator: ...]).

venv\\Scripts\\python voices\\make_my_voice.py <video_or_audio_file> [--seconds 15] [--start 12.5]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cineforge.ffmpeg_utils import FFMPEG  # noqa: E402
from cineforge.tts import load_audio  # noqa: E402

SR = 24000


def best_window(audio: np.ndarray, seconds: float) -> float:
    """Start time of the window with the most continuous speech and fewest long pauses."""
    hop = int(0.05 * SR)
    frames = np.array([np.sqrt(np.mean(audio[i:i + hop] ** 2) + 1e-12) for i in range(0, len(audio) - hop, hop)])
    speech = frames > max(1e-4, np.percentile(frames, 60) * 0.35)
    win = int(seconds / 0.05)
    best, best_score = 0, -1.0
    for s in range(0, max(1, len(speech) - win), 4):
        seg = speech[s:s + win]
        runs = np.diff(np.flatnonzero(np.diff(np.r_[1, seg.astype(int), 1]) != 0))[::2]  # silent run lengths
        long_pauses = int(np.sum(runs > 16))  # pauses longer than 0.8 s
        score = seg.mean() - 0.08 * long_pauses
        if score > best_score:
            best, best_score = s, score
    return best * 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--start", type=float, default=None, help="force a start time instead of auto-picking")
    ap.add_argument("--out", default=os.path.join(HERE, "recordings", "me.wav"))
    a = ap.parse_args()
    audio = load_audio(a.source, SR)
    start = a.start if a.start is not None else best_window(audio, a.seconds)
    out = a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{a.seconds:.2f}", "-i", a.source,
                    "-vn", "-ac", "1", "-af",
                    "highpass=f=70,lowpass=f=12000,afftdn=nf=-28,loudnorm=I=-18:TP=-2,aresample=24000",
                    out], check=True)
    print(f"voice sample: {out}  ({a.seconds:.0f}s from {start:.1f}s of {len(audio) / SR:.0f}s source)")


if __name__ == "__main__":
    main()
