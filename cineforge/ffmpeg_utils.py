from __future__ import annotations

import glob
import os
import shutil
import subprocess


def _find(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    # winget installs ffmpeg here; the PATH change only applies to new shells
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
                           "Gyan.FFmpeg*", "**", "bin", f"{name}.exe")
    hits = glob.glob(pattern, recursive=True)
    if hits:
        return hits[0]
    raise RuntimeError(f"{name} not found. Install it with: winget install Gyan.FFmpeg")


FFMPEG = _find("ffmpeg")
FFPROBE = _find("ffprobe")


def run(args: list[str], cwd: str | None = None) -> None:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                          cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + proc.stderr[-3000:])


def probe_duration(path: str) -> float:
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def raw_writer(out_path: str, width: int, height: int, fps: int) -> subprocess.Popen:
    """ffmpeg process that encodes raw RGB frames written to its stdin."""
    return subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
