"""Smooth sub-pixel Ken Burns camera moves rendered with PIL and piped into ffmpeg."""
from __future__ import annotations

import math

from PIL import Image

from .ffmpeg_utils import raw_writer


def _ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def cover_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize + center-crop an image so it exactly fills width x height."""
    scale = max(width / img.width, height / img.height)
    w, h = round(img.width * scale), round(img.height * scale)
    img = img.resize((w, h), Image.LANCZOS)
    left, top = (w - width) // 2, (h - height) // 2
    return img.crop((left, top, left + width, top + height))


def render(image_path: str, out_path: str, duration: float, camera: str,
           width: int = 1920, height: int = 1080, fps: int = 24, strength: float = 1.0) -> None:
    img = Image.open(image_path).convert("RGB")
    # Work on a canvas 1.5x the output so zooms stay sharp even for small inputs.
    base = cover_crop(img, int(width * 1.5), int(height * 1.5))
    bw, bh = base.size

    zoom = 0.15 * strength  # how far the camera travels
    if camera == "zoom_in":
        z0, z1, p0, p1 = 1.0, 1.0 + zoom, (0, 0), (0, 0)
    elif camera == "zoom_out":
        z0, z1, p0, p1 = 1.0 + zoom, 1.0, (0, 0), (0, 0)
    else:
        z0 = z1 = 1.0 + zoom
        d = {"pan_left": (1, 0), "pan_right": (-1, 0), "pan_up": (0, 1), "pan_down": (0, -1)}.get(camera, (-1, 0))
        p0, p1 = (d[0], d[1]), (-d[0], -d[1])

    frames = max(1, round(duration * fps))
    proc = raw_writer(out_path, width, height, fps)
    try:
        for f in range(frames):
            t = _ease(f / max(1, frames - 1))
            z = z0 + (z1 - z0) * t
            vw, vh = bw / z, bh / z                  # visible rect in base-image pixels
            mx, my = (bw - vw) / 2, (bh - vh) / 2    # max pan margin
            px = p0[0] + (p1[0] - p0[0]) * t
            py = p0[1] + (p1[1] - p0[1]) * t
            cx, cy = bw / 2 + px * mx, bh / 2 + py * my
            box = (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)
            frame = base.transform((width, height), Image.EXTENT, box, resample=Image.BICUBIC)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(proc.stderr.read().decode(errors="ignore")[-2000:])
    finally:
        if proc.poll() is None:
            proc.kill()
