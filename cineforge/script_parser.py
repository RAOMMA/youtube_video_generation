"""Parse a plain-text script into scenes.

Format (all tags optional):

    [voice Nisa: af_heart, pitch=1.3, speed=0.95]   # cast definition (anywhere, applies to whole film)

    [camera: zoom_in]          # zoom_in | zoom_out | pan_left | pan_right | pan_up | pan_down | auto
    [motion: fog drifts, slow push toward the castle]   # prompt for AI motion
    [mode: ltx]                # ltx (talking, lip-sync) | ai (Wan 2.2) | kenburns  (overrides the global setting)
    [image: 2]                 # use image #2 (1-based) instead of the next one
    [duration: 10]             # minimum seconds for this scene
    Plain text lines are spoken by the Narrator.
    Nisa: "Character lines use the character's voice."
    Nisa & Yusuf: "Both speak together."
    [sfx: ambulance siren wailing | 3]   # AI sound effect (prompt | seconds), plays at this point
    Ambulance: "Nee-naw! Nee-naw!"
    ---
    Next scene...

Scenes are separated by a line containing only `---`. If no `---` is present,
blank lines separate scenes instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DIRECTIVE = re.compile(r"^\s*\[(\w+)\s*:\s*(.*?)\]\s*$")
CAST = re.compile(r"^\s*\[voice\s+([^:\]]+?)\s*:\s*(.*?)\]\s*$", re.I)
DESIGN = re.compile(r"^\s*\[design\s+([^:\]]+?)\s*:\s*(.*?)\]\s*$", re.I)
REF = re.compile(r"^\s*\[ref\s+([^:\]]+?)\s*:\s*(.*?)\]\s*$", re.I)   # [ref Baba: C:\path\voice.wav]
SPEAKER = re.compile(r"^\s*([A-Za-z][\w .&'-]{0,40}?)\s*(?:\(([^)]{1,80})\))?\s*:\s*(.+)$")  # Name (emotion): text
CAMERA_MOVES = ["zoom_in", "pan_right", "zoom_out", "pan_left", "pan_up", "pan_down"]
QUOTES = "\"'“”‘’"


@dataclass
class Event:
    kind: str            # "speech" | "sfx"
    speaker: str = ""    # speech: who talks ("Narrator" for plain lines)
    text: str = ""       # speech: words; sfx: prompt
    seconds: float = 3.0 # sfx length
    emotion: str = ""    # speech: optional delivery direction, e.g. "giggling", "suspicious"


@dataclass
class Scene:
    index: int
    text: str
    image_index: int
    events: list = field(default_factory=list)
    camera: str = "auto"
    motion: str = ""
    mode: str = ""          # "" means use the global setting
    min_duration: float = 0.0
    crop_bottom: float = 0.0   # % of the frame to trim from the bottom (e.g. AI-drawn caption bars)
    # filled in by the pipeline
    audio_path: str = ""
    duration: float = 0.0
    sentence_times: list = field(default_factory=list)  # [(start, end, text)] relative to scene


def _split_blocks(script: str) -> list[str]:
    script = script.replace("\r\n", "\n").strip()
    if re.search(r"^\s*---\s*$", script, flags=re.M):
        blocks = re.split(r"^\s*---\s*$", script, flags=re.M)
    else:
        blocks = re.split(r"\n\s*\n", script)
    return [b.strip() for b in blocks if b.strip()]


def parse_cast(script: str) -> dict[str, dict]:
    """[voice Name: kokoro_voice, pitch=1.3, speed=0.9] -> {"name": {...}} (lower-case keys).
    [design Name: free-text voice description] switches that character to a Qwen3-TTS designed voice."""
    cast = {}
    for line in script.splitlines():
        r = REF.match(line)
        if r:
            name = r.group(1).strip().lower()
            cast.setdefault(name, {"voice": "af_heart", "pitch": 1.0, "speed": 1.0})["ref"] = r.group(2).strip()
            continue
        d = DESIGN.match(line)
        if d:
            name = d.group(1).strip().lower()
            cast.setdefault(name, {"voice": "af_heart", "pitch": 1.0, "speed": 1.0})["design"] = d.group(2).strip()
            continue
        m = CAST.match(line)
        if not m:
            continue
        parts = [p.strip() for p in m.group(2).split(",") if p.strip()]
        cfg = {"voice": parts[0] if parts else "af_heart", "pitch": 1.0, "speed": 1.0}
        for p in parts[1:]:
            if "=" in p:
                k, v = (x.strip() for x in p.split("=", 1))
                cfg[k.lower()] = float(v)
        name = m.group(1).strip().lower()
        cast[name] = {**cfg, **{k: cast[name][k] for k in ("design", "ref") if k in cast.get(name, {})}}
    return cast


def _clean(text: str) -> str:
    text = text.strip().strip(QUOTES).strip()
    # drop emoji / pictographs so the TTS doesn't try to read them
    return re.sub(r"[\U0001F000-\U0001FFFF☀-➿️‍]", "", text).strip().strip(QUOTES).strip()


def parse_script(script: str, num_images: int) -> list[Scene]:
    if num_images < 1:
        raise ValueError("Add at least one image.")
    scenes: list[Scene] = []
    next_image = 0
    for block in _split_blocks(script):
        opts: dict[str, str] = {}
        events: list[Event] = []
        for line in block.split("\n"):
            if not line.strip() or CAST.match(line) or DESIGN.match(line) or REF.match(line):
                continue
            m = DIRECTIVE.match(line)
            if m:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key == "sfx":
                    prompt, _, secs = val.partition("|")
                    events.append(Event("sfx", text=prompt.strip(), seconds=float(secs or 3)))
                else:
                    opts[key] = val
                continue
            sm = SPEAKER.match(line)
            if sm and not sm.group(1).lower().startswith("http"):
                text = _clean(sm.group(3))
                if text:
                    events.append(Event("speech", speaker=sm.group(1).strip(), text=text,
                                        emotion=(sm.group(2) or "").strip()))
            else:
                text = _clean(line)
                if text:
                    events.append(Event("speech", speaker="Narrator", text=text))
        if not events and not opts:
            continue
        i = len(scenes)

        if "image" in opts:
            img = max(0, int(opts["image"]) - 1) % num_images
        else:
            img = next_image % num_images
        next_image = img + 1

        camera = opts.get("camera", "auto").lower().replace(" ", "_").replace("-", "_")
        if camera == "auto" or camera not in CAMERA_MOVES:
            camera = CAMERA_MOVES[i % len(CAMERA_MOVES)]

        mode = opts.get("mode", "").lower().replace(" ", "")
        if mode in ("kb", "ken_burns"):
            mode = "kenburns"

        scenes.append(Scene(
            index=i,
            text=" ".join(f"{e.speaker}: {e.text}" if e.speaker != "Narrator" else e.text
                          for e in events if e.kind == "speech"),
            image_index=img,
            events=events,
            camera=camera,
            motion=opts.get("motion", ""),
            mode=mode if mode in ("ai", "ltx", "kenburns") else "",
            min_duration=float(opts.get("duration", 0) or 0),
            crop_bottom=float(opts.get("crop_bottom", 0) or 0),
        ))
    if not scenes:
        raise ValueError("The script is empty.")
    return scenes


def split_sentences(text: str, max_chars: int = 90) -> list[str]:
    """Split narration into subtitle-sized sentences."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        while len(p) > max_chars:
            cut = p.rfind(",", 0, max_chars)
            if cut < max_chars // 3:
                cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                break
            out.append(p[: cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out
