"""Render a faceless explainer script to a YouTube-ready package:
final.mp4 (1080p), subtitles.srt, chapters.txt (description timestamps), thumbnail.png.

venv\\Scripts\\python -m cineforge.explainer_render <script.txt> [--music music.wav] [--kokoro]
"""
from __future__ import annotations

import argparse
import os
import hashlib
import re
import time

from .explainer import (FPS, H, W, ensure_art, host_background, parse_explainer, parse_theme,
                        render_segments, set_theme, thumbnail, thumbnail_kids, write_clip)
from .ffmpeg_utils import run as ffmpeg
from .pipeline import OUTPUT_DIR, VOICE_CACHE_DIR, Pipeline, Settings
from .script_parser import parse_cast
from .tts import Narrator


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


HOST_LOOK = ("3D animated cartoon, Pixar-inspired premium family animation, vibrant colors, soft cinematic "
             "lighting, consistent characters. The characters talk naturally with expressive faces and clear lip "
             "movements on whoever is speaking, gentle gestures, steady camera.")


def host_clip(x, length: float, out: str, comfy, job: str, log=print) -> None:
    """Characters talking on screen: LTX lip-sync of the host image to the scene's dialogue audio,
    shown in a rounded window on the themed slide."""
    image = x.arg.partition("|")[0].strip()
    speakers = [e.speaker for e in x.scene.events if e.speaker != "Narrator"]
    prompt = " ".join(p for p in [HOST_LOOK, x.art, ("Speaking order: " + ", then ".join(speakers) + ".")
                                  if speakers else ""] if p)
    import hashlib
    with open(image, "rb") as fh, open(x.scene.audio_path, "rb") as fa:
        key = hashlib.sha1(fh.read() + fa.read() + prompt.encode() + f"{length:.2f}".encode()).hexdigest()[:16]
    talk = os.path.join(os.path.dirname(job), "_ai_cache", f"host_{key}.mp4")
    os.makedirs(os.path.dirname(talk), exist_ok=True)
    if not os.path.isfile(talk):
        log(f"  host shot {x.scene.index + 1}: LTX lip-sync {length:.1f}s")
        comfy.ltx_talking(image, x.scene.audio_path, talk, prompt, seconds=length, width=704, height=1280,
                          fps=FPS, seed=900 + x.scene.index)
    bg = os.path.join(job, f"hostbg_{x.scene.index:03d}.png")
    win = host_background(x, bg)
    ww, wh = win[2] - win[0], win[3] - win[1]
    mask = os.path.join(job, f"hostmask_{x.scene.index:03d}.png")
    from PIL import Image, ImageDraw
    m = Image.new("L", (ww, wh), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, ww - 1, wh - 1], radius=40, fill=255)
    m.save(mask)
    fc = (f"[1:v]scale={ww}:{wh}:force_original_aspect_ratio=increase:flags=lanczos,crop={ww}:{wh},fps={FPS},"
          f"format=rgba,tpad=stop_mode=clone:stop_duration={length:.3f}[v];[2:v]format=gray[m];[v][m]alphamerge[w];"
          f"[0:v][w]overlay={win[0]}:{win[1]},trim=duration={length:.3f},setsar=1,format=yuv420p[o]")
    ffmpeg(["-loop", "1", "-framerate", str(FPS), "-i", bg, "-i", talk, "-loop", "1", "-i", mask,
            "-filter_complex", fc, "-map", "[o]", "-t", f"{length:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "16", out])


MOTION_LOOK = "lively smooth natural motion, cheerful, steady camera, 3D cartoon animation, consistent characters"
MOTION_NEGATIVE = ("new objects entering the frame, flying objects, morphing, melting, distortion, glitch, smear, "
                   "warped faces, extra limbs, flicker, text, watermark, camera shake, scene change")


def motion_clip(x, length: float, out: str, comfy, job: str, log=print) -> None:
    """[motion: prompt | WORD]: the scene's cartoon comes alive. Wan 2.2 animates the illustration into a ~3.4 s
    clip (cached), slowed up to 2x and held on its last frame to fill the line, with the word sticker on top."""
    from .explainer import word_sticker
    prompt, _, word = (p.strip() for p in x.arg.partition("|"))
    key = hashlib.sha1(f"{x.art_path}|{prompt}|{MOTION_LOOK}|v1".encode()).hexdigest()[:16]
    mv = os.path.join(os.path.dirname(job), "_ai_cache", f"motion_{key}.mp4")
    os.makedirs(os.path.dirname(mv), exist_ok=True)
    frames = 81
    if not os.path.isfile(mv):
        log(f"  motion shot {x.scene.index + 1}: Wan animation")
        comfy.animate(x.art_path, mv, f"{prompt}. {MOTION_LOOK}", MOTION_NEGATIVE, width=832, height=480, frames=frames,
                      steps=20, seed=700 + x.scene.index, fps=FPS)
    slow = min(2.0, max(1.0, length / (frames / FPS)))
    sticker = os.path.join(job, f"sticker_{x.scene.index:03d}.png")
    word_sticker(word, x.scene.index).save(sticker)
    fc = (f"[0:v]setpts={slow:.3f}*PTS,fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={W}:{H},tpad=stop_mode=clone:stop_duration={length:.3f},trim=duration={length:.3f}[v];"
          f"[1:v]format=rgba,fade=t=in:st=0.3:d=0.2:alpha=1[s];[v][s]overlay=0:0,setsar=1,format=yuv420p[o]")
    ffmpeg(["-i", mv, "-loop", "1", "-framerate", str(FPS), "-i", sticker, "-filter_complex", fc, "-map", "[o]",
            "-t", f"{length:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", out])


def _tone(kind: str, sr: int):
    """Tiny synthesized kid-friendly sound effects: a bubbly "pop" and an airy "whoosh". No music or bells:
    the channel doesn't use music."""
    import numpy as np
    if kind == "pop":
        t = np.arange(int(0.14 * sr)) / sr
        f = 520 + 900 * t / t[-1]
        return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 28)).astype(np.float32) * 0.5
    t = np.arange(int(0.38 * sr)) / sr
    noise = np.random.default_rng(1).standard_normal(len(t)).astype(np.float32)
    noise = np.convolve(noise, np.ones(24) / 24, mode="same")
    env = np.sin(np.pi * t / t[-1]) ** 2
    return (noise * env * 0.9).astype(np.float32)


def add_kids_sfx(xs, starts, narration: str) -> None:
    """Kids theme: a whoosh when a new picture or slide arrives and a pop for word stickers and quizzes,
    mixed quietly under the voices so scene changes feel playful."""
    import numpy as np
    import soundfile as sf
    track, sr = sf.read(narration, dtype="float32")
    whoosh, pop = _tone("whoosh", sr), _tone("pop", sr)
    for x, st in zip(xs, starts):
        if x.visual == "host" or x.scene.index == 0:
            continue
        sounds = [(whoosh, st, 0.16)]
        if x.visual in ("show", "motion") and x.arg.partition("|")[2].strip():
            sounds.append((pop, st + 0.3, 0.22))  # the word sticker pops in
        if x.visual == "quiz" and len(x.scene.sentence_times) > 1:  # the right answer lights up (no bells/music)
            sounds.append((pop, st + x.scene.sentence_times[-1][0], 0.3))
        for snd, at, vol in sounds:
            i = int(at * sr)
            seg = track[i:i + len(snd)]
            seg += snd[:len(seg)] * vol
    sf.write(narration, track, sr)


def kids_thumbnail(script_path: str, out: str) -> bool:
    """Kids episodes with a [thumb: hook words | code snippet | color | badge | number] line get the bright
    face-forward thumbnail. Badge and number default to the title card, e.g. "Python for Kids | Episode 3: ..."."""
    script = open(script_path, encoding="utf-8").read()
    m = re.search(r"^\s*\[thumb\s*:\s*(.*)\]\s*$", script, flags=re.M | re.I)
    if not m or parse_theme(script) != "kids":
        return False
    text, snippet, color, badge, number = ([p.strip() for p in m.group(1).split("|")] + [""] * 5)[:5]
    t = re.search(r"^\s*\[title\s*:\s*([^|\]]*)\|?\s*([^\]]*)\]", script, flags=re.M)
    badge = badge or (t.group(1).strip() if t else "")
    ep = re.search(r"Episode\s+(\d+)", t.group(2)) if t else None
    number = number or (ep.group(1) if ep else "")
    set_theme("kids")
    art = None
    if not snippet:  # no code sticker: show the title card's cartoon art (already in the art cache) instead
        from . import explainer as E
        tx = next((x for x in parse_explainer(script) if x.visual == "title" and x.art), None)
        if tx:
            key = hashlib.sha1(f"{tx.art}. {E.ART_STYLE}|{11 + tx.scene.index}".encode()).hexdigest()[:16]
            art = os.path.join(OUTPUT_DIR, "_art_cache", key + ".png")
    thumbnail_kids(out, text, badge, number, snippet, color or "yellow", art=art)
    return True


def render(script_path: str, music: str = "", kokoro: bool = False, log=print) -> str:
    start = time.time()
    script = open(script_path, encoding="utf-8").read()
    name = os.path.splitext(os.path.basename(script_path))[0]
    job = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S") + f"_{name}")
    os.makedirs(job, exist_ok=True)
    set_theme(parse_theme(script))
    xs = parse_explainer(script)
    scenes = [x.scene for x in xs]
    log(f"{len(xs)} scenes -> {job}")
    mm = re.search(r"^\s*\[music\s*:\s*(.+?)\s*\]\s*$", script, flags=re.M | re.I)
    music = music or (mm.group(1) if mm and os.path.isfile(mm.group(1)) else "")

    # 1. narration in the creator's voice (Chatterbox clone of [ref Narrator: ...]); --kokoro = quick preview voice
    cast = parse_cast(script)
    narrator = cast.get("narrator", {}) if not kokoro else {}
    voice = {"voice": "am_michael", "speed": 1.0, "pitch": 1.0, **narrator}
    n = Narrator()
    if kokoro:  # stock voices for a quick preview
        cast = {k: {kk: vv for kk, vv in v.items() if kk not in ("ref", "design")} for k, v in cast.items()}
    m = re.search(r"^\s*\[engine\s*:\s*(\w+)\s*\]", script, flags=re.M | re.I)
    engine = m.group(1).lower() if m else "chatterbox"
    log(f"voice engine: {engine}")
    n.prerender_designed(scenes, cast, voice, VOICE_CACHE_DIR, log, engine=engine)
    for sc in scenes:
        n.narrate_scene(sc, os.path.join(job, f"voice_{sc.index:03d}.wav"), cast, voice, {},
                        lead_in=0.35, gap=0.3, tail=0.55)
    n.release()
    total = sum(sc.duration for sc in scenes)
    log(f"narration: {total / 60:.1f} min")

    # 2. visuals
    settings = Settings(aspect="16:9", crossfade=0.35, grade="Natural", letterbox=False, vignette=False,
                        grain=False, subtitles=False, music_path=music, music_volume=0.07)
    comfy = None
    if any(x.art or x.visual in ("image", "host", "show", "motion") for x in xs):
        from .comfy_client import ComfyClient
        comfy = ComfyClient(log)
        for x in xs:  # all illustrations first, so the GPU is free for the rest
            ensure_art(x, comfy, job)
        log("illustrations ready")
    clips, t0 = [], 0.0
    for x in xs:
        last = x.scene.index == len(xs) - 1
        length = x.scene.duration + (0 if last else settings.crossfade)
        out = os.path.join(job, f"clip_{x.scene.index:03d}.mp4")
        if x.visual == "host":
            host_clip(x, length, out, comfy, job, log)
        elif x.visual == "motion":
            motion_clip(x, length, out, comfy, job, log)
        else:
            write_clip(x, render_segments(x, length), out, t0, total)
        clips.append(out)
        t0 += x.scene.duration
        log(f"  scene {x.scene.index + 1}/{len(xs)} [{x.visual}] {x.scene.duration:.1f}s")

    # 3. audio mix + final cut
    p = Pipeline(log=log)
    p.W, p.H = W, H
    narration = p._build_narration(scenes, job)
    if parse_theme(script) == "kids":
        add_kids_sfx(xs, p._starts(scenes)[0], narration)
    srt = p._build_srt(scenes, job)
    final = os.path.join(job, "final.mp4")
    p._assemble(clips, scenes, narration, srt, final, settings)

    # 4. YouTube extras: chapters (first must be 0:00), thumbnail
    chapters, t, seen = [], 0.0, set()
    for x in xs:
        if x.chapter and x.chapter not in seen:
            seen.add(x.chapter)
            chapters.append((t, x.chapter))
        t += x.scene.duration
    if chapters and chapters[0][0] > 0.5:  # YouTube needs the first chapter at 0:00
        chapters.insert(0, (0.0, "Intro"))
    with open(os.path.join(job, "chapters.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"{fmt_ts(ts)} {title}" for ts, title in chapters) + "\n")
    title = next((x.arg.split("|")[0].strip() for x in xs if x.visual == "title"), name)
    art = (next((x.art_path for x in xs if x.visual == "title" and x.art_path), None)
           or next((x.art_path for x in xs if x.art_path), None))
    if not kids_thumbnail(script_path, os.path.join(job, "thumbnail.png")):
        thumbnail(title, os.path.join(job, "thumbnail.png"), art)
    log(f"FINISHED in {(time.time() - start) / 60:.1f} min: {final}")
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--music", default="")
    ap.add_argument("--kokoro", action="store_true", help="quick preview with a stock voice")
    a = ap.parse_args()
    render(a.script, a.music, a.kokoro)
