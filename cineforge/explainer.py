"""Faceless long-form explainer videos (YouTube, 16:9): narration in the creator's voice over
code-drawn visuals that reveal in sync with the narration.

Script format (one scene per block, blocks separated by `---`):

    [ref Narrator: D:\\path\\my_voice.wav]        # voice to clone (global)
    [chapter: Neural networks]                   # starts a YouTube chapter (badge + description timestamp)
    [title: Big title | small subtitle]
    [bullets: First point | Second point | Third point]   # one card appears per narration sentence
    [code: python | train.py]                    # followed by a ``` fenced block, typed line by line
    [nn: 3,4,4,2]                                # neural-network diagram, signal flows layer by layer
    [plot: sigmoid, relu]                        # function plots drawn progressively
    [equation: y = w1·x1 + b | caption]
    [image: prompt]                              # full-screen AI illustration (Wan 2.2) with slow camera move
    [steps: Study hours × 0.6 = 4.2 | => Chance to pass = 83%]   # worked example, one row per sentence
    [fit: both]                                  # good fit vs overfitting (also: good / overfit)
    [quiz: Is this a cat or a dog? | Cat | Dog | Cat]   # answer revealed on the scene's last line
    [flow: Question -> Retrieve: top chunks -> LLM -> Answer | loop]   # boxes appear one per sentence
    [rnn: The | cat | sat | down | fade]         # unrolled RNN reading word by word ('fade' = vanishing memory)
    [gates: lstm]  /  [gates: gru]               # cell diagram, one part lights up per sentence
    [conv: edge]  /  [conv: pool]                # sliding 3x3 filter building a feature map / 2x2 max pooling
    [tokens: Un|believ|able| results]            # token chips with ids
    [next: The cat sat on the | mat 62 | floor 21 | sofa 9]   # next-token probabilities
    [attention: The animal didn't cross the street because it was tired | it -> animal]
    [host: D:/path/characters.png | Caption]    # characters talk on screen (LTX lip-sync), kids theme
    [theme: kids]                                # global: bright kids look (default: adult)
    Name (emotion): "line"                       # dialogue, voiced with that character's [ref]/[design]
    [art: prompt]                                # AI illustration beside the cards / behind the title
    Plain lines are the narration for that scene.

Each scene is rendered as content images (which slowly push in) plus a fixed chrome layer
(chapter badge, channel logo, progress bar).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .ffmpeg_utils import raw_writer
from .kenburns import cover_crop
from .script_parser import CAST, DESIGN, REF, SPEAKER, Event, Scene, _clean

W, H, FPS = 1920, 1080, 24
NAVY = (8, 12, 26)
TEAL, AMBER, VIOLET = (32, 214, 186), (255, 190, 72), (140, 110, 255)
WHITE, SOFT, MUTED = (242, 246, 252), (196, 205, 220), (128, 140, 162)
FONT = r"C:\Windows\Fonts\segoeui.ttf"
FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_SEMI = r"C:\Windows\Fonts\seguisb.ttf"
FONT_BLACK = r"C:\Windows\Fonts\seguibl.ttf"
FONT_MONO = r"C:\Windows\Fonts\consola.ttf"
CHANNEL = "ML wali Sarkar"
ART_STYLE = ("3D render, sleek modern tech illustration, glowing neon teal and warm amber light, deep navy "
             "background, soft volumetric lighting, depth of field, clean minimal composition, high detail")
# theme-dependent colours (set_theme switches them); TEXT* = text on the background, DARK = text on accent fills
THEME, BG_BASE, DARK, TEXT, TEXT2, TEXT3 = "adult", NAVY, NAVY, WHITE, SOFT, MUTED
KIDS_ART_STYLE = ("cute 3D cartoon illustration, Pixar style, bright cheerful pastel colors, soft rounded shapes, "
                  "soft studio lighting, simple clean background, child-friendly")


def set_theme(name: str) -> None:
    """'adult' (dark navy tech look) or 'kids' (bright sky pastel, white cards, cartoon art)."""
    global THEME, TEAL, AMBER, VIOLET, BG_BASE, DARK, TEXT, TEXT2, TEXT3, ART_STYLE, _bg
    THEME, _bg = name, None
    if name == "kids":
        TEAL, AMBER, VIOLET = (28, 184, 164), (255, 128, 64), (155, 93, 229)
        BG_BASE, DARK = (214, 236, 255), (255, 255, 255)
        TEXT, TEXT2, TEXT3 = (30, 36, 64), (70, 78, 110), (120, 128, 160)
        ART_STYLE = KIDS_ART_STYLE
    else:
        TEAL, AMBER, VIOLET = (32, 214, 186), (255, 190, 72), (140, 110, 255)
        BG_BASE, DARK, TEXT, TEXT2, TEXT3 = NAVY, NAVY, WHITE, SOFT, MUTED
        ART_STYLE = ("3D render, sleek modern tech illustration, glowing neon teal and warm amber light, deep navy "
                     "background, soft volumetric lighting, depth of field, clean minimal composition, high detail")

_font_cache: dict = {}


def font(path, size):
    if not os.path.isfile(path):
        path = FONT_BOLD
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


@dataclass
class XScene:
    scene: Scene
    visual: str = "bullets"
    arg: str = ""
    code: str = ""
    chapter: str = ""
    chapter_no: int = 0
    art: str = ""                    # prompt for an illustration beside / behind the content
    art_path: str = ""
    segments: list = field(default_factory=list)


# ----------------------------------------------------------------------------- parsing
def parse_explainer(script: str) -> list[XScene]:
    blocks = [b.strip() for b in re.split(r"^\s*---\s*$", script.replace("\r\n", "\n"), flags=re.M)]
    out, chapter, chapter_no = [], "", 0
    # only declared characters count as speakers, so narration like "Try this: ..." stays with the narrator
    names = {"narrator"} | {m.group(1).strip().lower() for rx in (CAST, DESIGN, REF)
                            for m in (rx.match(ln) for ln in script.splitlines()) if m}
    for block in blocks:
        if not block:
            continue
        code, fence = "", re.search(r"```[\w+-]*\n(.*?)```", block, flags=re.S)
        if fence:
            code = fence.group(1).rstrip("\n")
            block = block[:fence.start()] + block[fence.end():]
        visual, arg, art, events = "", "", "", []
        for line in block.split("\n"):
            s = line.strip()
            if not s or CAST.match(s) or DESIGN.match(s) or REF.match(s):
                continue
            m = re.match(r"^\[(\w+)\s*:\s*(.*)\]$", s)
            if m:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key in ("theme", "engine", "thumb", "music"):
                    continue
                if key == "chapter":
                    chapter, chapter_no = val, chapter_no + 1
                elif key == "art":
                    art = val
                else:
                    visual, arg = key, val
                continue
            sm = SPEAKER.match(s)
            if sm and not s.startswith("[") and sm.group(1).strip().lower() in names:  # Name (emotion): "line"
                text = _clean(sm.group(3))
                if text:
                    events.append(Event("speech", speaker=sm.group(1).strip(), text=text,
                                        emotion=(sm.group(2) or "").strip()))
                continue
            text = _clean(s)
            if text:
                events.append(Event("speech", speaker="Narrator", text=text))
        if not events and not visual:
            continue
        i = len(out)
        sc = Scene(index=i, text=" ".join(e.text for e in events), image_index=0, events=events)
        out.append(XScene(scene=sc, visual=visual or "bullets", arg=arg, code=code, chapter=chapter,
                          chapter_no=chapter_no, art=art))
    return out


def parse_theme(script: str) -> str:
    m = re.search(r"^\s*\[theme\s*:\s*(\w+)\s*\]", script, flags=re.M | re.I)
    return m.group(1).lower() if m else "adult"


# ----------------------------------------------------------------------------- design system
_bg: Image.Image | None = None


def background() -> Image.Image:
    """Adult: deep navy with teal/violet glows, grain, vignette. Kids: sky-to-lavender with soft pastel blobs."""
    global _bg
    if _bg is None:
        yy, xx = np.mgrid[0:H, 0:W]
        if THEME == "kids":
            t = (yy / (H - 1))[..., None]
            arr = np.array([205, 233, 255], np.float32) * (1 - t) + np.array([236, 224, 255], np.float32) * t
            blobs = Image.new("RGB", (W // 8, H // 8), (0, 0, 0))
            bd = ImageDraw.Draw(blobs)
            for (cx, cy, r, col) in [(30, 20, 40, (60, 40, -20)), (210, 120, 50, (60, -10, -40)),
                                     (130, 70, 30, (-20, 25, 10))]:
                bd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(max(0, c) for c in col))
            blob = np.asarray(blobs.filter(ImageFilter.GaussianBlur(20)).resize((W, H), Image.BICUBIC), np.float32)
            arr = arr + blob * 0.6
            d = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            dots = ImageDraw.Draw(d)
            for gx in range(48, W, 48):
                for gy in range(48, H, 48):
                    dots.point((gx, gy), fill=(190, 205, 235))
            _bg = d
            return _bg.copy()
        base = Image.new("RGB", (W, H), NAVY)
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        g = ImageDraw.Draw(glow)
        g.ellipse([-350, -420, 900, 520], fill=(10, 70, 72))
        g.ellipse([1150, 520, 2350, 1500], fill=(46, 30, 96))
        g.ellipse([700, 300, 1300, 800], fill=(14, 26, 52))
        glow = glow.resize((W // 8, H // 8)).filter(ImageFilter.GaussianBlur(28)).resize((W, H), Image.BICUBIC)
        arr = np.asarray(base, dtype=np.float32) + np.asarray(glow, dtype=np.float32)
        vig = 1 - 0.45 * (((xx - W / 2) / (W / 1.6)) ** 2 + ((yy - H / 2) / (H / 1.4)) ** 2)
        arr *= np.clip(vig, 0.55, 1)[..., None]
        arr += np.random.default_rng(1).normal(0, 3.2, arr.shape[:2])[..., None]  # film grain
        d = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        dots = ImageDraw.Draw(d)
        for gx in range(48, W, 48):
            for gy in range(48, H, 48):
                dots.point((gx, gy), fill=(30, 40, 62))
        _bg = d
    return _bg.copy()


def layer() -> Image.Image:
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def paste_layer(base: Image.Image, top: Image.Image) -> Image.Image:
    out = base.convert("RGBA")
    out.alpha_composite(top)
    return out.convert("RGB")


def glass(lay: Image.Image, box, radius=26, alpha=16, outline=40, accent=None):
    d = ImageDraw.Draw(lay)
    if THEME == "kids":
        x0, y0, x1, y1 = box
        d.rounded_rectangle([x0 + 4, y0 + 8, x1 + 4, y1 + 8], radius=radius, fill=(80, 90, 160, 40))
        d.rounded_rectangle(box, radius=radius, fill=(255, 255, 255, 235 if alpha < 20 else 255),
                            outline=(accent or (200, 210, 240)) + (255,), width=3 if accent else 2)
    else:
        d.rounded_rectangle(box, radius=radius, fill=(255, 255, 255, alpha), outline=(255, 255, 255, outline), width=2)
    if accent:
        x0, y0, _, y1 = box
        d.rounded_rectangle([x0, y0 + 14, x0 + 7, y1 - 14], radius=4, fill=accent + (255,))


def shadow_glow(lay: Image.Image, box, color, radius=26, blur=28, alpha=110):
    g = layer()
    ImageDraw.Draw(g).rounded_rectangle(box, radius=radius, fill=color + (alpha,))
    lay.alpha_composite(g.filter(ImageFilter.GaussianBlur(blur)))


def wrap(d, text, fnt, width):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=fnt) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def framed_art(lay: Image.Image, path: str, box, radius=30):
    """AI illustration in a rounded frame with a teal glow."""
    x0, y0, x1, y1 = [int(v) for v in box]
    shadow_glow(lay, box, TEAL, radius, blur=36, alpha=70)
    art = cover_crop(Image.open(path).convert("RGB"), x1 - x0, y1 - y0).convert("RGBA")
    mask = Image.new("L", art.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, art.width - 1, art.height - 1], radius=radius, fill=255)
    art.putalpha(mask)
    lay.alpha_composite(art, (x0, y0))
    ImageDraw.Draw(lay).rounded_rectangle(box, radius=radius, outline=(255, 255, 255, 50), width=2)


def chrome(x: XScene, caption: str = "") -> Image.Image:
    """Fixed overlay: chapter badge (top-left), channel logo (bottom-left), optional lower-third caption."""
    lay = layer()
    d = ImageDraw.Draw(lay)
    if x.chapter and x.visual not in ("title",):
        label = f"{x.chapter_no:02d}  ·  {x.chapter.upper()}"
        f = font(FONT_SEMI, 24)
        w = d.textlength(label, font=f) + 56
        d.rounded_rectangle([70, 52, 70 + w, 100], radius=24, fill=(32, 214, 186, 38),
                            outline=TEAL + (200,), width=2)
        d.text((98, 76), label, font=f, fill=TEXT + (255,), anchor="lm")
    d.ellipse([70, H - 92, 112, H - 50], fill=TEAL + (235,))
    d.text((91, H - 71), "M", font=font(FONT_BLACK, 24), fill=DARK + (255,), anchor="mm")
    d.text((126, H - 71), CHANNEL, font=font(FONT_SEMI, 24), fill=TEXT2 + (220,), anchor="lm")
    if caption:
        cf = font(FONT_SEMI, 46)
        lines = wrap(d, caption, cf, W - 600)[:2]
        bw = max(d.textlength(ln, font=cf) for ln in lines) + 90
        bh = len(lines) * 58 + 44
        y0 = H - 130 - bh
        box = ((W - bw) / 2, y0, (W + bw) / 2, y0 + bh)
        g = layer()
        ImageDraw.Draw(g).rounded_rectangle(box, radius=26, fill=(0, 0, 0, 120))
        lay.alpha_composite(g.filter(ImageFilter.GaussianBlur(14)))
        d.rounded_rectangle(box, radius=26, fill=(10, 14, 28, 175))
        for k, ln in enumerate(lines):
            d.text((W // 2, y0 + 22 + 29 + k * 58), ln, font=cf, fill=(255, 255, 255, 255), anchor="mm")
    return lay


_STOP = set("a an the and or of to in on at is are was were be it its this that these those for with as by from "
             "so but if then than into out up down over under we you i he she they them his her our your their "
             "can will just like also very much more most each every all some any".split())


def _words(text: str) -> set:
    ws = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in ws if w not in _STOP}


def reveal_times(sc: Scene, n: int, items: list[str] | None = None) -> list[float]:
    """When each of n items should appear, in sync with the narration.

    With item texts, each item is matched to the narration sentence that talks about it (word overlap,
    keeping the order), so an intro line like 'AI is all around us' doesn't steal the first item.
    Without texts (or no overlap at all), items go on successive sentences, skipping one intro line when
    there are more sentences than items."""
    st = sc.sentence_times
    starts = [a for a, _, _ in st] or [0.3]
    m = len(starts)
    if items and n <= m and st:
        sw = [_words(t) for _, _, t in st]
        iw = [_words(t) for t in items]
        score = [[len(iw[i] & sw[j]) / max(1, len(iw[i])) for j in range(m)] for i in range(n)]
        # DP: best strictly increasing assignment of items to sentences
        NEG = -1e9
        best = [[NEG] * m for _ in range(n)]
        back = [[-1] * m for _ in range(n)]
        for j in range(m):
            best[0][j] = score[0][j]
        for i in range(1, n):
            run, arg = NEG, -1
            for j in range(m):
                if j - 1 >= 0 and best[i - 1][j - 1] > run:
                    run, arg = best[i - 1][j - 1], j - 1
                if arg >= 0:
                    best[i][j] = run + score[i][j]
                    back[i][j] = arg
        j = max(range(m), key=lambda jj: (best[n - 1][jj], -jj))
        total = best[n - 1][j]
        if total > 0.34 * n * 0.5:  # enough evidence: use the matched sentences
            idx = [0] * n
            for i in range(n - 1, -1, -1):
                idx[i] = j
                j = back[i][j]
            return [starts[k] for k in idx]
    if n < m:  # no usable overlap: assume one intro sentence before the items
        offset = min(1, m - n)
        return [starts[k + offset] for k in range(n)]
    times = [starts[min(k, m - 1)] for k in range(n)]
    if n > m:
        a, b = st[-1][:2] if st else (0.3, sc.duration)
        extra = n - m + 1
        for jj, k in enumerate(range(m - 1, n)):
            times[k] = a + (b - a) * jj / extra
    return times


def segments_from_states(states, length):
    states = sorted(states, key=lambda s: s[0])
    states[0] = (0.0, states[0][1])
    segs = []
    for k, (t, im) in enumerate(states):
        end = states[k + 1][0] if k + 1 < len(states) else length
        if end - t > 0.02:
            segs.append((im, end - t))
    return segs


def heading(lay, text, x=120, y=150, size=60, width=1600):
    d = ImageDraw.Draw(lay)
    f = font(FONT_BOLD, size)
    lines = wrap(d, text, f, width)[:2]
    for k, ln in enumerate(lines):
        d.text((x, y + k * (size + 12)), ln, font=f, fill=TEXT + (255,))
    bottom = y + len(lines) * (size + 12)
    d.rounded_rectangle([x, bottom + 14, x + 90, bottom + 21], radius=4, fill=AMBER + (255,))
    return bottom + 60


# ----------------------------------------------------------------------------- visuals
def v_title(x: XScene, length):
    im = background()
    lay = layer()
    if x.art_path:  # art fills the right ~58% and fades into the background, text keeps the clean left side
        aw = int(W * 0.58)
        art = cover_crop(Image.open(x.art_path).convert("RGB"), aw, H).convert("RGBA")
        fade = Image.new("L", (aw, H), 255)
        fd = ImageDraw.Draw(fade)
        for xx in range(int(aw * 0.45)):
            fd.line([(xx, 0), (xx, H)], fill=int(255 * (xx / (aw * 0.45)) ** 1.6))
        art.putalpha(fade)
        im = im.convert("RGBA")
        im.alpha_composite(art, (W - aw, 0))
        im = im.convert("RGB")
    d = ImageDraw.Draw(lay)
    title, _, sub = x.arg.partition("|")
    words = title.strip().split()
    tf = font(FONT_BLACK, 112)
    lines = wrap(d, title.strip(), tf, 860 if x.art_path else 1500)
    y = H // 2 - (len(lines) * 124) // 2 - 50
    d.text((120, y - 70), CHANNEL.upper(), font=font(FONT_SEMI, 28), fill=TEAL + (255,))
    for k, ln in enumerate(lines):
        d.text((116, y + k * 124), ln, font=tf, fill=(AMBER if k == len(lines) - 1 and len(words) > 2 else TEXT) + (255,))
    y += len(lines) * 124 + 20
    d.rounded_rectangle([120, y, 260, y + 8], radius=4, fill=TEAL + (255,))
    if sub.strip():
        d.text((120, y + 40), sub.strip(), font=font(FONT, 44), fill=TEXT2 + (255,))
    return [(paste_layer(im, lay), length)]


def v_bullets(x: XScene, length):
    items = [b.strip() for b in x.arg.split("|") if b.strip()] if x.arg else []
    if not items:
        items = [s for _, _, s in x.scene.sentence_times][:4] or [x.scene.text]
    times = reveal_times(x.scene, len(items), items)
    has_art = bool(x.art_path)
    col_w = 980 if has_art else 1620
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    size = (40 if len(items) > 4 else 44) + (8 if THEME == "kids" else 0)
    while True:  # shrink until the cards fit between the heading and the logo
        bf = font(FONT_SEMI, size)
        lh = int(size * 1.25)
        heights = [max(1, len(wrap(probe, it, bf, col_w - 170))) * lh + 54 for it in items]
        if sum(heights) + 22 * (len(items) - 1) <= H - 300 - 130 or size <= 30:
            break
        size -= 2
    base = background()
    static = layer()
    top = heading(static, x.chapter or "Key ideas", width=col_w)
    gap = 22
    total = sum(heights) + gap * (len(items) - 1)
    top = max(top, (H - total) // 2 + 40)
    if has_art:
        framed_art(static, x.art_path, (1180, 190, 1800, 890))
    base = paste_layer(base, static)
    states = []
    for k in range(len(items) + 1):
        lay = layer()
        d = ImageDraw.Draw(lay)
        y = top
        for j, item in enumerate(items[:k]):
            cur = j == k - 1
            box = (120, y, 120 + col_w, y + heights[j])
            if cur:
                shadow_glow(lay, box, AMBER, blur=30, alpha=45)
            glass(lay, box, alpha=26 if cur else 13, outline=60 if cur else 32, accent=AMBER if cur else None)
            cx, cy = 186, y + heights[j] // 2
            d.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=(TEAL if not cur else AMBER) + (255,))
            d.text((cx, cy), str(j + 1), font=font(FONT_BLACK, 28), fill=DARK + (255,), anchor="mm")
            lines = wrap(d, item, bf, col_w - 170)
            ty = cy - (len(lines) * lh) // 2 + lh // 2
            for ln in lines:
                d.text((236, ty), ln, font=bf, fill=(TEXT if cur else TEXT2) + (255,), anchor="lm")
                ty += lh
            y += heights[j] + gap
        states.append((times[k - 1] if k else 0.0, paste_layer(base, lay)))
    return segments_from_states(states, length)


def _eq_color(tok: str):
    t = tok.lower()
    if re.match(r"^w\d*$", t):
        return TEAL
    if re.match(r"^x\d*$", t):
        return AMBER
    if t == "b":
        return VIOLET
    if re.match(r"^[a-z]\w*$", t):
        return TEXT
    return TEXT3


def v_equation(x: XScene, length):
    im = background()
    lay = layer()
    d = ImageDraw.Draw(lay)
    eq, _, cap = x.arg.partition("|")
    heading(lay, x.chapter or "The formula")
    box = (220, 330, W - 220, 700)
    shadow_glow(lay, box, TEAL, blur=50, alpha=60)
    glass(lay, box, radius=34, alpha=22, outline=55)
    f = font(FONT_MONO, 104)
    toks = re.findall(r"[A-Za-z]\w*|\d+|\S", eq.strip())
    widths = [d.textlength(t, font=f) for t in toks]
    space = 22
    xpos = W / 2 - (sum(widths) + space * (len(toks) - 1)) / 2
    for t, w_ in zip(toks, widths):
        d.text((xpos, 515), t, font=f, fill=_eq_color(t) + (255,), anchor="lm")
        xpos += w_ + space
    legend = [(c, n) for c, n, rx in [(AMBER, "x  input", r"\bx\d*\b"), (TEAL, "w  weight", r"\bw\d*\b"),
                                      (VIOLET, "b  bias", r"\bb\b")] if re.search(rx, eq)]
    lx = W / 2 - len(legend) * 150
    for c, n in legend:
        d.rounded_rectangle([lx, 770, lx + 270, 830], radius=30, fill=c + (40,), outline=c + (220,), width=2)
        d.text((lx + 135, 800), n, font=font(FONT_SEMI, 32), fill=TEXT + (255,), anchor="mm")
        lx += 300
    if cap.strip() and not legend:
        d.text((W // 2, 800), cap.strip(), font=font(FONT, 42), fill=TEXT2 + (255,), anchor="mm")
    return [(paste_layer(im, lay), length)]


def _code_tokens(code: str, lang: str):
    from pygments import lex
    from pygments.lexers import get_lexer_by_name
    from pygments.token import Comment, Keyword, Name, Number, Operator, String
    colors = [(Comment, (110, 126, 150)), (Keyword, (199, 146, 234)), (String, (195, 232, 141)),
              (Number, (247, 140, 108)), (Name.Function, (130, 170, 255)), (Name.Builtin, (255, 203, 107)),
              (Name.Namespace, (255, 203, 107)), (Operator, (137, 221, 255))]
    lines, cur = [], []
    for tok, val in lex(code, get_lexer_by_name(lang or "python")):
        color = next((c for t, c in colors if tok in t), (220, 226, 236))
        parts = val.split("\n")
        for k, part in enumerate(parts):
            if part:
                cur.append((part, color))
            if k < len(parts) - 1:
                lines.append(cur)
                cur = []
    if cur:
        lines.append(cur)
    return lines


def v_code(x: XScene, length):
    lang, _, fname = x.arg.partition("|")
    # lines starting with "> " are what the program prints; they appear under the code as its output
    raw = x.code.split("\n")
    outs = [ln[2:] for ln in raw if ln.startswith("> ")]
    src = "\n".join(ln for ln in raw if not ln.startswith("> ")).strip("\n")
    lines = _code_tokens(src, lang.strip()) if src else []
    n_code = len(lines)
    if outs:
        lines = lines + [[("Output:", (120, 132, 156))]] + [[(o, AMBER)] for o in outs]
    n = len(lines)
    big = THEME == "kids" and n <= 10  # young viewers: short programs get a larger, easier-to-read font
    size = 46 if big else 36 if n <= 14 else 31
    lh = 64 if big else 52 if n <= 14 else 45
    # shrink until the longest line fits inside the window (long comments used to run off the edge)
    texts = ["".join(t for t, _ in toks) for toks in lines]
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    while size > 22 and max(probe.textlength(s, font=font(FONT_MONO, size)) for s in texts) > W - 180 - 280 - 30:
        size -= 1
        lh = round(lh * size / (size + 1))
    mono = font(FONT_MONO, size)
    win_h = 90 + n * lh + 40
    y0 = max(130, (H - win_h) // 2) if big else 130  # kids: short programs sit in the middle of the screen
    win = (180, y0, W - 180, min(H - 120, y0 + win_h))
    top = win[1] + 90
    base = background()
    static = layer()
    shadow_glow(static, win, (0, 0, 0), radius=24, blur=40, alpha=200)
    ds = ImageDraw.Draw(static)
    ds.rounded_rectangle(win, radius=24, fill=(17, 22, 36, 250), outline=(255, 255, 255, 40), width=2)
    ds.rounded_rectangle([win[0], win[1], win[2], win[1] + 62], radius=24, fill=(26, 32, 50, 255))
    ds.rectangle([win[0], win[1] + 40, win[2], win[1] + 62], fill=(26, 32, 50, 255))
    for c, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        ds.ellipse([win[0] + 28 + c * 32, win[1] + 22, win[0] + 46 + c * 32, win[1] + 40], fill=col + (255,))
    tab = (fname.strip() or "main.py")
    ds.rounded_rectangle([win[0] + 150, win[1] + 12, win[0] + 190 + ds.textlength(tab, font=font(FONT_SEMI, 24)),
                          win[1] + 62], radius=10, fill=(17, 22, 36, 255))
    ds.text((win[0] + 170, win[1] + 38), tab, font=font(FONT_SEMI, 24), fill=SOFT + (255,), anchor="lm")
    base = paste_layer(base, static)
    typing_end = max(1.0, min(length * 0.75, length - 1.0))
    states = []
    for k in range(n + 1):
        lay = layer()
        d = ImageDraw.Draw(lay)
        if k:
            ly = top + (k - 1) * lh
            d.rectangle([win[0] + 2, ly - 4, win[2] - 2, ly + lh - 8], fill=(32, 214, 186, 26))
            d.rectangle([win[0] + 2, ly - 4, win[0] + 7, ly + lh - 8], fill=TEAL + (255,))
        for li, toks in enumerate(lines[:k]):
            if li < n_code:
                d.text((win[0] + 70, top + li * lh), f"{li + 1}", font=mono, fill=(80, 92, 116, 255), anchor="ra")
            elif li == n_code and n_code:  # divider between the code and its output
                d.line([win[0] + 30, top + li * lh - 10, win[2] - 30, top + li * lh - 10],
                       fill=(255, 255, 255, 40), width=2)
            xpos = win[0] + 100
            for txt, col in toks:
                d.text((xpos, top + li * lh), txt, font=mono, fill=col + (255,))
                xpos += d.textlength(txt, font=mono)
        if k:
            cx = win[0] + 100 + sum(d.textlength(t, font=mono) for t, _ in lines[k - 1])
            d.rectangle([cx + 4, top + (k - 1) * lh + 4, cx + 18, top + (k - 1) * lh + lh - 12], fill=TEAL + (255,))
        states.append((typing_end * k / max(1, n) if k else 0.0, paste_layer(base, lay)))
    return segments_from_states(states, length)


def v_nn(x: XScene, length):
    sizes = [int(s) for s in re.findall(r"\d+", x.arg)] or [3, 4, 2]
    L = len(sizes)
    xs = [360 + (W - 720) * i / max(1, L - 1) for i in range(L)]
    spacing = min(150, 600 / max(1, max(sizes) - 1))
    pos = [[(xs[i], 590 + (j - (n - 1) / 2) * spacing) for j in range(n)] for i, n in enumerate(sizes)]
    labels = ["Input"] + ["Hidden"] * (L - 2) + ["Output"]
    base_bg = background()
    static = layer()
    heading(static, x.chapter or "The network")
    base_bg = paste_layer(base_bg, static)

    def draw(active, pulse=None):
        lay = layer()
        d = ImageDraw.Draw(lay)
        for i in range(L - 1):
            lit = active is not None and i < active
            for a in pos[i]:
                for b in pos[i + 1]:
                    d.line([a, b], fill=(TEAL + (150,)) if lit else (90, 110, 150, 60), width=3 if lit else 2)
        if pulse is not None:  # bright dots travelling along the edges of the layer being fired
            i, frac = pulse
            for a in pos[i]:
                for b in pos[i + 1]:
                    px, py = a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac
                    d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=AMBER + (230,))
        glow = layer()
        gd = ImageDraw.Draw(glow)
        for i, lyr in enumerate(pos):
            lit = active is not None and i <= active
            for (cx, cy) in lyr:
                if lit:
                    gd.ellipse([cx - 46, cy - 46, cx + 46, cy + 46], fill=AMBER + (150,))
        lay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))
        for i, lyr in enumerate(pos):
            lit = active is not None and i <= active
            for (cx, cy) in lyr:
                d.ellipse([cx - 28, cy - 28, cx + 28, cy + 28], fill=(AMBER if lit else (24, 32, 52)) + (255,),
                          outline=(255, 230, 170, 255) if lit else (110, 128, 160, 255), width=3)
            lab = labels[i]
            tw = d.textlength(lab, font=font(FONT_SEMI, 26)) + 40
            d.rounded_rectangle([xs[i] - tw / 2, 900, xs[i] + tw / 2, 946], radius=23,
                                fill=((32, 214, 186, 40) if lit else (255, 255, 255, 12)),
                                outline=(TEAL + (220,)) if lit else (255, 255, 255, 40), width=2)
            d.text((xs[i], 923), lab, font=font(FONT_SEMI, 26), fill=TEXT + (255,), anchor="mm")
        return paste_layer(base_bg, lay)
    states = [(0.0, draw(None))]
    step = max(0.5, min(1.1, (length - 0.6) / (L + 1)))
    t = 0.6
    while t < length - 0.3:
        for a in range(L):
            states.append((t, draw(a)))
            if a < L - 1:
                for f in (0.33, 0.66):
                    states.append((t + step * f, draw(a, (a, f))))
            t += step
        states.append((t, draw(None)))
        t += step * 0.6
    return segments_from_states(states, length)


FUNCS = {
    "sigmoid": (lambda v: 1 / (1 + math.exp(-v)), (-6, 6), (-0.2, 1.2)),
    "relu": (lambda v: max(0.0, v), (-4, 4), (-1, 4)),
    "tanh": (math.tanh, (-4, 4), (-1.3, 1.3)),
    "line": (lambda v: v, (-4, 4), (-4, 4)),
    "parabola": (lambda v: v * v, (-3, 3), (-1, 9)),
}
PLOT_NAMES = {"sigmoid": "Sigmoid", "relu": "ReLU", "tanh": "Tanh", "line": "Linear", "parabola": "x²"}


def v_plot(x: XScene, length):
    names = [n.strip().lower() for n in x.arg.split(",") if n.strip().lower() in FUNCS] or ["sigmoid"]
    n = len(names)
    pw = (W - 240 - (n - 1) * 70) / n
    base_bg = background()
    static = layer()
    top_y = heading(static, x.chapter or "Plot")
    boxes = []
    for k, name in enumerate(names):
        left = 120 + k * (pw + 70)
        box = (left, max(top_y, 300), left + pw, 960)
        glass(static, box, radius=28, alpha=16)
        boxes.append(box)
        sd = ImageDraw.Draw(static)
        f, (x0, x1), (y0, y1) = FUNCS[name]
        L_, T_, R_, B_ = box[0] + 70, box[1] + 90, box[2] - 40, box[3] - 60
        sx = lambda v: L_ + (v - x0) / (x1 - x0) * (R_ - L_)  # noqa: E731
        sy = lambda v: B_ - (v - y0) / (y1 - y0) * (B_ - T_)  # noqa: E731
        for gx in range(math.ceil(x0), math.floor(x1) + 1):
            sd.line([(sx(gx), T_), (sx(gx), B_)], fill=(255, 255, 255, 16), width=1)
            sd.text((sx(gx), B_ + 24), str(gx), font=font(FONT, 22), fill=TEXT3 + (255,), anchor="mm")
        for gy in range(math.ceil(y0), math.floor(y1) + 1):
            sd.line([(L_, sy(gy)), (R_, sy(gy))], fill=(255, 255, 255, 16), width=1)
            sd.text((L_ - 22, sy(gy)), str(gy), font=font(FONT, 22), fill=TEXT3 + (255,), anchor="rm")
        sd.line([(L_, sy(0)), (R_, sy(0))], fill=(255, 255, 255, 70), width=2)
        sd.line([(sx(0), T_), (sx(0), B_)], fill=(255, 255, 255, 70), width=2)
        sd.text(((box[0] + box[2]) / 2, box[1] + 46), PLOT_NAMES.get(name, name.title()),
                font=font(FONT_BOLD, 44), fill=TEXT + (255,), anchor="mm")
    base_bg = paste_layer(base_bg, static)
    steps = 16

    def draw(progress):
        lay, glow = layer(), layer()
        d, gd = ImageDraw.Draw(lay), ImageDraw.Draw(glow)
        for box, name in zip(boxes, names):
            f, (x0, x1), (y0, y1) = FUNCS[name]
            L_, T_, R_, B_ = box[0] + 70, box[1] + 90, box[2] - 40, box[3] - 60
            pts = [(L_ + (v - x0) / (x1 - x0) * (R_ - L_), B_ - (f(v) - y0) / (y1 - y0) * (B_ - T_))
                   for v in [x0 + (x1 - x0) * i / 300 for i in range(int(301 * progress))]]
            if len(pts) > 1:
                gd.line(pts, fill=AMBER + (200,), width=16, joint="curve")
                d.line(pts, fill=(255, 220, 140, 255), width=6, joint="curve")
        lay = Image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)), lay)
        return paste_layer(base_bg, lay)
    states = [(min(length * 0.6, 3.2) * s / steps, draw(s / steps)) for s in range(steps + 1)]
    return segments_from_states(states, length)


def v_steps(x: XScene, length):
    """Worked example: rows 'label = value' revealed one per sentence; a row starting with '=>' is the result."""
    rows = [r.strip() for r in x.arg.split("|") if r.strip()]
    times = reveal_times(x.scene, len(rows), [r.lstrip('=>') for r in rows])
    base = background()
    static = layer()
    top = heading(static, x.chapter or "Worked example")
    rh = 92
    box = (220, top + 10, W - 220, top + 10 + rh * len(rows) + 60)
    shadow_glow(static, box, TEAL, blur=44, alpha=40)
    glass(static, box, radius=30, alpha=16)
    base = paste_layer(base, static)
    states = []
    for k in range(len(rows) + 1):
        lay = layer()
        d = ImageDraw.Draw(lay)
        y = box[1] + 30
        for j, row in enumerate(rows[:k]):
            result = row.startswith("=>")
            label, _, value = row.lstrip("=>").partition("=")
            cur = j == k - 1
            if result:
                d.line([(box[0] + 50, y - 8), (box[2] - 50, y - 8)], fill=(255, 255, 255, 60), width=2)
            if cur:
                d.rounded_rectangle([box[0] + 20, y - 4, box[2] - 20, y + rh - 16], radius=16, fill=(255, 190, 72, 30))
            lf = font(FONT_BOLD if result else FONT_SEMI, 46 if result else 42)
            d.text((box[0] + 60, y + rh // 2 - 10), label.strip(), font=lf,
                   fill=(AMBER if result else (TEXT if cur else TEXT2)) + (255,), anchor="lm")
            vf = font(FONT_MONO, 54 if result else 46)
            d.text((box[2] - 60, y + rh // 2 - 10), value.strip(), font=vf,
                   fill=(AMBER if result else (TEAL if cur else TEXT)) + (255,), anchor="rm")
            y += rh
        states.append((times[k - 1] if k else 0.0, paste_layer(base, lay)))
    return segments_from_states(states, length)


def v_fit(x: XScene, length):
    """Data points with a sensible fit vs a wiggly overfit curve (arg: 'good', 'overfit' or 'both')."""
    modes = ["good", "overfit"] if x.arg.strip().lower() in ("", "both") else [x.arg.strip().lower()]
    rng = np.random.default_rng(4)
    xs_ = np.linspace(-1, 1, 13)
    ys_ = 0.6 * np.sin(2.2 * xs_) + rng.normal(0, 0.18, len(xs_))
    n = len(modes)
    pw = (W - 240 - (n - 1) * 70) / n
    base = background()
    static = layer()
    top = heading(static, x.chapter or "Fitting the data")
    boxes = []
    for k, mode in enumerate(modes):
        left = 120 + k * (pw + 70)
        box = (left, max(top, 300), left + pw, 960)
        glass(static, box, radius=28, alpha=16)
        ImageDraw.Draw(static).text(((box[0] + box[2]) / 2, box[1] + 46),
                                    {"good": "Good fit", "overfit": "Overfitting"}.get(mode, mode.title()),
                                    font=font(FONT_BOLD, 44), fill=(TEAL if mode == "good" else AMBER) + (255,),
                                    anchor="mm")
        boxes.append(box)
    base = paste_layer(base, static)
    fits = {"good": np.poly1d(np.polyfit(xs_, ys_, 3)), "overfit": np.poly1d(np.polyfit(xs_, ys_, len(xs_) - 1))}

    def draw(points, progress):
        lay, glow = layer(), layer()
        d, gd = ImageDraw.Draw(lay), ImageDraw.Draw(glow)
        for box, mode in zip(boxes, modes):
            L_, T_, R_, B_ = box[0] + 60, box[1] + 100, box[2] - 60, box[3] - 50
            sx = lambda v: L_ + (v + 1.1) / 2.2 * (R_ - L_)  # noqa: E731
            sy = lambda v: min(B_, max(T_, B_ - (v + 1.3) / 2.6 * (B_ - T_)))  # noqa: E731
            if progress > 0:
                vs = np.linspace(-1.05, 1.05, 400)[: int(400 * progress)]
                pts = [(sx(v), sy(fits[mode](v))) for v in vs]
                col = TEAL if mode == "good" else AMBER
                if len(pts) > 1:
                    gd.line(pts, fill=col + (190,), width=14, joint="curve")
                    d.line(pts, fill=col + (255,), width=5, joint="curve")
            for px, py in list(zip(xs_, ys_))[:points]:
                d.ellipse([sx(px) - 11, sy(py) - 11, sx(px) + 11, sy(py) + 11], fill=TEXT + (255,),
                          outline=BG_BASE + (255,), width=3)
        return paste_layer(base, Image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(9)), lay))
    states = [(0.0, draw(0, 0))]
    t = 0.3
    for p in range(1, len(xs_) + 1):
        states.append((t, draw(p, 0)))
        t += 0.12
    for s in range(1, 13):
        states.append((t, draw(len(xs_), s / 12)))
        t += 0.18
    return segments_from_states(states, length)


def v_quiz(x: XScene, length):
    """[quiz: Question? | Option A | Option B | correct option] + [art: picture]. The answer is revealed
    at the start of the scene's last narration line, so kids get a moment to guess."""
    parts = [p.strip() for p in x.arg.split("|")]
    question, options = parts[0], [p for p in parts[1:-1] if p] or ["Yes", "No"]
    answer = parts[-1] if len(parts) > 2 else options[0]
    reveal = x.scene.sentence_times[-1][0] if len(x.scene.sentence_times) > 1 else length * 0.6
    base = background()
    static = layer()
    if x.art_path:
        framed_art(static, x.art_path, (120, 170, 840, 890))
    left = 940 if x.art_path else 200
    d = ImageDraw.Draw(static)
    qf = font(FONT_BLACK, 64)
    y = 230
    for ln in wrap(d, question, qf, W - left - 120)[:3]:
        d.text((left, y), ln, font=qf, fill=TEXT + (255,))
        y += 80
    base = paste_layer(base, static)
    opt_top = max(y + 60, 520)

    def draw(show):
        lay = layer()
        dd = ImageDraw.Draw(lay)
        for k, opt in enumerate(options):
            box = (left, opt_top + k * 150, W - 140, opt_top + k * 150 + 120)
            right = show and opt.lower() == answer.lower()
            dim = show and not right
            glass(lay, box, radius=60, alpha=26, accent=(TEAL if right else (AMBER if not dim else None)))
            dd.ellipse([box[0] + 26, box[1] + 26, box[0] + 94, box[1] + 94],
                       fill=(TEAL if right else (TEXT3 if dim else AMBER)) + (255,))
            dd.text((box[0] + 60, box[1] + 60), "ABCD"[k], font=font(FONT_BLACK, 36), fill=DARK + (255,), anchor="mm")
            of = font(FONT_BLACK, 52)
            dd.text((box[0] + 130, box[1] + 60), opt, font=of,
                    fill=(TEAL if right else (TEXT3 if dim else TEXT)) + (255,), anchor="lm")
            if right:  # drawn checkmark (fonts may lack the glyph)
                cx = box[0] + 130 + dd.textlength(opt, font=of) + 60
                cy = box[1] + 60
                dd.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], fill=TEAL + (255,))
                dd.line([(cx - 16, cy + 1), (cx - 4, cy + 14), (cx + 18, cy - 12)], fill=DARK + (255,), width=8,
                        joint="curve")
        return paste_layer(base, lay)
    states = [(0.0, draw(False)), (reveal, draw(True))]
    if THEME == "kids":  # celebrate the right answer: a 1 s confetti burst from the correct option
        k = next((i for i, o in enumerate(options) if o.lower() == answer.lower()), 0)
        ox, oy = (left + W - 140) / 2, opt_top + k * 150 + 60
        rng = np.random.default_rng(x.scene.index)
        parts_ = [(rng.uniform(-1, 1) * 900, rng.uniform(-1.3, -0.4) * 900, rng.integers(len(STICKER_COLORS)),
                   rng.uniform(22, 40)) for _ in range(140)]
        shown = draw(True)
        for f in range(1, 25):
            t = f / FPS
            lay = layer()
            dc = ImageDraw.Draw(lay)
            for vx, vy, c, sz in parts_:
                px, py = ox + vx * t, oy + vy * t + 1400 * t * t  # thrown up, then falls with gravity
                a = int(255 * min(1.0, max(0.0, 1.6 - 1.6 * t)))
                dc.rectangle([px, py, px + sz, py + sz * 0.6], fill=STICKER_COLORS[c] + (a,))
            states.append((reveal + t, paste_layer(shown, lay)))
        states.append((reveal + 25 / FPS, shown))
    return segments_from_states(states, length)


def host_background(x: XScene, path: str) -> tuple:
    """Slide for a talking-characters shot: window on the left for the LTX clip, caption on the right.
    Returns the window box so the clip can be composited into it."""
    lay = layer()
    win = (150, 90, 150 + 506, 90 + 900)
    shadow_glow(lay, win, TEAL, radius=40, blur=40, alpha=90)
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle([win[0] - 8, win[1] - 8, win[2] + 8, win[3] + 8], radius=46, fill=(255, 255, 255, 255))
    caption = x.arg.partition("|")[2].strip()
    names = " & ".join(dict.fromkeys(e.speaker for e in x.scene.events if e.speaker != "Narrator"))
    if names:
        tw = d.textlength(names, font=font(FONT_SEMI, 30)) + 50
        d.rounded_rectangle([760, 330, 760 + tw, 384], radius=27, fill=TEAL + (60,), outline=TEAL + (255,), width=2)
        d.text((785, 357), names, font=font(FONT_SEMI, 30), fill=TEXT + (255,), anchor="lm")
    if caption:
        cf = font(FONT_BLACK, 84)
        y = 420
        for k, ln in enumerate(wrap(d, caption, cf, W - 760 - 140)[:3]):
            d.text((760, y), ln, font=cf, fill=(AMBER if k else TEXT) + (255,))
            y += 100
    paste_layer(background(), lay).save(path)
    return win


def _box(lay, box, title, sub="", lit=True, color=None, radius=24):
    d = ImageDraw.Draw(lay)
    color = color or TEAL
    if lit:
        shadow_glow(lay, box, color, radius=radius, blur=26, alpha=55)
    glass(lay, box, radius=radius, alpha=24 if lit else 10, outline=70 if lit else 30, accent=color if lit else None)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    tf = font(FONT_BOLD, 38 if len(title) < 14 else 32)
    lines = wrap(d, title, tf, box[2] - box[0] - 40)[:2]
    yy = cy - (len(lines) * 44 + (40 if sub else 0)) / 2 + 22
    for ln in lines:
        d.text((cx, yy), ln, font=tf, fill=(TEXT if lit else TEXT3) + (255,), anchor="mm")
        yy += 44
    if sub:
        d.text((cx, yy + 6), sub, font=font(FONT, 28), fill=(TEXT2 if lit else TEXT3) + (255,), anchor="mm")


def _arrow(d, a, b, color, width=5, head=18):
    d.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45))
    p2 = (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))
    d.polygon([b, p1, p2], fill=color)


def v_flow(x: XScene, length):
    """[flow: Question -> Retrieve: top chunks -> Prompt -> LLM -> Answer | loop]
    Boxes appear one per narration sentence; '| loop' draws an arrow from the last box back to the first."""
    spec, _, opts = x.arg.partition("|")
    steps = [s.strip() for s in re.split(r"->|→", spec) if s.strip()]
    loop = "loop" in opts.lower()
    n = len(steps)
    times = reveal_times(x.scene, n, [st.replace(":", " ") for st in steps])
    base = background()
    static = layer()
    top = heading(static, x.chapter or "How it flows")
    base = paste_layer(base, static)
    gap = 60
    bw = min(320, (W - 240 - gap * (n - 1)) / n)
    bh = 220
    y0 = max(top + 40, (H - bh) // 2 + 10)
    x0 = (W - (n * bw + (n - 1) * gap)) / 2
    boxes = [(x0 + i * (bw + gap), y0, x0 + i * (bw + gap) + bw, y0 + bh) for i in range(n)]
    states = []
    for k in range(n + 1):
        lay = layer()
        d = ImageDraw.Draw(lay)
        for i, (box, st) in enumerate(zip(boxes, steps)):
            if i >= k:
                continue
            title, _, sub = st.partition(":")
            _box(lay, box, title.strip(), sub.strip(), lit=True, color=(AMBER if i == k - 1 else TEAL))
            if i > 0:
                _arrow(d, (boxes[i - 1][2] + 8, y0 + bh / 2), (box[0] - 8, y0 + bh / 2), TEAL + (255,))
        if loop and k == n and n > 1:
            ly = y0 + bh + 70
            d.line([(boxes[-1][0] + bw / 2, y0 + bh + 8), (boxes[-1][0] + bw / 2, ly)], fill=AMBER + (255,), width=5)
            d.line([(boxes[-1][0] + bw / 2, ly), (boxes[0][0] + bw / 2, ly)], fill=AMBER + (255,), width=5)
            _arrow(d, (boxes[0][0] + bw / 2, ly), (boxes[0][0] + bw / 2, y0 + bh + 10), AMBER + (255,))
            d.text((W / 2, ly + 34), "repeat", font=font(FONT_SEMI, 30), fill=AMBER + (255,), anchor="mm")
        states.append((times[k - 1] if k else 0.0, paste_layer(base, lay)))
    return segments_from_states(states, length)


def v_rnn(x: XScene, length):
    """[rnn: The | cat | sat | down | fade] unrolled recurrent network reading word by word.
    'fade' dims the memory of early words as the sequence grows (vanishing gradient)."""
    parts = [p.strip() for p in x.arg.split("|") if p.strip()]
    fade = parts and parts[-1].lower() == "fade"
    words = parts[:-1] if fade else parts
    words = words or ["The", "cat", "sat", "down"]
    n = len(words)
    base = background()
    static = layer()
    heading(static, x.chapter or "Unrolled through time")
    base = paste_layer(base, static)
    gap = 70
    cw = min(230, (W - 240 - gap * (n - 1)) / n)
    x0 = (W - (n * cw + (n - 1) * gap)) / 2
    cy0, ch = 430, 170
    step = max(0.6, min(1.4, (length - 1.0) / (n + 1)))

    def draw(k):
        lay = layer()
        d = ImageDraw.Draw(lay)
        for i, w_ in enumerate(words):
            bx = x0 + i * (cw + gap)
            box = (bx, cy0, bx + cw, cy0 + ch)
            seen = i < k
            strength = 1.0 if not fade else max(0.18, 1 - 0.28 * (k - 1 - i)) if seen else 0.0
            # input word below
            wf = font(FONT_BOLD, 40)
            d.rounded_rectangle([bx + 20, 760, bx + cw - 20, 840], radius=40,
                                fill=(AMBER + (60,)) if seen else (255, 255, 255, 14),
                                outline=(AMBER + (255,)) if seen else (255, 255, 255, 50), width=2)
            d.text((bx + cw / 2, 800), w_, font=wf, fill=(TEXT if seen else TEXT3) + (255,), anchor="mm")
            if seen:
                _arrow(d, (bx + cw / 2, 752), (bx + cw / 2, cy0 + ch + 10), AMBER + (255,), width=4)
            col = tuple(int(c * strength + 40 * (1 - strength)) for c in TEAL)
            _box(lay, box, f"h{i + 1}", "memory" if i == k - 1 else "", lit=seen and strength > 0.45, color=col)
            if i > 0 and seen:
                _arrow(d, (box[0] - gap + 6, cy0 + ch / 2), (box[0] - 8, cy0 + ch / 2),
                       TEAL + (int(255 * max(0.3, strength)),), width=5)
        if fade and k == n:
            d.text((W / 2, 930), "early words fade from memory", font=font(FONT_SEMI, 36), fill=AMBER + (255,),
                   anchor="mm")
        return paste_layer(base, lay)
    states = [(0.0, draw(0))] + [(0.5 + i * step, draw(i + 1)) for i in range(n)]
    return segments_from_states(states, length)


def v_gates(x: XScene, length):
    """[gates: lstm] or [gates: gru]: cell diagram; one part lights up per narration sentence.
    LSTM order: cell state, forget, input, output.  GRU order: hidden state, reset, update."""
    kind = (x.arg.strip().lower() or "lstm")
    if kind == "gru":
        parts = [("Hidden state", "carries the memory"), ("Reset gate", "how much past to use"),
                 ("Update gate", "keep old vs take new")]
    else:
        parts = [("Cell state", "the long-term memory"), ("Forget gate", "what to erase"),
                 ("Input gate", "what to write"), ("Output gate", "what to reveal")]
    n = len(parts)
    times = reveal_times(x.scene, n, [f"{a} {b}" for a, b in parts])
    base = background()
    static = layer()
    heading(static, x.chapter or kind.upper())
    base = paste_layer(base, static)
    cell = (260, 300, W - 260, 900)

    def draw(active):
        lay = layer()
        d = ImageDraw.Draw(lay)
        glass(lay, cell, radius=36, alpha=12, outline=40)
        d.text((cell[0] + 40, cell[1] + 36), kind.upper() + " cell", font=font(FONT_SEMI, 30), fill=TEXT3 + (255,))
        lane_y = cell[1] + 120  # memory "conveyor belt" across the top
        lit0 = active is not None and active >= 0
        d.line([(cell[0] - 120, lane_y), (cell[2] + 120, lane_y)], fill=(TEAL if active == 0 else (90, 110, 150)) +
               (255,), width=10 if active == 0 else 6)
        _arrow(d, (cell[2] + 60, lane_y), (cell[2] + 125, lane_y), (TEAL if lit0 else (90, 110, 150)) + (255,))
        d.text((cell[0] - 120, lane_y - 40), parts[0][0], font=font(FONT_SEMI, 28),
               fill=(TEAL if active == 0 else TEXT3) + (255,))
        gn = n - 1
        gw = (cell[2] - cell[0] - 140 - 60 * (gn - 1)) / gn
        for g in range(gn):
            gx = cell[0] + 70 + g * (gw + 60)
            box = (gx, cell[1] + 260, gx + gw, cell[1] + 460)
            on = active == g + 1
            _box(lay, box, parts[g + 1][0], parts[g + 1][1], lit=on or (active is not None and active > g + 1),
                 color=AMBER if on else TEAL)
            nx = gx + gw / 2  # gate's effect on the memory lane
            d.line([(nx, box[1] - 6), (nx, lane_y + 26)], fill=(AMBER if on else (90, 110, 150)) + (255,),
                   width=5 if on else 3)
            d.ellipse([nx - 24, lane_y - 24, nx + 24, lane_y + 24], fill=((AMBER if on else (40, 52, 80)) + (255,)),
                      outline=TEXT3 + (255,), width=2)
            sym = "×" if g == 0 else ("+" if g == 1 else "→")
            d.text((nx, lane_y), sym, font=font(FONT_BOLD, 34), fill=(DARK if on else TEXT) + (255,), anchor="mm")
        d.text((cell[0] + 40, cell[3] - 50), "input word  +  previous memory", font=font(FONT, 28),
               fill=TEXT3 + (255,))
        return paste_layer(base, lay)
    states = [(0.0, draw(None))] + [(times[k], draw(k)) for k in range(n)]
    return segments_from_states(states, length)


def _grid(d, x0, y0, vals, cell, vmin, vmax, color, label=None, hi=None, numbers=True):
    rows, cols = len(vals), len(vals[0])
    for r in range(rows):
        for c in range(cols):
            v = vals[r][c]
            box = [x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - 3, y0 + (r + 1) * cell - 3]
            if v is None:
                d.rectangle(box, fill=(255, 255, 255, 10))
                continue
            t = 0 if vmax == vmin else (v - vmin) / (vmax - vmin)
            fill = tuple(int(30 + (cc - 30) * t) for cc in color) + (255,)
            d.rectangle(box, fill=fill)
            if numbers:
                d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), f"{v:g}", font=font(FONT_MONO, int(cell * 0.36)),
                       fill=(255, 255, 255, 230) if t < 0.6 else (10, 14, 28, 255), anchor="mm")
    if hi:
        r0, c0, rh, cw_ = hi
        d.rectangle([x0 + c0 * cell - 4, y0 + r0 * cell - 4, x0 + (c0 + cw_) * cell + 1, y0 + (r0 + rh) * cell + 1],
                    outline=AMBER + (255,), width=6)
    if label:
        d.text((x0 + cols * cell / 2, y0 - 40), label, font=font(FONT_SEMI, 32), fill=TEXT + (255,), anchor="mm")


def v_conv(x: XScene, length):
    """[conv: edge] a 3x3 vertical-edge filter slides over an image and builds the feature map (real values).
    [conv: pool] 2x2 max pooling of a 4x4 feature map."""
    base = background()
    static = layer()
    heading(static, x.chapter or "Convolution")
    base = paste_layer(base, static)
    if x.arg.strip().lower() == "pool":
        fm = [[1, 3, 2, 0], [4, 6, 1, 2], [0, 2, 9, 5], [1, 1, 3, 7]]
        out = [[max(fm[r][c], fm[r][c + 1], fm[r + 1][c], fm[r + 1][c + 1]) for c in (0, 2)] for r in (0, 2)]
        states = []
        for k in range(5):
            lay = layer()
            d = ImageDraw.Draw(lay)
            hi = None if k == 0 else ((k - 1) // 2 * 2, (k - 1) % 2 * 2, 2, 2)
            _grid(d, 330, 400, fm, 120, 0, 9, TEAL, "Feature map (4×4)", hi)
            shown = [[out[r][c] if r * 2 + c < k else None for c in range(2)] for r in range(2)]
            _grid(d, 1250, 520, shown, 120, 0, 9, AMBER, "Max pooled (2×2)")
            _arrow(d, (870, 640), (1190, 640), TEXT3 + (255,), width=5)
            d.text((1030, 600), "keep the max", font=font(FONT_SEMI, 30), fill=TEXT2 + (255,), anchor="mm")
            states.append((0.0 if k == 0 else min(length * 0.8, 0.8 + (k - 1) * 1.1), paste_layer(base, lay)))
        return segments_from_states(states, length)
    img = [[0, 0, 0, 1, 1, 1] for _ in range(6)]  # dark left half, bright right half: a vertical edge
    ker = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    outv = [[sum(img[r + i][c + j] * ker[i][j] for i in range(3) for j in range(3)) for c in range(4)] for r in range(4)]
    positions = [(r, c) for r in range(4) for c in range(4)]
    t_end = min(length * 0.7, 0.6 + 0.35 * len(positions))
    states = []
    for k in range(len(positions) + 1):
        lay = layer()
        d = ImageDraw.Draw(lay)
        hi = (positions[k - 1][0], positions[k - 1][1], 3, 3) if k else None
        _grid(d, 170, 360, img, 92, 0, 1, (240, 244, 252), "Input image", hi, numbers=True)
        _grid(d, 860, 500, ker, 92, -1, 1, VIOLET, "Filter (3×3)")
        shown = [[outv[r][c] if r * 4 + c < k else None for c in range(4)] for r in range(4)]
        _grid(d, 1340, 420, shown, 104, 0, 3, AMBER, "Feature map", None)
        d.text((W / 2, 960), "bright cells = the filter found a vertical edge", font=font(FONT_SEMI, 32),
               fill=TEXT2 + (255,), anchor="mm")
        states.append((t_end * k / len(positions) if k else 0.0, paste_layer(base, lay)))
    return segments_from_states(states, length)


def _chips(d, words, y, colors=None, ids=False, maxw=W - 240, size=48):
    f = font(FONT_SEMI, size)
    pad, gap = 26, 16
    widths = [d.textlength(w_, font=f) + 2 * pad for w_ in words]
    rows, cur, cw_ = [], [], 0
    for i, w_ in enumerate(widths):
        if cur and cw_ + w_ + gap > maxw:
            rows.append(cur)
            cur, cw_ = [], 0
        cur.append(i)
        cw_ += w_ + gap
    if cur:
        rows.append(cur)
    boxes = {}
    for r, idxs in enumerate(rows):
        total = sum(widths[i] for i in idxs) + gap * (len(idxs) - 1)
        xx = (W - total) / 2
        yy = y + r * (size + 2 * pad + (60 if ids else 30))
        for i in idxs:
            col = (colors[i] if colors else TEAL)
            box = (xx, yy, xx + widths[i], yy + size + 2 * pad - 10)
            d.rounded_rectangle(box, radius=18, fill=col + (60,), outline=col + (255,), width=3)
            d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), words[i], font=f, fill=TEXT + (255,), anchor="mm")
            if ids:
                tid = int(hashlib.md5(words[i].encode()).hexdigest()[:6], 16) % 50000
                d.text(((box[0] + box[2]) / 2, box[3] + 26), str(tid), font=font(FONT_MONO, 26),
                       fill=TEXT3 + (255,), anchor="mm")
            boxes[i] = box
            xx += widths[i] + gap
    return boxes


PALETTE = [(32, 214, 186), (255, 190, 72), (140, 110, 255), (255, 120, 140), (90, 170, 255)]


def v_tokens(x: XScene, length):
    """[tokens: Un|believ|able| results|!] text split into tokens (with ids); chips appear left to right."""
    toks = [t for t in x.arg.split("|") if t != ""]
    base = background()
    static = layer()
    heading(static, x.chapter or "Tokens")
    base = paste_layer(base, static)
    t_end = min(length * 0.6, 0.5 + 0.35 * len(toks))
    states = []
    for k in range(len(toks) + 1):
        lay = layer()
        d = ImageDraw.Draw(lay)
        shown = [t.replace(" ", "·") for t in toks[:k]]
        if shown:
            _chips(d, shown, 470, [PALETTE[i % len(PALETTE)] for i in range(k)], ids=True)
        states.append((t_end * k / len(toks) if k else 0.0, paste_layer(base, lay)))
    return segments_from_states(states, length)


def v_next(x: XScene, length):
    """[next: The cat sat on the | mat 62 | floor 21 | sofa 9] next-token prediction with probability bars."""
    parts = [p.strip() for p in x.arg.split("|")]
    context, cands = parts[0], []
    for p in parts[1:]:
        m = re.match(r"(.+?)\s+([\d.]+)%?$", p)
        if m:
            cands.append((m.group(1), float(m.group(2))))
    base = background()
    static = layer()
    heading(static, x.chapter or "Next-token prediction")
    d0 = ImageDraw.Draw(static)
    _chips(d0, context.split() + ["?"], 290, [TEAL] * len(context.split()) + [AMBER])
    base = paste_layer(base, static)
    reveal = reveal_times(x.scene, 3)
    states = [(0.0, base)]
    for k in range(1, len(cands) + 1):
        lay = layer()
        d = ImageDraw.Draw(lay)
        for i, (w_, p) in enumerate(cands[:k]):
            y = 560 + i * 110
            d.text((560, y + 35), w_, font=font(FONT_BOLD, 44), fill=TEXT + (255,), anchor="rm")
            d.rounded_rectangle([600, y, 600 + 900, y + 70], radius=20, fill=(255, 255, 255, 14))
            d.rounded_rectangle([600, y, 600 + 900 * p / 100, y + 70], radius=20,
                                fill=(AMBER if i == 0 else TEAL) + (230,))
            d.text((1530, y + 35), f"{p:g}%", font=font(FONT_MONO, 40), fill=TEXT + (255,), anchor="lm")
        states.append((min(length * 0.85, reveal[min(1, len(reveal) - 1)] + 0.5 * (k - 1)), paste_layer(base, lay)))
    return segments_from_states(states, length)


def v_attention(x: XScene, length):
    """[attention: The animal didn't cross the street because it was tired | it -> animal]
    Lines from the focus word to every other word; thicker = more attention."""
    sent, _, rule = x.arg.partition("|")
    words = sent.split()
    m = re.match(r"\s*(\S+)\s*->\s*(\S+)", rule)
    focus_w, target_w = (m.group(1), m.group(2)) if m else (words[-1], words[0])
    fi = next((i for i, w_ in enumerate(words) if w_.lower().strip(".,") == focus_w.lower()), len(words) - 1)
    ti = next((i for i, w_ in enumerate(words) if w_.lower().strip(".,") == target_w.lower()), 0)
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.02, 0.08, len(words))
    weights[ti] = 0.55
    weights[fi] = 0.10
    weights = weights / weights.sum()
    base = background()
    static = layer()
    heading(static, x.chapter or "Self-attention")
    base = paste_layer(base, static)

    def draw(show):
        lay = layer()
        d = ImageDraw.Draw(lay)
        cols = [AMBER if i == fi else (TEAL if (show and i == ti) else (90, 110, 150)) for i in range(len(words))]
        boxes = _chips(d, words, 640, cols, size=44)
        if show:
            fb = boxes[fi]
            fx, fy = (fb[0] + fb[2]) / 2, fb[1] - 6
            for i, w_ in enumerate(words):
                if i == fi:
                    continue
                b = boxes[i]
                tx, ty = (b[0] + b[2]) / 2, b[1] - 6
                peak = min(fy, ty) - 80 - 260 * abs(tx - fx) / W
                pts = [((1 - t) ** 2 * fx + 2 * (1 - t) * t * (fx + tx) / 2 + t ** 2 * tx,
                        (1 - t) ** 2 * fy + 2 * (1 - t) * t * peak + t ** 2 * ty) for t in np.linspace(0, 1, 40)]
                wgt = float(weights[i])
                d.line(pts, fill=(TEAL if i == ti else (150, 170, 210)) + (int(80 + 175 * min(1, wgt * 2)),),
                       width=max(2, int(wgt * 40)))
                if wgt > 0.08:
                    d.text((pts[20][0], pts[20][1] - 26), f"{wgt:.0%}", font=font(FONT_SEMI, 30),
                           fill=TEXT + (255,), anchor="mm")
        return paste_layer(base, lay)
    t1 = x.scene.sentence_times[1][0] if len(x.scene.sentence_times) > 1 else length * 0.3
    return segments_from_states([(0.0, draw(False)), (t1, draw(True))], length)


def v_image(x: XScene, length):
    return [(cover_crop(Image.open(x.art_path).convert("RGB"), W, H), length)]


STICKER_COLORS = [(255, 214, 10), (255, 120, 200), (60, 230, 200), (255, 150, 40), (150, 120, 255)]


def word_sticker(word: str, seed: int = 0) -> Image.Image:
    """A big, tilted, outlined cartoon word ("GAMES!") on a transparent full-frame layer."""
    lay = layer()
    if not word:
        return lay
    d = ImageDraw.Draw(lay)
    size = 150
    while size > 70 and d.textlength(word.upper(), font=font(FONT_BLACK, size)) > W * 0.55:
        size -= 6
    f = font(FONT_BLACK, size)
    tw = d.textlength(word.upper(), font=f)
    col = STICKER_COLORS[seed % len(STICKER_COLORS)]
    txt = layer()
    dt = ImageDraw.Draw(txt)
    x0, y0 = W - tw - 150, 120  # top-right, clear of the chapter badge
    dt.rounded_rectangle([x0 - 40, y0 - 20, x0 + tw + 40, y0 + size * 1.25], radius=40, fill=(255, 255, 255, 235),
                         outline=(25, 30, 60, 255), width=8)
    dt.text((x0, y0), word.upper(), font=f, fill=col + (255,), stroke_width=10, stroke_fill=(25, 30, 60, 255))
    return txt.rotate(-5 if seed % 2 else 4, center=(x0 + tw / 2, y0 + size / 2), resample=Image.BICUBIC)


def v_show(x: XScene, length):
    """Kids "show what's said": a full-screen cartoon of the line, popping in with a bounce, plus an optional
    word sticker. [show: picture prompt | WORD]"""
    word = x.arg.partition("|")[2].strip()
    art = cover_crop(Image.open(x.art_path).convert("RGB"), W, H).convert("RGBA")
    sticker = word_sticker(word, x.scene.index)
    states = []
    for k in range(13):  # 0.5 s bounce: 0.86 -> 1.04 -> 1.0, sticker pops in a little later
        t = k / FPS
        p = min(1.0, t / 0.42)
        s = 0.86 + 0.18 * math.sin(p * math.pi * 0.75) / math.sin(math.pi * 0.75) if p < 1 else 1.0
        s = min(s, 1.04) if p < 0.8 else 1.0 + (s - 1.0) * (1 - p) * 5
        frame = Image.new("RGBA", (W, H), BG_BASE + (255,))
        a = art.resize((int(W * s), int(H * s)), Image.BILINEAR)
        frame.alpha_composite(a, ((W - a.width) // 2, (H - a.height) // 2))
        if t >= 0.25 and word:
            q = min(1.0, (t - 0.25) / 0.2)
            st = sticker.resize((int(W * (0.6 + 0.4 * q)), int(H * (0.6 + 0.4 * q))), Image.BILINEAR)
            frame.alpha_composite(st, ((W - st.width) // 2 + int(W * 0.2 * (1 - q)), (H - st.height) // 2))
        states.append((t, frame.convert("RGB")))
    final = art.copy()
    final.alpha_composite(sticker)
    states.append((13 / FPS, final.convert("RGB")))
    return segments_from_states(states, length)


VISUALS = {"title": v_title, "bullets": v_bullets, "code": v_code, "nn": v_nn, "plot": v_plot,
           "equation": v_equation, "image": v_image, "chapter": v_bullets, "steps": v_steps, "fit": v_fit, "quiz": v_quiz,
           "flow": v_flow, "rnn": v_rnn, "gates": v_gates, "conv": v_conv, "tokens": v_tokens, "next": v_next,
           "attention": v_attention, "show": v_show}
ZOOM = {"code": 0.012, "image": 0.07, "title": 0.05, "show": 0.08}  # how far each visual pushes in over the scene
# Burned-in captions on picture scenes. Off: the full subtitles.srt is uploaded to YouTube instead, so viewers
# get captions for the whole video (or none) rather than only on some scenes.
BURN_CAPTIONS = False


def ensure_art(x: XScene, comfy, work: str) -> None:
    if x.visual == "host":  # for host shots [art: ...] describes the motion, not an illustration
        return
    prompt = x.arg.partition("|")[0].strip() if x.visual in ("image", "show", "motion") else x.art
    if not prompt:
        return
    full, seed = f"{prompt}. {ART_STYLE}", 11 + x.scene.index
    key = hashlib.sha1(f"{full}|{seed}".encode()).hexdigest()[:16]
    cache = os.path.join(os.path.dirname(work), "_art_cache", key + ".png")  # reused across renders
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.isfile(cache):
        comfy.text_to_image(full, cache, width=1280, height=704, steps=30, seed=seed)
    x.art_path = cache


def render_segments(x: XScene, length: float) -> list:
    return VISUALS.get(x.visual, v_bullets)(x, length)


def write_clip(x: XScene, segs: list, out: str, t0: float, total: float) -> None:
    """Render every frame: content slowly pushes in, chrome stays fixed, progress bar advances."""
    length = sum(s for _, s in segs)
    st = x.scene.sentence_times if x.visual == "image" and BURN_CAPTIONS else []
    overs = [(0.0, chrome(x, st[0][2] if st else ""))]
    for a, _, text in st[1:]:  # picture scenes: caption changes with every narration line
        overs.append((a, chrome(x, text)))
    zoom = ZOOM.get(x.visual, 0.03)
    proc = raw_writer(out, W, H, FPS)
    try:
        bounds = np.cumsum([0] + [s for _, s in segs])
        frames = max(1, round(length * FPS))
        for f in range(frames):
            t = f / FPS
            k = min(len(segs) - 1, int(np.searchsorted(bounds, t, side="right") - 1))
            img = segs[k][0]
            z = 1 + zoom * (t / max(length, 1e-6))
            vw, vh = W / z, H / z
            frame = img.transform((W, H), Image.EXTENT, ((W - vw) / 2, (H - vh) / 2, (W + vw) / 2, (H + vh) / 2),
                                  resample=Image.BICUBIC).convert("RGBA")
            frame.alpha_composite(next(o for t_, o in reversed(overs) if t_ <= t))
            d = ImageDraw.Draw(frame)
            p = min(1.0, (t0 + t) / max(total, 1e-6))
            d.rectangle([0, H - 6, W, H], fill=(255, 255, 255, 18))
            d.rectangle([0, H - 6, int(W * p), H], fill=TEAL + (255,))
            proc.stdin.write(frame.convert("RGB").tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(proc.stderr.read().decode(errors="ignore")[-2000:])
    finally:
        if proc.poll() is None:
            proc.kill()


THUMB_COLORS = {  # bright, saturated two-tone backgrounds for kids thumbnails
    "yellow": ((255, 214, 10), (255, 140, 0)), "blue": ((40, 200, 255), (30, 90, 230)),
    "pink": ((255, 120, 200), (170, 60, 230)), "green": ((120, 230, 80), (0, 160, 120)),
    "orange": ((255, 170, 40), (240, 70, 50)), "purple": ((170, 120, 255), (90, 40, 200)),
    "teal": ((60, 230, 200), (0, 140, 200)), "red": ((255, 110, 90), (210, 30, 70)),
}
KIDS_CUTOUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "projects", "nisa_baba", "images", "02_open_mouth_cutout.png")


def thumbnail_kids(out: str, text: str, badge: str = "", number: str = "", snippet: str = "",
                   color: str = "yellow", art: str | None = None, cutout: str = KIDS_CUTOUT) -> None:
    """Kids thumbnail, built to read at phone size: a big expressive Nisa & Baba sticker, 2-4 huge words,
    a saturated sunburst background, a series badge with the episode number, and an optional code sticker."""
    c1, c2 = THUMB_COLORS.get(color, THUMB_COLORS["yellow"])
    # diagonal gradient + sunburst rays from behind the characters
    gx = np.linspace(0, 1, W)[None, :] * 0.6 + np.linspace(0, 1, H)[:, None] * 0.4
    bg = (np.array(c1)[None, None] * (1 - gx[..., None]) + np.array(c2)[None, None] * gx[..., None]).astype(np.uint8)
    im = Image.fromarray(bg, "RGB").convert("RGBA")
    rays = layer()
    dr = ImageDraw.Draw(rays)
    cx, cy = int(W * 0.72), int(H * 0.45)
    for k in range(0, 360, 20):
        a1, a2 = math.radians(k), math.radians(k + 9)
        dr.polygon([(cx, cy), (cx + 2600 * math.cos(a1), cy + 2600 * math.sin(a1)),
                    (cx + 2600 * math.cos(a2), cy + 2600 * math.sin(a2))], fill=(255, 255, 255, 34))
    im = Image.alpha_composite(im, rays)
    if art and os.path.isfile(art):  # the episode's cartoon art as a tilted photo sticker, left-bottom
        a = cover_crop(Image.open(art).convert("RGB"), 440, 310).convert("RGBA")
        card = Image.new("RGBA", (a.width + 28, a.height + 28), (255, 255, 255, 255))
        card.paste(a, (14, 14))
        card = card.rotate(6, expand=True, resample=Image.BICUBIC)
        sh = Image.new("RGBA", card.size, (0, 0, 0, 0))
        sh.paste((0, 0, 0, 110), mask=card.split()[3])
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(14)), (70 + 10, H - card.height - 40 + 14))
        im.alpha_composite(card, (70, H - card.height - 40))
    if snippet:  # a little code window sticker, so parents see it's a coding lesson
        mono = font(FONT_MONO, 62)
        tw = int(ImageDraw.Draw(layer()).textlength(snippet, font=mono))
        st = Image.new("RGBA", (tw + 90, 170), (0, 0, 0, 0))
        ds = ImageDraw.Draw(st)
        ds.rounded_rectangle([0, 0, st.width - 1, st.height - 1], radius=26, fill=(22, 27, 45, 255),
                             outline=(255, 255, 255, 255), width=6)
        for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            ds.ellipse([26 + i * 30, 22, 44 + i * 30, 40], fill=col + (255,))
        x0 = 45
        for t, col in _code_tokens(snippet, "python")[0]:
            ds.text((x0, 112), t, font=mono, fill=col + (255,), anchor="lm")
            x0 += ds.textlength(t, font=mono)
        st = st.rotate(-4, expand=True, resample=Image.BICUBIC)
        sh = Image.new("RGBA", st.size, (0, 0, 0, 0))
        sh.paste((0, 0, 0, 120), mask=st.split()[3])
        sx = 70 if not art else 620
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)), (sx + 10, H - st.height - 60 + 14))
        im.alpha_composite(st, (sx, H - st.height - 60))
    if cutout and os.path.isfile(cutout):  # Nisa & Baba, big, with a white sticker outline and a shadow
        ch = Image.open(cutout).convert("RGBA")
        ch = ch.crop((0, 0, ch.width, int(ch.height * 0.6)))  # both faces, heads and shoulders
        s = 1000 / ch.height
        ch = ch.resize((int(ch.width * s), 1000), Image.LANCZOS)
        alpha = ch.split()[3]
        ring = alpha.filter(ImageFilter.MaxFilter(23))
        sticker = Image.new("RGBA", ch.size, (255, 255, 255, 0))
        sticker.putalpha(ring)
        sticker.alpha_composite(ch)
        sh = Image.new("RGBA", ch.size, (0, 0, 0, 0))
        sh.paste((0, 0, 0, 130), mask=ring)
        px, py = W - sticker.width + 30, H - sticker.height + 40
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(20)), (px - 18, py + 18))
        im.alpha_composite(sticker, (px, py))
    lay = layer()
    d = ImageDraw.Draw(lay)
    if badge:  # series badge + episode number, top-left
        bf = font(FONT_BLACK, 50)
        bw = int(d.textlength(badge.upper(), font=bf))
        x = 70
        if number:
            d.ellipse([x, 50, x + 120, 170], fill=(255, 255, 255, 255), outline=(25, 30, 60, 255), width=8)
            d.text((x + 60, 112), number, font=font(FONT_BLACK, 72), fill=(25, 30, 60, 255), anchor="mm")
            x += 140
        d.rounded_rectangle([x, 72, x + bw + 60, 150], radius=39, fill=(25, 30, 60, 255))
        d.text((x + 30, 111), badge.upper(), font=bf, fill=(255, 255, 255, 255), anchor="lm")
    # the hook: 2-4 huge words, white with a thick dark outline; last line in sunny yellow
    words = text.upper().split()
    lines = wrap(d, " ".join(words), font(FONT_BLACK, 200), 1050)[:2]
    size = 200
    while size > 110 and max(d.textlength(ln, font=font(FONT_BLACK, size)) for ln in lines) > 1080:
        size -= 6
    tf = font(FONT_BLACK, size)
    y = 215
    txt = layer()
    dt = ImageDraw.Draw(txt)
    for k, ln in enumerate(lines):
        fill = (255, 232, 40) if k == len(lines) - 1 and len(lines) > 1 else (255, 255, 255)
        dt.text((80, y), ln, font=tf, fill=fill + (255,), stroke_width=18, stroke_fill=(25, 30, 60, 255))
        y += int(size * 1.05)
    txt = txt.rotate(3, center=(80, 215), resample=Image.BICUBIC)
    sh = Image.new("RGBA", txt.size, (0, 0, 0, 0))
    sh.paste((0, 0, 0, 120), mask=txt.split()[3])
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)), (8, 12))
    im.alpha_composite(txt)
    im.alpha_composite(lay)
    im.convert("RGB").resize((1280, 720), Image.LANCZOS).save(out)


def thumbnail(title: str, out: str, image: str | None = None) -> None:
    im = background()
    if image and os.path.isfile(image):
        im = Image.blend(im, cover_crop(Image.open(image).convert("RGB"), W, H), 0.75)
    lay = layer()
    shade = ImageDraw.Draw(lay)
    for xx in range(W):
        shade.line([(xx, 0), (xx, H)], fill=BG_BASE + (int(230 * max(0.0, 1 - xx / (W * 0.8))),))
    d = ImageDraw.Draw(lay)
    tf = font(FONT_BLACK, 170)
    lines = wrap(d, title.upper(), tf, W - 300)[:3]
    y = H // 2 - len(lines) * 95
    for k, ln in enumerate(lines):
        d.text((110, y), ln, font=tf, fill=(AMBER if k == len(lines) - 1 else TEXT) + (255,),
               stroke_width=6, stroke_fill=(0, 0, 0, 255))
        y += 190
    d.rectangle([0, 0, 26, H], fill=TEAL + (255,))
    d.ellipse([110, 70, 186, 146], fill=TEAL + (255,))
    d.text((148, 108), "M", font=font(FONT_BLACK, 44), fill=DARK + (255,), anchor="mm")
    d.text((206, 108), CHANNEL, font=font(FONT_BOLD, 52), fill=TEXT + (255,), anchor="lm")
    paste_layer(im, lay).resize((1280, 720), Image.LANCZOS).save(out)
