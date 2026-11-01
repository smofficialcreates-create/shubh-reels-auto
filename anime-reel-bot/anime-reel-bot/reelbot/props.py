"""On-screen props: UI cards, rolling numbers, keyword scribbles, follow button.

Every function returns a BGRA uint8 sprite (straight alpha). Cards are drawn at 2x and
reduced, so edges stay smooth.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import fonts

ORANGE = (255, 86, 34)       # RGB
INK = (22, 22, 26)
PAPER_CARD = (255, 255, 255)
GREY = (140, 140, 146)
S = 2


def _to_bgra(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def ease_out_back(x, s=1.8):
    x = min(max(x, 0.0), 1.0)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


@lru_cache(maxsize=16)
def _shadow(w: int, h: int, r: int) -> Image.Image:
    pad = 46
    sh = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([pad, pad + 14, pad + w, pad + h + 14], radius=r, fill=(0, 0, 0, 70))
    return sh.filter(ImageFilter.GaussianBlur(18))


def card_base(w: int, h: int, left: str, right: str = "", dark: bool = False) -> Image.Image:
    """White card with header line, returned at 2x with a shadow margin."""
    pad = 46
    r = 26
    sh = _shadow(w, h, r).resize(((w + 2 * pad) * S, (h + 2 * pad) * S), Image.LANCZOS)
    img = sh
    d = ImageDraw.Draw(img)
    x0, y0 = pad * S, pad * S
    d.rounded_rectangle([x0, y0, x0 + w * S, y0 + h * S], radius=r * S, fill=(26, 26, 30, 255) if dark else PAPER_CARD + (255,))
    mono = fonts.get("mono", 22 * S)
    d.text((x0 + 28 * S, y0 + 22 * S), left, font=mono, fill=GREY + (255,))
    if right:
        wr = mono.getlength(right)
        d.text((x0 + w * S - 28 * S - wr, y0 + 22 * S), right, font=mono, fill=GREY + (255,))
    return img


def _finish(img: Image.Image) -> np.ndarray:
    w, h = img.size
    return _to_bgra(img.resize((w // S, h // S), Image.LANCZOS))


# ---------------------------------------------------------------------------
def bars_card(t: float, w: int = 800, h: int = 400) -> np.ndarray:
    """A messy 'problem' card: redacted text lines that jitter, plus a warning pill."""
    img = card_base(w, h, "// the problem", "unsorted")
    d = ImageDraw.Draw(img)
    pad = 46
    lens = [0.92, 0.7, 0.86, 0.55, 0.8, 0.64]
    for i, ln in enumerate(lens):
        jit = 6 * math.sin(t * 9 + i * 1.7)
        y = (pad + 84 + i * 46) * S
        d.rounded_rectangle([(pad + 30) * S + jit * S, y, (pad + 30 + (w - 60) * ln) * S + jit * S, y + 22 * S],
                            radius=11 * S, fill=(30, 30, 34, 255))
    # warning pill
    f = fonts.get("bold", 26 * S)
    txt = "too much at once"
    tw = f.getlength(txt)
    px, py = (pad + w - 40) * S - tw - 40 * S, (pad + h - 62) * S
    d.rounded_rectangle([px, py, px + tw + 40 * S, py + 44 * S], radius=22 * S, fill=ORANGE + (255,))
    d.text((px + 20 * S, py + 6 * S), txt, font=f, fill=(255, 255, 255, 255))
    return _finish(img)


def checklist_card(labels: list[str], starts: list[float], t: float, w: int = 800) -> np.ndarray:
    rows = max(1, len(labels))
    h = 128 + rows * 104
    img = card_base(w, h, "// the plan", f"{sum(1 for s in starts if t >= s)}/{rows}")
    d = ImageDraw.Draw(img)
    pad = 46
    f = fonts.get("bold", 40 * S)
    active = max([i for i, s in enumerate(starts) if t >= s], default=-1)
    for i, lab in enumerate(labels):
        y = (pad + 86 + i * 104) * S
        reached = t >= starts[i] if i < len(starts) else False
        if i == active:
            d.rounded_rectangle([(pad + 18) * S, y - 12 * S, (pad + w - 18) * S, y + 78 * S], radius=18 * S,
                                fill=(255, 238, 230, 255))
            d.rounded_rectangle([(pad + 18) * S, y - 12 * S, (pad + 28) * S, y + 78 * S], radius=5 * S, fill=ORANGE + (255,))
        bx, by = (pad + 46) * S, y + 6 * S
        d.rounded_rectangle([bx, by, bx + 56 * S, by + 56 * S], radius=14 * S,
                            fill=ORANGE + (255,) if reached else (255, 255, 255, 255),
                            outline=INK + (255,) if not reached else None, width=4 * S)
        if reached:
            u = min(1.0, (t - starts[i]) / 0.28)
            pts = [(bx + 13 * S, by + 29 * S), (bx + 24 * S, by + 40 * S), (bx + 44 * S, by + 16 * S)]
            seg = u * 2
            path = [pts[0]]
            if seg <= 1:
                path.append((pts[0][0] + (pts[1][0] - pts[0][0]) * seg, pts[0][1] + (pts[1][1] - pts[0][1]) * seg))
            else:
                path += [pts[1], (pts[1][0] + (pts[2][0] - pts[1][0]) * (seg - 1), pts[1][1] + (pts[2][1] - pts[1][1]) * (seg - 1))]
            d.line(path, fill=(255, 255, 255, 255), width=7 * S, joint="curve")
        col = INK if reached or i == active else (170, 170, 176)
        d.text((bx + 84 * S, y + 4 * S), lab, font=f, fill=col + (255,))
    return _finish(img)


@lru_cache(maxsize=64)
def _digit_strip(ch_h: int) -> Image.Image:
    f = fonts.get("bold", ch_h)
    wid = int(f.getlength("0") * 1.05)
    strip = Image.new("RGBA", (wid, ch_h * 21), (0, 0, 0, 0))
    d = ImageDraw.Draw(strip)
    for i in range(21):
        d.text((wid // 2, i * ch_h + ch_h // 2), str(i % 10), font=f, fill=ORANGE + (255,), anchor="mm")
    return strip


def stat_prop(value: int, label: str, t: float, size: int = 300) -> np.ndarray:
    """A big rolling number (slot-machine style) with an italic label under it."""
    digits = str(value)
    strip = _digit_strip(size)
    wid = strip.width
    ch_h = size
    total_w = wid * len(digits)
    lab_f = fonts.get("serif", int(size * 0.36))
    img = Image.new("RGBA", (max(total_w, int(lab_f.getlength(label)) + 20) + 40, ch_h + int(size * 0.5)), (0, 0, 0, 0))
    ox = (img.width - total_w) // 2
    for k, dch in enumerate(digits):
        target = int(dch)
        dur = 0.75 + 0.18 * k
        u = min(1.0, max(0.0, t / dur))
        e = ease_out(u)
        spins = 1 + k                                   # later digits spin more before landing
        pos = (target + 10 * spins) * e
        offset = (pos % 10) * ch_h
        col = strip.crop((0, int(offset), wid, int(offset) + ch_h))
        col_np = np.array(col)
        if u < 0.95:                                    # motion blur while spinning
            kh = int(40 * (1 - e)) | 1
            col_np = cv2.blur(col_np, (1, max(1, kh)))
        img.paste(Image.fromarray(col_np), (ox + k * wid, 0), Image.fromarray(col_np))
    d = ImageDraw.Draw(img)
    if label:
        d.text((img.width // 2, ch_h + int(size * 0.12)), label, font=lab_f, fill=INK + (255,), anchor="mt")
    return _to_bgra(img)


def keyword_prop(word: str, t: float, size: int = 230, color=ORANGE) -> np.ndarray:
    """A single big italic word that gets circled by a hand-drawn scribble."""
    f = fonts.get("serif", size)
    tw = int(f.getlength(word))
    pad = 70
    img = Image.new("RGBA", ((tw + 2 * pad) * S, int(size * 1.6) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f2 = fonts.get("serif", size * S)
    d.text((pad * S, int(size * 0.2) * S), word, font=f2, fill=color + (255,))
    u = min(1.0, max(0.0, (t - 0.35) / 0.55))
    if u > 0:
        cx, cy = (tw / 2 + pad) * S, int(size * 0.82) * S
        rx, ry = (tw / 2 + 34) * S, int(size * 0.62) * S
        n = 90
        pts = []
        for i in range(int(n * u) + 1):
            a = -2.2 + 2 * math.pi * 1.07 * i / n
            wob = 1 + 0.035 * math.sin(a * 3.0)
            pts.append((cx + rx * wob * math.cos(a), cy + ry * wob * math.sin(a) + i * 0.18 * S))
        if len(pts) > 1:
            d.line(pts, fill=INK + (255,), width=7 * S, joint="curve")
    return _finish(img)


def follow_prop(t: float, w: int = 420, h: int = 120) -> np.ndarray:
    """Follow button, a cursor that glides in and clicks, then 'Following'."""
    cw, ch = 760, 340
    img = Image.new("RGBA", (cw * S, ch * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    click_t = 0.95
    pressed = click_t <= t < click_t + 0.12
    done = t >= click_t + 0.1
    bx0, by0 = (cw - w) // 2, 70
    sc = 0.94 if pressed else 1.0
    ww, hh = int(w * sc), int(h * sc)
    x0, y0 = bx0 + (w - ww) // 2, by0 + (h - hh) // 2
    f = fonts.get("bold", 46 * S)
    if not done:
        d.rounded_rectangle([x0 * S, y0 * S, (x0 + ww) * S, (y0 + hh) * S], radius=hh * S // 2, fill=ORANGE + (255,))
        d.text(((x0 + ww / 2) * S, (y0 + hh / 2) * S), "Follow", font=f, fill=(255, 255, 255, 255), anchor="mm")
    else:
        d.rounded_rectangle([x0 * S, y0 * S, (x0 + ww) * S, (y0 + hh) * S], radius=hh * S // 2, fill=(255, 255, 255, 255),
                            outline=INK + (255,), width=5 * S)
        d.text(((x0 + ww / 2 - 18) * S, (y0 + hh / 2) * S), "Following", font=f, fill=INK + (255,), anchor="mm")
        cx, cy = (x0 + ww - 62) * S, (y0 + hh / 2) * S
        d.line([(cx - 14 * S, cy), (cx - 3 * S, cy + 12 * S), (cx + 16 * S, cy - 12 * S)], fill=INK + (255,), width=6 * S, joint="curve")
        ru = (t - click_t) / 0.5
        if 0 <= ru < 1:
            r = (50 + 150 * ru) * S
            a = int(200 * (1 - ru))
            d.ellipse([(cw / 2) * S - r, (by0 + h / 2) * S - r, (cw / 2) * S + r, (by0 + h / 2) * S + r],
                      outline=ORANGE + (a,), width=5 * S)
    # cursor
    u = min(1.0, t / 0.85)
    e = ease_out(u)
    sx, sy = cw + 40, by0 + h + 180
    tx, ty = cw / 2 + 60, by0 + h / 2 + 30
    px, py = sx + (tx - sx) * e, sy + (ty - sy) * e - 40 * math.sin(math.pi * e)
    if t > click_t + 0.45:
        px, py = tx + (t - click_t - 0.45) * 90, ty + (t - click_t - 0.45) * 60
    if t < click_t + 1.2:
        pts = [(px, py), (px, py + 44), (px + 12, py + 33), (px + 24, py + 56), (px + 33, py + 51), (px + 22, py + 29), (px + 36, py + 29)]
        d.polygon([(x * S, y * S) for x, y in pts], fill=(18, 18, 20, 255), outline=(255, 255, 255, 255), width=3 * S)
    return _finish(img)


def _wrap(text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if font.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def chat_card(prompt: str, reply: str, t: float, w: int = 840, h: int = 470) -> np.ndarray:
    """An AI-assistant window: the prompt types itself, three dots pulse, then the answer streams in."""
    img = card_base(w, h, "// ask your AI assistant", "demo")
    d = ImageDraw.Draw(img)
    pad = 46
    f = fonts.get("sans", 31 * S)
    fb = fonts.get("bold", 31 * S)
    x0, y0 = (pad + 28) * S, (pad + 84) * S
    maxw = (w - 56 - 150) * S
    # user bubble: types at 52 characters per second
    typed = prompt[: max(0, int((t - 0.25) * 52))]
    if typed:
        lines = _wrap(typed, f, maxw - 48 * S)
        bh = (len(lines) * 44 + 34) * S
        bw = min(maxw, max(f.getlength(l) for l in lines) + 48 * S)
        bx1 = (pad + w - 28) * S
        d.rounded_rectangle([bx1 - bw, y0, bx1, y0 + bh], radius=26 * S, fill=INK + (255,))
        for i, l in enumerate(lines):
            d.text((bx1 - bw + 24 * S, y0 + (17 + i * 44) * S), l, font=f, fill=(255, 255, 255, 255))
        y_reply = y0 + bh + 26 * S
    else:
        y_reply = y0
    t_type_end = 0.25 + len(prompt) / 52.0
    t_reply = t_type_end + 0.55
    if typed and t >= t_type_end and t < t_reply:                  # thinking dots
        for i in range(3):
            a = 0.35 + 0.65 * max(0.0, math.sin((t - t_type_end) * 11 - i * 0.9))
            d.ellipse([x0 + i * 30 * S, y_reply + 14 * S, x0 + i * 30 * S + 18 * S, y_reply + 32 * S],
                      fill=(int(255 * (1 - a) + ORANGE[0] * a), int(255 * (1 - a) + ORANGE[1] * a), int(255 * (1 - a) + ORANGE[2] * a), 255))
    if t >= t_reply:
        words = reply.split()
        shown = " ".join(words[: int((t - t_reply) * 14) + 1])
        lines = _wrap(shown, f, maxw)
        full = _wrap(reply, f, maxw)
        bh = (len(full) * 44 + 34) * S
        bw = min(maxw, max(f.getlength(l) for l in full) + 48 * S)
        d.rounded_rectangle([x0, y_reply, x0 + bw, y_reply + bh], radius=26 * S, fill=(255, 237, 228, 255))
        d.ellipse([x0 - 4 * S, y_reply - 12 * S, x0 + 20 * S, y_reply + 12 * S], fill=ORANGE + (255,))
        for i, l in enumerate(lines):
            d.text((x0 + 24 * S, y_reply + (17 + i * 44) * S), l, font=f, fill=INK + (255,))
    return _finish(img)
