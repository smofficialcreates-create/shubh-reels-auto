"""On-screen props: UI cards, rolling numbers, keyword scribbles, step badges, comparison cards,
flows, layer stacks and definitions. All colours come from palette.Theme.

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
from .palette import NAVY_T, Theme

S = 2


def _to_bgra(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def ease_out_back(x, s=1.8):
    x = min(max(x, 0.0), 1.0)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


@lru_cache(maxsize=16)
def _shadow(w: int, h: int, r: int) -> Image.Image:
    pad = 46
    sh = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([pad, pad + 14, pad + w, pad + h + 14], radius=r, fill=(0, 0, 0, 80))
    return sh.filter(ImageFilter.GaussianBlur(18))


def card_base(w: int, h: int, left: str, right: str = "", th: Theme = NAVY_T) -> Image.Image:
    """Card with a mono header line, returned at 2x with a shadow margin."""
    pad = 46
    r = 26
    sh = _shadow(w, h, r).resize(((w + 2 * pad) * S, (h + 2 * pad) * S), Image.LANCZOS)
    img = sh
    d = ImageDraw.Draw(img)
    x0, y0 = pad * S, pad * S
    d.rounded_rectangle([x0, y0, x0 + w * S, y0 + h * S], radius=r * S, fill=th.card + (255,))
    mono = fonts.get("mono", 22 * S)
    d.text((x0 + 28 * S, y0 + 22 * S), left, font=mono, fill=th.mute + (255,))
    if right:
        wr = mono.getlength(right)
        d.text((x0 + w * S - 28 * S - wr, y0 + 22 * S), right, font=mono, fill=th.mute + (255,))
    return img


def _finish(img: Image.Image) -> np.ndarray:
    w, h = img.size
    return _to_bgra(img.resize((w // S, h // S), Image.LANCZOS))


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


def _fit(font_kind: str, text: str, size: int, maxw: int) -> "ImageFont.FreeTypeFont":
    while size > 14 and fonts.get(font_kind, size).getlength(text) > maxw:
        size -= 2
    return fonts.get(font_kind, size)


# ---------------------------------------------------------------------------
def bars_card(t: float, th: Theme = NAVY_T, w: int = 800, h: int = 400) -> np.ndarray:
    """A messy 'problem' card: redacted text lines that jitter, plus a warning pill."""
    img = card_base(w, h, "// the problem", "unsorted", th)
    d = ImageDraw.Draw(img)
    pad = 46
    lens = [0.92, 0.7, 0.86, 0.55, 0.8, 0.64]
    for i, ln in enumerate(lens):
        jit = 6 * math.sin(t * 9 + i * 1.7)
        y = (pad + 84 + i * 46) * S
        d.rounded_rectangle([(pad + 30) * S + jit * S, y, (pad + 30 + (w - 60) * ln) * S + jit * S, y + 22 * S],
                            radius=11 * S, fill=th.ink + (255,) if th.name == "ivory" else th.ink + (235,))
    f = fonts.get("bold", 26 * S)
    txt = "too much at once"
    tw = f.getlength(txt)
    px, py = (pad + w - 40) * S - tw - 40 * S, (pad + h - 62) * S
    d.rounded_rectangle([px, py, px + tw + 40 * S, py + 44 * S], radius=22 * S, fill=th.c_fill + (255,))
    d.text((px + 20 * S, py + 6 * S), txt, font=f, fill=th.on_fill + (255,))
    return _finish(img)


def checklist_card(labels: list[str], starts: list[float], t: float, th: Theme = NAVY_T, w: int = 800) -> np.ndarray:
    rows = max(1, len(labels))
    h = 128 + rows * 104
    img = card_base(w, h, "// the plan", f"{sum(1 for s in starts if t >= s)}/{rows}", th)
    d = ImageDraw.Draw(img)
    pad = 46
    f = fonts.get("bold", 40 * S)
    active = max([i for i, s in enumerate(starts) if t >= s], default=-1)
    for i, lab in enumerate(labels):
        y = (pad + 86 + i * 104) * S
        reached = t >= starts[i] if i < len(starts) else False
        if i == active:
            d.rounded_rectangle([(pad + 18) * S, y - 12 * S, (pad + w - 18) * S, y + 78 * S], radius=18 * S, fill=th.soft + (255,))
            d.rounded_rectangle([(pad + 18) * S, y - 12 * S, (pad + 28) * S, y + 78 * S], radius=5 * S, fill=th.c_fill + (255,))
        bx, by = (pad + 46) * S, y + 6 * S
        d.rounded_rectangle([bx, by, bx + 56 * S, by + 56 * S], radius=14 * S,
                            fill=th.c_fill + (255,) if reached else th.card + (255,),
                            outline=th.ink + (255,) if not reached else None, width=4 * S)
        if reached:
            u = min(1.0, (t - starts[i]) / 0.28)
            pts = [(bx + 13 * S, by + 29 * S), (bx + 24 * S, by + 40 * S), (bx + 44 * S, by + 16 * S)]
            seg = u * 2
            path = [pts[0]]
            if seg <= 1:
                path.append((pts[0][0] + (pts[1][0] - pts[0][0]) * seg, pts[0][1] + (pts[1][1] - pts[0][1]) * seg))
            else:
                path += [pts[1], (pts[1][0] + (pts[2][0] - pts[1][0]) * (seg - 1), pts[1][1] + (pts[2][1] - pts[1][1]) * (seg - 1))]
            d.line(path, fill=th.on_fill + (255,), width=7 * S, joint="curve")
        col = th.ink if reached or i == active else th.mute
        d.text((bx + 84 * S, y + 4 * S), lab, font=_fit("bold", lab, 40 * S, (w - 190) * S), fill=col + (255,))
    return _finish(img)


@lru_cache(maxsize=64)
def _digit_strip(ch_h: int, color: tuple) -> Image.Image:
    f = fonts.get("bold", ch_h)
    wid = int(f.getlength("0") * 1.05)
    strip = Image.new("RGBA", (wid, ch_h * 21), (0, 0, 0, 0))
    d = ImageDraw.Draw(strip)
    for i in range(21):
        d.text((wid // 2, i * ch_h + ch_h // 2), str(i % 10), font=f, fill=color + (255,), anchor="mm")
    return strip


def stat_prop(value: int, label: str, t: float, th: Theme = NAVY_T, size: int = 300) -> np.ndarray:
    """A big rolling number (slot-machine style) with an italic label under it."""
    digits = str(value)
    strip = _digit_strip(size, th.accent if not th.marker else th.c_text)
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
        spins = 1 + k
        pos = (target + 10 * spins) * e
        offset = (pos % 10) * ch_h
        col = strip.crop((0, int(offset), wid, int(offset) + ch_h))
        col_np = np.array(col)
        if u < 0.95:
            kh = int(40 * (1 - e)) | 1
            col_np = cv2.blur(col_np, (1, max(1, kh)))
        img.paste(Image.fromarray(col_np), (ox + k * wid, 0), Image.fromarray(col_np))
    d = ImageDraw.Draw(img)
    if label:
        d.text((img.width // 2, ch_h + int(size * 0.12)), label, font=lab_f, fill=th.text + (255,), anchor="mt")
    return _to_bgra(img)


def keyword_prop(word: str, t: float, th: Theme = NAVY_T, size: int = 230) -> np.ndarray:
    """A single big italic word that gets circled by a hand-drawn scribble."""
    color = th.c_text if th.marker else th.accent
    f = fonts.get("serif", size)
    tw = int(f.getlength(word))
    pad = 70
    img = Image.new("RGBA", ((tw + 2 * pad) * S, int(size * 1.6) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f2 = fonts.get("serif", size * S)
    d.text((pad * S, int(size * 0.2) * S), word, font=f2, fill=th.text + (255,) if th.marker else color + (255,))
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
            d.line(pts, fill=(th.c_fill if th.marker else th.text) + (255,), width=7 * S, joint="curve")
    return _finish(img)


def follow_prop(t: float, th: Theme = NAVY_T, w: int = 420, h: int = 120) -> np.ndarray:
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
        d.rounded_rectangle([x0 * S, y0 * S, (x0 + ww) * S, (y0 + hh) * S], radius=hh * S // 2, fill=th.c_fill + (255,))
        d.text(((x0 + ww / 2) * S, (y0 + hh / 2) * S), "Follow", font=f, fill=th.on_fill + (255,), anchor="mm")
    else:
        d.rounded_rectangle([x0 * S, y0 * S, (x0 + ww) * S, (y0 + hh) * S], radius=hh * S // 2, fill=th.card + (255,),
                            outline=th.c_fill + (255,), width=5 * S)
        d.text(((x0 + ww / 2 - 18) * S, (y0 + hh / 2) * S), "Following", font=f, fill=th.ink + (255,), anchor="mm")
        cx, cy = (x0 + ww - 62) * S, (y0 + hh / 2) * S
        d.line([(cx - 14 * S, cy), (cx - 3 * S, cy + 12 * S), (cx + 16 * S, cy - 12 * S)], fill=th.c_text + (255,), width=6 * S, joint="curve")
        ru = (t - click_t) / 0.5
        if 0 <= ru < 1:
            r = (50 + 150 * ru) * S
            a = int(200 * (1 - ru))
            d.ellipse([(cw / 2) * S - r, (by0 + h / 2) * S - r, (cw / 2) * S + r, (by0 + h / 2) * S + r],
                      outline=th.c_fill + (a,), width=5 * S)
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


def chat_card(prompt: str, reply: str, t: float, th: Theme = NAVY_T, w: int = 840, h: int = 470) -> np.ndarray:
    """An AI-assistant window: the prompt types itself, three dots pulse, then the answer streams in."""
    img = card_base(w, h, "// ask your AI assistant", "demo", th)
    d = ImageDraw.Draw(img)
    pad = 46
    f = fonts.get("sans", 31 * S)
    x0, y0 = (pad + 28) * S, (pad + 84) * S
    maxw = (w - 56 - 150) * S
    user_bg = th.ink if th.name != "ivory" else IVORY_BUBBLE
    typed = prompt[: max(0, int((t - 0.25) * 52))]
    if typed:
        lines = _wrap(typed, f, maxw - 48 * S)
        bh = (len(lines) * 44 + 34) * S
        bw = min(maxw, max(f.getlength(l) for l in lines) + 48 * S)
        bx1 = (pad + w - 28) * S
        d.rounded_rectangle([bx1 - bw, y0, bx1, y0 + bh], radius=26 * S, fill=user_bg + (255,))
        for i, l in enumerate(lines):
            d.text((bx1 - bw + 24 * S, y0 + (17 + i * 44) * S), l, font=f, fill=(th.card if th.name != "ivory" else th.ink) + (255,))
        y_reply = y0 + bh + 26 * S
    else:
        y_reply = y0
    t_type_end = 0.25 + len(prompt) / 52.0
    t_reply = t_type_end + 0.55
    if typed and t >= t_type_end and t < t_reply:
        for i in range(3):
            a = 0.35 + 0.65 * max(0.0, math.sin((t - t_type_end) * 11 - i * 0.9))
            col = tuple(int(th.card[k] * (1 - a) + th.c_fill[k] * a) for k in range(3))
            d.ellipse([x0 + i * 30 * S, y_reply + 14 * S, x0 + i * 30 * S + 18 * S, y_reply + 32 * S], fill=col + (255,))
    if t >= t_reply:
        words = reply.split()
        shown = " ".join(words[: int((t - t_reply) * 14) + 1])
        lines = _wrap(shown, f, maxw)
        full = _wrap(reply, f, maxw)
        bh = (len(full) * 44 + 34) * S
        bw = min(maxw, max(f.getlength(l) for l in full) + 48 * S)
        d.rounded_rectangle([x0, y_reply, x0 + bw, y_reply + bh], radius=26 * S, fill=th.soft + (255,))
        d.ellipse([x0 - 4 * S, y_reply - 12 * S, x0 + 20 * S, y_reply + 12 * S], fill=th.c_fill + (255,))
        for i, l in enumerate(lines):
            d.text((x0 + 24 * S, y_reply + (17 + i * 44) * S), l, font=f, fill=th.ink + (255,))
    return _finish(img)


IVORY_BUBBLE = (72, 78, 86)


# ===================== new visual types ====================================================
def numeral_prop(n: int, total: int, title: str, t: float, th: Theme = NAVY_T) -> np.ndarray:
    """A big step badge that pops in, a row of progress dots, and a bold title that slides up."""
    cw, ch = 760, 520
    img = Image.new("RGBA", (cw * S, ch * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    e = ease_out_back(t / 0.5, 2.2)
    side = int(250 * max(0.05, e))
    cx, cy = cw // 2, 190
    x0, y0 = (cx - side // 2) * S, (cy - side // 2) * S
    d.rounded_rectangle([x0 + 8 * S, y0 + 14 * S, x0 + side * S + 8 * S, y0 + side * S + 14 * S], radius=56 * S, fill=(0, 0, 0, 70))
    d.rounded_rectangle([x0, y0, x0 + side * S, y0 + side * S], radius=56 * S, fill=th.c_fill + (255,))
    f = fonts.get("bold", int(190 * max(0.2, e)) * S)
    d.text((cx * S, (cy + 6) * S), str(n), font=f, fill=th.on_fill + (255,), anchor="mm")
    dots = max(1, total)
    gap = 36
    sx = cx - (dots - 1) * gap / 2
    for i in range(dots):
        on = i < n
        r = 9 if on else 7
        col = th.accent if not th.marker else th.c_fill
        d.ellipse([(sx + i * gap - r) * S, (350 - r) * S, (sx + i * gap + r) * S, (350 + r) * S],
                  fill=(col + (255,)) if on else None, outline=(th.mono + (255,)), width=2 * S)
    u = ease_out((t - 0.25) / 0.45)
    tf = _fit("bold", title, 54 * S, (cw - 40) * S)
    ty = int((420 + 24 * (1 - u)) * S)
    d.text((cx * S, ty), title, font=tf, fill=th.text + (int(255 * u),), anchor="mm")
    return _finish(img)


def counter_prop(value: int, unit: str, label: str, t: float, th: Theme = NAVY_T) -> np.ndarray:
    """A number that counts up, with a rising line under it. Used for examples ('about forty minutes')."""
    cw, ch = 820, 520
    img = Image.new("RGBA", (cw * S, ch * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = ease_out(t / 1.3)
    shown = int(round(value * u))
    big = fonts.get("bold", 250 * S)
    txt = str(shown)
    col = th.c_text if th.marker else th.accent
    wnum = big.getlength(txt)
    uf = fonts.get("serif", 84 * S)
    wun = uf.getlength(unit) if unit else 0
    total = wnum + (24 * S + wun if unit else 0)
    x = (cw * S - total) / 2
    d.text((x, 150 * S), txt, font=big, fill=th.text + (255,), anchor="lm")
    if unit:
        d.text((x + wnum + 24 * S, 190 * S), unit, font=uf, fill=col + (255,), anchor="lm")
    lf = fonts.get("sans", 36 * S)
    d.text((cw * S / 2, 300 * S), label, font=_fit("sans", label, 36 * S, (cw - 60) * S), fill=th.mono + (255,), anchor="mm")
    # rising line
    gx0, gx1, gy0, gy1 = 90, cw - 90, 470, 350
    d.line([(gx0 * S, gy0 * S), (gx1 * S, gy0 * S)], fill=th.mono + (110,), width=2 * S)
    pts = []
    n = 60
    for i in range(int(n * u) + 1):
        k = i / n
        yv = gy0 - (gy0 - gy1) * (k ** 1.6) - 8 * math.sin(k * 14) * (1 - k)
        pts.append((gx0 * S + (gx1 - gx0) * S * k, yv * S))
    if len(pts) > 1:
        d.line(pts, fill=(th.accent if not th.marker else th.c_fill) + (255,), width=7 * S, joint="curve")
        ex, ey = pts[-1]
        d.ellipse([ex - 12 * S, ey - 12 * S, ex + 12 * S, ey + 12 * S], fill=(th.accent if not th.marker else th.c_fill) + (255,))
    return _finish(img)


def compare_card(left: dict, right: dict, starts: list[float], t: float, th: Theme = NAVY_T) -> np.ndarray:
    """Two columns: the weak way (crosses) against the better way (ticks). Rows appear one by one."""
    w, h = 880, 470
    img = card_base(w, h, "// side by side", "", th)
    d = ImageDraw.Draw(img)
    pad = 46
    colw = (w - 90) // 2
    for side, spec in enumerate((left, right)):
        x0 = (pad + 30 + side * (colw + 30)) * S
        good = side == 1
        chip_fill = th.c_fill if good else th.soft
        chip_txt = th.on_fill if good else th.ink
        f = fonts.get("bold", 30 * S)
        title = str(spec.get("title", ""))[:22]
        d.rounded_rectangle([x0, (pad + 80) * S, x0 + colw * S, (pad + 140) * S], radius=30 * S, fill=chip_fill + (255,))
        d.text((x0 + colw * S / 2, (pad + 110) * S), title, font=_fit("bold", title, 30 * S, (colw - 30) * S), fill=chip_txt + (255,), anchor="mm")
        rows = list(spec.get("rows", []))[:3]
        for i, row in enumerate(rows):
            st = starts[i] if i < len(starts) else 0.5 + i * 0.5
            st = st + (0.18 if good else 0.0)
            u = ease_out((t - st) / 0.35)
            if u <= 0:
                continue
            y = (pad + 168 + i * 92 + 14 * (1 - u)) * S
            cx, cy = x0 + 26 * S, y + 28 * S
            if good:
                d.ellipse([cx - 22 * S, cy - 22 * S, cx + 22 * S, cy + 22 * S], fill=th.c_fill + (int(255 * u),))
                d.line([(cx - 10 * S, cy + 1 * S), (cx - 3 * S, cy + 9 * S), (cx + 11 * S, cy - 9 * S)],
                       fill=th.on_fill + (int(255 * u),), width=5 * S, joint="curve")
            else:
                d.ellipse([cx - 22 * S, cy - 22 * S, cx + 22 * S, cy + 22 * S], outline=th.mute + (int(255 * u),), width=3 * S)
                d.line([(cx - 9 * S, cy - 9 * S), (cx + 9 * S, cy + 9 * S)], fill=th.mute + (int(255 * u),), width=4 * S)
                d.line([(cx + 9 * S, cy - 9 * S), (cx - 9 * S, cy + 9 * S)], fill=th.mute + (int(255 * u),), width=4 * S)
            rf = _fit("sans", row, 29 * S, (colw - 80) * S)
            d.text((x0 + 62 * S, cy), row, font=rf, fill=(th.ink if good else th.mute) + (int(255 * u),), anchor="lm")
    return _finish(img)


def flow_card(nodes: list[str], starts: list[float], t: float, th: Theme = NAVY_T, header: str = "// how it moves") -> np.ndarray:
    """A vertical chain of boxes joined by arrows. A gold dot travels down as each step is spoken."""
    n = max(2, min(5, len(nodes)))
    nodes = nodes[:n]
    nh, gap = 78, 40
    w = 760
    h = 104 + n * nh + (n - 1) * gap + 30
    img = card_base(w, h, header, f"{sum(1 for s in starts if t >= s)}/{n}", th)
    d = ImageDraw.Draw(img)
    pad = 46
    active = max([i for i, s in enumerate(starts) if t >= s], default=-1)
    f = fonts.get("bold", 36 * S)
    for i, label in enumerate(nodes):
        y = (pad + 76 + i * (nh + gap)) * S
        reached = i <= active
        cur = i == active
        x0, x1 = (pad + 60) * S, (pad + w - 60) * S
        fill = th.c_fill if cur else (th.soft if reached else th.card)
        outline = None if cur else (th.c_fill if reached else th.line)
        d.rounded_rectangle([x0, y, x1, y + nh * S], radius=22 * S, fill=fill + (255,), outline=outline, width=3 * S)
        bx = x0 + 44 * S
        d.ellipse([bx - 20 * S, y + nh * S / 2 - 20 * S, bx + 20 * S, y + nh * S / 2 + 20 * S],
                  fill=(th.on_fill if cur else th.ink) + (255,))
        d.text((bx, y + nh * S / 2), str(i + 1), font=fonts.get("bold", 26 * S), fill=(th.c_fill if cur else th.card) + (255,), anchor="mm")
        d.text((bx + 44 * S, y + nh * S / 2), label, font=_fit("bold", label, 36 * S, (x1 - bx - 70 * S)),
               fill=(th.on_fill if cur else (th.ink if reached else th.mute)) + (255,), anchor="lm")
        if i < n - 1:
            ax = (pad + w / 2) * S
            ay0, ay1 = y + nh * S + 6 * S, y + (nh + gap) * S - 6 * S
            done = i < active
            col = th.c_fill if done else th.line
            d.line([(ax, ay0), (ax, ay1)], fill=col + (255,), width=5 * S)
            d.polygon([(ax - 10 * S, ay1 - 12 * S), (ax + 10 * S, ay1 - 12 * S), (ax, ay1 + 2 * S)], fill=col + (255,))
            # travelling packet between the current node and the next one
            if i == active and i + 1 < n:
                nxt = starts[i + 1] if i + 1 < len(starts) else starts[i] + 3.0
                span = max(0.4, nxt - starts[i])
                k = ((t - starts[i]) / span) % 1.0
                py = ay0 + (ay1 - ay0) * ease_io(k)
                d.ellipse([ax - 9 * S, py - 9 * S, ax + 9 * S, py + 9 * S], fill=th.c_fill + (255,))
    return _finish(img)


def layers_card(labels: list[str], starts: list[float], t: float, th: Theme = NAVY_T, header: str = "// the layers") -> np.ndarray:
    """A stack of slanted plates. The plate being spoken about lifts, lights up and slides forward."""
    n = max(2, min(7, len(labels)))
    labels = labels[:n]
    step = 74 if n <= 5 else 62
    pw, ph = 560, step - 10
    cw = 820
    ch = 90 + n * step + 120
    img = Image.new("RGBA", (cw * S, ch * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    active = max([i for i, s in enumerate(starts) if t >= s], default=-1)
    mono = fonts.get("mono", 22 * S)
    d.text((40 * S, 20 * S), header, font=mono, fill=th.mono + (255,))
    d.text((cw * S - 40 * S, 20 * S), f"{max(0, active + 1)}/{n}", font=mono, fill=th.mono + (255,), anchor="ra")
    for i in range(n):                               # draw top plate last so it overlaps the ones beneath
        pass
    order = [i for i in range(n - 1, -1, -1) if i != active] + ([active] if active >= 0 else [])   # active plate on top
    for i in order:
        base_y = 90 + i * step
        lift = 0.0
        if i == active:
            u = ease_out_back((t - starts[i]) / 0.4, 1.6)
            lift = 26 * u
        slide = 0.0
        if i == active:
            slide = 38 * ease_out((t - starts[i]) / 0.4)
        cx = cw / 2 + slide - 0 * 1
        y = base_y - lift
        skew = 90
        pts = [(cx - pw / 2 + skew, y), (cx + pw / 2 + skew, y), (cx + pw / 2, y + ph), (cx - pw / 2, y + ph)]
        pts_s = [(x * S, yy * S) for x, yy in pts]
        shadow = [(x * S + 6 * S, yy * S + 14 * S) for x, yy in pts]
        d.polygon(shadow, fill=(0, 0, 0, 70))
        reached = i <= active
        cur = i == active
        fill = th.c_fill if cur else (th.card if not reached else th.soft)
        edge = th.c_fill if reached else th.line
        d.polygon(pts_s, fill=fill + (255,), outline=edge + (255,))
        d.line(pts_s + [pts_s[0]], fill=edge + (255,), width=3 * S)
        label = labels[i]
        lf = _fit("bold", label, 34 * S, (pw - 150) * S)
        d.text(((cx + skew / 2) * S, (y + ph / 2) * S), label, font=lf,
               fill=(th.on_fill if cur else (th.ink if reached else th.mute)) + (255,), anchor="mm")
    return _finish(img)


def definition_card(term: str, meaning: str, t: float, th: Theme = NAVY_T) -> np.ndarray:
    """A term in big italic serif, a rule, and a one-line meaning that types in word by word."""
    w, h = 840, 440
    img = card_base(w, h, "// in plain words", "", th)
    d = ImageDraw.Draw(img)
    pad = 46
    u = ease_out(t / 0.5)
    tf = _fit("serif", term, 104 * S, (w - 90) * S)
    d.text(((pad + 40) * S, (pad + 160 + 18 * (1 - u)) * S), term, font=tf, fill=(th.c_text if not th.marker else th.c_fill) + (int(255 * u),), anchor="ls")
    lw = ease_out((t - 0.2) / 0.5)
    d.line([((pad + 40) * S, (pad + 186) * S), ((pad + 40 + (w - 80) * lw) * S, (pad + 186) * S)], fill=th.line + (255,), width=3 * S)
    words = meaning.split()
    shown = " ".join(words[: max(0, int((t - 0.5) * 6.5))])
    mf = fonts.get("sans", 34 * S)
    for i, ln in enumerate(_wrap(shown, mf, (w - 90) * S)[:4]):
        d.text(((pad + 40) * S, (pad + 222 + i * 50) * S), ln, font=mf, fill=th.ink + (255,))
    return _finish(img)
