"""Reel cover: five-to-seven words big and readable, Shubh underneath, inside Instagram's safe area.

The grid shows the middle 4:5 of a 9:16 cover, so everything important sits between y=285 and y=1635.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import fonts, plan
from .animator import Animator
from .editor import ORANGE, make_bg
from .puppet import Shubh

W, H = 1080, 1920


def _wrap(words, accent_i, maxw=900):
    """[(text, kind, size, x, baseline_y, accent)] with the accent word in orange italic serif."""
    out, x, y = [], 90, 470
    for i, w in enumerate(words):
        kind, size = ("serif", 140) if i == accent_i else ("bold", 118)
        wd = int(fonts.get(kind, size).getlength(w))
        if x + wd > 90 + maxw and x > 90:
            x, y = 90, y + 150
        out.append((w, kind, size, x, y, i == accent_i))
        x += wd + 28
    return out


def render_cover(script: dict, puppet: Shubh, out_path: str) -> str:
    words = script.get("cover_text", "").split() or script["hook"].split()[:7]
    ai = plan.accent_index(words, script.get("hook_focus"))
    frame = make_bg("paper").copy()
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((90, 330), "// SHUBH", font=fonts.get("mono", 30), fill=(150, 150, 156, 255))
    for w, kind, size, x, y, acc in _wrap(words, ai):
        f = fonts.get(kind, size)
        d.text((x, y), w, font=f, fill=(ORANGE if acc else (22, 22, 26)) + (255,), anchor="ls")
        if acc:
            d.line([(x, y + 22), (x + int(f.getlength(w)), y + 22)], fill=ORANGE + (255,), width=9)
    over = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    a = over[..., 3:4] / 255.0
    frame[:] = (frame * (1 - a) + over[..., :3] * a).astype(np.uint8)
    an = Animator()
    an.play("present_r", at=0.02)
    p = None
    for i in range(48):
        p = an.pose(i / 30, 1 / 30, mouth_open=0.0)
    p.smile = 0.6
    p.mouth_open = 0.18
    puppet.draw(frame, p, anchor_frame=(540.0, 1330.0), zoom=1.12)
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path
