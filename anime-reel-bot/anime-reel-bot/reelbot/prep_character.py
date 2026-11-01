"""One-off image clean-up: remove the cigarette and its smoke from the source art.

The coordinates are in the pixel space of assets/character.jpg (736x981).
Run:  python -m reelbot.prep_character  ->  writes assets/character_clean.png
Telea inpainting plus a smoke-detection pass, so no AI service is involved.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

# centre line of the smoke trail, ember -> top-left (image px)
SMOKE_PATH = [(322, 272), (306, 245), (303, 215), (290, 190), (262, 170), (232, 160),
              (224, 135), (236, 100), (232, 82), (200, 68), (168, 64), (152, 44), (112, 36)]


def clean(src: str, dst: str) -> None:
    im = cv2.imread(src)
    if im is None:
        raise SystemExit(f"cannot read {src}")
    h, w = im.shape[:2]

    # 1) smoke: bright, low-saturation wisps over a darker local background
    corridor = np.zeros((h, w), np.uint8)
    cv2.polylines(corridor, [np.array(SMOKE_PATH, np.int32)], False, 255, 46)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    luma = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.medianBlur(im, 41)
    bg_luma = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    wisp = ((luma - bg_luma) > 7) & (hsv[..., 1] < 95)
    smoke = (wisp & (corridor > 0)).astype(np.uint8) * 255
    smoke = cv2.dilate(smoke, np.ones((5, 5), np.uint8), iterations=1)

    # 2) cigarette: body, ember and the falling ash fleck
    cig = np.zeros((h, w), np.uint8)
    cv2.fillPoly(cig, [np.array([(324, 274), (383, 299), (381, 306), (322, 283)], np.int32)], 255)
    cv2.circle(cig, (321, 277), 8, 255, -1)
    cv2.line(cig, (314, 281), (316, 288), 255, 4)
    cv2.circle(cig, (327, 271), 8, 255, -1)      # leftover smoke puff by the fingertip
    cig = cv2.dilate(cig, np.ones((5, 5), np.uint8))

    # smoke first (Telea is fine on the dark wood panels) ...
    out = cv2.inpaint(im, smoke, 5, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(out, (0, 0), 3)
    feather = cv2.GaussianBlur(smoke, (0, 0), 4).astype(np.float32)[..., None] / 255.0
    out = (out * (1 - feather) + soft * feather).astype(np.uint8)

    # ... then the cigarette with frequency-selective reconstruction, which keeps
    # the finger outlines intact where the cigarette crossed the hand.
    x0, y0, x1, y1 = 290, 245, 430, 335
    crop = out[y0:y1, x0:x1].copy()
    cmask = cig[y0:y1, x0:x1]
    fixed = np.zeros_like(crop)
    try:
        cv2.xphoto.inpaint(crop, 255 - cmask, fixed, cv2.xphoto.INPAINT_FSR_BEST)
    except (AttributeError, cv2.error):          # opencv without contrib
        fixed = cv2.inpaint(crop, cmask, 5, cv2.INPAINT_TELEA)
    out[y0:y1, x0:x1] = fixed
    cv2.imwrite(dst, out)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "assets/character.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "assets/character_clean.png"
    clean(src, dst)
    print("wrote", dst)
