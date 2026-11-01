"""One-time cut of assets/shubh_source.jpg into rig layers (assets/shubh/*.png).

Layers (source px, 457x1600 image):
  head   hair, face, glasses, neck
  torso  hoodie body with both sleeves removed and the hole behind them filled in
  strap  the two backpack straps (drawn over the arms)
  arm    the character's right arm (hoodie sleeve + hand); the left arm is its mirror
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

SRC = "assets/shubh_source.jpg"
OUT = "assets/shubh"
UP = 3                                           # layers are stored 3x upscaled
AXIS = 205.0                                     # body mirror axis (source x)

HEAD_HULL = [(80, 30), (270, 30), (280, 140), (262, 200), (242, 232), (222, 250), (218, 285), (152, 285), (148, 250), (124, 232), (98, 200), (80, 140)]
ARM_R_HULL = [(318, 292), (340, 300), (366, 322), (384, 380), (397, 470), (399, 600), (392, 690), (372, 740),
              (366, 800), (366, 880), (352, 880), (300, 880), (299, 800), (305, 745), (309, 700), (309, 560), (309, 420), (312, 330)]
STRAP_R = [(272, 296), (316, 296), (318, 560), (272, 560)]


def bg_model(f):
    H, W = f.shape[:2]
    bg = np.zeros_like(f)
    t = np.linspace(0, 1, W)[:, None]
    for y in range(H):
        l = np.median(f[y, 10:45], axis=0)
        r = np.median(f[y, 300:350], axis=0)
        bg[y] = l * (1 - t) + r * t
    return bg


def poly(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    return m


def soft_alpha(d, lo, hi):
    return np.clip((d - lo) / (hi - lo), 0, 1)


def main():
    img = cv2.imread(SRC)
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    bg = bg_model(f)
    d = np.linalg.norm(f - bg, axis=2)
    lum = f.mean(axis=2)
    os.makedirs(OUT, exist_ok=True)

    # ---- head ---------------------------------------------------------------
    hh = poly(img.shape, HEAD_HULL)
    m = ((d > 24) & (hh > 0)).astype(np.uint8)
    m[:, :70] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = (1 - m).astype(np.uint8)
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 2)
    m = m | (ff == 1).astype(np.uint8)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.6)
    a_head = np.clip((mf - 0.40) / 0.26, 0, 1)
    yy = np.arange(H)[:, None].astype(np.float32)
    a_head *= np.clip((270 - yy) / 22.0, 0, 1)
    save_layer("head", img, a_head, (60, 20, 290, 300))

    # ---- right arm (sleeve + hand) ----------------------------------------------
    ah = poly(img.shape, ARM_R_HULL)
    a_arm = np.zeros((H, W), np.float32)
    sleeve = (lum > 100) & (ah > 0)
    sleeve[:, 398:] = False
    skin_hand = (ah > 0) & (np.arange(H)[:, None] > 742) & ((f[..., 2] - f[..., 1]) > 9) & (lum > 85)
    ma = (sleeve | skin_hand).astype(np.uint8)
    ma = cv2.morphologyEx(ma, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ma = cv2.morphologyEx(ma, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ma)
    ma = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = (1 - ma).astype(np.uint8)
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 2)
    ma = ma | (ff == 1).astype(np.uint8)
    ma = cv2.medianBlur(ma * 255, 9) // 255
    a_arm = np.clip((cv2.GaussianBlur(ma.astype(np.float32), (0, 0), 1.6) - 0.4) / 0.35, 0, 1)
    save_layer("arm", img, a_arm, (290, 250, 400, 900))
    np.save(os.path.join(OUT, "_arm_mask.npy"), ma)

    # ---- torso: body between the sleeves, hole behind the arms filled ------------
    body = np.zeros((H, W), np.uint8)
    body_pts = [(112, 300), (150, 232), (180, 222), (232, 226), (262, 240), (298, 296), (312, 330), (312, 704), (98, 704), (98, 330)]
    cv2.fillPoly(body, [np.array(body_pts, np.int32)], 1)
    fill = img.copy()
    # arm band on both sides: replace by the neighbouring hoodie texture (mirror from inside the body)
    torso_a = body.astype(np.float32)
    torso_a = cv2.GaussianBlur(torso_a, (0, 0), 1.2)
    # head's neck region belongs to the head layer; keep the collar in the torso
    torso_a[:228, :] = 0
    save_layer("torso", fill, torso_a, (90, 215, 322, 712))

    # ---- straps: the two dark backpack bands (over the shoulder), cut from the source -------
    dark = (lum < 62) & (np.arange(H)[:, None] > 290) & (np.arange(H)[:, None] < 570)
    sl = poly(img.shape, [(40, 290), (92, 290), (92, 570), (40, 570)])
    sr = poly(img.shape, STRAP_R)
    strap = dark & ((sl > 0) | (sr > 0))
    strap = cv2.morphologyEx(strap.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(np.float32)
    sm = np.zeros_like(strap)
    for box in ((40, 300, 95, 570), (270, 300, 322, 570)):
        bx = np.zeros_like(strap); bx[box[1]:box[3], box[0]:box[2]] = 1
        part = (strap * bx).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(part)
        if n > 1:
            part = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        sm += part
    sm = cv2.medianBlur((np.clip(sm, 0, 1) * 255).astype(np.uint8), 9).astype(np.float32) / 255
    strap = cv2.GaussianBlur(sm, (0, 0), 1.1)
    save_layer("strap", img, strap, (30, 280, 330, 580))


def save_layer(name, img, alpha, box):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    a = alpha[y0:y1, x0:x1]
    # edge colours: replace low-alpha pixels' colour by the nearest solid colour so no navy halo shows
    solid = (a > 0.9).astype(np.uint8)
    solid = cv2.erode(solid, np.ones((3, 3), np.uint8), iterations=2)
    inp = cv2.inpaint(crop, ((1 - solid) * 255).astype(np.uint8), 4, cv2.INPAINT_TELEA)
    big = cv2.resize(inp, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(big, (0, 0), 2.0)
    big = cv2.addWeighted(big, 1.5, blur, -0.5, 0)
    ab = cv2.resize(a, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_CUBIC)
    ab = np.clip(cv2.GaussianBlur(ab, (0, 0), 1.0), 0, 1)
    out = np.dstack([big, (ab * 255).astype(np.uint8)])
    cv2.imwrite(os.path.join(OUT, f"{name}.png"), out)
    meta_p = os.path.join(OUT, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta[name] = {"x0": x0, "y0": y0, "scale": UP}
    json.dump(meta, open(meta_p, "w"), indent=1)


if __name__ == "__main__":
    main()
