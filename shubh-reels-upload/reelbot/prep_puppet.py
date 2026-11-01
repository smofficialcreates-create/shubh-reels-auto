"""One-off: cut assets/shubh_sheet.png into rig layers (assets/puppet/*.png + meta.json).

The sheet already carries pivot markers (grey discs). They are located here, recorded as
joint positions, and painted out of the artwork so they never show in a render.
Run:  python -m reelbot.prep_puppet
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

SHEET = "assets/shubh_sheet.png"
OUT = "assets/puppet"
BG = np.array([235, 236, 237], np.int32)


def find_dots(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    b, gg, r = [im[..., i].astype(int) for i in range(3)]
    spread = np.maximum(np.maximum(abs(b - gg), abs(gg - r)), abs(b - r))
    dm = ((spread < 14) & (g > 120) & (g < 215)).astype(np.uint8)
    dm[:, :500] = 0
    dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(dm)
    return [(float(cen[i][0]), float(cen[i][1])) for i in range(1, n)
            if 50 < st[i, 4] < 260 and 7 <= st[i, 2] <= 13 and 7 <= st[i, 3] <= 13]


def components(im):
    d = np.abs(im.astype(np.int32) - BG).sum(2)
    m = (d > 30).astype(np.uint8)
    m[:, :500] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return lab, [(i, tuple(st[i])) for i in range(1, n) if st[i, 4] > 300]


def cut(im, lab, idx, box, dots, pad=5):
    """BGRA sprite of component idx (straight alpha).

    The sheet background is one flat colour, so the edge pixels are un-mixed against it:
    the outermost ring gets a fractional alpha from how far it is from the background
    and its colour is re-synthesised from the piece's own interior. Joint discs are painted
    out the same way. Returns (sprite, origin_xy)."""
    x, y, w, h, _ = box
    x0, y0, x1, y1 = x - pad, y - pad, x + w + pad, y + h + pad
    crop = im[y0:y1, x0:x1].copy()
    dsum = np.abs(crop.astype(np.int32) - BG).sum(2)
    comp = ((lab[y0:y1, x0:x1] == idx) & (dsum > 70)).astype(np.uint8)
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(comp)
    for ci, c in enumerate(cnts):                              # outer shapes filled; real gaps stay open
        if hier[0][ci][3] == -1:
            cv2.drawContours(solid, [c], -1, 1, -1)
    for ci, c in enumerate(cnts):
        if hier[0][ci][3] != -1 and cv2.contourArea(c) > 40:
            cv2.drawContours(solid, [c], -1, 0, -1)
    dotm = np.zeros(comp.shape, np.uint8)
    for (dx, dy) in dots:
        cv2.circle(dotm, (int(round(dx - x0)), int(round(dy - y0))), 11, 1, -1)
    near = cv2.dilate(solid, np.ones((25, 25), np.uint8))
    solid = np.maximum(solid, dotm & near)                      # a disc on the edge becomes shape
    core1 = cv2.erode(solid, np.ones((3, 3), np.uint8))
    # joint discs: copy a matching patch of the piece's own texture over each disc
    known = ((core1 > 0)).astype(np.uint8)
    for (dx, dy) in dots:
        known = cv2.circle(known, (int(round(dx - x0)), int(round(dy - y0))), 13, 0, -1)
    cf = crop.astype(np.float32)
    fixed = cf.copy()
    for (dx, dy) in dots:
        cx, cy = int(round(dx - x0)), int(round(dy - y0))
        rr = 11
        yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
        dd = np.hypot(xx - cx, yy - cy)
        ring = (dd > rr + 1) & (dd < rr + 5) & (core1 > 0)
        best, bo = 1e18, None
        for dist_ in (26, 32, 38, 46):
            for ang in range(0, 360, 20):
                ox = int(round(dist_ * math.cos(math.radians(ang)))); oy = int(round(dist_ * math.sin(math.radians(ang))))
                sx, sy = cx + ox, cy + oy
                if sx - rr - 5 < 0 or sy - rr - 5 < 0 or sx + rr + 5 >= crop.shape[1] or sy + rr + 5 >= crop.shape[0]:
                    continue
                src = (np.hypot(xx - sx, yy - sy) < rr + 5)
                if known[src].min() == 0 or (core1[src] == 0).any():
                    continue
                ys_, xs_ = np.where(ring)
                if len(ys_) < 20:
                    continue
                ok = (ys_ + oy >= 0) & (ys_ + oy < crop.shape[0]) & (xs_ + ox >= 0) & (xs_ + ox < crop.shape[1])
                ys_, xs_ = ys_[ok], xs_[ok]
                if len(ys_) < 20 or not known[ys_ + oy, xs_ + ox].all():
                    continue
                sc_ = np.abs(cf[ys_, xs_] - cf[ys_ + oy, xs_ + ox]).mean()
                if sc_ < best:
                    best, bo = sc_, (ox, oy)
        m = np.clip((rr + 2.5 - dd) / 3.0, 0, 1)[..., None]
        if bo is None:                                         # no clean patch nearby: smooth fill
            kn0 = ((core1 > 0) & (dd > rr + 1)).astype(np.float32)
            for sg in (3.0, 6.0, 12.0):
                num = cv2.GaussianBlur(fixed * kn0[..., None], (0, 0), sg)
                den = cv2.GaussianBlur(kn0, (0, 0), sg)[..., None]
                fixed = np.where((m > 0) & (den > 0.05), fixed * (1 - m) + (num / np.maximum(den, 1e-4)) * m, fixed)
                kn0 = np.maximum(kn0, (m[..., 0] > 0).astype(np.float32) * (den[..., 0] > 0.05))
            continue
        ox, oy = bo
        src_img = np.zeros_like(cf)
        ys0, ys1 = max(0, -oy), min(crop.shape[0], crop.shape[0] - oy)
        xs0, xs1 = max(0, -ox), min(crop.shape[1], crop.shape[1] - ox)
        src_img[ys0:ys1, xs0:xs1] = cf[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox]
        fixed = fixed * (1 - m) + src_img * m
    # outside the outline: colours continued from the interior (no background tint)
    unknown = ((1 - core1)).astype(np.uint8)
    kn = (core1 > 0).astype(np.float32)
    for sg in (3.0, 7.0, 16.0):
        num = cv2.GaussianBlur(fixed * kn[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(kn, (0, 0), sg)[..., None]
        fill = num / np.maximum(den, 1e-4)
        todo = (unknown > 0) & (den[..., 0] > 0.02)
        fixed[todo] = fill[todo]
        kn = np.maximum(kn, todo.astype(np.float32))
    fixed = np.clip(fixed, 0, 255).astype(np.uint8)
    lum = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a_edge = np.clip((235.0 - lum) / (235.0 - 45.0), 0, 1)
    a = np.where(core1 > 0, 1.0, np.where(solid > 0, np.maximum(a_edge, 0.0), 0.0)).astype(np.float32)
    a = np.where((dotm > 0) & (solid > 0), 1.0, a).astype(np.float32)   # joint areas stay solid
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    a = np.clip(a * (solid > 0), 0, 1)
    spr = np.dstack([fixed, (a * 255).astype(np.uint8)])
    return spr, (x0, y0)


NAMES = {1: "head_top", 2: "torso", 3: "arm_up_L", 4: "arm_up_R", 5: "fore_L", 6: "fore_R", 7: "pelvis",
         8: "hand_L", 9: "hand_R", 10: "leg_up_L", 11: "leg_up_R", 12: "leg_lo_L", 13: "leg_lo_R",
         14: "shoe_L", 15: "shoe_R", 16: "head_open", 17: "head_closed", 18: "head_neutral"}

# joints in sheet coordinates (the grey discs on the sheet, plus a few read off the silhouettes)
JOINTS = {
    "torso": {"collar": (804, 207), "hem": (808, 490), "shoulder_L": (704, 262), "shoulder_R": (907, 262)},
    "pelvis": {"waist": (808, 525), "hip_L": (755, 590), "hip_R": (863, 589)},
    "arm_up_L": {"shoulder": (646, 248), "elbow": (622, 341)},
    "arm_up_R": {"shoulder": (965, 248), "elbow": (988, 341)},
    "fore_L": {"elbow": (618, 378), "wrist": (606, 497)},
    "fore_R": {"elbow": (994, 378), "wrist": (1010, 497)},
    "hand_L": {"wrist": (605, 533)},
    "hand_R": {"wrist": (1016, 533)},
    "leg_up_L": {"hip": (754, 625), "knee": (745, 750)},
    "leg_up_R": {"hip": (866, 625), "knee": (874, 750)},
    "leg_lo_L": {"knee": (744, 788), "ankle": (738, 923)},
    "leg_lo_R": {"knee": (875, 788), "ankle": (882, 923)},
    "shoe_L": {"ankle": (738, 941)},
    "shoe_R": {"ankle": (882, 941)},
}
HEAD_DOT = {"head_neutral": (618, 1227), "head_open": (830, 1227), "head_closed": (1043, 1228)}
HEAD_SCALE = 0.81            # the three expression heads were drawn larger than the body parts
HEAD_K = 3                   # stored resolution multiplier
HEAD_W, HEAD_H, HEAD_PIVOT = 260, 300, (130, 262)


def head_canvas(spr, name, shift=(0.0, 0.0)):
    s, (x0, y0) = spr[name]
    d = HEAD_DOT[name]
    k = HEAD_SCALE * HEAD_K
    r = cv2.resize(s, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    ox = HEAD_PIVOT[0] * HEAD_K - (d[0] - x0) * k + shift[0] * HEAD_K
    oy = HEAD_PIVOT[1] * HEAD_K - (d[1] - y0) * k + shift[1] * HEAD_K
    return cv2.warpAffine(r, np.float32([[1, 0, ox], [0, 1, oy]]), (HEAD_W * HEAD_K, HEAD_H * HEAD_K),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    im = cv2.imread(SHEET)
    dots = find_dots(im)
    lab, comps = components(im)
    box = {i: b for i, b in comps}
    spr = {}
    for i, nm in NAMES.items():
        x, y, w, h, _ = box[i]
        inside = [d for d in dots if x - 6 <= d[0] <= x + w + 6 and y - 6 <= d[1] <= y + h + 6]
        spr[nm] = cut(im, lab, i, box[i], inside)
    meta = {"parts": {}, "head": {}}
    for nm, joints in JOINTS.items():
        s, (x0, y0) = spr[nm]
        cv2.imwrite(os.path.join(OUT, nm + ".png"), s)
        meta["parts"][nm] = {"origin": [int(x0), int(y0)], "joints": {k: list(v) for k, v in joints.items()}}
    # expression heads: the three drawings are aligned on the neck disc, then refined with ECC
    base = head_canvas(spr, "head_neutral")
    g = lambda a: cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    K = HEAD_K
    for nm, excl in (("head_open", (100, 160, 210, 250)), ("head_closed", (88, 172, 178, 212))):
        a = head_canvas(spr, nm)
        mask = np.zeros(base.shape[:2], np.uint8)
        mask[95 * K:260 * K, 85 * K:180 * K] = 255
        mask[excl[1] * K:excl[3] * K, excl[0] * K:excl[2] * K] = 0
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(g(base), g(a), warp, cv2.MOTION_TRANSLATION,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
            a = cv2.warpAffine(a, warp, (HEAD_W * K, HEAD_H * K), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            pass
        cv2.imwrite(os.path.join(OUT, nm + ".png"), a)
    cv2.imwrite(os.path.join(OUT, "head_neutral.png"), base)
    meta["head"] = {"k": K, "w": HEAD_W, "h": HEAD_H, "pivot": list(HEAD_PIVOT),
                    "mouth": [114, 221, 148, 243], "eye_L": [90, 178, 122, 204], "eye_R": [136, 178, 170, 204],
                    "brow_L": [94, 166, 124, 184], "brow_R": [138, 166, 168, 184]}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print("wrote", OUT, "-", len(JOINTS), "body parts + 3 heads")


if __name__ == "__main__":
    main()
