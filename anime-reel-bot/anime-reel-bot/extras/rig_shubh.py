"""Layered 2D rig for Shubh (assets/shubh/*.png, cut by reelbot/prep_shubh.py).

Layers: head, torso, two arms (the left one is the mirrored right arm, each split into
upper arm / forearm / hand) and the backpack straps. Every frame:

* the head sprite is warped (jaw drop for speech, a small yaw squash, hair lag) and a
  mouth interior plus eyelids are painted on it,
* arms are posed through shoulder, elbow and wrist joints with forearm foreshortening,
* the whole body leans, breathes, shifts weight and shrugs.

Everything is CPU only (OpenCV + numpy).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

ASSET_DIR = "assets/shubh"
DS = 2.0                          # stored sprite pixels per source pixel
K0 = 1.7                          # frame pixels per source pixel at zoom 1
AXIS = 205.0                      # body mirror axis

# joints (source px). Left side = mirror of right about AXIS
SHOULDER_R, ELBOW_R, WRIST_R = (335.0, 322.0), (352.0, 548.0), (338.0, 746.0)
NECK = (180.0, 262.0)
HEM = (205.0, 700.0)
FACE_X = 175.0
MOUTH = (175.0, 210.0)
EYES = ((152.0, 162.0), (195.0, 163.0))
HEAD_CENTER = (180.0, 150.0)      # used for placement
ANCHOR_SRC = (190.0, 150.0)       # source point that sits at anchor_frame

HEAD = HEAD_CENTER


def mirror_pt(p):
    return (2 * AXIS - p[0], p[1])


@dataclass
class Pose:
    # shoulder / elbow / wrist angles in degrees. Positive shoulder = arm away from the body.
    # Positive elbow = forearm swings away from the body, negative = hand comes in toward the chest.
    l_sh: float = 2.0
    l_el: float = 0.0
    l_wr: float = 0.0
    r_sh: float = 2.0
    r_el: float = 0.0
    r_wr: float = 0.0
    head_rot: float = 0.0
    head_dx: float = 0.0
    head_dy: float = 0.0
    yaw: float = 0.0              # -1..1 head turn illusion
    lean: float = 0.0
    bob: float = 0.0
    sway: float = 0.0             # whole body sideways (source px)
    breath: float = 0.0
    shrug: float = 0.0
    mouth_open: float = 0.0
    mouth_w: float = 0.5
    blink: float = 0.0
    smile: float = 0.0
    hair: float = 0.0
    l_fore: float = 0.0           # forearm foreshortening 0..1 (toward camera)
    r_fore: float = 0.0
    l_up: float = 0.0             # upper-arm foreshortening 0..1 (elbow forward)
    r_up: float = 0.0


def _mat_rot(angle_deg, cx, cy):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]], np.float64)


def _mat_t(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


def _mat_s(sx, sy, cx=0.0, cy=0.0):
    return np.array([[sx, 0, cx - sx * cx], [0, sy, cy - sy * cy], [0, 0, 1]], np.float64)


class Sprite:
    def __init__(self, img, x0, y0):
        self.img, self.x0, self.y0 = img, float(x0), float(y0)    # premultiplied float32 RGBA at DS

    @property
    def h(self):
        return self.img.shape[0]

    @property
    def w(self):
        return self.img.shape[1]

    def flipped(self):
        """Mirror about the body axis."""
        img = np.ascontiguousarray(self.img[:, ::-1])
        x1 = self.x0 + self.w / DS
        return Sprite(img, 2 * AXIS - x1, self.y0)


def _load(name, meta):
    m = meta[name]
    im = cv2.imread(os.path.join(ASSET_DIR, f"{name}.png"), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"{ASSET_DIR}/{name}.png (run python -m reelbot.prep_shubh)")
    rgb = im[..., :3][..., ::-1].astype(np.float32) / 255.0
    a = im[..., 3:4].astype(np.float32) / 255.0
    pm = np.dstack([rgb * a, a])
    f = DS / m["scale"]
    pm = cv2.resize(pm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return Sprite(pm.astype(np.float32), m["x0"], m["y0"])


def _piece(sp: Sprite, keep: np.ndarray) -> Sprite:
    img = sp.img * keep[..., None]
    return Sprite(img.astype(np.float32), sp.x0, sp.y0)


class Mascot:
    def __init__(self, path: str = "assets/shubh"):
        global ASSET_DIR
        ASSET_DIR = path if os.path.isdir(path) else ASSET_DIR
        meta = json.load(open(os.path.join(ASSET_DIR, "meta.json")))
        self.head = _load("head", meta)
        self.torso = _load("torso", meta)
        self.strap = _load("strap", meta)
        arm = _load("arm", meta)
        # ---- split the right arm into upper / fore / hand ------------------------------
        h, w = arm.h, arm.w
        ys = (arm.y0 + (np.arange(h)[:, None] + 0.5) / DS) * np.ones((1, w), np.float32)
        xs = (arm.x0 + (np.arange(w)[None, :] + 0.5) / DS) * np.ones((h, 1), np.float32)
        ex, ey = ELBOW_R
        wx, wy = WRIST_R
        cap = ((xs - ex) ** 2 + (ys - ey) ** 2 < 46.0 ** 2).astype(np.float32)
        ramp = lambda v, edge, soft=3.0: np.clip((edge - v) / soft + 0.5, 0, 1)     # 1 above the edge
        upper = np.maximum(ramp(ys, ey), cap)
        fore = np.clip(1 - ramp(ys, ey - 14.0), 0, 1) * ramp(ys, wy + 6.0)
        hand = 1 - ramp(ys, wy - 6.0)
        self.arm = {"up": _piece(arm, upper), "fore": _piece(arm, fore), "hand": _piece(arm, hand)}
        self.arm_l = {k: v.flipped() for k, v in self.arm.items()}
        self._grid = None
        self._skin = self._sample_skin()
        self.SC = K0

    # ---------------------------------------------------------------- helpers
    def _sample_skin(self):
        src = cv2.imread("assets/shubh_source.jpg")
        patch = src[177:184, 146:160].reshape(-1, 3)
        return (np.median(patch, axis=0)[::-1] / 255.0).astype(np.float32)               # RGB 0..1

    def head_screen(self, anchor_frame, zoom):
        k = K0 * zoom
        return (anchor_frame[0] + (HEAD_CENTER[0] - ANCHOR_SRC[0]) * k, anchor_frame[1] + (HEAD_CENTER[1] - ANCHOR_SRC[1]) * k)

    # ------------------------------------------------------------- head warp
    def _head_grid(self):
        if self._grid is None:
            sp = self.head
            u = np.arange(sp.w, dtype=np.float32)[None, :]
            v = np.arange(sp.h, dtype=np.float32)[:, None]
            sx = sp.x0 + (u + 0.5) / DS + 0 * v
            sy = sp.y0 + (v + 0.5) / DS + 0 * u
            self._grid = (sx.astype(np.float32), sy.astype(np.float32))
        return self._grid

    def _head_frame(self, p: Pose) -> np.ndarray:
        sp = self.head
        sx, sy = self._head_grid()
        need = abs(p.mouth_open) > 0.02 or abs(p.yaw) > 0.01 or abs(p.hair) > 0.05 or p.blink > 0.05
        img = sp.img
        if need:
            dx = np.zeros_like(sx)
            dy = np.zeros_like(sx)
            # hair lag: only the top of the hair shifts
            if abs(p.hair) > 0.02:
                dx += p.hair * np.clip((150.0 - sy) / 80.0, 0, 1) ** 1.3
            # jaw: lower lip and chin move down while talking
            if p.mouth_open > 0.02:
                wy = np.clip((sy - 209.0) / 6.0, 0, 1) * np.clip((268.0 - sy) / 30.0, 0, 1)
                wx = np.exp(-((sx - FACE_X) / 40.0) ** 2)
                dy += (2.4 * min(p.mouth_open, 1.0)) * wy * wx
            # blink: the upper lid comes down by squashing the eye strip toward its top edge
            if p.blink > 0.05:
                b = min(1.0, p.blink) * 0.78
                top, bot, back = 156.0, 172.0, 190.0
                for ex, _ey in EYES:
                    wxm = np.exp(-((sx - ex) / 17.0) ** 2)
                    inside = np.clip((sy - top) / 2.0, 0, 1) * np.clip((back - sy) / (back - bot), 0, 1)
                    shift = np.where(sy < bot, (sy - top) * b, (bot - top) * b * np.clip((back - sy) / (back - bot), 0, 1))
                    dy += -shift * wxm * inside
            # yaw: squash the face toward its centre line and shift the features
            if abs(p.yaw) > 0.01:
                c = 1.0 - 0.035 * abs(p.yaw)
                dx += -(sx - FACE_X) * (1 - c) / c + 3.2 * p.yaw * np.exp(-((sy - 170.0) / 60.0) ** 2)
            map_x = (sx - dx - sp.x0) * DS - 0.5
            map_y = (sy - dy - sp.y0) * DS - 0.5
            img = cv2.remap(sp.img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = img.copy()
        if p.mouth_open > 0.06:
            self._paint_mouth(img, p)
        if p.blink > 0.25:
            self._paint_blink(img, p)
        return img

    def _px(self, x, y):
        return ((x - self.head.x0) * DS, (y - self.head.y0) * DS)

    def _paint_mouth(self, img, p):
        sp = self.head
        S = 2
        h, w = img.shape[:2]
        mask = np.zeros((h * S // 2 * 2, w * S // 2 * 2), np.float32)
        op = min(1.0, p.mouth_open)
        wide = 0.5 + 0.5 * p.mouth_w
        half = (8.5 + 4.0 * wide - 2.0 * op) * DS * S
        hgt = (0.7 + 4.4 * op) * DS * S
        cx, cy = self._px(MOUTH[0] + 0.4, MOUTH[1] + 0.9 + 1.0 * op)
        cx, cy = cx * S, cy * S
        # D-shaped opening: flatter on top, rounder at the bottom
        pts_top = [(cx + t * half, cy - hgt * 0.30 * (1 - t * t) ** 0.8) for t in np.linspace(-1, 1, 30)]
        pts_bot = [(cx + t * half, cy + hgt * (1 - t * t) ** 0.65) for t in np.linspace(1, -1, 30)]
        cv2.fillPoly(mask, [np.array(pts_top + pts_bot, np.int32)], 1.0, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 0.9 * S)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)[..., None]
        inner = np.array([0.16, 0.06, 0.08], np.float32)
        # tongue hint when wide open
        col = np.broadcast_to(inner, (h, w, 3)).copy()
        if op > 0.45:
            tm = np.zeros((h * S, w * S), np.float32)
            cv2.ellipse(tm, (int(cx), int(cy + hgt * 0.62)), (int(half * 0.55), int(hgt * 0.34)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            tm = cv2.resize(cv2.GaussianBlur(tm, (0, 0), 1.2 * S), (w, h), interpolation=cv2.INTER_AREA)[..., None]
            col = col * (1 - tm * 0.8) + np.array([0.42, 0.20, 0.23], np.float32) * tm * 0.7
        # upper teeth sliver
        tt = np.zeros((h * S, w * S), np.float32)
        if op > 0.25:
            cv2.ellipse(tt, (int(cx), int(cy - hgt * 0.12)), (int(half * 0.78), max(1, int(hgt * 0.20))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            tt = cv2.resize(cv2.GaussianBlur(tt, (0, 0), 0.8 * S), (w, h), interpolation=cv2.INTER_AREA)[..., None]
            col = col * (1 - tt * 0.9) + np.array([0.92, 0.90, 0.88], np.float32) * tt * 0.9
        a = mask * img[..., 3:4]
        img[..., :3] = img[..., :3] * (1 - a) + col * a
        # keep the pixel opaque where the head was opaque
        img[..., 3:4] = np.maximum(img[..., 3:4], a)

    def _paint_blink(self, img, p):
        """Dark lash line along the closed lid."""
        h, w = img.shape[:2]
        S = 2
        b = min(1.0, p.blink)
        line = np.zeros((h * S, w * S), np.float32)
        for ex, ey in EYES:
            lid_y = 156.0 + 16.0 * (1 - 0.78 * b) + 0.5
            cx, cy = self._px(ex, lid_y)
            cv2.ellipse(line, (int(cx * S), int(cy * S)), (int(10.5 * DS * S), int(1.6 * DS * S)), 0, 5, 175, 1.0,
                        max(1, int(1.1 * DS * S)), cv2.LINE_AA)
        line = cv2.resize(cv2.GaussianBlur(line, (0, 0), 0.8 * S), (w, h), interpolation=cv2.INTER_AREA)[..., None]
        la = line * img[..., 3:4] * min(1.0, (b - 0.25) / 0.5) * 0.7
        img[..., :3] = img[..., :3] * (1 - la) + np.array([0.10, 0.07, 0.09], np.float32) * la

    # -------------------------------------------------------------- composition
    def draw(self, frame, p: Pose, anchor_frame=(540.0, 1190.0), anchor_src=ANCHOR_SRC, zoom=1.0, shear=0.0, alpha=1.0):
        k = K0 * zoom
        base = _mat_t(anchor_frame[0], anchor_frame[1]) @ _mat_s(k, k) @ _mat_t(-anchor_src[0], -anchor_src[1])
        body = (_mat_t(p.sway, p.bob) @ _mat_rot(p.lean, *HEM) @ _mat_s(1 + p.breath, 1 + 1.6 * p.breath, AXIS, 500.0))
        B = base @ body
        sh = p.shrug
        # torso
        _blend(frame, self.torso, B, alpha)
        # arms (hand under forearm under upper arm)
        for side in ("L", "R"):
            sets = self.arm_l if side == "L" else self.arm
            sho = SHOULDER_R if side == "R" else mirror_pt(SHOULDER_R)
            elb = ELBOW_R if side == "R" else mirror_pt(ELBOW_R)
            wri = WRIST_R if side == "R" else mirror_pt(WRIST_R)
            sg = 1.0 if side == "L" else -1.0
            s_, e_, w_ = (getattr(p, f"{side.lower()}_{n}") for n in ("sh", "el", "wr"))
            fore_amt = getattr(p, f"{side.lower()}_fore")
            up_amt = getattr(p, f"{side.lower()}_up")
            up = _mat_t(0, -sh) @ _mat_rot(s_ * sg, *sho) @ _mat_s(1.0, 1.0 - 0.5 * min(up_amt, 1.0), *sho)
            sy = 1.0 - 0.38 * min(fore_amt, 1.0)
            fo = up @ _mat_rot(e_ * sg, *elb) @ _mat_s(1.0, sy, *elb)
            ha = fo @ _mat_rot(w_ * sg, wri[0], wri[1] + 0)
            _blend(frame, sets["hand"], B @ ha, alpha)
            _blend(frame, sets["fore"], B @ fo, alpha)
            _blend(frame, sets["up"], B @ up, alpha)
        _blend(frame, self.strap, B @ _mat_t(0, -sh * 0.6), alpha)
        head_m = _mat_t(p.head_dx, p.head_dy - sh * 0.7) @ _mat_rot(p.head_rot, *NECK)
        _blend(frame, Sprite(self._head_frame(p), self.head.x0, self.head.y0), B @ head_m, alpha)

    def head_position(self, p, anchor_frame, anchor_src, zoom):
        return self.head_screen(anchor_frame, zoom)


def _blend(frame, sp: Sprite, M_src, alpha):
    img = sp.img
    h, w = img.shape[:2]
    to_src = _mat_t(sp.x0, sp.y0) @ _mat_s(1 / DS, 1 / DS)
    full = M_src @ to_src
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T
    c = (full @ corners)[:2]
    fx0, fy0 = np.floor(c.min(axis=1)).astype(int) - 1
    fx1, fy1 = np.ceil(c.max(axis=1)).astype(int) + 1
    H, W = frame.shape[:2]
    cx0, cy0, cx1, cy1 = max(fx0, 0), max(fy0, 0), min(fx1, W), min(fy1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    M = _mat_t(-cx0, -cy0) @ full
    warped = cv2.warpAffine(img, M[:2], (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_AREA if False else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4] * alpha
    roi = frame[cy0:cy1, cx0:cx1].astype(np.float32)
    frame[cy0:cy1, cx0:cx1] = np.clip(roi * (1 - a) + warped[..., :3][..., ::-1] * 255.0 * alpha, 0, 255).astype(np.uint8)
