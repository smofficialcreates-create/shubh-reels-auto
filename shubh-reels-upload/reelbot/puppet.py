"""Shubh 2D puppet: a reusable character component.

Layers (cut from assets/shubh_sheet.png by reelbot/prep_puppet.py): head (hair, face, glasses,
neck), eyelids, eyebrows, mouth, torso, pelvis, upper arms, forearms, hands, upper legs,
lower legs, shoes. Joints (neck, shoulders, elbows, wrists, hips, knees, ankles) come from
the pivot discs printed on the sheet.

    from reelbot.puppet import Shubh, Pose
    s = Shubh()                                   # loads assets/puppet
    img = s.render_rgba(Pose(r_sh=30, r_el=150))  # transparent premultiplied-free BGRA image
    s.draw(frame_bgr, pose, anchor_frame=(540, 1400), zoom=1.0)   # onto an opaque frame

Pose angles are degrees. Positive shoulder = arm away from the body; positive elbow = forearm
swings outward/up, negative = hand comes in toward the chest. Left/right are screen sides.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields

import cv2
import numpy as np

ASSET_DIR = "assets/puppet"
UP = 2.0                       # sprites are stored at 1x and resampled once to UP x at load


@dataclass
class Pose:
    # arms (screen-left "l", screen-right "r")
    l_sh: float = 2.0
    l_el: float = -5.0
    l_wr: float = 0.0
    r_sh: float = 2.0
    r_el: float = -5.0
    r_wr: float = 0.0
    # head
    head_rot: float = 0.0          # tilt, degrees (+ = clockwise on screen)
    yaw: float = 0.0               # -1..1  turns the face features left/right inside the head outline
    pitch: float = 0.0             # -1..1  looks up (-) / down (+)
    head_dx: float = 0.0           # world units
    head_dy: float = 0.0
    # body
    lean: float = 0.0              # torso lean about the waist, degrees
    bob: float = 0.0               # whole body vertical, world units
    sway: float = 0.0              # whole body sideways, world units
    breath: float = 0.0            # torso scale pulse (0.004 is plenty)
    shrug: float = 0.0             # shoulders up, world units
    # face
    mouth_open: float = 0.0        # 0..1
    mouth_w: float = 0.5           # 0 round .. 1 wide
    smile: float = 0.0             # 0..1 (widens the mouth shape a little)
    blink: float = 0.0             # 0 open .. 1 closed
    brow: float = 0.0              # -1 lowered .. 1 raised
    hair: float = 0.0              # hair sway, world units (+ = to the right)


POSE_FIELDS = [f.name for f in fields(Pose)]


# ---------------------------------------------------------------------------------------
def _rot(deg, cx, cy):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy, ], [0, 0, 1]], np.float64)


def _tr(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


def _sc(sx, sy, cx=0.0, cy=0.0):
    return np.array([[sx, 0, cx - sx * cx], [0, sy, cy - sy * cy], [0, 0, 1]], np.float64)


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class _Sprite:
    """Premultiplied BGRA float32 sprite, resampled to UP x, with its sheet-space placement."""

    def __init__(self, bgra_u8: np.ndarray, origin, pad=0):
        a = bgra_u8.astype(np.float32)
        a[..., :3] *= a[..., 3:4] / 255.0
        if pad:
            a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)))
        self.img = cv2.resize(a, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
        self.img = np.clip(self.img, 0, 255)
        self.img[..., :3] = np.minimum(self.img[..., :3], self.img[..., 3:4])
        self.ox, self.oy = origin[0] - pad, origin[1] - pad
        self.h, self.w = self.img.shape[:2]


def _cap(bgra: np.ndarray, joint_local, radius: int, pad: int):
    """Hidden overlap artwork: a disc in the garment/skin colour centred on a joint, painted
    into the sprite's padding so that a rotating limb never shows a gap or a square corner."""
    h, w = bgra.shape[:2]
    jx, jy = int(round(joint_local[0])) + pad, int(round(joint_local[1])) + pad
    big = np.pad(bgra, ((pad, pad), (pad, pad), (0, 0)))
    yy, xx = np.mgrid[0:big.shape[0], 0:big.shape[1]]
    row = np.where(big[jy, :, 3] > 200)[0] if 0 <= jy < big.shape[0] else []
    if len(row) > 4:                                    # never wider than the limb itself
        radius = min(radius, 0.5 * (row.max() - row.min()) * 0.96)
    near = (np.hypot(xx - jx, yy - jy) < radius * 0.8) & (big[..., 3] > 250)
    if near.sum() < 30:
        return big
    col = np.median(big[near][:, :3], axis=0)
    disc = (np.hypot(xx - jx, yy - jy) <= radius) & (big[..., 3] < 255)
    soft = np.clip((radius + 0.5 - np.hypot(xx - jx, yy - jy)), 0, 1)
    blend = soft * (big[..., 3] < 255)
    for c in range(3):
        big[..., c] = np.where(disc, (big[..., c] * (1 - blend) + col[c] * blend), big[..., c])
    big[..., 3] = np.maximum(big[..., 3], (soft * 255).astype(np.uint8))
    return big


class Shubh:
    # chain: (part, parent joint on the parent, own joint)
    def __init__(self, asset_dir: str = ASSET_DIR, scale: float = 2.2, neck_drop: float = -7.0, leg_stretch: float = 1.4):
        self.dir = asset_dir
        meta = json.load(open(os.path.join(asset_dir, "meta.json")))
        self.meta = meta
        self.scale = scale                 # frame pixels per world unit at zoom 1
        P = meta["parts"]
        raw = {}
        for nm in P:
            raw[nm] = cv2.imread(os.path.join(asset_dir, nm + ".png"), cv2.IMREAD_UNCHANGED)
        J = {nm: {k: np.array(v, np.float64) for k, v in P[nm]["joints"].items()} for nm in P}
        O = {nm: P[nm]["origin"] for nm in P}
        # the sheet's leg pieces are shorter than the full-figure reference: stretch them to match
        for nm, k in (("leg_up_L", leg_stretch), ("leg_up_R", leg_stretch), ("leg_lo_L", leg_stretch), ("leg_lo_R", leg_stretch)):
            h0 = raw[nm].shape[0]
            raw[nm] = cv2.resize(raw[nm], (raw[nm].shape[1], int(round(h0 * k))), interpolation=cv2.INTER_CUBIC)
            for jn in J[nm]:
                J[nm][jn][1] = O[nm][1] + (J[nm][jn][1] - O[nm][1]) * k
        # --- rest assembly: translation of each part's sheet coords into the assembled body ---
        T = {"torso": np.zeros(2)}
        T["pelvis"] = J["torso"]["hem"] - J["pelvis"]["waist"]
        self.rest = {}
        for s in ("L", "R"):
            T[f"arm_up_{s}"] = J["torso"][f"shoulder_{s}"] - J[f"arm_up_{s}"]["shoulder"]
            T[f"fore_{s}"] = J[f"arm_up_{s}"]["elbow"] + T[f"arm_up_{s}"] - J[f"fore_{s}"]["elbow"]
            T[f"hand_{s}"] = J[f"fore_{s}"]["wrist"] + T[f"fore_{s}"] - J[f"hand_{s}"]["wrist"]
            T[f"leg_up_{s}"] = J["pelvis"][f"hip_{s}"] + T["pelvis"] - J[f"leg_up_{s}"]["hip"]
            T[f"leg_lo_{s}"] = J[f"leg_up_{s}"]["knee"] + T[f"leg_up_{s}"] - J[f"leg_lo_{s}"]["knee"]
            T[f"shoe_{s}"] = J[f"leg_lo_{s}"]["ankle"] + T[f"leg_lo_{s}"] - J[f"shoe_{s}"]["ankle"]
        self.T = T
        self.J = J
        # joints in assembled (rest) world coordinates
        W = lambda part, jn: J[part][jn] + T[part]
        self.W = {
            "hem": W("torso", "hem"),
            "collar": W("torso", "collar"),
            "shoulder_L": W("torso", "shoulder_L"), "shoulder_R": W("torso", "shoulder_R"),
            "elbow_L": W("arm_up_L", "elbow"), "elbow_R": W("arm_up_R", "elbow"),
            "wrist_L": W("fore_L", "wrist"), "wrist_R": W("fore_R", "wrist"),
        }
        self.neck = self.W["collar"] + np.array([0.0, neck_drop])
        # --- sprites, with hidden overlap caps on the joints ---
        self.sp = {}
        caps = {  # part: (joint, radius, pad)
            "arm_up_L": ("shoulder", 33, 40), "arm_up_R": ("shoulder", 33, 40),
            "fore_L": ("elbow", 36, 40), "fore_R": ("elbow", 36, 40),
            "hand_L": ("wrist", 21, 30), "hand_R": ("wrist", 21, 30),
            "leg_lo_L": ("knee", 33, 40), "leg_lo_R": ("knee", 33, 40),
        }
        for nm, img in raw.items():
            ox, oy = O[nm]
            if nm in caps:
                jn, r, pad = caps[nm]
                local = J[nm][jn] - np.array([ox, oy])
                img = _cap(img, local, r, pad)
                self.sp[nm] = _Sprite(img, (ox - pad, oy - pad), 0)
            else:
                self.sp[nm] = _Sprite(img, (ox, oy), 0)
        self._load_head()

    # ------------------------------------------------------------------ head / face layers
    def _load_head(self):
        H = self.meta["head"]
        self.K = H["k"]
        self.pivot = np.array(H["pivot"], np.float64)
        self.rect = {k: H[k] for k in ("mouth", "eye_L", "eye_R", "brow_L", "brow_R")}
        rd = lambda n: cv2.imread(os.path.join(self.dir, n + ".png"), cv2.IMREAD_UNCHANGED)
        self.base = rd("head_neutral")                      # eyes open, mouth closed
        self.open_h = rd("head_open")
        self.closed_h = rd("head_closed")
        K = self.K
        h, w = self.base.shape[:2]
        self.hw = (w, h)
        r = lambda key: [int(v * K) for v in self.rect[key]]
        # mouth: a skin-only patch (closed-mouth line painted out) + the lips from the open drawing
        x0, y0, x1, y1 = r("mouth")
        nb = self.base[y0:y1, x0:x1, :3]
        gray = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)
        line = (gray < np.percentile(gray, 22)).astype(np.uint8) * 255
        line = cv2.dilate(line, np.ones((5, 5), np.uint8))
        self.mouth_skin = cv2.inpaint(nb, line, 5, cv2.INPAINT_TELEA)
        op = self.open_h[y0:y1, x0:x1, :3]
        d = np.abs(op.astype(np.int32) - self.mouth_skin.astype(np.int32)).sum(2).astype(np.float32)
        m = (d > 55).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        if n > 1:
            big = 1 + int(np.argmax(st[1:, 4]))
            keep[lab == big] = 1
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * K + 1, 2 * K + 1)))
        al = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.8 * K / 2)
        self.lips = np.dstack([op.astype(np.float32), np.clip(al * 1.3, 0, 1) * 255])
        ys, xs = np.where(keep > 0)
        self.lip_c = (float(xs.mean()), float(ys.mean())) if len(xs) else ((x1 - x0) / 2, (y1 - y0) / 2)
        self.lip_top = float(ys.min()) if len(ys) else 0.0
        # blink patches: closed drawing, both eyes, with a soft rectangular feather
        self.eye_patch = {}
        for side in ("eye_L", "eye_R"):
            ex0, ey0, ex1, ey1 = r(side)
            self.eye_patch[side] = (ex0, ey0, ex1, ey1, self.closed_h[ey0:ey1, ex0:ex1, :3].astype(np.float32))
        # displacement weight maps (head canvas, K resolution)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ux, uy = xx / K, yy / K
        face = _smooth(1.0 - np.sqrt(((ux - 130) / 44.0) ** 2 + ((uy - 205) / 56.0) ** 2))
        self.w_face = face
        lum = cv2.cvtColor(self.base[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        dark = _smooth((110.0 - lum) / 60.0) * (self.base[..., 3] > 128)
        ramp = _smooth((178.0 - uy) / 62.0)
        side = _smooth((np.abs(ux - 130) - 38) / 20.0) * _smooth((200.0 - uy) / 40.0) * _smooth((uy - 140) / 30.0)
        hair = cv2.GaussianBlur(dark * np.maximum(ramp, 0.45 * side), (0, 0), 2.0 * K) * (1 - 0.85 * face)
        self.w_hair = np.clip(hair * 1.6, 0, 1)
        self.w_brow = {}
        for side_ in ("brow_L", "brow_R"):
            bx0, by0, bx1, by1 = self.rect[side_]
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            self.w_brow[side_] = _smooth(1 - np.sqrt(((ux - cx) / ((bx1 - bx0) / 2 + 4)) ** 2 + ((uy - cy) / ((by1 - by0) / 2 + 2)) ** 2))
        self.w_brows = np.maximum(self.w_brow["brow_L"], self.w_brow["brow_R"])
        self.gx, self.gy = xx, yy

    def compose_head(self, p: Pose) -> np.ndarray:
        """Head canvas (K x resolution, straight BGRA uint8) for this expression."""
        K = self.K
        img = self.base.astype(np.float32).copy()
        # eyelids sweep down from the top of each lens
        if p.blink > 0.02:
            for side, (ex0, ey0, ex1, ey1, patch) in self.eye_patch.items():
                hh = ey1 - ey0
                yy = (np.arange(hh, dtype=np.float32) + 0.5)[:, None]
                edge = p.blink * (hh + 2 * K)
                m = np.clip((edge - yy) / (2.5 * K), 0, 1)
                fx = np.minimum(np.arange(ex1 - ex0, dtype=np.float32) + 0.5, (ex1 - ex0) - np.arange(ex1 - ex0, dtype=np.float32) - 0.5)
                m = m * np.clip(fx / (3.0 * K), 0, 1)[None, :]
                roi = img[ey0:ey1, ex0:ex1, :3]
                roi[:] = roi * (1 - m[..., None]) + patch * m[..., None]
        # mouth
        mx0, my0, mx1, my1 = [int(v * K) for v in self.rect["mouth"]]
        o = float(np.clip(p.mouth_open, 0, 1))
        wide = float(np.clip(0.9 + 0.35 * (p.mouth_w - 0.5) + 0.12 * p.smile, 0.7, 1.25))
        if o > 0.07:
            a = float(_smooth((o - 0.07) / 0.12))
            roi = img[my0:my1, mx0:mx1, :3]
            roi[:] = roi * (1 - a) + self.mouth_skin.astype(np.float32) * a
            sy = 0.25 + 0.95 * o
            cx, cy = self.lip_c
            top = self.lip_top
            M = np.float32([[wide, 0, cx - wide * cx], [0, sy, top - sy * top + (1 - sy) * 0.0]])
            lp = cv2.warpAffine(self.lips, M, (mx1 - mx0, my1 - my0), flags=cv2.INTER_LINEAR, borderValue=0)
            la = lp[..., 3:4] / 255.0 * a
            roi[:] = roi * (1 - la) + lp[..., :3] * la
        # displacement: yaw/pitch of the face features, brow raise, hair sway
        dx = np.zeros(img.shape[:2], np.float32)
        dy = np.zeros(img.shape[:2], np.float32)
        used = False
        if abs(p.yaw) > 1e-3 or abs(p.pitch) > 1e-3:
            dx += p.yaw * 7.0 * K * self.w_face
            dy += p.pitch * 5.0 * K * self.w_face
            used = True
        if abs(p.brow) > 1e-3:
            dy -= p.brow * 2.6 * K * self.w_brows
            used = True
        if abs(p.hair) > 1e-3:
            dx += p.hair * K * self.w_hair
            used = True
        if used:
            img = cv2.remap(img, self.gx - dx, self.gy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ skeleton
    def matrices(self, p: Pose):
        """world-rest -> world-posed matrices for every part (3x3), no camera applied."""
        W = self.W
        root = _tr(p.sway, p.bob)
        hem = W["hem"]
        torso = root @ _rot(p.lean, hem[0], hem[1]) @ _sc(1 + 0.4 * p.breath, 1 + p.breath, hem[0], hem[1])
        M = {"torso": torso, "pelvis": root}
        for s, sg in (("L", 1.0), ("R", -1.0)):
            sh = W[f"shoulder_{s}"]
            up = torso @ _tr(0, -p.shrug) @ _rot(sg * (p.l_sh if s == "L" else p.r_sh), sh[0], sh[1])
            el = W[f"elbow_{s}"]
            fo = up @ _rot(sg * (p.l_el if s == "L" else p.r_el), el[0], el[1])
            wr = W[f"wrist_{s}"]
            ha = fo @ _rot(sg * (p.l_wr if s == "L" else p.r_wr), wr[0], wr[1])
            M[f"arm_up_{s}"], M[f"fore_{s}"], M[f"hand_{s}"] = up, fo, ha
            M[f"leg_up_{s}"] = root
            M[f"leg_lo_{s}"] = root
            M[f"shoe_{s}"] = root
        n = self.neck
        M["head"] = torso @ _tr(p.head_dx, p.head_dy) @ _rot(p.head_rot, n[0], n[1])
        return M

    # ------------------------------------------------------------------ drawing
    ORDER = ["shoe_L", "shoe_R", "leg_lo_L", "leg_lo_R", "leg_up_L", "leg_up_R", "pelvis", "head", "torso",
             "hand_L", "hand_R", "fore_L", "fore_R", "arm_up_L", "arm_up_R"]

    def _camera(self, anchor_frame, anchor_world, zoom):
        s = self.scale * zoom
        return np.array([[s, 0, anchor_frame[0] - s * anchor_world[0]],
                         [0, s, anchor_frame[1] - s * anchor_world[1]], [0, 0, 1]], np.float64)

    def _blit(self, dst, img, A, alpha=1.0):
        """warp premultiplied float BGRA img by 3x3 A (img px -> dst px) and composite 'over'."""
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T
        pts = (A @ corners)[:2]
        H_, W_ = dst.shape[:2]
        x0, y0 = int(max(0, math.floor(pts[0].min()) - 1)), int(max(0, math.floor(pts[1].min()) - 1))
        x1, y1 = int(min(W_, math.ceil(pts[0].max()) + 1)), int(min(H_, math.ceil(pts[1].max()) + 1))
        if x1 <= x0 or y1 <= y0:
            return
        A2 = _tr(-x0, -y0) @ A
        patch = cv2.warpAffine(img, A2[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = patch[..., 3:4] / 255.0 * alpha
        roi = dst[y0:y1, x0:x1]
        if dst.shape[2] == 3:
            roi[:] = np.clip(patch[..., :3] * alpha + roi.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)
        else:                                            # float32 premultiplied canvas
            roi[:] = patch * alpha + roi * (1 - a)

    def _draw_on(self, dst, p: Pose, cam):
        M = self.matrices(p)
        head = self.compose_head(p)
        hpm = head.astype(np.float32)
        hpm[..., :3] *= hpm[..., 3:4] / 255.0
        for nm in self.ORDER:
            if nm == "head":
                # canvas px (K res) -> world: u/K + (neck - pivot)
                off = self.neck - self.pivot
                A = cam @ M["head"] @ _tr(off[0], off[1]) @ _sc(1.0 / self.K, 1.0 / self.K)
                self._blit(dst, hpm, A)
                continue
            sp = self.sp[nm]
            t = self.T[nm]
            # sprite px -> sheet (u/UP + origin) -> world rest (+T) -> posed -> camera
            A = cam @ M[nm] @ _tr(sp.ox + t[0], sp.oy + t[1]) @ _sc(1.0 / UP, 1.0 / UP)
            self._blit(dst, sp.img, A)

    def draw(self, frame: np.ndarray, p: Pose, anchor_frame=(540.0, 1405.0), zoom: float = 1.0, anchor_world=None):
        """Draw onto an opaque BGR uint8 frame (in place). anchor_world defaults to the neck."""
        aw = self.neck if anchor_world is None else anchor_world
        self._draw_on(frame, p, self._camera(anchor_frame, aw, zoom))

    def render_rgba(self, p: Pose, size=(1080, 1920), anchor_frame=None, zoom: float = 1.0, full_body=False):
        """Transparent render: straight-alpha BGRA uint8 of the given size."""
        w, h = size
        if full_body:
            sc_ = self.scale
            self.scale = 0.92 * h / 1000.0
            anchor_frame = anchor_frame or (w / 2, h * 0.17)
        anchor_frame = anchor_frame or (w / 2, h * 0.73)
        cv = np.zeros((h, w, 4), np.float32)
        self._draw_on(cv, p, self._camera(anchor_frame, self.neck, zoom))
        if full_body:
            self.scale = sc_
        a = cv[..., 3:4]
        rgb = np.where(a > 0.5, cv[..., :3] * 255.0 / np.maximum(a, 1e-3), 0)
        return np.dstack([np.clip(rgb, 0, 255), np.clip(a, 0, 255)]).astype(np.uint8)

    def head_position(self, p: Pose, anchor_frame, zoom=1.0):
        """Frame position of the head centre (for effects that follow the head)."""
        cam = self._camera(anchor_frame, self.neck, zoom)
        M = self.matrices(p)["head"]
        v = cam @ M @ np.array([self.neck[0], self.neck[1] - 75.0, 1.0])
        return float(v[0]), float(v[1])
