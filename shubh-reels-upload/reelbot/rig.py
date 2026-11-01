"""A 2D puppet rig for the mascot (assets/mascot_full.jpg).

The art is cut into layers once: torso, head, two upper arms and two forearms.
Each frame the layers are moved with affine transforms (shoulder, elbow, neck,
body sway) and a talking mouth and blinking eyelids are painted onto the head.
Everything is CPU only (OpenCV + Pillow).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

SC = 2.2                       # frame pixels per source pixel at zoom 1
MIRROR_X = 1024                # arm polygons for the right side are mirrored about x = 512

# joints (source px)
SHOULDER = {"L": (410.0, 258.0), "R": (614.0, 258.0)}
ELBOW = {"L": (396.0, 412.0), "R": (628.0, 412.0)}
NECK = (512.0, 188.0)
MOUTH = (512.0, 167.0)
EYES = ((491.0, 131.0), (534.0, 131.0))

ARM_L = [(398, 236), (430, 246), (431, 330), (433, 420), (437, 505), (437, 540), (434, 614),
         (382, 614), (376, 540), (362, 500), (356, 440), (366, 350), (384, 280)]


@dataclass
class Pose:
    # arm angles in degrees: shoulder and elbow, positive = away from the body
    l_sh: float = 3.0
    l_el: float = 0.0
    r_sh: float = 3.0
    r_el: float = 0.0
    head_rot: float = 0.0
    head_dx: float = 0.0
    head_dy: float = 0.0
    lean: float = 0.0           # whole upper body lean (degrees)
    bob: float = 0.0            # whole body vertical offset (source px)
    breath: float = 0.0         # scale pulse
    shrug: float = 0.0          # shoulders up (source px)
    mouth_open: float = 0.0
    mouth_w: float = 0.5
    blink: float = 0.0
    smile: float = 0.0
    hair: float = 0.0           # hair sway in source px


def _mat_rot(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]], np.float64)


def _mat_t(dx, dy) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


def _mat_s(sx, sy, cx=0.0, cy=0.0) -> np.ndarray:
    return np.array([[sx, 0, cx - sx * cx], [0, sy, cy - sy * cy], [0, 0, 1]], np.float64)


class Sprite:
    def __init__(self, rgba_premul: np.ndarray, x0: float, y0: float):
        self.img = rgba_premul     # float32 H x W x 4, premultiplied, already scaled by SC
        self.x0, self.y0 = x0, y0  # source coords of the sprite's top-left corner


class Mascot:
    def __init__(self, path: str = "assets/mascot_full.jpg"):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        self.src = bgr
        h, w = bgr.shape[:2]
        # ---- character mask: colour distance from the flat background -----------
        corners = np.concatenate([bgr[:20, :20].reshape(-1, 3), bgr[:20, -20:].reshape(-1, 3),
                                  bgr[-20:, :20].reshape(-1, 3)])
        bg = np.median(corners, axis=0)
        dist = np.linalg.norm(bgr.astype(np.float32) - bg.astype(np.float32), axis=2)
        m = (dist > 34).astype(np.uint8)
        box = np.zeros_like(m)
        box[20:1000, 340:690] = 1                       # drops the corner watermark and shadows
        m &= box
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8)
        # fill enclosed holes (the pale irises are close to the background colour)
        inv = (1 - m).astype(np.uint8)
        ff = inv.copy()
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
        m = m | (ff == 1).astype(np.uint8)
        self.mask = m
        a_soft = np.clip((dist - 22.0) / 40.0, 0.0, 1.0)
        interior = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)
        solid = (interior > 0) | ((dist > 70) & (m > 0))
        edge = (~solid).astype(np.uint8) * 255
        self.fg = cv2.inpaint(bgr, edge, 3, cv2.INPAINT_TELEA).astype(np.float32)
        self.a_soft = a_soft
        self.skin = self._sample_skin(bgr)
        # ---- region masks -------------------------------------------------------
        arm_l = np.zeros((h, w), np.uint8)
        cv2.fillPoly(arm_l, [np.array(ARM_L, np.int32)], 1)
        arm_r = np.zeros((h, w), np.uint8)
        cv2.fillPoly(arm_r, [np.array([(MIRROR_X - x, y) for x, y in ARM_L], np.int32)], 1)
        trouser = ((np.abs(bgr[..., 2].astype(int) - bgr[..., 0].astype(int)) < 16) &
                   (bgr.max(axis=2) < 125) & (np.arange(h)[:, None] > 505)).astype(np.uint8)
        arm_l = arm_l & m & (1 - trouser)
        arm_r = arm_r & m & (1 - trouser)
        yy = np.arange(h)[:, None]
        upper = (yy < 412).astype(np.uint8)
        lower = 1 - upper
        # forearm gets a round cap around the elbow so the bend reads as a joint
        cap = {}
        for side, (ex, ey) in ELBOW.items():
            c = np.zeros((h, w), np.uint8)
            cv2.circle(c, (int(ex), int(ey)), 40, 1, -1)
            cap[side] = c
        pieces = {
            "L_up": (arm_l & upper) | (arm_l & cap["L"]), "R_up": (arm_r & upper) | (arm_r & cap["R"]),
            "L_fore": (arm_l & lower) | (arm_l & cap["L"]), "R_fore": (arm_r & lower) | (arm_r & cap["R"]),
        }
        arms = arm_l | arm_r
        # head: everything above the chin that is not the navy collar
        navy = (bgr[..., 0].astype(int) > bgr[..., 2].astype(int) + 8) & (bgr.sum(axis=2) < 330)
        head_m = np.zeros((h, w), np.uint8)
        head_m[20:206, 430:600] = 1
        head_m &= m
        collar = np.zeros((h, w), bool)
        collar[:206, :] = navy[:206, :]
        collar[:176, :] = False
        head_m[collar] = 0
        head_m = cv2.morphologyEx(head_m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # torso: the body with the arms taken out, shoulders filled in
        torso_m = m & (1 - arms) & (1 - head_m)
        torso_m[:180, :] = 0
        torso_m = self._add_shoulders(torso_m)
        torso_m[:180, :] = 0
        torso_m[330:, :418] = 0
        torso_m[330:, 606:] = 0
        torso_m[540:, :] = 0
        self.sprites = {"torso": self._cut(bgr, torso_m, (330, 150, 700, 560), fill_navy=True),
                        "head": self._cut(bgr, head_m, (430, 20, 600, 210))}
        for k, pm in pieces.items():
            self.sprites[k] = self._cut(bgr, pm, (340, 225, 700, 625))
        self.head_clean = self.sprites["head"]
        self.hair_mask_rows = None

    # ------------------------------------------------------------------ cutting
    @staticmethod
    def _sample_skin(bgr):
        p = bgr[150:160, 500:520].reshape(-1, 3)
        return tuple(int(v) for v in np.median(p, axis=0))       # BGR

    def _add_shoulders(self, torso_m):
        h, w = torso_m.shape
        extra = np.zeros((h, w), np.uint8)
        for side in ("L", "R"):
            sx = 1 if side == "L" else -1
            pts = [(480, 205), (398, 238), (430, 262), (432, 330), (512, 330)]
            if side == "R":
                pts = [(MIRROR_X - x, y) for x, y in pts]
            cv2.fillPoly(extra, [np.array(pts, np.int32)], 1)
        return torso_m | (extra & self.mask)

    def _cut(self, bgr, mask, box, fill_navy=False) -> Sprite:
        x0, y0, x1, y1 = box
        mk = mask[y0:y1, x0:x1]
        if mk.sum() < 10:
            raise ValueError("empty piece")
        region = cv2.dilate(mk, np.ones((3, 3), np.uint8), iterations=2).astype(np.float32)
        a = np.clip(self.a_soft[y0:y1, x0:x1] * region, 0, 1)
        a = np.maximum(a, mk.astype(np.float32) * 0.0 + (mk.astype(np.float32)))   # interior fully opaque
        crop = self.fg[y0:y1, x0:x1].astype(np.uint8)
        crop = cv2.resize(crop, None, fx=SC, fy=SC, interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(crop, (0, 0), 1.6)
        crop = cv2.addWeighted(crop, 1.35, blur, -0.35, 0)
        a = cv2.resize(a, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
        a = np.clip(cv2.GaussianBlur(a, (0, 0), 0.7), 0, 1)
        rgb = crop.astype(np.float32)[..., ::-1] / 255.0
        # soft edge pixels take the outline colour instead of the purple backdrop
        wgt = np.clip((a - 0.35) / 0.55, 0, 1)[..., None]
        rgb = rgb * wgt + np.array([0.11, 0.10, 0.15], np.float32) * (1 - wgt)
        pm = np.dstack([rgb * a[..., None], a])
        return Sprite(pm.astype(np.float32), x0, y0)

    # --------------------------------------------------------------- face paint
    def _paint_face(self, p: Pose) -> np.ndarray:
        """Return the head sprite pixels (premultiplied RGBA float32) with the mouth
        and eyelids drawn for this pose."""
        sp = self.sprites["head"]
        img = sp.img.copy()
        need_mouth = p.mouth_open > 0.06
        need_blink = p.blink > 0.15
        if not (need_mouth or need_blink):
            return img
        S = 3
        H, W = img.shape[:2]
        layer = Image.new("RGBA", (W * S // 1, H * S // 1), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        skin = (self.skin[2], self.skin[1], self.skin[0], 255)

        def px(x, y):                                       # source -> supersampled sprite px
            return ((x - sp.x0) * SC * S, (y - sp.y0) * SC * S)

        line = (52, 30, 34, 255)
        if need_blink:
            for ex, ey in EYES:
                x0, y0 = px(ex - 11, ey - 7.5)
                x1, y1 = px(ex + 11, ey + 8.5)
                d.ellipse([x0, y0, x1, y1], fill=skin)
                cx, cy = px(ex, ey + 1.5)
                wdt = 10.5 * SC * S
                d.arc([cx - wdt, cy - 3.5 * SC * S, cx + wdt, cy + 3.5 * SC * S], 20, 160, fill=line,
                      width=int(1.5 * SC * S))
        if need_mouth:
            mx, my = MOUTH
            x0, y0 = px(mx - 13, my - 4.5)
            x1, y1 = px(mx + 13, my + 12)
            d.ellipse([x0, y0, x1, y1], fill=skin)                                     # wipes the closed mouth
            op = min(1.0, p.mouth_open)
            half = (6.2 + 5.0 * p.mouth_w) * (1 - 0.2 * op)
            hgt = 1.6 + 11.5 * op
            smile = 0.8 + 1.8 * p.smile
            cx, cy = px(mx, my - 1)
            hw = half * SC * S
            # outline + dark interior
            top = [(cx + t * hw, cy - (smile * (1 - t * t)) * SC * S * 0.45 + (t * t) * 1.2 * SC * S) for t in np.linspace(-1, 1, 24)]
            bot = [(cx + t * hw, cy + (hgt * (1 - t * t) ** 0.7) * SC * S + (t * t) * 0.6 * SC * S) for t in np.linspace(1, -1, 24)]
            d.polygon(top + bot, fill=(110, 36, 46, 255), outline=line, width=int(1.1 * SC * S))
            if op > 0.28:
                d.polygon([(cx + t * hw * 0.8, cy + 0.2 * SC * S + abs(t) * 0.5) for t in np.linspace(-1, 1, 12)] +
                          [(cx + t * hw * 0.8, cy + min(hgt * 0.34, 2.8) * SC * S) for t in np.linspace(1, -1, 12)],
                          fill=(246, 244, 240, 255))
            if op > 0.5:
                d.ellipse([cx - hw * 0.5, cy + hgt * 0.55 * SC * S, cx + hw * 0.5, cy + hgt * 1.02 * SC * S], fill=(212, 96, 106, 255))
        layer = layer.resize((W, H), Image.LANCZOS)
        lay = np.asarray(layer).astype(np.float32) / 255.0
        a = lay[..., 3:4]
        rgb = lay[..., :3] * a                              # premultiplied
        img = img * (1 - a) + np.dstack([rgb, a])
        return img

    # -------------------------------------------------------------- composition
    def draw(self, frame: np.ndarray, p: Pose, anchor_frame=(540.0, 1920.0), anchor_src=(512.0, 420.0),
             zoom: float = 1.0, shear: float = 0.0, alpha: float = 1.0) -> None:
        """Composite the character onto `frame` (uint8 BGR H x W x 3)."""
        k = SC * zoom
        # source -> frame
        base = (_mat_t(anchor_frame[0], anchor_frame[1]) @ _mat_s(k, k) @ _mat_t(-anchor_src[0], -anchor_src[1]))
        body = (_mat_t(0, p.bob) @ _mat_rot(p.lean, 512, 560) @ _mat_s(1 + p.breath, 1 + p.breath, 512, 420))
        B = base @ body
        sh = p.shrug

        def sh_off(side):
            return _mat_t(0, -sh)

        # head
        head_m = (_mat_t(p.head_dx, p.head_dy - sh * 0.6) @ _mat_rot(p.head_rot, *NECK))
        pieces = []
        pieces.append(("torso", self.sprites["torso"].img, self.sprites["torso"].x0, self.sprites["torso"].y0, B))
        hs = self._paint_face(p)
        hair_img = hs
        pieces.append(("head", hair_img, self.sprites["head"].x0, self.sprites["head"].y0, B @ head_m))
        for side, sg in (("L", 1.0), ("R", -1.0)):
            sho, elb = SHOULDER[side], ELBOW[side]
            a_s = getattr(p, f"{side.lower()}_sh") * sg
            a_e = getattr(p, f"{side.lower()}_el") * sg
            up = _mat_t(0, -sh) @ _mat_rot(a_s, *sho)
            # elbow position after the shoulder rotation, then bend the forearm about it
            fore = up @ _mat_rot(a_e, *elb)
            pieces.append((f"{side}_fore", self.sprites[f"{side}_fore"].img, self.sprites[f"{side}_fore"].x0,
                           self.sprites[f"{side}_fore"].y0, B @ fore))
            pieces.append((f"{side}_up", self.sprites[f"{side}_up"].img, self.sprites[f"{side}_up"].x0,
                           self.sprites[f"{side}_up"].y0, B @ up))
        for name, img, x0, y0, M in pieces:
            _blend(frame, img, x0, y0, M, k, alpha)

    def head_position(self, p: Pose, anchor_frame, anchor_src, zoom) -> tuple[float, float]:
        k = SC * zoom
        return (anchor_frame[0] + (512 - anchor_src[0]) * k, anchor_frame[1] + (110 - anchor_src[1]) * k)


def _blend(frame, img, x0, y0, M_src, k, alpha):
    """Warp a pre-scaled premultiplied sprite with the source-space matrix M_src and blend."""
    h, w = img.shape[:2]
    # sprite px -> source coords -> frame px.  Sprite px (u,v) = (src - (x0,y0)) * SC
    to_src = _mat_t(x0, y0) @ _mat_s(1 / SC, 1 / SC)
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
    warped = cv2.warpAffine(img, M[:2], (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4] * alpha
    roi = frame[cy0:cy1, cx0:cx1].astype(np.float32)
    rgb_bgr = warped[..., :3][..., ::-1] * 255.0 * alpha
    frame[cy0:cy1, cx0:cx1] = np.clip(roi * (1 - a) + rgb_bgr, 0, 255).astype(np.uint8)
