"""Brings one still illustration to life: camera moves, breathing, head sway,
hair flutter, blinking, lamp glow, dust and film grain. CPU only (OpenCV).

All coordinates below are in pixels of the 736x981 source art.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

OUT_W, OUT_H = 1080, 1920
MASTER_K = 2.6                      # master upscale (Lanczos) used for camera crops
GRID = 8                            # displacement field is evaluated every 8 output px

# camera shots: (centre x, centre y, zoom, drift zoom per second, pan x/s, pan y/s)
SHOTS = {
    "wide":       (440, 490, 1.00, 0.012, -1.5, 0.0),
    "face":       (452, 330, 1.50, 0.018, 0.0, 2.0),
    "face_tight": (455, 285, 1.70, 0.015, 1.5, 1.0),
    "hand":       (505, 400, 1.75, 0.012, -2.0, 0.0),
    "tie":        (430, 400, 1.30, 0.014, 1.0, 2.0),
}

EYES = ((387, 210), (482, 235))     # left, right eye centres in the source art
LAMP = (680, 290)
NECK = (410, 340)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class Motion:
    nod_px: float = 0.0      # head bob (source px), driven by the voice
    tilt: float = 0.0        # head tilt in radians
    blink: float = 0.0       # 0 open .. 1 closed
    glow: float = 1.0        # lamp intensity multiplier
    flash: float = 0.0       # white flash amount for cuts


class Stage:
    def __init__(self, image_path: str, seed: int = 7):
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        self.src_h, self.src_w = bgr.shape[:2]
        self.fit = OUT_H / self.src_h
        master = cv2.resize(bgr, None, fx=MASTER_K, fy=MASTER_K, interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(master, (0, 0), 2.2)
        master = cv2.addWeighted(master, 1.55, blur, -0.55, 0)           # unsharp mask
        master = self._grade(master)
        self.master = master
        rng = np.random.default_rng(seed)

        # coarse output grid for the displacement field
        self.gx = np.arange(0, OUT_W + GRID, GRID, dtype=np.float32)
        self.gy = np.arange(0, OUT_H + GRID, GRID, dtype=np.float32)
        self.X = np.arange(OUT_W, dtype=np.float32)[None, :]
        self.Y = np.arange(OUT_H, dtype=np.float32)[:, None]

        # vignette + grain tiles
        yy, xx = np.mgrid[0:OUT_H // 4, 0:OUT_W // 4].astype(np.float32)
        r = np.sqrt(((xx / (OUT_W / 4) - 0.5) * 1.15) ** 2 + ((yy / (OUT_H / 4) - 0.5) * 0.9) ** 2)
        vig = 1.0 - 0.55 * np.clip((r - 0.28) / 0.55, 0, 1) ** 1.6
        self.vignette = cv2.resize(vig, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)[..., None]
        self.grain = [cv2.resize(rng.normal(0, 1, (OUT_H // 2, OUT_W // 2)).astype(np.float32),
                                 (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR) for _ in range(6)]

        # lamp glow sprite
        s = 256
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        d = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / (s / 2)
        self.glow = np.clip(1 - d, 0, 1) ** 2.2

        # dust motes
        n = 46
        self.dust = dict(
            x=rng.uniform(0, OUT_W, n), y=rng.uniform(0, OUT_H, n),
            vx=rng.uniform(-6, 6, n), vy=rng.uniform(-22, -6, n),
            r=rng.uniform(1.2, 3.4, n), ph=rng.uniform(0, 6.28, n),
            a=rng.uniform(0.25, 0.7, n))

    # ------------------------------------------------------------------ grade
    @staticmethod
    def _grade(img):
        f = img.astype(np.float32) / 255.0
        f = np.clip((f - 0.5) * 1.07 + 0.5, 0, 1)                      # a touch more contrast
        hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.06, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        out[..., 0] += 3.0                                              # whisper of cool in the shadows
        return np.clip(out, 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------- camera
    def camera(self, shot: str, u: float, punch: float = 0.0, extra_zoom: float = 1.0):
        cx, cy, z0, dz, px, py = SHOTS[shot]
        z = z0 * (1 + dz * u) * (1 + punch) * extra_zoom
        cx += px * u
        cy += py * u
        vw, vh = OUT_W / (self.fit * z), OUT_H / (self.fit * z)
        cx = float(np.clip(cx, vw / 2, self.src_w - vw / 2))
        cy = float(np.clip(cy, vh / 2, self.src_h - vh / 2))
        return cx, cy, z

    # ----------------------------------------------------------- deformations
    def _displacement(self, sx, sy, t, m: Motion):
        """Source-pixel displacement for every point of the coarse grid."""
        dx = np.zeros_like(sx)
        dy = np.zeros_like(sx)
        # breathing: the torso swells and settles
        w_chest = _smoothstep(260, 430, sy)
        b = 0.0045 * math.sin(2 * math.pi * t / 3.8) + 0.0012 * math.sin(2 * math.pi * t / 1.3)
        dx += (sx - 330) * b * w_chest
        dy += (sy - 700) * b * w_chest
        # head + hand move together: slow sway, tilt and voice-driven nod
        wh = np.exp(-(((sx - 445) / 165) ** 2 + ((sy - 220) / 150) ** 2))
        ang = 0.010 * math.sin(2 * math.pi * t / 5.1) + 0.006 * math.sin(2 * math.pi * t / 2.3 + 1.0) + m.tilt
        rx, ry = sx - NECK[0], sy - NECK[1]
        dx += wh * (-ang * ry)
        dy += wh * (ang * rx + m.nod_px + 1.6 * math.sin(2 * math.pi * t / 4.4))
        # hair flutter, strongest at the top of the head
        wf = wh * np.exp(-(((sy - 60) / 85) ** 2))
        dx += wf * 1.7 * np.sin(0.055 * sx + 0.045 * sy + 2 * math.pi * t / 1.8)
        dy += wf * 0.9 * np.sin(0.045 * sx - 0.06 * sy + 2 * math.pi * t / 2.4)
        # blink: squash each eye vertically toward its lid line
        if m.blink > 0.01:
            for ex, ey in EYES:
                we = np.exp(-(((sx - ex) / 36) ** 2 + ((sy - (ey + 4)) / 17) ** 2))
                dy += we * (sy - (ey - 3)) * (0.55 * m.blink)
        return dx, dy

    # ------------------------------------------------------------------ frame
    def render(self, t: float, shot: str, u: float, punch: float, motion: Motion,
               frame_idx: int = 0, extra_zoom: float = 1.0) -> np.ndarray:
        cx, cy, z = self.camera(shot, u, punch, extra_zoom)
        k = self.fit * z
        # coarse grid -> source coords -> displacement
        gsx = cx + (self.gx[None, :] - OUT_W / 2) / k
        gsy = cy + (self.gy[:, None] - OUT_H / 2) / k
        gsx = np.broadcast_to(gsx, (len(self.gy), len(self.gx))).astype(np.float32)
        gsy = np.broadcast_to(gsy, (len(self.gy), len(self.gx))).astype(np.float32)
        ddx, ddy = self._displacement(gsx, gsy, t, motion)
        ddx = cv2.resize(ddx, (len(self.gx) * GRID, len(self.gy) * GRID), interpolation=cv2.INTER_LINEAR)[:OUT_H, :OUT_W]
        ddy = cv2.resize(ddy, (len(self.gx) * GRID, len(self.gy) * GRID), interpolation=cv2.INTER_LINEAR)[:OUT_H, :OUT_W]
        map_x = ((cx + (self.X - OUT_W / 2) / k) + ddx) * MASTER_K
        map_y = ((cy + (self.Y - OUT_H / 2) / k) + ddy) * MASTER_K
        frame = cv2.remap(self.master, map_x.astype(np.float32), map_y.astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame = frame.astype(np.float32)

        # lamp glow follows the camera
        lx = OUT_W / 2 + (LAMP[0] - cx) * k
        ly = OUT_H / 2 + (LAMP[1] - cy) * k
        flick = motion.glow * (0.92 + 0.08 * math.sin(t * 7.3) * math.sin(t * 2.9))
        self._add_glow(frame, lx, ly, 520 * z, (70, 150, 255), 0.55 * flick)   # BGR warm amber
        self._add_dust(frame, t)

        frame *= self.vignette
        frame += self.grain[frame_idx % len(self.grain)][..., None] * 4.5
        if motion.flash > 0.01:
            frame = frame * (1 - motion.flash) + 255.0 * motion.flash
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _add_glow(self, frame, cx, cy, radius, bgr, strength):
        r = int(radius)
        if r < 8:
            return
        sprite = cv2.resize(self.glow, (2 * r, 2 * r), interpolation=cv2.INTER_LINEAR)
        x0, y0 = int(cx - r), int(cy - r)
        x1, y1 = x0 + 2 * r, y0 + 2 * r
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(OUT_W, x1), min(OUT_H, y1)
        if x1c <= x0c or y1c <= y0c:
            return
        sp = sprite[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)][..., None]
        frame[y0c:y1c, x0c:x1c] += sp * np.array(bgr, np.float32)[None, None, :] * strength

    def _add_dust(self, frame, t):
        d = self.dust
        layer = np.zeros((OUT_H // 4, OUT_W // 4), np.float32)
        for i in range(len(d["x"])):
            x = (d["x"][i] + d["vx"][i] * t + 14 * math.sin(t * 0.5 + d["ph"][i])) % OUT_W
            y = (d["y"][i] + d["vy"][i] * t) % OUT_H
            a = d["a"][i] * (0.55 + 0.45 * math.sin(t * 1.7 + d["ph"][i]))
            cv2.circle(layer, (int(x / 4), int(y / 4)), max(1, int(d["r"][i] / 2)), float(a), -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.2)
        layer = cv2.resize(layer, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        frame += layer[..., None] * np.array([150, 205, 255], np.float32)[None, None, :] * 0.55


if __name__ == "__main__":
    import sys
    st = Stage(sys.argv[1] if len(sys.argv) > 1 else "assets/character_clean.png")
    import time
    t0 = time.time()
    for i in range(10):
        f = st.render(i / 30, "face", i / 30, 0.0, Motion(), i)
    print("per frame", (time.time() - t0) / 10)
    cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "stage_test.png", f)
