"""Human-like body language for Shubh.

No looping animation: every joint is a damped spring chasing a target that is built from
what is being said (gesture per sentence, a beat on each stressed word), plus slow
non-repeating idle motion (breathing, weight shift, head drift, gaze turns, blinks).
Hands lag the forearms and overshoot a little, like a real wrist.
"""
from __future__ import annotations

import math

import numpy as np

from .rig import Pose

# arm poses. sh: shoulder out (deg), el: elbow (negative = hand comes in and up), wr: wrist,
# fore/up: foreshortening (limb pointing toward the camera). Sides are screen left (l) and right (r).
REST = dict(sh=2.0, el=-5.0, wr=0.0, fore=0.0, up=0.0)
RELAX = dict(sh=5.0, el=-12.0, wr=3.0, fore=0.05, up=0.05)


def arm(sh, el, wr=0.0, fore=0.0, up=0.0):
    return dict(sh=sh, el=el, wr=wr, fore=fore, up=up)


GESTURES = {
    "rest":      dict(l=REST, r=REST, head=0, yaw=0.0, lean=0, shrug=0),
    "wave":      dict(l=RELAX, r=arm(28, 150, 0, 0.3, 0.85), head=-3, yaw=0.15, lean=1, shrug=1, wave="r"),
    "point_up":  dict(l=RELAX, r=arm(18, 162, 0, 0.3, 0.7), head=2, yaw=0.1, lean=1, shrug=1),
    "present_r": dict(l=RELAX, r=arm(24, -100, -10, 0.5, 0.8), head=3, yaw=0.15, lean=2, shrug=0),
    "present_l": dict(l=arm(24, -100, 10, 0.5, 0.8), r=RELAX, head=-3, yaw=-0.15, lean=-2, shrug=0),
    "count":     dict(l=RELAX, r=arm(26, -114, -8, 0.5, 0.8), head=-2, yaw=0.1, lean=1, shrug=0, beat="r"),
    "shrug":     dict(l=arm(16, 58, 16, 0.75, 0.6), r=arm(16, 58, -16, 0.75, 0.6), head=5, yaw=0.0, lean=0, shrug=8),
    "think":     dict(l=RELAX, r=arm(14, -165, -12, 0.7, 1.0), head=-5, yaw=0.45, lean=-1, shrug=0),
    "both_up":   dict(l=arm(28, -118, 8, 0.5, 0.8), r=arm(28, -118, -8, 0.5, 0.8), head=0, yaw=0.0, lean=0, shrug=2),
    "cheer":     dict(l=arm(45, 135, 8, 0.3, 1.0), r=arm(45, 135, -8, 0.3, 1.0), head=0, yaw=0.0, lean=0, shrug=3, beat="both"),
}


class Spring:
    def __init__(self, x=0.0, omega=12.0, zeta=0.85):
        self.x, self.v, self.w, self.z = x, 0.0, omega, zeta

    def step(self, target, dt):
        n = 4
        h = dt / n
        for _ in range(n):
            a = self.w * self.w * (target - self.x) - 2 * self.z * self.w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x


class Noise:
    """Smooth, non-repeating wander in about [-1, 1]."""

    def __init__(self, seed, base_hz=0.11):
        rng = np.random.default_rng(seed)
        self.f = base_hz * rng.uniform(0.7, 1.9, 4) * np.array([1, 1.7, 2.9, 4.3])
        self.p = rng.uniform(0, 2 * math.pi, 4)
        self.a = np.array([1.0, 0.55, 0.3, 0.15])
        self.a /= self.a.sum()

    def __call__(self, t):
        return float(np.sum(self.a * np.sin(2 * math.pi * self.f * t + self.p)))


class Body:
    """Turns (time, speech, gesture plan) into a Pose each frame."""

    ARM_KEYS = ("sh", "el", "wr", "fore", "up")

    def __init__(self, events, seed=7, hold=1.9):
        self.ev, self.idx, self.hold = events, 0, hold
        self.rng = np.random.default_rng(seed)
        self.sp = {}
        for side in ("l", "r"):
            self.sp[side] = dict(sh=Spring(2, 11, 0.9), el=Spring(-5, 10, 0.82), wr=Spring(0, 17, 0.42),
                                 fore=Spring(0, 9, 1.0), up=Spring(0, 9, 1.0))
        self.sp_body = dict(head=Spring(0, 9, 0.85), yaw=Spring(0, 5, 0.9), lean=Spring(0, 5, 0.9), shrug=Spring(0, 14, 0.8),
                            nod=Spring(0, 16, 0.6), nodx=Spring(0, 12, 0.7))
        self.n = {k: Noise(i + seed * 13) for i, k in enumerate(("hr", "hx", "hy", "ln", "sw", "yw", "ls", "rs", "lw", "rw"))}
        self.scale = 1.0
        self._last_idx = -1
        self.beat_t = -9.0

    def _arm_targets(self, g, side, t, t_ev, imp):
        base = dict(g[side])
        s = self.scale
        tgt = {k: base[k] for k in self.ARM_KEYS}
        if base["sh"] > 8:                      # only an active arm gets the amplitude variation
            tgt["sh"] = base["sh"] * s
            tgt["el"] = base["el"] * (0.94 + 0.06 * s)
        age = t - t_ev
        # wave: the wrist swings, the forearm follows a little
        if g.get("wave") == side and age > 0.35:
            ph = math.sin(2 * math.pi * 2.3 * (age - 0.35))
            tgt["wr"] += 16 * ph * (1 if side == "l" else -1)
            tgt["el"] += 5 * ph
        if g.get("beat") in (side, "both"):
            tgt["el"] += -6 * math.sin(2 * math.pi * 1.6 * age) * 0.0
        # a stressed word taps the active hand
        if base["sh"] > 8:
            tgt["el"] += -7 * imp
            tgt["wr"] += 6 * imp * (1 if side == "l" else -1)
            tgt["sh"] += 2.5 * imp
        else:
            # idle drift for the resting arm
            tgt["sh"] += 1.6 * self.n["ls" if side == "l" else "rs"](t)
            tgt["el"] += 3.0 * self.n["lw" if side == "l" else "rw"](t)
        return tgt

    def step(self, t, dt, env, cen, imp, blink, sec_kind):
        while self.idx + 1 < len(self.ev) and self.ev[self.idx + 1][0] <= t:
            self.idx += 1
        if self.idx != self._last_idx:                     # new gesture: vary the size a little
            self._last_idx = self.idx
            self.scale = float(self.rng.uniform(0.88, 1.12))
        t_ev, name = self.ev[self.idx]
        g = GESTURES[name]
        age = t - t_ev
        if name != "rest" and age > self.hold:             # nobody holds a gesture forever
            g = dict(GESTURES["rest"], l=RELAX if name != "present_l" else GESTURES["rest"]["l"], r=RELAX)
        # anticipation: the arm dips a hair before the stroke
        lift = 1.0
        if name != "rest" and age < 0.12:
            lift = 0.55 + 0.45 * (age / 0.12)
        out = {}
        for side in ("l", "r"):
            tgt = self._arm_targets(g, side, t, t_ev, imp)
            res = {}
            for k in self.ARM_KEYS:
                v = tgt[k]
                if k in ("sh", "el") and abs(g[side]["sh"]) > 8 and lift < 1.0:
                    v = REST[k] + (v - REST[k]) * lift
                res[k] = self.sp[side][k].step(v, dt)
            out[side] = res
        sp = self.sp_body
        head_t = g["head"] + 1.6 * self.n["hr"](t) - 1.2 * imp
        yaw_t = g["yaw"] + 0.35 * self.n["yw"](t)
        lean_t = g["lean"] + 0.9 * self.n["ln"](t) + 0.9 * imp
        head = sp["head"].step(head_t, dt)
        yaw = sp["yaw"].step(yaw_t, dt)
        lean = sp["lean"].step(lean_t, dt)
        shrug = sp["shrug"].step(g["shrug"] + 1.2 * imp, dt)
        nod = sp["nod"].step(2.4 * imp + 1.0 * env, dt)        # head dips with stress and loudness
        nodx = sp["nodx"].step(0.8 * self.n["hx"](t), dt)
        smile = 0.55 if sec_kind in ("intro", "hook", "cta", "payoff") else 0.1
        return Pose(
            l_sh=out["l"]["sh"], l_el=out["l"]["el"], l_wr=out["l"]["wr"], l_fore=max(0, out["l"]["fore"]), l_up=max(0, out["l"]["up"]),
            r_sh=out["r"]["sh"], r_el=out["r"]["el"], r_wr=out["r"]["wr"], r_fore=max(0, out["r"]["fore"]), r_up=max(0, out["r"]["up"]),
            head_rot=head, head_dx=nodx + 1.2 * self.n["hx"](t + 3.1), head_dy=nod + 0.8 * self.n["hy"](t), yaw=float(np.clip(yaw, -1, 1)),
            lean=lean, bob=1.0 * math.sin(2 * math.pi * t / 3.6) * 0.6 + 0.5 * self.n["sw"](t + 1.7),
            sway=3.0 * self.n["sw"](t), breath=0.0045 * math.sin(2 * math.pi * t / 3.9) + 0.0015 * imp,
            shrug=shrug + 0.6 * math.sin(2 * math.pi * t / 3.9),
            mouth_open=0.0 if env < 0.1 else float(np.clip(env ** 0.8 * 1.2, 0, 1)), mouth_w=float(cen),
            blink=blink, smile=smile, hair=float(-(sp["head"].v) * 0.06 + 0.8 * self.n["hx"](t)),
        )
