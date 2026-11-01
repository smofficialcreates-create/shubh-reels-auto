"""Human-like body language for Shubh (animation presets, speed control, narration lip-sync).

No looping animation: every joint is a damped spring chasing a target that is built from
what is being said (gesture per sentence, a beat on each stressed word), plus slow
non-repeating idle motion (breathing, weight shift, head drift, gaze turns, blinks).
Hands lag the forearms and overshoot a little, like a real wrist.
"""
from __future__ import annotations

import math

import numpy as np

from .puppet import Pose

# arm poses. sh: shoulder out (deg), el: elbow (negative = hand comes in and up), wr: wrist,
# fore/up: foreshortening (limb pointing toward the camera). Sides are screen left (l) and right (r).
REST = dict(sh=2.0, el=-5.0, wr=0.0, fore=0.0, up=0.0)
RELAX = dict(sh=5.0, el=-12.0, wr=3.0, fore=0.05, up=0.05)


def arm(sh, el, wr=0.0, fore=0.0, up=0.0):
    return dict(sh=sh, el=el, wr=wr, fore=fore, up=up)


GESTURES = {
    "look_l":    dict(l=REST, r=REST, head=-3, yaw=-0.85, pitch=0.05, lean=-1, shrug=0),
    "look_r":    dict(l=REST, r=REST, head=3, yaw=0.85, pitch=0.05, lean=1, shrug=0),
    "look_up":   dict(l=RELAX, r=RELAX, head=-2, yaw=0.3, pitch=-0.9, lean=0, shrug=0, brow=0.8),
    "lean_in":   dict(l=RELAX, r=RELAX, head=1, yaw=0.0, pitch=0.2, lean=4, shrug=0, brow=0.4),
    "rest":      dict(l=REST, r=REST, head=0, yaw=0.0, lean=0, shrug=0),
    "wave":      dict(l=RELAX, r=arm(28, 150, 0, 0.3, 0.85), head=-3, yaw=0.15, lean=1, shrug=1, wave="r"),
    "point_up":  dict(l=RELAX, r=arm(18, 162, 0, 0.3, 0.7), head=2, yaw=0.1, lean=1, shrug=1),
    "present_r": dict(l=RELAX, r=arm(24, -100, -10, 0.5, 0.8), head=3, yaw=0.15, lean=2, shrug=0),
    "present_l": dict(l=arm(24, -100, 10, 0.5, 0.8), r=RELAX, head=-3, yaw=-0.15, lean=-2, shrug=0),
    "count":     dict(l=RELAX, r=arm(26, -114, -8, 0.5, 0.8), head=-2, yaw=0.1, lean=1, shrug=0, beat="r"),
    "shrug":     dict(l=arm(13, 38, 14, 0.75, 0.6), r=arm(13, 38, -14, 0.75, 0.6), head=5, yaw=0.0, lean=0, shrug=8, brow=0.7),
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

    def __init__(self, events, seed=7, hold=1.9, speed=1.0):
        self.ev, self.idx, self.hold, self.speed = events, 0, hold, float(speed)
        self.rng = np.random.default_rng(seed)
        self.sp = {}
        for side in ("l", "r"):
            self.sp[side] = dict(sh=Spring(2, 11, 0.9), el=Spring(-5, 10, 0.82), wr=Spring(0, 17, 0.42),
                                 fore=Spring(0, 9, 1.0), up=Spring(0, 9, 1.0))
        self.sp_body = dict(head=Spring(0, 9, 0.85), yaw=Spring(0, 5, 0.9), lean=Spring(0, 5, 0.9), shrug=Spring(0, 14, 0.8),
                            nod=Spring(0, 16, 0.6), nodx=Spring(0, 12, 0.7), pitch=Spring(0, 6, 0.9), brow=Spring(0, 14, 0.7))
        self.n = {k: Noise(i + seed * 13) for i, k in enumerate(("hr", "hx", "hy", "ln", "sw", "yw", "ls", "rs", "lw", "rw"))}
        self.scale = 1.0
        self._last_idx = -1
        self.beat_t = -9.0

    def _arm_targets(self, g, side, t, t_ev, imp):
        tn = t * self.speed
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
            tgt["sh"] += 1.6 * self.n["ls" if side == "l" else "rs"](tn)
            tgt["el"] += 3.0 * self.n["lw" if side == "l" else "rw"](tn)
        return tgt

    def step(self, t, dt, env, cen, imp, blink, sec_kind):
        dt = dt * self.speed                               # animation speed scales the dynamics
        tn = t * self.speed
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
        head_t = g["head"] + 1.6 * self.n["hr"](tn) - 1.2 * imp
        yaw_t = g["yaw"] + 0.35 * self.n["yw"](tn)
        lean_t = g["lean"] + 0.9 * self.n["ln"](tn) + 0.9 * imp
        head = sp["head"].step(head_t, dt)
        yaw = sp["yaw"].step(yaw_t, dt)
        lean = sp["lean"].step(lean_t, dt)
        shrug = sp["shrug"].step(g["shrug"] + 1.2 * imp, dt)
        nod = sp["nod"].step(2.4 * imp + 1.0 * env, dt)        # head dips with stress and loudness
        nodx = sp["nodx"].step(0.8 * self.n["hx"](tn), dt)
        pitch = sp["pitch"].step(g.get("pitch", 0.0) + 0.12 * self.n["hy"](tn + 5.0), dt)
        brow = sp["brow"].step(g.get("brow", 0.0) + 0.5 * imp + 0.12 * self.n["hy"](tn + 9.0), dt)
        smile = 0.55 if sec_kind in ("intro", "hook", "cta", "payoff") else 0.1
        return Pose(
            l_sh=out["l"]["sh"], l_el=out["l"]["el"], l_wr=out["l"]["wr"], 
            r_sh=out["r"]["sh"], r_el=out["r"]["el"], r_wr=out["r"]["wr"], 
            head_rot=head, head_dx=nodx + 1.2 * self.n["hx"](tn + 3.1), head_dy=nod + 0.8 * self.n["hy"](tn), yaw=float(np.clip(yaw, -1, 1)), pitch=float(np.clip(pitch, -1, 1)),
            lean=lean, bob=1.0 * math.sin(2 * math.pi * tn / 3.6) * 0.6 + 0.5 * self.n["sw"](tn + 1.7),
            sway=3.0 * self.n["sw"](tn), breath=0.0045 * math.sin(2 * math.pi * tn / 3.9) + 0.0015 * imp,
            shrug=shrug + 0.6 * math.sin(2 * math.pi * tn / 3.9),
            mouth_open=0.0 if env < 0.1 else float(np.clip(env ** 0.8 * 1.2, 0, 1)), mouth_w=float(cen),
            blink=blink, brow=float(np.clip(brow, -1, 1)), smile=smile, hair=float(-(sp["head"].v) * 0.06 + 0.8 * self.n["hx"](tn)),
        )


# ---------------------------------------------------------------------------------------
PRESETS = sorted(GESTURES)


def blink_schedule(duration: float, seed: int = 21):
    rng = np.random.default_rng(seed)
    t, out = float(rng.uniform(0.8, 1.5)), []
    while t < duration:
        out.append(t)
        if rng.random() < 0.2:
            out.append(t + 0.26)                       # the occasional double blink
        t += float(rng.uniform(2.0, 4.4))
    return out


def blink_value(t: float, times, dur: float = 0.16) -> float:
    for b in times:
        if b <= t < b + dur:
            return math.sin(math.pi * (t - b) / dur) ** 1.3
    return 0.0


def lip_sync(voice: np.ndarray, sr: int, fps: int, n_frames: int):
    """Narration -> per-frame (mouth_open 0..1, mouth_w 0..1, loudness 0..1)."""
    from . import audio as A
    v = A.resample(voice, sr) if sr != A.SR else voice.astype(np.float32)
    env, cen = A.voice_features(v, fps, n_frames)
    mo = np.where(env < 0.1, 0.0, np.clip(env ** 0.8 * 1.25, 0, 1)).astype(np.float32)
    return mo, cen.astype(np.float32), env


class Animator:
    """Reusable animation driver.

        a = Animator(speed=1.0)
        a.play("wave", at=0.5); a.play("present_r", at=3.0); a.play("rest", at=6.0)
        pose = a.pose(t, dt=1/30, mouth_open=.., mouth_w=.., blink=..)   # call once per frame in order

    speed < 1 is calmer, speed > 1 snappier (springs and idle drift scale together).
    """

    def __init__(self, speed: float = 1.0, seed: int = 7, hold: float = 1.9):
        self.speed, self.seed, self.hold = speed, seed, hold
        self.events = [(0.0, "rest")]
        self._body = None

    def play(self, name: str, at: float):
        if name not in GESTURES:
            raise KeyError(f"unknown preset {name!r}; available: {', '.join(PRESETS)}")
        self.events.append((float(at), name))
        self.events.sort()
        self._body = None
        return self

    def load_events(self, events):
        self.events = [(0.0, "rest")] + [(float(t), n) for t, n in events if t > 0 or n != "rest"]
        self.events.sort()
        self._body = None
        return self

    def pose(self, t: float, dt: float = 1 / 30, mouth_open: float = 0.0, mouth_w: float = 0.5,
             blink: float = 0.0, loud: float = 0.0, stress: float = 0.0, mood: str = "hook") -> Pose:
        if self._body is None:
            self._body = Body(self.events, seed=self.seed, hold=self.hold, speed=self.speed)
        p = self._body.step(t, dt, loud, mouth_w, stress, blink, mood)
        p.mouth_open, p.mouth_w = float(mouth_open), float(mouth_w)
        return p
