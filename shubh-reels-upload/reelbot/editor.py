"""Motion-graphics edit in the brand palette: navy / ivory / forest scenes, four kinds of wipe,
kinetic top captions (sans + italic serif accent), a different visual prop per beat, a gesturing
mascot and a synthesized sound bed.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import wave
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import audio as A
from . import brand, fonts, palette as PAL, plan, props
from .puppet import Shubh as Mascot, Pose
from .animator import Animator
from .tts import Narration

W, H = 1080, 1920
FPS = 30
THEMES = PAL.THEMES
HEAD_SRC = (512.0, 115.0)
ANCHOR_SRC = (512.0, 420.0)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out_back(x, s=1.8):
    x = min(max(x, 0.0), 1.0)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


# ---------------------------------------------------------------------------
def make_bg(theme: str) -> np.ndarray:
    """Flat brand colour, a fine dot grid, a soft light behind the mascot and a charcoal floor."""
    t = THEMES[theme]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.empty((H, W, 3), np.float32)
    f[:] = np.array(t.bg, np.float32)
    cell = 44
    cy, cx = (yy % cell) - cell / 2, (xx % cell) - cell / 2
    dots = (np.sqrt(cx * cx + cy * cy) < 1.7).astype(np.float32)
    if theme == "ivory":
        f += dots[..., None] * (np.array(PAL.CHARCOAL, np.float32) - f[0, 0]) * 0.10
        glow = np.clip(1 - np.sqrt((xx - 540) ** 2 + (yy - 1250) ** 2) / 900, 0, 1)
        f += glow[..., None] * np.array([6, 6, 4], np.float32)
        band = np.clip((yy - 1640) / 60.0, 0, 1)[..., None]
        f = f * (1 - band * 0.06)
    else:
        f += dots[..., None] * (np.array(PAL.IVORY, np.float32) - f[0, 0]) * 0.07
        glow = np.clip(1 - np.sqrt((xx - 540) ** 2 + ((yy - 1250) * 1.25) ** 2) / 760, 0, 1) ** 1.6
        lift = np.array(PAL.GREY, np.float32) - f[0, 0]
        f += glow[..., None] * lift * 0.22
        # charcoal floor, slightly slanted, with a thin gold edge
        edge = 1545 + (xx - 540) * 0.05
        floor = np.clip((yy - edge) / 3.0, 0, 1)
        f = f * (1 - floor[..., None]) + np.array(PAL.CHARCOAL, np.float32) * floor[..., None]
        line = np.clip(1 - np.abs(yy - edge) / 2.2, 0, 1)
        f = f * (1 - line[..., None] * 0.55) + np.array(PAL.GOLD, np.float32) * (line[..., None] * 0.55)
    return np.clip(f, 0, 255).astype(np.uint8)[..., ::-1].copy()


def blit(frame, sp, x, y, alpha=1.0):
    h, w = sp.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    s = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (s[..., 3:4].astype(np.float32) / 255.0) * min(1.0, alpha)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - a) + s[..., :3].astype(np.float32) * a).astype(np.uint8)


def scaled(sp, k):
    if abs(k - 1) < 0.01:
        return sp
    h, w = sp.shape[:2]
    return cv2.resize(sp, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)


_WORD_CACHE: dict = {}


def word_sprite(text: str, kind: str, size: int, color) -> np.ndarray:
    key = (text, kind, size, color)
    sp = _WORD_CACHE.get(key)
    if sp is not None:
        return sp
    f = fonts.get(kind, size)
    w = int(f.getlength(text)) + 16
    h = int(size * 1.6)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((8, int(size * 1.15)), text, font=f, fill=color + (255,), anchor="ls")
    sp = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    _WORD_CACHE[key] = sp
    return sp


@dataclass
class CapWord:
    text: str
    t0: float
    kind: str
    size: int
    x: int
    y: int          # baseline
    accent: bool


def _layout(words, accent_i, top, left, maxw, sans, serif, step):
    out, x, y, lines = [], left, top + 70, 1
    gap = int(sans * 0.3)
    for i, (text, t0, _t1) in enumerate(words):
        acc = (i == accent_i)
        kind, size = ("serif", serif) if acc else ("sans", sans)
        wd = int(fonts.get(kind, size).getlength(text))
        if x + wd > left + maxw and x > left:
            x, y, lines = left, y + step, lines + 1
        out.append(CapWord(text, t0, kind, size, x, y, acc))
        x += wd + gap
    return out, lines


def layout_sentence(words, accent_i: int, top: int = 330, left: int = 70, maxw: int = 930) -> list[CapWord]:
    for sans, serif, step, mx in ((68, 82, 90, 2), (58, 70, 72, 3), (50, 60, 62, 3)):
        out, lines = _layout(words, accent_i, top, left, maxw, sans, serif, step)
        if lines <= mx:
            break
    return out


def section_labels(sections, script) -> dict[int, str]:
    """'// 03 - the layers' for every sentence, numbered by scene (the spoken intro is not counted)."""
    labels, n = {}, 0
    for sec in sections:
        if sec.kind != "intro":
            n += 1
        for i in range(sec.i0, sec.i1 + 1):
            labels[i] = f"// {n:02d} - {sec.label}" if sec.kind != "intro" else "// shubh"
    return labels


class Gestures:
    KEYS = ("l_sh", "l_el", "r_sh", "r_el", "head", "lean", "shrug")

    def __init__(self, events):
        self.ev, self.idx = events, 0
        rest = plan.POSES["rest"]
        self.sp = {k: plan.Spring(rest[k]) for k in self.KEYS}

    def step(self, t, dt, imp):
        while self.idx + 1 < len(self.ev) and self.ev[self.idx + 1][0] <= t:
            self.idx += 1
        t_ev, name = self.ev[self.idx]
        pose = plan.POSES[name]
        tgt = {k: pose[k] for k in self.KEYS}
        if pose.get("osc") and t - t_ev > 0.3:
            ph = math.sin(2 * math.pi * pose["hz"] * (t - t_ev))
            if pose["osc"] == "both":
                tgt["l_el"] += pose["amp"] * ph
                tgt["r_el"] += pose["amp"] * ph
            else:
                tgt[pose["osc"]] += pose["amp"] * ph
        for side in ("l", "r"):
            if abs(tgt[f"{side}_el"]) > 20:
                tgt[f"{side}_el"] += math.copysign(9.0, tgt[f"{side}_el"]) * imp
        tgt["shrug"] += 3.0 * imp
        return {k: self.sp[k].step(tgt[k], dt) for k in self.KEYS}, name


def blink_schedule(duration, seed=21):
    rng = np.random.default_rng(seed)
    t, out = rng.uniform(0.8, 1.5), []
    while t < duration:
        out.append(t)
        if rng.random() < 0.2:
            out.append(t + 0.26)
        t += rng.uniform(2.0, 4.4)
    return out


def blink_value(t, times, dur=0.16):
    for b in times:
        if b <= t < b + dur:
            return math.sin(math.pi * (t - b) / dur) ** 1.3
    return 0.0


def write_wav(path, stereo):
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(A.SR)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
def render_reel(script: dict, nar: Narration, mascot: Mascot, out_path: str, work_dir: str,
                tag: str = "", debug_times=None, max_seconds=None, handle: str = "") -> dict:
    os.makedirs(work_dir, exist_ok=True)
    duration = len(nar.audio) / nar.sr
    if max_seconds:
        duration = min(duration, max_seconds)
    n_frames = int(duration * FPS)
    voice = A.resample(nar.audio, nar.sr)
    env, cen = A.voice_features(voice, FPS, n_frames)
    hits = A.emphasis_hits(env, FPS)

    sections = plan.build_sections(script, nar)
    labels = section_labels(sections, script)
    events = plan.gesture_events(sections, nar)
    o = brand.offset(script)
    emph = script.get("emphasis") or []
    layouts = []
    for i, wt in enumerate(nar.word_times):
        j = i - o
        hint = emph[j] if 0 <= j < len(emph) else None
        if i == 0:
            hint = (emph[0] if emph else None) or script.get("hook_focus")
        elif i == 1 and o:
            hint = "Shubh"
        ai = plan.accent_index([w[0] for w in wt], hint)
        layouts.append(layout_sentence(wt, ai))
    step_starts = [x for s in sections for x in s.data.get("abs_starts", [])]

    # sound design
    cuts = [s.t0 for s in sections[1:]]
    pops = [s.t0 + 0.12 for s in sections if s.prop != "none"] + list(step_starts)
    cta = sections[-1]
    pops.append(cta.data["t_follow"] + 0.95)
    stereo = A.mix(voice, duration, cuts, pops, thud_time=0.0)
    wav_path = os.path.join(work_dir, "mix.wav")
    write_wav(wav_path, stereo)

    bgs = {k: make_bg(k) for k in THEMES}
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hx0, hy0 = 540.0, 1247.0
    dist = np.sqrt((xx - hx0) ** 2 + (yy - hy0) ** 2)
    anim = Animator(speed=1.0).load_events(events)
    blinks = blink_schedule(duration)
    mono_f = fonts.get("mono", 26)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav_path, "-c:v", "libx264", "-preset", "fast", "-crf", "21",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-r", str(FPS),
           "-af", "loudnorm=I=-15:TP=-2:LRA=11,alimiter=limit=0.84:level=disabled",
           "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-movflags", "+faststart", "-t", f"{duration:.3f}", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dbg = {int(t * FPS): t for t in (debug_times or [])}
    prev_frame = None
    try:
        for i in range(n_frames):
            t = i / FPS
            k = max(j for j, s in enumerate(sections) if s.t0 <= t + 1e-6)
            sec = sections[k]
            # ---- background and the wipe into this scene ------------------------
            frame = bgs[sec.theme].copy()
            if k > 0:
                u = (t - sec.t0) / 0.5
                if u < 1.0:
                    old, new = bgs[sections[k - 1].theme], bgs[sec.theme]
                    e = ease_io(u)
                    if sec.transition == "slide_up":
                        edge = H * (1 - e)
                        m = yy > edge
                        ring = np.abs(yy - edge) < 4
                    elif sec.transition == "push":
                        edge = W * (1 - e)
                        m = xx > edge
                        ring = np.abs(xx - edge) < 4
                    elif sec.transition == "iris":
                        r = e * 1900
                        dd = np.abs(xx - 540) + np.abs(yy - 960)
                        m = dd < r
                        ring = np.abs(dd - r) < 6
                    else:
                        r = e * 1750
                        m = dist < r
                        ring = np.abs(dist - r) < 5
                    frame = old.copy()
                    frame[m] = new[m]
                    frame[ring & (u > 0.02)] = PAL.GOLD[::-1]
            th = THEMES[sec.theme if (k == 0 or t - sec.t0 > 0.22) else sections[k - 1].theme]
            tx = tuple(th.text)
            ac = tuple(th.accent)

            # ---- header + ticker ---------------------------------------------
            si = max(j for j, (a, b) in enumerate(nar.sentence_times) if a <= t + 1e-6) if t >= nar.sentence_times[0][0] else 0
            hdr = Image.new("RGBA", (W, 90), (0, 0, 0, 0))
            hd = ImageDraw.Draw(hdr)
            hd.text((70, 24), labels.get(si, ""), font=mono_f, fill=tuple(th.mono) + (255,))
            marks = sec.data.get("abs_starts") if sec.prop in ("flow", "layers", "checklist") else None
            if marks:
                done = sum(1 for x in marks if t >= x)
                vf = fonts.get("bold", 36)
                hd.text((W - 70, 12), f"{done}/{len(marks)}", font=vf, fill=(th.c_text if th.marker else th.accent) + (255,), anchor="ra")
            blit(frame, cv2.cvtColor(np.array(hdr), cv2.COLOR_RGBA2BGRA), 0, 238)
            pw = int((W - 140) * min(1.0, t / duration))
            cv2.rectangle(frame, (70, 226), (W - 70, 229), PAL.mix(th.bg, th.mono, 0.35)[::-1], -1)
            cv2.rectangle(frame, (70, 226), (70 + pw, 229), (th.c_fill if th.marker else th.accent)[::-1], -1)

            # ---- kinetic caption ---------------------------------------------
            s0, s1 = nar.sentence_times[si]
            if s0 - 0.1 <= t < s1 + 0.3:
                fade = 1.0 if t < s1 + 0.1 else max(0.0, 1 - (t - s1 - 0.1) / 0.2)
                for cw in layouts[si]:
                    u = (t - cw.t0 + 0.03) / 0.17
                    if u <= 0:
                        continue
                    e = ease_out(u)
                    if cw.accent and th.marker:                       # gold highlighter behind the accent word
                        wd = int(fonts.get(cw.kind, cw.size).getlength(cw.text))
                        hl = ease_out((t - cw.t0) / 0.28)
                        x0, x1 = cw.x - 8, cw.x - 8 + int((wd + 16) * hl)
                        y0, y1 = int(cw.y - cw.size * 0.66), int(cw.y + cw.size * 0.12)
                        roi = frame[y0:y1, x0:x1]
                        if roi.size:
                            roi[:] = (roi * (1 - 0.92 * fade) + np.array(PAL.GOLD[::-1], np.float32) * 0.92 * fade).astype(np.uint8)
                    col = tx if (cw.accent and th.marker) else (ac if cw.accent else tx)
                    spr = word_sprite(cw.text, cw.kind, cw.size, col)
                    blit(frame, spr, cw.x - 8, int(cw.y - cw.size * 1.15 + 16 * (1 - e)), e * fade)
                    if cw.accent and not th.marker:
                        wd = int(fonts.get(cw.kind, cw.size).getlength(cw.text))
                        ul = ease_out((t - cw.t0) / 0.3)
                        cv2.line(frame, (cw.x, cw.y + 14), (cw.x + int(wd * ul), cw.y + 14), PAL.GOLD[::-1], 4, cv2.LINE_AA)

            # ---- prop ---------------------------------------------------------
            if sec.prop != "none":
                tl = t - (sec.t0 + 0.12)
                exit_u = (t - sec.t1) / 0.2 if k + 1 < len(sections) else 0.0
                if tl > 0 and exit_u < 1:
                    spr = self_prop(sec, tl, sec_th(sec))
                    fit = min(1.12, 560 / spr.shape[0], 980 / spr.shape[1])
                    e = ease_out_back(tl / 0.5)
                    kscale = fit * (0.80 + 0.20 * e) * (1 + 0.05 * max(exit_u, 0))
                    spr = scaled(spr, kscale)
                    alpha = min(1.0, tl / 0.18) * (1 - max(exit_u, 0))
                    cx, cy = 540, 825 + int(70 * (1 - e)) + int(40 * max(exit_u, 0))
                    blit(frame, spr, cx - spr.shape[1] // 2, cy - spr.shape[0] // 2, alpha)

            # ---- mascot --------------------------------------------------------
            imp = 0.0
            for h in hits:
                dt_h = t - h / FPS
                if 0 <= dt_h < 0.6:
                    imp += math.exp(-dt_h / 0.2)
            mo = 0.0 if env[i] < 0.1 else float(np.clip(env[i] ** 0.8 * 1.25, 0, 1))
            p = anim.pose(t, 1 / FPS, mouth_open=mo, mouth_w=float(cen[i]), blink=blink_value(t, blinks),
                          loud=float(env[i]), stress=imp, mood=sec.kind)
            zoom = 1.0 + 0.012 * imp
            if k > 0:
                zoom += 0.10 * math.exp(-(t - sec.t0) / 0.22) * (1 if t >= sec.t0 else 0)
            ay = hy0 + 77 * mascot.scale * zoom          # keep the head where it is while zooming
            enter = 0.0 if t > 0.6 else 700 * (1 - ease_out_back(t / 0.6, 1.4))
            mascot.draw(frame, p, anchor_frame=(540.0, ay + enter), zoom=zoom)
            hx, hy = 540.0, hy0 + enter

            # ---- little extras ---------------------------------------------------
            if sec.kind == "problem":
                draw_sweat(frame, hx + 168, hy - 95 + 6 * math.sin(t * 5), t)
            if sec.kind in ("payoff", "cta") and imp > 0.15:
                draw_sparkle(frame, hx - 210, hy - 130, 26 * min(1, imp), PAL.GOLD)
                draw_sparkle(frame, hx + 215, hy - 60, 18 * min(1, imp), PAL.IVORY)

            # ---- transition glitch ---------------------------------------------------
            if k > 0 and 0 <= t - sec.t0 < 0.16:
                sh = int(8 * (1 - (t - sec.t0) / 0.16))
                if sh:
                    frame[:, :, 0] = np.roll(frame[:, :, 0], sh, axis=1)
                    frame[:, :, 2] = np.roll(frame[:, :, 2], -sh, axis=1)
                if prev_frame is not None:
                    frame = cv2.addWeighted(frame, 0.8, prev_frame, 0.2, 0)
            prev_frame = frame

            if i in dbg:
                cv2.imwrite(os.path.join(work_dir, f"debug_{dbg[i]:05.2f}.png"), frame)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return {"duration": duration, "frames": n_frames,
            "sections": [(s.kind, round(s.t0, 2), s.prop) for s in sections]}


def sec_th(sec):
    return THEMES[sec.theme]


def self_prop(sec, tl, th):
    d = sec.data
    rel = [x - (sec.t0 + 0.12) for x in d.get("abs_starts", [])]
    dur = max(1.0, sec.t1 - sec.t0)
    typ = sec.prop
    if typ == "stat":
        return props.stat_prop(d["value"], d["label"], tl, th)
    if typ == "keyword":
        return props.keyword_prop(d["word"], tl, th)
    if typ == "bars":
        return props.bars_card(tl, th)
    if typ == "checklist":
        return props.checklist_card(list(d["labels"])[:5], rel, tl, th)
    if typ == "flow":
        return props.flow_card(list(d["nodes"]), rel, tl, th)
    if typ == "layers":
        return props.layers_card(list(d["labels"] if "labels" in d else d["nodes"]), rel, tl, th)
    if typ == "compare":
        n = max(1, min(3, len(d["left"].get("rows", []))))
        starts = [0.5 + j * max(0.7, (dur - 1.2) / n) for j in range(n)]
        return props.compare_card(d["left"], d["right"], starts, tl, th)
    if typ == "numeral":
        return props.numeral_prop(int(d.get("n", 1)), int(d.get("total", 3)), str(d.get("title", "")), tl, th)
    if typ == "counter":
        return props.counter_prop(int(d["value"]), str(d.get("unit", "")), str(d.get("label", "")), tl, th)
    if typ == "definition":
        return props.definition_card(str(d["term"]), str(d["meaning"]), tl, th)
    if typ == "chat":
        return props.chat_card(d["prompt"], d["result"], max(0.0, tl - 0.3), th)
    if typ == "follow":
        return props.follow_prop(max(0.0, tl - (d["t_follow"] - (sec.t0 + 0.12))), th)
    raise ValueError(sec.prop)


def draw_sweat(frame, x, y, t):
    x, y = int(x), int(y)
    pts = np.array([[x, y - 34], [x - 17, y + 6], [x + 17, y + 6]], np.int32)
    cv2.fillPoly(frame, [pts], (250, 205, 120), cv2.LINE_AA)
    cv2.circle(frame, (x, y + 8), 17, (250, 205, 120), -1, cv2.LINE_AA)
    cv2.circle(frame, (x - 6, y + 5), 5, (255, 255, 255), -1, cv2.LINE_AA)


def draw_sparkle(frame, x, y, r, rgb):
    x, y, r = int(x), int(y), max(4, int(r))
    pts = np.array([[x, y - r], [x + r * 0.28, y - r * 0.28], [x + r, y], [x + r * 0.28, y + r * 0.28],
                    [x, y + r], [x - r * 0.28, y + r * 0.28], [x - r, y], [x - r * 0.28, y - r * 0.28]], np.int32)
    cv2.fillPoly(frame, [pts], rgb[::-1], cv2.LINE_AA)
