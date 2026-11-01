"""The edit: shot director, kinetic captions, hook card, step stickers, HUD,
sound design and the final encode. Draws straight onto numpy frames.
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import audio as A
from .stage import OUT_H, OUT_W, Motion, Stage
from .tts import Narration

FPS = 30
AMBER = (255, 200, 87)        # RGB
WHITE = (255, 255, 255)
INK = (18, 14, 22)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf",
    "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
    "/usr/share/fonts/opentype/montserrat/Montserrat-ExtraBold.otf",
    "assets/fonts/Montserrat-ExtraBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def find_font() -> str:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no usable bold font found; apt install fonts-montserrat or fonts-dejavu-core")


def ease_out_back(x: float, s: float = 1.9) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


# ---------------------------------------------------------------------------
# sprites
# ---------------------------------------------------------------------------
class Sprites:
    def __init__(self):
        self.font_path = find_font()
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self._cache: dict = {}

    def font(self, size: int):
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def text_width(self, text: str, size: int) -> int:
        return int(self.font(size).getlength(text))

    def word(self, text: str, size: int, fill=WHITE, stroke: int = 9) -> np.ndarray:
        key = ("w", text, size, fill, stroke)
        if key in self._cache:
            return self._cache[key]
        f = self.font(size)
        pad = stroke + 14
        w = int(f.getlength(text)) + 2 * pad
        h = int(size * 1.35) + 2 * pad
        shadow = Image.new("L", (w, h), 0)
        ImageDraw.Draw(shadow).text((pad, pad + 8), text, font=f, fill=170, stroke_width=stroke, stroke_fill=170)
        shadow = shadow.filter(ImageFilter.GaussianBlur(7))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        img.paste((0, 0, 0, 255), (0, 0), shadow)
        ImageDraw.Draw(img).text((pad, pad), text, font=f, fill=fill + (255,), stroke_width=stroke,
                                 stroke_fill=INK + (255,))
        sp = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        self._cache[key] = sp
        return sp

    def pill(self, text: str, size: int, bg=(20, 18, 28, 215), fg=WHITE, badge: str | None = None,
             badge_fill=AMBER, pad_x: int = 34, pad_y: int = 20, min_w: int = 0) -> np.ndarray:
        key = ("p", text, size, bg, fg, badge, badge_fill, pad_x, pad_y, min_w)
        if key in self._cache:
            return self._cache[key]
        f = self.font(size)
        tw = int(f.getlength(text))
        bw = int(size * 1.5) if badge else 0
        h = int(size * 1.25) + 2 * pad_y
        w = max(min_w, tw + 2 * pad_x + bw)
        S = 2
        img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=h * S // 2, fill=bg)
        x = pad_x
        if badge:
            r = int(size * 0.62)
            cx, cy = pad_x - 6 + r, h // 2
            d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=badge_fill + (255,))
            bf = ImageFont.truetype(self.font_path, int(size * 0.85) * S)
            bwid = bf.getlength(badge)
            d.text((cx * S - bwid / 2, cy * S - int(size * 0.85) * S * 0.56), badge, font=bf, fill=INK + (255,))
            x = pad_x + bw - 8
        tf = ImageFont.truetype(self.font_path, size * S)
        d.text((x * S, (h * S - size * S * 1.15) / 2 - size * S * 0.02), text, font=tf, fill=fg + (255,))
        img = img.resize((w, h), Image.LANCZOS)
        sp = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        self._cache[key] = sp
        return sp

    def highlighter(self, w: int, h: int) -> np.ndarray:
        key = ("h", w, h)
        if key in self._cache:
            return self._cache[key]
        S = 2
        img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        ImageDraw.Draw(img).rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=18 * S,
                                              fill=(235, 90, 60, 255))
        img = img.resize((w, h), Image.LANCZOS)
        sp = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        self._cache[key] = sp
        return sp


def blit(frame: np.ndarray, sp: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    h, w = sp.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(OUT_W, x + w), min(OUT_H, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    s = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (s[..., 3:4].astype(np.float32) / 255.0) * min(1.0, alpha)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - a) + s[..., :3].astype(np.float32) * a).astype(np.uint8)


def scaled(sp: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    h, w = sp.shape[:2]
    nw, nh = max(1, int(w * sx)), max(1, int(h * sy))
    return cv2.resize(sp, (nw, nh), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    t0: float
    t1: float
    words: list  # (text, t0, t1)


def make_chunks(word_times: list[tuple[str, float, float]], max_words: int = 3, max_chars: int = 22) -> list[Chunk]:
    chunks, cur = [], []
    for w in word_times:
        cur.append(w)
        text = " ".join(x[0] for x in cur)
        end_punct = w[0].endswith((".", "!", "?", ",", ";", ":"))
        if len(cur) >= max_words or len(text) >= max_chars or end_punct:
            chunks.append(Chunk(cur[0][1], cur[-1][2], cur))
            cur = []
    if cur:
        chunks.append(Chunk(cur[0][1], cur[-1][2], cur))
    return chunks


SHOT_CYCLE = ["face", "wide", "hand", "face_tight", "tie", "wide", "face", "hand", "face_tight", "tie"]


def plan_shots(sentence_times, min_len: float = 2.6):
    cuts = [(0.0, SHOT_CYCLE[0])]
    k = 1
    for i, (s, _e) in enumerate(sentence_times):
        if i == 0:
            continue
        if s - cuts[-1][0] >= min_len:
            cuts.append((s, SHOT_CYCLE[k % len(SHOT_CYCLE)]))
            k += 1
    return cuts


def pick_focus_word(words: list[str], hint: str | None = None) -> int:
    clean = [re.sub(r"[^A-Za-z0-9$%']", "", w).lower() for w in words]
    if hint:
        h = re.sub(r"[^A-Za-z0-9$%']", "", hint).lower()
        if h in clean:
            return clean.index(h)
    for i, w in enumerate(clean):
        if any(c.isdigit() for c in w):
            return i
    stop = {"your", "that", "this", "with", "from", "they", "have", "what", "when", "will", "just", "about", "every"}
    best, bi = -1, 0
    for i, w in enumerate(clean):
        if w in stop:
            continue
        if len(w) > best:
            best, bi = len(w), i
    return bi


def blink_schedule(duration: float, seed: int = 21) -> list[float]:
    rng = np.random.default_rng(seed)
    t, out = rng.uniform(0.9, 1.6), []
    while t < duration:
        out.append(t)
        if rng.random() < 0.18:
            out.append(t + 0.24)
        t += rng.uniform(2.2, 4.8)
    return out


def blink_value(t: float, times: list[float], dur: float = 0.17) -> float:
    for b in times:
        if b <= t < b + dur:
            u = (t - b) / dur
            return math.sin(math.pi * u) ** 1.4
    return 0.0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def write_wav(path: str, stereo: np.ndarray) -> None:
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(A.SR)
        wf.writeframes(pcm.tobytes())


def render_reel(script: dict, nar: Narration, stage: Stage, out_path: str, work_dir: str,
                tag: str = "AI TOOL TIP", debug_times: list[float] | None = None,
                max_seconds: float | None = None) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    sprites = Sprites()
    duration = len(nar.audio) / nar.sr
    if max_seconds:
        duration = min(duration, max_seconds)
    n_frames = int(duration * FPS)

    voice = A.resample(nar.audio, nar.sr)
    env = A.envelope(voice, FPS, n_frames)
    hits = A.emphasis_hits(env, FPS)

    cuts = plan_shots(nar.sentence_times)
    cut_times = [c[0] for c in cuts]
    texts = nar.word_times
    n_sent = len(texts)
    hook_words = texts[0]
    hook_end = nar.sentence_times[0][1]
    focus_i = pick_focus_word([w[0] for w in hook_words], script.get("hook_focus"))
    chunks_by_sentence = [make_chunks(texts[i]) for i in range(n_sent)]

    # key-point stickers
    stickers = []
    for kp in script.get("key_points", [])[:3]:
        si = int(kp["sentence"])
        if 0 <= si < n_sent:
            s, e = nar.sentence_times[si]
            stickers.append(dict(t0=s + 0.1, t1=e + 0.35, label=str(kp["label"]).upper()[:30], n=len(stickers) + 1))
    cta_start = nar.sentence_times[-1][0]

    # audio
    pop_times = [s["t0"] for s in stickers]
    stereo = A.mix(voice, duration, [c for c in cut_times if c > 0.05], pop_times, thud_time=0.0)
    wav_path = os.path.join(work_dir, "mix.wav")
    write_wav(wav_path, stereo)

    blinks = blink_schedule(duration)
    rng = np.random.default_rng(4)

    # darkening ramps for legibility
    ramp = np.zeros((OUT_H, 1), np.float32)
    yy = np.arange(OUT_H)[:, None].astype(np.float32)
    ramp = 1.0 - 0.50 * np.clip((yy - 950) / 700, 0, 1) ** 1.3
    ramp *= 1.0 - 0.35 * np.clip((330 - yy) / 330, 0, 1)
    ramp = ramp[..., None].astype(np.float32)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-", "-i", wav_path,
           "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-level", "4.1", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-r", str(FPS), "-af", "loudnorm=I=-15:TP=-2:LRA=11,alimiter=limit=0.84:level=disabled", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
           "-movflags", "+faststart", "-t", f"{duration:.3f}", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    debug_frames = {int(t * FPS): t for t in (debug_times or [])}
    shot_idx = 0
    try:
        for i in range(n_frames):
            t = i / FPS
            while shot_idx + 1 < len(cuts) and t >= cuts[shot_idx + 1][0]:
                shot_idx += 1
            t_cut, shot = cuts[shot_idx]
            u = t - t_cut
            punch = (0.18 * math.exp(-u / 0.26)) if shot_idx == 0 else 0.085 * math.exp(-u / 0.16)
            # beat kick on emphasised syllables
            kick = 0.0
            nod_imp = 0.0
            for h in hits:
                dt = t - h / FPS
                if 0 <= dt < 0.6:
                    kick += 0.013 * math.exp(-dt / 0.12)
                    nod_imp += math.exp(-dt / 0.2)
            m = Motion(
                nod_px=1.8 * float(env[i]) + 7.0 * nod_imp,
                tilt=0.010 * nod_imp * (1 if (len(hits) % 2) else -1),
                blink=blink_value(t, blinks),
                glow=1.0 + 0.16 * float(env[i]),
                flash=(0.30 * math.exp(-u / 0.045) if (shot_idx > 0 and u < 0.16) else 0.0),
            )
            frame = stage.render(t, shot, u, punch, m, i, extra_zoom=1.0 + kick)
            frame = (frame.astype(np.float32) * ramp).astype(np.uint8)

            draw_hud(frame, sprites, t, duration, tag)
            if t < hook_end + 0.35:
                draw_hook(frame, sprites, t, hook_words, focus_i, hook_end + 0.35)
            else:
                draw_captions(frame, sprites, t, chunks_by_sentence[1:])
            for st in stickers:
                draw_sticker(frame, sprites, t, st)
            if t >= cta_start:
                draw_follow(frame, sprites, t, cta_start)

            if i in debug_frames:
                cv2.imwrite(os.path.join(work_dir, f"debug_{debug_frames[i]:05.2f}.png"), frame)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return {"duration": duration, "frames": n_frames, "cuts": cuts, "stickers": len(stickers)}


# ---------------------------------------------------------------------------
# overlay drawing
# ---------------------------------------------------------------------------
def draw_hud(frame, sp: Sprites, t: float, duration: float, tag: str) -> None:
    # progress bar (inside the platform safe zone, below the status area)
    x0, x1, y = 60, OUT_W - 60, 244
    track = frame[y:y + 6, x0:x1].astype(np.float32)
    frame[y:y + 6, x0:x1] = (track * 0.35 + 64).astype(np.uint8)
    fx = int(x0 + (x1 - x0) * min(1.0, t / max(duration, 0.1)))
    cv2.rectangle(frame, (x0, y), (fx, y + 6), (87, 200, 255), -1)       # BGR amber
    pill = sp.pill(tag, 30, bg=(20, 18, 28, 200), pad_x=26, pad_y=12)
    a = min(1.0, t / 0.4)
    blit(frame, pill, 60, 270, a)


def draw_hook(frame, sp: Sprites, t: float, words, focus_i: int, t_end: float) -> None:
    size = 98
    maxw = 920
    # lay words out in lines
    sizes = [sp.text_width(w[0].upper() + " ", size) for w in words]
    lines, cur, curw = [], [], 0
    for idx, wd in enumerate(sizes):
        if cur and curw + wd > maxw:
            lines.append(cur)
            cur, curw = [], 0
        cur.append(idx)
        curw += wd
    if cur:
        lines.append(cur)
    line_h = int(size * 1.22)
    block_h = line_h * len(lines)
    y_top = 1250 - block_h // 2
    exit_u = max(0.0, (t - (t_end - 0.2)) / 0.2)
    for li, line in enumerate(lines):
        total = sum(sizes[i] for i in line)
        x = (OUT_W - total) // 2
        for i in line:
            text, w0, _w1 = words[i]
            text_u = text.upper()
            u = (t - (w0 - 0.04)) / 0.2
            if u <= 0:
                x += sizes[i]
                continue
            s = 0.55 + 0.45 * ease_out_back(u, 2.4)
            fill = AMBER if i == focus_i else WHITE
            spr = sp.word(text_u, size, fill=fill, stroke=10)
            if i == focus_i:
                hl_w = sp.text_width(text_u, size) + 36
                hl = sp.highlighter(hl_w, int(size * 1.08))
                wipe = ease_out((t - w0) / 0.25)
                hl_s = scaled(hl, max(0.02, wipe), 1.0)
                blit(frame, hl_s, x - 12 + (OUT_W * 0), y_top + li * line_h + 6, 0.92 * (1 - exit_u))
                spr = sp.word(text_u, size, fill=WHITE, stroke=6)
            ss = scaled(spr, s)
            cx = x + sp.text_width(text_u, size) // 2 + 14
            cy = y_top + li * line_h + int(size * 0.7)
            blit(frame, ss, cx - ss.shape[1] // 2 - 14 + 14, int(cy - ss.shape[0] / 2 - 8 * exit_u * 6),
                 min(1.0, u * 2.5) * (1 - exit_u))
            x += sizes[i]


def draw_captions(frame, sp: Sprites, t: float, chunks_by_sentence) -> None:
    active = None
    allc = [c for ch in chunks_by_sentence for c in ch]
    for j, c in enumerate(allc):
        nxt = allc[j + 1].t0 if j + 1 < len(allc) else c.t1 + 1.0
        if c.t0 - 0.05 <= t < min(nxt, c.t1 + 0.35):
            active = c
    if active is None:
        return
    size = 84
    maxw = 880
    texts = [w[0].upper() for w in active.words]
    widths = [sp.text_width(x + " ", size) for x in texts]
    lines, cur, curw = [], [], 0
    for idx, wd in enumerate(widths):
        if cur and curw + wd > maxw:
            lines.append(cur)
            cur, curw = [], 0
        cur.append(idx)
        curw += wd
    if cur:
        lines.append(cur)
    appear = (t - active.t0 + 0.05) / 0.14
    scale_in = 0.84 + 0.16 * ease_out_back(appear)
    alpha = min(1.0, appear * 2.5)
    line_h = int(size * 1.2)
    y_top = 1290 - (line_h * len(lines)) // 2
    cur_word = max((k for k, w in enumerate(active.words) if w[1] <= t), default=0)
    for li, line in enumerate(lines):
        total = sum(widths[i] for i in line)
        x = (OUT_W - total) // 2
        for i in line:
            hot = (i == cur_word)
            pop = 1.0 + 0.16 * ease_out_back((t - active.words[i][1]) / 0.12, 2.8) if hot else 1.0
            spr = sp.word(texts[i], size, fill=AMBER if hot else WHITE, stroke=9)
            ss = scaled(spr, scale_in * pop)
            wcx = x + widths[i] // 2
            cy = y_top + li * line_h + line_h // 2
            blit(frame, ss, int(wcx - ss.shape[1] / 2), int(cy - ss.shape[0] / 2), alpha)
            x += widths[i]


def draw_sticker(frame, sp: Sprites, t: float, st: dict) -> None:
    if not (st["t0"] <= t <= st["t1"]):
        return
    pill = sp.pill(st["label"], 46, badge=str(st["n"]), pad_x=36, pad_y=22)
    w = pill.shape[1]
    u_in = (t - st["t0"]) / 0.42
    u_out = (t - (st["t1"] - 0.25)) / 0.25
    x_rest = 60
    x = int(-w + (x_rest + w) * ease_out_back(u_in, 1.6))
    if u_out > 0:
        x = int(x_rest - (w + x_rest) * ease_out(u_out))
    blit(frame, pill, x, 1075, 1.0)


def draw_follow(frame, sp: Sprites, t: float, cta_start: float) -> None:
    u = (t - (cta_start + 0.2)) / 0.35
    if u <= 0:
        return
    pulse = 1.0 + 0.035 * math.sin((t - cta_start) * 7.0)
    pill = sp.pill("FOLLOW FOR DAILY TIPS", 44, bg=AMBER + (255,), fg=INK, pad_x=44, pad_y=24)
    s = scaled(pill, (0.6 + 0.4 * ease_out_back(u)) * pulse)
    blit(frame, s, (OUT_W - s.shape[1]) // 2 - 20, 1075, min(1.0, u * 2))
