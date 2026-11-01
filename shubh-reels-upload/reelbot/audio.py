"""Audio: voice envelope, a synthesized lo-fi noir music bed, whooshes and pops.

Everything is generated in numpy, so there is no music licence to worry about.
"""
from __future__ import annotations

import math
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

SR = 44100


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _sos(kind, fc, sr=SR, order=2):
    return butter(order, fc, btype=kind, fs=sr, output="sos")


def lowpass(x, fc, sr=SR, order=2):
    return sosfilt(_sos("low", fc, sr, order), x).astype(np.float32)


def highpass(x, fc, sr=SR, order=2):
    return sosfilt(_sos("high", fc, sr, order), x).astype(np.float32)


def envelope(voice: np.ndarray, fps: int, n_frames: int, sr: int = SR) -> np.ndarray:
    """Per-video-frame loudness 0..1 (fast attack, slower release)."""
    hop = sr / fps
    out = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        a, b = int(i * hop), int((i + 2) * hop)
        seg = voice[a:b]
        out[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    ref = np.percentile(out[out > 1e-4], 95) if np.any(out > 1e-4) else 1.0
    out = np.clip(out / max(ref, 1e-6), 0, 1.3)
    sm = np.zeros_like(out)
    for i in range(1, n_frames):
        k = 0.65 if out[i] > sm[i - 1] else 0.22
        sm[i] = sm[i - 1] + k * (out[i] - sm[i - 1])
    return sm


def emphasis_hits(env: np.ndarray, fps: int, min_gap: float = 0.45) -> list[int]:
    """Frame indices where the voice punches (used for nods and camera kicks)."""
    hits, last = [], -10 ** 6
    for i in range(3, len(env) - 3):
        window = env[max(0, i - int(0.5 * fps)):i]
        base = float(np.mean(window)) if len(window) else 0.0
        if env[i] > 0.6 and env[i] > base * 1.35 and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            if i - last > min_gap * fps:
                hits.append(i)
                last = i
    return hits


# ---------------------------------------------------------------------------
# music
# ---------------------------------------------------------------------------
def _midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def make_music(duration: float, seed: int = 3, bpm: float = 76.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * SR) + SR
    t = np.arange(n) / SR
    beat = 60.0 / bpm
    bar = 4 * beat
    chords = [
        (50, [53, 57, 60, 64]),     # Dm9
        (55, [58, 62, 65, 69]),     # Gm9
        (48, [52, 55, 59, 62]),     # Cmaj9
        (53, [57, 60, 64, 67]),     # Fmaj9
    ]
    pad = np.zeros(n, np.float32)
    bass = np.zeros(n, np.float32)
    for ci in range(int(duration / (2 * bar)) + 2):
        root, notes = chords[ci % len(chords)]
        t0, t1 = ci * 2 * bar, (ci + 1) * 2 * bar + 0.6
        i0, i1 = int(t0 * SR), min(n, int(t1 * SR))
        if i0 >= n:
            break
        tt = t[i0:i1] - t0
        att = np.clip(tt / 0.9, 0, 1) ** 2
        rel = np.clip((2 * bar + 0.6 - tt) / 0.6, 0, 1)
        env = att * rel
        for nn in notes:
            f = _midi(nn)
            for det in (0.997, 1.003):
                ph = rng.uniform(0, 6.28)
                pad[i0:i1] += (np.sin(2 * math.pi * f * det * tt + ph)
                               + 0.28 * np.sin(2 * math.pi * 2 * f * det * tt + ph)) * env * 0.045
        # plucked bass on beats 1 and 3
        for b in (0, 2, 4, 6):
            bs = t0 + b * beat
            j0 = int(bs * SR)
            if j0 >= n:
                continue
            j1 = min(n, j0 + int(1.1 * SR))
            bt = t[j0:j1] - bs
            bass[j0:j1] += np.sin(2 * math.pi * _midi(root - 12) * bt) * np.exp(-bt * 3.2) * 0.22
    # soft kick on 1 & 3, brushed hat on off-beats
    drums = np.zeros(n, np.float32)
    for k in range(int(duration / beat) + 2):
        bs = k * beat
        j0 = int(bs * SR)
        if j0 >= n:
            break
        if k % 4 in (0, 2):
            j1 = min(n, j0 + int(0.22 * SR))
            kt = t[j0:j1] - bs
            drums[j0:j1] += np.sin(2 * math.pi * (52 + 40 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 16) * 0.28
        if k % 2 == 1:
            j1 = min(n, j0 + int(0.12 * SR))
            ht = t[j0:j1] - bs
            drums[j0:j1] += rng.normal(0, 1, j1 - j0).astype(np.float32) * np.exp(-ht * 40) * 0.05
    drums = highpass(drums, 70) * 0.9
    # vinyl crackle
    crackle = (rng.random(n) > 0.9993).astype(np.float32) * rng.normal(0, 1, n).astype(np.float32)
    crackle = lowpass(crackle, 4500) * 0.12
    hiss = lowpass(rng.normal(0, 0.004, n).astype(np.float32), 6000)
    music = pad + bass + drums + crackle + hiss
    music = lowpass(music, 5200)
    # gentle tape wobble via slow tremolo
    music *= (1 + 0.04 * np.sin(2 * math.pi * 0.35 * t)).astype(np.float32)
    return music[: int(duration * SR)].astype(np.float32)


def whoosh(dur: float = 0.42, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    x = rng.normal(0, 1, n).astype(np.float32)
    x = highpass(lowpass(x, 5200), 350)
    tt = np.linspace(0, 1, n)
    env = (np.sin(np.pi * tt ** 0.7)) ** 2
    return (x * env * 0.5).astype(np.float32)


def pop(dur: float = 0.14) -> np.ndarray:
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = 520 + 500 * np.exp(-tt * 40)
    return (np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-tt * 28) * 0.35).astype(np.float32)


def thud(dur: float = 0.5) -> np.ndarray:
    n = int(dur * SR)
    tt = np.arange(n) / SR
    return (np.sin(2 * math.pi * (44 + 60 * np.exp(-tt * 18)) * tt) * np.exp(-tt * 7) * 0.6).astype(np.float32)


def _place(dst: np.ndarray, src: np.ndarray, at: float, gain: float):
    i = int(at * SR)
    if i < 0 or i >= len(dst):
        return
    j = min(len(dst), i + len(src))
    dst[i:j] += src[: j - i] * gain


def mix(voice: np.ndarray, duration: float, whoosh_times, pop_times, thud_time: float | None = None) -> np.ndarray:
    """Return a stereo float32 array (N, 2) with ducked music, SFX and voice."""
    n = int(duration * SR)
    v = np.zeros(n, np.float32)
    v[: min(n, len(voice))] = voice[:n]
    peak = np.max(np.abs(v)) or 1.0
    v = v / peak * 0.88

    music = make_music(duration)
    music = music[:n] if len(music) >= n else np.pad(music, (0, n - len(music)))
    rms = float(np.sqrt(np.mean(music ** 2))) or 1.0
    music = music / rms * 0.05
    # side-chain: music dips while the voice speaks
    follower = lowpass(np.abs(v), 7.0, order=1)
    ref = np.percentile(follower, 95) or 1.0
    duck = 1.0 - 0.62 * np.clip(follower / ref, 0, 1)
    music = music * duck
    # fade in / out
    fade = int(0.6 * SR)
    music[:fade] *= np.linspace(0, 1, fade)
    music[-int(1.2 * SR):] *= np.linspace(1, 0, int(1.2 * SR))

    sfx = np.zeros(n, np.float32)
    w = whoosh()
    for tm in whoosh_times:
        _place(sfx, w, tm - 0.18, 0.55)
    p = pop()
    for tm in pop_times:
        _place(sfx, p, tm, 0.5)
    if thud_time is not None:
        _place(sfx, thud(), thud_time, 0.7)

    left = v + music + sfx
    # haas-style width on the music + sfx only
    d = int(0.012 * SR)
    right = v + np.concatenate([np.zeros(d, np.float32), (music + sfx)[:-d]])
    out = np.stack([left, right], axis=1)
    m = np.max(np.abs(out))
    out = out * 0.82                      # land near -14 LUFS, leave true-peak headroom
    m = np.max(np.abs(out))
    if m > 0.9:
        out = out / m * 0.9
    return out.astype(np.float32)


def voice_features(voice: np.ndarray, fps: int, n_frames: int, sr: int = SR):
    """Per video frame: loudness 0..1 and a 0..1 'brightness' (spectral centroid) used to
    choose between round (oo) and wide (ee) mouth shapes."""
    env = envelope(voice, fps, n_frames, sr)
    hop = sr / fps
    cen = np.zeros(n_frames, np.float32)
    win = np.hanning(int(2 * hop))
    freqs = np.fft.rfftfreq(len(win), 1 / sr)
    for i in range(n_frames):
        seg = voice[int(i * hop): int(i * hop) + len(win)]
        if len(seg) < len(win) or env[i] < 0.05:
            cen[i] = 0.5
            continue
        mag = np.abs(np.fft.rfft(seg * win))
        c = float((freqs * mag).sum() / (mag.sum() + 1e-9))
        cen[i] = np.clip((c - 600.0) / 2200.0, 0.0, 1.0)
    sm = np.copy(cen)
    for i in range(1, n_frames):
        sm[i] = sm[i - 1] + 0.35 * (cen[i] - sm[i - 1])
    return env, sm
