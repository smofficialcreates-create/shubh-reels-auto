"""Export the Shubh puppet on a transparent background.

  python -m reelbot.puppet_cli --audio narration.wav --out out/shubh.webm [--speed 1.0] [--size 720x1280]
  python -m reelbot.puppet_cli --presets "wave@0.3,present_r@2.5,think@5,rest@7" --seconds 8 --png-dir out/frames

Without --audio the mouth stays closed. Output: VP9 with alpha (.webm), ProRes 4444 (.mov) or a PNG sequence.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import wave

import numpy as np

from .animator import PRESETS, Animator, blink_schedule, blink_value, lip_sync
from .puppet import Shubh

FPS = 30


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return (x.reshape(-1, ch).mean(1) if ch > 1 else x), sr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio")
    ap.add_argument("--presets", default="", help="name@seconds,... from: " + ", ".join(PRESETS))
    ap.add_argument("--speed", type=float, default=1.0, help="animation speed (0.7 calm .. 1.4 snappy)")
    ap.add_argument("--size", default="720x1280")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--full-body", action="store_true")
    ap.add_argument("--out", default="out/shubh.webm")
    ap.add_argument("--png-dir")
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    voice, sr = (read_wav(a.audio) if a.audio else (np.zeros(1, np.float32), 24000))
    dur = a.seconds or (len(voice) / sr if a.audio else 6.0)
    n = int(dur * FPS)
    mo, mw, env = lip_sync(voice, sr, FPS, n) if a.audio else (np.zeros(n), np.full(n, 0.5), np.zeros(n))
    anim = Animator(speed=a.speed)
    for tok in filter(None, a.presets.split(",")):
        name, _, at = tok.partition("@")
        anim.play(name.strip(), float(at or 0))
    puppet = Shubh(scale=2.2 * w / 1080.0)
    blinks = blink_schedule(dur)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    proc = None
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
    else:
        codec = (["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"] if a.out.endswith(".mov")
                 else ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28", "-auto-alt-ref", "0"])
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
        if a.audio:
            cmd += ["-i", a.audio, "-c:a", "libopus" if a.out.endswith(".webm") else "pcm_s16le"]
        proc = subprocess.Popen(cmd + codec + [a.out], stdin=subprocess.PIPE)
    import cv2
    for i in range(n):
        t = i / FPS
        p = anim.pose(t, 1 / FPS, mouth_open=float(mo[i]), mouth_w=float(mw[i]), blink=blink_value(t, blinks), loud=float(env[i]))
        img = puppet.render_rgba(p, size=(w, h), zoom=a.zoom, full_body=a.full_body)
        if a.png_dir:
            cv2.imwrite(os.path.join(a.png_dir, f"shubh_{i:05d}.png"), img)
        else:
            proc.stdin.write(img.tobytes())
    if proc:
        proc.stdin.close()
        proc.wait()
    print("wrote", a.png_dir or a.out)


if __name__ == "__main__":
    main()
