"""Weekly carousel: turn one posted reel's script into a swipeable post (1080x1350) and publish it.

python -m reelbot.carousel               # pick a script, render slides, post (PUBLISH_MODE=dry renders only)
Slides are drawn with Pillow in the brand palette. Instagram's official API publishes the carousel.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

import requests
from PIL import Image, ImageDraw

from . import fonts, palette as PAL, plan, series
from .config import Config

W, H = 1080, 1350
PAD = 90
MAX_BODY = 6


# ------------------------------------------------------------------ drawing helpers
def wrap(text: str, font, maxw: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_block(d, text, x, y, font, fill, maxw, gap=1.28, accent=None, accent_fill=None, accent_font=None) -> int:
    """Draw wrapped text; the accent word (if any) is set in gold italic serif. Returns the y below the block."""
    lh = int(font.size * gap)
    for line in wrap(text, font, maxw):
        cx = x
        for w in line.split():
            is_acc = accent and re.sub(r"[^a-z']", "", w.lower()) == accent.lower()
            f = accent_font if (is_acc and accent_font) else font
            d.text((cx, y), w, font=f, fill=accent_fill if is_acc else fill)
            cx += int(f.getlength(w + " "))
        y += lh
    return y


def frame(theme: PAL.Theme, idx: int, total: int, label: str):
    img = Image.new("RGB", (W, H), theme.bg)
    d = ImageDraw.Draw(img)
    for gx in range(PAD, W, 90):                       # fine dot grid
        for gy in range(PAD, H, 90):
            d.ellipse((gx - 2, gy - 2, gx + 2, gy + 2), fill=PAL.mix(theme.bg, theme.text, 0.10))
    d.text((PAD, 70), label, font=fonts.get("mono", 28), fill=theme.mono)
    d.text((W - PAD, 70), f"{idx:02d}/{total:02d}", font=fonts.get("mono", 28), fill=theme.mono, anchor="ra")
    d.rectangle((0, 0, int(W * idx / total), 8), fill=PAL.GOLD)
    d.text((PAD, H - 90), "SHUBH", font=fonts.get("bold", 30), fill=theme.text)
    if idx < total:
        d.text((W - PAD, H - 90), "swipe  →", font=fonts.get("mono", 30), fill=theme.accent, anchor="ra")
    return img, d


def slide_cover(script, theme, total, label):
    img, d = frame(theme, 1, total, label)
    words = script["hook"].split()
    focus = re.sub(r"[^a-z']", "", (script.get("hook_focus") or "").lower())
    y = 430
    y = text_block(d, script["hook"], PAD, y, fonts.get("bold", 112), theme.text, W - 2 * PAD, 1.18,
                   accent=focus or None, accent_fill=theme.accent, accent_font=fonts.get("serif", 118))
    d.text((PAD, y + 40), script.get("series", "") and f"// {series.label(script)}", font=fonts.get("mono", 34), fill=theme.accent)
    return img


def slide_text(script, theme, i, total, label, scene, body):
    img, d = frame(theme, i, total, label)
    txt = " ".join(body[scene["from"] - 1:scene["to"]])
    d.text((PAD, 250), "// " + (scene.get("label") or "next"), font=fonts.get("mono", 32), fill=theme.accent)
    size = 70 if len(txt) < 120 else 58
    text_block(d, txt, PAD, 340, fonts.get("bold", size), theme.text, W - 2 * PAD)
    return img


def slide_chat(script, theme, i, total, label, v):
    img, d = frame(theme, i, total, label)
    d.text((PAD, 250), "// try it", font=fonts.get("mono", 32), fill=theme.accent)
    f = fonts.get("sans", 50)
    p_lines = wrap(v.get("prompt", ""), f, 700)
    ph = len(p_lines) * 66 + 70
    d.rounded_rectangle((W - PAD - 760, 340, W - PAD, 340 + ph), 36, fill=PAL.NAVY if theme.name != "navy" else PAL.FOREST)
    for k, ln in enumerate(p_lines):
        d.text((W - PAD - 730, 372 + k * 66), ln, font=f, fill=PAL.IVORY)
    r_lines = wrap(v.get("result", ""), f, 760)
    rh = len(r_lines) * 66 + 70
    y0 = 340 + ph + 40
    d.rounded_rectangle((PAD, y0, PAD + 820, y0 + rh), 36, fill=PAL.WHITE)
    for k, ln in enumerate(r_lines):
        d.text((PAD + 36, y0 + 32 + k * 66), ln, font=f, fill=PAL.NAVY)
    return img


def slide_compare(script, theme, i, total, label, v):
    img, d = frame(theme, i, total, label)
    d.text((PAD, 250), "// weak vs better", font=fonts.get("mono", 32), fill=theme.accent)
    y = 340
    for side, fill, ink in (("left", PAL.CHARCOAL, PAL.IVORY), ("right", PAL.WHITE, PAL.NAVY)):
        blk = v.get(side) or {}
        rows = blk.get("rows", [])[:3]
        h = 120 + 70 * len(rows)
        d.rounded_rectangle((PAD, y, W - PAD, y + h), 30, fill=fill)
        d.text((PAD + 40, y + 34), (blk.get("title") or side).upper(), font=fonts.get("mono", 32), fill=PAL.GOLD)
        for k, r in enumerate(rows):
            d.text((PAD + 40, y + 100 + k * 70), "• " + r, font=fonts.get("sans", 46), fill=ink)
        y += h + 40
    return img


def slide_list(script, theme, i, total, label, v, kind):
    img, d = frame(theme, i, total, label)
    items = v.get("nodes") or v.get("labels") or []
    d.text((PAD, 250), "// " + ("the steps" if kind != "checklist" else "remember"), font=fonts.get("mono", 32), fill=theme.accent)
    y = 340
    for k, it in enumerate(items[:6], 1):
        d.ellipse((PAD, y, PAD + 80, y + 80), fill=PAL.GOLD)
        d.text((PAD + 40, y + 40), str(k), font=fonts.get("bold", 44), fill=PAL.NAVY, anchor="mm")
        text_block(d, it, PAD + 120, y + 8, fonts.get("bold", 54), theme.text, W - 2 * PAD - 120)
        y += 130
    return img


def slide_definition(script, theme, i, total, label, v):
    img, d = frame(theme, i, total, label)
    d.text((PAD, 250), "// in plain words", font=fonts.get("mono", 32), fill=theme.accent)
    d.text((PAD, 330), v.get("term", ""), font=fonts.get("serif", 150), fill=theme.accent)
    text_block(d, v.get("meaning", ""), PAD, 560, fonts.get("bold", 62), theme.text, W - 2 * PAD)
    return img


def slide_final(script, theme, i, total, label):
    img, d = frame(theme, i, total, label)
    d.text((PAD, 300), "// your turn", font=fonts.get("mono", 32), fill=theme.accent)
    y = text_block(d, "Save this for later.", PAD, 380, fonts.get("bold", 100), theme.text, W - 2 * PAD, 1.15,
                   accent="later", accent_fill=theme.accent, accent_font=fonts.get("serif", 112))
    y = text_block(d, script.get("cta", ""), PAD, y + 50, fonts.get("sans", 58), theme.text, W - 2 * PAD)
    d.rounded_rectangle((PAD, y + 80, PAD + 620, y + 190), 55, fill=PAL.GOLD)
    d.text((PAD + 310, y + 135), "Follow Shubh", font=fonts.get("bold", 50), fill=PAL.NAVY, anchor="mm")
    return img


def render_slides(script: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    body = script["sentences"]
    scenes = plan.repair_scenes(script.get("scenes") or [], len(body)) or plan.repair_scenes(plan.legacy_scenes(script), len(body))
    scenes = scenes[:MAX_BODY]
    total = len(scenes) + 2
    cycle = [PAL.NAVY_T, PAL.IVORY_T, PAL.FOREST_T]
    seed = sum(ord(c) for c in script.get("topic", ""))
    label = "// " + (series.label(script) or "SHUBH")
    imgs = [slide_cover(script, PAL.NAVY_T, total, label)]
    for k, sc in enumerate(scenes):
        th = cycle[(seed + k + 1) % 3]
        if th.name == "navy" and k == 0:
            th = PAL.IVORY_T
        v = sc["visual"]
        t = v.get("type")
        i = k + 2
        if t == "chat" and v.get("prompt"):
            imgs.append(slide_chat(script, th, i, total, label, v))
        elif t == "compare" and v.get("left") and v.get("right"):
            imgs.append(slide_compare(script, th, i, total, label, v))
        elif t in ("flow", "layers", "checklist") and (v.get("nodes") or v.get("labels")):
            imgs.append(slide_list(script, th, i, total, label, v, t))
        elif t == "definition" and v.get("term"):
            imgs.append(slide_definition(script, th, i, total, label, v))
        else:
            imgs.append(slide_text(script, th, i, total, label, sc, body))
    imgs.append(slide_final(script, PAL.FOREST_T, total, total, label))
    paths = []
    for n, im in enumerate(imgs, 1):
        p = os.path.join(out_dir, f"slide-{n:02d}.jpg")
        im.save(p, quality=93)
        paths.append(p)
    return paths


def carousel_caption(script: dict) -> str:
    from .script_gen import FIXED_TAGS
    parts = [script["hook"], script.get("takeaway") or "", "Save this for later and send it to someone who uses AI every day.",
             script.get("cta", ""), series.label(script), "AI-generated visuals."]
    return "\n".join(p for p in parts if p).strip() + "\n\n" + FIXED_TAGS


# ------------------------------------------------------------------ picking what to post
def load_scripts(folder="data/scripts") -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            out.append(json.load(open(p, encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def pick_script(scripts: list[dict], done: list[str], insights: dict | None = None) -> dict | None:
    """Best reel not yet turned into a carousel. With insights: highest saves+shares per reach; otherwise the newest
    reel that has a chat demo."""
    pool = [s for s in scripts if s.get("media_id") not in done and s.get("sentences")]
    if not pool:
        return None
    ins = (insights or {}).get("media", {})

    def score(s):
        m = ins.get(s.get("media_id"), {}).get("metrics", {})
        reach = max(1, m.get("reach", 0))
        return ((m.get("saved", 0) * 2 + m.get("shares", 0) * 3) / reach * 1000, s.get("date", ""))
    if ins and any(s.get("media_id") in ins for s in pool):
        return max(pool, key=score)
    with_chat = [s for s in pool if any((x.get("visual") or {}).get("type") == "chat" for x in (s.get("scenes") or []))]
    return max(with_chat or pool, key=lambda s: s.get("date", ""))


# ------------------------------------------------------------------ publishing
def _graph(cfg: Config, path: str) -> str:
    return f"https://{cfg.graph_host}/{cfg.graph_version}/{path}"


def _check(r, what):
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code >= 400 or "error" in data:
        err = data.get("error", {})
        raise RuntimeError(f"{what} failed: HTTP {r.status_code} code={err.get('code')} {err.get('message', '')[:200]}")
    return data


def push_files(cfg: Config, paths: list[str], branch: str = "media-carousel") -> list[str]:
    """Host the slides on a throw-away branch (same trick as the reel video); returns their public URLs."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    tmp = tempfile.mkdtemp(prefix="car-")
    try:
        os.makedirs(os.path.join(tmp, "slides"))
        names = []
        for p in paths:
            n = f"{stamp}-{os.path.basename(p)}"
            shutil.copy(p, os.path.join(tmp, "slides", n))
            names.append(n)
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, env=env, capture_output=True, text=True)  # noqa: E731
        run("init", "-q"); run("config", "user.name", "reel-bot"); run("config", "user.email", "reel-bot@users.noreply.github.com")
        run("checkout", "-q", "-b", branch); run("add", "-A"); run("commit", "-q", "-m", "media: carousel")
        remote = f"https://x-access-token:{cfg.gh_token}@github.com/{cfg.repo}.git"
        r = subprocess.run(["git", "-C", tmp, "push", "-q", "--force", remote, f"{branch}:{branch}"], env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("could not push slides: " + r.stderr.replace(cfg.gh_token, "***")[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    urls = [f"https://raw.githubusercontent.com/{cfg.repo}/{branch}/slides/{n}" for n in names]
    for u in urls:
        for _ in range(12):
            if requests.head(u, timeout=20, allow_redirects=True).status_code == 200:
                break
            time.sleep(5)
        else:
            raise RuntimeError("uploaded slide is not publicly reachable (is the repository public?)")
    return urls


def publish_carousel(cfg: Config, urls: list[str], caption: str) -> dict:
    auth = {"Authorization": f"Bearer {cfg.ig_token}"}
    kids = []
    for u in urls:
        r = requests.post(_graph(cfg, f"{cfg.ig_user_id}/media"), headers=auth, timeout=60,
                          data={"image_url": u, "is_carousel_item": "true"})
        kids.append(_check(r, "create slide")["id"])
    r = requests.post(_graph(cfg, f"{cfg.ig_user_id}/media"), headers=auth, timeout=60,
                      data={"media_type": "CAROUSEL", "children": ",".join(kids), "caption": caption})
    container = _check(r, "create carousel")["id"]
    deadline = time.time() + 300
    while True:
        s = _check(requests.get(_graph(cfg, container), headers=auth, timeout=30, params={"fields": "status_code,status"}), "poll")
        if s.get("status_code") == "FINISHED":
            break
        if s.get("status_code") in ("ERROR", "EXPIRED") or time.time() > deadline:
            raise RuntimeError(f"carousel not accepted: {s}")
        time.sleep(8)
    r = requests.post(_graph(cfg, f"{cfg.ig_user_id}/media_publish"), headers=auth, timeout=60, data={"creation_id": container})
    return {"id": _check(r, "publish carousel")["id"]}


def main(argv=None) -> int:
    cfg = Config()
    out = "out/carousel"
    scripts = load_scripts()
    state_path = "data/carousels.json"
    try:
        done = json.load(open(state_path))
    except (OSError, json.JSONDecodeError):
        done = []
    try:
        insights = json.load(open("data/insights.json"))
    except (OSError, json.JSONDecodeError):
        insights = None
    script = pick_script(scripts, [d["media_id"] for d in done], insights)
    if not script:
        print("no posted reel is waiting to become a carousel yet; nothing to do")
        return 0
    paths = render_slides(script, out)
    caption = carousel_caption(script)
    open(os.path.join(out, "caption.txt"), "w").write(caption)
    print(f"carousel from '{script['topic']}': {len(paths)} slides in {out}/")
    if not cfg.can_publish:
        print("not posting (dry run or missing secrets); slides are saved as a workflow artifact")
        return 0
    from . import publish
    if cfg.post_at:
        publish.wait_until(cfg.post_at)
    result = publish_carousel(cfg, push_files(cfg, paths), caption)
    print("published carousel:", result)
    done.append({"media_id": script["media_id"], "topic": script["topic"], "carousel_id": result["id"],
                 "date": dt.date.today().isoformat()})
    json.dump(done[-100:], open(state_path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
