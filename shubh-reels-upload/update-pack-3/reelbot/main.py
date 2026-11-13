"""Entry point: script -> voice -> video -> (optional) Instagram.

python -m reelbot.main                  # real run (needs secrets)
python -m reelbot.main --sample --fake-tts --no-publish --max-seconds 8   # offline smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

from . import brand, cover, editor, publish, script_gen, tts
from .config import Config
from .sample import SAMPLE
from .puppet import Shubh as Mascot


def load_history(path="data/history.json") -> list[dict]:
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use the built-in script, no LLM")
    ap.add_argument("--fake-tts", action="store_true", help="offline stand-in voice (tests only)")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    os.makedirs(args.out_dir, exist_ok=True)
    history = load_history()
    t0 = time.time()

    script = dict(SAMPLE) if args.sample else script_gen.generate(cfg, history)
    json.dump(script, open(os.path.join(args.out_dir, "script.json"), "w"), indent=2)
    print(f"topic: {script['topic']}  (hook score {script.get('hook_score', 'n/a')})")

    texts = brand.spoken(script)
    nar = tts.synth_fake(texts) if args.fake_tts else tts.synth_voice(texts, cfg.voice, cfg.speed, cfg.model_dir)
    total = len(nar.audio) / nar.sr
    print(f"voice: {total:.1f}s")
    limit = 55 if script.get("format") == "short" else 85
    if total > limit and not args.sample:
        print(f"narration is {total:.0f}s, longer than {limit}s; switching to a backup lesson so today still posts")
        script = script_gen.backup_script(history, dt.date.today(), script.get("format") or "long")
        json.dump(script, open(os.path.join(args.out_dir, "script.json"), "w"), indent=2)
        texts = brand.spoken(script)
        nar = tts.synth_fake(texts) if args.fake_tts else tts.synth_voice(texts, cfg.voice, cfg.speed, cfg.model_dir)
        total = len(nar.audio) / nar.sr
        print(f"voice: {total:.1f}s")
    if total > 88:
        print("::error::narration longer than 88s; Instagram reels allow 90s. Skipping.")
        return 1

    stage = Mascot(cfg.character)
    video = os.path.join(args.out_dir, "reel.mp4")
    info = editor.render_reel(script, nar, stage, video, os.path.join(args.out_dir, "work"),
                              tag=script.get("tag", "AI TOOL TIP"), max_seconds=args.max_seconds)
    caption = script_gen.build_caption(script)
    open(os.path.join(args.out_dir, "caption.txt"), "w").write(caption)
    cover_path = cover.render_cover(script, stage, os.path.join(args.out_dir, "cover.jpg"))
    json.dump({"topic": script["topic"], "category": script.get("category"), "mode": script.get("mode"),
               "source": script.get("source"), "hook": script["hook"], "hook_mechanism": script.get("hook_mechanism"),
               "takeaway": script.get("takeaway"), "quality": script.get("quality"), "topic_scores": script.get("topic_scores"),
               "research": script.get("research")}, open(os.path.join(args.out_dir, "research.json"), "w"), indent=2, default=str)
    print(f"rendered {info['frames']} frames in {time.time() - t0:.0f}s -> {video}")

    if args.no_publish or args.sample or not cfg.can_publish:
        why = "dry run" if (args.no_publish or args.sample) else "publishing not configured (missing secrets)"
        print(f"not posting: {why}. The video is in {args.out_dir}/ (also uploaded as a workflow artifact).")
        return 0

    if cfg.post_at:
        publish.wait_until(cfg.post_at)
    urls = publish.push_media(cfg, video, cover_path)
    result = publish.publish_reel(cfg, urls["video"], caption, urls["cover"])
    print("published:", result)
    if script.get("first_comment"):
        publish.post_comment(cfg, result["id"], script["first_comment"])
    history.append({"date": dt.date.today().isoformat(), "topic": script["topic"], "category": script.get("category"),
                    "hook": script["hook"], "format": script.get("format"), "hook_mechanism": script.get("hook_mechanism"),
                    "cta_type": script.get("cta_type"), "words": len(" ".join(brand.spoken(script)).split()),
                    "source_url": (script.get("source") or {}).get("url"),
                    "mode": script.get("mode"), "quality": (script.get("quality") or {}).get("total"),
                    "media_id": result["id"]})
    json.dump(history[-200:], open("data/history.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
