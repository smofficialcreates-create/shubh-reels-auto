"""Comment-keyword rewards: when someone comments the keyword of a recent reel, send them the promised text as a
private reply through Instagram's official messaging API (one reply per comment, within seven days).

python -m reelbot.responder
Runs every 15 minutes from GitHub Actions. It never replies twice to the same comment and never fails the workflow.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import requests

from . import insights
from .config import Config

WINDOW_DAYS = 7          # Instagram only allows a private reply within seven days of the comment
MAX_PER_RUN = 25
MAX_TRIES = 3


def due_scripts(folder: str = "data/scripts", today: dt.date | None = None) -> list[dict]:
    today = today or dt.date.today()
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        try:
            s = json.load(open(os.path.join(folder, name), encoding="utf-8"))
            age = (today - dt.date.fromisoformat(s["date"])).days
        except (OSError, ValueError, KeyError):
            continue
        if s.get("keyword") and s.get("reward_text") and s.get("media_id") and 0 <= age <= WINDOW_DAYS:
            out.append(s)
    return out


def keyword_hit(text: str, keyword: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])" + re.escape(keyword) + r"(?![A-Za-z])", text or "", re.I))


def message_for(script: dict) -> str:
    return script["reward_text"].strip()[:900]


def send_private_reply(cfg: Config, comment_id: str, text: str) -> None:
    """Tries the Instagram-user endpoint first, then the 'me' endpoint (the Facebook-login flavour)."""
    last = ""
    for path in (f"{cfg.ig_user_id}/messages", "me/messages"):
        r = requests.post(f"https://{cfg.graph_host}/{cfg.graph_version}/{path}", timeout=30,
                          headers={"Authorization": f"Bearer {cfg.ig_token}"},
                          json={"recipient": {"comment_id": comment_id}, "message": {"text": text}})
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code < 400 and "error" not in data:
            return
        e = data.get("error", {})
        last = f"HTTP {r.status_code} code={e.get('code')} {e.get('message', '')[:160]}"
    raise RuntimeError(last)


def run(cfg: Config, scripts: list[dict], replied: dict, send=send_private_reply, owner: str = "") -> tuple[int, list[str]]:
    sent, notes = 0, []
    for s in scripts:
        try:
            comments = insights._get(cfg, f"{s['media_id']}/comments", {"fields": "id,text,username,timestamp", "limit": 50}).get("data", [])
        except insights.InsightsError as e:
            notes.append(f"could not read comments of {s['media_id']}: {e}")
            continue
        for c in comments:
            cid = c.get("id")
            rec = replied.get(cid, {})
            if not cid or rec.get("status") == "sent" or rec.get("tries", 0) >= MAX_TRIES:
                continue
            if not keyword_hit(c.get("text", ""), s["keyword"]):
                continue
            if (owner and c.get("username", "").lower() == owner.lower()) or c.get("text", "").strip() == s.get("first_comment", "").strip():
                continue                      # our own comment
            if sent >= MAX_PER_RUN:
                notes.append("limit per run reached; the rest go out next time")
                return sent, notes
            try:
                send(cfg, cid, message_for(s))
                replied[cid] = {"status": "sent", "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "media_id": s["media_id"]}
                sent += 1
            except Exception as e:  # noqa: BLE001
                replied[cid] = {"status": "failed", "tries": rec.get("tries", 0) + 1, "error": str(e)[:200]}
                notes.append(f"reply failed: {str(e)[:200]}")
                if "code=10" in str(e) or "code=200" in str(e) or "code=3" in str(e):
                    notes.append("the token probably lacks the messages permission (instagram_manage_messages); stopping")
                    return sent, notes
    return sent, notes


def main(argv=None) -> int:
    cfg = Config()
    scripts = due_scripts()
    if not scripts:
        print("no keyword reels in the last seven days; nothing to do")
        return 0
    if not (cfg.ig_token and cfg.ig_user_id):
        print("Instagram secrets missing")
        return 0
    try:
        replied = json.load(open("data/replied.json"))
    except (OSError, json.JSONDecodeError):
        replied = {}
    owner = ""
    try:
        owner = insights._get(cfg, cfg.ig_user_id, {"fields": "username"}).get("username", "")
    except insights.InsightsError:
        pass
    send = send_private_reply if cfg.publish_mode != "dry" else (lambda c, cid, t: print(f"[dry] would reply to {cid}: {t[:60]}..."))
    sent, notes = run(cfg, scripts, replied, send, owner)
    json.dump(replied, open("data/replied.json", "w"), indent=1)
    for n in notes:
        print(n)
    print(f"sent {sent} private replies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
