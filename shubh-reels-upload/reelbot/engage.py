"""Engagement mechanics: rotating closing styles, comment-keyword rewards and part-two teasers."""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import re

from . import safety

STYLES = {
    "question": ("ONE direct question the viewer can answer in a comment, tied to the content.", "question"),
    "choice": ('ONE either-or question that makes the viewer pick a side, for example "Which would you use first, the weak prompt or the sharp one?"', "choice"),
    "blank": ('A fill-in-the-blank the viewer completes in a comment, spoken with the word "blank", for example "Finish this in the comments: the first thing I ask AI is blank."', "blank"),
    "rate": ('Ask the viewer to rate how useful this was, spoken as "from one to five", in the comments.', "rate"),
    "next": ("Ask which topic they want explained next, in the comments.", "next"),
    "share": ('Tell the viewer to send this reel to ONE specific kind of friend who needs it, for example "Send this to a friend who still asks AI vague questions."', "share"),
    "keyword": ("Offer a reward for a keyword comment, see KEYWORD REWARD below.", "keyword"),
}
KEYWORDS = ("PROMPT", "STEPS", "GUIDE", "CHECKLIST", "TEMPLATE")
TEASER_EVERY = 4
KEYWORD_EVERY = 4


def prompt_block(eng: dict) -> str:
    """Text appended to the writer's task for the chosen closing style."""
    style = eng.get("cta_style", "question")
    out = [f'CLOSING STYLE "{style}": the closing line is: {STYLES[style][0]}']
    if style == "keyword":
        out.append(
            'KEYWORD REWARD: the closing line tells the viewer to comment one keyword to get something useful sent to them, for example '
            '"Comment PROMPT and I will send you the exact prompt." Choose "keyword" from ' + ", ".join(KEYWORDS) + '. Also return "reward_text": the '
            "actual message that is sent to everyone who comments the keyword: 25 to 90 words, plain text, no links, no emojis, no prices, "
            "genuinely useful on its own (the full prompt template, the checklist or the steps promised in the reel, ready to copy). "
            'Start it with "Here is what I promised from the reel:".')
    return "\n".join(out)


def check_cta(style: str | None, cta: str, script: dict) -> list[str]:
    style = style or "question"
    low = cta.lower()
    issues = []
    if style in ("question", "choice", "next") and not cta.rstrip().endswith("?"):
        issues.append(f"closing line must be a question ({style})")
    if style == "choice" and " or " not in f" {low} " and " ya " not in f" {low} ":
        issues.append("an either-or closing question needs the word 'or'")
    if style == "blank" and "blank" not in low and "khali" not in low:
        issues.append("a fill-in closing line must say the word blank")
    if style == "rate" and not re.search(r"one to five|ek se p[a]+n?ch", low):
        issues.append("a rating closing line must say 'from one to five'")
    if style == "share" and not re.search(r"\b(send|bhej|share)", low):
        issues.append("a share closing line must tell the viewer to send the reel to a friend")
    if style == "keyword":
        kw = (script.get("keyword") or "").upper()
        if kw not in KEYWORDS or kw.lower() not in low or "comment" not in low:
            issues.append("keyword closing line must say 'comment' and the chosen keyword")
        issues += check_reward(script.get("reward_text", ""))
    return issues


def check_reward(text: str) -> list[str]:
    n = len(text.split())
    issues = []
    if not (20 <= n <= 110):
        issues.append(f"reward_text has {n} words (need 25 to 90)")
    if re.search(r"https?:|www\.|@|#", text):
        issues.append("reward_text must not contain links, mentions or hashtags")
    issues += safety.check_text("reward", text)
    return issues


def pick_style(history: list[dict], today: dt.date, keyword_ok: bool = True, strategy: dict | None = None) -> str:
    """Rotate closing styles so none repeats back to back; a keyword reward every few reels."""
    recent = [h.get("cta_style") or "question" for h in history][-8:]
    since_kw = next((i for i, s in enumerate(reversed(recent)) if s == "keyword"), len(recent) + KEYWORD_EVERY)
    if keyword_ok and since_kw >= KEYWORD_EVERY - 1 and len(history) >= 2:
        return "keyword"
    pool = [s for s in STYLES if s != "keyword"]
    uses = {s: recent.count(s) for s in pool}
    ordered = sorted(pool, key=lambda s: (uses[s], random.Random(today.toordinal() * 31 + len(s)).random()))
    ordered = [s for s in ordered if s != (recent[-1] if recent else None)]
    st = strategy or {}
    liked = [s for s in ordered if s in st.get("prefer_cta", [])]
    if liked and random.Random(today.toordinal()).random() < 0.5:
        return liked[0]
    return ordered[0]


def syllabus(path: str | None = None) -> list[dict]:
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "syllabus.json")
    return json.load(open(path, encoding="utf-8"))


def entry_by_title(title: str, items: list[dict] | None = None) -> dict | None:
    return next((e for e in (items or syllabus()) if e["title"] == title), None)


def teaser_title(entry: dict) -> str | None:
    t = entry["title"].strip().rstrip(".")
    if len(t.split()) > 9 or re.search(r"\d", t) or safety.check_text("teaser", t):
        return None
    return t


def pick_teaser(history: list[dict], today: dt.date, picker, keyword_style: bool) -> str | None:
    """Every few reels, promise tomorrow's topic. `picker(history, day)` returns a syllabus entry."""
    if keyword_style or len(history) < 2:
        return None
    if history and history[-1].get("promise"):         # today fulfils a promise: no new one
        return None
    since = next((i for i, h in enumerate(reversed(history)) if h.get("promise")), len(history) + TEASER_EVERY)
    if since < TEASER_EVERY - 1:
        return None
    entry = picker(history, today + dt.timedelta(days=1))
    return teaser_title(entry) if entry else None


def pending_promise(history: list[dict]) -> str | None:
    """A topic promised by the last reel that has not been made yet."""
    return (history[-1].get("promise") or None) if history else None


def plan_for(cfg, history: list[dict], today: dt.date, fmt: str, picker, strategy: dict | None = None, lang: str = "en") -> dict:
    hi = lang == "hi"          # Hinglish reels: no keyword reward and no teaser (both need English text)
    keyword_ok = getattr(cfg, "keyword_dm", "on") != "off" and fmt == "short" and bool(getattr(cfg, "ig_token", "")) and not hi
    style = pick_style(history, today, keyword_ok, strategy) if fmt == "short" else "question"
    promised = pending_promise(history)
    teaser = pick_teaser(history, today, picker, style == "keyword") if fmt == "short" and not promised and not hi else None
    return {"cta_style": style, "teaser": teaser, "promised": promised}
