"""Pick what to talk about: score real, current stories on the six-criterion rubric.

The model rates each criterion 0-10; the weighted total is computed here, in code, so a
model cannot simply declare a favourite. Category rotation keeps the feed varied.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

from . import llm, safety
from .config import Config
from .research import Candidate

CATEGORIES = [
    "ai_discoveries", "practical_ai_tutorials", "career_skills", "productivity_methods", "psychology_learning",
    "business_insights", "underrated_tools", "myths_vs_reality", "research_findings", "step_by_step_workflows",
]
WEIGHTS = {"audience": 20, "curiosity": 20, "practical": 20, "timeliness": 15, "originality": 15, "story": 10}
MIN_TOTAL = 62

SELECT_SYSTEM = """You are the topic scout for SHUBH, an Instagram Reels brand about AI tools, productivity and career skills.
Audience: students, young professionals, creators and entrepreneurs who are smart but not technical.
Brand: intelligent, modern, practical, curiosity-driven. Not generic motivation, not hype.

You get a list of REAL recent stories (id, title, publisher, age in days, snippet). Choose up to 8 that could become a genuinely useful 30-second reel.
A story qualifies only if it teaches something useful, reveals something surprising, solves a real problem, gives a practical advantage or explains an important development. Trending is not the same as valuable: skip developer-only tools and code topics (anything a student who does not code cannot use tomorrow), skip gossip, funding rounds, politics, hardware specs, and developer-only topics (code, servers, CI pipelines, command lines): a non-coder student must be able to use the idea tomorrow. Skip anything the viewer cannot use or learn from.
Rate each chosen story 0-10 on: audience (relevance), curiosity (stopping power), practical (value to the viewer), timeliness (is there a reason to post it now), originality (fresh angle), story (storytelling potential). Be harsh; most stories deserve 4-6.
Give each a category from the list given, and one sentence "angle": what the viewer will be able to do or understand after the reel.
Reply with JSON only: {"ranked": [{"id": 1, "category": "...", "angle": "...", "scores": {"audience": 0, "curiosity": 0, "practical": 0, "timeliness": 0, "originality": 0, "story": 0}, "reject_reason": ""}]}"""


def total_score(scores: dict) -> float:
    return round(sum(WEIGHTS[k] * max(0.0, min(10.0, float(scores.get(k, 0)))) / 10.0 for k in WEIGHTS), 1)


def category_bonus(category: str, history: list[dict]) -> float:
    """+ for a category not used lately, - for one used in the last two reels."""
    recent = [h.get("category") for h in history[-8:] if h.get("category")]
    if not recent:
        return 0.0
    last2 = recent[-2:]
    if category in last2:
        return -8.0
    return 6.0 if category not in recent else 0.0


def choose(cfg: Config, cands: list[Candidate], history: list[dict], today: dt.date | None = None) -> list[dict]:
    """Ranked picks: [{candidate, category, angle, scores, total}], best first, only those above the bar."""
    if not cands:
        return []
    past = [h.get("topic", "") for h in history][-40:]
    fresh = [c for c in cands if not safety.too_similar(c.title, past, 0.5)
             and c.url not in {h.get("source_url") for h in history}]
    if not fresh:
        return []
    by_id = {c.id: c for c in fresh}
    user = (f"Today: {(today or dt.date.today()).isoformat()}\nCategories: {', '.join(CATEGORIES)}\n"
            f"Recent reels, avoid repeating their topics or categories: "
            f"{json.dumps([(h.get('topic'), h.get('category')) for h in history[-12:]])}\n"
            f"Stories: {json.dumps([c.brief() for c in fresh[:25]])}")
    try:
        raw = llm.chat_json(cfg, SELECT_SYSTEM, user, temperature=0.3)
    except Exception as e:  # noqa: BLE001
        print(f"[topic] scoring unavailable ({type(e).__name__}); using evergreen mode", file=sys.stderr)
        return []
    ranked = []
    for r in raw.get("ranked", []):
        try:
            c = by_id[int(r["id"])]
            cat = r.get("category")
            if cat not in CATEGORIES:
                continue
            base = total_score(r["scores"])
        except (KeyError, ValueError, TypeError):
            continue
        total = round(base + category_bonus(cat, history), 1)
        if base >= MIN_TOTAL and total >= MIN_TOTAL and not r.get("reject_reason"):
            ranked.append({"candidate": c, "category": cat, "angle": str(r.get("angle", ""))[:240],
                           "scores": r["scores"], "base": base, "total": total})
    ranked.sort(key=lambda x: -x["total"])
    return ranked


def evergreen_category(history: list[dict], today: dt.date | None = None) -> str:
    """Least recently used category for the evergreen fallback."""
    last_used = {c: -1 for c in CATEGORIES}
    for i, h in enumerate(history):
        if h.get("category") in last_used:
            last_used[h["category"]] = i
    order = sorted(CATEGORIES, key=lambda c: (last_used[c], CATEGORIES.index(c)))
    return order[0]
