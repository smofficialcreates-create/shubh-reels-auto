"""Deterministic guard rails. If anything trips, the reel is NOT posted."""
from __future__ import annotations

import re

BLOCK = [
    (r"\$\s?\d|\d\s?(usd|dollars|rupees|inr|euros?)\b|₹|€|£", "price or money amount"),
    (r"\b(per month|a month|per year|pricing|subscription|premium plan|free plan|free tier|100% free|completely free|unlimited)\b"
     r"|\b(is|are|it's|totally|for|and) free\b|\bfree (to use|tool|app|version|trial|forever)\b",
     "pricing or plan claim"),
    (r"\d+\s?%|\bpercent\b", "statistic"),
    (r"\b(study|studies|research|survey|scientists?|experts?) (shows?|says?|found|proves?)\b|according to", "unsourced claim"),
    (r"\b(guarantee[ds]?|get rich|passive income|make money|earn \$|six figures|overnight)\b", "income or guarantee claim"),
    (r"\b(invest|stocks?|crypto|bitcoin|forex|loan|lawsuit|sue|diagnos\w*|prescri\w*|medication|therapy)\b",
     "financial, legal or medical territory"),
    (r"\b(cigarettes?|smok\w*|vape|tobacco|whisky|whiskey|alcohol|beer|wine|drunk|drugs?)\b", "tobacco, alcohol or drugs"),
    (r"\b(election|politic\w*|vote for|president|prime minister|religion|god)\b", "politics or religion"),
    (r"\b(gpt|claude|gemini|llama|sonnet|opus)[- ]?v?\d", "model version number (goes stale)"),
    (r"https?://|www\.|@\w|#\w", "link, handle or hashtag inside spoken text"),
    (r"\b(fuck|shit|bitch|nsfw|sex|nude|kill|suicide|rape)\w*", "unsafe language"),
]
BLOCK_RE = [(re.compile(p, re.I), why) for p, why in BLOCK]


GROUNDED_OK = {"statistic", "unsourced claim"}      # allowed only when checked against a source (see grounding.py)


def check_text(label: str, text: str, grounded: bool = False) -> list[str]:
    issues = []
    for rx, why in BLOCK_RE:
        if grounded and why in GROUNDED_OK:
            continue
        m = rx.search(text)
        if m:
            issues.append(f"{label}: {why} ('{m.group(0)}')")
    return issues


def check_script(script: dict, grounded: bool = False) -> list[str]:
    issues = []
    body = script.get("sentences", [])
    if not (5 <= len(body) <= 10):
        issues.append(f"body has {len(body)} sentences (need 5-10)")
    spoken = [script.get("hook", "")] + list(body) + [script.get("cta", "")]
    words = sum(len(s.split()) for s in spoken)
    if not (50 <= words <= 95):
        issues.append(f"{words} spoken words (need 50-95)")
    for i, s in enumerate(spoken):
        if len(s.split()) > 24:
            issues.append(f"sentence {i} is {len(s.split())} words, too long to say well")
        issues += check_text(f"sentence {i}", s, grounded)
        if re.search(r"\d", s):
            issues.append(f"sentence {i}: digits in spoken text (write numbers as words)")
    for kp in script.get("key_points", []):
        issues += check_text("key point", str(kp.get("label", "")))
        if not (1 <= int(kp.get("sentence", 0)) <= len(body)):
            issues.append("key point points at a sentence that does not exist")
    cap = script.get("caption", "")
    issues += check_text("caption", re.sub(r"#\w+", "", cap), grounded)
    if len(script.get("hashtags", [])) > 5:
        issues.append("more than 5 hashtags")
    if "ai-generated" not in cap.lower():
        issues.append("caption is missing the AI-generated disclosure")
    return issues


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def too_similar(topic: str, past: list[str], threshold: float = 0.5) -> str | None:
    a = tokens(topic)
    for p in past:
        b = tokens(p)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return p
    return None
