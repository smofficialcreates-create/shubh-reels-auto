"""Named series: every reel belongs to one, with an episode number, so viewers learn what to expect."""
from __future__ import annotations

SERIES = {
    "practical_ai_tutorials": "PROMPT FIX", "step_by_step_workflows": "PROMPT FIX",
    "ai_discoveries": "AI DECODED", "underrated_tools": "AI DECODED", "research_findings": "AI DECODED",
    "productivity_methods": "FOCUS LAB", "psychology_learning": "BRAIN LAB",
    "career_skills": "CAREER MOVE", "business_insights": "CAREER MOVE", "myths_vs_reality": "MYTH CHECK",
}
DEFAULT = "AI DECODED"


def assign(script: dict, history: list[dict]) -> dict:
    name = SERIES.get(script.get("category") or "", DEFAULT)
    n = 1 + sum(1 for h in history if h.get("series") == name)
    script["series"], script["episode"] = name, n
    script["tag"] = name
    return script


def label(script: dict) -> str:
    return f"{script['series']} #{script['episode']:02d}" if script.get("series") else ""
