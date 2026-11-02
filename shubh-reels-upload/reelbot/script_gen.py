"""Research, write, verify and score a reel script. Fails closed: no good script, no post.

Order of work
  1. Find what is current (research.py) and score it on the six-criterion rubric (topic.py).
  2. Read the winning article. The script may only use facts that appear in it, and every
     factual line needs a verbatim quote that is checked here in code (grounding.py).
  3. Hard safety rules (safety.py), then a reviewer scores the reel out of 100 and must
     reach 80 with accuracy of at least 16/20.
  4. If no story survives, fall back to a vetted evergreen topic written with no factual claims at all.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
import sys

from vendor.ig_skill import hookscore, humanize

from . import grounding, llm, research, safety, topic
from .config import Config

PERSONA = """You write the voice of SHUBH, an animated narrator: an intelligent friend who has just found something fascinating and wants you to have it too.
He is confident, curious and slightly witty. Short plain sentences, spoken to one person ("you"). No influencer hype, no motivational clichés, no filler.
Audience: students, young professionals, creators and entrepreneurs: smart, not technical. Explain simply without being shallow."""

STRUCTURE = """SHAPE of the reel (about 30 seconds): a stopper hook; one line on why it matters now; the discovery; three concrete steps the viewer can actually use; a payoff line with the most useful insight; then a closing line.
The viewer must leave with something tangible: a method, a workflow, a new way to see something. Never "here are five amazing tools". Reveal progressively; no repeated hook in other words."""

HARD_RULES = """HARD RULES
- Spoken text only: no emojis, no hashtags, no links, no stage directions, no em dashes, NO DIGITS (write every number as words).
- Never include: prices, plans, "free" claims, version numbers, income or money claims, guarantees, or anything about politics, religion, health, law, investing, alcohol, tobacco or drugs.
- Every sentence at most 16 words. Total spoken words (hook + body + closing line) between 55 and 80.
- The intro (Shubh introduces himself) and outro (thanks and a follow ask) are added automatically: never write a greeting, a name or a follow request.
- The closing line is a natural call to action linked to the content: save it for later, send it to someone who needs it, or a real question to the viewer. Never "follow for more"."""

SOURCE_RULES = """GROUNDING (this reel is based on a real article, given below as SOURCE)
- Use ONLY facts stated in the SOURCE. Do not add anything from memory: no extra numbers, features, names, dates or results.
- Separate fact from opinion. Do not present predictions or claims of a company as established truth: attribute them ("the company says").
- "evidence": for at least three body sentences that state something from the article, give the sentence number (1-based) and a quote of 6 to 25 words copied EXACTLY, character for character, from the SOURCE. Quotes are machine-checked; a wrong quote fails the whole script.
- A number may appear in the script only if it appears in the SOURCE.
- The three steps should turn the idea into something the viewer can do with any general AI assistant or on their own."""

EVERGREEN_RULES = """EVERGREEN MODE (no article)
- Teach a practical habit or generic AI-assistant workflow. Make no factual claims about products, studies, numbers or news: if you are not certain, leave it out.
- You may name assistants only generically ("an AI assistant like ChatGPT, Claude or Gemini")."""

SCHEMA = """OUTPUT: one JSON object with exactly these keys
{
 "topic": "short title",
 "category": "one of: %s",
 "tag": "2-3 word on-screen label in caps, e.g. AI TOOL TIP",
 "hook_options": ["three hooks, 6-12 words each, built on three different mechanisms, concrete, aimed at 'you', true to the content"],
 "hook_mechanisms": ["curiosity gap | contrarian | specific outcome | problem | open loop | pattern interrupt"],
 "hook_focus": "the single most striking word of your best hook",
 "sentences": ["6 to 8 body sentences: why it matters now, the discovery, then three clear steps, then the payoff"],
 "key_points": [{"sentence": <1-based index of a step sentence>, "label": "2-4 word step title"}],
 "takeaway": "one sentence: what the viewer can now do or understand",
 "demo": null or {"prompt": "what to type into an AI assistant, at most 14 words", "result": "what comes back, at most 22 words"},
 "emphasis": ["one word taken from each body sentence, then one from the closing line: the word to highlight on screen"],
 "evidence": [{"sentence": <1-based body sentence>, "quote": "verbatim from SOURCE"}],
 "opinions": ["any line that is interpretation, opinion or prediction rather than fact"],
 "cta_type": "save | send | question",
 "cta": "the closing line",
 "cover_text": "5 to 7 words for the cover image",
 "caption": "line one is the hook. Then one takeaway line. Then a line of 3-5 plain search keywords. Then exactly: AI-generated video.",
 "hashtags": ["#a", "#b", "#c"]
}
key_points: 3 items, one per step. hashtags: 3 to 5. Use demo only when the steps are about typing a request to an AI assistant, else null. In evergreen mode evidence is [].""" % ", ".join(topic.CATEGORIES)

REVIEW_SYSTEM = """You are a strict editor and fact-checker for a short-video brand about AI tools, productivity and career skills. Reject mediocre work.
You receive the script and, when present, the SOURCE article. Score honestly; do not inflate:
 hook /20 (stops the scroll AND is truthfully fulfilled), accuracy /20 (every factual line supported by the SOURCE; evergreen scripts make no factual claims), value /20 (the viewer leaves with something tangible and usable),
 retention /20 (every sentence earns its place, curiosity is kept, sounds spoken by a human), visual /10 (concrete things that could be shown on screen), originality /10 (not a generic tip).
Also list "unsupported_claims": any statement not backed by the SOURCE (or any factual claim at all in evergreen mode).
Reject for: medical, legal or financial advice; income promises; politics, religion, alcohol, tobacco, drugs; disparaging a person or company; claiming a tool is free or its price without proof in the SOURCE; fewer than three concrete steps; padding.
Reply with JSON only: {"hook": 0, "accuracy": 0, "value": 0, "retention": 0, "visual": 0, "originality": 0, "unsupported_claims": [], "problems": []}"""

PASS_TOTAL = 76          # normal bar
FALLBACK_TOTAL = 70      # last resort, only after every plan failed; accuracy and claim checks still apply
HOOK_FLOOR = 45
_NEAR: list = []         # scripts that passed every hard check but missed the quality total
PASS_ACCURACY = 16


# ------------------------------------------------------------------------------ topics
def pick_topic(history: list[dict], today: dt.date | None = None) -> tuple[str, str]:
    topics = json.load(open("data/topics.json"))
    today = today or dt.date.today()
    order = topics["order"]
    pillar = order[today.toordinal() % len(order)]
    past = [h.get("topic", "") for h in history]
    pool = [t for t in topics["pillars"][pillar] if not safety.too_similar(t, past[-30:], 0.6)]
    rng = random.Random(today.toordinal())
    return pillar, rng.choice(pool or topics["pillars"][pillar])


def clean_text(text: str, lex) -> str:
    out, _ = humanize.humanize(text, lex)
    out = out.strip()
    out = re.sub(r"[\U00010000-\U0010ffff☀-➿]", "", out)      # emoji
    return re.sub(r"\s+", " ", out)


def _word_in(word: str, line: str) -> str | None:
    w = re.sub(r"[^A-Za-z']", "", word or "")
    if len(w) >= 3 and re.search(r"\b" + re.escape(w) + r"\b", line, re.I):
        return w
    return None


def finalize(raw: dict, plan: dict | None = None) -> dict:
    lex = humanize.load_lexicon()
    opts = [clean_text(h, lex) for h in raw.get("hook_options", []) if h]
    mech = list(raw.get("hook_mechanisms", []))
    scored = []
    for i, h in enumerate(opts):
        _res, overall, verdict, flags = hookscore.run(h)
        scored.append((overall, h, i))
    if not scored:
        raise ValueError("no hooks")
    scored.sort(reverse=True)
    best = scored[0]
    hook = best[1]
    body = [clean_text(s, lex) for s in raw.get("sentences", [])]
    cta = clean_text(raw.get("cta", "Save this for your next work session."), lex)
    emph_raw = list(raw.get("emphasis", []))
    emph = [_word_in(raw.get("hook_focus", ""), hook)]
    for i, line in enumerate(body + [cta]):
        emph.append(_word_in(emph_raw[i], line) if i < len(emph_raw) else None)
    cap = humanize.humanize(raw.get("caption", ""), lex)[0].strip()
    if "ai-generated" not in cap.lower():
        cap = (cap + "\nAI-generated video.").strip()
    demo = raw.get("demo")
    if isinstance(demo, dict) and demo.get("prompt") and demo.get("result"):
        demo = {"prompt": clean_text(str(demo["prompt"]), lex), "result": clean_text(str(demo["result"]), lex)}
        if len(demo["prompt"].split()) > 16 or len(demo["result"].split()) > 26 or re.search(r"\d", json.dumps(demo)):
            demo = None
    else:
        demo = None
    cover = clean_text(str(raw.get("cover_text", "")), lex).strip(" .!")
    if not (3 <= len(cover.split()) <= 8):
        cover = " ".join(hook.rstrip(".?!").split()[:7])
    return {
        "topic": clean_text(raw.get("topic", ""), lex),
        "category": raw.get("category") if raw.get("category") in topic.CATEGORIES else None,
        "tag": re.sub(r"[^A-Z0-9 +&]", "", str(raw.get("tag", "AI TOOL TIP")).upper())[:24] or "AI TOOL TIP",
        "hook": hook,
        "hook_score": round(best[0], 1),
        "hook_mechanism": mech[best[2]] if best[2] < len(mech) else "",
        "hook_focus": emph[0] or "",
        "sentences": body,
        "key_points": [{"sentence": int(k["sentence"]), "label": clean_text(str(k["label"]), lex)}
                       for k in raw.get("key_points", [])][:3],
        "takeaway": clean_text(str(raw.get("takeaway", "")), lex),
        "demo": demo,
        "emphasis": emph,
        "evidence": [e for e in raw.get("evidence", []) if isinstance(e, dict)],
        "opinions": [clean_text(str(o), lex) for o in raw.get("opinions", [])][:6],
        "cta_type": raw.get("cta_type", ""),
        "cta": cta,
        "cover_text": cover,
        "caption": cap,
        "hashtags": [("#" + re.sub(r"[^A-Za-z0-9]", "", h)) for h in raw.get("hashtags", [])][:5],
    }


# ------------------------------------------------------------------------------ plans
def build_plans(cfg: Config, history: list[dict], today: dt.date, log: dict) -> list[dict]:
    """Source-grounded plans first (best story with a readable article), evergreen last."""
    plans: list[dict] = []
    try:
        cands = research.gather()
    except Exception as e:  # noqa: BLE001
        print(f"[research] gather failed: {type(e).__name__}", file=sys.stderr)
        cands = []
    log["candidates_seen"] = len(cands)
    ranked = topic.choose(cfg, cands, history, today) if cands else []
    log["shortlist"] = [{"title": r["candidate"].title, "publisher": r["candidate"].publisher, "url": r["candidate"].url,
                         "category": r["category"], "total": r["total"], "scores": r["scores"]} for r in ranked[:10]]
    for r in ranked[:4]:
        art = research.fetch_article(r["candidate"].url)
        if not art:
            continue
        plans.append({"mode": "source", "cand": r["candidate"], "category": r["category"], "angle": r["angle"],
                      "scores": r["scores"], "total": r["total"], "excerpt": art["text"]})
        if len(plans) == 3:
            break
    pillar, seed = pick_topic(history, today)
    plans.append({"mode": "evergreen", "pillar": pillar, "seed": seed, "category": topic.evergreen_category(history, today)})
    return plans


def writer_prompt(plan: dict, history: list[dict], problems: list[str], today: dt.date) -> tuple[str, str]:
    past_topics = [h.get("topic", "") for h in history][-40:]
    past_hooks = [h.get("hook", "") for h in history][-12:]
    avoid = (f"Topics already covered (do not repeat): {json.dumps(past_topics)}\n"
             f"Recent hooks (do not reuse their wording or structure): {json.dumps(past_hooks)}\n")
    fix = f"Fix these problems from the last attempt: {problems[-6:]}\n" if problems else ""
    if plan["mode"] == "source":
        c = plan["cand"]
        system = "\n\n".join([PERSONA, STRUCTURE, SOURCE_RULES, HARD_RULES, SCHEMA])
        user = (f"Today: {today.isoformat()}\nCategory to use: {plan['category']}\nAngle: {plan['angle']}\n"
                f"SOURCE title: {c.title}\nSOURCE publisher: {c.publisher} (published {c.published})\n"
                f"SOURCE text:\n\"\"\"\n{plan['excerpt']}\n\"\"\"\n{avoid}{fix}")
    else:
        system = "\n\n".join([PERSONA, STRUCTURE, EVERGREEN_RULES, HARD_RULES, SCHEMA])
        user = (f"Today: {today.isoformat()}\nPillar: {plan['pillar']}\nCategory to use: {plan['category']}\n"
                f"Angle to teach today: {plan['seed']}\n{avoid}{fix}")
    return system, user


def rubric_total(r: dict) -> tuple[int, dict]:
    caps = {"hook": 20, "accuracy": 20, "value": 20, "retention": 20, "visual": 10, "originality": 10}
    got = {k: max(0, min(cap, int(float(r.get(k, 0))))) for k, cap in caps.items()}
    return sum(got.values()), got


def attempt(cfg: Config, plan: dict, history: list[dict], problems: list[str], today: dt.date) -> tuple[dict | None, list[str]]:
    system, user = writer_prompt(plan, history, problems, today)
    try:
        raw = llm.chat_json(cfg, system, user, temperature=0.75)
        script = finalize(raw, plan)
    except Exception as e:  # noqa: BLE001
        print(f"generation error: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
        return None, [f"generation error: {type(e).__name__}"]
    source = plan["mode"] == "source"
    issues = safety.check_script(script, grounded=source)
    past = [h.get("topic", "") for h in history][-40:]
    dup = safety.too_similar(script["topic"], past)
    if dup:
        issues.append(f"too similar to an earlier topic: {dup}")
    if script["hook_score"] < HOOK_FLOOR:
        issues.append(f"best hook only scores {script['hook_score']} (need {HOOK_FLOOR}+)")
    verified = []
    if source:
        verified, g_issues = grounding.verify(script, plan["excerpt"])
        issues += g_issues
    if issues:
        return None, issues
    review_input = {"script": {k: script[k] for k in ("hook", "sentences", "cta", "takeaway", "key_points", "demo", "opinions")},
                    "mode": plan["mode"]}
    if source:
        review_input["SOURCE"] = plan["excerpt"]
        review_input["verified_evidence"] = verified
    try:
        review = llm.chat_json(cfg, REVIEW_SYSTEM, json.dumps(review_input), temperature=0.0)
    except Exception as e:  # noqa: BLE001 - reviewer unavailable: fail closed
        return None, [f"reviewer unavailable ({type(e).__name__})"]
    total, parts = rubric_total(review)
    problems_r = [str(p) for p in review.get("problems", [])][:4] + \
                 [f"unsupported: {u}" for u in review.get("unsupported_claims", [])][:4]
    if review.get("unsupported_claims"):
        return None, problems_r or ["unsupported claims"]
    if total < PASS_TOTAL and total >= FALLBACK_TOTAL and parts["accuracy"] >= PASS_ACCURACY:
        script["quality"] = {"total": total, **parts}
        script["evidence_verified"] = verified
        _NEAR.append((total, script, plan))
    if total < PASS_TOTAL or parts["accuracy"] < PASS_ACCURACY:
        return None, [f"quality {total}/100 (hook {parts['hook']}, accuracy {parts['accuracy']}, value {parts['value']}, "
                      f"retention {parts['retention']}) is below the bar of {PASS_TOTAL}"] + problems_r
    script["quality"] = {"total": total, **parts}
    script["evidence_verified"] = verified
    return script, []


def finish(script: dict, plan: dict, log: dict) -> dict:
    script["pillar"] = plan.get("pillar") or plan["category"]
    script["category"] = script.get("category") or plan["category"]
    script["mode"] = plan["mode"]
    if plan["mode"] == "source":
        c = plan["cand"]
        script["source"] = {"title": c.title, "publisher": c.publisher, "url": c.url, "published": c.published}
        script["topic_scores"] = {"scores": plan["scores"], "total": plan["total"]}
    script["research"] = {**log, "mode": plan["mode"], "rubric": script["quality"],
                          "fact_check": fact_check(script), "opinions": script["opinions"]}
    return script


def generate(cfg: Config, history: list[dict], today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    log: dict = {"date": today.isoformat()}
    plans = build_plans(cfg, history, today, log)
    for plan in plans:
        tries = 3
        problems: list[str] = []
        label = plan["cand"].title if plan["mode"] == "source" else plan["seed"]
        for n in range(tries):
            print(f"[{plan['mode']}] attempt {n + 1}: {label[:80]}", file=sys.stderr)
            script, issues = attempt(cfg, plan, history, problems, today)
            if script:
                return finish(script, plan, log)
            problems += issues
            print(f"  rejected: {issues[:4]}", file=sys.stderr)
    if _NEAR:
        total, script, plan = max(_NEAR, key=lambda t: t[0])
        print(f"using best near-miss script ({total}/100): every safety and accuracy check passed", file=sys.stderr)
        return finish(script, plan, log)
    raise RuntimeError("no script passed research, safety and quality checks; skipping today's post")


def fact_check(script: dict) -> list[dict]:
    """Structured record for later review: each supported line, its quote and where it came from."""
    out = []
    src = script.get("source", {})
    for e in script.get("evidence_verified", []):
        out.append({"line": script["sentences"][e["sentence"] - 1], "quote": e["quote"],
                    "publisher": src.get("publisher"), "url": src.get("url"), "status": "supported by source"})
    for o in script.get("opinions", []):
        out.append({"line": o, "status": "opinion or prediction, attributed"})
    return out


def build_caption(script: dict) -> str:
    cap = script["caption"].strip()
    src = script.get("source")
    if src and "source:" not in cap.lower():
        cap = cap.replace("AI-generated video.", f"Source: {src['publisher']}\nAI-generated video.") \
            if "AI-generated video." in cap else cap + f"\nSource: {src['publisher']}"
    tags = " ".join(script["hashtags"][:5])
    return f"{cap}\n\n{tags}".strip()
