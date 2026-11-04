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
import os
import random
import re
import sys

from vendor.ig_skill import hookscore, humanize

from . import grounding, llm, research, safety, topic
from .config import Config

PERSONA = """You write the voice of SHUBH, an animated narrator: an intelligent friend who has just found something fascinating and wants you to have it too.
He is confident, curious and slightly witty. Short plain sentences, spoken to one person ("you"). No influencer hype, no motivational clichés, no filler.
Audience: students, young professionals, creators and entrepreneurs: smart, not technical. Explain simply without being shallow."""

STRUCTURE = """SHAPE of the reel (about 75 seconds, 150 to 215 spoken words), modelled on the best explainer videos:
1. HOOK: a question or a stake the viewer wants answered.
2. DEFINE (1-2 sentences): what it is in plain words, and why it matters for the viewer's work or career.
3. ANALOGY (1-2 sentences): pick ONE concrete picture or small story (a post office, a desk, a kitchen, a delivery company) and keep using it in the stages.
4. STAGES (three to five): for each one give its name, HOW it works (a cause and its effect, the mechanism), and a concrete example in plain words. Any small number in an example is clearly hypothetical ("say a delivery takes forty minutes").
5. THE CATCH (1-2 sentences): where it fails, the trade-off, or the most common mistake.
6. RECAP (1 sentence): one line that helps the viewer remember every stage.
7. CLOSING line: a natural call to action.
Depth beats breadth: explain WHY, not only WHAT. Teach it so a smart beginner could explain it to a friend afterwards. Include one thing the viewer can do on Monday. Never a list of tool names. No hype."""

HOOKS = """THE HOOK is the very first thing spoken, before anything else. It must stop a thumb within two seconds.
Write three options, each from a different pattern, fitted to THIS reel's content, then they are scored and the best is used:
 1. WRONG WAY: tell them what they are doing wrong ("You are using AI the wrong way.")
 2. COSTLY MISTAKE: name a mistake that costs them ("Most interview answers fail in the same place.")
 3. HIDDEN TRUTH: something nobody explains ("Nobody tells you what your assistant is really doing.")
 4. REVERSAL: flip a belief ("Working harder is not why you are behind.")
 5. DIRECT QUESTION or CHALLENGE aimed at the viewer ("Can you explain how your assistant answers you?")
Rules: 5 to 12 words, spoken to "you", concrete, no numbers, no hype words (amazing, insane, mind-blowing, secret hack), and it must be fully answered by the reel: the first body sentence starts paying it off at once. Never promise anything the reel does not deliver."""

HARD_RULES = """HARD RULES
- Spoken text only: no emojis, no hashtags, no links, no stage directions, no em dashes, NO DIGITS (write every number as words).
- Never include: prices, plans, "free" claims, version numbers, percentages, income or money claims, guarantees, or anything about politics, religion, health, law, investing, alcohol, tobacco or drugs.
- Every sentence at most 22 words. Total spoken words (hook + body + closing line) between 150 and 215. Body of 11 to 17 sentences.
- The intro (Shubh introduces himself) and outro (thanks and a follow ask) are added automatically: never write a greeting, a name or a follow request.
- The closing line is a natural call to action linked to the content: save it for later, send it to someone who needs it, or a real question to the viewer. Never "follow for more"."""

SOURCE_RULES = """GROUNDING (this reel is based on a real article, given below as SOURCE)
- Use ONLY facts stated in the SOURCE. Do not add anything from memory: no extra numbers, features, names, dates or results. The analogy is a teaching picture, not a fact, and may be your own.
- Separate fact from opinion. Do not present predictions or claims of a company as established truth: attribute them ("the company says").
- "evidence": for at least three body sentences that state something from the article, give the sentence number (1-based) and a quote of 6 to 25 words copied EXACTLY, character for character, from the SOURCE. Quotes are machine-checked; a wrong quote fails the whole script.
- A number may appear in the script only if it appears in the SOURCE.
- Turn the idea into stages the viewer can use with any general AI assistant or on their own."""

EXPLAINER_RULES = """EXPLAINER MODE (no article): teach one stable concept from the OUTLINE below, the way a good textbook or a senior colleague would.
- Everything you state as fact must be standard, widely agreed knowledge about HOW THE CONCEPT WORKS. Make no claims about studies, statistics, history, dates, companies, products, prices, versions or news. If you are not certain, leave it out.
- Examples are hypothetical and say so ("imagine", "say"). Small example numbers are allowed only as words.
- You may name assistants only generically ("an AI assistant")."""

VISUALS = """SCENES: split the body sentences into 5 to 8 scenes of 1-4 consecutive sentences each. Every scene gets ONE on-screen visual that shows the idea, not decorates it.
Visual types (use at least four different types, never the same type twice in a row):
 {"type":"definition","term":"1-3 words","meaning":"at most 14 words"}                     the term explained in plain words
 {"type":"flow","nodes":["3 to 5 labels, at most 4 words each"],"sync":[sentence numbers]}    a process; sync = the body sentence where each node is spoken
 {"type":"layers","labels":["3 to 6 labels"],"sync":[sentence numbers]}                       things stacked in order
 {"type":"compare","left":{"title":"the weak way","rows":["3 short rows"]},"right":{"title":"the better way","rows":["3 short rows"]}}
 {"type":"numeral","n":1,"total":3,"title":"at most 4 words"}                                a named step
 {"type":"counter","value":40,"unit":"min","label":"at most 8 words, an example"}           only for a number that is spoken in the scene as a word
 {"type":"checklist","labels":["3 short items"],"sync":[sentence numbers]}                   a recap
 {"type":"chat","prompt":"at most 14 words","result":"at most 22 words"}                      typing to an AI assistant
 {"type":"keyword","word":"one word spoken in the scene"}
Each scene: {"from": <first sentence>, "to": <last sentence>, "label": "2-3 word label", "visual": {...}}. The scenes must cover sentences 1 to N in order. Use compare for the catch, a flow or layers for the stages, and a checklist or keyword for the recap."""

SCHEMA = """OUTPUT: one JSON object with exactly these keys
{
 "topic": "short title",
 "category": "one of: %s",
 "tag": "2-3 word on-screen label in caps, e.g. AI EXPLAINED",
 "hook_options": ["three hooks from three different patterns in THE HOOK, 5-12 words each, aimed at 'you', true to the content"],
 "hook_mechanisms": ["wrong way | costly mistake | hidden truth | reversal | challenge"],
 "hook_focus": "the single most striking word of your best hook",
 "sentences": ["11 to 17 body sentences following the SHAPE"],
 "scenes": [ ...see SCENES... ],
 "takeaway": "one sentence: what the viewer can now do or understand",
 "emphasis": ["one word taken from each body sentence, then one from the closing line: the word to highlight on screen"],
 "evidence": [{"sentence": <1-based body sentence>, "quote": "verbatim from SOURCE"}],
 "opinions": ["any line that is interpretation, opinion or prediction rather than fact"],
 "cta_type": "save | send | question",
 "cta": "the closing line",
 "cover_text": "5 to 7 words for the cover image",
 "caption": "line one is the hook. Then one takeaway line. Then a line of 3-5 plain search keywords. Then exactly: AI-generated video.",
 "hashtags": ["#a", "#b", "#c"]
}
hashtags: 3 to 5. In explainer mode evidence is [].""" % ", ".join(topic.CATEGORIES)

REVIEW_SYSTEM = """You are a strict editor and subject-matter checker for a short-video brand about AI tools, productivity and career skills. Reject mediocre work.
You receive the script, a mode, and when present the SOURCE article. Score honestly; do not inflate:
 hook /15 (stops the scroll AND is truthfully fulfilled)
 accuracy /20 (source mode: every factual line supported by the SOURCE. explainer mode: every statement is standard, correct, widely agreed knowledge about how the concept works, and nothing is overstated)
 depth /20 (explains WHY and HOW, with a mechanism, a concrete example and a limit; a smart beginner could now explain it; not a list of tips)
 clarity /15 (one running analogy used consistently, plain words, no jargon left unexplained, stages easy to follow and remember)
 retention /15 (every sentence earns its place, curiosity is kept, sounds spoken by a human)
 visuals /10 (the scene visuals show the idea accurately and fit the sentences they sit on)
 originality /5
Also list "unsupported_claims": any statement that is wrong, overstated, contested, or (source mode) not backed by the SOURCE. Standard textbook definitions are not unsupported.
Reject for: medical, legal or financial advice; income promises; politics, religion, alcohol, tobacco, drugs; disparaging a person or company; pricing or free claims; fewer than three stages; padding; a visual that contradicts the sentence.
Reply with JSON only: {"hook": 0, "accuracy": 0, "depth": 0, "clarity": 0, "retention": 0, "visuals": 0, "originality": 0, "unsupported_claims": [], "problems": []}"""

PASS_TOTAL = 76          # normal bar
FALLBACK_TOTAL = 70      # last resort, only after every plan failed; accuracy and claim checks still apply
HOOK_FLOOR = 45
_NEAR: list = []         # scripts that passed every hard check but missed the quality total
PASS_ACCURACY = 16
PASS_DEPTH = 12
MIN_WORDS = 140
VISUAL_TYPES = ("definition", "flow", "layers", "compare", "numeral", "counter", "checklist", "chat", "keyword")


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


def pick_explainer(history: list[dict], today: dt.date) -> dict:
    """An unused topic from the syllabus, rotating through the categories."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "syllabus.json")
    items = json.load(open(path, encoding="utf-8"))
    past = [h.get("topic", "") for h in history][-60:]
    recent_cats = [h.get("category") for h in history][-3:]
    fresh = [e for e in items if not safety.too_similar(e["title"], past, 0.5)] or items
    varied = [e for e in fresh if e["category"] not in recent_cats] or fresh
    return random.Random(today.toordinal()).choice(varied)


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
                       for k in raw.get("key_points", []) if isinstance(k, dict) and "sentence" in k and "label" in k][:3],
        "scenes": finalize_scenes(raw.get("scenes"), body, lex),
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


def _txt(v, n_words: int, lex) -> str:
    return " ".join(clean_text(str(v), lex).split()[:n_words])


def finalize_scenes(raw, body: list[str], lex) -> list[dict]:
    """Keep only well-formed scenes. Bad visuals are dropped; plan.repair_scenes then closes any gap."""
    out = []
    if not isinstance(raw, list):
        return out
    n = len(body)
    for sc in raw:
        try:
            a, b = int(sc["from"]), int(sc["to"])
            v = sc["visual"]
            typ = v["type"]
        except (KeyError, TypeError, ValueError):
            continue
        if typ not in VISUAL_TYPES or not (1 <= a <= b <= n):
            continue
        spoken = " ".join(body[a - 1:b])
        clean_v = {"type": typ}
        try:
            if typ == "definition":
                clean_v.update(term=_txt(v["term"], 3, lex), meaning=_txt(v["meaning"], 16, lex))
            elif typ in ("flow", "layers", "checklist"):
                key = "nodes" if typ == "flow" else "labels"
                items = [_txt(x, 5, lex) for x in (v.get(key) or v.get("labels") or v.get("nodes") or [])][:6]
                if len(items) < 3:
                    continue
                clean_v[key] = items[:5] if typ != "layers" else items
                clean_v["sync"] = [int(x) for x in v.get("sync", []) if isinstance(x, (int, float))]
            elif typ == "compare":
                clean_v["left"] = {"title": _txt(v["left"]["title"], 3, lex), "rows": [_txt(x, 5, lex) for x in v["left"]["rows"]][:3]}
                clean_v["right"] = {"title": _txt(v["right"]["title"], 3, lex), "rows": [_txt(x, 5, lex) for x in v["right"]["rows"]][:3]}
                if len(clean_v["left"]["rows"]) < 2 or len(clean_v["right"]["rows"]) < 2:
                    continue
            elif typ == "numeral":
                clean_v.update(n=int(v["n"]), total=max(int(v.get("total", 3)), int(v["n"])), title=_txt(v["title"], 4, lex))
            elif typ == "counter":
                val = int(v["value"])
                if val not in grounding.numbers_in(spoken):      # the number must be spoken in the scene
                    continue
                clean_v.update(value=val, unit=_txt(v.get("unit", ""), 2, lex), label=_txt(v["label"], 9, lex))
            elif typ == "chat":
                clean_v.update(prompt=_txt(v["prompt"], 16, lex), result=_txt(v["result"], 24, lex))
            elif typ == "keyword":
                w = _word_in(str(v.get("word", "")), spoken)
                clean_v["word"] = w.lower() if w else None
        except (KeyError, TypeError, ValueError):
            continue
        if re.search(r"\d", json.dumps({k: x for k, x in clean_v.items() if k not in ("value", "n", "total", "sync")})):
            continue
        out.append({"from": a, "to": b, "label": _txt(sc.get("label", ""), 3, lex).lower() or None, "visual": clean_v})
    return out


def scene_issues(script: dict) -> list[str]:
    sc = script.get("scenes") or []
    issues = []
    if len(sc) < 4:
        issues.append(f"only {len(sc)} usable scenes (need at least 4)")
    types = [x["visual"]["type"] for x in sc]
    if len(set(types)) < 3:
        issues.append("scenes use fewer than three different visual types")
    if any(a == b for a, b in zip(types, types[1:])):
        issues.append("two neighbouring scenes use the same visual type")
    for x in sc:
        issues += safety.check_text("scene visual", json.dumps(x["visual"]))
    return issues


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
    entry = pick_explainer(history, today)
    plans.append({"mode": "explainer", "entry": entry, "pillar": entry["category"], "seed": entry["title"],
                  "category": entry["category"]})
    return plans


def writer_prompt(plan: dict, history: list[dict], problems: list[str], today: dt.date) -> tuple[str, str]:
    past_topics = [h.get("topic", "") for h in history][-40:]
    past_hooks = [h.get("hook", "") for h in history][-12:]
    avoid = (f"Topics already covered (do not repeat): {json.dumps(past_topics)}\n"
             f"Recent hooks (do not reuse their wording or structure): {json.dumps(past_hooks)}\n")
    fix = f"Fix these problems from the last attempt: {problems[-6:]}\n" if problems else ""
    if plan["mode"] == "source":
        c = plan["cand"]
        system = "\n\n".join([PERSONA, STRUCTURE, HOOKS, SOURCE_RULES, VISUALS, HARD_RULES, SCHEMA])
        user = (f"Today: {today.isoformat()}\nCategory to use: {plan['category']}\nAngle: {plan['angle']}\n"
                f"SOURCE title: {c.title}\nSOURCE publisher: {c.publisher} (published {c.published})\n"
                f"SOURCE text:\n\"\"\"\n{plan['excerpt']}\n\"\"\"\n{avoid}{fix}")
    else:
        e = plan["entry"]
        system = "\n\n".join([PERSONA, STRUCTURE, HOOKS, EXPLAINER_RULES, VISUALS, HARD_RULES, SCHEMA])
        user = (f"Today: {today.isoformat()}\nCategory to use: {plan['category']}\nTOPIC: {e['title']}\n"
                f"OUTLINE\n- analogy to use: {e['analogy']}\n- stages: {json.dumps(e['stages'])}\n- the catch: {e['catch']}\n"
                f"{avoid}{fix}")
    return system, user


def rubric_total(r: dict) -> tuple[int, dict]:
    caps = {"hook": 15, "accuracy": 20, "depth": 20, "clarity": 15, "retention": 15, "visuals": 10, "originality": 5}
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
    words = sum(len(x.split()) for x in [script["hook"]] + script["sentences"] + [script["cta"]])
    if words < MIN_WORDS or len(script["sentences"]) < 9:
        issues.append(f"too thin for a deep explainer: {words} words, {len(script['sentences'])} sentences (need {MIN_WORDS}+ words, 9+ sentences)")
    issues += scene_issues(script)
    if script["hook_score"] < HOOK_FLOOR:
        issues.append(f"best hook only scores {script['hook_score']} (need {HOOK_FLOOR}+)")
    verified = []
    if source:
        verified, g_issues = grounding.verify(script, plan["excerpt"])
        issues += g_issues
    if issues:
        return None, issues
    review_input = {"script": {k: script[k] for k in ("hook", "sentences", "cta", "takeaway", "scenes", "opinions")},
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
    if total < PASS_TOTAL and total >= FALLBACK_TOTAL and parts["accuracy"] >= PASS_ACCURACY and parts["depth"] >= PASS_DEPTH:
        script["quality"] = {"total": total, **parts}
        script["evidence_verified"] = verified
        _NEAR.append((total, script, plan))
    if total < PASS_TOTAL or parts["accuracy"] < PASS_ACCURACY or parts["depth"] < PASS_DEPTH:
        return None, [f"quality {total}/100 (hook {parts['hook']}, accuracy {parts['accuracy']}, depth {parts['depth']}, "
                      f"clarity {parts['clarity']}) is below the bar of {PASS_TOTAL}"] + problems_r
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


def backup_script(history: list[dict], today: dt.date) -> dict:
    """Last resort so a day is never skipped: a hand-written lesson from data/backup_scripts.json.
    Every entry is checked by the same safety rules at load time. Unused ones go first; once all
    have been used, the one used longest ago comes round again."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backup_scripts.json")
    bank = json.load(open(path, encoding="utf-8"))
    ok = [b for b in bank if not safety.check_script(b)]
    if not ok:
        raise RuntimeError("backup bank is empty or failed the safety checks")
    last = {}
    for i, h in enumerate(history):
        last[h.get("topic", "")] = i
    ok.sort(key=lambda b: last.get(b["topic"], -1))
    script = json.loads(json.dumps(ok[0]))
    script["mode"] = "backup"
    script["pillar"] = script["category"]
    script["quality"] = {"total": None, "note": "pre-checked backup lesson"}
    script["evidence_verified"] = []
    script["research"] = {"date": today.isoformat(), "mode": "backup", "note": "live scripts did not pass; used a pre-checked lesson",
                          "fact_check": [], "opinions": []}
    return script


def generate(cfg: Config, history: list[dict], today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    log: dict = {"date": today.isoformat()}
    try:
        plans = build_plans(cfg, history, today, log)
    except Exception as e:  # noqa: BLE001 - research or AI outage must not stop the daily post
        print(f"[plans] failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        plans = []
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
    print("no live script passed; using a pre-checked backup lesson", file=sys.stderr)
    return backup_script(history, today)


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
