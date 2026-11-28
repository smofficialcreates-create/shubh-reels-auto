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

from . import grounding, llm, research, safety, series, topic
from . import engage
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
 6. NOBODY MENTIONS THIS: an open secret nobody spells out ("Nobody mentions this about AI prompts.", "Nobody tells you this about your to do list.")
 7. JUST FIGURED IT OUT: a personal discovery ("I just figured out why my AI answers were vague.")
 9. RESULT FIRST: invite the viewer to watch the outcome ("Watch a weak prompt turn into a sharp one.", "Watch AI fix this resume line.") - only when the first scene really shows that outcome; never promise jobs, money or grades.
 8. MILD OPINION: a bold but defensible take about a method or habit, never about people, groups, companies or politics ("Most AI advice is backwards.", "Stop asking AI for the answer.")
At least one of the three options should use pattern 6, 7, 8 or 9, and they are preferred when they fit the content honestly. A mild opinion must be something the reel then explains and supports.
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
- "evidence": for at least two body sentences that state something from the article, give the sentence number (1-based) and a quote of 6 to 25 words copied EXACTLY, character for character, from the SOURCE. Quotes are machine-checked; a wrong quote fails the whole script.
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
 "hook_mechanisms": ["wrong way | costly mistake | hidden truth | reversal | challenge | nobody mentions | just figured out | opinion | result first"],
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
LANGUAGE "Hinglish": judge it as natural, correct, friendly Hinglish, and check that "speech" (Devanagari) says the same thing as the Hinglish lines, line by line, with nothing mistranslated, unsafe or changed in meaning.
FORMAT "short" (about 35 seconds): judge depth as one clear mechanism, one concrete demo and one limit; do NOT reject it for having fewer than three stages. Reject a short reel whose closing line is only a request to save or follow, or whose demo is not useful tomorrow.
Reject for: medical, legal or financial advice; income promises; politics, religion, alcohol, tobacco, drugs; disparaging a person or company; pricing or free claims; fewer than three stages; padding; a visual that contradicts the sentence.
Reply with JSON only: {"hook": 0, "accuracy": 0, "depth": 0, "clarity": 0, "retention": 0, "visuals": 0, "originality": 0, "unsupported_claims": [], "problems": []}"""

SHORT_STRUCTURE = """SHAPE of the reel (about 35 seconds, 70 to 100 spoken words in hook + body + closing line). RESULTS FIRST: show the outcome before explaining it. ONE idea, one mechanism, no recap.
1. HOOK (first words spoken, see THE HOOK). When it fits, use the RESULT FIRST pattern.
2. THE RESULT, body sentences 1 and 2, straight after the hook and covered by the on-screen chat demo: say what the viewer is looking at, for example "Here is a weak resume line. Here is the same line after one better prompt." No greeting, no setup, no background before it.
3. THE PROMPT, one or two sentences: the better prompt in plain words, so the viewer can copy the idea.
4. WHY IT WORKS, one or two sentences: the single reason, in plain words.
5. THE CATCH, one short sentence: when it does not work or what to check.
6. CLOSING LINE: follows the CLOSING STYLE given in the task. Never "save this" and never "follow".
The viewer must be able to use it tomorrow with a free general AI assistant or on paper. Audience: students and early-career people who do NOT code. Skip developer-only tools and topics (code, repositories, servers, APIs)."""

SHORT_RULES = """HARD RULES (short reel)
- Spoken text only: no emojis, no hashtags, no links, no stage directions, no em dashes, NO DIGITS (numbers as words).
- Never include: prices, plans, "free" claims, version numbers, percentages, income or money claims, guarantees, or anything about politics, religion, health, law, investing, alcohol, tobacco or drugs.
- Every sentence at most 18 words. Hook + body + closing line: 70 to 100 words in total. Body of 5 to 8 sentences.
- Do NOT write a greeting, a name, "I'm Shubh" or a follow request: the brand line is added automatically at the end.
- The closing line follows the CLOSING STYLE given in the task, tied to the content, one sentence, spoken, no digits (numbers as words).
- "loop_line": 3 to 8 words, a lead-in clause that is spoken as the very last words and flows grammatically into the HOOK when the video restarts (for example "And that is exactly why" then the hook "Nobody mentions this."). No question mark, no digits."""

SHORT_VISUALS = """SCENES: split the body sentences into 3 to 5 scenes of 1-3 consecutive sentences each, ONE on-screen visual each. Use at least two different types, never the same type twice in a row, and ALWAYS exactly one "chat" scene for the demo (the better prompt and the answer it gets), and it MUST be the FIRST scene, starting at sentence 1.
Types: {"type":"chat","prompt":"at most 14 words","result":"at most 22 words"}
 {"type":"compare","left":{"title":"the weak way","rows":["2-3 short rows"]},"right":{"title":"the better way","rows":["2-3 short rows"]}}
 {"type":"definition","term":"1-3 words","meaning":"at most 14 words"}
 {"type":"flow","nodes":["3 to 4 labels"],"sync":[sentence numbers]}   {"type":"checklist","labels":["3 short items"],"sync":[sentence numbers]}
 {"type":"keyword","word":"one word spoken in the scene"}
Each scene: {"from": <first sentence>, "to": <last sentence>, "label": "2-3 word label", "visual": {...}}. Scenes cover sentences 1 to N in order."""

SHORT_SCHEMA = SCHEMA.replace("11 to 17 body sentences following the SHAPE", "5 to 8 body sentences following the SHAPE") \
    .replace('"cta": "the closing line",', '"cta": "ONE question for the comments",\n "first_comment": "a one-line prompt (at most 120 characters, no digits, no links) the account posts as the first comment to start a conversation",') \
    .replace('"cta_type": "save | send | question"', '"cta_type": "the CLOSING STYLE name"') \
    .replace(' "cover_text"', ' "keyword": "ONLY for the keyword style: one of PROMPT, STEPS, GUIDE, CHECKLIST, TEMPLATE, else empty",\n "reward_text": "ONLY for the keyword style: the message sent to commenters, else empty",\n "cover_text"', 1) \
    .replace(' "cover_text"', ' "loop_line": "3 to 8 word lead-in that flows into the hook when the video restarts",\n "cover_text"', 1)

HI_BLOCK = """LANGUAGE: HINGLISH. Write ALL on-screen and caption text (hook_options, sentences, scene labels and rows, cta, loop_line, caption, cover_text, takeaway, first_comment) in natural, friendly Hinglish: Hindi grammar and everyday Hindi words written in the ENGLISH ALPHABET, with common English words kept as they are (AI, prompt, resume, interview, chatbot). Short and conversational, like a helpful senior talking to a college student; address the viewer as "aap". The chat demo prompt and its answer stay in simple English, because that is what people type to an assistant.
ALSO return "speech": the SAME lines written in DEVANAGARI script for the voice, line by line and word for word parallel to the Hinglish (same meaning, same order), English words spelled phonetically in Devanagari (for example एआई, प्रॉम्प्ट, रिज़्यूमे, इंटरव्यू, चैटबॉट). Shape: {"hook_options": [one per hook option, same order], "sentences": [one per body sentence, same count], "cta": "...", "loop_line": "..."}. No digits in any script (numbers as words). Every rule above (closing style, safety, length) applies to the Hinglish lines."""
HI_BLOCK_RE = [(re.compile(r"\b(muft|mufat|guarantee[d]?|pakka|kamai|kamayi|lakh|crore|paisa kamao)\b", re.I), "pricing, income or guarantee claim (Hinglish)"),
               (re.compile(r"मुफ़्त|मुफ्त|फ्री|गारंटी|कमाई|लाख|करोड़|पैसे कमा"), "pricing, income or guarantee claim (Hindi)")]
DEVANAGARI = re.compile(r"[\u0900-\u097F]")

PASS_TOTAL = 76          # normal bar
FALLBACK_TOTAL = 70      # last resort, only after every plan failed; accuracy and claim checks still apply
HOOK_FLOOR = 45
_NEAR: list = []         # scripts that passed every hard check but missed the quality total
PASS_ACCURACY = 16
PASS_DEPTH = 12
MIN_WORDS = 140
SHORT_WORDS = (70, 105)
SHORT_DEPTH = 10
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


def load_strategy(path: str | None = None) -> dict:
    """What the weekly growth manager learned. Empty (no effect) until it has enough data."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "strategy.json")
    try:
        st = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return st if st.get("level") in ("early", "reliable") else {}


def pick_explainer(history: list[dict], today: dt.date) -> dict:
    """An unused topic from the syllabus, rotating through the categories."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "syllabus.json")
    items = json.load(open(path, encoding="utf-8"))
    past = [h.get("topic", "") for h in history][-60:]
    recent_cats = [h.get("category") for h in history][-3:]
    fresh = [e for e in items if not safety.too_similar(e["title"], past, 0.5)] or items
    varied = [e for e in fresh if e["category"] not in recent_cats] or fresh
    st = load_strategy()
    rng = random.Random(today.toordinal())
    if st:                                   # 70 percent follow what works, 30 percent keep exploring
        pool = [e for e in varied if e["category"] not in st.get("avoid_categories", [])] or varied
        liked = [e for e in pool if e["category"] in st.get("prefer_categories", [])]
        if liked and rng.random() < 0.7:
            pool = liked
        return rng.choice(pool)
    return rng.choice(varied)


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
        if re.match(r"(nobody (mentions|tells|talks)|i just (figured|found|realized|learned))", h, re.I):
            overall += 8          # the two hook styles the owner prefers
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
    speech = None
    if (plan or {}).get("lang") == "hi":
        sp = raw.get("speech") or {}
        h_all = list(sp.get("hook_options") or [])
        speech = {"hook": str(h_all[best[2]]).strip() if best[2] < len(h_all) else "",
                  "sentences": [str(x).strip() for x in sp.get("sentences") or []],
                  "cta": str(sp.get("cta") or "").strip(), "loop_line": str(sp.get("loop_line") or "").strip()}
    return {
        "lang": "hi" if (plan or {}).get("lang") == "hi" else "en",
        "speech": speech,
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
        "first_comment": (f"Comment {re.sub('[^A-Za-z]', '', str(raw.get('keyword') or '')).upper()} below and I will send it to you."
                          if (plan or {}).get("cta_style") == "keyword" and raw.get("keyword") else first_comment(raw.get("first_comment"), lex)),
        "loop_line": loop_line(raw.get("loop_line"), lex),
        "cta_type": plan.get("cta_style", "question") if plan else raw.get("cta_type", ""),
        "cta_style": plan.get("cta_style", "question") if plan else "question",
        "keyword": re.sub(r"[^A-Za-z]", "", str(raw.get("keyword") or "")).upper() if (plan or {}).get("cta_style") == "keyword" else "",
        "reward_text": _reward(raw.get("reward_text")) if (plan or {}).get("cta_style") == "keyword" else "",
        "teaser": (plan or {}).get("teaser") or "",
        "cta": cta,
        "cover_text": cover,
        "caption": cap,
        "hashtags": [("#" + re.sub(r"[^A-Za-z0-9]", "", h)) for h in raw.get("hashtags", [])][:5],
    }


FIXED_TAGS = "#fyp #doubletap #growthmindset #viral #reels #explorepage"      # on every reel, as requested
COMMENT_PROMPTS = ["Which one will you try first? Tell me below.", "What would you ask an AI first? Write it below.",
                   "Did you know this already? Yes or no below.", "What is your biggest struggle with this? Tell me below.",
                   "Want a part two? Comment your topic below.", "Which tip surprised you most? Tell me below."]


def _reward(text) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return re.sub(r"[\U00010000-\U0010ffff☀-➿]", "", t).replace("\u2014", ",")


def loop_line(text, lex) -> str:
    t = clean_text(str(text or ""), lex).strip(" .!?,")
    if not (2 <= len(t.split()) <= 9) or re.search(r"\d|https?://|@|#", t):
        return ""
    return t


def first_comment(text, lex=None, seed: int = 0) -> str:
    """The comment the account posts under its own reel. Safe fallback if the model's one is unusable."""
    t = re.sub(r"\s+", " ", clean_text(str(text or ""), lex)) if lex else str(text or "").strip()
    if not t or len(t) > 120 or re.search(r"\d|https?:|@|#", t) or safety.check_text("comment", t):
        return COMMENT_PROMPTS[seed % len(COMMENT_PROMPTS)]
    return t


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


def scene_issues(script: dict, short: bool = False) -> list[str]:
    sc = script.get("scenes") or []
    issues = []
    if len(sc) < (3 if short else 4):
        issues.append(f"only {len(sc)} usable scenes (need at least {3 if short else 4})")
    types = [x["visual"]["type"] for x in sc]
    if len(set(types)) < (2 if short else 3):
        issues.append(f"scenes use too few different visual types")
    if short and "chat" not in types:
        issues.append("short reel needs an on-screen chat demo scene")
    elif short and (types[0] != "chat" or int(sc[0].get("from", 1)) != 1):
        issues.append("results first: the chat demo must be the first scene, starting at sentence 1")
    if any(a == b for a, b in zip(types, types[1:])):
        issues.append("two neighbouring scenes use the same visual type")
    for x in sc:
        issues += safety.check_text("scene visual", json.dumps(x["visual"]))
    return issues


# ------------------------------------------------------------------------------ plans
def build_plans(cfg: Config, history: list[dict], today: dt.date, log: dict, fmt: str = "long", lang: str = "en") -> list[dict]:
    """Source-grounded plans first (best story with a readable article), evergreen last.
    Hinglish reels use evergreen topics only: a translated news story is too easy to get wrong."""
    plans: list[dict] = []
    cands = []
    if lang != "hi":
        try:
            cands = research.gather()
        except Exception as e:  # noqa: BLE001
            print(f"[research] gather failed: {type(e).__name__}", file=sys.stderr)
    log["candidates_seen"] = len(cands)
    ranked = topic.choose(cfg, cands, history, today) if cands else []
    log["shortlist"] = [{"title": r["candidate"].title, "publisher": r["candidate"].publisher, "url": r["candidate"].url,
                         "category": r["category"], "total": r["total"], "scores": r["scores"]} for r in ranked[:10]]
    for r in ranked[:4]:
        art = research.fetch_article(r["candidate"].url)
        if not art:
            continue
        plans.append({"mode": "source", "cand": r["candidate"], "category": r["category"], "angle": r["angle"],
                      "scores": r["scores"], "total": r["total"], "excerpt": art["text"], "fmt": fmt})
        if len(plans) == 3:
            break
    entry = pick_explainer(history, today)
    plans.append({"mode": "explainer", "entry": entry, "pillar": entry["category"], "seed": entry["title"],
                  "category": entry["category"], "fmt": fmt})
    return plans


def writer_prompt(plan: dict, history: list[dict], problems: list[str], today: dt.date) -> tuple[str, str]:
    short = plan.get("fmt") == "short"
    S, R, V, SC = (SHORT_STRUCTURE, SHORT_RULES, SHORT_VISUALS, SHORT_SCHEMA) if short else (STRUCTURE, HARD_RULES, VISUALS, SCHEMA)
    past_topics = [h.get("topic", "") for h in history][-40:]
    past_hooks = [h.get("hook", "") for h in history][-12:]
    avoid = (f"Topics already covered (do not repeat): {json.dumps(past_topics)}\n"
             f"Recent hooks (do not reuse their wording or structure): {json.dumps(past_hooks)}\n")
    fix = f"Fix these problems from the last attempt: {problems[-6:]}\n" if problems else ""
    st = load_strategy()
    if st and (st.get("prefer_hooks") or st.get("avoid_hooks")):
        fix += (f"Our own data ({st.get('sample_size')} reels): hook patterns that did best: {st.get('prefer_hooks') or 'unknown'}; "
                f"weakest: {st.get('avoid_hooks') or 'none'}. Use a best pattern for at least one hook option.\n")
    if short and plan.get("lang") == "hi":
        R = R + "\n\n" + HI_BLOCK
    if short:
        fix += engage.prompt_block(plan) + "\n"
        if plan.get("promised"):
            fix += (f'Yesterday\'s reel promised this topic for today: "{plan["promised"]}". Deliver exactly that topic. '
                    'Do not mention tomorrow.\n')
    if plan["mode"] == "source":
        c = plan["cand"]
        system = "\n\n".join([PERSONA, S, HOOKS, SOURCE_RULES, V, R, SC])
        user = (f"Today: {today.isoformat()}\nCategory to use: {plan['category']}\nAngle: {plan['angle']}\n"
                f"SOURCE title: {c.title}\nSOURCE publisher: {c.publisher} (published {c.published})\n"
                f"SOURCE text:\n\"\"\"\n{plan['excerpt']}\n\"\"\"\n{avoid}{fix}")
    else:
        e = plan["entry"]
        system = "\n\n".join([PERSONA, S, HOOKS, EXPLAINER_RULES, V, R, SC])
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
    short = plan.get("fmt") == "short"
    script["format"] = "short" if short else "long"
    words = sum(len(x.split()) for x in [script["hook"]] + script["sentences"] + [script["cta"]])
    if short:
        if not (SHORT_WORDS[0] <= words <= SHORT_WORDS[1]):
            issues.append(f"{words} words; a short reel needs {SHORT_WORDS[0]} to {SHORT_WORDS[1]}")
        issues += engage.check_cta(plan.get("cta_style"), script["cta"], script)
    elif words < MIN_WORDS or len(script["sentences"]) < 9:
        issues.append(f"too thin for a deep explainer: {words} words, {len(script['sentences'])} sentences (need {MIN_WORDS}+ words, 9+ sentences)")
    issues += scene_issues(script, short)
    if script.get("loop_line"):
        issues += safety.check_text("loop line", script["loop_line"])
    if plan.get("lang") == "hi":
        issues += speech_issues(script)
    elif script["hook_score"] < HOOK_FLOOR:
        issues.append(f"best hook only scores {script['hook_score']} (need {HOOK_FLOOR}+)")
    verified = []
    if source:
        verified, g_issues = grounding.verify(script, plan["excerpt"], min_quotes=1 if short else 2)
        issues += g_issues
    if issues:
        return None, issues
    review_input = {"script": {k: script[k] for k in ("hook", "sentences", "cta", "takeaway", "scenes", "opinions")},
                    "mode": plan["mode"], "format": "short" if short else "long",
                    "language": "Hinglish" if plan.get("lang") == "hi" else "English"}
    if plan.get("lang") == "hi":
        review_input["speech"] = script.get("speech")
    if source:
        review_input["SOURCE"] = plan["excerpt"]
        review_input["verified_evidence"] = verified
    try:
        review = llm.chat_json(cfg, REVIEW_SYSTEM, json.dumps(review_input), temperature=0.0)
    except Exception as e:  # noqa: BLE001 - reviewer unavailable: fail closed
        return None, [f"reviewer unavailable ({type(e).__name__})"]
    total, parts = rubric_total(review)
    min_depth = SHORT_DEPTH if short else PASS_DEPTH
    problems_r = [str(p) for p in review.get("problems", [])][:4] + \
                 [f"unsupported: {u}" for u in review.get("unsupported_claims", [])][:4]
    if review.get("unsupported_claims"):
        return None, problems_r or ["unsupported claims"]
    if total < PASS_TOTAL and total >= FALLBACK_TOTAL and parts["accuracy"] >= PASS_ACCURACY and parts["depth"] >= min_depth:
        script["quality"] = {"total": total, **parts}
        script["evidence_verified"] = verified
        _NEAR.append((total, script, plan))
    if total < PASS_TOTAL or parts["accuracy"] < PASS_ACCURACY or parts["depth"] < min_depth:
        return None, [f"quality {total}/100 (hook {parts['hook']}, accuracy {parts['accuracy']}, depth {parts['depth']}, "
                      f"clarity {parts['clarity']}) is below the bar of {PASS_TOTAL}"] + problems_r
    script["quality"] = {"total": total, **parts}
    script["evidence_verified"] = verified
    return script, []


def speech_issues(script: dict) -> list[str]:
    """The Devanagari voice lines must mirror the Hinglish display lines, be safe, and have no digits."""
    sp = script.get("speech") or {}
    issues = []
    lines = [sp.get("hook", "")] + list(sp.get("sentences", [])) + [sp.get("cta", "")]
    if len(sp.get("sentences", [])) != len(script["sentences"]):
        issues.append(f"speech has {len(sp.get('sentences', []))} sentences, the on-screen text has {len(script['sentences'])}")
    if any(not DEVANAGARI.search(x) for x in lines):
        issues.append("every speech line must be written in Devanagari")
    if script.get("loop_line") and not DEVANAGARI.search(sp.get("loop_line", "")):
        issues.append("loop line needs a Devanagari speech version")
    for x in lines + [sp.get("loop_line", "")]:
        if re.search(r"\d", x) or len(x.split()) > 30:
            issues.append("speech lines must have no digits and stay under thirty words")
            break
    shown = " ".join([script["hook"]] + script["sentences"] + [script["cta"]])
    for rx, why in HI_BLOCK_RE:
        m = rx.search(shown + " " + " ".join(lines))
        if m:
            issues.append(f"{why}: '{m.group(0)}'")
    return issues


def demo_run(cfg: Config, script: dict) -> bool:
    """Run the on-screen demo prompt through the real assistant so the result shown is a genuine answer,
    not text we invented. Keeps the written result if the call fails or the answer looks unusable."""
    scene = next((x for x in script.get("scenes") or [] if x["visual"].get("type") == "chat"), None)
    if not scene or not scene["visual"].get("prompt"):
        return False
    try:
        res = llm.chat_json(cfg, "You are a helpful AI assistant. Answer the user's request directly and concretely in plain text, "
                            'at most 24 words, no markdown, no lists, no preamble. Reply with JSON only: {"answer": "..."}',
                            scene["visual"]["prompt"], temperature=0.4)
        ans = re.sub(r"\s+", " ", str(res.get("answer", ""))).strip().replace("\u2014", ",")
    except Exception as e:  # noqa: BLE001
        print(f"demo run skipped: {type(e).__name__}", file=sys.stderr)
        return False
    if not (4 <= len(ans.split()) <= 28) or re.search(r"https?:|www\.|[*#_`]", ans) or safety.check_text("demo answer", ans):
        return False
    scene["visual"]["result"] = ans
    script["demo_real"] = True
    return True


def finish(script: dict, plan: dict, log: dict, cfg: Config | None = None) -> dict:
    if cfg is not None and plan.get("fmt") == "short":
        demo_run(cfg, script)
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


def backup_script(history: list[dict], today: dt.date, fmt: str = "long") -> dict:
    """Last resort so a day is never skipped: a hand-written lesson from data/backup_scripts.json.
    Every entry is checked by the same safety rules at load time. Unused ones go first; once all
    have been used, the one used longest ago comes round again."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backup_scripts.json")
    bank = json.load(open(path, encoding="utf-8"))
    ok = [b for b in bank if not safety.check_script(b)]
    if fmt == "short":                       # short reels use the hand-written short lessons (no deep scenes)
        ok = [b for b in ok if not b.get("scenes")] or ok
    if not ok:
        raise RuntimeError("backup bank is empty or failed the safety checks")
    last = {}
    for i, h in enumerate(history):
        last[h.get("topic", "")] = i
    ok.sort(key=lambda b: last.get(b["topic"], -1))
    script = json.loads(json.dumps(ok[0]))
    script["mode"] = "backup"
    script["format"] = "long" if script.get("scenes") else "short"
    script.setdefault("cta_style", "question")
    script["first_comment"] = script.get("first_comment") or COMMENT_PROMPTS[today.toordinal() % len(COMMENT_PROMPTS)]
    script["pillar"] = script["category"]
    script["quality"] = {"total": None, "note": "pre-checked backup lesson"}
    script["evidence_verified"] = []
    script["research"] = {"date": today.isoformat(), "mode": "backup", "note": "live scripts did not pass; used a pre-checked lesson",
                          "fact_check": [], "opinions": []}
    return script


def pick_format(cfg: Config, today: dt.date) -> str:
    f = (cfg.reel_format or "short").lower()
    if f == "alternate":
        return "short" if today.toordinal() % 2 else "long"
    if f == "auto":                            # follow the growth manager once it is sure, otherwise alternate
        w = load_strategy().get("format_winner") if load_strategy().get("level") == "reliable" else None
        return w or ("short" if today.toordinal() % 2 else "long")
    return "long" if f == "long" else "short"


def pick_lang(cfg: Config, today: dt.date, fmt: str) -> str:
    """Hinglish only for short reels; 'alternate' swaps days, 'auto' follows the growth manager once it is sure."""
    if fmt != "short":
        return "en"
    f = (cfg.reel_lang or "en").lower()
    if f == "hi":
        return "hi"
    if f == "alternate":
        return "hi" if today.toordinal() % 2 == 0 else "en"
    if f == "auto":
        st = load_strategy()
        w = st.get("lang_winner") if st.get("level") == "reliable" else None
        return w or ("hi" if today.toordinal() % 2 == 0 else "en")
    return "en"


def generate(cfg: Config, history: list[dict], today: dt.date | None = None, lang: str | None = None) -> dict:
    today = today or dt.date.today()
    log: dict = {"date": today.isoformat()}
    fmt = pick_format(cfg, today)
    lang = lang or pick_lang(cfg, today, fmt)
    log["format"], log["lang"] = fmt, lang
    try:
        plans = build_plans(cfg, history, today, log, fmt, lang)
    except Exception as e:  # noqa: BLE001 - research or AI outage must not stop the daily post
        print(f"[plans] failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        plans = []
    strategy = load_strategy()
    eng = engage.plan_for(cfg, history, today, fmt, pick_explainer, strategy, lang)
    log["engagement"] = {k: eng[k] for k in ("cta_style", "teaser", "promised")}
    promised = eng.get("promised") and engage.entry_by_title(eng["promised"])
    if promised:                                 # yesterday's teaser: keep the promise first
        plans.insert(0, {"mode": "explainer", "entry": promised, "pillar": promised["category"], "seed": promised["title"],
                         "category": promised["category"], "fmt": fmt})
    for plan in plans:
        plan.setdefault("fmt", fmt)
        plan.update({k: v for k, v in eng.items() if k in ("cta_style", "teaser", "promised")})
        plan["lang"] = lang
        if plan["mode"] == "source":
            plan["promised"] = None
        tries = 3
        problems: list[str] = []
        label = plan["cand"].title if plan["mode"] == "source" else plan["seed"]
        for n in range(tries):
            print(f"[{plan['mode']}] attempt {n + 1}: {label[:80]}", file=sys.stderr)
            script, issues = attempt(cfg, plan, history, problems, today)
            if script:
                return finish(script, plan, log, cfg)
            problems += issues
            print(f"  rejected: {issues[:4]}", file=sys.stderr)
    if _NEAR:
        total, script, plan = max(_NEAR, key=lambda t: t[0])
        print(f"using best near-miss script ({total}/100): every safety and accuracy check passed", file=sys.stderr)
        return finish(script, plan, log, cfg)
    if lang == "hi" and cfg.hi_strict:
        raise RuntimeError("no Hinglish script passed the checks (strict test run: not falling back to English)")
    if lang == "hi":
        print("no Hinglish script passed; making today's reel in English instead", file=sys.stderr)
        _NEAR.clear()
        return generate(cfg, history, today, lang="en")
    print("no live script passed; using a pre-checked backup lesson", file=sys.stderr)
    return backup_script(history, today, fmt)


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
    if script.get("format") == "short" and script.get("first_comment") and "?" not in cap:
        cap = cap.replace("AI-generated video.", f"{script['cta']}\nAI-generated video.") if "AI-generated video." in cap else cap
    if script.get("teaser") and "AI-generated video." in cap:
        cap = cap.replace("AI-generated video.", f"Tomorrow: {script['teaser']}\nAI-generated video.")
    if script.get("series") and "AI-generated video." in cap:
        cap = cap.replace("AI-generated video.", f"{series.label(script)}\nAI-generated video.")
    return f"{cap}\n\n{FIXED_TAGS}".strip()
