"""Weekly growth manager: read the numbers, find what works, write a strategy the script writer follows,
and produce a plain-language report plus DRAFT replies to comments (never posted automatically).

python -m reelbot.growth
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import sys

from . import insights, llm
from .config import Config

MIN_EARLY, MIN_RELIABLE = 12, 20
BUCKETS = (("morning", 5, 12), ("afternoon", 12, 17), ("evening", 17, 21), ("night", 21, 29))


def hour_bucket(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        h = t.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30))).hour
    except ValueError:
        return None
    h = h if h >= 5 else h + 24
    return next((n for n, a, b in BUCKETS if a <= h < b), None)


def join(history: list[dict], data: dict) -> list[dict]:
    """One row per posted reel: what we made + how it did."""
    media = data.get("media", {})
    rows = []
    for h in history:
        m = media.get(h.get("media_id"))
        if not m or not m.get("metrics"):
            continue
        k = m["metrics"]
        reach = k.get("reach") or 0
        inter = (k.get("shares") or 0) * 3 + (k.get("saved") or 0) * 2 + (k.get("comments") or 0) * 2 + (k.get("likes") or 0)
        rows.append({"media_id": h["media_id"], "topic": h.get("topic"), "category": h.get("category"),
                     "hook": h.get("hook"), "cta_style": h.get("cta_style"), "hook_mechanism": (h.get("hook_mechanism") or "").lower() or None,
                     "format": h.get("format") or "long", "series": h.get("series"),
                     "bucket": hour_bucket(h.get("posted_at") or m.get("timestamp")), "reach": reach,
                     "views": k.get("views") or 0, "watch_s": (k.get("ig_reels_avg_watch_time") or 0) / 1000.0,
                     "shares": k.get("shares") or 0, "saved": k.get("saved") or 0, "comments": k.get("comments") or 0,
                     "likes": k.get("likes") or 0, "score": round(inter / reach * 1000, 1) if reach else 0.0,
                     "permalink": m.get("permalink")})
    return rows


def value(g: dict) -> float:
    """One number to rank a group by: typical reach, lifted by how engaged those viewers were."""
    return g["reach"] * (1 + g["score"] / 100.0)


def group(rows: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        v = r.get(key)
        if v:
            out.setdefault(v, []).append(r)
    return {k: {"n": len(v), "reach": round(statistics.mean(x["reach"] for x in v), 1),
                "score": round(statistics.mean(x["score"] for x in v), 1),
                "watch_s": round(statistics.mean(x["watch_s"] for x in v), 1)} for k, v in out.items()}


def level(n: int) -> str:
    return "reliable" if n >= MIN_RELIABLE else "early" if n >= MIN_EARLY else "collecting"


def best_worst(groups: dict, min_best: int = 2, min_worst: int = 3):
    ranked = sorted(((value(g), k) for k, g in groups.items() if g["n"] >= min_best), reverse=True)
    if len(ranked) < 2:
        return [], []
    med = statistics.median(v for v, _ in ranked)
    best = [k for v, k in ranked[:3] if v >= med]
    worst = [k for v, k in ranked if groups[k]["n"] >= min_worst and v < 0.5 * med]
    return best, worst


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    lv = level(n)
    cat, hook, fmt, ser, buck, cta = (group(rows, k) for k in ("category", "hook_mechanism", "format", "series", "bucket", "cta_style"))
    pc, ac = best_worst(cat)
    ph, ah = best_worst(hook)
    pcta, acta = best_worst(cta)
    winner = None
    if lv == "reliable" and len(fmt) == 2 and all(g["n"] >= 4 for g in fmt.values()):
        (a, ga), (b, gb) = sorted(fmt.items(), key=lambda kv: -value(kv[1]))
        if value(ga) >= 1.3 * value(gb):
            winner = a
    pb, _ = best_worst(buck, 3, 99)
    notes = []
    if lv == "collecting":
        notes.append(f"Only {n} reels with data so far. Collecting; no changes are made to the writer yet (it needs {MIN_EARLY}).")
    elif lv == "early":
        notes.append(f"{n} reels with data: early signals only. Topic and hook hints are applied lightly; format stays as set.")
    return {"updated": dt.date.today().isoformat(), "sample_size": n, "level": lv,
            "prefer_categories": pc if lv != "collecting" else [], "avoid_categories": ac if lv != "collecting" else [],
            "prefer_hooks": ph if lv != "collecting" else [], "avoid_hooks": ah if lv != "collecting" else [],
            "prefer_cta": pcta if lv != "collecting" else [], "avoid_cta": acta if lv != "collecting" else [],
            "format_winner": winner, "best_time": pb[0] if pb and lv != "collecting" else None,
            "groups": {"category": cat, "hook": hook, "format": fmt, "series": ser, "time": buck, "cta": cta}, "notes": notes}


def draft_replies(cfg: Config, comments: list[dict]) -> list[dict]:
    comments = [c for c in comments if len((c.get("text") or "").split()) >= 2][:8]
    if not comments or not (cfg.groq_key or cfg.gemini_key):
        return []
    system = ("You draft replies to Instagram comments for Shubh, an animated narrator who teaches AI tools, productivity and career "
              "skills. Warm, brief (under 25 words), specific to the comment, end with a short question when natural. No emojis spam, "
              "no em dashes, no promises, no health, money or legal advice. If a comment is spam or abusive, reply with an empty string. "
              'Reply with JSON only: {"replies":[{"id":"comment id","reply":"text"}]}')
    try:
        res = llm.chat_json(cfg, system, json.dumps([{"id": c["id"], "text": c["text"]} for c in comments]), temperature=0.6)
        by = {c["id"]: c for c in comments}
        return [{"comment": by[r["id"]]["text"], "user": by[r["id"]]["username"], "draft": str(r["reply"]).replace("—", ",")}
                for r in res.get("replies", []) if r.get("id") in by and r.get("reply")]
    except Exception as e:  # noqa: BLE001
        print(f"reply drafts skipped: {type(e).__name__}", file=sys.stderr)
        return []


def explain(cfg: Config, strategy: dict, rows: list[dict]) -> str:
    if strategy["level"] == "collecting" or not (cfg.groq_key or cfg.gemini_key):
        return ""
    try:
        res = llm.chat_json(cfg, "You are a careful social media analyst. Given grouped stats for an Instagram Reels account, write "
                            "3 to 5 short plain sentences: what seems to work, what does not, and ONE experiment for next week. "
                            "Label guesses as guesses; never claim a cause you cannot see; small samples are noisy and you must say so. "
                            'JSON only: {"summary":"..."}', json.dumps({"groups": strategy["groups"], "sample": strategy["sample_size"]}),
                            temperature=0.2)
        return str(res.get("summary", ""))
    except Exception:  # noqa: BLE001
        return ""


def table(groups: dict) -> str:
    if not groups:
        return "_no data yet_\n"
    lines = ["| value | reels | avg reach | engagement per 1000 reach | avg watch (s) |", "|---|---|---|---|---|"]
    for k, g in sorted(groups.items(), key=lambda kv: -value(kv[1])):
        lines.append(f"| {k} | {g['n']} | {g['reach']} | {g['score']} | {g['watch_s']} |")
    return "\n".join(lines) + "\n"


def report(strategy: dict, rows: list[dict], account: dict, replies: list[dict], summary: str, log: list[dict]) -> str:
    out = [f"# Growth report {strategy['updated']}", ""]
    out.append(f"Followers: **{account.get('followers', 'n/a')}**  |  reels with data: **{strategy['sample_size']}**  |  data level: **{strategy['level']}**")
    if len(log) >= 2:
        out.append(f"Change since last week: followers {log[-1].get('followers', 0) - log[-2].get('followers', 0):+d}, "
                   f"reach of the week {log[-1].get('reach_total', 0) - log[-2].get('reach_total', 0):+d}")
    out += [""] + [f"- {n}" for n in strategy["notes"]]
    if summary:
        out += ["", "## Analyst summary (guesses, not facts)", summary]
    best = sorted(rows, key=lambda r: -r["score"])[:3]
    worst = sorted(rows, key=lambda r: r["score"])[:3]
    out += ["", "## Best three reels (engagement per 1000 reach)"] + [f"- {r['score']} | reach {r['reach']} | {r['topic']} | {r['hook']}" for r in best]
    out += ["", "## Weakest three"] + [f"- {r['score']} | reach {r['reach']} | {r['topic']} | {r['hook']}" for r in worst]
    for title, key in (("Category", "category"), ("Hook pattern", "hook"), ("Closing style", "cta"), ("Format", "format"), ("Series", "series"), ("Time of day (IST)", "time")):
        out += ["", f"## {title}", table(strategy["groups"][key])]
    out += ["## What the writer will do next week",
            f"- prefer categories: {strategy['prefer_categories'] or 'no change'}", f"- avoid categories: {strategy['avoid_categories'] or 'none'}",
            f"- prefer hook patterns: {strategy['prefer_hooks'] or 'no change'}", f"- avoid hook patterns: {strategy['avoid_hooks'] or 'none'}",
            f"- prefer closing styles: {strategy['prefer_cta'] or 'no change'}",
            f"- format winner: {strategy['format_winner'] or 'none yet'}", f"- best time bucket: {strategy['best_time'] or 'not enough data'}"]
    if replies:
        out += ["", "## Draft replies (copy, edit and post them yourself)"]
        for r in replies:
            out += [f"- **{r['user']}**: {r['comment']}", f"  - draft: {r['draft']}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    cfg = Config()
    os.makedirs("data/reports", exist_ok=True)
    try:
        history = json.load(open("data/history.json"))
    except (OSError, json.JSONDecodeError):
        history = []
    if not (cfg.ig_token and cfg.ig_user_id):
        print("Instagram secrets missing; nothing to analyse")
        return 0
    try:
        data = insights.collect(cfg)
    except insights.InsightsError as e:
        msg = (f"Could not read insights: {e}\nMost likely the token lacks the insights permission. In the Meta app, add "
               "instagram_manage_insights to the configuration, generate a new token and update the IG_ACCESS_TOKEN secret.")
        print(msg)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            open(summary, "a").write("## Growth manager\n" + msg + "\n")
        return 0
    json.dump(data, open("data/insights.json", "w"), indent=1)
    rows = join(history, data)
    strategy = analyze(rows)
    json.dump(strategy, open("data/strategy.json", "w"), indent=1)
    try:
        log = json.load(open("data/growth_log.json"))
    except (OSError, json.JSONDecodeError):
        log = []
    log.append({"date": strategy["updated"], "followers": data["account"].get("followers"), "reels": strategy["sample_size"],
                "reach_total": sum(r["reach"] for r in rows)})
    json.dump(log[-120:], open("data/growth_log.json", "w"), indent=1)
    recent = [r["media_id"] for r in sorted(rows, key=lambda r: r["media_id"], reverse=True)[:8]]
    replies = draft_replies(cfg, insights.comments_for(cfg, recent))
    text = report(strategy, rows, data["account"], replies, explain(cfg, strategy, rows), log)
    path = f"data/reports/{strategy['updated']}.md"
    open(path, "w").write(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text)
    print(f"report written to {path}; level={strategy['level']}, reels={strategy['sample_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
