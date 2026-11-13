"""Turns a script + narration timings into scenes, props and gestures."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
             "sixty": 60}
STOP = {"your", "that", "this", "with", "from", "they", "have", "what", "when", "will", "just", "about", "every",
        "here", "there", "then", "than", "into", "only", "first", "second", "third", "next", "after", "before"}


def clean(w: str) -> str:
    return re.sub(r"[^A-Za-z0-9']", "", w).lower()


def accent_index(words: list[str], hint: str | None = None) -> int:
    c = [clean(w) for w in words]
    if hint and clean(hint) in c:
        return c.index(clean(hint))
    for i, w in enumerate(c):
        if w in NUM_WORDS or any(ch.isdigit() for ch in w):
            return i
    best, bi = -1, 0
    for i, w in enumerate(c):
        if w in STOP or len(w) < 5:
            continue
        if len(w) > best:
            best, bi = len(w), i
    return bi


def find_number(sentence: str):
    """(value, following word) for the first spelled-out or digit number, else None."""
    ws = sentence.split()
    for i, w in enumerate(ws):
        c = clean(w)
        val = NUM_WORDS.get(c) or (int(c) if c.isdigit() and int(c) < 1000 else None)
        if val is not None:
            nxt = clean(ws[i + 1]) if i + 1 < len(ws) else ""
            return val, nxt
    return None


@dataclass
class Section:
    kind: str                      # intro | hook | problem | steps | payoff | cta   (drives the gestures)
    i0: int
    i1: int
    t0: float
    t1: float
    theme: str                     # navy | ivory | forest
    prop: str                      # visual type, or none
    data: dict = field(default_factory=dict)
    label: str = ""
    transition: str = "circle"


VISUAL_KIND = {"compare": "problem", "bars": "problem", "flow": "steps", "layers": "steps", "numeral": "steps",
               "checklist": "steps", "definition": "steps", "chat": "payoff", "counter": "payoff", "keyword": "payoff",
               "stat": "payoff"}
DEFAULT_LABEL = {"compare": "side by side", "bars": "the problem", "flow": "how it flows", "layers": "the layers",
                 "numeral": "the step", "checklist": "the plan", "definition": "in plain words", "chat": "try it",
                 "counter": "the numbers", "keyword": "remember this", "stat": "the numbers"}
THEME_CYCLE = ["navy", "ivory", "navy", "navy", "forest", "navy", "ivory", "navy", "navy", "ivory", "navy", "forest"]
TRANSITIONS = ["circle", "slide_up", "push", "circle", "iris", "slide_up"]


def _seed(script: dict) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(script.get("topic", "reel"))) % 9973


def legacy_scenes(script: dict) -> list[dict]:
    """Older scripts (no 'scenes'): problem / steps / payoff derived from the key points."""
    n = len(script["sentences"])
    keys = sorted({int(k["sentence"]) for k in script.get("key_points", []) if 1 <= int(k["sentence"]) <= n})
    out = []
    if keys:
        if keys[0] > 1:
            out.append({"from": 1, "to": keys[0] - 1, "visual": {"type": "bars"}})
        out.append({"from": keys[0], "to": keys[-1], "visual": {
            "type": "checklist", "labels": [k["label"] for k in sorted(script["key_points"], key=lambda k: int(k["sentence"]))][:3],
            "sync": keys[:3]}})
        if keys[-1] < n:
            demo = script.get("demo")
            if demo and demo.get("prompt") and demo.get("result"):
                v = {"type": "chat", "prompt": demo["prompt"], "result": demo["result"]}
            else:
                v = {"type": "keyword", "word": None}
            out.append({"from": keys[-1] + 1, "to": n, "visual": v})
    else:
        mid = max(1, n // 2)
        out.append({"from": 1, "to": mid, "visual": {"type": "bars"}})
        if mid < n:
            out.append({"from": mid + 1, "to": n, "visual": {"type": "keyword", "word": None}})
    return out


def repair_scenes(scenes: list[dict], n: int) -> list[dict]:
    """Sorted, inside 1..n, no overlaps, no gaps (a gap is absorbed by the scene before it)."""
    good = []
    for sc in sorted((s for s in scenes if isinstance(s, dict) and isinstance(s.get("visual"), dict)),
                     key=lambda s: int(s.get("from", 0))):
        a, b = max(1, int(sc.get("from", 1))), min(n, int(sc.get("to", sc.get("from", 1))))
        if good and a <= good[-1]["to"]:
            a = good[-1]["to"] + 1
        if a > b:
            continue
        good.append({**sc, "from": a, "to": b})
    if not good:
        return []
    good[0]["from"] = 1
    for x, y in zip(good, good[1:]):
        x["to"] = y["from"] - 1
    good[-1]["to"] = n
    return good


def build_sections(script: dict, nar) -> list[Section]:
    from . import brand
    o = brand.offset(script)                            # 1 when the Shubh intro is spoken first
    outro = 1 if brand.has_outro(script) else 0
    n_body = len(script["sentences"])
    last = n_body + 1                                   # cta index
    text_all = [script["hook"]] + script["sentences"] + [script["cta"]]
    scenes = repair_scenes(script.get("scenes") or [], n_body) or repair_scenes(legacy_scenes(script), n_body)
    seed = _seed(script)

    def times(a, b):
        ia, ib = a + o, b + o
        return nar.sentence_times[ia][0] if ia > 0 else 0.0, nar.sentence_times[ib][1] + 0.25

    secs: list[Section] = []
    # the hook is spoken first (index 0); Shubh introduces himself right after it (index 1)
    num = find_number(text_all[0])
    if num:
        hv = {"type": "stat", "value": num[0], "label": num[1]}
    else:
        ws = text_all[0].split()
        hv = {"type": "keyword", "word": clean(ws[accent_index(ws, script.get("hook_focus"))])}
    secs.append(Section("hook", 0, 0, 0.0, nar.sentence_times[0][1] + 0.25, "navy", hv["type"], hv, "the hook"))
    if o:
        secs.append(Section("intro", 1, 1, nar.sentence_times[1][0], nar.sentence_times[1][1] + 0.25, "navy", "keyword",
                            {"word": "Shubh"}, "intro"))
    # body scenes
    for k, sc in enumerate(scenes):
        a, b = sc["from"], sc["to"]
        v = dict(sc["visual"])
        typ = v.get("type", "keyword")
        if typ == "keyword" and not v.get("word"):
            ws = text_all[b].split()
            v["word"] = clean(ws[accent_index(ws)])
        t0, t1 = times(a, b)
        sync = [int(x) for x in v.get("sync", []) if isinstance(x, (int, float)) and a <= int(x) <= b]
        nodes = v.get("nodes") or v.get("labels") or []
        if typ in ("flow", "layers", "checklist"):
            need = len(nodes)
            if len(sync) != need:                       # spread evenly over the sentences of the scene
                span = list(range(a, b + 1))
                sync = [span[min(len(span) - 1, int(j * len(span) / max(1, need)))] for j in range(need)]
            v["abs_starts"] = [nar.sentence_times[x + o][0] for x in sync]
            if len({round(x, 2) for x in v["abs_starts"]}) < len(v["abs_starts"]):      # several items in one sentence: spread them
                lo, hi = t0 + 0.4, max(t0 + 1.2, t1 - 1.2)
                v["abs_starts"] = [lo + (hi - lo) * j / max(1, need) for j in range(need)]
        sec = Section(VISUAL_KIND.get(typ, "payoff"), a + o, b + o, t0, t1,
                      THEME_CYCLE[(seed + k * 1) % len(THEME_CYCLE)], typ, v,
                      sc.get("label") or DEFAULT_LABEL.get(typ, "next"),
                      TRANSITIONS[(seed + k) % len(TRANSITIONS)])
        secs.append(sec)
    # cta (with the spoken outro)
    ia, ib = last + o, last + o + outro
    t0 = nar.sentence_times[ia][0]
    t1 = nar.sentence_times[ib][1] + 0.25
    secs.append(Section("cta", ia, ib, t0, t1, "forest", "follow",
                        {"t_follow": nar.sentence_times[ib][0] if outro else t0}, "follow"))
    # no two neighbouring scenes share a theme or a transition
    for prev, cur in zip(secs, secs[1:]):
        if cur.kind != "cta" and cur.theme == prev.theme:
            cur.theme = "ivory" if prev.theme == "navy" else "navy"
        if cur.transition == prev.transition:
            cur.transition = TRANSITIONS[(TRANSITIONS.index(cur.transition) + 1) % len(TRANSITIONS)]
    for a, b in zip(secs, secs[1:]):
        a.t1 = b.t0
    return secs


# ---------------------------------------------------------------------------
# gestures. Angles in degrees, positive = away from the body.
# keys: l_sh l_el r_sh r_el head lean shrug osc_arm osc_amp osc_hz
# ---------------------------------------------------------------------------
POSES = {
    "rest":      dict(l_sh=3, l_el=0, r_sh=3, r_el=0, head=0, lean=0, shrug=0),
    "wave":      dict(l_sh=3, l_el=0, r_sh=40, r_el=138, head=-3, lean=1, shrug=0, osc="r_el", amp=17, hz=2.4),
    "point_up":  dict(l_sh=3, l_el=0, r_sh=30, r_el=165, head=2, lean=1, shrug=0, osc="r_el", amp=5, hz=1.2),
    "present_r": dict(l_sh=3, l_el=0, r_sh=62, r_el=52, head=3, lean=2, shrug=0),
    "present_l": dict(l_sh=62, l_el=52, r_sh=3, r_el=0, head=-3, lean=-2, shrug=0),
    "count":     dict(l_sh=3, l_el=0, r_sh=24, r_el=122, head=-2, lean=1, shrug=0, osc="r_el", amp=8, hz=3.0),
    "shrug":     dict(l_sh=30, l_el=95, r_sh=30, r_el=95, head=5, lean=0, shrug=9),
    "think":     dict(l_sh=3, l_el=0, r_sh=-6, r_el=-150, head=-4, lean=-1, shrug=0),
    "both_up":   dict(l_sh=55, l_el=150, r_sh=55, r_el=150, head=0, lean=0, shrug=3),
    "cheer":     dict(l_sh=70, l_el=165, r_sh=70, r_el=165, head=0, lean=0, shrug=4, osc="both", amp=10, hz=3.0),
}
SEQ = {
    "intro": ["wave", "present_r"],
    "hook": ["present_r", "present_l"],
    "problem": ["shrug", "think", "shrug"],
    "steps": ["present_l", "present_r", "count"],
    "payoff": ["both_up", "think"],
    "cta": ["wave", "cheer"],
}


def gesture_events(sections: list[Section], nar) -> list[tuple[float, str]]:
    """(time, pose name) targets. Between sentences the arms drop back to rest."""
    ev = [(0.0, "rest")]
    counters = {k: 0 for k in SEQ}
    for sec in sections:
        for i in range(sec.i0, sec.i1 + 1):
            s, e = nar.sentence_times[i]
            seq = SEQ[sec.kind]
            name = seq[counters[sec.kind] % len(seq)]
            counters[sec.kind] += 1
            ev.append((s + (0.05 if i else 0.12), name))
            if e - s > 2.8:                                   # long sentence: a second gesture mid-way
                name2 = seq[counters[sec.kind] % len(seq)]
                counters[sec.kind] += 1
                ev.append((s + (e - s) * 0.5, name2))
            ev.append((max(s + 0.6, e - 0.12), "rest"))
    ev.sort(key=lambda x: x[0])
    return ev


class Spring:
    """Slightly under-damped follower: gives gestures a little overshoot."""

    def __init__(self, x=0.0, omega=15.0, zeta=0.58):
        self.x, self.v, self.w, self.z = x, 0.0, omega, zeta

    def step(self, target: float, dt: float) -> float:
        for _ in range(3):
            h = dt / 3
            a = self.w * self.w * (target - self.x) - 2 * self.z * self.w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x
