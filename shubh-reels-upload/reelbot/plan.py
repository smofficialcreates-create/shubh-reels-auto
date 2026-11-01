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
    kind: str                      # hook | problem | steps | payoff | cta
    i0: int
    i1: int
    t0: float
    t1: float
    theme: str                     # paper | orange | dark
    prop: str                      # stat | keyword | bars | checklist | follow | none
    data: dict = field(default_factory=dict)


def build_sections(script: dict, nar) -> list[Section]:
    from . import brand
    o = brand.offset(script)                            # 1 when the Shubh intro is spoken first
    outro = 1 if o else 0
    n_body = len(script["sentences"])
    last = n_body + 1                                   # cta index
    keys = sorted({int(k["sentence"]) for k in script.get("key_points", []) if 1 <= int(k["sentence"]) <= n_body})
    cuts = [("hook", 0, 0)]
    if keys:
        if keys[0] > 1:
            cuts.append(("problem", 1, keys[0] - 1))
        cuts.append(("steps", keys[0], keys[-1]))
        if keys[-1] < n_body:
            cuts.append(("payoff", keys[-1] + 1, n_body))
    else:
        mid = max(1, n_body // 2)
        cuts += [("problem", 1, mid)] + ([("payoff", mid + 1, n_body)] if mid < n_body else [])
    cuts.append(("cta", last, last))
    themes = {"hook": "paper", "problem": "orange", "steps": "paper", "payoff": "dark", "cta": "paper"}
    secs = []
    if o:
        secs.append(Section("intro", 0, 0, 0.0, nar.sentence_times[0][1] + 0.25, "paper", "keyword", {"word": "Shubh"}))
    for kind, a, b in cuts:
        ia, ib = a + o, b + o + (outro if kind == "cta" else 0)
        t0 = nar.sentence_times[ia][0] if ia > 0 else 0.0
        t1 = nar.sentence_times[ib][1] + 0.25
        sec = Section(kind, ia, ib, t0, t1, themes[kind], "none")
        text_all = [script["hook"]] + script["sentences"] + [script["cta"]]
        if kind == "hook":
            num = find_number(text_all[0])
            if num:
                sec.prop, sec.data = "stat", {"value": num[0], "label": num[1]}
            else:
                ws = text_all[0].split()
                sec.prop, sec.data = "keyword", {"word": clean(ws[accent_index(ws, script.get("hook_focus"))])}
        elif kind == "problem":
            sec.prop = "bars"
        elif kind == "steps":
            sec.prop = "checklist"
            sec.data = {"labels": [k["label"] for k in sorted(script["key_points"], key=lambda k: int(k["sentence"]))][:3],
                        "starts": [nar.sentence_times[int(k) + o][0] for k in keys][:3]}
        elif kind == "payoff":
            demo = script.get("demo")
            if demo and demo.get("prompt") and demo.get("result"):
                sec.prop, sec.data = "chat", {"prompt": demo["prompt"], "result": demo["result"]}
            else:
                ws = text_all[b].split()
                sec.prop, sec.data = "keyword", {"word": clean(ws[accent_index(ws)])}
        elif kind == "cta":
            sec.prop = "follow"
            sec.data = {"t_follow": nar.sentence_times[ib][0] if outro else t0}
        secs.append(sec)
    # sections run back to back
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
