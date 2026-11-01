"""Hold a script to its source: every factual line needs a verbatim quote, every number must be in the article."""
from __future__ import annotations

import re

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
MIN_CHECKED = 6          # "three steps", "one goal" are structure, not claims


def norm(text: str) -> str:
    t = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.sub(r"[^a-z0-9%$.']+", " ", t).split())


def numbers_in(text: str) -> set[float]:
    """Numeric values written as digits or as English number words (up to billions)."""
    vals: set[float] = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            vals.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    toks = re.findall(r"[a-z]+|[^a-z\s-]", text.lower().replace("-", " "))
    cur, total, seen = 0, 0, False

    def flush():
        nonlocal cur, total, seen
        if seen:
            vals.add(float(total + cur))
        cur, total, seen = 0, 0, False

    def is_num(w):
        return w in UNITS or w in TENS or w in SCALES

    for i, w in enumerate(toks):
        if w in UNITS:
            if seen and 20 <= cur % 100 < 100 and cur % 10 == 0 and UNITS[w] < 10:
                cur += UNITS[w]
            elif seen and cur % 100 == 0 and cur and UNITS[w] > 0:
                cur += UNITS[w]
            else:
                flush()
                cur, seen = UNITS[w], True
        elif w in TENS:
            if seen and cur % 100 == 0 and cur:
                cur += TENS[w]
            else:
                flush()
                cur, seen = TENS[w], True
        elif w in SCALES and seen:
            if w == "hundred":
                cur *= 100
            else:
                total += cur * SCALES[w]
                cur = 0
        elif w == "and" and seen and i + 1 < len(toks) and is_num(toks[i + 1]):
            continue
        else:
            flush()
    flush()
    return vals


def checked_numbers(spoken: list[str]) -> set[float]:
    out = set()
    for s in spoken:
        out |= {v for v in numbers_in(s) if v >= MIN_CHECKED}
    return out


def verify(script: dict, excerpt: str, min_quotes: int = 2) -> tuple[list[dict], list[str]]:
    """Returns (verified evidence, issues). Evidence refers to body sentences, 1-based."""
    issues: list[str] = []
    body = script.get("sentences", [])
    ex = norm(excerpt)
    ok, bad = [], 0
    for e in script.get("evidence", []) or []:
        try:
            idx, quote = int(e["sentence"]), str(e["quote"])
        except (KeyError, TypeError, ValueError):
            bad += 1
            continue
        nq = norm(quote)
        if 1 <= idx <= len(body) and len(nq.split()) >= 4 and nq in ex:
            ok.append({"sentence": idx, "quote": quote.strip()})
        else:
            bad += 1
    if len({e["sentence"] for e in ok}) < min_quotes:
        issues.append(f"only {len({e['sentence'] for e in ok})} sentence(s) backed by a verbatim quote (need {min_quotes})"
                      + (f"; {bad} quote(s) are not in the article" if bad else ""))
    spoken = [script.get("hook", "")] + body + [script.get("cta", "")]
    have = numbers_in(excerpt)
    for v in sorted(checked_numbers(spoken)):
        if v not in have:
            issues.append(f"the number {v:g} is not in the source article")
    return ok, issues
