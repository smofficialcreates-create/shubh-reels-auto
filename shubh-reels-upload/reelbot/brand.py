"""Fixed brand lines spoken in every reel."""
INTRO = "Want to stay ahead of the curve? I'm Shubh. Let's upgrade the way you think, work and grow."
OUTRO = "Thanks for watching. Follow for more smart ways to work, and I'll see you in the next one."


OUTRO_SHORT = "I'm Shubh. Follow me to upgrade the way you think, work and grow."   # short reels: name goes last, payoff comes first


def is_short(script: dict) -> bool:
    return script.get("format") == "short"


def spoken(script: dict) -> list[str]:
    """Every spoken line in order: hook, intro, body, closing line, outro.
    The hook comes first so the first two seconds are the scroll-stopper; Shubh introduces himself right after it."""
    if not script.get("with_brand", True):
        return [script["hook"]] + script["sentences"] + [script["cta"]]
    if is_short(script):                       # short reel: payoff first, the name comes last
        return [script["hook"]] + script["sentences"] + [script["cta"], OUTRO_SHORT]
    return [script["hook"], INTRO] + script["sentences"] + [script["cta"], OUTRO]


def offset(script: dict) -> int:
    """1 when the intro is spoken right after the hook (long reels only)."""
    return 1 if script.get("with_brand", True) and not is_short(script) else 0


def has_outro(script: dict) -> bool:
    return bool(script.get("with_brand", True))


def say(text: str) -> str:
    """Text as the voice should read it. 'Shubh' is one word (rhymes with 'boob'),
    never the letters S-H-U-B-H."""
    import re
    return re.sub(r"\bShubh\b", "Shoob", text)
