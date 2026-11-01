"""Fixed brand lines spoken in every reel."""
INTRO = "Want to stay ahead of the curve? I'm Shubh. Let's upgrade the way you think, work and grow."
OUTRO = "Thanks for watching. Follow for more smart ways to work, and I'll see you in the next one."


def spoken(script: dict) -> list[str]:
    """Every spoken line in order: intro, hook, body, closing line, outro."""
    return [INTRO, script["hook"]] + script["sentences"] + [script["cta"], OUTRO]


def offset(script: dict) -> int:
    return 1 if script.get("with_brand", True) else 0


def say(text: str) -> str:
    """Text as the voice should read it. 'Shubh' is one word (rhymes with 'boob'),
    never the letters S-H-U-B-H."""
    import re
    return re.sub(r"\bShubh\b", "Shoob", text)
