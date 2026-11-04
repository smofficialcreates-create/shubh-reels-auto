"""The brand palette. One place, so every scene, card and cover agrees.

Hierarchy (share of the screen): Midnight Navy 50 %, Warm Ivory 20 %, Charcoal 15 %,
Forest Green 10 %, Antique Gold 5 %. Gold is the signature highlight and is used sparingly.
"""
from __future__ import annotations

from dataclasses import dataclass

NAVY = (20, 36, 58)          # #14243A  primary backgrounds
FOREST = (36, 72, 62)        # #24483E  growth, secondary sections
GOLD = (197, 161, 91)        # #C5A15B  signature highlights
IVORY = (241, 235, 221)      # #F1EBDD  text and light backgrounds
CHARCOAL = (41, 45, 50)      # #292D32  secondary dark surfaces
WHITE = (255, 255, 255)      # #FFFFFF  cards and interface elements
GREY = (167, 173, 181)       # #A7ADB5  secondary text


def mix(a, b, t):
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


@dataclass(frozen=True)
class Theme:
    name: str
    bg: tuple
    text: tuple          # headline colour on the background
    accent: tuple        # accent word colour on the background
    mono: tuple          # small labels
    marker: bool         # accent word sits on a gold highlighter stroke (light backgrounds)
    card: tuple          # card surface
    ink: tuple           # text on cards
    mute: tuple          # secondary text on cards
    c_fill: tuple        # accent fill on cards (badges, active rows, progress)
    c_text: tuple        # accent text on cards
    soft: tuple          # tint for the active row on a card
    on_fill: tuple       # text on an accent fill
    line: tuple          # hairlines on cards


NAVY_T = Theme("navy", NAVY, IVORY, GOLD, GREY, False,
               WHITE, NAVY, (112, 120, 132), GOLD, FOREST, mix(WHITE, GOLD, 0.16), NAVY, mix(WHITE, NAVY, 0.12))
IVORY_T = Theme("ivory", IVORY, NAVY, NAVY, CHARCOAL, True,
                CHARCOAL, IVORY, GREY, GOLD, GOLD, mix(CHARCOAL, GOLD, 0.20), NAVY, mix(CHARCOAL, IVORY, 0.18))
FOREST_T = Theme("forest", FOREST, IVORY, GOLD, mix(IVORY, FOREST, 0.35), False,
                 WHITE, NAVY, (112, 120, 132), GOLD, FOREST, mix(WHITE, GOLD, 0.16), NAVY, mix(WHITE, NAVY, 0.12))
THEMES = {"navy": NAVY_T, "ivory": IVORY_T, "forest": FOREST_T}
