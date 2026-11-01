"""Font lookup with graceful fallbacks (apt: fonts-inter fonts-liberation fonts-dejavu-core)."""
from __future__ import annotations

import os
from functools import lru_cache

from PIL import ImageFont

CANDIDATES = {
    "sans": ["/usr/share/fonts/opentype/inter/Inter-Medium.otf", "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
             "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["/usr/share/fonts/opentype/inter/Inter-Bold.otf", "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf",
             "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "serif": ["/usr/share/fonts/truetype/google-fonts/Lora-Italic-Variable.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Italic.ttf"],
    "mono": ["/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
             "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}


@lru_cache(maxsize=None)
def path(kind: str) -> str:
    for p in CANDIDATES[kind]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no font for '{kind}'. apt install fonts-inter fonts-liberation fonts-dejavu-core")


@lru_cache(maxsize=256)
def get(kind: str, size: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path(kind), size)
    if kind == "serif" and "Variable" in path(kind):
        try:
            f.set_variation_by_name("Italic")
        except Exception:  # noqa: BLE001
            pass
    return f
