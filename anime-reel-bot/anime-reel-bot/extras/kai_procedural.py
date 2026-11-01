"""Kai: a constant 2D anime character drawn in code.

Every pixel is drawn from parameters (mouth, eyes, brows, head pose), so the
character is pixel-identical across videos and runs on a plain CPU runner with
no GPU and no image-generation API. Pillow only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter

SS = 2                      # supersampling factor for anti-aliasing
W, H = 1000, 1150           # design canvas (output px at 1x)

SKIN = (255, 225, 205)
SKIN_SH = (238, 194, 170)
HAIR = (33, 38, 82)
HAIR_HI = (74, 90, 160)
HOOD = (255, 122, 69)
HOOD_SH = (221, 94, 46)
HOOD_HI = (255, 156, 108)
IRIS = (62, 112, 255)
IRIS_DK = (28, 52, 160)
IRIS_LT = (130, 190, 255)
LINE = (26, 24, 54)
MOUTH_IN = (122, 34, 56)
TONGUE = (238, 110, 124)
WHITE = (255, 255, 255)


@dataclass
class Pose:
    mouth_open: float = 0.0     # 0 closed .. 1 wide open
    mouth_w: float = 0.6        # 0 round "oo" .. 1 wide "ee"
    smile: float = 0.2          # -1 frown .. 1 big smile
    eye_open: float = 1.0       # 0 closed .. 1.15 wide
    gaze_x: float = 0.0         # -1 left .. 1 right
    gaze_y: float = 0.0         # -1 up .. 1 down
    brow_raise: float = 0.0     # px up (negative = lowered)
    brow_tilt: float = 0.0      # >0 inner ends down (determined), <0 worried
    head_rot: float = 0.0       # degrees
    head_dx: float = 0.0
    head_dy: float = 0.0
    breath: float = 0.0         # px body rise
    shrug: float = 0.0          # px shoulder lift
    hair_sway: float = 0.0      # px lag of the bangs


def _s(v):
    return v * SS


def _pts(pts, dx=0.0, dy=0.0):
    return [(_s(x + dx), _s(y + dy)) for x, y in pts]


def _poly(d, pts, fill, dx=0.0, dy=0.0):
    d.polygon(_pts(pts, dx, dy), fill=fill)


def _ellipse(d, cx, cy, rx, ry, fill=None, outline=None, width=1):
    d.ellipse([_s(cx - rx), _s(cy - ry), _s(cx + rx), _s(cy + ry)],
              fill=fill, outline=outline, width=_s(width) if outline else 0)


def _stroke(d, pts, fill, width):
    """Thick polyline with round caps and joins."""
    sp = [(_s(x), _s(y)) for x, y in pts]
    d.line(sp, fill=fill, width=_s(width), joint="curve")
    r = _s(width) / 2
    for x, y in (sp[0], sp[-1]):
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _quad(p0, p1, p2, n=14):
    out = []
    for i in range(n + 1):
        t = i / n
        out.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]))
    return out


# --------------------------------------------------------------------------
# body (hoodie, neck) - drawn first, moves with breathing and shrugs
# --------------------------------------------------------------------------
def draw_body(p: Pose) -> Image.Image:
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lift = -p.breath - p.shrug
    # hood behind the neck
    _ellipse(d, 500, 790 + lift, 250, 105, fill=HOOD_SH)
    # neck
    _poly(d, [(446, 640), (554, 640), (562, 800 + lift), (438, 800 + lift)], SKIN_SH)
    # torso
    torso = [(70, H + 40), (96, 960 + lift), (190, 840 + lift), (330, 780 + lift),
             (440, 764 + lift), (560, 764 + lift), (670, 780 + lift),
             (810, 840 + lift), (904, 960 + lift), (930, H + 40)]
    _poly(d, torso, HOOD)
    # soft fold highlights
    _stroke(d, [(180, 990 + lift), (250, 900 + lift), (350, 850 + lift)], HOOD_HI, 10)
    _stroke(d, [(820, 990 + lift), (750, 900 + lift), (650, 850 + lift)], HOOD_SH, 10)
    # neckline (V cut showing skin) and hood rim
    _poly(d, [(430, 770 + lift), (570, 770 + lift), (500, 880 + lift)], SKIN_SH)
    _stroke(d, [(420, 760 + lift), (500, 890 + lift), (580, 760 + lift)], HOOD_SH, 12)
    # drawstrings
    for sx, ex in ((462, 452), (538, 548)):
        _stroke(d, [(sx, 840 + lift), (ex, 960 + lift)], (250, 245, 240), 8)
        _ellipse(d, ex, 966 + lift, 8, 8, fill=(250, 245, 240))
    # chest logo: tiny sparkle so Kai is recognisable even cropped
    cx, cy = 690, 1010 + lift
    for ang in (0, 90):
        a = math.radians(ang)
        dx, dy = 26 * math.cos(a), 26 * math.sin(a)
        _stroke(d, [(cx - dx, cy - dy), (cx + dx, cy + dy)], WHITE, 7)
    _ellipse(d, cx, cy, 6, 6, fill=WHITE)
    return img


# --------------------------------------------------------------------------
# head parts
# --------------------------------------------------------------------------
def _face_outline(cx=500, cy=470, rx=212, ry=236, taper=0.26, n=90):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        sa = math.sin(a)
        k = 1 - taper * (max(0.0, sa) ** 2.2)
        pts.append((cx + rx * math.cos(a) * k, cy + ry * sa))
    return pts


def _draw_back_hair(d, sway):
    # big mass behind the head with side tufts
    _ellipse(d, 500, 400, 262, 250, fill=HAIR)
    _poly(d, [(250, 420), (214, 560), (252, 700), (318, 610), (330, 470)], HAIR, dx=-sway * 0.2)
    _poly(d, [(750, 420), (786, 560), (748, 700), (682, 610), (670, 470)], HAIR, dx=-sway * 0.2)
    # spiky top
    spikes = [
        [(300, 240), (330, 120), (390, 220)],
        [(380, 215), (440, 70), (500, 205)],
        [(490, 205), (570, 60), (620, 215)],
        [(600, 220), (690, 110), (710, 250)],
    ]
    for s in spikes:
        _poly(d, s, HAIR, dx=sway * 0.3)


def _draw_ears(d):
    for sx in (-1, 1):
        cx = 500 + sx * 214
        _ellipse(d, cx, 520, 32, 44, fill=SKIN)
        _ellipse(d, cx - sx * 4, 522, 14, 24, fill=SKIN_SH)


def _blush(img):
    tile = Image.new("RGBA", (120 * SS, 70 * SS), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    td.ellipse([10 * SS, 10 * SS, 110 * SS, 60 * SS], fill=(255, 140, 150, 120))
    tile = tile.filter(ImageFilter.GaussianBlur(7 * SS))
    for cx in (340, 660):
        img.alpha_composite(tile, (_s(cx - 60), _s(572 - 35)))


def _eye(img, cx, cy, p: Pose, flip: int):
    """One anime eye. Rendered in a small tile and masked so the iris never
    leaves the eye opening, whatever the gaze or blink state."""
    o = max(0.0, min(1.2, p.eye_open))
    rx, ry = 60, 74 * o
    d_img = ImageDraw.Draw(img)
    if o < 0.16:
        # closed: a happy curved lash line
        _stroke(d_img, _quad((cx - rx, cy + 4), (cx, cy + 22), (cx + rx, cy + 4)), LINE, 11)
        _stroke(d_img, [(cx + flip * rx, cy + 4), (cx + flip * (rx + 18), cy - 8)], LINE, 8)
        return
    tw, th = 220, 220
    tile = Image.new("RGBA", (tw * SS, th * SS), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    mx, my = tw / 2, th / 2
    mask = Image.new("L", tile.size, 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([_s(mx - rx), _s(my - ry), _s(mx + rx), _s(my + ry)], fill=255)
    # sclera
    td.ellipse([_s(mx - rx), _s(my - ry), _s(mx + rx), _s(my + ry)], fill=(252, 252, 255, 255))
    # iris (follows gaze, kept inside the eye)
    ix = mx + p.gaze_x * 17
    iy = my + p.gaze_y * 12 + 4
    irx, iry = 55, min(72, ry + 4)
    td.ellipse([_s(ix - irx), _s(iy - iry), _s(ix + irx), _s(iy + iry)], fill=IRIS_DK)
    td.ellipse([_s(ix - irx + 4), _s(iy - iry * 0.45), _s(ix + irx - 4), _s(iy + iry - 2)], fill=IRIS)
    td.ellipse([_s(ix - irx * 0.62), _s(iy + iry * 0.1), _s(ix + irx * 0.62), _s(iy + iry - 6)], fill=IRIS_LT)
    td.ellipse([_s(ix - 22), _s(iy - 24), _s(ix + 22), _s(iy + 22)], fill=(14, 18, 52, 255))
    # highlights (these make the eyes look alive)
    hx, hy = ix - 17, iy - iry * 0.45
    td.ellipse([_s(hx - 15), _s(hy - 15), _s(hx + 15), _s(hy + 15)], fill=WHITE)
    td.ellipse([_s(ix + 17), _s(iy + 16), _s(ix + 27), _s(iy + 26)], fill=WHITE)
    tile.putalpha(Image.composite(tile.getchannel("A"), Image.new("L", tile.size, 0), mask))
    img.alpha_composite(tile, (int(round(_s(cx - mx))), int(round(_s(cy - my)))))
    # lid lines drawn over the tile
    top = [(cx + rx * math.cos(math.radians(a)), cy - ry * math.sin(math.radians(a)))
           for a in range(190, -11, -10)]
    _stroke(d_img, top, LINE, 15)
    # lash flick at the outer corner
    ox = cx + flip * rx
    _stroke(d_img, [(ox - flip * 6, cy - ry * 0.25), (ox + flip * 20, cy - ry * 0.55 - 8)], LINE, 9)



def _brows(d, p: Pose):
    base = 392 - p.brow_raise
    for sx in (-1, 1):
        ex = 500 + sx * 88
        outer = (ex + sx * 62, base + 8 + p.brow_tilt * -0.4)
        mid = (ex, base - 6 + p.brow_tilt * 0.2)
        inner = (ex - sx * 56, base + p.brow_tilt)
        _stroke(d, _quad(outer, mid, inner, 10), HAIR, 12)


def _nose(d):
    _stroke(d, [(500, 548), (493, 566), (505, 570)], (205, 150, 130), 4)


def _mouth(img, p: Pose):
    d = ImageDraw.Draw(img)
    cx, cy = 500, 628
    half = 22 + 26 * p.mouth_w - 8 * p.mouth_open * (1 - p.mouth_w)
    smile = p.smile
    op = max(0.0, min(1.0, p.mouth_open))
    if op < 0.07:
        pts = [(cx + x, cy - smile * 15 * (x / half) ** 2 + 6 * (1 - (x / half) ** 2) * (smile < 0))
               for x in [(-half + 2 * half * i / 24) for i in range(25)]]
        _stroke(d, pts, (150, 60, 74), 6)
        if smile > 0.5:   # cheeky corners
            for sx in (-1, 1):
                _stroke(d, [(cx + sx * half, cy - smile * 15), (cx + sx * (half + 8), cy - smile * 15 - 6)],
                        (150, 60, 74), 5)
        return
    height = 8 + 58 * op
    xs = [(-half + 2 * half * i / 28) for i in range(29)]
    upper = [(cx + x, cy - smile * 12 * (x / half) ** 2) for x in xs]
    lower = [(cx + x, cy - smile * 12 * (x / half) ** 2 + height * max(0.0, 1 - (x / half) ** 2) ** 0.6)
             for x in xs]
    shape = upper + lower[::-1]
    tw, th = int(half * 2 + 60), int(height + 80)
    ox, oy = cx - tw / 2, cy - 30 - 10
    tile = Image.new("RGBA", (tw * SS, th * SS), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    local = [((x - ox) * SS, (y - oy) * SS) for x, y in shape]
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).polygon(local, fill=255)
    td.polygon(local, fill=MOUTH_IN)
    # tongue
    td.ellipse([(cx - ox - half * 0.55) * SS, (cy - oy + height * 0.45) * SS,
                (cx - ox + half * 0.55) * SS, (cy - oy + height * 1.25) * SS], fill=TONGUE)
    # upper teeth
    if op > 0.22:
        th_h = min(14.0, 6 + 10 * op)
        td.rectangle([(cx - ox - half) * SS, (cy - oy - 14) * SS,
                      (cx - ox + half) * SS, (cy - oy + th_h) * SS], fill=WHITE)
    tile.putalpha(Image.composite(tile.getchannel("A"), Image.new("L", tile.size, 0), mask))
    img.alpha_composite(tile, (int(ox * SS), int(oy * SS)))
    d = ImageDraw.Draw(img)
    _stroke(d, upper, (150, 60, 74), 5)
    _stroke(d, lower, (150, 60, 74), 5)


def _front_hair(d, sway):
    # hair cap over the forehead
    cap = [(262, 470), (270, 330), (330, 240), (420, 200), (500, 190), (580, 200), (670, 240),
           (730, 330), (738, 470), (690, 400), (640, 330), (500, 300), (360, 330), (310, 400)]
    _poly(d, cap, HAIR)
    # bangs: sharp anime locks that lag behind head movement
    locks = [
        [(300, 330), (316, 438), (374, 452), (404, 330)],
        [(380, 300), (428, 448), (470, 408), (480, 300)],
        [(470, 300), (512, 424), (558, 452), (596, 300)],
        [(590, 300), (644, 440), (694, 446), (714, 330)],
    ]
    for lk in locks:
        pts = [(x + sway * (0.35 + 0.65 * max(0.0, (y - 300) / 160)), y) for x, y in lk]
        _poly(d, pts, HAIR)
    # glossy highlight band
    _stroke(d, _quad((340, 285), (500, 235), (660, 285)), HAIR_HI, 14)
    _stroke(d, [(395, 360), (408, 430)], HAIR_HI, 8)
    _stroke(d, [(560, 380), (572, 440)], HAIR_HI, 8)
    # hair clip (signature accessory)
    _stroke(d, [(318, 372), (368, 352)], (255, 214, 64), 11)
    _stroke(d, [(326, 394), (376, 374)], (255, 214, 64), 11)


def draw_head(p: Pose) -> Image.Image:
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    _draw_back_hair(d, p.hair_sway)
    _draw_ears(d)
    _poly(d, _face_outline(), SKIN)
    # chin shading
    _poly(d, [(x, y) for x, y in _face_outline(ry=236)][50:90] + [(500, 640)], SKIN)
    _blush(img)
    d = ImageDraw.Draw(img)
    _eye(img, 500 - 96, 508, p, -1)
    _eye(img, 500 + 96, 508, p, +1)
    d = ImageDraw.Draw(img)
    _brows(d, p)
    _nose(d)
    _mouth(img, p)
    d = ImageDraw.Draw(img)
    _front_hair(d, p.hair_sway)
    return img


_BODY_CACHE: dict = {}


def render_character(p: Pose) -> Image.Image:
    """Return Kai as an RGBA image of size (W, H).

    Parts are drawn supersampled, downsampled, then the head is rotated at 1x:
    rotating the 2x canvas was 10x slower for no visible gain.
    """
    key = round((p.breath + p.shrug) * 2) / 2
    body = _BODY_CACHE.get(key)
    if body is None:
        body = draw_body(Pose(breath=key)).resize((W, H), Image.LANCZOS)
        _BODY_CACHE[key] = body
    out = body.copy()
    head = draw_head(p).resize((W, H), Image.LANCZOS)
    head = head.rotate(-p.head_rot, resample=Image.BILINEAR, center=(500, 740))
    dx = int(round(p.head_dx))
    dy = int(round(p.head_dy - p.shrug * 0.5 - p.breath * 0.6))
    out.alpha_composite(head, (0, 0), ) if (dx == 0 and dy == 0) else _paste_shifted(out, head, dx, dy)
    return out


def _paste_shifted(dst: Image.Image, src: Image.Image, dx: int, dy: int) -> None:
    shifted = Image.new("RGBA", src.size, (0, 0, 0, 0))
    shifted.paste(src, (dx, dy))
    dst.alpha_composite(shifted)


if __name__ == "__main__":  # quick visual check: python -m reelbot.character out.png
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "kai_preview.png"
    poses = [
        Pose(),
        Pose(mouth_open=0.8, mouth_w=0.6, smile=0.3, brow_raise=10, head_rot=3),
        Pose(eye_open=0.0, smile=0.8),
        Pose(mouth_open=0.5, mouth_w=0.1, gaze_x=0.8, gaze_y=-0.6, head_rot=-3, brow_tilt=-10),
    ]
    sheet = Image.new("RGBA", (W * 2 // 2, H * 2 // 2 + 0), (225, 232, 255, 255))
    tiles = [render_character(p).resize((W // 2, H // 2), Image.LANCZOS) for p in poses]
    sheet = Image.new("RGBA", (W // 2 * 4, H // 2), (225, 232, 255, 255))
    for i, t in enumerate(tiles):
        sheet.alpha_composite(t, (i * (W // 2), 0))
    sheet.convert("RGB").save(out)
    print("saved", out)
