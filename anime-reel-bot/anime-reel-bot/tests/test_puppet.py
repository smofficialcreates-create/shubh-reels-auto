import numpy as np

from reelbot.animator import Animator, PRESETS, lip_sync
from reelbot.puppet import Pose, Shubh


def test_render_transparent_and_consistent():
    s = Shubh()
    a = s.render_rgba(Pose(), size=(540, 960))
    b = s.render_rgba(Pose(r_sh=30, r_el=150, mouth_open=0.8, blink=0.5), size=(540, 960))
    assert a.shape == (960, 540, 4) and a[..., 3].min() == 0 and a[..., 3].max() == 255
    assert (b[..., 3] > 0).sum() > (a[..., 3] > 0).sum() * 0.8           # same character, no vanishing parts


def test_every_preset_renders():
    s = Shubh()
    for name in PRESETS:
        an = Animator().play(name, at=0.0)
        p = None
        for i in range(20):
            p = an.pose(i / 30)
        assert s.render_rgba(p, size=(270, 480))[..., 3].max() == 255


def test_lipsync_follows_loudness():
    sr = 24000
    t = np.arange(sr) / sr
    voice = np.where(t > 0.5, np.sin(2 * np.pi * 180 * t), 0).astype(np.float32)
    mo, _, _ = lip_sync(voice, sr, 30, 30)
    assert mo[:10].max() < 0.1 and mo[20:].max() > 0.5
