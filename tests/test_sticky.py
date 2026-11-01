"""Sticky hold-out: blips stay, a sustained new look becomes the next scene."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from autocuts.sticky import (
    DEFAULT_SIMILARITY,
    cosine_similarity,
    detect_sticky_intervals,
    hsv_histogram,
    sample_video,
    sticky_scenes,
)
from tests.synth import ffmpeg_available, write_color_video

A = np.array([1.0, 0.0])
B = np.array([0.0, 1.0])


def _cover(scenes, duration):
    assert scenes
    assert scenes[0].start == pytest.approx(0.0)
    assert scenes[-1].end == pytest.approx(duration)
    for left, right in zip(scenes, scenes[1:]):
        assert left.end == pytest.approx(right.start)
        assert left.duration > 0


def _run(samples, duration, **kwargs):
    opts = {"hold_out": 20.0, "similarity": 0.55, "prototype_alpha": 0.05}
    opts.update(kwargs)
    return sticky_scenes(samples, duration=duration, **opts)


def _repeat(vec, start, count, dt=1.0):
    return [(start + i * dt, vec) for i in range(count)]


def test_empty_and_single_sample():
    assert sticky_scenes([]) == []
    scenes = sticky_scenes([(0.0, A)], duration=12.0)
    assert len(scenes) == 1
    _cover(scenes, 12.0)


def test_constant_signature_is_one_scene():
    samples = _repeat(A, 0.0, 50)
    scenes = _run(samples, 50.0)
    assert len(scenes) == 1
    _cover(scenes, 50.0)


def test_brief_blip_does_not_split():
    # 5s unlike, then back. Well under a 20s hold-out.
    samples = _repeat(A, 0, 40) + _repeat(B, 40, 5) + _repeat(A, 45, 30)
    scenes = _run(samples, 75.0)
    assert len(scenes) == 1
    _cover(scenes, 75.0)


def test_blip_just_under_hold_out_does_not_split():
    # Unlike from t=40 through t=59 (19s), then the original look returns.
    samples = _repeat(A, 0, 40) + _repeat(B, 40, 20) + _repeat(A, 60, 20)
    scenes = _run(samples, 80.0, hold_out=20)
    assert len(scenes) == 1


def test_sustained_change_splits_at_first_unlike_sample():
    samples = _repeat(A, 0, 40) + _repeat(B, 40, 50)
    scenes = _run(samples, 90.0, hold_out=20)
    assert len(scenes) == 2
    assert scenes[0].end == pytest.approx(40.0)
    assert scenes[1].start == pytest.approx(40.0)
    _cover(scenes, 90.0)


def test_exact_hold_out_boundary_splits():
    # First unlike at t=40, confirming sample at t=60 (elapsed == 20).
    samples = _repeat(A, 0, 40) + _repeat(B, 40, 21)
    scenes = _run(samples, 61.0, hold_out=20)
    assert len(scenes) == 2
    assert scenes[0].end == pytest.approx(40.0)


def test_return_after_confirmed_cut_splits_again():
    # B lasts through the hold-out, then A returns and holds.
    samples = _repeat(A, 0, 40) + _repeat(B, 40, 21) + _repeat(A, 61, 40)
    scenes = _run(samples, 101.0, hold_out=20)
    assert [s.start for s in scenes] == pytest.approx([0.0, 40.0, 61.0])
    _cover(scenes, 101.0)


def test_two_palette_changes_make_three_scenes():
    C = np.array([-1.0, 0.0])
    samples = _repeat(A, 0, 30) + _repeat(B, 30, 30) + _repeat(C, 60, 30)
    scenes = _run(samples, 90.0, hold_out=15)
    assert [s.start for s in scenes] == pytest.approx([0.0, 30.0, 60.0])
    _cover(scenes, 90.0)


def test_interleaved_short_cuts_stay_one_scene():
    # Scenes are assumed not to interleave for long: 10s visits do not close.
    samples = (
        _repeat(A, 0, 30)
        + _repeat(B, 30, 10)
        + _repeat(A, 40, 10)
        + _repeat(B, 50, 10)
        + _repeat(A, 60, 30)
    )
    scenes = _run(samples, 90.0, hold_out=20)
    assert len(scenes) == 1


def test_hold_out_is_time_not_sample_count():
    # 5s grid. Unlike at t=5,15,25,... The window is met at t=25, not after 20 hits.
    samples = [(0.0, A)] + [(float(t), B) for t in (5, 10, 15, 20, 25, 30)]
    scenes = _run(samples, 35.0, hold_out=20)
    assert len(scenes) == 2
    assert scenes[0].end == pytest.approx(5.0)


def test_one_unlike_sample_does_not_split_on_unsampled_tail():
    samples = _repeat(A, 0, 21) + [(21.0, B)]
    scenes = _run(samples, 80.0, hold_out=20)
    assert len(scenes) == 1
    assert scenes[0].end == pytest.approx(80.0)


def test_slow_drift_stays_one_scene():
    angles = np.linspace(0.0, np.pi * 0.75, 90)
    samples = [
        (float(i), np.array([np.cos(a), np.sin(a)]))
        for i, a in enumerate(angles)
    ]
    scenes = _run(samples, 90.0, hold_out=20, similarity=0.55, prototype_alpha=0.05)
    assert len(scenes) == 1


def test_hard_cut_is_not_absorbed_when_alpha_is_one():
    # alpha=1 would swallow a new look immediately if unlike samples updated
    # the prototype. They must not.
    samples = _repeat(A, 0, 10) + _repeat(B, 10, 30)
    scenes = _run(samples, 40.0, hold_out=10, prototype_alpha=1.0)
    assert len(scenes) == 2
    assert scenes[0].end == pytest.approx(10.0)


def test_huge_matching_sample_does_not_yank_prototype():
    odd = np.array([0.8, 0.6]) * 1e6  # cosine 0.8 with A, huge magnitude
    follow = np.array([0.75, -np.sqrt(1.0 - 0.75**2)])  # matches A, not `odd`
    assert cosine_similarity(A, odd) >= 0.55
    assert cosine_similarity(odd, follow) < 0.55
    assert cosine_similarity(A, follow) >= 0.55
    samples = [(0.0, A), (1.0, odd)] + _repeat(follow, 2.0, 30)
    scenes = _run(samples, 32.0, hold_out=10, similarity=0.55, prototype_alpha=0.05)
    assert len(scenes) == 1


def test_threshold_equality_counts_as_a_match():
    same = np.array([0.6, 0.8])  # already unit length, cosine 0.6 with A
    samples = [(0.0, A)] + _repeat(same, 1.0, 40)
    assert cosine_similarity(A, same) == pytest.approx(0.6)
    scenes = _run(samples, 41.0, similarity=0.6, hold_out=10)
    assert len(scenes) == 1


def test_samples_must_be_ordered():
    with pytest.raises(ValueError):
        sticky_scenes([(1.0, A), (0.0, A)], duration=2.0)


def _solid_hsv(hue: int, sat: int = 180, val: int = 140, shape=(72, 128)):
    hsv = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = sat
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _scene_frame(zoom: float = 1.0, value: float = 1.0):
    img = np.zeros((180, 320, 3), np.float32)
    img[:] = (70, 55, 40)
    img[40:150, 90:250] = (40, 145, 225)
    img[16:55, 16:80] = (165, 175, 55)
    img[145:175, 240:310] = (90, 40, 140)
    if zoom != 1.0:
        h, w = img.shape[:2]
        ch, cw = max(1, int(round(h / zoom))), max(1, int(round(w / zoom)))
        y, x = (h - ch) // 2, (w - cw) // 2
        img = cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(img * value, 0, 255).astype(np.uint8)


def _outdoor():
    img = np.zeros((180, 320, 3), np.uint8)
    img[:] = (190, 130, 50)
    img[100:180, :] = (50, 150, 60)
    img[30:90, 140:200] = (30, 40, 210)
    return img


def test_histogram_ignores_brightness_and_wraps_hue():
    red = _solid_hsv(0, val=120)
    red_dark = _solid_hsv(0, val=40)
    red_wrap = _solid_hsv(179, val=120)
    green = _solid_hsv(60, val=120)
    assert cosine_similarity(hsv_histogram(red), hsv_histogram(red_dark)) > 0.95
    assert cosine_similarity(hsv_histogram(red), hsv_histogram(red_wrap)) > 0.9
    assert cosine_similarity(hsv_histogram(red), hsv_histogram(green)) < 0.2


def test_histogram_keeps_zoom_and_light_splits_new_palette():
    base = _scene_frame()
    hb = hsv_histogram(base)
    assert cosine_similarity(hb, hsv_histogram(_scene_frame(value=0.45))) > 0.9
    assert cosine_similarity(hb, hsv_histogram(_scene_frame(value=1.6))) > 0.9
    assert cosine_similarity(hb, hsv_histogram(_scene_frame(zoom=1.5))) >= DEFAULT_SIMILARITY
    assert cosine_similarity(hb, hsv_histogram(_scene_frame(zoom=2.0))) >= DEFAULT_SIMILARITY
    assert cosine_similarity(hb, hsv_histogram(_outdoor())) < 0.3


def test_held_zoom_stays_one_scene_palette_change_does_not():
    # A punch-in that stays should not close the scene. A new palette that
    # stays for the hold-out should, and the cut sits at the first new sample.
    frames = [_scene_frame(zoom=z) for z in np.linspace(1.0, 2.2, 40)]
    zoom_samples = [(float(i), hsv_histogram(f)) for i, f in enumerate(frames)]
    zoom_scenes = _run(zoom_samples, 40.0, hold_out=10, similarity=DEFAULT_SIMILARITY)
    assert len(zoom_scenes) == 1

    base_h = hsv_histogram(_scene_frame())
    out_h = hsv_histogram(_outdoor())
    blip = [(float(i), base_h) for i in range(30)]
    blip += [(float(i), out_h) for i in range(30, 35)]
    blip += [(float(i), base_h) for i in range(35, 55)]
    assert len(_run(blip, 55.0, hold_out=10, similarity=DEFAULT_SIMILARITY)) == 1

    change = [(float(i), base_h) for i in range(20)]
    change += [(float(i), out_h) for i in range(20, 50)]
    scenes = _run(change, 50.0, hold_out=15, similarity=DEFAULT_SIMILARITY)
    assert len(scenes) == 2
    assert scenes[0].end == pytest.approx(20.0)
    _cover(scenes, 50.0)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is required to synthesize a clip")
def test_sampled_blip_stays_one_scene_and_sustained_cut_splits(tmp_path):
    blip = tmp_path / "blip.avi"
    cut = tmp_path / "cut.avi"
    write_color_video(blip, [("red", 6), ("blue", 2), ("red", 6)], fps=10)
    write_color_video(cut, [("red", 8), ("green", 10)], fps=10)

    blip_scenes = detect_sticky_intervals(str(blip), hold_out=4, similarity=0.4, sample_fps=2)
    assert len(blip_scenes) == 1
    assert blip_scenes[0].start == pytest.approx(0.0)
    assert blip_scenes[0].end == pytest.approx(14.0, abs=0.6)

    cut_scenes = detect_sticky_intervals(str(cut), hold_out=4, similarity=0.4, sample_fps=2)
    assert len(cut_scenes) == 2
    assert cut_scenes[0].end == pytest.approx(8.0, abs=1.0)
    assert cut_scenes[1].start == pytest.approx(cut_scenes[0].end)
    assert cut_scenes[1].end == pytest.approx(18.0, abs=0.6)

    samples, _duration = sample_video(str(cut), sample_fps=2)
    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    assert gaps
    assert max(gaps) == pytest.approx(0.5, abs=0.15)


def test_unreadable_video_raises(tmp_path):
    bogus = tmp_path / "nope.mp4"
    bogus.write_bytes(b"this is not a video")
    with pytest.raises(RuntimeError):
        sample_video(str(bogus))
