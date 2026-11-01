"""Last-occurrence: one first-to-last span per lasting look, junk left inside."""
from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from autocuts.cli import main
from autocuts.last_occurrence import (
    DEFAULT_MIN_LOOK,
    detect_last_occurrence_intervals,
    last_occurrence_scenes,
)
from autocuts.sticky import hsv_histogram
from tests.synth import ffmpeg_available, write_color_video

A = np.array([1.0, 0.0])
B = np.array([0.0, 1.0])
C = np.array([-1.0, 0.0])
F = np.array([0.0, -1.0])


def _repeat(vec, start, count, dt=1.0):
    return [(start + i * dt, vec) for i in range(count)]


def _run(samples, duration, **kwargs):
    opts = {"similarity": 0.55, "prototype_alpha": 0.05, "sample_interval": 1.0}
    opts.update(kwargs)
    return last_occurrence_scenes(samples, duration=duration, **opts)


def _span_pair(scene):
    return (scene.start, scene.end)


def test_empty_samples():
    assert last_occurrence_scenes([]) == []


def test_return_after_foreign_stretch_is_one_span_first_to_last():
    # A, then a lasting foreign look, then A again. One A file covers the middle.
    samples = _repeat(A, 0, 20) + _repeat(B, 20, 20) + _repeat(A, 40, 30)
    scenes = _run(samples, 70.0)
    assert len(scenes) == 2
    assert _span_pair(scenes[0]) == pytest.approx((0.0, 70.0))
    assert _span_pair(scenes[1]) == pytest.approx((20.0, 40.0))
    # The foreign stretch is inside A's file, not cut out.
    assert scenes[0].start < scenes[1].start < scenes[1].end < scenes[0].end


def test_look_that_never_returns_ends_at_its_last_samples():
    samples = _repeat(A, 0, 20) + _repeat(B, 20, 20) + _repeat(C, 40, 20)
    scenes = _run(samples, 100.0, min_look=8)
    assert len(scenes) == 3
    assert _span_pair(scenes[0]) == pytest.approx((0.0, 20.0))
    assert _span_pair(scenes[1]) == pytest.approx((20.0, 40.0))
    assert _span_pair(scenes[2]) == pytest.approx((40.0, 60.0))
    # B does not run on to the end of the video.
    assert scenes[1].end == pytest.approx(40.0)
    assert scenes[1].end < 100.0


def test_brief_blip_stays_inside_without_its_own_export():
    samples = _repeat(A, 0, 20) + _repeat(B, 20, 3) + _repeat(A, 23, 20)
    scenes = _run(samples, 43.0, min_look=DEFAULT_MIN_LOOK)
    assert len(scenes) == 1
    assert _span_pair(scenes[0]) == pytest.approx((0.0, 43.0))
    assert scenes[0].start <= 20.0 < 23.0 <= scenes[0].end


def test_nested_lasting_scene_overlaps_parent_and_a_later_return():
    # A, ~10s of distinct B, A for ~30s more, then A ends and later resumes.
    # A is one file first→last. B is its own file, inside A's span.
    samples = (
        _repeat(A, 0, 40)
        + _repeat(B, 40, 10)
        + _repeat(A, 50, 30)
        + _repeat(C, 80, 40)
        + _repeat(A, 120, 30)
    )
    scenes = _run(samples, 150.0, min_look=DEFAULT_MIN_LOOK)
    assert len(scenes) == 3
    by_start = sorted(scenes, key=lambda s: s.start)
    assert _span_pair(by_start[0]) == pytest.approx((0.0, 150.0))  # A
    assert _span_pair(by_start[1]) == pytest.approx((40.0, 50.0))  # B
    assert _span_pair(by_start[2]) == pytest.approx((80.0, 120.0))  # C
    parent, nested = by_start[0], by_start[1]
    assert parent.start <= nested.start and nested.end <= parent.end
    # B's file is the whole stretch between its own first and last sample,
    # not a trimmed fragment with the edges dropped.
    assert nested.duration == pytest.approx(10.0)


def test_transition_fluff_stays_inside_the_parent():
    # Fountain, then a mountain: neither lasts. Both sit inside A.
    mountain = C
    samples = (
        _repeat(A, 0, 30)
        + _repeat(F, 30, 2)
        + _repeat(mountain, 32, 3)
        + _repeat(A, 35, 25)
    )
    scenes = _run(samples, 60.0, min_look=8)
    assert len(scenes) == 1
    assert _span_pair(scenes[0]) == pytest.approx((0.0, 60.0))


def test_early_glimpse_plus_later_run_starts_at_the_first_glimpse():
    # The 2s glimpse is the same look as the later 12s run, so the file
    # starts at the glimpse and keeps the middle.
    samples = _repeat(B, 0, 2) + _repeat(A, 2, 30) + _repeat(B, 32, 12)
    scenes = _run(samples, 44.0, min_look=8)
    b_spans = [s for s in scenes if s.start == pytest.approx(0.0)]
    assert len(b_spans) == 1
    assert b_spans[0].end == pytest.approx(44.0)
    a_spans = [s for s in scenes if s.start == pytest.approx(2.0)]
    assert len(a_spans) == 1
    assert a_spans[0].end == pytest.approx(32.0)


def test_repeated_look_is_one_file():
    samples = _repeat(A, 0, 15) + _repeat(B, 15, 10) + _repeat(A, 25, 10) + _repeat(B, 35, 12)
    scenes = _run(samples, 47.0, min_look=8)
    assert len(scenes) == 2
    assert _span_pair(scenes[0]) == pytest.approx((0.0, 35.0))
    assert _span_pair(scenes[1]) == pytest.approx((15.0, 47.0))


def test_run_just_under_min_look_is_not_its_own_file():
    samples = _repeat(A, 0, 20) + _repeat(B, 20, 7) + _repeat(A, 27, 10)
    scenes = _run(samples, 37.0, min_look=8)
    assert len(scenes) == 1
    assert scenes[0].end == pytest.approx(37.0)


def test_run_exactly_min_look_is_kept():
    samples = _repeat(A, 0, 10) + _repeat(B, 10, 8) + _repeat(A, 18, 10)
    scenes = _run(samples, 28.0, min_look=8)
    assert len(scenes) == 2
    assert _span_pair(scenes[1]) == pytest.approx((10.0, 18.0))


def test_slow_drift_stays_one_look():
    angles = np.linspace(0.0, np.pi * 0.75, 90)
    samples = [
        (float(i), np.array([np.cos(a), np.sin(a)]))
        for i, a in enumerate(angles)
    ]
    scenes = _run(samples, 90.0, min_look=8, similarity=0.55, prototype_alpha=0.05)
    assert len(scenes) == 1
    assert _span_pair(scenes[0]) == pytest.approx((0.0, 90.0))


def test_hard_cut_is_not_merged_when_alpha_is_one():
    samples = _repeat(A, 0, 12) + _repeat(B, 12, 12)
    scenes = _run(samples, 24.0, min_look=8, prototype_alpha=1.0)
    assert len(scenes) == 2
    assert _span_pair(scenes[0]) == pytest.approx((0.0, 12.0))
    assert _span_pair(scenes[1]) == pytest.approx((12.0, 24.0))


def test_samples_must_be_ordered():
    with pytest.raises(ValueError):
        last_occurrence_scenes([(1.0, A), (0.0, B)], duration=2.0)


def _solid_hsv(hue: int, sat: int = 200, val: int = 140, shape=(72, 128)):
    hsv = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = sat
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_histogram_nested_palette_overlaps_and_a_brightness_blip_does_not():
    red = hsv_histogram(_solid_hsv(0))
    red_dark = hsv_histogram(_solid_hsv(0, val=40))
    green = hsv_histogram(_solid_hsv(60))
    # 20s red, 2s darker red (same look), 12s green, 15s red.
    samples = [(float(i), red) for i in range(20)]
    samples += [(float(i), red_dark) for i in range(20, 22)]
    samples += [(float(i), green) for i in range(22, 34)]
    samples += [(float(i), red) for i in range(34, 49)]
    scenes = _run(samples, 49.0, min_look=DEFAULT_MIN_LOOK)
    assert len(scenes) == 2
    assert scenes[0].start == pytest.approx(0.0)
    assert scenes[0].end == pytest.approx(49.0)
    assert scenes[1].start == pytest.approx(22.0)
    assert scenes[1].end == pytest.approx(34.0)
    assert scenes[0].start < scenes[1].start < scenes[1].end < scenes[0].end


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is required to synthesize a clip")
def test_sampled_clip_nests_a_real_scene_and_keeps_a_blip_inside(tmp_path):
    # 12s red, 10s green, 12s red, 3s blue, 10s red.
    video = tmp_path / "nested.avi"
    write_color_video(
        video,
        [("red", 12), ("green", 10), ("red", 12), ("blue", 3), ("red", 10)],
        fps=10,
    )
    scenes = detect_last_occurrence_intervals(
        str(video), min_look=8, similarity=0.4, sample_fps=1,
    )
    assert len(scenes) == 2
    red, green = sorted(scenes, key=lambda s: s.duration, reverse=True)
    assert red.start == pytest.approx(0.0, abs=0.6)
    assert red.end == pytest.approx(47.0, abs=1.5)
    assert green.start == pytest.approx(12.0, abs=1.5)
    assert green.end == pytest.approx(22.0, abs=1.5)
    assert red.start <= green.start and green.end <= red.end

    out = tmp_path / "out"
    assert main([
        str(video), "-o", str(out),
        "--min-look", "8", "--similarity", "0.4", "--copy",
    ]) == 0
    manifest = json.loads((out / "nested_autocuts.json").read_text())
    assert manifest["mode"] == "last-occurrence"
    assert len(manifest["exports"]) == 2
    parent = manifest["exports"][0]["ranges"]
    nested = manifest["exports"][1]["ranges"]
    assert len(parent) == 1 and len(nested) == 1
    assert parent[0]["start"] < nested[0]["start"] < nested[0]["end"] <= parent[0]["end"] + 0.6
    for item in manifest["exports"]:
        assert (out / item["file"]).stat().st_size > 0
