"""Legacy drop/stitch behavior, kept behind --mode adaptive|content."""
import pytest

from autocuts.logic import Interval, detect_intervals, drop_short, stitch_across_drops
from tests.synth import ffmpeg_available, write_color_video


def test_drop_short_splits_keep_and_burst():
    intervals = [Interval(0, 10), Interval(10, 12), Interval(12, 12.5)]
    keep, dropped = drop_short(intervals, 4)
    assert keep == [Interval(0, 10)]
    assert len(dropped) == 2


def test_stitch_excises_a_short_burst_and_keeps_adjacent_scenes_apart():
    intervals = [
        Interval(0, 10),
        Interval(10, 12),
        Interval(12, 40),
        Interval(40, 55),
    ]
    groups = stitch_across_drops(intervals, min_keep=4)
    assert len(groups) == 2
    assert [(iv.start, iv.end) for iv in groups[0]] == [(0, 10), (12, 40)]
    assert [(iv.start, iv.end) for iv in groups[1]] == [(40, 55)]


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is required to synthesize a clip")
def test_legacy_detectors_still_split_a_hard_color_cut(tmp_path):
    video = tmp_path / "cut.avi"
    write_color_video(video, [("red", 2), ("green", 2)], fps=10)
    for detector in ("content", "adaptive"):
        intervals = detect_intervals(str(video), detector=detector, min_scene_len=0.3)
        assert len(intervals) == 2
        assert intervals[0].start == pytest.approx(0.0, abs=0.2)
        assert intervals[0].end == pytest.approx(2.0, abs=0.3)
        assert intervals[1].start == pytest.approx(intervals[0].end)
