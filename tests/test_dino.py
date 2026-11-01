"""DINOv2 dry-run: group embeddings, report frames, write no scene files."""
from __future__ import annotations

import json

import numpy as np
import pytest

from autocuts.cli import main
from autocuts.dino import (
    DEFAULT_DINO_GAP,
    DEFAULT_DINO_SIMILARITY,
    Stretch,
    build_dino_report,
    filter_stretches,
    format_dino_summary,
    preprocess_bgr,
    resolve_device,
)
from autocuts.last_occurrence import DEFAULT_MIN_LOOK


A = np.array([1.0, 0.0])
B = np.array([0.0, 1.0])
C = np.array([-1.0, 0.0])


def _timeline(pieces, dt=1.0, fps=10.0):
    samples = []
    frames = []
    t = 0.0
    for vec, count in pieces:
        for _ in range(count):
            samples.append((t, vec))
            frames.append(int(round(t * fps)))
            t += dt
    return samples, frames, t


def _report(pieces, **kwargs):
    samples, frames, duration = _timeline(pieces)
    opts = {
        "frames": frames,
        "native_fps": 10.0,
        "frame_count": int(round(duration * 10)),
        "duration": duration,
        "min_look": DEFAULT_MIN_LOOK,
        "similarity": DEFAULT_DINO_SIMILARITY,
        "sample_interval": 1.0,
        "sample_fps": 1.0,
    }
    opts.update(kwargs)
    if "duration" in kwargs and kwargs["duration"] is not None:
        opts["frame_count"] = int(round(kwargs["duration"] * 10))
    return build_dino_report(samples, **opts)


def test_return_after_foreign_stretch_is_one_span_with_frames():
    report = _report([(A, 40), (B, 30), (A, 20)], umbrella=True)
    assert report["dry_run"] is True
    assert report["mode"] == "dino"
    assert len(report["looks"]) == 2
    parent, nested = report["looks"]
    assert parent["start"] == pytest.approx(0.0)
    assert parent["end"] == pytest.approx(90.0)
    assert parent["start_frame"] == 0
    assert parent["end_frame"] == 900
    assert len(parent["runs"]) == 2
    assert nested["start"] == pytest.approx(40.0)
    assert nested["end"] == pytest.approx(70.0)
    assert nested["duration"] == pytest.approx(30.0)
    assert nested["start_frame"] == 400
    assert nested["end_frame"] == 700
    assert nested["nested_in"] == [parent["id"]]
    assert parent["contains"] == [nested["id"]]
    assert parent["start"] < nested["start"] < nested["end"] < parent["end"]


def test_look_that_never_returns_stops_at_its_last_samples():
    report = _report([(A, 40), (B, 30), (C, 20)], duration=250.0, min_look=20)
    ends = [look["end"] for look in report["looks"]]
    assert ends == pytest.approx([40.0, 70.0, 90.0])
    assert all(end < 250.0 for end in ends)
    assert report["looks"][2]["end_frame"] == 900


def test_brief_blip_is_skipped_and_stays_inside_the_parent():
    report = _report([(A, 40), (B, 3), (A, 15)])
    assert len(report["looks"]) == 1
    parent = report["looks"][0]
    assert parent["start"] == pytest.approx(0.0)
    assert parent["end"] == pytest.approx(58.0)
    assert len(report["skipped_looks"]) == 1
    blip = report["skipped_looks"][0]
    assert blip["start"] == pytest.approx(40.0)
    assert blip["duration"] == pytest.approx(3.0)
    assert parent["start"] <= blip["start"] < blip["end"] <= parent["end"]


def test_default_min_look_keeps_30s_and_drops_29s():
    short = _report([(A, 40), (B, 29), (A, 10)], umbrella=True)
    assert len(short["looks"]) == 1
    assert len(short["skipped_looks"]) == 1
    assert short["min_look"] == 30.0

    held = _report([(A, 40), (B, 30), (A, 10)], umbrella=True)
    assert len(held["looks"]) == 2
    assert held["looks"][1]["duration"] == pytest.approx(30.0)
    assert held["looks"][1]["nested_in"] == [held["looks"][0]["id"]]


def test_summary_names_frames_and_nesting():
    report = _report([(A, 40), (B, 30), (A, 10)], umbrella=True)
    text = format_dino_summary(report)
    assert "frames 0-800" in text
    assert "frames 400-700" in text
    assert "nested in 1" in text
    assert "contains 2" in text


def _piece(look_id, start, end):
    return Stretch(look_id, start, end)


def test_filter_drops_the_first_last_umbrella():
    # A returns long after the gap limit, so the 0–250 umbrella is not a span.
    kept, skipped = filter_stretches(
        [_piece(1, 0, 40), _piece(1, 200, 250), _piece(2, 40, 80)],
        gap=60,
        min_look=30,
    )
    assert [(item.look_id, item.start, item.end) for item in kept] == [
        (1, 0, 40),
        (2, 40, 80),
        (1, 200, 250),
    ]
    assert skipped == []
    assert all(item.end - item.start < 250 for item in kept)


def test_filter_rejoins_a_short_gap():
    kept, skipped = filter_stretches(
        [_piece(1, 0, 40), _piece(2, 40, 55), _piece(1, 55, 100)],
        gap=60,
        min_look=30,
    )
    assert len(kept) == 1
    assert kept[0].look_id == 1
    assert (kept[0].start, kept[0].end) == (0, 100)
    assert len(skipped) == 1
    assert skipped[0].reason == "overlap"
    # A hole of exactly 60s still rejoins; 61s stays split.
    joined, _ = filter_stretches(
        [_piece(1, 0, 40), _piece(1, 100, 140)],
        gap=60,
        min_look=30,
    )
    assert len(joined) == 1
    assert (joined[0].start, joined[0].end) == (0, 140)
    split, _ = filter_stretches(
        [_piece(1, 0, 40), _piece(1, 101, 141)],
        gap=60,
        min_look=30,
    )
    assert [(item.start, item.end) for item in split] == [(0, 40), (101, 141)]


def test_filter_greedy_keeps_the_longest_nonoverlapping():
    kept, skipped = filter_stretches(
        [
            _piece(1, 0, 100),
            _piece(2, 10, 50),
            _piece(3, 110, 150),
        ],
        gap=60,
        min_look=30,
    )
    assert [(item.look_id, item.start, item.end) for item in kept] == [
        (1, 0, 100),
        (3, 110, 150),
    ]
    assert [(item.look_id, item.reason) for item in skipped] == [(2, "overlap")]


def test_filter_min_look_drops_a_short_survivor():
    kept, skipped = filter_stretches(
        [_piece(1, 0, 29), _piece(2, 100, 130)],
        gap=60,
        min_look=30,
    )
    assert [(item.look_id, item.start, item.end) for item in kept] == [(2, 100, 130)]
    assert [(item.look_id, item.reason) for item in skipped] == [(1, "short")]


def test_default_report_uses_the_stretch_filter():
    report = _report([(A, 40), (B, 30), (A, 20)])
    assert report["filter"] == "stretch"
    assert report["dino_gap"] == DEFAULT_DINO_GAP
    assert report["similarity"] == pytest.approx(0.60)
    assert len(report["looks"]) == 1
    parent = report["looks"][0]
    assert parent["start"] == pytest.approx(0.0)
    assert parent["end"] == pytest.approx(90.0)
    assert parent["start_frame"] == 0
    assert parent["end_frame"] == 900
    assert len(parent["runs"]) == 2
    assert report["looks"][0]["nested_in"] == []
    assert len(report["skipped_looks"]) == 1
    assert report["skipped_looks"][0]["reason"] == "overlap"


def test_preprocess_is_imagenet_chw():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = (40, 80, 200)
    arr = preprocess_bgr(frame)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32


def test_cpu_device_does_not_need_a_gpu(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match=r"\[dino\]"):
        resolve_device("auto")


def test_dino_mode_writes_a_report_and_no_scene_files(tmp_path, monkeypatch, capsys):
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"x")
    seen = {}

    def fake_report(path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        return {
            "version": "0.4.0",
            "mode": "dino",
            "dry_run": True,
            "looks": [{
                "id": 1,
                "start": 0.0,
                "end": 40.0,
                "duration": 40.0,
                "start_frame": 0,
                "end_frame": 1200,
                "nested_in": [],
                "contains": [],
                "runs": [{"start": 0.0, "end": 40.0}],
            }],
            "skipped_looks": [],
        }

    monkeypatch.setattr("autocuts.cli.detect_dino_report", fake_report)
    monkeypatch.setattr(
        "autocuts.cli.export_group",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("exported a clip")),
    )

    out = tmp_path / "out"
    assert main([
        str(video), "-o", str(out), "--mode", "dino", "--min-look", "30",
    ]) == 0
    assert seen["min_look"] == 30
    assert seen["similarity"] == pytest.approx(0.60)
    assert seen["gap"] == DEFAULT_DINO_GAP
    assert seen["umbrella"] is False
    assert seen["model_name"] == "dinov2_vits14"
    report_path = out / "reel_dino.json"
    report = json.loads(report_path.read_text())
    assert report["dry_run"] is True
    assert report["looks"][0]["start_frame"] == 0
    assert report["looks"][0]["end_frame"] == 1200
    assert list(out.glob("*.mp4")) == []
    printed = capsys.readouterr().out
    assert "frames 0-1200" in printed
    assert "no scene files written" in printed
