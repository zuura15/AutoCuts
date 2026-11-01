"""CLI defaults: last-occurrence is the path you get; sticky and adaptive stay available."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocuts import __version__
from autocuts.cli import main
from autocuts.logic import Interval
from tests.synth import ffmpeg_available, write_color_video


def test_version_matches_pyproject():
    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    assert f'version = "{__version__}"' in text
    assert __version__ == "0.3.0"


def test_help_documents_sticky_hold_out(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["-h"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--min-look" in out
    assert "--hold-out" in out
    assert "last-occurrence" in out
    assert "sticky" in out
    assert "--mode" in out


def test_hold_out_is_limited_to_15_30():
    for bad in ("14", "14.9", "30.1", "60"):
        with pytest.raises(SystemExit) as exc:
            main(["clip.mp4", "--hold-out", bad])
        assert exc.value.code == 2
    # Accepted by the parser; the missing file is a later check.
    for ok in ("15", "20", "30"):
        assert main(["/no/such/clip.mp4", "--hold-out", ok]) == 2


def test_mode_and_detector_must_agree():
    assert main(["clip.mp4", "--mode", "sticky", "--detector", "adaptive"]) == 2


def test_default_mode_is_last_occurrence(tmp_path, monkeypatch):
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"x")
    seen = {}

    def fake_detect(path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        # Nested looks: B sits inside A's first-to-last span. Both are exported.
        return [Interval(0.0, 80.0), Interval(20.0, 30.0)]

    monkeypatch.setattr("autocuts.cli.detect_last_occurrence_intervals", fake_detect)
    monkeypatch.setattr(
        "autocuts.cli.detect_sticky_intervals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("sticky")),
    )
    monkeypatch.setattr(
        "autocuts.cli.detect_intervals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy")),
    )
    monkeypatch.setattr("autocuts.cli.export_group", lambda *a, **k: None)

    out = tmp_path / "out"
    rc = main([
        str(video), "-o", str(out),
        "--min-look", "10", "--similarity", "0.5", "--sample-fps", "2",
    ])
    assert rc == 0
    assert seen["min_look"] == 10
    assert seen["similarity"] == 0.5
    assert seen["sample_fps"] == 2
    manifest = json.loads((out / "reel_autocuts.json").read_text())
    assert manifest["version"] == "0.3.0"
    assert manifest["mode"] == "last-occurrence"
    assert manifest["min_look"] == 10
    assert manifest["dropped"] == []
    assert [item["file"] for item in manifest["exports"]] == [
        "reel_scene_001.mp4",
        "reel_scene_002.mp4",
    ]
    # The nested look is not excised from the parent file.
    assert manifest["exports"][0]["ranges"] == [
        {"start": 0.0, "end": 80.0, "duration": 80.0},
    ]
    assert manifest["exports"][1]["ranges"] == [
        {"start": 20.0, "end": 30.0, "duration": 10.0},
    ]


def test_sticky_mode_still_available(tmp_path, monkeypatch):
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"x")
    seen = {}

    def fake_detect(path, **kwargs):
        seen.update(kwargs)
        return [Interval(0.0, 40.0)]

    monkeypatch.setattr("autocuts.cli.detect_sticky_intervals", fake_detect)
    monkeypatch.setattr(
        "autocuts.cli.detect_last_occurrence_intervals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("last-occurrence")),
    )
    monkeypatch.setattr("autocuts.cli.export_group", lambda *a, **k: None)

    out = tmp_path / "out"
    rc = main([
        str(video), "-o", str(out), "--mode", "sticky",
        "--hold-out", "25", "--similarity", "0.5",
    ])
    assert rc == 0
    assert seen["hold_out"] == 25
    assert seen["similarity"] == 0.5
    manifest = json.loads((out / "reel_autocuts.json").read_text())
    assert manifest["mode"] == "sticky"
    assert manifest["hold_out"] == 25


def test_adaptive_mode_still_drops_and_stitches(tmp_path, monkeypatch):
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"x")
    seen = {}

    def fake_detect(path, **kwargs):
        seen.update(kwargs)
        return [Interval(0.0, 10.0), Interval(10.0, 12.0), Interval(12.0, 30.0)]

    monkeypatch.setattr("autocuts.cli.detect_intervals", fake_detect)
    monkeypatch.setattr(
        "autocuts.cli.detect_sticky_intervals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("sticky")),
    )
    monkeypatch.setattr(
        "autocuts.cli.detect_last_occurrence_intervals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("last-occurrence")),
    )
    monkeypatch.setattr("autocuts.cli.export_group", lambda *a, **k: None)

    out = tmp_path / "out"
    rc = main([str(video), "-o", str(out), "--detector", "content", "--min-keep", "4"])
    assert rc == 0
    assert seen["detector"] == "content"
    manifest = json.loads((out / "reel_autocuts.json").read_text())
    assert manifest["mode"] == "content"
    assert manifest["stitch"] is True
    assert len(manifest["dropped"]) == 1
    assert len(manifest["exports"]) == 1
    assert len(manifest["exports"][0]["ranges"]) == 2

    out2 = tmp_path / "out2"
    rc = main([
        str(video), "-o", str(out2), "--mode", "adaptive", "--no-stitch", "--min-keep", "4",
    ])
    assert rc == 0
    manifest2 = json.loads((out2 / "reel_autocuts.json").read_text())
    assert manifest2["mode"] == "adaptive"
    assert manifest2["stitch"] is False
    assert len(manifest2["exports"]) == 2


def test_decode_failure_returns_1(tmp_path, monkeypatch):
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"x")

    def boom(path, **kwargs):
        raise RuntimeError("no frames decoded")

    monkeypatch.setattr("autocuts.cli.detect_last_occurrence_intervals", boom)
    assert main([str(video), "-o", str(tmp_path / "out")]) == 1


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is required to synthesize a clip")
def test_cli_sticky_on_a_real_clip(tmp_path):
    blip = tmp_path / "blip.avi"
    cut = tmp_path / "cut.avi"
    write_color_video(blip, [("red", 8), ("blue", 3), ("red", 8)], fps=10)
    write_color_video(cut, [("red", 6), ("0x228822", 18)], fps=10)

    blip_out = tmp_path / "blip_out"
    assert main([
        str(blip), "-o", str(blip_out),
        "--mode", "sticky", "--hold-out", "15", "--copy",
    ]) == 0
    blip_manifest = json.loads((blip_out / "blip_autocuts.json").read_text())
    assert len(blip_manifest["exports"]) == 1
    assert (blip_out / "blip_scene_001.mp4").stat().st_size > 0

    cut_out = tmp_path / "cut_out"
    assert main([
        str(cut), "-o", str(cut_out),
        "--mode", "sticky", "--hold-out", "15", "--similarity", "0.4", "--copy",
    ]) == 0
    cut_manifest = json.loads((cut_out / "cut_autocuts.json").read_text())
    assert len(cut_manifest["exports"]) == 2
    assert cut_manifest["exports"][0]["ranges"][0]["end"] == pytest.approx(6.0, abs=1.5)
    for item in cut_manifest["exports"]:
        assert (cut_out / item["file"]).stat().st_size > 0
