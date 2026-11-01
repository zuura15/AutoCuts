"""ffmpeg export of kept ranges (drop gaps)."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence

from .logic import Interval


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def export_group(
    video: Path,
    ranges: Sequence[Interval],
    out_path: Path,
    *,
    reencode: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(ranges) == 1 and not reencode:
        iv = ranges[0]
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{iv.start:.3f}", "-to", f"{iv.end:.3f}",
            "-i", str(video), "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(out_path),
        ])
        return

    # Frame-accurate trim+concat (handles excised middles)
    parts_v = []
    parts_a = []
    maps = []
    for i, iv in enumerate(ranges):
        parts_v.append(
            f"[0:v]trim=start={iv.start:.3f}:end={iv.end:.3f},setpts=PTS-STARTPTS[v{i}];"
        )
        parts_a.append(
            f"[0:a]atrim=start={iv.start:.3f}:end={iv.end:.3f},asetpts=PTS-STARTPTS[a{i}];"
        )
        maps.append(f"[v{i}][a{i}]")
    n = len(ranges)
    fc = "".join(parts_v) + "".join(parts_a) + "".join(maps) + f"concat=n={n}:v=1:a=1[outv][outa]"
    try:
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-filter_complex", fc,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-movflags", "+faststart",
            str(out_path),
        ])
    except subprocess.CalledProcessError:
        # video-only fallback
        fc_v = "".join(
            f"[0:v]trim=start={iv.start:.3f}:end={iv.end:.3f},setpts=PTS-STARTPTS[v{i}];"
            for i, iv in enumerate(ranges)
        )
        fc_v += "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-filter_complex", fc_v,
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-an", "-movflags", "+faststart",
            str(out_path),
        ])


def write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
