"""Tiny synthetic videos for sampler and CLI smoke tests."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def write_color_video(
    path: Path,
    segments: list[tuple[str, float]],
    *,
    fps: int = 10,
    size: str = "160x90",
) -> None:
    """Write an MJPEG AVI made of solid-color stretches.

    ``segments`` is ``(ffmpeg_color, duration_seconds)``.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not on PATH")
    inputs: list[str] = []
    parts: list[str] = []
    for i, (color, dur) in enumerate(segments):
        inputs += [
            "-f", "lavfi", "-t", f"{dur}",
            "-i", f"color=c={color}:s={size}:r={fps}",
        ]
        parts.append(f"[{i}:v]")
    fc = "".join(parts) + f"concat=n={len(segments)}:v=1:a=0,format=yuv420p[v]"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex", fc,
            "-map", "[v]",
            "-c:v", "mjpeg", "-q:v", "3",
            str(path),
        ],
        check=True,
    )
