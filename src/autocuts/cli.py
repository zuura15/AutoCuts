"""autocuts — local scene splitter that drops short pan-outs."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .export import export_group, write_manifest
from .logic import detect_intervals, drop_short, stitch_across_drops


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="autocuts",
        description="Split a compilation video into scenes; drop short pan-outs; keep main action.",
    )
    p.add_argument("video", type=Path, help="Input video path")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory (default: <video_dir>/AutoCuts)",
    )
    p.add_argument(
        "--min-keep", type=float, default=4.0,
        help="Scenes shorter than this many seconds are dropped (default: 4)",
    )
    p.add_argument(
        "--detector", choices=("adaptive", "content"), default="adaptive",
        help="PySceneDetect detector (default: adaptive; better for dark/night-ish cuts)",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Detector threshold override (adaptive_threshold or content threshold)",
    )
    p.add_argument(
        "--no-stitch", action="store_true",
        help="Do not merge long scenes across short bursts; export each keep separately",
    )
    p.add_argument(
        "--copy", action="store_true",
        help="Try stream-copy for single-range exports (faster, less precise)",
    )
    p.add_argument("--version", action="version", version=f"autocuts {__version__}")
    args = p.parse_args(argv)

    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"error: not a file: {video}", file=sys.stderr)
        return 2

    out_dir = (args.output or (video.parent / "AutoCuts")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"AutoCuts {__version__}")
    print(f"input:    {video}")
    print(f"output:   {out_dir}")
    print(f"detector: {args.detector}  min-keep: {args.min_keep}s")
    t0 = time.time()
    intervals = detect_intervals(
        str(video),
        detector=args.detector,
        threshold=args.threshold,
        min_scene_len=0.6,
    )
    print(f"detected {len(intervals)} raw scenes in {time.time()-t0:.1f}s")

    keep, dropped = drop_short(intervals, args.min_keep)
    print(f"keep {len(keep)}  drop {len(dropped)} (short bursts)")

    if args.no_stitch:
        groups = [[iv] for iv in keep]
    else:
        groups = stitch_across_drops(intervals, min_keep=args.min_keep)
    print(f"export groups: {len(groups)}")

    stem = video.stem
    exported = []
    for i, group in enumerate(groups, start=1):
        name = f"{stem}_scene_{i:03d}.mp4"
        dest = out_dir / name
        print(f"  writing {name}  ({len(group)} range(s), {sum(g.duration for g in group):.1f}s kept)")
        export_group(video, group, dest, reencode=not args.copy)
        exported.append({
            "file": name,
            "ranges": [{"start": g.start, "end": g.end, "duration": g.duration} for g in group],
        })

    write_manifest(out_dir / f"{stem}_autocuts.json", {
        "version": __version__,
        "source": str(video),
        "min_keep": args.min_keep,
        "detector": args.detector,
        "raw_scenes": len(intervals),
        "dropped": [{"start": d.start, "end": d.end, "duration": d.duration} for d in dropped],
        "exports": exported,
    })
    print(f"done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
