"""autocuts — split a compilation into scenes."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .export import export_group, write_manifest
from .last_occurrence import DEFAULT_MIN_LOOK, detect_last_occurrence_intervals
from .logic import detect_intervals, drop_short, stitch_across_drops
from .sticky import (
    DEFAULT_HOLD_OUT,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SIMILARITY,
    HOLD_OUT_MAX,
    HOLD_OUT_MIN,
    detect_sticky_intervals,
)

_MODES = ("last-occurrence", "sticky", "adaptive", "content")


def _bounded_float(name: str, lo: float, hi: float):
    def parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from exc
        if value < lo or value > hi:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {lo:g} and {hi:g} (got {value:g})"
            )
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autocuts",
        description=(
            "Split a compilation video into scenes. The default "
            "last-occurrence mode exports one file per lasting look, from "
            "the first time it appears to the last, and leaves the middle "
            "in place."
        ),
        epilog=(
            "examples:\n"
            "  autocuts video.mp4 -o ./out\n"
            "  autocuts video.mp4 --min-look 10 --similarity 0.5\n"
            "  autocuts video.mp4 --mode sticky --hold-out 20\n"
            "  autocuts video.mp4 --mode adaptive --min-keep 4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", type=Path, help="Input video path")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory (default: <video_dir>/AutoCuts)",
    )
    p.add_argument(
        "--mode", choices=_MODES, default=None,
        help=(
            "last-occurrence (default), sticky, or adaptive/content "
            "for the PySceneDetect path"
        ),
    )
    p.add_argument(
        "--detector", choices=_MODES, default=None,
        help="Alias of --mode",
    )
    p.add_argument(
        "--min-look",
        type=_bounded_float("min-look", 1e-3, 600.0),
        default=None,
        help=(
            "Last-occurrence only. Contiguous seconds a look must last to "
            f"get its own file (default {DEFAULT_MIN_LOOK:g}). Shorter "
            "stretches stay inside other scenes."
        ),
    )
    p.add_argument(
        "--hold-out",
        type=_bounded_float("hold-out", HOLD_OUT_MIN, HOLD_OUT_MAX),
        default=None,
        help=(
            "Sticky only. Seconds a new look must last before the scene "
            f"closes ({HOLD_OUT_MIN:g}–{HOLD_OUT_MAX:g}, default {DEFAULT_HOLD_OUT:g}). "
            "Shorter blips stay inside the scene."
        ),
    )
    p.add_argument(
        "--similarity",
        type=_bounded_float("similarity", 0.0, 1.0),
        default=None,
        help=(
            "Last-occurrence and sticky. Minimum cosine similarity to a "
            f"look's prototype (default {DEFAULT_SIMILARITY:g}). Lower "
            "under-splits more."
        ),
    )
    p.add_argument(
        "--sample-fps",
        type=_bounded_float("sample-fps", 1e-3, 30.0),
        default=None,
        help=(
            "Last-occurrence and sticky. Signature samples per second "
            f"(default {DEFAULT_SAMPLE_FPS:g})"
        ),
    )
    p.add_argument(
        "--min-keep", type=float, default=4.0,
        help="Adaptive/content only. Drop scenes shorter than this many seconds (default: 4)",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Adaptive/content only. Detector threshold override",
    )
    p.add_argument(
        "--no-stitch", action="store_true",
        help="Adaptive/content only. Do not merge long scenes across short bursts",
    )
    p.add_argument(
        "--copy", action="store_true",
        help="Try stream-copy for single-range exports (faster, less precise)",
    )
    p.add_argument("--version", action="version", version=f"autocuts {__version__}")
    return p


def _resolve_mode(args: argparse.Namespace) -> str | None:
    if args.mode is not None and args.detector is not None and args.mode != args.detector:
        print("error: --mode and --detector disagree", file=sys.stderr)
        return None
    return args.mode or args.detector or "last-occurrence"


def _progress(at: float, duration: float) -> None:
    print(f"  sampling {at:.0f}s / {duration:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = _resolve_mode(args)
    if mode is None:
        return 2

    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"error: not a file: {video}", file=sys.stderr)
        return 2

    out_dir = (args.output or (video.parent / "AutoCuts")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hold_out = DEFAULT_HOLD_OUT if args.hold_out is None else args.hold_out
    similarity = DEFAULT_SIMILARITY if args.similarity is None else args.similarity
    sample_fps = DEFAULT_SAMPLE_FPS if args.sample_fps is None else args.sample_fps
    min_look = DEFAULT_MIN_LOOK if args.min_look is None else args.min_look

    print(f"AutoCuts {__version__}")
    print(f"input:  {video}")
    print(f"output: {out_dir}")

    t0 = time.time()
    dropped: list = []
    if mode == "last-occurrence":
        print(
            f"mode:   last-occurrence  min-look: {min_look:g}s  "
            f"similarity: {similarity:g}  sample: {sample_fps:g} fps"
        )
        try:
            intervals = detect_last_occurrence_intervals(
                str(video),
                min_look=min_look,
                similarity=similarity,
                sample_fps=sample_fps,
                on_progress=_progress,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"lasting looks: {len(intervals)} in {time.time() - t0:.1f}s")
        # One continuous span per look. Do not cut the middle out.
        groups = [[iv] for iv in intervals]
    elif mode == "sticky":
        print(
            f"mode:   sticky  hold-out: {hold_out:g}s  "
            f"similarity: {similarity:g}  sample: {sample_fps:g} fps"
        )
        try:
            intervals = detect_sticky_intervals(
                str(video),
                hold_out=hold_out,
                similarity=similarity,
                sample_fps=sample_fps,
                on_progress=_progress,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"sticky scenes: {len(intervals)} in {time.time() - t0:.1f}s")
        groups = [[iv] for iv in intervals]
    elif mode in ("adaptive", "content"):
        print(f"mode:   {mode}  min-keep: {args.min_keep:g}s")
        intervals = detect_intervals(
            str(video),
            detector=mode,
            threshold=args.threshold,
            min_scene_len=0.6,
        )
        print(f"detected {len(intervals)} raw scenes in {time.time() - t0:.1f}s")
        _keep, dropped = drop_short(intervals, args.min_keep)
        print(f"keep {len(_keep)}  drop {len(dropped)} (short bursts)")
        if args.no_stitch:
            groups = [[iv] for iv in _keep]
        else:
            groups = stitch_across_drops(intervals, min_keep=args.min_keep)
    else:
        print(f"error: unknown mode: {mode}", file=sys.stderr)
        return 2
    print(f"export groups: {len(groups)}")

    stem = video.stem
    exported = []
    for i, group in enumerate(groups, start=1):
        name = f"{stem}_scene_{i:03d}.mp4"
        dest = out_dir / name
        span = f"{group[0].start:.1f}-{group[-1].end:.1f}s"
        kept = sum(g.duration for g in group)
        print(f"  writing {name}  ({span}, {len(group)} range(s), {kept:.1f}s kept)")
        export_group(video, group, dest, reencode=not args.copy)
        exported.append({
            "file": name,
            "ranges": [{"start": g.start, "end": g.end, "duration": g.duration} for g in group],
        })

    manifest = {
        "version": __version__,
        "source": str(video),
        "mode": mode,
        "detector": mode,
        "raw_scenes": len(intervals),
        "dropped": [{"start": d.start, "end": d.end, "duration": d.duration} for d in dropped],
        "exports": exported,
    }
    if mode == "last-occurrence":
        manifest["min_look"] = min_look
        manifest["similarity"] = similarity
        manifest["sample_fps"] = sample_fps
    elif mode == "sticky":
        manifest["hold_out"] = hold_out
        manifest["similarity"] = similarity
        manifest["sample_fps"] = sample_fps
    else:
        manifest["min_keep"] = args.min_keep
        manifest["threshold"] = args.threshold
        manifest["stitch"] = not args.no_stitch
    write_manifest(out_dir / f"{stem}_autocuts.json", manifest)
    print(f"done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
