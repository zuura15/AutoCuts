"""DINOv2 dry-run look report.

Embed a coarse sample of frames, assign look identities, then report the
stretches that survive the post-filter. This path does not encode or write
scene clips.

The default filter splits each look into contiguous stretches, rejoins a
same-look gap of at most ``--dino-gap`` seconds, keeps the longest
non-overlapping stretches, and drops anything shorter than ``--min-look``.
``--dino-umbrella`` restores the raw first-to-last spans instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from . import __version__
from .last_occurrence import (
    DEFAULT_MIN_LOOK,
    DEFAULT_PROTOTYPE_ALPHA,
    LookGroup,
    group_looks,
)
from .sticky import DEFAULT_SAMPLE_FPS, _decoded_duration

DEFAULT_DINO_MODEL = "dinov2_vits14"
DINO_MODELS = (
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
)
# CLS-token cosine on DINOv2. 0.60 matched the FU10 sweep better than 0.70.
# Histogram modes stay at 0.55.
DEFAULT_DINO_SIMILARITY = 0.60
DEFAULT_DINO_BATCH = 8
# Rejoin same-look stretches when the hole between them is at most this long.
DEFAULT_DINO_GAP = 60.0

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DINO_SIZE = 224


def preprocess_bgr(frame: np.ndarray) -> np.ndarray:
    """BGR uint8 frame → ImageNet-normalized CHW float32 at 224²."""
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_DINO_SIZE, _DINO_SIZE), interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'DINOv2 mode needs PyTorch. Install with: pip install -e ".[dino]"'
        ) from exc
    return torch


def resolve_device(choice: str) -> str:
    """``auto`` picks CUDA when PyTorch can see a GPU, otherwise CPU."""
    if choice == "cpu":
        return "cpu"
    torch = _require_torch()
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this PyTorch build has no GPU"
            )
        return "cuda"
    if choice != "auto":
        raise ValueError(f"unknown device: {choice}")
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dinov2(model_name: str, device: str):
    """Load a DINOv2 backbone. The first call downloads the weights."""
    if model_name not in DINO_MODELS:
        raise ValueError(f"unknown DINOv2 model: {model_name}")
    torch = _require_torch()
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            verbose=False,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load {model_name}. The first run downloads weights "
            f"from GitHub. ({exc})"
        ) from exc
    model.eval()
    model.to(device)
    return model


def embed_batch(model, batch: np.ndarray, device: str) -> np.ndarray:
    """Run one NCHW batch. Returns ``(N, D)`` float32 on CPU."""
    torch = _require_torch()
    tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
    with torch.inference_mode():
        out = model(tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state[:, 0]
    if out.ndim == 3:
        out = out[:, 0]
    return out.detach().float().cpu().numpy()


def _frame_at(time_s: float, native_fps: float, frame_count: int | None) -> int:
    frame = int(round(float(time_s) * float(native_fps)))
    if frame < 0:
        frame = 0
    if frame_count is not None and frame_count > 0:
        frame = min(frame, int(frame_count))
    return frame


def _look_record(
    group: LookGroup,
    look_id: int,
    frames: Sequence[int],
    native_fps: float,
    frame_count: int | None,
) -> dict:
    start_frame = int(frames[group.first_index])
    end_frame = _frame_at(group.end, native_fps, frame_count)
    if end_frame < start_frame:
        end_frame = start_frame
    runs = []
    for run in group.runs:
        run_start = int(frames[run.first_index])
        run_end = _frame_at(run.end, native_fps, frame_count)
        if run_end < run_start:
            run_end = run_start
        runs.append({
            "start": run.start,
            "end": run.end,
            "duration": run.duration,
            "start_frame": run_start,
            "end_frame": run_end,
        })
    return {
        "id": look_id,
        "start": group.start,
        "end": group.end,
        "duration": group.duration,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "runs": runs,
    }


@dataclass(frozen=True)
class Stretch:
    """One candidate span, possibly several contiguous runs rejoined."""

    look_id: int
    start: float
    end: float
    runs: tuple = ()
    reason: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _overlaps(left: Stretch, right: Stretch) -> bool:
    return left.start < right.end - 1e-6 and right.start < left.end - 1e-6


def filter_stretches(
    pieces: Sequence[Stretch],
    *,
    gap: float = DEFAULT_DINO_GAP,
    min_look: float = DEFAULT_MIN_LOOK,
) -> tuple[list[Stretch], list[Stretch]]:
    """Turn contiguous pieces into the default dino spans.

    Pieces of one ``look_id`` rejoin when the hole between them is at most
    ``gap`` seconds. Overlapping candidates then compete: longer duration
    wins, ties go to the earlier start. Survivors shorter than ``min_look``
    are dropped. Returned kept stretches are in start order.
    """
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if min_look <= 0:
        raise ValueError("min_look must be positive")

    by_look: dict[int, list[Stretch]] = {}
    for piece in pieces:
        by_look.setdefault(piece.look_id, []).append(piece)

    merged: list[Stretch] = []
    for look_id, runs in by_look.items():
        ordered = sorted(runs, key=lambda run: (run.start, run.end))
        current = ordered[0]
        for nxt in ordered[1:]:
            hole = nxt.start - current.end
            if hole <= gap + 1e-6:
                current = Stretch(
                    look_id,
                    current.start,
                    max(current.end, nxt.end),
                    current.runs + nxt.runs,
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    ranked = sorted(merged, key=lambda item: (-item.duration, item.start, item.look_id))
    accepted: list[Stretch] = []
    skipped: list[Stretch] = []
    for candidate in ranked:
        if any(_overlaps(candidate, kept) for kept in accepted):
            skipped.append(Stretch(
                candidate.look_id,
                candidate.start,
                candidate.end,
                candidate.runs,
                "overlap",
            ))
        else:
            accepted.append(candidate)

    kept: list[Stretch] = []
    for candidate in accepted:
        if candidate.duration + 1e-6 < min_look:
            skipped.append(Stretch(
                candidate.look_id,
                candidate.start,
                candidate.end,
                candidate.runs,
                "short",
            ))
        else:
            kept.append(candidate)
    kept.sort(key=lambda item: (item.start, item.look_id))
    return kept, skipped


def _nesting(looks: list[dict]) -> None:
    for look in looks:
        look["nested_in"] = []
        look["contains"] = []
    for outer in looks:
        for inner in looks:
            if outer["id"] == inner["id"]:
                continue
            inside = (
                inner["start"] >= outer["start"] - 1e-6
                and inner["end"] <= outer["end"] + 1e-6
                and (
                    inner["start"] > outer["start"] + 1e-6
                    or inner["end"] < outer["end"] - 1e-6
                )
            )
            if inside:
                inner["nested_in"].append(outer["id"])
                outer["contains"].append(inner["id"])


def _runs_payload(stretch: Stretch, frames: Sequence[int], native_fps: float, frame_count: int | None) -> list[dict]:
    payload = []
    for run in stretch.runs:
        run_start = int(frames[run.first_index])
        run_end = _frame_at(run.end, native_fps, frame_count)
        if run_end < run_start:
            run_end = run_start
        payload.append({
            "start": run.start,
            "end": run.end,
            "duration": run.duration,
            "start_frame": run_start,
            "end_frame": run_end,
        })
    return payload


def _stretch_record(
    stretch: Stretch,
    report_id: int,
    frames: Sequence[int],
    native_fps: float,
    frame_count: int | None,
) -> dict:
    if stretch.runs:
        start_frame = int(frames[stretch.runs[0].first_index])
    else:
        start_frame = _frame_at(stretch.start, native_fps, frame_count)
    end_frame = _frame_at(stretch.end, native_fps, frame_count)
    if end_frame < start_frame:
        end_frame = start_frame
    record = {
        "id": report_id,
        "look_id": int(stretch.look_id) + 1,
        "start": stretch.start,
        "end": stretch.end,
        "duration": stretch.duration,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "runs": _runs_payload(stretch, frames, native_fps, frame_count),
        "nested_in": [],
        "contains": [],
    }
    if stretch.reason:
        record["reason"] = stretch.reason
    return record


def build_dino_report(
    samples: Sequence[tuple[float, np.ndarray]],
    *,
    frames: Sequence[int] | None = None,
    native_fps: float = 1.0,
    frame_count: int | None = None,
    duration: float | None = None,
    min_look: float = DEFAULT_MIN_LOOK,
    similarity: float = DEFAULT_DINO_SIMILARITY,
    sample_interval: float = 1.0,
    prototype_alpha: float = DEFAULT_PROTOTYPE_ALPHA,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    model: str = DEFAULT_DINO_MODEL,
    device: str = "cpu",
    source: str = "",
    gap: float = DEFAULT_DINO_GAP,
    umbrella: bool = False,
) -> dict:
    """Group embedding samples into a dry-run look report.

    ``samples`` are ``(time_seconds, embedding)``. ``frames`` aligns with
    them; when omitted, frame numbers are ``round(time * native_fps)``.

    The default path reports filtered stretches. ``umbrella=True`` reports
    each lasting look's raw first-to-last span instead.
    """
    if native_fps <= 0:
        raise ValueError("native_fps must be positive")
    ordered = list(samples)
    if frames is None:
        frame_list = [_frame_at(t, native_fps, frame_count) for t, _sig in ordered]
    else:
        frame_list = [int(f) for f in frames]
        if len(frame_list) != len(ordered):
            raise ValueError("frames and samples must have the same length")

    grouped = group_looks(
        ordered,
        similarity=similarity,
        prototype_alpha=prototype_alpha,
        min_look=min_look,
        sample_interval=sample_interval,
        duration=duration,
    )
    if umbrella:
        lasting = [group for group in grouped if group.lasting]
        skipped_groups = [group for group in grouped if not group.lasting]
        looks = [
            _look_record(group, i, frame_list, native_fps, frame_count)
            for i, group in enumerate(lasting, start=1)
        ]
        for look, group in zip(looks, lasting):
            look["look_id"] = group.creation_index + 1
        _nesting(looks)
        skipped_looks = [
            _look_record(group, i, frame_list, native_fps, frame_count)
            for i, group in enumerate(skipped_groups, start=1)
        ]
        for skipped, group in zip(skipped_looks, skipped_groups):
            skipped["look_id"] = group.creation_index + 1
            skipped["reason"] = "short"
    else:
        pieces = [
            Stretch(group.creation_index, run.start, run.end, (run,))
            for group in grouped
            for run in group.runs
        ]
        kept, dropped = filter_stretches(pieces, gap=gap, min_look=min_look)
        looks = [
            _stretch_record(stretch, i, frame_list, native_fps, frame_count)
            for i, stretch in enumerate(kept, start=1)
        ]
        skipped_looks = [
            _stretch_record(stretch, i, frame_list, native_fps, frame_count)
            for i, stretch in enumerate(dropped, start=1)
        ]
    return {
        "version": __version__,
        "mode": "dino",
        "dry_run": True,
        "filter": "umbrella" if umbrella else "stretch",
        "dino_gap": None if umbrella else float(gap),
        "source": source,
        "model": model,
        "device": device,
        "fps": float(native_fps),
        "frame_count": int(frame_count) if frame_count else None,
        "duration": None if duration is None else float(duration),
        "sample_fps": float(sample_fps),
        "sample_count": len(ordered),
        "min_look": float(min_look),
        "similarity": float(similarity),
        "looks": looks,
        "skipped_looks": skipped_looks,
    }


def format_dino_summary(report: dict) -> str:
    """Human-readable list of proposed look spans."""
    looks = report.get("looks") or []
    kind = report.get("filter") or "stretch"
    lines = [f"lasting looks: {len(looks)}  ({kind})"]
    for look in looks:
        extra = []
        if look.get("look_id") not in (None, look["id"]):
            extra.append(f"identity {look['look_id']}")
        if look.get("nested_in"):
            extra.append("nested in " + ",".join(str(i) for i in look["nested_in"]))
        if look.get("contains"):
            extra.append("contains " + ",".join(str(i) for i in look["contains"]))
        suffix = ("  " + "; ".join(extra)) if extra else ""
        lines.append(
            f"  look {look['id']}  frames {look['start_frame']}-{look['end_frame']}"
            f"  {look['start']:.1f}-{look['end']:.1f}s"
            f"  ({look['duration']:.1f}s)"
            f"  runs {len(look.get('runs') or [])}"
            f"{suffix}"
        )
    skipped = report.get("skipped_looks") or []
    if skipped:
        lines.append(f"short looks left inside: {len(skipped)}")
    return "\n".join(lines)


def _sample_and_embed(
    path: str,
    model,
    device: str,
    *,
    sample_fps: float,
    batch_size: int,
    on_progress: Callable[[float, float], None] | None,
) -> tuple[list[tuple[float, np.ndarray]], list[int], float, float, int]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        native = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native <= 1e-3:
            native = 30.0
        reported_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        reported = reported_frames / native if reported_frames > 0 else 0.0
        frame_count = int(reported_frames) if reported_frames > 0 else 0
        step = max(1, int(round(native / float(sample_fps))))
        times: list[float] = []
        frames: list[int] = []
        pending: list[np.ndarray] = []
        vectors: list[np.ndarray] = []
        index = 0
        last_report = 0.0

        def flush() -> None:
            if not pending:
                return
            batch = np.stack(pending)
            pending.clear()
            for vec in embed_batch(model, batch, device):
                vectors.append(np.asarray(vec, dtype=np.float64).ravel())

        while True:
            if index % step == 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t = index / native
                times.append(t)
                frames.append(index)
                pending.append(preprocess_bgr(frame))
                if len(pending) >= batch_size:
                    flush()
                if on_progress is not None and t - last_report >= 30.0:
                    on_progress(t, reported or t)
                    last_report = t
            else:
                if not cap.grab():
                    break
            index += 1
        flush()
        if not times:
            raise RuntimeError(f"no frames decoded: {path}")
        if len(vectors) != len(times):
            raise RuntimeError(
                f"embedded {len(vectors)} frames but sampled {len(times)}"
            )
        duration = _decoded_duration(index / native, reported)
        samples = list(zip(times, vectors))
        return samples, frames, native, duration, frame_count or index
    finally:
        cap.release()


def detect_dino_report(
    path: str,
    *,
    min_look: float = DEFAULT_MIN_LOOK,
    similarity: float = DEFAULT_DINO_SIMILARITY,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    prototype_alpha: float = DEFAULT_PROTOTYPE_ALPHA,
    model_name: str = DEFAULT_DINO_MODEL,
    device: str = "auto",
    batch_size: int = DEFAULT_DINO_BATCH,
    gap: float = DEFAULT_DINO_GAP,
    umbrella: bool = False,
    on_progress: Callable[[float, float], None] | None = None,
) -> dict:
    """Sample ``path``, embed with DINOv2, and return a dry-run report."""
    chosen = resolve_device(device)
    model = load_dinov2(model_name, chosen)
    samples, frames, native_fps, duration, frame_count = _sample_and_embed(
        path,
        model,
        chosen,
        sample_fps=sample_fps,
        batch_size=batch_size,
        on_progress=on_progress,
    )
    return build_dino_report(
        samples,
        frames=frames,
        native_fps=native_fps,
        frame_count=frame_count,
        duration=duration,
        min_look=min_look,
        similarity=similarity,
        sample_interval=1.0 / float(sample_fps),
        prototype_alpha=prototype_alpha,
        sample_fps=sample_fps,
        model=model_name,
        device=chosen,
        source=path,
        gap=gap,
        umbrella=umbrella,
    )
