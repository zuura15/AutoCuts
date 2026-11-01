"""DINOv2 dry-run look report.

Embed a coarse sample of frames and group them with the same first-to-last
rules as last-occurrence. The result is a JSON report of proposed spans.
This path does not encode or write scene clips.

Change-points come from cosine distance to each look's running prototype,
not from a separate segmentation model. A lasting nested look still gets
its own overlapping span; a short run stays inside the parent.
"""
from __future__ import annotations

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
# Starting point for CLS-token cosine on DINOv2. Histogram mode stays at
# 0.55; this space is tighter, so the dry-run default is higher. Tune it
# against a ground-truth timeline.
DEFAULT_DINO_SIMILARITY = 0.70
DEFAULT_DINO_BATCH = 8

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
) -> dict:
    """Group embedding samples into a dry-run look report.

    ``samples`` are ``(time_seconds, embedding)``. ``frames`` aligns with
    them; when omitted, frame numbers are ``round(time * native_fps)``.
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
    lasting = [group for group in grouped if group.lasting]
    skipped = [group for group in grouped if not group.lasting]
    looks = [
        _look_record(group, i, frame_list, native_fps, frame_count)
        for i, group in enumerate(lasting, start=1)
    ]
    _nesting(looks)
    skipped_looks = [
        _look_record(group, i, frame_list, native_fps, frame_count)
        for i, group in enumerate(skipped, start=1)
    ]
    return {
        "version": __version__,
        "mode": "dino",
        "dry_run": True,
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
    lines = [f"lasting looks: {len(looks)}"]
    for look in looks:
        extra = []
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
    )
