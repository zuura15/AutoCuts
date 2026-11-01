"""Sticky scene detection.

Samples a coarse visual signature and keeps one scene until the picture
has looked different for a sustained hold-out window. Brief zooms and
lighting blips that revert stay inside the current scene. Slow drift is
absorbed by updating the prototype only while samples still match it.

This prefers under-splitting: a little junk left inside a scene is fine.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import cv2
import numpy as np

from .logic import Interval

# Hue × saturation. Value is omitted so an exposure change is not a new
# scene. Bins are coarse and assigned bilinearly (hue is circular) so a
# small white-balance nudge does not fall off a bin edge into "unlike".
_H_BINS = 8
_S_BINS = 4
_PREVIEW_WIDTH = 160

DEFAULT_HOLD_OUT = 20.0
HOLD_OUT_MIN = 15.0
HOLD_OUT_MAX = 30.0
# Cosine similarity of the soft HS histogram. ~0.55 keeps a punch-in of
# the same palette and a lighting shift, and still treats a real palette
# change as unlike. Lower values under-split more.
DEFAULT_SIMILARITY = 0.55
DEFAULT_SAMPLE_FPS = 1.0
# Per matching sample, at ~1 fps. Slow pans walk the prototype; unlike
# samples do not, or a hard cut would be absorbed before the hold-out.
DEFAULT_PROTOTYPE_ALPHA = 0.05


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]``. Two empty signatures count as a match."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == 0.0 and nb == 0.0 else 0.0
    return float(np.dot(va, vb) / (na * nb))


def hsv_histogram(frame: np.ndarray) -> np.ndarray:
    """Soft HS histogram of a BGR frame. Brightness is ignored.

    Each pixel is split between the nearest hue and saturation bins, and
    hue wraps (red meets red). The result is a palette signature: a zoom
    or a brightness blip stays close to the wide shot, a new location
    usually does not.
    """
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    small = frame
    height, width = frame.shape[:2]
    if width > _PREVIEW_WIDTH:
        scale = _PREVIEW_WIDTH / float(width)
        small = cv2.resize(
            frame,
            (_PREVIEW_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float64)
    sat = hsv[..., 1].astype(np.float64)
    hue_pos = hue * (_H_BINS / 180.0)
    sat_pos = np.clip(sat * (_S_BINS / 256.0), 0.0, _S_BINS - 1e-6)
    hue_i = np.floor(hue_pos).astype(np.int32) % _H_BINS
    hue_f = hue_pos - np.floor(hue_pos)
    hue_j = (hue_i + 1) % _H_BINS
    sat_i = np.floor(sat_pos).astype(np.int32)
    sat_f = sat_pos - np.floor(sat_pos)
    sat_j = np.minimum(sat_i + 1, _S_BINS - 1)
    sat_f = np.where(sat_j == sat_i, 0.0, sat_f)
    hist = np.zeros((_H_BINS, _S_BINS), dtype=np.float64)
    np.add.at(hist, (hue_i, sat_i), (1.0 - hue_f) * (1.0 - sat_f))
    np.add.at(hist, (hue_j, sat_i), hue_f * (1.0 - sat_f))
    np.add.at(hist, (hue_i, sat_j), (1.0 - hue_f) * sat_f)
    np.add.at(hist, (hue_j, sat_j), hue_f * sat_f)
    return hist.ravel()


def _as_vec(sample: np.ndarray) -> np.ndarray:
    return np.asarray(sample, dtype=np.float64).ravel().copy()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _ema(prototype: np.ndarray, sample: np.ndarray, alpha: float) -> np.ndarray:
    # Normalize both sides. Raw histograms have magnitude ~= pixel count,
    # which would otherwise drown a unit-length prototype in one update.
    mixed = (1.0 - alpha) * _normalize(prototype) + alpha * _normalize(sample)
    return _normalize(mixed)


def sticky_scenes(
    samples: Sequence[tuple[float, np.ndarray]],
    *,
    hold_out: float = DEFAULT_HOLD_OUT,
    similarity: float = DEFAULT_SIMILARITY,
    prototype_alpha: float = DEFAULT_PROTOTYPE_ALPHA,
    duration: float | None = None,
) -> List[Interval]:
    """Split ordered ``(time_seconds, signature)`` samples into sticky scenes.

    The scene closes only when every sample from some time ``t`` through a
    sample at least ``hold_out`` seconds later is below ``similarity`` with
    the frozen prototype. The cut is placed at ``t`` (the first unlike
    sample), so the hold-out stretch belongs to the next scene. If the new
    look reverts before the window elapses, the blip stays in the current
    scene and the prototype resumes updating.

    Sample times are seconds from the start of the video. The returned
    intervals cover ``[0, duration]`` with no gaps. ``duration`` defaults
    to the last sample time. The hold-out is measured on observed samples
    only; time past the last sample is not treated as evidence, which
    under-splits if sampling stops early. ``hold_out`` may be any positive
    number; the CLI restricts the flag to 15–30s.
    """
    if hold_out <= 0:
        raise ValueError("hold_out must be positive")
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be between 0 and 1")
    if not 0.0 < prototype_alpha <= 1.0:
        raise ValueError("prototype_alpha must be in (0, 1]")

    ordered = list(samples)
    if not ordered:
        return []

    previous_t: float | None = None
    for time_s, _sig in ordered:
        current = float(time_s)
        if previous_t is not None and current < previous_t:
            raise ValueError("samples must be in non-decreasing time order")
        previous_t = current

    last_t = float(ordered[-1][0])
    end = last_t if duration is None else max(float(duration), last_t)
    if end <= 0.0:
        return []

    prototype = _normalize(_as_vec(ordered[0][1]))
    scene_start = 0.0
    diverge_at: float | None = None
    candidate: np.ndarray | None = None
    scenes: List[Interval] = []

    def close_at(cut: float, new_prototype: np.ndarray) -> None:
        nonlocal scene_start, prototype, diverge_at, candidate
        cut = float(cut)
        if cut > scene_start:
            scenes.append(Interval(scene_start, cut))
            scene_start = cut
        prototype = _normalize(_as_vec(new_prototype))
        diverge_at = None
        candidate = None

    for time_s, sig in ordered[1:]:
        t = float(time_s)
        sample = _as_vec(sig)
        sim = cosine_similarity(prototype, sample)
        if sim >= similarity:
            # Back inside the scene. A blip that did not last the hold-out
            # is discarded; matching samples keep the prototype drifting.
            diverge_at = None
            candidate = None
            prototype = _ema(prototype, sample, prototype_alpha)
            continue
        if diverge_at is None:
            diverge_at = t
            candidate = _normalize(sample)
            continue
        assert candidate is not None
        candidate = _ema(candidate, sample, prototype_alpha)
        if (t - diverge_at) + 1e-6 >= hold_out:
            close_at(diverge_at, candidate)

    if end > scene_start:
        scenes.append(Interval(scene_start, end))
    return scenes


def _decoded_duration(decoded: float, reported: float) -> float:
    """Prefer the container duration when it agrees with what we decoded."""
    if reported <= 0:
        return decoded
    slack = max(1.5, 0.02 * max(decoded, 0.0))
    if reported > decoded + slack:
        # Some files report a nonsense frame count. Trust the decoder.
        return decoded
    return max(decoded, reported)


def sample_video(
    path: str,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    on_progress: Callable[[float, float], None] | None = None,
) -> tuple[list[tuple[float, np.ndarray]], float]:
    """Read ``path`` and return ``(samples, duration_seconds)``.

    Samples are taken on a coarse grid (default 1 fps). The file is decoded
    in order; skipped frames are grabbed and not histogrammed.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        native = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native <= 1e-3:
            native = 30.0
        reported_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        reported = reported_frames / native if reported_frames > 0 else 0.0
        step = max(1, int(round(native / float(sample_fps))))
        samples: list[tuple[float, np.ndarray]] = []
        index = 0
        last_report = 0.0
        while True:
            take = index % step == 0
            if take:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t = index / native
                samples.append((t, hsv_histogram(frame)))
                if on_progress is not None and t - last_report >= 30.0:
                    on_progress(t, reported or t)
                    last_report = t
            else:
                if not cap.grab():
                    break
            index += 1
        if not samples:
            raise RuntimeError(f"no frames decoded: {path}")
        duration = _decoded_duration(index / native, reported)
        return samples, duration
    finally:
        cap.release()


def detect_sticky_intervals(
    path: str,
    *,
    hold_out: float = DEFAULT_HOLD_OUT,
    similarity: float = DEFAULT_SIMILARITY,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    prototype_alpha: float = DEFAULT_PROTOTYPE_ALPHA,
    on_progress: Callable[[float, float], None] | None = None,
) -> List[Interval]:
    """Sample ``path`` and return one interval per sticky scene."""
    samples, duration = sample_video(path, sample_fps=sample_fps, on_progress=on_progress)
    return sticky_scenes(
        samples,
        hold_out=hold_out,
        similarity=similarity,
        prototype_alpha=prototype_alpha,
        duration=duration,
    )
