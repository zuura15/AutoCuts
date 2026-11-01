"""Global last-occurrence scenes.

Sample the whole video, match samples into visual looks, and export one
span per lasting look: from the first time that look appears to the last
time it appears anywhere later. The file keeps everything in between,
including other scenes and transition junk. A short look with no lasting
run does not get its own file.

Overlapping spans are expected. A nested scene that itself lasts is its
own export, sitting inside the parent's first-to-last range.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np

from .logic import Interval
from .sticky import (
    DEFAULT_PROTOTYPE_ALPHA,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SIMILARITY,
    cosine_similarity,
    sample_video,
)

# A look needs a contiguous run at least this long before it is its own
# export. ~10s of a distinct nested scene qualifies; a few-second fountain
# or camera flourish does not.
DEFAULT_MIN_LOOK = 8.0


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.copy()
    return arr / norm


def _ema(prototype: np.ndarray, sample: np.ndarray, alpha: float) -> np.ndarray:
    mixed = (1.0 - alpha) * _normalize(prototype) + alpha * _normalize(sample)
    norm = float(np.linalg.norm(mixed))
    if norm == 0.0:
        return mixed
    return mixed / norm


def last_occurrence_scenes(
    samples: Sequence[tuple[float, np.ndarray]],
    *,
    similarity: float = DEFAULT_SIMILARITY,
    prototype_alpha: float = DEFAULT_PROTOTYPE_ALPHA,
    min_look: float = DEFAULT_MIN_LOOK,
    sample_interval: float = 1.0,
    duration: float | None = None,
) -> List[Interval]:
    """One interval per lasting look, each running first occurrence → last.

    Samples are ``(time_seconds, signature)`` in order. A sample joins the
    existing look whose prototype is most similar when that similarity is
    at least ``similarity``; otherwise it opens a new look. The prototype
    is a slow average of the samples that match it, same idea as sticky
    mode, so a gradual zoom stays one look.

    A look is exported only when some contiguous run of its samples lasts
    at least ``min_look`` seconds. The exported range still runs from that
    look's first sample to its last sample anywhere in the video, so a
    later return pulls the middle back in. Shorter looks are omitted;
    their time stays inside any longer span that already covers it.
    """
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be between 0 and 1")
    if not 0.0 < prototype_alpha <= 1.0:
        raise ValueError("prototype_alpha must be in (0, 1]")
    if min_look <= 0:
        raise ValueError("min_look must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    ordered = list(samples)
    if not ordered:
        return []

    previous_t: float | None = None
    for time_s, _sig in ordered:
        current = float(time_s)
        if previous_t is not None and current < previous_t:
            raise ValueError("samples must be in non-decreasing time order")
        previous_t = current

    looks: list[dict] = []
    assignment: list[tuple[float, int]] = []
    for time_s, sig in ordered:
        t = float(time_s)
        sample = np.asarray(sig, dtype=np.float64).ravel().copy()
        best_i: int | None = None
        best_sim = -2.0
        for i, look in enumerate(looks):
            sim = cosine_similarity(look["prototype"], sample)
            if sim >= similarity and sim > best_sim:
                best_sim = sim
                best_i = i
        if best_i is None:
            looks.append({"prototype": _normalize(sample), "times": [t]})
            best_i = len(looks) - 1
        else:
            looks[best_i]["times"].append(t)
            looks[best_i]["prototype"] = _ema(
                looks[best_i]["prototype"], sample, prototype_alpha,
            )
        assignment.append((t, best_i))

    max_run = [0.0] * len(looks)
    index = 0
    total = len(assignment)
    while index < total:
        look_i = assignment[index][1]
        end_index = index + 1
        while end_index < total and assignment[end_index][1] == look_i:
            end_index += 1
        start_t = assignment[index][0]
        end_t = assignment[end_index - 1][0]
        length = (end_t - start_t) + sample_interval
        if length > max_run[look_i]:
            max_run[look_i] = length
        index = end_index

    spans: list[tuple[float, float, int]] = []
    for look_i, look in enumerate(looks):
        if max_run[look_i] + 1e-6 < min_look:
            continue
        first = float(look["times"][0])
        last = float(look["times"][-1])
        end = last + sample_interval
        if duration is not None:
            end = min(end, float(duration))
        if end > first:
            spans.append((first, end, look_i))

    spans.sort(key=lambda item: (item[0], item[2]))
    return [Interval(start, end) for start, end, _look_i in spans]


def detect_last_occurrence_intervals(
    path: str,
    *,
    min_look: float = DEFAULT_MIN_LOOK,
    similarity: float = DEFAULT_SIMILARITY,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    prototype_alpha: float = DEFAULT_PROTOTYPE_ALPHA,
    on_progress: Callable[[float, float], None] | None = None,
) -> List[Interval]:
    """Sample ``path`` and return one first-to-last interval per lasting look."""
    samples, duration = sample_video(path, sample_fps=sample_fps, on_progress=on_progress)
    return last_occurrence_scenes(
        samples,
        similarity=similarity,
        prototype_alpha=prototype_alpha,
        min_look=min_look,
        sample_interval=1.0 / float(sample_fps),
        duration=duration,
    )
