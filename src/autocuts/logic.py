"""Scene detect + drop short bursts + optional stitch of similar flanks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from scenedetect import FrameTimecode, SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector


@dataclass(frozen=True)
class Interval:
    start: float  # seconds
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_intervals(
    path: str,
    *,
    detector: str = "adaptive",
    threshold: float | None = None,
    min_scene_len: float | str = "0.6s",
) -> List[Interval]:
    video = open_video(path)
    sm = SceneManager()
    msl = min_scene_len if isinstance(min_scene_len, str) else f"{min_scene_len}s"
    if detector == "content":
        kwargs = {"min_scene_len": msl}
        if threshold is not None:
            kwargs["threshold"] = threshold
        sm.add_detector(ContentDetector(**kwargs))
    else:
        kwargs = {"min_scene_len": msl}
        if threshold is not None:
            kwargs["adaptive_threshold"] = threshold
        sm.add_detector(AdaptiveDetector(**kwargs))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    out: List[Interval] = []
    for a, b in scenes:
        out.append(Interval(a.get_seconds(), b.get_seconds()))
    return out


def drop_short(intervals: Sequence[Interval], min_keep: float) -> Tuple[List[Interval], List[Interval]]:
    """Split into keep (>= min_keep) and dropped (< min_keep)."""
    keep, dropped = [], []
    for iv in intervals:
        (keep if iv.duration >= min_keep else dropped).append(iv)
    return keep, list(dropped)


def stitch_across_drops(
    all_intervals: Sequence[Interval],
    *,
    min_keep: float,
) -> List[List[Interval]]:
    """Group kept intervals so short bursts between them are excised.

    Consecutive long scenes with only short scenes between them become one
    export group (multiple ranges concatenated, gaps omitted). A new group
    starts whenever a long scene follows another long scene with no short
    burst between (adjacent keeps) — those stay separate files.

    Actually: user wants short pan excised from the *prior* scene flow.
    If we have L D L, merge into one group [L,L] (concat, skip D).
    If we have L L (adjacent keeps), two groups.
    If we have L D L D L, one group of three L's.
    """
    if not all_intervals:
        return []

    groups: List[List[Interval]] = []
    current: List[Interval] = []
    pending_drop = False

    for iv in all_intervals:
        is_keep = iv.duration >= min_keep
        if is_keep:
            if not current:
                current = [iv]
                pending_drop = False
            elif pending_drop:
                # short burst between two longs → same group (excise middle)
                current.append(iv)
                pending_drop = False
            else:
                # two keeps back-to-back → new file
                groups.append(current)
                current = [iv]
                pending_drop = False
        else:
            if current:
                pending_drop = True
            # leading shorts before first keep: ignore
    if current:
        groups.append(current)
    return groups


def flat_export_plan(groups: Sequence[Sequence[Interval]]) -> List[List[Interval]]:
    return [list(g) for g in groups]
