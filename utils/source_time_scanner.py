"""Source-level mosaic time scanning for OneClick source-scan mode."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TimeInterval:
    start_s: float
    end_s: float
    conf_max: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_intervals(intervals: list[TimeInterval], gap_s: float = 0.05) -> list[TimeInterval]:
    if not intervals:
        return []
    merged: list[TimeInterval] = []
    for interval in sorted(intervals, key=lambda item: (item.start_s, item.end_s)):
        if not merged or interval.start_s > merged[-1].end_s + gap_s:
            merged.append(TimeInterval(interval.start_s, interval.end_s, interval.conf_max))
            continue
        prev = merged[-1]
        prev.end_s = max(prev.end_s, interval.end_s)
        prev.conf_max = max(prev.conf_max, interval.conf_max)
    return merged


def _merge_by_gap(intervals: list[TimeInterval], gap_s: float,
                  max_segment_s: float = 0.0) -> list[TimeInterval]:
    if not intervals:
        return []
    out: list[TimeInterval] = []
    for interval in sorted(intervals, key=lambda item: (item.start_s, item.end_s)):
        if not out or interval.start_s > out[-1].end_s + gap_s:
            out.append(TimeInterval(interval.start_s, interval.end_s, interval.conf_max))
            continue
        prev = out[-1]
        new_end = max(prev.end_s, interval.end_s)
        if max_segment_s > 0.0 and new_end - prev.start_s > max_segment_s:
            out.append(TimeInterval(interval.start_s, interval.end_s, interval.conf_max))
            continue
        prev.end_s = new_end
        prev.conf_max = max(prev.conf_max, interval.conf_max)
    return out


def _coarse_merge(intervals: list[TimeInterval], duration_s: float = 0.0) -> list[TimeInterval]:
    """Source-scan level coarse merge, independent from inner pre-extract tuning."""
    if not intervals:
        return []
    from utils import app_config

    def cfg_float(key: str, default: float) -> float:
        value = app_config.get(key, default)
        return float(default if value is None else value)

    merge_gap = max(0.0, cfg_float("source_scan_merge_gap_s", 30.0))
    min_segment = max(0.0, cfg_float("source_scan_min_segment_s", 30.0))
    pad = max(0.0, cfg_float("source_scan_head_tail_pad_s", 5.0))
    max_segment = max(0.0, cfg_float("source_scan_max_segment_s", 0.0))
    duration = max(0.0, float(duration_s or 0.0))

    padded: list[TimeInterval] = []
    for interval in intervals:
        start = max(0.0, float(interval.start_s) - pad)
        end = float(interval.end_s) + pad
        if duration > 0.0:
            end = min(duration, end)
        if end > start:
            padded.append(TimeInterval(start, end, float(interval.conf_max)))

    merged = _merge_by_gap(padded, merge_gap, max_segment)
    if min_segment <= 0.0:
        return merged

    extended: list[TimeInterval] = []
    for interval in merged:
        dur = interval.duration_s
        if dur >= min_segment:
            extended.append(interval)
            continue
        grow = (min_segment - dur) * 0.5
        start = max(0.0, interval.start_s - grow)
        end = interval.end_s + grow
        if duration > 0.0 and end > duration:
            shift = end - duration
            end = duration
            start = max(0.0, start - shift)
        extended.append(TimeInterval(start, end, interval.conf_max))
    return _merge_by_gap(extended, merge_gap, max_segment)


def scan_source_time_segments(source_sbs: str | Path, log_callback=None,
                              cancel_token=None) -> list[TimeInterval]:
    """Scan source SBS and return time intervals only.

    mosaic_prescan may return multiple spatial regions for the same time span.
    Source scan only cares about time, so overlapping time ranges are merged.
    """
    from utils import mosaic_prescan
    from utils import app_config
    from gpu_engine import probe

    strategy = str(app_config.get("source_scan_strategy", "keyframes") or "keyframes").strip().lower()
    if not strategy:
        strategy = "keyframes"
    if log_callback:
        log_callback(f"[source-scan] scan strategy: {strategy}")
    segments = mosaic_prescan.scan_segments(
        source_sbs,
        log_callback=log_callback,
        cancel_token=cancel_token,
        scan_strategy=strategy,
    )
    intervals = [
        TimeInterval(start_s=float(seg.start_s), end_s=float(seg.end_s), conf_max=float(seg.conf_max))
        for seg in segments
        if float(seg.end_s) > float(seg.start_s)
    ]
    intervals = _merge_intervals(intervals)
    try:
        meta = probe.probe_video(source_sbs)
        duration = float(meta.duration or 0.0)
    except Exception:
        duration = 0.0
    merged = _coarse_merge(intervals, duration)
    if log_callback and len(merged) != len(intervals):
        log_callback(f"[source-scan] coarse merged {len(intervals)} intervals -> {len(merged)} intervals")
    return merged


def save_source_intervals_json(intervals: list[TimeInterval], path: str | Path,
                               source: str | Path | None = None) -> None:
    data = {
        "source": str(source) if source else "",
        "intervals": [interval.to_dict() for interval in intervals],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
