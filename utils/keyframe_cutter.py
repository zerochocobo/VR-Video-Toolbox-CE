"""Keyframe-aware pre-extract segment cutting."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.mosaic_prescan import MosaicSegment


@dataclass
class TimelineEntry:
    start_s: float
    end_s: float
    path: Path
    kind: str  # "mosaic" or "gap"
    conf_max: float = 0.0
    inpoint_s: float | None = None
    outpoint_s: float | None = None


def _hidden_kwargs() -> dict:
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        return {"startupinfo": startupinfo}
    return {}


def _run(cmd: list[str], log_callback=None, process_callback=None) -> None:
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_kwargs(),
    )
    if process_callback:
        process_callback(proc)
    try:
        if proc.stdout:
            for line in proc.stdout:
                if log_callback:
                    log_callback(line.rstrip())
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}")


def get_video_size(path: str | Path) -> tuple[int, int] | None:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, errors="replace", **_hidden_kwargs()).strip()
        width, height = [int(x) for x in out.split(",")[:2]]
        return width, height
    except Exception:
        return None


def segment_file_matches_rect(path: str | Path, segment: MosaicSegment) -> bool:
    size = get_video_size(path)
    return size == (int(segment.w), int(segment.h))


def list_keyframes(path: str | Path) -> list[float]:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_frames",
        "-show_entries", "frame=pts_time,best_effort_timestamp_time",
        "-of", "json",
        str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **_hidden_kwargs())
        data = json.loads(raw)
    except Exception:
        return []
    out: list[float] = []
    for frame in data.get("frames", []):
        value = frame.get("pts_time") or frame.get("best_effort_timestamp_time")
        try:
            ts = float(value)
        except Exception:
            continue
        if ts >= 0:
            out.append(ts)
    return sorted(set(out))


def _floor_kf(value: float, keyframes: list[float]) -> float:
    best = 0.0
    for kf in keyframes:
        if kf <= value + 1e-6:
            best = kf
        else:
            break
    return best


def _ceil_kf(value: float, keyframes: list[float], duration: float | None) -> float:
    for kf in keyframes:
        if kf >= value - 1e-6:
            return kf
    return max(value, duration or value)


def _interval_value(interval, name: str, default: float = 0.0) -> float:
    if isinstance(interval, dict):
        return float(interval.get(name, default))
    return float(getattr(interval, name, default))


def cut_source_by_intervals(src: str | Path, intervals, out_dir: str | Path,
                            keyframes: list[float] | None = None,
                            log_callback=None, process_callback=None,
                            materialize_gaps: bool = False,
                            materialize_mosaic: bool = True) -> list[TimelineEntry]:
    from gpu_engine import probe

    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe.probe_video(src)
    duration = float(meta.duration or 0.0)
    if keyframes is None:
        keyframes = list_keyframes(src)
    if log_callback:
        log_callback(
            f"[source-scan] Stage 2 keyframe copy setup: source={src}, "
            f"out_dir={out_dir}, duration={duration:.3f}s, keyframes={len(keyframes or [])}"
        )

    mosaic_specs: list[tuple[float, float, float]] = []
    for interval in intervals:
        start = max(0.0, _interval_value(interval, "start_s"))
        end = _interval_value(interval, "end_s")
        conf = _interval_value(interval, "conf_max")
        if duration > 0:
            end = min(duration, end)
        if end - start > 0.05:
            mosaic_specs.append((start, end, conf))

    mosaic_specs.sort(key=lambda item: (item[0], item[1]))
    if log_callback:
        log_callback(f"[source-scan] Stage 2 requested mosaic intervals: {len(mosaic_specs)}")

    aligned_specs: list[tuple[float, float, float, float, float]] = []
    for start, end, conf in mosaic_specs:
        cut_start = _floor_kf(start, keyframes) if keyframes else start
        cut_end = _ceil_kf(end, keyframes, duration) if keyframes else end
        if duration > 0:
            cut_start = max(0.0, min(cut_start, duration))
            cut_end = max(cut_start, min(cut_end, duration))
        if cut_end - cut_start > 0.05:
            aligned_specs.append((cut_start, cut_end, conf, start, end))
    if log_callback and not keyframes:
        log_callback("[source-scan] no keyframe list available; copy cut will use raw interval boundaries")

    merged: list[tuple[float, float, float]] = []
    for start, end, conf, _raw_start, _raw_end in aligned_specs:
        if merged and start <= merged[-1][1] + 1e-3:
            prev_start, prev_end, prev_conf = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), max(prev_conf, conf))
        else:
            merged.append((start, end, conf))
    if log_callback and len(merged) != len(aligned_specs):
        log_callback(f"[source-scan] Stage 2 merged overlapping intervals: {len(aligned_specs)} -> {len(merged)}")

    mosaic_entries: list[TimelineEntry] = []
    for idx, (start, end, conf) in enumerate(merged):
        if materialize_mosaic:
            path = out_dir / f"mosaic_seg{idx:03d}{src.suffix}"
            if log_callback:
                log_callback(f"[source-scan] keyframe copy mosaic segment {idx}: {start:.3f}-{end:.3f}s -> {path.name}")
            _cut_copy(src, path, start, end, log_callback=log_callback, process_callback=process_callback)
            mosaic_entries.append(TimelineEntry(start_s=start, end_s=end, path=path, kind="mosaic", conf_max=conf))
        else:
            if log_callback:
                log_callback(f"[source-scan] virtual mosaic segment {idx}: {start:.3f}-{end:.3f}s -> source interval")
            mosaic_entries.append(
                TimelineEntry(
                    start_s=start,
                    end_s=end,
                    path=src,
                    kind="mosaic",
                    conf_max=conf,
                    inpoint_s=start,
                    outpoint_s=end,
                )
            )

    gap_specs: list[tuple[float, float]] = []
    cursor = 0.0
    for entry in mosaic_entries:
        if entry.start_s - cursor > 0.05:
            gap_specs.append((cursor, entry.start_s))
        cursor = max(cursor, entry.end_s)
    if duration > 0 and duration - cursor > 0.05:
        gap_specs.append((cursor, duration))

    gap_entries: list[TimelineEntry] = []
    for idx, (start, end) in enumerate(gap_specs):
        if materialize_gaps:
            path = out_dir / f"gap_seg{idx:03d}{src.suffix}"
            if log_callback:
                log_callback(f"[source-scan] keyframe copy gap segment {idx}: {start:.3f}-{end:.3f}s -> {path.name}")
            _cut_copy(src, path, start, end, log_callback=log_callback, process_callback=process_callback)
            gap_entries.append(TimelineEntry(start_s=start, end_s=end, path=path, kind="gap", conf_max=0.0))
        else:
            if log_callback:
                log_callback(f"[source-scan] virtual gap segment {idx}: {start:.3f}-{end:.3f}s -> source inpoint/outpoint")
            gap_entries.append(
                TimelineEntry(
                    start_s=start,
                    end_s=end,
                    path=src,
                    kind="gap",
                    conf_max=0.0,
                    inpoint_s=start,
                    outpoint_s=end,
                )
            )

    return sorted(mosaic_entries + gap_entries, key=lambda entry: (entry.start_s, entry.end_s, entry.kind))


def _cut_copy(input_path: str | Path, output_path: str | Path,
              start_s: float, end_s: float, log_callback=None,
              process_callback=None) -> None:
    duration = max(0.001, float(end_s) - float(start_s))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-ss", f"{float(start_s):.6f}",
        "-i", str(input_path),
        "-t", f"{duration:.6f}",
        "-map", "0:v:0",
        "-c:v", "copy",
        "-an",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    _run(cmd, log_callback=log_callback, process_callback=process_callback)


def _merge_rect(a: MosaicSegment, b: MosaicSegment) -> tuple[int, int, int, int]:
    x1 = min(a.x, b.x)
    y1 = min(a.y, b.y)
    x2 = max(a.x + a.w, b.x + b.w)
    y2 = max(a.y + a.h, b.y + b.h)
    return x1, y1, x2 - x1, y2 - y1


def _rects_overlap_with_gap(a: MosaicSegment, b: MosaicSegment, gap: int = 16) -> bool:
    return (
        a.x + a.w + gap >= b.x
        and b.x + b.w + gap >= a.x
        and a.y + a.h + gap >= b.y
        and b.y + b.h + gap >= a.y
    )


def align_segments(segments: list[MosaicSegment], keyframes: list[float],
                   duration: float | None = None) -> list[MosaicSegment]:
    if not segments:
        return []
    for seg in segments:
        if keyframes:
            seg.start_s_kf = _floor_kf(seg.start_s, keyframes)
            seg.end_s_kf = _ceil_kf(seg.end_s, keyframes, duration)
        else:
            seg.start_s_kf = seg.start_s
            seg.end_s_kf = seg.end_s
        if seg.end_s_kf <= seg.start_s_kf:
            seg.end_s_kf = max(seg.end_s, seg.start_s_kf + 0.1)

    aligned = sorted(segments, key=lambda s: (s.start_s_kf, s.end_s_kf))
    merged: list[MosaicSegment] = []
    for seg in aligned:
        if merged:
            prev = merged[-1]
            time_overlap = seg.start_s_kf <= prev.end_s_kf + 1e-3
            rect_overlap = _rects_overlap_with_gap(prev, seg)
        else:
            prev = None
            time_overlap = False
            rect_overlap = False
        if prev is not None and time_overlap and rect_overlap:
            prev.start_s = min(prev.start_s, seg.start_s)
            prev.end_s = max(prev.end_s, seg.end_s)
            prev.start_s_kf = min(prev.start_s_kf, seg.start_s_kf)
            prev.end_s_kf = max(prev.end_s_kf, seg.end_s_kf)
            prev.x, prev.y, prev.w, prev.h = _merge_rect(prev, seg)
            prev.conf_max = max(prev.conf_max, seg.conf_max)
        else:
            merged.append(seg)
    for idx, seg in enumerate(merged):
        seg.seg_id = idx
    return merged


def cut_segment(input_path: str | Path, output_path: str | Path, segment: MosaicSegment,
                log_callback=None, process_callback=None) -> None:
    from gpu_engine import probe

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    meta = probe.probe_video(input_path)
    duration = max(0.001, segment.end_s_kf - segment.start_s_kf)
    crop = f"crop={segment.w}:{segment.h}:{segment.x}:{segment.y}"
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-ss", f"{segment.start_s_kf:.6f}",
        "-i", str(input_path),
        "-t", f"{duration:.6f}",
        "-vf", crop,
        "-an",
        "-c:v", "hevc_nvenc",
        "-preset", "p7",
        "-rc", "vbr",
        "-cq", "18",
        "-g", "60",
        "-bf", "0",
    ]
    if meta.bit_depth > 8:
        cmd.extend(["-pix_fmt", "p010le", "-profile:v", "main10"])
    if meta.color is not None:
        cmd.extend(meta.color.ffmpeg_args())
    cmd.extend(["-movflags", "+faststart", str(output_path)])
    _run(cmd, log_callback=log_callback, process_callback=process_callback)
    size = get_video_size(output_path)
    expected = (int(segment.w), int(segment.h))
    if size != expected:
        raise RuntimeError(
            f"segment crop failed: output size {size}, expected {expected}, "
            f"rect={segment.x},{segment.y},{segment.w}x{segment.h}"
        )
