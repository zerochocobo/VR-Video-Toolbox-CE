"""Paste restored pre-extract segments back onto the base video."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from gpu_engine import mux, probe
from gpu_engine.fallback import OperationCancelled
from utils import app_config
from utils.mosaic_prescan import MosaicSegment


@dataclass
class PasteSeg:
    seg_id: int
    path: Path
    base_frame_start: int
    base_frame_end: int
    start_s: float
    end_s: float
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class _PasteSubSegment:
    kind: str  # "paste" or "passthrough"
    start_frame: int
    end_frame: int


def _hidden_kwargs() -> dict:
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        return {"startupinfo": startupinfo}
    return {}


def build_paste_segments(base_path: str | Path, segments: list[MosaicSegment],
                         restored_paths: list[str | Path]) -> list[PasteSeg]:
    meta = probe.probe_video(base_path)
    fps = meta.source_fps or 30.0
    out: list[PasteSeg] = []
    for seg, restored in zip(segments, restored_paths):
        start = max(0, int(round(seg.start_s_kf * fps)))
        end = max(start + 1, int(round(seg.end_s_kf * fps)))
        out.append(PasteSeg(
            seg_id=seg.seg_id,
            path=Path(restored),
            base_frame_start=start,
            base_frame_end=end,
            start_s=seg.start_s_kf,
            end_s=seg.end_s_kf,
            x=seg.x,
            y=seg.y,
            w=seg.w,
            h=seg.h,
        ))
    return out


def _merge_frame_intervals(intervals: list[tuple[int, int]], total_frames: int) -> list[tuple[int, int]]:
    total = max(0, int(total_frames))
    clipped = []
    for start, end in intervals:
        s = max(0, min(total, int(start)))
        e = max(s, min(total, int(end)))
        if e > s:
            clipped.append((s, e))
    clipped.sort()
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _keyframe_frames(keyframes: list[float], fps: float, total_frames: int) -> list[int]:
    if fps <= 0:
        return []
    total = max(0, int(total_frames))
    frames = set()
    for ts in keyframes:
        try:
            value = float(ts)
        except Exception:
            continue
        if value >= 0.0:
            frames.add(max(0, min(total, int(round(value * fps)))))
    if total > 0:
        frames.add(0)
        frames.add(total)
    return sorted(frames)


def _ceil_frame(value: int, frames: list[int]) -> int | None:
    for frame in frames:
        if frame >= int(value):
            return frame
    return None


def _floor_frame(value: int, frames: list[int]) -> int | None:
    best = None
    for frame in frames:
        if frame <= int(value):
            best = frame
        else:
            break
    return best


def _build_passthrough_plan(segments: list[PasteSeg],
                            total_frames: int,
                            keyframes: list[float],
                            fps: float,
                            min_passthrough_frames: int,
                            max_passthrough_count: int) -> list[_PasteSubSegment]:
    total = max(0, int(total_frames))
    if total <= 0 or not segments:
        return []
    active = _merge_frame_intervals(
        [(int(seg.base_frame_start), int(seg.base_frame_end)) for seg in segments],
        total,
    )
    if not active:
        return []
    key_frames = _keyframe_frames(keyframes, fps, total)
    if not key_frames:
        return [_PasteSubSegment("paste", 0, total)]

    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in active:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        gaps.append((cursor, total))

    passthrough: list[tuple[int, int]] = []
    min_frames = max(1, int(min_passthrough_frames or 1))
    for start, end in gaps:
        aligned_start = _ceil_frame(start, key_frames)
        aligned_end = _floor_frame(end, key_frames)
        if aligned_start is None or aligned_end is None:
            continue
        if aligned_end - aligned_start >= min_frames:
            passthrough.append((aligned_start, aligned_end))

    if max_passthrough_count > 0 and len(passthrough) > int(max_passthrough_count):
        return []
    if not passthrough:
        return [_PasteSubSegment("paste", 0, total)]

    plan: list[_PasteSubSegment] = []
    cursor = 0
    for start, end in passthrough:
        if start > cursor:
            plan.append(_PasteSubSegment("paste", cursor, start))
        plan.append(_PasteSubSegment("passthrough", start, end))
        cursor = end
    if cursor < total:
        plan.append(_PasteSubSegment("paste", cursor, total))
    return [part for part in plan if part.end_frame > part.start_frame]


def _overlapping_paste_segments(segments: list[PasteSeg], start_frame: int, end_frame: int) -> list[PasteSeg]:
    start = int(start_frame)
    end = int(end_frame)
    return [
        seg for seg in segments
        if int(seg.base_frame_start) < end and int(seg.base_frame_end) > start
    ]


def _try_paste_segments_gpu_passthrough(base_path: str | Path, restored_path: str | Path,
                                        paste_segments: list[PasteSeg],
                                        keep_audio: bool,
                                        bitrate_bps: int | None,
                                        log_callback=None,
                                        process_callback=None) -> bool:
    if not bool(app_config.get("paste_passthrough_enabled", True)):
        return False
    if not paste_segments:
        return False

    base_path = Path(base_path)
    restored_path = Path(restored_path)
    meta = probe.probe_video(base_path)
    if str(meta.codec_name or "").lower() != "hevc":
        return False
    fps = float(meta.source_fps or 30.0)
    total_frames = int(meta.nb_frames or 0)
    if total_frames <= 0 and meta.duration and fps > 0:
        total_frames = int(round(float(meta.duration) * fps))
    if total_frames <= 0:
        return False

    from utils import keyframe_cutter, sbs_concat

    keyframes = keyframe_cutter.list_keyframes(base_path)
    min_frames = max(1, int(app_config.get("paste_passthrough_min_frames", 60) or 60))
    max_subseg = max(0, int(app_config.get("paste_passthrough_max_subseg", 32) or 32))
    plan = _build_passthrough_plan(
        paste_segments,
        total_frames,
        keyframes,
        fps,
        min_frames,
        max_subseg,
    )
    if not plan or not any(part.kind == "passthrough" for part in plan):
        return False

    restored_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{restored_path.stem}.passthrough_", dir=str(restored_path.parent)))
    timeline = []
    try:
        if log_callback:
            passthrough_frames = sum(
                part.end_frame - part.start_frame
                for part in plan
                if part.kind == "passthrough"
            )
            log_callback(
                f"[pre-extract] paste passthrough plan: parts={len(plan)}, "
                f"passthrough_frames={passthrough_frames}/{total_frames}"
            )

        from gpu_engine import files as gpu_files
        from utils.keyframe_cutter import TimelineEntry

        for idx, part in enumerate(plan):
            start_s = float(part.start_frame) / fps
            end_s = float(part.end_frame) / fps
            part_path = temp_dir / (
                f"{restored_path.stem}.{idx:04d}.{part.kind}."
                f"f{part.start_frame:08d}-{part.end_frame:08d}.mp4"
            )
            if part.kind == "passthrough":
                keyframe_cutter._cut_copy(
                    base_path,
                    part_path,
                    start_s,
                    end_s,
                    log_callback=log_callback,
                    process_callback=process_callback,
                )
                kind = "gap"
            else:
                active = _overlapping_paste_segments(paste_segments, part.start_frame, part.end_frame)
                if not active:
                    raise RuntimeError(
                        f"paste subsegment has no active rects: {part.start_frame}-{part.end_frame}"
                    )
                token = gpu_files.CancelToken()
                if process_callback:
                    process_callback(token)
                gpu_files.paste_segments_gpu(
                    base_path,
                    part_path,
                    active,
                    keep_audio=False,
                    bitrate_bps=bitrate_bps,
                    start_frame=part.start_frame,
                    end_frame=part.end_frame,
                    log_callback=log_callback,
                    cancel_token=token,
                )
                kind = "mosaic"
            timeline.append(TimelineEntry(start_s, end_s, part_path, kind))

        sbs_concat.concat_timeline_hevc_fast(
            timeline,
            restored_path,
            source_src=base_path,
            audio_source=base_path if keep_audio else None,
            log_callback=log_callback,
            process_callback=process_callback,
        )
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def paste_segments_gpu_or_fallback(base_path: str | Path, restored_path: str | Path,
                                   segments: list[MosaicSegment],
                                   restored_paths: list[str | Path],
                                   keep_audio: bool = True,
                                   bitrate_bps: int | None = None,
                                   log_callback=None, process_callback=None) -> None:
    paste_segments = build_paste_segments(base_path, segments, restored_paths)
    try:
        from gpu_engine import files as gpu_files

        try:
            if _try_paste_segments_gpu_passthrough(
                base_path,
                restored_path,
                paste_segments,
                keep_audio,
                bitrate_bps,
                log_callback=log_callback,
                process_callback=process_callback,
            ):
                return
        except OperationCancelled:
            raise
        except Exception as exc:
            if log_callback:
                log_callback(
                    f"[pre-extract] paste passthrough failed: {type(exc).__name__}: {exc}; "
                    f"retrying full GPU paste"
                )

        token = gpu_files.CancelToken()
        if process_callback:
            process_callback(token)
        gpu_files.paste_segments_gpu(
            base_path,
            restored_path,
            paste_segments,
            keep_audio=keep_audio,
            bitrate_bps=bitrate_bps,
            log_callback=log_callback,
            cancel_token=token,
        )
        return
    except OperationCancelled:
        raise
    except Exception as exc:
        if any(seg.path.suffix.lower() == ".hevc" for seg in paste_segments):
            temp_inputs: list[Path] = []
            try:
                if log_callback:
                    log_callback(
                        f"[pre-extract] GPU paste failed on raw HEVC input: {type(exc).__name__}: {exc}; "
                        "retrying with temporary mp4-wrapped restored segments"
                    )
                from gpu_engine import files as gpu_files

                materialized_segments, temp_inputs = _materialize_raw_hevc_segments(
                    restored_path,
                    paste_segments,
                    log_callback=log_callback,
                )
                token = gpu_files.CancelToken()
                if process_callback:
                    process_callback(token)
                gpu_files.paste_segments_gpu(
                    base_path,
                    restored_path,
                    materialized_segments,
                    keep_audio=keep_audio,
                    bitrate_bps=bitrate_bps,
                    log_callback=log_callback,
                    cancel_token=token,
                )
                return
            except OperationCancelled:
                raise
            except Exception as retry_exc:
                if log_callback:
                    log_callback(
                        f"[pre-extract] GPU paste retry with mp4-wrapped raw failed: "
                        f"{type(retry_exc).__name__}: {retry_exc}; using ffmpeg overlay fallback"
                    )
            finally:
                for path in temp_inputs:
                    try:
                        path.unlink()
                    except OSError:
                        pass
        if log_callback:
            log_callback(f"[pre-extract] GPU paste failed: {type(exc).__name__}: {exc}; using ffmpeg overlay fallback")
    _paste_segments_ffmpeg(
        base_path,
        restored_path,
        paste_segments,
        keep_audio,
        log_callback,
        process_callback,
        bitrate_bps=bitrate_bps,
    )


def _materialize_raw_hevc_segments(restored_path: str | Path,
                                   segments: list[PasteSeg],
                                   log_callback=None) -> tuple[list[PasteSeg], list[Path]]:
    temp_inputs: list[Path] = []
    materialized_segments: list[PasteSeg] = []
    parent = Path(restored_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        if seg.path.suffix.lower() == ".hevc":
            from gpu_engine import restored_sidecar

            meta = restored_sidecar.metadata_from_sidecar(seg.path)
            if meta is None:
                raise RuntimeError(f"raw restored segment has no sidecar: {seg.path}")
            fd, temp_name = tempfile.mkstemp(
                prefix=f"{seg.path.stem}.wrapped_",
                suffix=".mp4",
                dir=str(parent),
            )
            os.close(fd)
            temp_mp4 = Path(temp_name)
            try:
                temp_mp4.unlink()
            except OSError:
                pass
            mux.mux_hevc_with_audio(
                seg.path,
                temp_mp4,
                fps=meta.source_fps or 30.0,
                color=meta.color,
                audio_source=None,
                log_callback=log_callback,
            )
            temp_inputs.append(temp_mp4)
            materialized_segments.append(replace(seg, path=temp_mp4))
        else:
            materialized_segments.append(seg)
    return materialized_segments, temp_inputs


def _paste_segments_ffmpeg(base_path: str | Path, restored_path: str | Path,
                           segments: list[PasteSeg], keep_audio: bool = True, log_callback=None,
                           process_callback=None, bitrate_bps: int | None = None) -> None:
    if not segments:
        shutil.copy2(base_path, restored_path)
        return

    temp_inputs: list[Path] = []
    try:
        materialized_segments, temp_inputs = _materialize_raw_hevc_segments(
            restored_path,
            segments,
            log_callback=log_callback,
        )
        _paste_segments_ffmpeg_impl(
            base_path,
            restored_path,
            materialized_segments,
            keep_audio,
            log_callback,
            process_callback,
            bitrate_bps=bitrate_bps,
        )
    finally:
        for path in temp_inputs:
            try:
                path.unlink()
            except OSError:
                pass


def _paste_segments_ffmpeg_impl(base_path: str | Path, restored_path: str | Path,
                                segments: list[PasteSeg], keep_audio: bool = True, log_callback=None,
                                process_callback=None, bitrate_bps: int | None = None) -> None:
    if not segments:
        shutil.copy2(base_path, restored_path)
        return

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y", "-i", str(base_path)]
    for seg in segments:
        cmd.extend(["-i", str(seg.path)])

    filter_parts: list[str] = []
    last = "[0:v]"
    for idx, seg in enumerate(segments, start=1):
        shifted = f"[s{idx}]"
        out = f"[v{idx}]"
        filter_parts.append(f"[{idx}:v]setpts=PTS+{seg.start_s:.6f}/TB{shifted}")
        filter_parts.append(
            f"{last}{shifted}overlay={seg.x}:{seg.y}:"
            f"enable='between(t,{seg.start_s:.6f},{seg.end_s:.6f})'{out}"
        )
        last = out

    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", last,
    ])
    cmd.extend([
        "-map", "0:a?",
        "-c:a", "copy",
    ] if keep_audio else [
        "-an",
    ])
    cmd.extend([
        "-c:v", "hevc_nvenc",
        "-preset", "p7",
        "-rc", "vbr",
    ])
    if bitrate_bps and bitrate_bps > 0:
        kbps = max(1, int(bitrate_bps) // 1000)
        cmd.extend([
            "-b:v", f"{kbps}k",
            "-maxrate:v", f"{int(kbps * 1.2)}k",
            "-bufsize:v", f"{int(kbps * 2)}k",
        ])
    else:
        cmd.extend(["-cq", "18"])
    size_hint = Path(base_path).stat().st_size if Path(base_path).exists() else None
    cmd.extend([
        "-g", "60",
        "-bf", "0",
    ])
    cmd.extend(mux.faststart_args(size_hint))
    cmd.extend([str(restored_path)])
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
        raise RuntimeError(f"ffmpeg overlay paste failed with code {proc.returncode}")
