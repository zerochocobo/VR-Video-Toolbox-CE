"""Paste restored pre-extract segments back onto the base video."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from gpu_engine import probe
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


def paste_segments_gpu_or_fallback(base_path: str | Path, restored_path: str | Path,
                                   segments: list[MosaicSegment],
                                   restored_paths: list[str | Path],
                                   keep_audio: bool = True,
                                   log_callback=None, process_callback=None) -> None:
    paste_segments = build_paste_segments(base_path, segments, restored_paths)
    try:
        from gpu_engine import files as gpu_files

        token = gpu_files.CancelToken()
        if process_callback:
            process_callback(token)
        gpu_files.paste_segments_gpu(
            base_path,
            restored_path,
            paste_segments,
            keep_audio=keep_audio,
            log_callback=log_callback,
            cancel_token=token,
        )
        return
    except Exception as exc:
        if log_callback:
            log_callback(f"[pre-extract] GPU paste failed: {type(exc).__name__}: {exc}; using ffmpeg overlay fallback")
    _paste_segments_ffmpeg(base_path, restored_path, paste_segments, keep_audio, log_callback, process_callback)


def _paste_segments_ffmpeg(base_path: str | Path, restored_path: str | Path,
                           segments: list[PasteSeg], keep_audio: bool = True, log_callback=None,
                           process_callback=None) -> None:
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
        "-cq", "18",
        "-g", "60",
        "-bf", "0",
        "-movflags", "+faststart",
        str(restored_path),
    ])
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
