"""ffmpeg muxing helpers for packaging raw HEVC bitstreams from NVENC with source audio into mp4.

PyNv encoders emit raw HEVC only, with no container, no colr atom, and no audio.
This module uses `ffmpeg -c copy` to package the video stream as-is, copy audio
from the source when needed, and explicitly write color metadata so HDR/10-bit
information is preserved end to end.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .probe import ColorMetadata

# Color name -> HEVC VUI numeric code (ITU-T H.265 Table E.3/E.4/E.5).
_PRIMARIES_CODE = {"bt709": 1, "unspecified": 2, "bt470bg": 5, "smpte170m": 6, "bt2020": 9}
_TRANSFER_CODE = {"bt709": 1, "unspecified": 2, "smpte170m": 6, "smpte2084": 16, "arib-std-b67": 18}
_MATRIX_CODE = {"bt709": 1, "unspecified": 2, "bt470bg": 5, "smpte170m": 6, "bt2020nc": 9, "bt2020_ncl": 9}
_FASTSTART_AUTO_DISABLE_BYTES = 4 * 1024 * 1024 * 1024


def _hevc_metadata_bsf(color: ColorMetadata) -> str | None:
    """Build a hevc_metadata bitstream filter from color metadata to write VUI into the HEVC stream.

    `-c:v copy` does not inject container-level parameters such as `-colorspace`
    into raw HEVC VUI. The hevc_metadata bitstream filter ensures players,
    including VR headsets, interpret the stream with the correct color metadata.
    """
    opts: list[str] = []
    cr = color.color_range.lower()
    if cr in {"tv", "mpeg", "limited"}:
        opts.append("video_full_range_flag=0")
    elif cr in {"pc", "jpeg", "full"}:
        opts.append("video_full_range_flag=1")
    prim = _PRIMARIES_CODE.get(color.color_primaries.lower())
    trc = _TRANSFER_CODE.get(color.color_transfer.lower())
    mtx = _MATRIX_CODE.get(color.color_space.lower())
    if prim is not None:
        opts.append(f"colour_primaries={prim}")
    if trc is not None:
        opts.append(f"transfer_characteristics={trc}")
    if mtx is not None:
        opts.append(f"matrix_coefficients={mtx}")
    if not opts:
        return None
    return "hevc_metadata=" + ":".join(opts)


def _hidden_kwargs() -> dict:
    if sys.platform.startswith("win"):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        return {"startupinfo": si}
    return {}


def _cfg(key: str, default):
    try:
        from utils import app_config

        value = app_config.get(key, default)
        return default if value is None else value
    except Exception:
        return default


def should_use_faststart(candidate_size_bytes: int | None = None, mode: object | None = None) -> bool:
    """Resolve the mp4 faststart policy.

    ``auto`` keeps faststart for small files and disables it for very large
    local outputs, avoiding a full-file rewrite after muxing.
    """
    value = str(mode if mode is not None else _cfg("output_mp4_faststart", "auto") or "auto").strip().lower()
    if value in {"always", "on", "true", "1", "yes"}:
        return True
    if value in {"off", "false", "0", "no", "none"}:
        return False
    try:
        size = int(candidate_size_bytes or 0)
    except (TypeError, ValueError):
        size = 0
    return size <= _FASTSTART_AUTO_DISABLE_BYTES


def faststart_args(candidate_size_bytes: int | None = None, mode: object | None = None) -> list[str]:
    return ["-movflags", "+faststart"] if should_use_faststart(candidate_size_bytes, mode) else []


def mux_hevc_with_audio(
    raw_hevc: str | Path,
    out_path: str | Path,
    *,
    fps: float,
    color: ColorMetadata | None = None,
    audio_source: str | Path | None = None,
    audio_start_sec: float | None = None,
    audio_duration: float | None = None,
    shortest: bool = True,
    faststart: object | None = None,
    log_callback=None,
) -> None:
    """Package a raw HEVC bitstream into mp4, optionally copying source audio.

    raw_hevc      : raw .hevc file written by NVENC
    out_path      : output mp4
    fps           : frame rate; raw streams have no timestamps and need explicit framerate
    color         : color metadata written to the container
    audio_source  : source file to copy audio from; None means no audio
    audio_start_sec / audio_duration : audio trimming aligned with the video's -ss/-to
    shortest      : whether to pass -shortest. It should be True for single-segment
                    synchronized output. For multi-segment fast HEVC merges, the
                    combined.hevc frame count and source audio duration may drift
                    slightly, causing -shortest to misclassify and drop the whole
                    audio stream, so it must be False there.
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    raw_hevc = str(raw_hevc)
    final_out = Path(out_path)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    out_path = str(final_out)

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y"]
    cmd += ["-f", "hevc", "-framerate", f"{fps:.6f}", "-i", raw_hevc]

    has_audio = audio_source is not None
    if has_audio:
        if audio_start_sec and audio_start_sec > 0.001:
            cmd += ["-ss", f"{audio_start_sec:.3f}"]
        if audio_duration and audio_duration > 0:
            cmd += ["-t", f"{audio_duration:.3f}"]
        cmd += ["-i", str(audio_source)]

    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0?"]

    cmd += ["-c:v", "copy"]
    if has_audio:
        cmd += ["-c:a", "copy"]

    if color is not None:
        # Container-level tags plus bitstream VUI injection for a redundant safeguard.
        cmd += color.ffmpeg_args()
        bsf = _hevc_metadata_bsf(color)
        if bsf:
            cmd += ["-bsf:v", bsf]

    raw_size = Path(raw_hevc).stat().st_size if Path(raw_hevc).exists() else 0
    cmd += faststart_args(raw_size, faststart)
    if has_audio:
        if shortest:
            cmd += ["-shortest"]
    else:
        cmd += ["-an"]
    cmd += [out_path]

    if log_callback:
        log_callback(
            f"[mux] setup: raw={raw_hevc} exists={Path(raw_hevc).exists()} "
            f"size={raw_size} faststart={should_use_faststart(raw_size, faststart)} "
            f"out={out_path} parent_exists={final_out.parent.exists()} "
            f"audio={audio_source} audio_exists={Path(audio_source).exists() if audio_source is not None else False}"
        )
        log_callback(f"[mux] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", **_hidden_kwargs(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed (code {proc.returncode}): {proc.stdout}")
