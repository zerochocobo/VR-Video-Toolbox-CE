"""Sidecar metadata for restored raw HEVC intermediates."""
from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path

from gpu_engine.probe import ColorMetadata, VideoMetadata


def raw_path_for_output(output_path: str | Path) -> Path:
    return Path(output_path).with_suffix(".hevc")


def sidecar_path_for(raw_path: str | Path) -> Path:
    return Path(raw_path).with_suffix(".json")


def is_restored_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".hevc" and sidecar_path_for(path).exists()


def _fps_parts(fps: float) -> tuple[int, int]:
    frac = Fraction(float(fps or 30.0)).limit_denominator(1001000)
    return int(frac.numerator), int(frac.denominator)


def _color_to_dict(color: ColorMetadata | None) -> dict:
    color = color or ColorMetadata()
    return {
        "primaries": color.color_primaries,
        "transfer": color.color_transfer,
        "matrix": color.color_space,
        "range": color.color_range,
    }


def _color_from_dict(data: dict | None) -> ColorMetadata:
    data = data or {}
    return ColorMetadata(
        color_range=str(data.get("range") or data.get("color_range") or ""),
        color_space=str(data.get("matrix") or data.get("color_space") or ""),
        color_transfer=str(data.get("transfer") or data.get("color_transfer") or ""),
        color_primaries=str(data.get("primaries") or data.get("color_primaries") or ""),
    )


def write_restored_sidecar(
    raw_path: str | Path,
    *,
    width: int,
    height: int,
    bit_depth: int,
    fps: float,
    frame_count: int,
    color: ColorMetadata | None = None,
    source: str | Path | None = None,
    rect: dict | None = None,
    time_range: dict | None = None,
    encoder: str = "",
) -> Path:
    raw = Path(raw_path)
    fps_num, fps_den = _fps_parts(fps)
    data = {
        "format_version": 1,
        "kind": "restored",
        "codec": "hevc",
        "width": int(width),
        "height": int(height),
        "bit_depth": int(bit_depth),
        "fps_num": fps_num,
        "fps_den": fps_den,
        "frame_count": int(frame_count),
        "color": _color_to_dict(color),
        "encoder": str(encoder or ""),
        "source": str(source or ""),
    }
    if rect is not None:
        data["rect"] = rect
    if time_range is not None:
        data["time"] = time_range

    sidecar = sidecar_path_for(raw)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_name(f"{sidecar.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, sidecar)
    return sidecar


def load_restored_sidecar(path: str | Path) -> dict:
    p = Path(path)
    sidecar = p if p.suffix.lower() == ".json" else sidecar_path_for(p)
    return json.loads(sidecar.read_text(encoding="utf-8-sig"))


def metadata_from_sidecar(path: str | Path) -> VideoMetadata | None:
    try:
        data = load_restored_sidecar(path)
    except Exception:
        return None
    fps_den = int(data.get("fps_den") or 1)
    fps = float(int(data.get("fps_num") or 0)) / max(1, fps_den)
    frame_count = int(data.get("frame_count") or 0)
    duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return VideoMetadata(
        path=str(path),
        codec_name=str(data.get("codec") or "hevc"),
        pix_fmt="p010le" if int(data.get("bit_depth") or 8) > 8 else "nv12",
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        bit_depth=int(data.get("bit_depth") or 8),
        duration=duration,
        nb_frames=frame_count,
        source_fps=fps or 30.0,
        is_cfr=True,
        bitrate_bps=0,
        color=_color_from_dict(data.get("color")),
        audio_codec="",
    )


def frame_count_from_sidecar(path: str | Path) -> int | None:
    try:
        value = int(load_restored_sidecar(path).get("frame_count") or 0)
    except Exception:
        return None
    return value if value > 0 else None
