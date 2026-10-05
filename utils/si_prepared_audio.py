"""Portable contract shared with PTMediaServer / VR Video Toolbox.

Keep this module identical in both repositories. Only a committed JSON manifest
whose inputs, filter and output still match can enable direct AAC reuse.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "pt-si-mix-v1"
AUDIO_FORMAT = {"codec": "aac", "sample_rate": 48000, "channels": 2, "bit_rate": 192000}


def prepared_paths(video: str | Path) -> tuple[Path, Path]:
    video = Path(video)
    return video.with_suffix(".si.mix.m4a"), video.with_suffix(".si.mix.json")


def file_signature(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def input_signature(video: Path, voice: Path, duck_key: Path | None) -> dict:
    return {
        "video": file_signature(video),
        "voice": file_signature(voice),
        "duck_key": file_signature(duck_key) if duck_key is not None else None,
    }


def make_manifest(inputs: dict, filter_string: str, audio: Path) -> dict:
    return {
        "schema": SCHEMA,
        "inputs": inputs,
        "filter": filter_string,
        "format": AUDIO_FORMAT,
        "audio": file_signature(audio),
    }


def find_prepared_audio(
    video: Path, voice: Path, duck_key: Path | None, filter_string: str,
) -> Path | None:
    audio, metadata = prepared_paths(video)
    try:
        # Bound corrupt/unrelated metadata without reading any video payload.
        if metadata.stat().st_size > 65536:
            return None
        data = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if data.get("schema") != SCHEMA or data.get("format") != AUDIO_FORMAT:
            return None
        if data.get("filter") != filter_string:
            return None
        if data.get("inputs") != input_signature(video, voice, duck_key):
            return None
        signature = file_signature(audio)
        if signature["size"] <= 0 or data.get("audio") != signature:
            return None
        return audio
    except (OSError, ValueError, TypeError):
        return None
