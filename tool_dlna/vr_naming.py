"""Centralized VR filename marker and generated-title helpers.

Only contains the physical file auto-naming mapping rules.
"""
from __future__ import annotations

import re

# Player naming references checked for this rule set:
# - HereSphere/SKYBOX/DeoVR: this project treats only underscore and hyphen as
#   reliable filename marker separators.
# - SKYBOX: 3D/angle/fisheye keywords can appear in any order/case.
# - DeoVR: local filenames use markers such as _LR_180 and _fisheye190.
_MARKER_RE = re.compile(
    r"(^|[_\-])("
    r"lr|rl|lrf|rlf|3dh|3dhf|sbs|sbsf|hsbs|"
    r"left[-_]*right|left[-_]*by[-_]*right|"
    r"half[-_]*sbs|half[-_]*side[-_]*by[-_]*side|"
    r"side[-_]*by[-_]*side|"
    r"tb|bt|tbf|btf|ou|ouf|3dv|3dvf|hou|"
    r"top[-_]*bottom|top[-_]*by[-_]*bottom|"
    r"over[-_]*under|half[-_]*ou|half[-_]*over[-_]*under|"
    r"3d|3dh|3dv|2d|"
    r"180|360|180x180|vr180|"
    r"f180|180f|fisheye|fisheye180|fisheye190|rf52|"
    r"mkx200|mkx22|vrca220|eac360|360eac"
    r")($|[_\-])",
    re.IGNORECASE,
)

SBS_180_SOURCE_SUFFIX = "_LR_180_SBS"
LEGACY_LR_180_SOURCE_SUFFIX = "_LR_180"
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".ts", ".m2ts"}


def _as_stem(stem_or_name: str) -> str:
    value = str(stem_or_name)
    dot = value.rfind(".")
    if dot > 0:
        ext = value[dot + 1:].lower()
        if f".{ext}" in _VIDEO_SUFFIXES:
            return value[:dot]
    return value


def has_vr_filename_marker(stem_or_name: str) -> bool:
    """Return whether a filename stem/name already carries known VR markers."""
    stem = _as_stem(stem_or_name)
    return bool(_MARKER_RE.search(stem))


def is_half_equirectangular_source(width: int = 0, height: int = 0) -> bool:
    width = int(width or 0)
    height = int(height or 0)
    if width <= 0 or height <= 0:
        return False
    return abs((width / height) - 2.0) <= 0.02


def source_display_stem(stem_or_name: str, width: int = 0, height: int = 0) -> str:
    """Return the DLNA display stem for a source video.

    2:1 half-equirectangular sources without known VR/player markers are exposed
    as SBS 180 virtual names so VR players enter the intended VR180 mode.
    """
    stem = _as_stem(stem_or_name)
    if is_half_equirectangular_source(width, height) and not has_vr_filename_marker(stem):
        return f"{stem}{SBS_180_SOURCE_SUFFIX}"
    if stem.lower().endswith(LEGACY_LR_180_SOURCE_SUFFIX.lower()):
        return f"{stem}_SBS"
    return stem
