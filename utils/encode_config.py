from __future__ import annotations

from dataclasses import dataclass

from . import app_config

VALID_NVENC_PRESETS = {f"P{i}" for i in range(1, 8)}
ENCODE_PROFILE_ORDER = ("highest_quality", "balanced_high_quality", "fast_quality", "ultra_fast_normal")
DEFAULT_ENCODE_PROFILE = "balanced_high_quality"

_ENCODE_PROFILES: dict[str, dict[str, object]] = {
    "highest_quality": {
        "preset": "P7",
        "multipass": "fullres",
        "aq": True,
        "temporal_aq": False,
        "aq_strength": 6,
    },
    "balanced_high_quality": {
        "preset": "P4",
        "multipass": "fullres",
        "aq": True,
        "temporal_aq": False,
        "aq_strength": 6,
    },
    "fast_quality": {
        "preset": "P1",
        "multipass": "fullres",
        "aq": True,
        "temporal_aq": False,
        "aq_strength": 6,
    },
    "ultra_fast_normal": {
        "preset": "P1",
        "multipass": "off",
        "aq": True,
        "temporal_aq": False,
        "aq_strength": 6,
    },
}

_PROFILE_ALIASES = {
    # Before 2026-06-25 this key meant P4+AQ+fullres. Keep old saved configs
    # on the same effective settings after the user-facing rename.
    "maximum_quality": "balanced_high_quality",
}


@dataclass(frozen=True)
class EncodeSettings:
    profile_key: str
    preset: str
    multipass: str
    aq: bool
    temporal_aq: bool
    aq_strength: int = 6


def _cfg_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _normalize_preset(value, default: str = "P4") -> str:
    preset = str(value or default).upper()
    return preset if preset in VALID_NVENC_PRESETS else default


def _normalize_multipass(value, default: str = "fullres") -> str:
    multipass = str(value or default).strip().lower()
    return multipass if multipass in {"off", "qres", "fullres"} else default


def _normalize_aq_strength(value, default: int = 6) -> int:
    try:
        strength = int(value)
    except (TypeError, ValueError):
        strength = int(default)
    return max(1, min(15, strength))


def get_profile_definition(key: str) -> dict[str, object]:
    return dict(_ENCODE_PROFILES[_normalize_profile_key(key)])


def get_profile_keys() -> tuple[str, ...]:
    return ENCODE_PROFILE_ORDER


def _normalize_profile_key(key: str | None) -> str:
    value = str(key or "").strip()
    return _PROFILE_ALIASES.get(value, value)


def resolve_encode_settings(profile_key: str | None = None) -> EncodeSettings:
    stored = _normalize_profile_key(
        str(profile_key if profile_key is not None else app_config.get("gpu_encode_profile", "") or "").strip()
    )
    if stored in _ENCODE_PROFILES:
        definition = _ENCODE_PROFILES[stored]
        return EncodeSettings(
            profile_key=stored,
            preset=str(definition["preset"]),
            multipass=str(definition["multipass"]),
            aq=bool(definition["aq"]),
            temporal_aq=bool(definition["temporal_aq"]),
            aq_strength=int(definition.get("aq_strength", 6)),
        )
    return EncodeSettings(
        profile_key="custom",
        preset=_normalize_preset(app_config.get("gpu_encode_preset", "P4"), "P4"),
        multipass=_normalize_multipass(app_config.get("gpu_encode_multipass", "fullres"), "fullres"),
        aq=_cfg_bool(app_config.get("gpu_encode_aq", True), True),
        temporal_aq=_cfg_bool(app_config.get("gpu_encode_temporal_aq", False), False),
        aq_strength=_normalize_aq_strength(app_config.get("gpu_encode_aq_strength", 6), 6),
    )


def current_encode_profile_key() -> str | None:
    stored = _normalize_profile_key(str(app_config.get("gpu_encode_profile", "") or "").strip())
    if stored in _ENCODE_PROFILES:
        return stored

    current = resolve_encode_settings("custom")
    for key in ENCODE_PROFILE_ORDER:
        profile = _ENCODE_PROFILES[key]
        if (
            current.preset == str(profile["preset"])
            and current.multipass == str(profile["multipass"])
            and current.aq == bool(profile["aq"])
            and current.temporal_aq == bool(profile["temporal_aq"])
            and current.aq_strength == int(profile.get("aq_strength", 6))
        ):
            return key

    return None


def apply_encode_profile(key: str) -> None:
    key = _normalize_profile_key(key)
    _ENCODE_PROFILES[key]
    app_config.set("gpu_encode_profile", key)


def build_pynv_encoder_kwargs() -> dict[str, str]:
    settings = resolve_encode_settings()
    kwargs: dict[str, str] = {
        "preset": settings.preset,
    }
    if settings.aq:
        kwargs["aq"] = "1"
    if settings.temporal_aq:
        kwargs["temporalaq"] = "1"
    if settings.multipass in {"fullres", "qres"}:
        kwargs["multipass"] = settings.multipass
    # Do not pass aq_strength here until PyNvVideoCodec's exact key is verified
    # on a real machine. Unknown encoder kwargs can fail setup in the main GPU
    # resident path, while ffmpeg/PyAV use aq-strength below.
    return kwargs


def maxrate_multiplier(default: float = 2.0) -> float:
    """Return the shared VBR peak-rate multiplier for non-explicit maxrate paths."""
    try:
        value = float(app_config.get("gpu_encode_maxrate_multiplier", default) or default)
    except (TypeError, ValueError):
        value = float(default)
    return max(1.0, value)


def final_maxrate_multiplier(default: float = 1.1) -> float:
    """VBR peak-rate multiplier for the FINAL delivered encode only.

    The effective value comes from ``app_config`` (``gpu_final_encode_maxrate_multiplier``,
    a code-default-only key whose default lives in ``app_config._DEFAULTS`` = 1.1);
    the ``default`` argument here only applies if that entry is ever removed.

    Intermediate crop/restore stages keep the looser ``maxrate_multiplier``
    headroom (they are decoded and re-encoded again downstream, so peak quality
    matters there). The final paste/merge/concat encode tightens the VBR ceiling
    so the delivered file converges near the kept source bitrate instead of
    drifting up toward a 2x peak.

    For bit-hungry restored content (notably fisheye-warped 8K) NVENC VBR hugs the
    ceiling, so the delivered average lands close to ``multiplier x source``. 1.1
    keeps the output ~1.1x source; it is a *ceiling*, so content that already fits
    under it is untouched and only genuine over-allocation gets clamped. Raise
    toward 2.0 for the old behaviour.
    """
    try:
        value = float(app_config.get("gpu_final_encode_maxrate_multiplier", default) or default)
    except (TypeError, ValueError):
        value = float(default)
    return max(1.0, value)


def build_ffmpeg_pix_fmt_args(bit_depth: int | str | None) -> list[str]:
    """Return explicit ffmpeg output pixel-format args for NVENC paths."""
    try:
        depth = int(bit_depth or 8)
    except (TypeError, ValueError):
        depth = 8
    if depth > 8:
        return ["-pix_fmt", "p010le", "-profile:v", "main10"]
    return ["-pix_fmt", "yuv420p"]


_BFRAMES_SUPPORTED: bool | None = None


def _run_nvenc_bframe_probe(bframes: int) -> bool:
    """Run a tiny hevc_nvenc encode and report whether ffmpeg exited cleanly."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=128x128:r=30:d=0.1",
        "-c:v", "hevc_nvenc", "-bf", str(int(bframes)), "-f", "null", "-",
    ]
    try:
        from utils.ffmpeg_checker import get_startupinfo
        startupinfo = get_startupinfo()
    except Exception:
        startupinfo = None
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            startupinfo=startupinfo, timeout=30,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _probe_hevc_nvenc_bframes() -> bool:
    """Decide whether hevc_nvenc on this machine accepts B-frames.

    Returns True when B-frames work, OR when NVENC is unavailable for an
    unrelated reason (so a missing GPU / ffmpeg never masquerades as a B-frame
    problem and we don't silently strip a working knob). Returns False only when
    NVENC encodes fine at -bf 0 but rejects -bf 2 — the older Maxwell/Pascal
    HEVC case the fallback exists for.
    """
    if _run_nvenc_bframe_probe(2):
        return True
    if _run_nvenc_bframe_probe(0):
        import logging
        logging.getLogger(__name__).warning(
            "hevc_nvenc rejected -bf 2 but works at -bf 0; final re-encode B-frames disabled."
        )
        return False
    return True


def hevc_nvenc_supports_bframes() -> bool:
    """Cached capability check; override with env VRVT_NVENC_BFRAMES=1/0."""
    import os

    override = os.environ.get("VRVT_NVENC_BFRAMES")
    if override is not None:
        return override.strip().lower() not in {"", "0", "false", "no", "off"}
    global _BFRAMES_SUPPORTED
    if _BFRAMES_SUPPORTED is None:
        _BFRAMES_SUPPORTED = _probe_hevc_nvenc_bframes()
    return _BFRAMES_SUPPORTED


def final_reencode_bframes(default: int = 2) -> int:
    """Return B-frame count for final ffmpeg re-encode stages only.

    Falls back to 0 when the GPU's hevc_nvenc rejects B-frames, so the final
    merge/concat/projection encode does not hard-fail on older NVENC silicon.
    """
    try:
        value = int(app_config.get("gpu_final_encode_bframes", default))
    except (TypeError, ValueError):
        value = int(default)
    value = max(0, min(4, value))
    if value > 0 and not hevc_nvenc_supports_bframes():
        return 0
    return value


def final_reencode_gop_frames(fps: float | int | None, default_seconds: float = 2.0) -> int:
    """Return GOP frame count for final ffmpeg re-encode stages; 0 means omit -g."""
    try:
        seconds = float(app_config.get("gpu_final_encode_gop_sec", default_seconds) or default_seconds)
    except (TypeError, ValueError):
        seconds = float(default_seconds)
    seconds = max(0.0, min(10.0, seconds))
    if seconds <= 0.0:
        return 0
    try:
        frame_rate = float(fps or 0.0)
    except (TypeError, ValueError):
        frame_rate = 0.0
    if frame_rate <= 0.0:
        frame_rate = 30.0
    return max(1, int(round(frame_rate * seconds)))


def build_final_ffmpeg_reencode_tail_args(fps: float | int | None = None) -> list[str]:
    """Return final-only efficiency args. Do not use for paste/crop intermediates."""
    args: list[str] = []
    gop = final_reencode_gop_frames(fps)
    if gop > 0:
        args.extend(["-g", str(gop)])
    args.extend(["-bf", str(final_reencode_bframes())])
    return args


def build_ffmpeg_nvenc_base_args() -> list[str]:
    settings = resolve_encode_settings()
    args = [
        "-c:v", "hevc_nvenc",
        "-preset", settings.preset.lower(),
        "-tune", "hq",
    ]
    if settings.aq:
        args.extend(["-spatial_aq", "1", "-aq-strength", str(settings.aq_strength)])
    if settings.temporal_aq:
        args.extend(["-temporal_aq", "1"])
    if settings.multipass in {"fullres", "qres"}:
        args.extend(["-multipass", settings.multipass])
    return args


def build_lada_encoder_options(cq: int | str = 18) -> str:
    settings = resolve_encode_settings()
    # Keep every option as a key/value pair. The native Lada VideoWriter fallback
    # parses this string as pairs for PyAV stream.options; a valueless flag would
    # shift all following keys and silently corrupt the option mapping.
    args = [
        "-rc vbr",
        f"-cq {cq}",
        f"-preset {settings.preset.lower()}",
        "-tune hq",
    ]
    if settings.aq:
        args.extend(["-spatial_aq 1", f"-aq-strength {settings.aq_strength}"])
    if settings.temporal_aq:
        args.append("-temporal_aq 1")
    if settings.multipass in {"fullres", "qres"}:
        args.append(f"-multipass {settings.multipass}")
    return " " + " ".join(args)
