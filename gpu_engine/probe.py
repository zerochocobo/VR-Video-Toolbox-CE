"""ffprobe metadata probing plus GPU/ffmpeg routing decisions.

Routing policy aligned with reference/PTMediaServer/utils/video_metadata.py:select_backend():
  - 8-bit yuv420p/nv12 SDR        -> gpu_nv12
  - 10-bit p010 bt709 SDR         -> gpu_p016  (required for the first project phase)
  - HDR10 / HLG / bt2020 / VFR    -> ffmpeg_fallback
  - Sources NVDEC cannot decode, such as MPEG-4 ASP -> ffmpeg_fallback
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

Backend = Literal["gpu_nv12", "gpu_p016", "ffmpeg_fallback"]

# Source pixel formats that PyNv/NVDEC can decode reliably.
_GPU_PIX_FMTS = {"yuv420p", "yuvj420p", "nv12", "p010le", "yuv420p10le"}
_GPU_CODECS = {"h264", "hevc"}  # av1/vp9 support depends on the GPU; fall back to ffmpeg conservatively for the first phase.
_SDR_OK = {"", "bt709", "unknown", "unspecified", "reserved"}
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def _hidden_kwargs() -> dict:
    if sys.platform.startswith("win"):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        return {"startupinfo": si}
    return {}


def _parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    text = str(value).strip()
    if not text or text == "0/0":
        return 0.0
    try:
        return float(Fraction(text))
    except Exception:
        try:
            return float(text)
        except Exception:
            return 0.0


@dataclass(frozen=True)
class ColorMetadata:
    color_range: str = ""
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""

    def ffmpeg_args(self) -> list[str]:
        """Build ffmpeg output color metadata arguments for end-to-end passthrough."""
        args: list[str] = []
        if self.color_range:
            args += ["-color_range", self.color_range]
        if self.color_primaries:
            args += ["-color_primaries", self.color_primaries]
        if self.color_transfer:
            args += ["-color_trc", self.color_transfer]
        if self.color_space:
            args += ["-colorspace", self.color_space]
        return args


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    codec_name: str = ""
    profile: str = ""
    pix_fmt: str = ""
    width: int = 0
    height: int = 0
    bit_depth: int = 8
    duration: float = 0.0
    nb_frames: int = 0
    source_fps: float = 0.0
    is_cfr: bool = True
    bitrate_bps: int = 0
    color: ColorMetadata = ColorMetadata()
    audio_codec: str = ""

    @property
    def is_hdr(self) -> bool:
        return self.color.color_transfer.lower() in _HDR_TRANSFERS

    @property
    def is_bt2020(self) -> bool:
        return "bt2020" in self.color.color_primaries.lower() or "bt2020" in self.color.color_space.lower()


@dataclass(frozen=True)
class BackendDecision:
    backend: Backend
    reason: str

    @property
    def is_gpu(self) -> bool:
        return self.backend in {"gpu_nv12", "gpu_p016"}

    @property
    def bit_depth(self) -> int:
        return 10 if self.backend == "gpu_p016" else 8


_probe_cache: dict[tuple, VideoMetadata] = {}


def probe_video(path: str | Path) -> VideoMetadata:
    """Probe video metadata with ffprobe, cached by path + size + mtime."""
    p = Path(path)
    try:
        st = p.stat()
        key = (str(p.resolve()), st.st_size, int(st.st_mtime))
    except OSError:
        key = (str(p), 0, 0)
    cached = _probe_cache.get(key)
    if cached is not None:
        return cached

    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe, "-hide_banner", "-v", "error", "-show_streams",
        "-show_entries", "format=duration,bit_rate",
        "-of", "json", str(p),
    ]
    video: dict = {}
    audio: dict = {}
    fmt: dict = {}
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **_hidden_kwargs())
        data = json.loads(out)
        streams = data.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), streams[0] if streams else {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        fmt = data.get("format") or {}
    except Exception:
        pass

    r_fps = _parse_rate(video.get("r_frame_rate"))
    avg_fps = _parse_rate(video.get("avg_frame_rate"))
    source_fps = avg_fps or r_fps or 30.0
    diff = abs(r_fps - avg_fps) / max(r_fps, avg_fps) if r_fps > 0 and avg_fps > 0 else 0.0
    is_cfr = r_fps > 0 and avg_fps > 0 and diff < 0.01

    pix_fmt = str(video.get("pix_fmt") or "")
    try:
        bit_depth = int(video.get("bits_per_raw_sample") or 0)
    except Exception:
        bit_depth = 0
    if bit_depth <= 0:
        bit_depth = 10 if "10" in pix_fmt else 8

    def _i(v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    def _f(v) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    meta = VideoMetadata(
        path=str(p),
        codec_name=str(video.get("codec_name") or ""),
        profile=str(video.get("profile") or ""),
        pix_fmt=pix_fmt,
        width=_i(video.get("width")),
        height=_i(video.get("height")),
        bit_depth=bit_depth,
        duration=_f(video.get("duration") or fmt.get("duration")),
        nb_frames=_i(video.get("nb_frames")),
        source_fps=source_fps,
        is_cfr=is_cfr,
        bitrate_bps=_i(video.get("bit_rate") or fmt.get("bit_rate")),
        color=ColorMetadata(
            color_range=str(video.get("color_range") or ""),
            color_space=str(video.get("color_space") or ""),
            color_transfer=str(video.get("color_transfer") or ""),
            color_primaries=str(video.get("color_primaries") or ""),
        ),
        audio_codec=str(audio.get("codec_name") or ""),
    )
    _probe_cache[key] = meta
    return meta


def decide_backend(meta: VideoMetadata) -> BackendDecision:
    """Decide whether metadata should route to GPU or fall back to ffmpeg."""
    codec = meta.codec_name.lower()
    pix_fmt = meta.pix_fmt.lower()
    profile = meta.profile.lower()
    transfer = meta.color.color_transfer.lower()
    primaries = meta.color.color_primaries.lower()

    if codec not in _GPU_CODECS:
        return BackendDecision("ffmpeg_fallback", f"codec {codec or 'unknown'} not on GPU route")
    if meta.width <= 0 or meta.height <= 0:
        return BackendDecision("ffmpeg_fallback", "missing dimensions")
    if not meta.is_cfr:
        return BackendDecision("ffmpeg_fallback", "VFR / weak-CFR source needs timestamp-preserving ffmpeg path")
    if pix_fmt and pix_fmt not in _GPU_PIX_FMTS:
        return BackendDecision("ffmpeg_fallback", f"pixel format {pix_fmt} not safe for PyNv NV12/P010 route")
    if meta.is_hdr:
        return BackendDecision("ffmpeg_fallback", f"HDR transfer {transfer} needs separate color policy")
    if meta.is_bt2020:
        # Treat bt2020 in either primaries or colorspace as wide gamut and fall back to ffmpeg.
        # Some mp4 containers retain only colorspace=bt2020nc and lose primaries.
        return BackendDecision("ffmpeg_fallback", "bt2020 wide gamut needs separate color policy")

    is_10bit = (
        meta.bit_depth > 8
        or "10" in pix_fmt
        or "main 10" in profile
        or "main10" in profile
    )
    if is_10bit:
        if (
            codec == "hevc"
            and (not pix_fmt or pix_fmt in {"p010le", "yuv420p10le"})
            and primaries in _SDR_OK
            and transfer in _SDR_OK
        ):
            return BackendDecision("gpu_p016", "10-bit SDR HEVC (Main10/P010) GPU route")
        return BackendDecision("ffmpeg_fallback", "10-bit non-bt709 / non-HEVC needs ffmpeg path")

    return BackendDecision("gpu_nv12", "8-bit SDR source GPU route")


def route(path: str | Path) -> tuple[VideoMetadata, BackendDecision]:
    """Convenience wrapper for probing and routing decision."""
    meta = probe_video(path)
    return meta, decide_backend(meta)
