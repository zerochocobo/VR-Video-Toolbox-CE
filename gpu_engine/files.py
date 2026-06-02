"""File layer that runs the GPU frame pipeline end to end: decode -> transform -> encode -> mux.

Called by each logic.py as a file-in/file-out API. Transforms are injected as
per-frame callbacks from (Y,UV) CuPy planes to (Y,UV) CuPy planes. Geometry
operators such as v360/crop/stack are implemented in nv12_kernels + v360_lut.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from . import probe, mux, runtime
from . import v360_lut, nv12_kernels
from .fallback import OperationCancelled
from .pynv_io import (
    PyNvSimpleDecoder, PyNvThreadedSerialDecoder,
    GpuNv12AppFrame, GpuP016AppFrame, PyNvEncoderSession,
)

# Per-frame transform signature: (y, uv, ctx) -> (out_y, out_uv)
#   y  : CuPy (H, W)            luma
#   uv : CuPy (H/2, W/2, 2)     interleaved chroma
#   ctx: dict containing source width/height, bit_depth, and related metadata
FrameTransform = Callable[..., tuple]


class CancelToken:
    """Popen-compatible cancellation handle for wiring GPU paths into UI process_callback / proc.kill().

    The UI expects an object with .kill()/.terminate()/.poll(). This class uses
    a flag, checked by process_video each frame, to abort and clean up when
    cancelled.
    """

    def __init__(self):
        self._cancelled = False

    def kill(self):
        self._cancelled = True

    def terminate(self):
        self._cancelled = True

    def poll(self):
        return 0 if self._cancelled else None

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class _Progress:
    """Throttled progress reporter that logs ffmpeg -stats-like progress/FPS/ETA to log_callback.

    A silent per-frame GPU loop makes long videos look stalled, so this reports a
    progress line periodically, once per second by default.
    """

    def __init__(self, total: int, log_callback, min_interval: float = 1.0,
                 window_sec: float = 60.0):
        from collections import deque
        self.total = max(0, int(total))
        self.log = log_callback
        self.min_interval = float(min_interval)
        # Use the rolling rate over the most recent window_sec seconds for fps/ETA
        # instead of a cumulative average. Otherwise, one-time startup costs before
        # the first frame, such as pipeline construction and cudnn autotune, can
        # hold the cumulative average down for a long time, making fps look like it
        # only rises and initial ETA jump to dozens of hours. The rolling window
        # reflects recent speed quickly and smooths Lada's batch-restore stalls
        # every 180 frames.
        self.window_sec = float(window_sec)
        self.t0 = time.perf_counter()
        self._last = 0.0
        self._samples = deque()  # (t, done)

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0, int(sec))
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def update(self, done: int, *, force: bool = False) -> None:
        if not self.log:
            return
        now = time.perf_counter()
        if not force and (now - self._last) < self.min_interval:
            return
        self._last = now
        el = now - self.t0
        # Rolling-window rate: keep the earliest sample still inside window_sec and measure from it to now.
        self._samples.append((now, done))
        while len(self._samples) > 2 and (now - self._samples[0][0]) > self.window_sec:
            self._samples.popleft()
        t_old, done_old = self._samples[0]
        dt = now - t_old
        dn = done - done_old
        if dt >= 1.0 and dn > 0:
            fps = dn / dt
        else:
            fps = done / el if el > 0 else 0.0  # Not enough warmup samples yet; fall back to cumulative speed.
        pct = (100.0 * done / self.total) if self.total else 0.0
        eta = ((self.total - done) / fps) if fps > 0 else 0.0
        self.log(f"[GPU] {done}/{self.total} ({pct:.1f}%) | {fps:.1f} fps | "
                 f"elapsed {self._fmt(el)} | ETA {self._fmt(eta)}")

    def finish(self, done: int) -> None:
        self.update(done, force=True)


def _pack_planes(out_y, out_uv, bit_depth: int):
    """Pack (Y, UV) planes into a contiguous buffer and wrap them as an encoder AppFrame."""
    import cupy as cp

    h, w = out_y.shape
    dtype = cp.uint16 if bit_depth > 8 else cp.uint8
    packed = cp.empty((h * 3 // 2, w), dtype=dtype)
    packed[:h, :] = out_y
    # Interleaved UV plane (h/2, w/2, 2) -> (h/2, w).
    packed[h:, :] = out_uv.reshape(h // 2, w)
    if bit_depth > 8:
        return GpuP016AppFrame(packed, w, h)
    return GpuNv12AppFrame(packed, w, h)


def _match_depth(arr, src_bd: int, dst_bd: int):
    import cupy as cp

    if src_bd == dst_bd:
        return arr
    if dst_bd > 8 and src_bd <= 8:
        return arr.astype(cp.uint16) * cp.uint16(257)
    if dst_bd <= 8 and src_bd > 8:
        return cp.rint(arr.astype(cp.float32) * (255.0 / 65535.0)).astype(cp.uint8)
    return arr


def _make_alpha_mask(w: int, h: int, px: int):
    import cupy as cp

    px = max(0, min(int(px), max(0, min(w, h) // 2)))
    if px <= 0:
        return cp.ones((h, w), dtype=cp.float32)
    xs = cp.arange(w, dtype=cp.float32)
    ys = cp.arange(h, dtype=cp.float32)
    ax = cp.minimum(cp.minimum(xs, w - 1 - xs), px) / float(px)
    ay = cp.minimum(cp.minimum(ys, h - 1 - ys), px) / float(px)
    return (ay[:, None] * ax[None, :]).astype(cp.float32)


def _default_bitrate_bps(meta: probe.VideoMetadata, keep_original: bool) -> int:
    """Encoding target bitrate in bits/s, preferring source bitrate and estimating from resolution when missing."""
    if meta.bitrate_bps > 0:
        return int(meta.bitrate_bps if keep_original else meta.bitrate_bps)
    px = max(1, meta.width * meta.height)
    # Heuristic: about 0.08 bit/px/frame at fps.
    return int(px * meta.source_fps * 0.08)


def _cfg(key: str, default):
    """Read app_config and fall back to the default on failure."""
    try:
        from utils import app_config
        v = app_config.get(key, default)
        return default if v is None else v
    except Exception:
        return default


def _quality_bitrate_bps(out_w: int, out_h: int, fps: float) -> int:
    """Fallback for unknown source bitrate: estimate VBR target bitrate from output resolution at about 0.07 bit/px/frame."""
    px = max(1, int(out_w) * int(out_h))
    return int(px * max(1.0, fps) * 0.07)


def _resolve_bitrate(out_w: int, out_h: int, fps: float,
                     bitrate_bps: int | None, source_bitrate_bps: int = 0) -> int:
    """Target bitrate strategy:
      1. Use the caller's explicit value when provided.
      2. Otherwise use source bitrate times gpu_bitrate_multiplier, default 2.0,
         keeping intermediate/converted file sizes controlled while staying
         slightly above source quality, per user preference.
      3. If source bitrate is unknown, estimate from output resolution.
    """
    if bitrate_bps and bitrate_bps > 0:
        return int(bitrate_bps)
    mult = float(_cfg("gpu_bitrate_multiplier", 2.0) or 2.0)
    if source_bitrate_bps and source_bitrate_bps > 0:
        return int(source_bitrate_bps * mult)
    return _quality_bitrate_bps(out_w, out_h, fps)


def _encoder_kwargs(meta: probe.VideoMetadata, bitrate_bps: int, *,
                    maxrate_multiplier: float = 1.0) -> dict:
    """Always use capped VBR to avoid uncontrolled constqp output sizes.

    Note the NVENC preset semantics: P1 is fastest/lowest quality and P7 is
    slowest/highest quality, which is counterintuitive. Default to P7. The
    frontend controls this through gpu_encode_preset, typically P4-P7 tradeoffs.
    """
    preset = str(_cfg("gpu_encode_preset", "P7") or "P7").upper()
    if preset not in {f"P{i}" for i in range(1, 8)}:
        preset = "P7"
    maxrate = max(int(bitrate_bps), int(bitrate_bps * max(1.0, float(maxrate_multiplier or 1.0))))
    return {
        "fps": f"{meta.source_fps:.6f}",
        "gop": "30",
        "bf": "0",
        "tuning_info": "high_quality",
        "preset": preset,
        "rc": "vbr",
        "bitrate": str(int(bitrate_bps)),
        "maxbitrate": str(maxrate),
    }


def _log_encoder_settings(label: str, out_w: int, out_h: int, bit_depth: int,
                          kwargs: dict, log_callback=None) -> None:
    if not log_callback:
        return
    log_callback(
        f"[gpu-encoder] {label}: out={int(out_w)}x{int(out_h)} {int(bit_depth)}bit "
        f"preset={kwargs.get('preset')} rc={kwargs.get('rc')} "
        f"bitrate={int(kwargs.get('bitrate', 0)) // 1000}kbps "
        f"maxbitrate={int(kwargs.get('maxbitrate', 0)) // 1000}kbps "
        f"gop={kwargs.get('gop')} bf={kwargs.get('bf')}"
    )


from collections import deque as _deque


def _media_temp_path(dst: str | Path, label: str, suffix: str = ".raw.hevc") -> Path:
    """Create a unique media temp path next to the intended output file."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(label or "tmp"))
    return dst.with_name(f"{dst.stem}.{safe_label}.{os.getpid()}.{time.time_ns()}{suffix}")


class _EncodeSink:
    """Lifetime-safe encode sink that synchronizes before encode and delays input-buffer release afterward.

    NVENC Encode() returns before it has actually finished reading the input GPU
    buffer, as verified in PTMediaServer. Synchronizing CUDA after Encode() does
    not fix that; immediately reusing or freeing the buffer can corrupt output
    with green flashing blocks on 4K/8K video. Therefore:
      - synchronize the device before encode so NVENC reads only after remap/pack kernels finish;
      - push input buffers into a ring after encode, retaining the most recent 4 frames by default, so release is delayed until NVENC has certainly finished reading.
    """

    def __init__(self, enc, fobj, ring: int = 4):
        self.enc = enc
        self.f = fobj
        self.pending = _deque(maxlen=max(2, int(ring)))
        self.count = 0

    def feed(self, app, *, force_idr: bool = False) -> None:
        import cupy as cp
        # Ensure kernels writing the packed buffer finish before NVENC reads it.
        cp.cuda.Device().synchronize()
        data = self.enc.encode(app, force_idr=force_idr)
        if data:
            self.f.write(data)
        # Hold a reference so CuPy's memory pool cannot reuse this buffer before NVENC finishes reading it.
        self.pending.append(app)
        self.count += 1

    def flush(self) -> None:
        tail = self.enc.flush()
        if tail:
            self.f.write(tail)
        self.pending.clear()


def process_video(
    src: str | Path,
    dst: str | Path,
    transform: FrameTransform,
    *,
    out_size: tuple[int, int] | None = None,
    bit_depth: int | None = None,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    log_callback=None,
    raw_only: bool = False,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """Decode src, transform each frame, encode with NVENC, and mux source audio into dst.

    out_size : (out_w, out_h); None means source dimensions.
    raw_only : write only raw .hevc without muxing, for benchmarks/debugging; returns the raw stream path.
    cancel_token : CancelToken checked every frame; cancellation aborts and cleans up.
    """
    import cupy as cp

    src = Path(src)
    dst = Path(dst)
    meta = probe.probe_video(src)
    bd = bit_depth if bit_depth is not None else (10 if meta.bit_depth > 8 else 8)

    # Use ThreadedDecoder for sequential processing; throughput is far higher than SimpleDecoder random access.
    dec = PyNvThreadedSerialDecoder(src, bit_depth=bd)
    info = dec.info
    out_w, out_h = out_size if out_size else (info.width, info.height)

    bitrate_bps = _resolve_bitrate(out_w, out_h, meta.source_fps, bitrate_bps, meta.bitrate_bps)
    enc_kwargs = _encoder_kwargs(meta, bitrate_bps)
    _log_encoder_settings("process video", out_w, out_h, bd, enc_kwargs, log_callback)
    enc = PyNvEncoderSession(
        out_w, out_h, bit_depth=bd, codec="hevc",
        **enc_kwargs,
    )

    raw = _media_temp_path(dst, "raw")
    ctx = {"width": info.width, "height": info.height, "bit_depth": bd,
           "out_w": out_w, "out_h": out_h}
    n = 0
    total = len(dec)
    cancelled = False
    prog = _Progress(total, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(total):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True
                    break
                frame = dec.frame_at(i)
                # ThreadedDecoder writes frames on an internal NVDEC stream that CuPy cannot see.
                # Synchronize the device before reading planes to ensure decoding has completed.
                cp.cuda.Device().synchronize()
                y, uv = frame.y_uv_cupy()
                out_y, out_uv = transform(y, uv, ctx)
                app = _pack_planes(out_y, out_uv, bd)
                sink.feed(app, force_idr=(i == 0))
                n += 1
                prog.update(n)
            if not cancelled:
                prog.finish(n)
                sink.flush()
    finally:
        dec.stop()
        runtime.free_memory_pool()

    if cancelled:
        try:
            raw.unlink()
        except OSError:
            pass
        if log_callback:
            log_callback("[gpu] cancelled by user")
        raise OperationCancelled("cancelled by user")

    if log_callback:
        log_callback(f"[gpu] encoded {n} frames {out_w}x{out_h} {bd}bit -> raw {raw.stat().st_size} bytes")

    if raw_only:
        return raw

    mux.mux_hevc_with_audio(
        raw, dst, fps=meta.source_fps, color=meta.color,
        audio_source=str(src) if keep_audio else None,
    )
    try:
        raw.unlink()
    except OSError:
        pass
    return dst


# ---------------------------------------------------------------------------
# v360 projection for tool_v360_trans and one_click fisheye stages.
# ---------------------------------------------------------------------------

def _make_v360_transform(kind: str, dual_screen: bool, fov: float = 180.0) -> FrameTransform:
    """Build a per-frame hequirect<->fisheye transform.

    kind in {heq2fisheye, fisheye2heq}. When dual_screen=True, process SBS
    eye-by-eye and then hstack.
    """
    def _remap_full(y, uv):
        h, w = y.shape
        ch, cw = uv.shape[0], uv.shape[1]
        lut_y = v360_lut.make_lut(kind, w, h, fov)
        lut_c = v360_lut.make_lut(kind, cw, ch, fov)
        out_y = nv12_kernels.remap_y(y, lut_y, w, h)
        out_uv = nv12_kernels.remap_uv(uv, lut_c, cw, ch)
        return out_y, out_uv

    def _transform(y, uv, ctx):
        if not dual_screen:
            return _remap_full(y, uv)
        h, w = y.shape
        ch, cw = uv.shape[0], uv.shape[1]
        hw = w // 2
        chw = cw // 2
        # Luma per eye.
        ly = nv12_kernels.crop_plane(y, 0, 0, hw, h)
        ry = nv12_kernels.crop_plane(y, hw, 0, hw, h)
        lut_y = v360_lut.make_lut(kind, hw, h, fov)
        out_ly = nv12_kernels.remap_y(ly, lut_y, hw, h)
        out_ry = nv12_kernels.remap_y(ry, lut_y, hw, h)
        out_y = nv12_kernels.hstack_planes(out_ly, out_ry)
        # Chroma per eye.
        luv = uv[:, :chw, :]
        ruv = uv[:, chw:, :]
        lut_c = v360_lut.make_lut(kind, chw, ch, fov)
        out_luv = nv12_kernels.remap_uv(luv, lut_c, chw, ch)
        out_ruv = nv12_kernels.remap_uv(ruv, lut_c, chw, ch)
        out_uv = nv12_kernels.hstack_planes(out_luv, out_ruv)
        return out_y, out_uv

    return _transform


def _make_flat_transform(out_w: int, out_h: int, yaw: float, pitch: float,
                         d_fov: float, roll: float = 0.0) -> FrameTransform:
    """Build a per-frame hequirect->flat transform with yaw/pitch/d_fov."""
    def _transform(y, uv, ctx):
        ih, iw = y.shape
        ch, cw = uv.shape[0], uv.shape[1]
        lut_y = v360_lut.make_heq_to_flat_lut(out_w, out_h, iw, ih, yaw, pitch, d_fov, roll)
        out_y = nv12_kernels.remap_y(y, lut_y, out_w, out_h)
        lut_c = v360_lut.make_heq_to_flat_lut(out_w // 2, out_h // 2, cw, ch, yaw, pitch, d_fov, roll)
        out_uv = nv12_kernels.remap_uv(uv, lut_c, out_w // 2, out_h // 2)
        return out_y, out_uv
    return _transform


def vr_to_flat(
    src: str | Path,
    dst: str | Path,
    yaw: float, pitch: float, d_fov: float,
    out_w: int, out_h: int,
    *,
    roll: float = 0.0,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """GPU VR (hequirect) -> flat perspective projection with yaw/pitch/d_fov."""
    transform = _make_flat_transform(out_w, out_h, yaw, pitch, d_fov, roll)
    return process_video(
        src, dst, transform, out_size=(out_w, out_h), cq=cq, bitrate_bps=bitrate_bps,
        keep_audio=keep_audio, log_callback=log_callback, cancel_token=cancel_token,
    )


def vr_projection(
    src: str | Path,
    dst: str | Path,
    mode: str,
    *,
    dual_screen: bool = False,
    fov: float = 180.0,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    log_callback=None,
    raw_only: bool = False,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """GPU VR projection conversion. mode in {heq2fisheye, fisheye2heq}."""
    transform = _make_v360_transform(mode, dual_screen, fov)
    return process_video(
        src, dst, transform, cq=cq, bitrate_bps=bitrate_bps,
        keep_audio=keep_audio, log_callback=log_callback, raw_only=raw_only,
        cancel_token=cancel_token,
    )


# ---------------------------------------------------------------------------
# split / combine for tool_split_combine and area_selection_rect_crop.
# ---------------------------------------------------------------------------

def process_video_multi(
    src: str | Path,
    jobs: list,
    *,
    bit_depth: int | None = None,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    start_sec: float | None = None,
    end_sec: float | None = None,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
):
    """Decode once, apply multiple job transforms to every frame, and encode each job to an independent output.

    jobs: list[dict], each containing {transform, dst(Path), out_size(w,h)}.
    Used to produce multiple files from one decode, such as left/right eye split outputs.
    start_sec/end_sec optionally limit the time range.
    """
    import cupy as cp

    src = Path(src)
    meta = probe.probe_video(src)
    bd = bit_depth if bit_depth is not None else (10 if meta.bit_depth > 8 else 8)

    fps = meta.source_fps or 30.0
    start_idx = int(round(start_sec * fps)) if start_sec else 0
    dec = PyNvThreadedSerialDecoder(src, bit_depth=bd, start_frame=start_idx)
    info = dec.info
    fps = meta.source_fps or info.fps or 30.0
    end_idx = min(int(round(end_sec * fps)) if end_sec else len(dec), len(dec))
    audio_start = start_sec if start_sec else None
    audio_dur = (end_sec - (start_sec or 0.0)) if end_sec else None

    states = []
    for job in jobs:
        ow, oh = job["out_size"]
        br = _resolve_bitrate(ow, oh, fps, bitrate_bps, meta.bitrate_bps)
        enc_kwargs = _encoder_kwargs(meta, br)
        _log_encoder_settings(f"multi output {Path(job['dst']).name}", ow, oh, bd, enc_kwargs, log_callback)
        enc = PyNvEncoderSession(ow, oh, bit_depth=bd, codec="hevc",
                                 **enc_kwargs)
        raw = _media_temp_path(Path(job["dst"]), "raw")
        fobj = open(raw, "wb")
        states.append({"job": job, "enc": enc, "raw": raw,
                       "f": fobj, "sink": _EncodeSink(enc, fobj), "ow": ow, "oh": oh})

    ctx = {"width": info.width, "height": info.height, "bit_depth": bd}
    n = 0
    cancelled = False
    prog = _Progress(end_idx - start_idx, log_callback)
    try:
        for i in range(start_idx, end_idx):
            if cancel_token is not None and cancel_token.cancelled:
                cancelled = True
                break
            frame = dec.frame_at(i)
            cp.cuda.Device().synchronize()
            y, uv = frame.y_uv_cupy()
            for st in states:
                oy, ouv = st["job"]["transform"](y, uv, ctx)
                app = _pack_planes(oy, ouv, bd)
                st["sink"].feed(app, force_idr=(i == start_idx))
            n += 1
            prog.update(n)
        if not cancelled:
            prog.finish(n)
            for st in states:
                st["sink"].flush()
    finally:
        dec.stop()
        for st in states:
            try:
                st["f"].close()
            except Exception:
                pass
        runtime.free_memory_pool()

    if cancelled:
        for st in states:
            try:
                st["raw"].unlink()
            except OSError:
                pass
        if log_callback:
            log_callback("[gpu] cancelled by user")
        raise OperationCancelled("cancelled by user")

    outs = []
    for st in states:
        mux.mux_hevc_with_audio(
            st["raw"], Path(st["job"]["dst"]), fps=meta.source_fps, color=meta.color,
            audio_source=str(src) if keep_audio else None,
            audio_start_sec=audio_start, audio_duration=audio_dur,
        )
        try:
            st["raw"].unlink()
        except OSError:
            pass
        outs.append(Path(st["job"]["dst"]))
    if log_callback:
        log_callback(f"[gpu] split: {n} frames -> {len(outs)} output(s)")
    return outs


def _crop_region(mode: str, w: int, h: int):
    """Return luma crop (x,y,cw,ch). Coordinates are even for chroma compatibility."""
    hw, hh = w // 2, h // 2
    return {
        "left":   (0, 0, hw, h),
        "right":  (hw, 0, hw, h),
        "top":    (0, 0, w, hh),
        "bottom": (0, hh, w, hh),
    }[mode]


def _make_crop_transform(mode: str, to_fisheye: bool):
    """Build a per-frame single-output crop transform, optionally followed by hequirect->fisheye."""
    def _transform(y, uv, ctx):
        h, w = y.shape
        x, yy, cw_, ch_ = _crop_region(mode, w, h)
        cy = y[yy:yy + ch_, x:x + cw_]
        # Halve coordinates for chroma.
        cuv = uv[yy // 2:(yy + ch_) // 2, x // 2:(x + cw_) // 2, :]
        if not to_fisheye:
            return cy, cuv
        lut_y = v360_lut.make_lut("heq2fisheye", cw_, ch_)
        oy = nv12_kernels.remap_y(cy, lut_y, cw_, ch_)
        ccw, cch = cuv.shape[1], cuv.shape[0]
        lut_c = v360_lut.make_lut("heq2fisheye", ccw, cch)
        ouv = nv12_kernels.remap_uv(cuv, lut_c, ccw, cch)
        return oy, ouv
    return _transform


def split_video(
    src: str | Path,
    out_paths: dict,
    *,
    to_fisheye: bool = False,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    start_sec: float | None = None,
    end_sec: float | None = None,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
):
    """GPU VR split. out_paths: {mode: dst}, with mode in {left,right,top,bottom}.

    Passing 1 output produces a single file; passing 2 outputs, left+right or
    top+bottom, produces both from one decode. start_sec/end_sec optionally limit
    the time range.
    """
    meta = probe.probe_video(src)
    w, h = meta.width, meta.height
    jobs = []
    for mode, dst in out_paths.items():
        x, yy, cw_, ch_ = _crop_region(mode, w, h)
        jobs.append({"transform": _make_crop_transform(mode, to_fisheye),
                     "dst": dst, "out_size": (cw_, ch_)})
    return process_video_multi(src, jobs, cq=cq, bitrate_bps=bitrate_bps,
                               keep_audio=keep_audio, start_sec=start_sec, end_sec=end_sec,
                               log_callback=log_callback, cancel_token=cancel_token)


def combine_video(
    src_a: str | Path,
    src_b: str | Path,
    dst: str | Path,
    mode: str,
    *,
    from_fisheye: bool = False,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """GPU combine for two video streams. mode in {left_right, top_bottom}; from_fisheye converts fisheye->hequirect first."""
    import cupy as cp

    src_a = Path(src_a); src_b = Path(src_b); dst = Path(dst)
    meta = probe.probe_video(src_a)
    bd = 10 if meta.bit_depth > 8 else 8
    da = PyNvThreadedSerialDecoder(src_a, bit_depth=bd)
    db = PyNvThreadedSerialDecoder(src_b, bit_depth=bd)
    ia, ib = da.info, db.info
    if mode == "left_right":
        out_w, out_h = ia.width + ib.width, max(ia.height, ib.height)
    elif mode == "top_bottom":
        out_w, out_h = max(ia.width, ib.width), ia.height + ib.height
    else:
        raise ValueError(f"unknown combine mode: {mode}")

    bitrate_bps = _resolve_bitrate(out_w, out_h, meta.source_fps, bitrate_bps, meta.bitrate_bps)
    enc_kwargs = _encoder_kwargs(meta, bitrate_bps)
    _log_encoder_settings("combine video", out_w, out_h, bd, enc_kwargs, log_callback)
    enc = PyNvEncoderSession(out_w, out_h, bit_depth=bd, codec="hevc",
                             **enc_kwargs)
    raw = _media_temp_path(dst, "raw")

    def _maybe_defish(y, uv):
        if not from_fisheye:
            return y, uv
        h, w = y.shape
        ch, cw = uv.shape[0], uv.shape[1]
        oy = nv12_kernels.remap_y(y, v360_lut.make_lut("fisheye2heq", w, h), w, h)
        ouv = nv12_kernels.remap_uv(uv, v360_lut.make_lut("fisheye2heq", cw, ch), cw, ch)
        return oy, ouv

    n = min(len(da), len(db))
    cancelled = False
    done = 0
    prog = _Progress(n, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(n):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True
                    break
                fa = da.frame_at(i); fb = db.frame_at(i)
                cp.cuda.Device().synchronize()
                ya, uva = fa.y_uv_cupy(); yb, uvb = fb.y_uv_cupy()
                ya, uva = _maybe_defish(ya, uva)
                yb, uvb = _maybe_defish(yb, uvb)
                if mode == "left_right":
                    oy = nv12_kernels.hstack_planes(ya, yb)
                    ouv = nv12_kernels.hstack_planes(uva, uvb)
                else:
                    oy = nv12_kernels.vstack_planes(ya, yb)
                    ouv = nv12_kernels.vstack_planes(uva, uvb)
                app = _pack_planes(oy, ouv, bd)
                sink.feed(app, force_idr=(i == 0))
                done += 1
                prog.update(done)
            if not cancelled:
                prog.finish(done)
                sink.flush()
    finally:
        da.stop(); db.stop()
        runtime.free_memory_pool()

    if cancelled:
        try:
            raw.unlink()
        except OSError:
            pass
        raise OperationCancelled("cancelled by user")

    mux.mux_hevc_with_audio(raw, dst, fps=meta.source_fps, color=meta.color,
                            audio_source=str(src_a) if keep_audio else None)
    try:
        raw.unlink()
    except OSError:
        pass
    return dst


def paste_segments_gpu(
    base_src: str | Path,
    dst: str | Path,
    segments: list,
    *,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    feather_px: int | None = None,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """GPU paste restored cropped segments back onto ``base_src``.

    Segment objects are duck-typed and must provide:
    ``path, base_frame_start, base_frame_end, x, y, w, h``.
    ``base_frame_end`` is exclusive.
    """
    import cupy as cp

    base_src = Path(base_src)
    dst = Path(dst)
    if not segments:
        raise ValueError("paste_segments_gpu requires at least one segment")

    meta, decision = probe.route(base_src)
    if not decision.is_gpu:
        raise RuntimeError(f"base video is not GPU-paste eligible: {decision.reason}")
    bd = 10 if meta.bit_depth > 8 else 8
    base_dec = PyNvThreadedSerialDecoder(base_src, bit_depth=bd)
    info = base_dec.info
    fps = meta.source_fps or info.fps or 30.0
    total = len(base_dec)
    feather = int(feather_px if feather_px is not None else _cfg("pre_extract_feather_px", 12) or 12)

    def _open_state(seg):
        seg_meta = probe.probe_video(seg.path)
        seg_bd = 10 if seg_meta.bit_depth > 8 else 8
        dec = PyNvThreadedSerialDecoder(seg.path, bit_depth=seg_bd)
        if int(seg.x) < 0 or int(seg.y) < 0 or int(seg.x + seg.w) > info.width or int(seg.y + seg.h) > info.height:
            dec.stop()
            raise ValueError(f"segment {seg.seg_id} rect out of bounds: {(seg.x, seg.y, seg.w, seg.h)} for {info.width}x{info.height}")
        if int(seg.w) <= 0 or int(seg.h) <= 0 or int(seg.w) % 2 or int(seg.h) % 2:
            dec.stop()
            raise ValueError(f"segment {seg.seg_id} rect must be positive even dimensions: {(seg.w, seg.h)}")
        chroma_feather = 0 if feather <= 0 else max(1, feather // 2)
        return {
            "seg": seg,
            "dec": dec,
            "bd": seg_bd,
            "frames": len(dec),
            "alpha_y": _make_alpha_mask(int(seg.w), int(seg.h), feather),
            "alpha_c": _make_alpha_mask(int(seg.w) // 2, int(seg.h) // 2, chroma_feather),
        }

    segs = sorted(segments, key=lambda s: (int(s.base_frame_start), int(s.base_frame_end), int(s.seg_id)))
    bitrate_bps = _resolve_bitrate(info.width, info.height, fps, bitrate_bps, meta.bitrate_bps)
    enc_kwargs = _encoder_kwargs(meta, bitrate_bps)
    _log_encoder_settings("pre-extract paste", info.width, info.height, bd, enc_kwargs, log_callback)
    enc = PyNvEncoderSession(
        info.width, info.height, bit_depth=bd, codec="hevc",
        **enc_kwargs,
    )
    raw = _media_temp_path(dst, "paste")

    next_idx = 0
    active: list[dict] = []
    done = 0
    cancelled = False
    prog = _Progress(total, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(total):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True
                    break

                while next_idx < len(segs) and int(segs[next_idx].base_frame_start) <= i:
                    seg = segs[next_idx]
                    next_idx += 1
                    if int(seg.base_frame_end) <= i:
                        continue
                    state = _open_state(seg)
                    active.append(state)
                    if log_callback:
                        log_callback(f"[pre-extract] paste segment {seg.seg_id}: frames {seg.base_frame_start}-{seg.base_frame_end}, rect {seg.x},{seg.y},{seg.w}x{seg.h}")

                still_active = []
                for state in active:
                    if i >= int(state["seg"].base_frame_end):
                        state["dec"].stop()
                    else:
                        still_active.append(state)
                active = still_active

                frame = base_dec.frame_at(i)
                cp.cuda.Device().synchronize()
                base_y_src, base_uv_src = frame.y_uv_cupy()
                # Intentionally edit this frame view in-place. _pack_planes() copies
                # the final planes into an independent packed buffer before NVENC
                # reads them, so decoder ring reuse on the next frame is safe.
                y = cp.ascontiguousarray(base_y_src)
                uv = cp.ascontiguousarray(base_uv_src)

                for state in list(active):
                    seg = state["seg"]
                    seg_idx = i - int(seg.base_frame_start)
                    if seg_idx < 0:
                        continue
                    if seg_idx >= int(state["frames"]):
                        state["dec"].stop()
                        active.remove(state)
                        continue
                    sframe = state["dec"].frame_at(seg_idx)
                    sy, suv = sframe.y_uv_cupy()
                    sy = _match_depth(sy, int(state["bd"]), bd)
                    suv = _match_depth(suv, int(state["bd"]), bd)
                    if sy.shape[0] != int(seg.h) or sy.shape[1] != int(seg.w):
                        raise RuntimeError(
                            f"segment {seg.seg_id} size mismatch: decoded {sy.shape[1]}x{sy.shape[0]}, expected {seg.w}x{seg.h}"
                        )
                    ry = y[int(seg.y):int(seg.y + seg.h), int(seg.x):int(seg.x + seg.w)]
                    a_y = state["alpha_y"]
                    ry[:] = cp.rint(a_y * sy.astype(cp.float32) + (1.0 - a_y) * ry.astype(cp.float32)).astype(y.dtype)

                    cy = int(seg.y) // 2
                    cx = int(seg.x) // 2
                    ch = int(seg.h) // 2
                    cw = int(seg.w) // 2
                    ruv = uv[cy:cy + ch, cx:cx + cw, :]
                    a_c = state["alpha_c"][..., None]
                    ruv[:] = cp.rint(a_c * suv.astype(cp.float32) + (1.0 - a_c) * ruv.astype(cp.float32)).astype(uv.dtype)

                app = _pack_planes(y, uv, bd)
                sink.feed(app, force_idr=(i == 0))
                done += 1
                prog.update(done)
            if not cancelled:
                prog.finish(done)
                sink.flush()
    finally:
        for state in active:
            try:
                state["dec"].stop()
            except Exception:
                pass
        base_dec.stop()
        runtime.free_memory_pool()

    if cancelled:
        try:
            raw.unlink()
        except OSError:
            pass
        if log_callback:
            log_callback("[pre-extract] GPU paste cancelled by user")
        raise OperationCancelled("cancelled by user")

    mux.mux_hevc_with_audio(
        raw,
        dst,
        fps=fps,
        color=meta.color,
        audio_source=str(base_src) if keep_audio else None,
        log_callback=log_callback,
    )
    try:
        raw.unlink()
    except OSError:
        pass
    return dst


def paste_fisheye_eye_rects_to_sbs_gpu(
    base_src: str | Path,
    dst: str | Path,
    segments: list,
    *,
    fov: float = 180.0,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    feather_px: int | None = None,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """Patch restored fisheye-eye rect clips into an SBS hequirect interval.

    ``base_src`` stays in hequirect SBS form on disk. For each frame this function
    converts each eye to fisheye in GPU memory, pastes restored fisheye rects, then
    converts the patched eyes back to hequirect before NVENC output. Segment ``x``
    coordinates are SBS fisheye coordinates: left eye starts at 0, right eye starts
    at ``base_width / 2``.
    """
    import cupy as cp

    base_src = Path(base_src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        raise ValueError("paste_fisheye_eye_rects_to_sbs_gpu requires at least one segment")

    meta, decision = probe.route(base_src)
    if not decision.is_gpu:
        raise RuntimeError(f"base video is not GPU fisheye-paste eligible: {decision.reason}")
    bd = 10 if meta.bit_depth > 8 else 8
    base_dec = PyNvThreadedSerialDecoder(base_src, bit_depth=bd)
    info = base_dec.info
    if int(info.width) % 4 or int(info.height) % 2:
        base_dec.stop()
        raise ValueError(f"SBS fisheye paste requires width divisible by 4 and even height: {info.width}x{info.height}")

    fps = meta.source_fps or info.fps or 30.0
    total = len(base_dec)
    eye_w = int(info.width // 2)
    eye_h = int(info.height)
    eye_cw = eye_w // 2
    eye_ch = eye_h // 2
    feather = int(feather_px if feather_px is not None else _cfg("pre_extract_feather_px", 12) or 12)

    if log_callback:
        log_callback(
            f"[source-scan] Stage 3: in-memory fisheye patch "
            f"{len(segments)} rect(s), base={info.width}x{info.height}, eye={eye_w}x{eye_h}"
        )

    lut_h2f_y = v360_lut.make_lut("heq2fisheye", eye_w, eye_h, fov)
    lut_h2f_c = v360_lut.make_lut("heq2fisheye", eye_cw, eye_ch, fov)
    lut_f2h_y = v360_lut.make_lut("fisheye2heq", eye_w, eye_h, fov)
    lut_f2h_c = v360_lut.make_lut("fisheye2heq", eye_cw, eye_ch, fov)

    def _open_state(seg):
        seg_meta = probe.probe_video(seg.path)
        seg_bd = 10 if seg_meta.bit_depth > 8 else 8
        dec = PyNvThreadedSerialDecoder(seg.path, bit_depth=seg_bd)
        sx = int(seg.x)
        sy = int(seg.y)
        sw = int(seg.w)
        sh = int(seg.h)
        if sx < 0 or sy < 0 or sx + sw > info.width or sy + sh > eye_h:
            dec.stop()
            raise ValueError(
                f"segment {seg.seg_id} rect out of SBS bounds: {(sx, sy, sw, sh)} for {info.width}x{eye_h}"
            )
        if sx < eye_w:
            side = "left"
            local_x = sx
            if sx + sw > eye_w:
                dec.stop()
                raise ValueError(f"segment {seg.seg_id} crosses SBS eye boundary: {(sx, sy, sw, sh)}")
        else:
            side = "right"
            local_x = sx - eye_w
            if local_x + sw > eye_w:
                dec.stop()
                raise ValueError(f"segment {seg.seg_id} rect out of right-eye bounds: {(sx, sy, sw, sh)}")
        if sw <= 0 or sh <= 0 or sw % 2 or sh % 2 or local_x % 2 or sy % 2:
            dec.stop()
            raise ValueError(f"segment {seg.seg_id} rect must use positive even coordinates/dimensions: {(sx, sy, sw, sh)}")
        chroma_feather = 0 if feather <= 0 else max(1, feather // 2)
        return {
            "seg": seg,
            "dec": dec,
            "bd": seg_bd,
            "frames": len(dec),
            "side": side,
            "x": local_x,
            "y": sy,
            "w": sw,
            "h": sh,
            "alpha_y": _make_alpha_mask(sw, sh, feather),
            "alpha_c": _make_alpha_mask(sw // 2, sh // 2, chroma_feather),
        }

    segs = sorted(segments, key=lambda s: (int(s.base_frame_start), int(s.base_frame_end), int(s.seg_id)))
    bitrate_bps = _resolve_bitrate(info.width, info.height, fps, bitrate_bps, meta.bitrate_bps)
    enc_kwargs = _encoder_kwargs(meta, bitrate_bps)
    _log_encoder_settings("fisheye eye rect paste", info.width, info.height, bd, enc_kwargs, log_callback)
    enc = PyNvEncoderSession(
        info.width, info.height, bit_depth=bd, codec="hevc",
        **enc_kwargs,
    )
    raw = _media_temp_path(dst, "fishpatch")

    def _paste_into_eye(state, dst_y, dst_uv, frame_idx: int) -> bool:
        seg = state["seg"]
        seg_idx = frame_idx - int(seg.base_frame_start)
        if seg_idx < 0:
            return True
        if seg_idx >= int(state["frames"]):
            state["dec"].stop()
            return False

        sframe = state["dec"].frame_at(seg_idx)
        sy, suv = sframe.y_uv_cupy()
        sy = _match_depth(sy, int(state["bd"]), bd)
        suv = _match_depth(suv, int(state["bd"]), bd)
        if sy.shape[0] != int(state["h"]) or sy.shape[1] != int(state["w"]):
            raise RuntimeError(
                f"segment {seg.seg_id} size mismatch: decoded {sy.shape[1]}x{sy.shape[0]}, "
                f"expected {state['w']}x{state['h']}"
            )

        x = int(state["x"])
        y = int(state["y"])
        w = int(state["w"])
        h = int(state["h"])
        target_y = dst_y[y:y + h, x:x + w]
        a_y = state["alpha_y"]
        target_y[:] = cp.rint(
            a_y * sy.astype(cp.float32) + (1.0 - a_y) * target_y.astype(cp.float32)
        ).astype(dst_y.dtype)

        cx = x // 2
        cy = y // 2
        cw = w // 2
        ch = h // 2
        target_uv = dst_uv[cy:cy + ch, cx:cx + cw, :]
        a_c = state["alpha_c"][..., None]
        target_uv[:] = cp.rint(
            a_c * suv.astype(cp.float32) + (1.0 - a_c) * target_uv.astype(cp.float32)
        ).astype(dst_uv.dtype)
        return True

    next_idx = 0
    active: list[dict] = []
    done = 0
    cancelled = False
    prog = _Progress(total, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(total):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True
                    break

                while next_idx < len(segs) and int(segs[next_idx].base_frame_start) <= i:
                    seg = segs[next_idx]
                    next_idx += 1
                    if int(seg.base_frame_end) <= i:
                        continue
                    state = _open_state(seg)
                    active.append(state)
                    if log_callback:
                        eye_label = "L" if state["side"] == "left" else "R"
                        log_callback(
                            f"[pre-extract] fisheye patch segment {seg.seg_id} {eye_label}: "
                            f"frames {seg.base_frame_start}-{seg.base_frame_end}, "
                            f"rect {state['x']},{state['y']},{state['w']}x{state['h']}"
                        )

                still_active = []
                for state in active:
                    if i >= int(state["seg"].base_frame_end):
                        state["dec"].stop()
                    else:
                        still_active.append(state)
                active = still_active

                frame = base_dec.frame_at(i)
                cp.cuda.Device().synchronize()
                base_y_src, base_uv_src = frame.y_uv_cupy()

                if not active:
                    app = _pack_planes(base_y_src, base_uv_src, bd)
                    sink.feed(app, force_idr=(i == 0))
                    done += 1
                    prog.update(done)
                    continue

                left_active = [state for state in active if state["side"] == "left"]
                right_active = [state for state in active if state["side"] == "right"]
                ly_src = base_y_src[:, :eye_w]
                ry_src = base_y_src[:, eye_w:eye_w * 2]
                luv_src = base_uv_src[:, :eye_cw, :]
                ruv_src = base_uv_src[:, eye_cw:eye_cw * 2, :]

                if left_active:
                    ly_fish = nv12_kernels.remap_y(ly_src, lut_h2f_y, eye_w, eye_h)
                    luv_fish = nv12_kernels.remap_uv(luv_src, lut_h2f_c, eye_cw, eye_ch)
                    left_patched = False
                    for state in list(left_active):
                        keep = _paste_into_eye(state, ly_fish, luv_fish, i)
                        if keep:
                            left_patched = True
                        elif state in active:
                            active.remove(state)
                    if left_patched:
                        ly_out = nv12_kernels.remap_y(ly_fish, lut_f2h_y, eye_w, eye_h)
                        luv_out = nv12_kernels.remap_uv(luv_fish, lut_f2h_c, eye_cw, eye_ch)
                    else:
                        ly_out = ly_src
                        luv_out = luv_src
                else:
                    ly_out = ly_src
                    luv_out = luv_src

                if right_active:
                    ry_fish = nv12_kernels.remap_y(ry_src, lut_h2f_y, eye_w, eye_h)
                    ruv_fish = nv12_kernels.remap_uv(ruv_src, lut_h2f_c, eye_cw, eye_ch)
                    right_patched = False
                    for state in list(right_active):
                        keep = _paste_into_eye(state, ry_fish, ruv_fish, i)
                        if keep:
                            right_patched = True
                        elif state in active:
                            active.remove(state)
                    if right_patched:
                        ry_out = nv12_kernels.remap_y(ry_fish, lut_f2h_y, eye_w, eye_h)
                        ruv_out = nv12_kernels.remap_uv(ruv_fish, lut_f2h_c, eye_cw, eye_ch)
                    else:
                        ry_out = ry_src
                        ruv_out = ruv_src
                else:
                    ry_out = ry_src
                    ruv_out = ruv_src

                out_y = nv12_kernels.hstack_planes(ly_out, ry_out)
                out_uv = nv12_kernels.hstack_planes(luv_out, ruv_out)

                app = _pack_planes(out_y, out_uv, bd)
                sink.feed(app, force_idr=(i == 0))
                done += 1
                prog.update(done)
            if not cancelled:
                prog.finish(done)
                sink.flush()
    finally:
        for state in active:
            try:
                state["dec"].stop()
            except Exception:
                pass
        base_dec.stop()
        runtime.free_memory_pool()

    if cancelled:
        try:
            raw.unlink()
        except OSError:
            pass
        if log_callback:
            log_callback("[pre-extract] GPU fisheye patch cancelled by user")
        raise OperationCancelled("cancelled by user")

    mux.mux_hevc_with_audio(
        raw,
        dst,
        fps=fps,
        color=meta.color,
        audio_source=str(base_src) if keep_audio else None,
        log_callback=log_callback,
    )
    try:
        raw.unlink()
    except OSError:
        pass
    return dst


def replace_timeline_segments_gpu(
    source_src: str | Path,
    dst: str | Path,
    timeline: list,
    *,
    audio_source: str | Path | None = None,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """Encode a full timeline by replacing mosaic intervals with restored clips.

    This avoids ffmpeg concat/copy timestamp edge cases. The source video is
    decoded once in order; frames inside ``kind == "mosaic"`` entries are taken
    from the restored entry path, while all other frames pass through from source.
    """
    import cupy as cp

    source_src = Path(source_src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta, decision = probe.route(source_src)
    if not decision.is_gpu:
        raise RuntimeError(f"source video is not GPU timeline eligible: {decision.reason}")
    bd = 10 if meta.bit_depth > 8 else 8
    src_dec = PyNvThreadedSerialDecoder(source_src, bit_depth=bd)
    info = src_dec.info
    fps = meta.source_fps or info.fps or 30.0
    total = len(src_dec)

    specs = []
    for entry in sorted(timeline, key=lambda e: (float(e.start_s), float(e.end_s), str(e.kind))):
        if getattr(entry, "kind", "") != "mosaic":
            continue
        path = Path(entry.path)
        if not path.exists():
            raise FileNotFoundError(f"restored mosaic segment missing: {path}")
        start = max(0, int(round(float(entry.start_s) * fps)))
        end = min(total, int(round(float(entry.end_s) * fps)))
        if end <= start:
            continue
        specs.append({"entry": entry, "path": path, "start": start, "end": end})

    if log_callback:
        covered = sum(max(0, item["end"] - item["start"]) for item in specs)
        log_callback(
            f"[source-scan] Stage 4 GPU timeline merge: source={source_src}, "
            f"entries={len(timeline)}, mosaic={len(specs)}, frames={total}, replaced_frames={covered}"
        )
        for idx, item in enumerate(specs[:30]):
            entry = item["entry"]
            log_callback(
                f"[source-scan] gpu merge mosaic {idx}: "
                f"{float(entry.start_s):.3f}-{float(entry.end_s):.3f}s "
                f"frames={item['start']}-{item['end']} path={item['path']}"
            )
        if len(specs) > 30:
            log_callback(f"[source-scan] ... {len(specs) - 30} more GPU merge mosaic entries")

    bitrate_bps = _resolve_bitrate(info.width, info.height, fps, bitrate_bps, meta.bitrate_bps)
    enc_kwargs = _encoder_kwargs(meta, bitrate_bps)
    _log_encoder_settings("source-scan timeline merge", info.width, info.height, bd, enc_kwargs, log_callback)
    enc = PyNvEncoderSession(
        info.width, info.height, bit_depth=bd, codec="hevc",
        **enc_kwargs,
    )
    raw = _media_temp_path(dst, "timeline")

    def _open_state(item):
        seg_meta, seg_decision = probe.route(item["path"])
        if not seg_decision.is_gpu:
            raise RuntimeError(f"restored segment is not GPU timeline eligible: {seg_decision.reason}")
        seg_bd = 10 if seg_meta.bit_depth > 8 else 8
        dec = PyNvThreadedSerialDecoder(item["path"], bit_depth=seg_bd)
        seg_info = dec.info
        if int(seg_info.width) != int(info.width) or int(seg_info.height) != int(info.height):
            dec.stop()
            raise ValueError(
                f"restored segment size mismatch: {item['path']} is {seg_info.width}x{seg_info.height}, "
                f"expected {info.width}x{info.height}"
            )
        return {**item, "dec": dec, "bd": seg_bd, "frames": len(dec), "warned_short": False}

    next_idx = 0
    active: list[dict] = []
    done = 0
    cancelled = False
    prog = _Progress(total, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(total):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True
                    break

                while next_idx < len(specs) and int(specs[next_idx]["start"]) <= i:
                    item = specs[next_idx]
                    next_idx += 1
                    if int(item["end"]) <= i:
                        continue
                    state = _open_state(item)
                    active.append(state)
                    if log_callback:
                        log_callback(
                            f"[source-scan] gpu merge open {Path(state['path']).name}: "
                            f"frames {state['start']}-{state['end']}, restored_frames={state['frames']}"
                        )

                still_active = []
                for state in active:
                    if i >= int(state["end"]):
                        state["dec"].stop()
                    else:
                        still_active.append(state)
                active = still_active

                src_frame = src_dec.frame_at(i)
                cp.cuda.Device().synchronize()
                out_y, out_uv = src_frame.y_uv_cupy()

                if active:
                    # Timeline intervals should not overlap; if they do, the latest
                    # active restored segment wins for this frame.
                    state = active[-1]
                    local_idx = i - int(state["start"])
                    if 0 <= local_idx < int(state["frames"]):
                        rframe = state["dec"].frame_at(local_idx)
                        ry, ruv = rframe.y_uv_cupy()
                        ry = _match_depth(ry, int(state["bd"]), bd)
                        ruv = _match_depth(ruv, int(state["bd"]), bd)
                        if ry.shape[0] != int(info.height) or ry.shape[1] != int(info.width):
                            raise RuntimeError(
                                f"restored segment frame size mismatch: decoded {ry.shape[1]}x{ry.shape[0]}, "
                                f"expected {info.width}x{info.height}"
                            )
                        out_y, out_uv = ry, ruv
                    elif log_callback and not state["warned_short"]:
                        state["warned_short"] = True
                        log_callback(
                            f"[source-scan] restored segment shorter than timeline; "
                            f"using source frames for remainder: {state['path']}"
                        )

                app = _pack_planes(out_y, out_uv, bd)
                sink.feed(app, force_idr=(i == 0))
                done += 1
                prog.update(done)
            if not cancelled:
                prog.finish(done)
                sink.flush()
    finally:
        for state in active:
            try:
                state["dec"].stop()
            except Exception:
                pass
        src_dec.stop()
        runtime.free_memory_pool()

    if cancelled:
        try:
            raw.unlink()
        except OSError:
            pass
        if log_callback:
            log_callback("[source-scan] GPU timeline merge cancelled by user")
        raise OperationCancelled("cancelled by user")

    mux.mux_hevc_with_audio(
        raw,
        dst,
        fps=fps,
        color=meta.color,
        audio_source=str(audio_source) if audio_source is not None else None,
        log_callback=log_callback,
    )
    try:
        raw.unlink()
    except OSError:
        pass
    return dst


def extract_clip(
    src: str | Path,
    dst: str | Path,
    *,
    crop_mode: str | None = None,
    to_fisheye: bool = False,
    fov: float = 180.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """GPU crop + optional hequirect->fisheye + time-range extraction, all completed in one decode.

    crop_mode in {None, left, right, top, bottom}. When to_fisheye is true, the
    crop is followed by heq->fisheye so a single eye reaches fisheye in one step.
    start/end_sec define the time range in seconds.
    """
    import cupy as cp

    src = Path(src); dst = Path(dst)
    meta = probe.probe_video(src)
    bd = 10 if meta.bit_depth > 8 else 8
    fps = meta.source_fps or 30.0
    start_idx = int(round(start_sec * fps)) if start_sec else 0
    dec = PyNvThreadedSerialDecoder(src, bit_depth=bd, start_frame=start_idx)
    info = dec.info
    fps = meta.source_fps or info.fps or 30.0
    end_idx = int(round(end_sec * fps)) if end_sec else len(dec)
    end_idx = min(end_idx, len(dec))

    w, h = info.width, info.height
    if crop_mode:
        x, yy, cw_, ch_ = _crop_region(crop_mode, w, h)
        out_w, out_h = cw_, ch_
    else:
        x = yy = 0; cw_, ch_ = w, h; out_w, out_h = w, h
    if log_callback:
        log_callback(
            f"[gpu] extract clip setup: src={src}, dst={dst}, "
            f"sec={start_sec}-{end_sec}, frames={start_idx}-{end_idx}, "
            f"crop_mode={crop_mode}, to_fisheye={to_fisheye}, out={out_w}x{out_h}"
        )

    # Build fisheye LUTs once and reuse them, using cropped-eye size and half-resolution chroma.
    lut_y = v360_lut.make_lut("heq2fisheye", out_w, out_h, fov) if to_fisheye else None
    lut_c = v360_lut.make_lut("heq2fisheye", out_w // 2, out_h // 2, fov) if to_fisheye else None

    bitrate_bps = _resolve_bitrate(out_w, out_h, fps, bitrate_bps, meta.bitrate_bps)
    enc_kwargs = _encoder_kwargs(meta, bitrate_bps)
    _log_encoder_settings("extract clip", out_w, out_h, bd, enc_kwargs, log_callback)
    enc = PyNvEncoderSession(out_w, out_h, bit_depth=bd, codec="hevc",
                             **enc_kwargs)
    raw = _media_temp_path(dst, "raw")
    cancelled = False
    written = 0
    prog = _Progress(end_idx - start_idx, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(start_idx, end_idx):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True
                    break
                frame = dec.frame_at(i)
                cp.cuda.Device().synchronize()
                y, uv = frame.y_uv_cupy()
                if crop_mode:
                    y = y[yy:yy + ch_, x:x + cw_]
                    uv = uv[yy // 2:(yy + ch_) // 2, x // 2:(x + cw_) // 2, :]
                if to_fisheye:
                    y = nv12_kernels.remap_y(y, lut_y, out_w, out_h)
                    uv = nv12_kernels.remap_uv(uv, lut_c, out_w // 2, out_h // 2)
                app = _pack_planes(y, uv, bd)
                sink.feed(app, force_idr=(written == 0))
                written += 1
                prog.update(written)
            if not cancelled:
                prog.finish(written)
                sink.flush()
    finally:
        dec.stop()
        runtime.free_memory_pool()

    if cancelled:
        try:
            raw.unlink()
        except OSError:
            pass
        raise OperationCancelled("cancelled by user")

    audio_start = start_sec if start_sec else None
    audio_dur = (end_sec - (start_sec or 0.0)) if end_sec else None
    mux.mux_hevc_with_audio(raw, dst, fps=fps, color=meta.color,
                            audio_source=str(src) if keep_audio else None,
                            audio_start_sec=audio_start, audio_duration=audio_dur)
    try:
        raw.unlink()
    except OSError:
        pass
    return dst


def extract_transformed_rect_clip(
    src: str | Path,
    dst: str | Path,
    *,
    crop_mode: str,
    rect: tuple[int, int, int, int],
    to_fisheye: bool = False,
    fov: float = 180.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = False,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """GPU extract a rect after optional eye crop + heq->fisheye transform."""
    import cupy as cp

    src = Path(src); dst = Path(dst)
    rx, ry, rw, rh = [int(v) for v in rect]
    if rw <= 0 or rh <= 0:
        raise ValueError(f"rect must be positive: {rect}")
    meta = probe.probe_video(src)
    bd = 10 if meta.bit_depth > 8 else 8
    fps = meta.source_fps or 30.0
    start_idx = int(round(start_sec * fps)) if start_sec else 0
    dec = PyNvThreadedSerialDecoder(src, bit_depth=bd, start_frame=start_idx)
    info = dec.info
    fps = meta.source_fps or info.fps or 30.0
    end_idx = int(round(end_sec * fps)) if end_sec else len(dec)
    end_idx = min(end_idx, len(dec))

    if crop_mode:
        x, yy, eye_w, eye_h = _crop_region(crop_mode, info.width, info.height)
    else:
        x = yy = 0
        eye_w, eye_h = info.width, info.height
    rx -= rx % 2
    ry -= ry % 2
    rw -= rw % 2
    rh -= rh % 2
    rx = max(0, min(rx, max(0, eye_w - 2)))
    ry = max(0, min(ry, max(0, eye_h - 2)))
    rw = max(2, min(rw, eye_w - rx))
    rh = max(2, min(rh, eye_h - ry))
    rw -= rw % 2
    rh -= rh % 2

    if log_callback:
        log_callback(
            f"[gpu] extract transformed rect: src={src}, dst={dst}, "
            f"sec={start_sec}-{end_sec}, crop_mode={crop_mode}, to_fisheye={to_fisheye}, "
            f"rect={rx},{ry},{rw}x{rh}"
        )

    lut_y = v360_lut.make_lut("heq2fisheye", eye_w, eye_h, fov) if to_fisheye else None
    lut_c = v360_lut.make_lut("heq2fisheye", eye_w // 2, eye_h // 2, fov) if to_fisheye else None
    bitrate_bps = _resolve_bitrate(rw, rh, fps, bitrate_bps, meta.bitrate_bps)
    enc_kwargs = _encoder_kwargs(meta, bitrate_bps)
    _log_encoder_settings("extract transformed rect", rw, rh, bd, enc_kwargs, log_callback)
    enc = PyNvEncoderSession(rw, rh, bit_depth=bd, codec="hevc",
                             **enc_kwargs)
    raw = _media_temp_path(dst, "raw")
    cancelled = False
    written = 0
    prog = _Progress(end_idx - start_idx, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(start_idx, end_idx):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True
                    break
                frame = dec.frame_at(i)
                cp.cuda.Device().synchronize()
                y, uv = frame.y_uv_cupy()
                y = y[yy:yy + eye_h, x:x + eye_w]
                uv = uv[yy // 2:(yy + eye_h) // 2, x // 2:(x + eye_w) // 2, :]
                if to_fisheye:
                    y = nv12_kernels.remap_y(y, lut_y, eye_w, eye_h)
                    uv = nv12_kernels.remap_uv(uv, lut_c, eye_w // 2, eye_h // 2)
                y = y[ry:ry + rh, rx:rx + rw]
                uv = uv[ry // 2:(ry + rh) // 2, rx // 2:(rx + rw) // 2, :]
                app = _pack_planes(y, uv, bd)
                sink.feed(app, force_idr=(written == 0))
                written += 1
                prog.update(written)
            if not cancelled:
                prog.finish(written)
                sink.flush()
    finally:
        dec.stop()
        runtime.free_memory_pool()

    if cancelled:
        try:
            raw.unlink()
        except OSError:
            pass
        raise OperationCancelled("cancelled by user")

    audio_start = start_sec if start_sec else None
    audio_dur = (end_sec - (start_sec or 0.0)) if end_sec else None
    mux.mux_hevc_with_audio(raw, dst, fps=fps, color=meta.color,
                            audio_source=str(src) if keep_audio else None,
                            audio_start_sec=audio_start, audio_duration=audio_dur,
                            log_callback=log_callback)
    try:
        raw.unlink()
    except OSError:
        pass
    return dst
