"""NativeMosaicEngine: in-process Lada mosaic-removal engine with singleton models and file restoration.

Replicates the core of Lada `cli.main.process_video_file`: FrameRestorer for
detection -> tracking -> restoration -> compositing, fully on torch CUDA, then
Lada VideoWriter (PyAV/libav, nvenc, in-process rather than subprocess), then
audio muxing.

Compared with the old lada-cli path, this removes subprocess cold start and
per-file model reloads. Detection and restoration already run on GPU.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from gpu_engine.native_mosaic import fisheye_delta


_FALSE_VALUES = {"", "0", "false", "no", "off"}
_DEFAULT_GPU_FRAME_SOURCE_MIN_PIXELS = 0
_GIB = 1024 ** 3


@dataclass(frozen=True)
class _NativeRestoreLimits:
    max_clip_length: int
    detector_batch_size: int
    frame_queue_mb: int
    clip_queue_mb: int
    detector_queue_size: int
    reason: str = ""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in _FALSE_VALUES


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return int(default)


def _env_int_optional(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _cuda_total_vram_bytes(torch_module=None, device=None) -> int | None:
    try:
        torch = torch_module
        if torch is None:
            import torch as _torch

            torch = _torch
        if not torch.cuda.is_available():
            return None
        if device is None:
            _free, total = torch.cuda.mem_get_info()
        else:
            with torch.cuda.device(device):
                _free, total = torch.cuda.mem_get_info()
        return int(total)
    except Exception:
        return None


def _video_dimensions(meta) -> tuple[int, int]:
    width = int(getattr(meta, "width", 0) or getattr(meta, "video_width", 0) or 0)
    height = int(getattr(meta, "height", 0) or getattr(meta, "video_height", 0) or 0)
    return max(0, width), max(0, height)


def _native_restore_limits(meta, requested_max_clip_length: int,
                           *, torch_module=None, device=None) -> _NativeRestoreLimits:
    """Pick conservative native restore limits for memory-bound GPU restores."""
    requested = max(1, int(requested_max_clip_length or 180))
    width, height = _video_dimensions(meta)
    pixels = width * height
    total_vram = _cuda_total_vram_bytes(torch_module=torch_module, device=device)

    max_clip = requested
    detector_batch = 4
    frame_queue_mb = 512
    clip_queue_mb = 512
    detector_queue_size = 8
    reason = ""

    guard_enabled = _env_bool("VRVT_NATIVE_VRAM_GUARD", True)
    guard_profile = None
    if total_vram is not None:
        if total_vram <= 10 * _GIB:
            guard_profile = (6_000_000, 24, 64, 1)
        elif total_vram <= 12 * _GIB:
            guard_profile = (8_000_000, 48, 96, 2)
        elif total_vram <= 18 * _GIB:
            guard_profile = (12_000_000, 64, 128, 2)
    if guard_enabled and guard_profile is not None and pixels >= guard_profile[0]:
        min_pixels, guarded_clip, guarded_queue_mb, guarded_queue_size = guard_profile
        max_clip = min(max_clip, guarded_clip)
        detector_batch = 1
        frame_queue_mb = guarded_queue_mb
        clip_queue_mb = guarded_queue_mb
        detector_queue_size = guarded_queue_size
        reason = (
            f"large restore frame {width}x{height} ({pixels} pixels, threshold {min_pixels}) "
            f"on {total_vram / _GIB:.1f}GiB GPU; max_clip_length {requested}->{max_clip}, "
            f"detector batch=1, queue_mb={guarded_queue_mb}, queues={detector_queue_size}"
        )

    env_clip = _env_int_optional("VRVT_NATIVE_MAX_CLIP_LENGTH")
    if env_clip is not None:
        old_clip = max_clip
        max_clip = max(1, min(max_clip, env_clip))
        reason = f"{reason}; " if reason else ""
        reason += f"VRVT_NATIVE_MAX_CLIP_LENGTH={env_clip} (max_clip_length {old_clip}->{max_clip})"

    detector_batch = max(1, _env_int("VRVT_NATIVE_DETECT_BATCH", detector_batch))
    frame_queue_mb = max(16, _env_int("VRVT_NATIVE_FRAME_QUEUE_MB", frame_queue_mb))
    clip_queue_mb = max(16, _env_int("VRVT_NATIVE_CLIP_QUEUE_MB", clip_queue_mb))
    detector_queue_size = max(1, _env_int("VRVT_NATIVE_DETECT_QUEUE", detector_queue_size))

    return _NativeRestoreLimits(
        max_clip_length=max_clip,
        detector_batch_size=detector_batch,
        frame_queue_mb=frame_queue_mb,
        clip_queue_mb=clip_queue_mb,
        detector_queue_size=detector_queue_size,
        reason=reason,
    )


def _gpu_frame_source_decision(src_meta) -> tuple[bool, str, int, int]:
    """Decide whether restore_file should use NVDEC-backed frame source."""
    width = int(getattr(src_meta, "width", 0) or 0)
    height = int(getattr(src_meta, "height", 0) or 0)
    pixels = max(0, width) * max(0, height)
    min_pixels = max(0, _env_int("VRVT_GPU_FRAME_SOURCE_MIN_PIXELS", _DEFAULT_GPU_FRAME_SOURCE_MIN_PIXELS))

    if _env_bool("VRVT_NATIVE_FORCE_CPU_FRAME_SOURCE", False):
        return False, "forced by VRVT_NATIVE_FORCE_CPU_FRAME_SOURCE", pixels, min_pixels
    if min_pixels > 0 and pixels < min_pixels:
        return False, f"input too small ({width}x{height}, {pixels} < {min_pixels} pixels)", pixels, min_pixels
    return True, f"gpu frame source eligible ({width}x{height}, {pixels} >= {min_pixels} pixels)", pixels, min_pixels


def _log_error_marker(elem, log_func, *, prefix: str = "[native]") -> None:
    message = str(elem) or type(elem).__name__
    log_func(f"{prefix} frame restorer error: {message}")
    stack = getattr(elem, "stack_trace", "") or ""
    if not stack:
        return
    lines = [line.rstrip() for line in stack.splitlines() if line.strip()]
    if not lines:
        return
    for line in lines[-10:]:
        log_func(f"{prefix} {line}")


class _GpuEncodeSetupError(Exception):
    """GPU NVENC setup-stage failure, allowing cheap fallback to VideoWriter.

    This covers probing, encoder creation, and temporary-file creation. Errors
    after entering the encode loop do not use this exception and do not fall
    back, avoiding a second full mosaic-removal pass.
    """


class NativeMosaicEngine:
    def __init__(self):
        # __init__._prepare() has already locked CuPy nvrtc and set vendor paths before torch import.
        import torch
        from lada.restorationpipeline import load_models
        from lada.utils.os_utils import gpu_has_fp16_acceleration
        from . import _torch_tuning
        from .models_cfg import detection_model_path, restoration_model_path

        self._tuning_state = _torch_tuning.apply_inference_tuning()
        self.torch = torch
        self.device = torch.device("cuda")
        try:
            self._decode_stream = torch.cuda.Stream(device=self.device)
        except Exception:
            self._decode_stream = None
        self.fp16 = bool(gpu_has_fp16_acceleration())
        self.restoration_name = "basicvsrpp"
        det_path = detection_model_path()
        res_path = restoration_model_path()
        if not os.path.isfile(det_path):
            raise FileNotFoundError(f"检测模型不存在: {det_path}")
        if not os.path.isfile(res_path):
            raise FileNotFoundError(f"恢复模型不存在: {res_path}")
        # Load once and reuse across files.
        self.detection_model, self.restoration_model, self.pad_mode = load_models(
            self.device, self.restoration_name, res_path, None, det_path,
            self.fp16, False,
        )
        self._tuning_state["restoration_channels_last"] = _torch_tuning.convert_module_channels_last(
            getattr(self.restoration_model, "model", None)
        )
        self._tuning_state["detection_channels_last"] = _torch_tuning.convert_module_channels_last(
            getattr(self.detection_model, "model", None)
        )
        self._patch_detection_gpu_preprocess()
        self._warmup_native_pipeline()

    def release(self) -> None:
        """Release model references held by the singleton before memory-heavy GPU stages."""
        self.detection_model = None
        self.restoration_model = None
        self.pad_mode = None
        self._decode_stream = None
        try:
            from gpu_engine import runtime

            runtime.free_memory_pool()
        except Exception:
            pass
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def _patch_detection_gpu_preprocess(self):
        """Fix LADA GPU frame preprocessing for HWC torch frames.

        The vendored YOLO wrapper's GPU path is rarely used by file-based LADA and may
        treat HWC frames as BCHW. Patch the loaded instance here so the fix lives in
        tracked native engine code even if ignored vendor model files are not committed.
        """
        import types

        from lada.utils.torch_letterbox import PyTorchLetterBox

        model = self.detection_model

        def _preprocess_gpu(_self, imgs):
            im = self.torch.stack(imgs, dim=0).permute(0, 3, 1, 2).contiguous()
            return _self.letterbox(im)

        def _preprocess(_self, imgs):
            if imgs[0].device.type == "cpu":
                return _self._preprocess_cpu(imgs)
            original_shape = getattr(_self.letterbox, "original_shape", None)
            if _self.letterbox is None or imgs[0].shape[:2] != original_shape:
                _self.letterbox = PyTorchLetterBox(_self.imgsz, imgs[0].shape[:2], stride=_self.stride)
            return _self._preprocess_gpu(imgs)

        model._preprocess_gpu = types.MethodType(_preprocess_gpu, model)
        model.preprocess = types.MethodType(_preprocess, model)

    def _warmup_native_pipeline(self):
        """Trigger CUDA kernels and model autotune once after model load.

        Warmup must never make engine startup fail; unsupported model shapes or
        driver hiccups are captured for diagnostics and the real restore path can
        still run normally.
        """
        value = str(os.environ.get("VRVT_NATIVE_WARMUP", "1")).strip().lower()
        if value in {"0", "false", "no", "off"}:
            self._warmup_error = None
            return

        errors: list[str] = []
        try:
            import cupy as cp
            from gpu_engine import nv12_kernels

            y = cp.zeros((16, 16), dtype=cp.uint8)
            uv = cp.full((8, 8, 2), 128, dtype=cp.uint8)
            bgr = nv12_kernels.nv12_to_bgr(y, uv, bit_depth=8)
            nv12_kernels.bgr_to_nv12(bgr)
            cp.cuda.get_current_stream().synchronize()
        except Exception as exc:
            errors.append(f"cupy/nv12: {type(exc).__name__}: {exc}")

        try:
            frames = [
                self.torch.zeros((256, 256, 3), device=self.device, dtype=self.torch.uint8)
                for _ in range(max(1, min(3, int(os.environ.get("VRVT_NATIVE_WARMUP_FRAMES", "3")))))
            ]
            with self.torch.inference_mode():
                batch = self.detection_model.preprocess(frames[:1])
                self.detection_model.inference_and_postprocess(batch, frames[:1])
            self.torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            errors.append(f"detection: {type(exc).__name__}: {exc}")

        try:
            frames = [
                self.torch.zeros((256, 256, 3), device=self.device, dtype=self.torch.uint8)
                for _ in range(max(1, min(3, int(os.environ.get("VRVT_NATIVE_WARMUP_FRAMES", "3")))))
            ]
            with self.torch.inference_mode():
                self.restoration_model.restore(frames, max_frames=1)
            self.torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            errors.append(f"restoration: {type(exc).__name__}: {exc}")

        try:
            # Startup capture is only a cheap validation that graph capture works
            # on this system; default small so it does not pin a large unused graph
            # in VRAM. The real per-clip-length capture happens single-threaded at
            # the start of restore_file (see warm(max_clip_length) there).
            warmup_clip_length = int(os.environ.get("VRVT_CUDA_GRAPH_WARMUP_FRAMES", "8"))
            if hasattr(self.restoration_model, "warmup_graph"):
                self.restoration_model.warmup_graph(warmup_clip_length)
                self.torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            errors.append(f"cuda_graph: {type(exc).__name__}: {exc}")

        self._warmup_error = "; ".join(errors) if errors else None

    def _resolve_encoder(self):
        """Choose an nvenc HEVC encoding preset, falling back to reasonable hevc_nvenc options if the CSV preset is unavailable."""
        try:
            from utils import encode_config

            return "hevc_nvenc", encode_config.build_lada_encoder_options(cq=18).strip()
        except Exception:
            pass
        # Defensive fallback for early import/bootstrap failures before the
        # shared OneClick encode profile layer is available.
        from lada.utils import video_utils
        try:
            default_name = video_utils.get_default_preset_name()
            for p in video_utils.get_encoding_presets():
                if p.name == default_name:
                    return p.encoder_name, p.encoder_options
        except Exception:
            pass
        return "hevc_nvenc", "-preset p5 -rc vbr -cq 20"

    def restore_file(self, input_path, output_path, *, max_clip_length=180,
                     bitrate_bps: int | None = None,
                     log_callback=None, cancel_token=None,
                     produce_mp4: bool = True,
                     sidecar_metadata: dict | None = None) -> bool:
        """Remove mosaics and write the output file.

        Plan A from 2026-05-31: prefer our GPU NVENC for output encoding. This
        encodes directly on GPU, avoiding Lada VideoWriter's per-frame GPU->CPU
        download, CPU swscale rgb24->yuv420p, and upload path. 4K/8K encoding is
        about 10x faster, and profiling showed Lada VideoWriter took about 59%
        of the mosaic-removal segment time. Enable this only for PyNv-safe
        8-bit SDR sources; HDR/10-bit/bt2020 or setup failure falls back to Lada
        VideoWriter. Our NVENC path only uses current-stream synchronization and
        one lightweight kernel per frame for NV12 conversion, so it does not
        strongly contend with torch model inference like the section 4.5
        streaming path did.
        """
        from gpu_engine._profile import DecodeProfile, get_active_profile, set_active_profile
        from lada.restorationpipeline.frame_restorer import FrameRestorer
        from lada.utils import video_utils

        def _log(m):
            if log_callback:
                log_callback(m)

        profile = DecodeProfile.from_env_or_argv()
        profile.metadata(
            input=str(input_path),
            output=str(output_path),
            max_clip_length=int(max_clip_length),
            produce_mp4=bool(produce_mp4),
            fp16=bool(self.fp16),
            warmup_error=self._warmup_error,
            torch_tuning=self._tuning_state,
        )
        previous_profile = get_active_profile()
        set_active_profile(profile)
        restore_limits = _NativeRestoreLimits(
            max_clip_length=max(1, int(max_clip_length or 180)),
            detector_batch_size=4,
            frame_queue_mb=512,
            clip_queue_mb=512,
            detector_queue_size=8,
        )

        def _make_restorer(frame_source_factory=None, video_meta_data=None):
            return FrameRestorer(
                self.device, input_path, restore_limits.max_clip_length, self.restoration_name,
                self.detection_model, self.restoration_model, self.pad_mode,
                video_meta_data=video_meta_data, frame_source_factory=frame_source_factory,
                progress_log_callback=log_callback,
                detector_batch_size=restore_limits.detector_batch_size,
                frame_queue_mb=restore_limits.frame_queue_mb,
                clip_queue_mb=restore_limits.clip_queue_mb,
                detector_queue_size=restore_limits.detector_queue_size,
            )

        try:
            with profile.section("restore_file.total"):
                with profile.section("metadata.lada_video"):
                    meta = video_utils.get_video_meta_data(input_path)

                gpu_ok = False
                frame_source_ok = False
                src_meta = None
                decision = None
                frame_source_reason = ""
                try:
                    from gpu_engine import probe
                    with profile.section("metadata.probe_route"):
                        src_meta, decision = probe.route(input_path)
                    gpu_ok = bool(decision.is_gpu and not src_meta.is_hdr and not src_meta.is_bt2020)
                    frame_source_ok = gpu_ok
                    if frame_source_ok:
                        frame_source_ok, frame_source_reason, frame_source_pixels, frame_source_min_pixels = (
                            _gpu_frame_source_decision(src_meta)
                        )
                    else:
                        frame_source_pixels = int(src_meta.width) * int(src_meta.height)
                        frame_source_min_pixels = _DEFAULT_GPU_FRAME_SOURCE_MIN_PIXELS
                    profile.metadata(
                        source_backend=decision.backend,
                        source_backend_reason=decision.reason,
                        source_width=int(src_meta.width),
                        source_height=int(src_meta.height),
                        source_bit_depth=int(src_meta.bit_depth),
                        source_fps=float(src_meta.source_fps or 0.0),
                        source_frames=int(src_meta.nb_frames or 0),
                        frame_source_pixels=int(frame_source_pixels),
                        frame_source_min_pixels=int(frame_source_min_pixels),
                        frame_source_reason=frame_source_reason,
                    )
                    restore_limits = _native_restore_limits(
                        src_meta,
                        max_clip_length,
                        torch_module=self.torch,
                        device=self.device,
                    )
                    profile.metadata(
                        native_max_clip_length=int(restore_limits.max_clip_length),
                        native_detector_batch_size=int(restore_limits.detector_batch_size),
                        native_frame_queue_mb=int(restore_limits.frame_queue_mb),
                        native_clip_queue_mb=int(restore_limits.clip_queue_mb),
                        native_detector_queue_size=int(restore_limits.detector_queue_size),
                        native_restore_limits_reason=restore_limits.reason,
                    )
                    if restore_limits.reason:
                        _log(f"[native] VRAM guard: {restore_limits.reason}")
                    if not gpu_ok:
                        _log(f"[native] GPU NVENC unavailable ({decision.reason or 'non-SDR/10-bit'}); using VideoWriter")
                        if not produce_mp4:
                            _log("[native] raw HEVC restore requested but GPU NVENC path is unavailable; writing mp4")
                except Exception as e:
                    _log(f"[native] probe failed, using VideoWriter: {type(e).__name__}: {e}")
                    gpu_ok = False
                    frame_source_ok = False
                    restore_limits = _native_restore_limits(
                        meta,
                        max_clip_length,
                        torch_module=self.torch,
                        device=self.device,
                    )
                    if restore_limits.reason:
                        _log(f"[native] VRAM guard: {restore_limits.reason}")

                # Pre-capture the restoration CUDA graph for this clip length now,
                # while we are still single-threaded. Most clips are exactly the
                # effective max_clip_length, so this captures the dominant shape.
                # Runtime capture is forbidden (it races with the detection/encode
                # threads), so any other clip length falls back to eager.
                try:
                    warm = getattr(self.restoration_model, "warmup_graph", None)
                    if callable(warm):
                        warm(int(restore_limits.max_clip_length))
                        self.torch.cuda.current_stream(self.device).synchronize()
                except Exception as exc:
                    _log(f"[native] graph warmup skipped: {type(exc).__name__}: {exc}")
                    try:
                        self.torch.cuda.empty_cache()
                    except Exception:
                        pass

                frame_source_factory = None
                restorer_meta = None
                if frame_source_ok:
                    try:
                        frame_source_factory, restorer_meta = self._make_gpu_bgr_frame_source_factory(
                            input_path,
                            crop_mode="passthrough",
                            to_fisheye=False,
                            start_sec=None,
                            end_sec=None,
                            profile=profile,
                        )
                        profile.metadata(frame_source="gpu_passthrough", frame_source_reason=frame_source_reason)
                        _log("[native] frame_source=nvdec_gpu_passthrough")
                    except Exception as e:
                        frame_source_factory = None
                        restorer_meta = None
                        profile.metadata(frame_source="cpu_videoreader", frame_source_error=f"{type(e).__name__}: {e}")
                        _log(f"[native] GPU frame source unavailable ({type(e).__name__}: {e}); using VideoReader")
                else:
                    profile.metadata(frame_source="cpu_videoreader", frame_source_reason=frame_source_reason)
                    if gpu_ok and frame_source_reason:
                        _log(f"[native] frame_source=cpu_videoreader ({frame_source_reason})")

                if gpu_ok:
                    try:
                        return self._restore_file_gpu_nvenc(
                            _make_restorer(frame_source_factory, restorer_meta),
                            input_path, output_path, meta, src_meta,
                            bitrate_bps=bitrate_bps,
                            log_callback=log_callback, cancel_token=cancel_token,
                            produce_mp4=produce_mp4,
                            sidecar_metadata=sidecar_metadata,
                            profile=profile,
                        )
                    except _GpuEncodeSetupError as e:
                        _log(f"[native] GPU NVENC setup failed ({e}); fallback to VideoWriter")

                return self._restore_file_videowriter(
                    _make_restorer(frame_source_factory, restorer_meta), input_path, output_path, meta,
                    log_callback=log_callback, cancel_token=cancel_token, profile=profile,
                )
        finally:
            try:
                profile.write(log_callback)
            except Exception as e:
                _log(f"[profile] write failed: {type(e).__name__}: {e}")
            set_active_profile(previous_profile)

    def _restore_file_gpu_nvenc(self, frame_restorer, input_path, output_path, meta, src_meta,
                                *, bitrate_bps: int | None = None,
                                log_callback=None, cancel_token=None,
                                produce_mp4: bool = True,
                                sidecar_metadata: dict | None = None,
                                profile=None) -> bool:
        """Plan A: encode restored frames directly from GPU through PyNv NVENC.

        Restored BGR frames from the file path, currently CPU torch, are uploaded
        to GPU, converted by a fused BGR->NV12 kernel (M1), encoded directly by
        NVENC from GPU, written as raw HEVC, then muxed with audio. Encoder setup
        failures raise _GpuEncodeSetupError so the caller can fall back before
        running Lada. Once the encode loop starts, failures are treated as
        failures without fallback to avoid repeating a long mosaic-removal run.
        """
        from gpu_engine import mux, runtime
        from gpu_engine.files import (
            _EncodeSink, _Progress, _encoder_kwargs, _log_encoder_settings,
            _media_temp_path,
            _pack_planes, _resolve_bitrate,
        )
        from gpu_engine.pynv_io import PyNvEncoderSession
        from lada.utils.threading_utils import STOP_MARKER, ErrorMarker

        def _log(m):
            if log_callback:
                log_callback(m)

        # --- setup, where failure can fall back cheaply ---
        try:
            with profile.section("gpu_nvenc.setup") if profile else nullcontext():
                out_w, out_h = int(meta.video_width), int(meta.video_height)
                fps = float(meta.video_fps_exact)
                bitrate = _resolve_bitrate(out_w, out_h, fps, bitrate_bps, getattr(src_meta, "bitrate_bps", None))
                enc_kwargs = _encoder_kwargs(src_meta, bitrate)
                _log_encoder_settings("native restore", out_w, out_h, 8, enc_kwargs, log_callback)
                enc = PyNvEncoderSession(
                    out_w, out_h, bit_depth=8, codec="hevc",
                    **enc_kwargs,
                )
                raw_path = _media_temp_path(output_path, "native")
        except Exception as e:
            raise _GpuEncodeSetupError(f"{type(e).__name__}: {e}") from e

        _log(f"[native] encoder=hevc_nvenc(gpu-resident) fp16={self.fp16}")

        total = getattr(meta, "frames_count", 0) or 0
        from gpu_engine.native_mosaic.progress import native_progress_interval_s, native_progress_min_pct

        prog = _Progress(
            total,
            log_callback,
            min_interval=native_progress_interval_s(),
            min_pct=native_progress_min_pct(),
            prefix="[native]",
        )
        written = 0
        success = True
        try:
            with profile.section("restorer.start") if profile else nullcontext():
                frame_restorer.start()
            with open(raw_path, "wb") as f:
                sink = _EncodeSink(enc, f)
                for elem in frame_restorer:
                    if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                        success = False
                        _log("[native] cancelled by user")
                        break
                    if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                        success = False
                        if isinstance(elem, ErrorMarker):
                            _log_error_marker(elem, _log)
                        else:
                            _log("[native] frame restorer stopped prematurely")
                        break
                    restored_frame, _pts = elem
                    with profile.section("encode.prepare_nv12", torch_module=self.torch, cuda=True) if profile else nullcontext():
                        y_plane, uv_plane = self._prepare_restored_nv12(
                            restored_frame, from_fisheye=False, out_w=out_w, out_h=out_h,
                            profile=profile,
                        )
                        app = _pack_planes(y_plane, uv_plane, 8)
                    with profile.section("encode.nvenc_feed") if profile else nullcontext():
                        sink.feed(app, force_idr=(written == 0))
                    written += 1
                    if profile:
                        profile.increment("frames_encoded")
                    prog.update(written)
                if success:
                    prog.finish(written)
                    with profile.section("encode.nvenc_flush") if profile else nullcontext():
                        sink.flush()
        except Exception as e:
            success = False
            _log(f"[native] error during GPU encode: {type(e).__name__}: {e}")
        finally:
            frame_restorer.stop()
            try:
                runtime.free_memory_pool()
                self.torch.cuda.empty_cache()
            except Exception:
                pass

        if not success:
            try:
                if raw_path.exists():
                    raw_path.unlink()
            except OSError:
                pass
            return False

        if not produce_mp4:
            from gpu_engine import restored_sidecar

            raw_output = restored_sidecar.raw_path_for_output(output_path)
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            try:
                if raw_output.exists():
                    raw_output.unlink()
                raw_path.replace(raw_output)
                sidecar_metadata = sidecar_metadata or {}
                restored_sidecar.write_restored_sidecar(
                    raw_output,
                    width=out_w,
                    height=out_h,
                    bit_depth=8,
                    fps=fps,
                    frame_count=written,
                    color=getattr(src_meta, "color", None),
                    source=input_path,
                    rect=sidecar_metadata.get("rect"),
                    time_range=sidecar_metadata.get("time"),
                    encoder=(
                        f"hevc_nvenc {enc_kwargs.get('preset')} {enc_kwargs.get('rc')} "
                        f"{int(enc_kwargs.get('bitrate', 0)) // 1000}kbps"
                    ),
                )
            except Exception as exc:
                try:
                    if raw_path.exists():
                        raw_path.unlink()
                    if raw_output.exists():
                        raw_output.unlink()
                except OSError:
                    pass
                _log(f"[native] raw sidecar write failed: {type(exc).__name__}: {exc}")
                return False
            _log(f"[native] done raw -> {raw_output}")
            return True

        _log("[native] muxing audio...")
        with profile.section("mux.audio") if profile else nullcontext():
            mux.mux_hevc_with_audio(
                raw_path, output_path, fps=fps, color=getattr(src_meta, "color", None),
                audio_source=input_path, audio_start_sec=None, audio_duration=None,
                log_callback=log_callback,
            )
        try:
            raw_path.unlink()
        except OSError:
            pass
        _log(f"[native] done -> {output_path}")
        return True

    def _restore_file_videowriter(self, frame_restorer, input_path, output_path, meta,
                                  *, log_callback=None, cancel_token=None,
                                  profile=None) -> bool:
        """Fallback to the original Lada VideoWriter encode path for HDR/10-bit/non-PyNv-safe sources."""
        from gpu_engine.files import _Progress, _media_temp_path
        from gpu_engine.native_mosaic import _gpu_ops
        from lada.utils import audio_utils, video_utils
        from lada.utils.threading_utils import STOP_MARKER, ErrorMarker

        def _log(m):
            if log_callback:
                log_callback(m)

        encoder, encoder_options = self._resolve_encoder()
        _log(f"[native] encoder={encoder} opts='{encoder_options}' fp16={self.fp16}")

        suffix = os.path.splitext(output_path)[1] or ".mp4"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = str(_media_temp_path(out_path, "native_tmp", suffix=suffix))

        success = True
        n = 0
        total = getattr(meta, "frames_count", 0) or 0
        from gpu_engine.native_mosaic.progress import native_progress_interval_s, native_progress_min_pct

        prog = _Progress(
            total,
            log_callback,
            min_interval=native_progress_interval_s(),
            min_pct=native_progress_min_pct(),
            prefix="[native]",
        )
        try:
            with profile.section("restorer.start") if profile else nullcontext():
                frame_restorer.start()
            with video_utils.VideoWriter(
                tmp_out, meta.video_width, meta.video_height, meta.video_fps_exact,
                encoder=encoder, encoder_options=encoder_options,
                time_base=meta.time_base, mp4_fast_start=False,
            ) as writer:
                for elem in frame_restorer:
                    if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                        success = False
                        _log("[native] cancelled by user")
                        break
                    if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                        success = False
                        if isinstance(elem, ErrorMarker):
                            _log_error_marker(elem, _log)
                        else:
                            _log("[native] frame restorer stopped prematurely")
                        break
                    restored_frame, restored_pts = elem
                    _gpu_ops.wait_decode_event(restored_frame)
                    if profile and isinstance(restored_frame, self.torch.Tensor) and restored_frame.is_cuda:
                        profile.increment("d2h_expected_count")
                    with profile.section("videowriter.write") if profile else nullcontext():
                        writer.write(restored_frame, restored_pts, bgr2rgb=True)
                    n += 1
                    if profile:
                        profile.increment("frames_written_videowriter")
                    prog.update(n)
                if success:
                    prog.finish(n)
        except Exception as e:
            success = False
            _log(f"[native] error: {type(e).__name__}: {e}")
        finally:
            frame_restorer.stop()

        if success:
            _log("[native] muxing audio...")
            with profile.section("mux.audio_videowriter") if profile else nullcontext():
                audio_utils.combine_audio_video_files(meta, tmp_out, output_path)
            _log(f"[native] done -> {output_path}")
        else:
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except OSError:
                pass
        return success

    @staticmethod
    def _crop_region(mode: str, w: int, h: int):
        if mode == "passthrough":
            return (0, 0, w, h)
        hw, hh = w // 2, h // 2
        return {
            "left": (0, 0, hw, h),
            "right": (hw, 0, hw, h),
            "top": (0, 0, w, hh),
            "bottom": (0, hh, w, hh),
        }[mode]

    @staticmethod
    def _cupy_to_torch(arr, profile=None, synchronize: bool = True):
        import cupy as cp
        import torch

        arr = cp.ascontiguousarray(arr)
        if synchronize and profile:
            profile.increment("sync_count")
            profile.increment("cupy_to_torch_sync_count")
        if synchronize:
            with profile.section("sync.cupy_to_torch") if profile else nullcontext():
                cp.cuda.get_current_stream().synchronize()
        try:
            return torch.utils.dlpack.from_dlpack(arr)
        except TypeError:
            return torch.utils.dlpack.from_dlpack(arr.toDlpack())

    @staticmethod
    def _torch_to_cupy(tensor, profile=None):
        import cupy as cp

        if not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=False)
        tensor = tensor.contiguous()
        try:
            import torch
            if profile:
                profile.increment("sync_count")
                profile.increment("torch_to_cupy_sync_count")
            with profile.section("sync.torch_to_cupy") if profile else nullcontext():
                torch.cuda.current_stream(tensor.device).synchronize()
        except Exception:
            pass
        try:
            return cp.from_dlpack(tensor)
        except TypeError:
            return cp.from_dlpack(tensor.detach())

    def _planes_to_torch_bgr(self, y_plane, uv_plane, bit_depth: int = 8, profile=None):
        """Convert NV12/P010 CuPy planes to BGR8 torch CUDA tensors.

        CuPy RawKernel fuses chroma upsampling with YUV->BGR conversion to avoid
        large temporary tensors from torch repeat_interleave on 4K/8K eye frames.
        """
        from gpu_engine import nv12_kernels

        with profile.section("decode.nv12_to_bgr", torch_module=self.torch, cuda=True) if profile else nullcontext():
            bgr = nv12_kernels.nv12_to_bgr(y_plane, uv_plane, bit_depth=bit_depth)
        return self._cupy_to_torch(bgr, profile=profile).to(device=self.device)

    def _planes_to_torch_bgr_async(self, y_plane, uv_plane, bit_depth: int = 8, profile=None):
        """Convert planes on the decode stream and attach a torch event to the tensor."""
        import cupy as cp
        from gpu_engine import nv12_kernels
        from gpu_engine.native_mosaic import _gpu_ops

        if self._decode_stream is None:
            return self._planes_to_torch_bgr(y_plane, uv_plane, bit_depth=bit_depth, profile=profile), None

        with self.torch.cuda.stream(self._decode_stream):
            with cp.cuda.ExternalStream(self._decode_stream.cuda_stream):
                with profile.section("decode.nv12_to_bgr", torch_module=self.torch, cuda=True) if profile else nullcontext():
                    bgr = nv12_kernels.nv12_to_bgr(y_plane, uv_plane, bit_depth=bit_depth)
                tensor = self._cupy_to_torch(bgr, profile=profile, synchronize=False).to(device=self.device)
                event = self.torch.cuda.Event()
                event.record(self._decode_stream)
        _gpu_ops.attach_decode_event(tensor, event)
        if profile:
            profile.increment("decode_event_count")
        return tensor, event

    def _torch_bgr_to_nv12_cupy(self, bgr_frame, profile=None):
        """Convert BGR8 torch CUDA tensors to NV12 CuPy planes."""
        from gpu_engine import nv12_kernels

        bgr = bgr_frame.to(device=self.device, dtype=self.torch.uint8)
        return nv12_kernels.bgr_to_nv12(self._torch_to_cupy(bgr, profile=profile))

    @staticmethod
    def _to_lada_video_meta(path: str, width: int, height: int, fps: float,
                            frames_count: int, duration: float):
        from lada.utils import VideoMetadata

        fps_fraction = Fraction(str(fps)).limit_denominator(1000000)
        return VideoMetadata(
            video_file=str(path),
            video_height=int(height),
            video_width=int(width),
            video_fps=float(fps),
            average_fps=float(fps),
            video_fps_exact=fps_fraction,
            codec_name="gpu_frame_source",
            frames_count=int(frames_count),
            duration=float(duration),
            time_base=Fraction(1, max(1, fps_fraction.denominator)),
            start_pts=0,
        )

    def _make_gpu_bgr_frame_source_factory(
        self,
        input_path,
        *,
        crop_mode: str,
        to_fisheye: bool,
        start_sec: float | None,
        end_sec: float | None,
        fov: float = 180.0,
        profile=None,
    ):
        from gpu_engine import probe

        meta = probe.probe_video(input_path)
        fps = meta.source_fps or 30.0
        start_idx = max(0, int(round((start_sec or 0.0) * fps)))
        end_idx = int(round(end_sec * fps)) if end_sec is not None else None
        if crop_mode in {"sbs", "passthrough"}:
            out_w, out_h = meta.width, meta.height
        else:
            _x, _y0, out_w, out_h = self._crop_region(crop_mode, meta.width, meta.height)
        frames_count = max(0, (end_idx if end_idx is not None else (meta.nb_frames or 0)) - start_idx)
        if frames_count <= 0 and meta.duration > 0:
            end_time = end_sec if end_sec is not None else meta.duration
            frames_count = max(0, int(round((end_time - (start_sec or 0.0)) * fps)))
        duration = (frames_count / fps) if fps > 0 and frames_count else 0.0
        lada_meta = self._to_lada_video_meta(input_path, out_w, out_h, fps, frames_count, duration)

        def _factory(start_ns=0, start_frame=0):
            source_start = (start_sec or 0.0) + (float(start_ns or 0) / 1_000_000_000.0)
            if crop_mode == "sbs":
                return self._iter_gpu_bgr_frames_sbs(
                    input_path,
                    to_fisheye=to_fisheye,
                    start_sec=source_start,
                    end_sec=end_sec,
                    fov=fov,
                )
            return self._iter_gpu_bgr_frames(
                input_path,
                crop_mode=crop_mode,
                to_fisheye=to_fisheye,
                start_sec=source_start,
                end_sec=end_sec,
                fov=fov,
                profile=profile,
            )

        return _factory, lada_meta

    def _iter_gpu_bgr_frames(
        self,
        input_path,
        *,
        crop_mode: str,
        to_fisheye: bool,
        start_sec: float | None,
        end_sec: float | None,
        fov: float = 180.0,
        profile=None,
    ):
        import cupy as cp
        from gpu_engine import nv12_kernels, probe, v360_lut
        from gpu_engine.pynv_io import PyNvThreadedSerialDecoder

        meta = probe.probe_video(input_path)
        bd = 10 if meta.bit_depth > 8 else 8
        fps = meta.source_fps or 30.0
        start_idx = max(0, int(round((start_sec or 0.0) * fps)))
        end_idx = int(round(end_sec * fps)) if end_sec is not None else None
        dec = PyNvThreadedSerialDecoder(Path(input_path), bit_depth=bd, start_frame=start_idx)
        pending_events = []
        try:
            total = len(dec)
            stop_idx = min(end_idx if end_idx is not None else total, total)
            info = dec.info
            x, y0, out_w, out_h = self._crop_region(crop_mode, info.width, info.height)
            lut_y = v360_lut.make_lut("heq2fisheye", out_w, out_h, fov) if to_fisheye else None
            lut_c = v360_lut.make_lut("heq2fisheye", out_w // 2, out_h // 2, fov) if to_fisheye else None
            decoder_batch_size = max(8, int(getattr(dec, "batch_size", 8) or 8))
            if profile:
                profile.metadata(decode_event_batch_size=int(decoder_batch_size))

            def _sync_pending_events():
                if not pending_events:
                    return
                with profile.section("sync.decode_batch_event") if profile else nullcontext():
                    for event in pending_events:
                        event.synchronize()
                if profile:
                    profile.increment("decode_batch_event_sync_count")
                pending_events.clear()

            for i in range(start_idx, stop_idx):
                if pending_events and ((i - start_idx) % decoder_batch_size == 0):
                    _sync_pending_events()
                with profile.section("decode.frame_at") if profile else nullcontext():
                    frame = dec.frame_at(i)
                y_plane, uv_plane = frame.y_uv_cupy()
                y_plane = y_plane[y0:y0 + out_h, x:x + out_w]
                uv_plane = uv_plane[y0 // 2:(y0 + out_h) // 2, x // 2:(x + out_w) // 2, :]
                if to_fisheye:
                    with profile.section("decode.remap_fisheye", torch_module=self.torch, cuda=True) if profile else nullcontext():
                        y_plane = nv12_kernels.remap_y(y_plane, lut_y, out_w, out_h)
                        uv_plane = nv12_kernels.remap_uv(uv_plane, lut_c, out_w // 2, out_h // 2)
                    bgr = self._planes_to_torch_bgr(y_plane, uv_plane, bit_depth=bd, profile=profile)
                    event = None
                else:
                    bgr, event = self._planes_to_torch_bgr_async(y_plane, uv_plane, bit_depth=bd, profile=profile)
                    if event is not None:
                        pending_events.append(event)
                if profile:
                    profile.increment("frames_decoded")
                yield bgr, getattr(frame, "pts", i - start_idx)
        finally:
            try:
                with profile.section("sync.decode_batch_event") if profile and pending_events else nullcontext():
                    for event in pending_events:
                        event.synchronize()
                if profile and pending_events:
                    profile.increment("decode_batch_event_sync_count")
            except Exception:
                pass
            dec.stop()

    def _iter_gpu_bgr_frames_sbs(
        self,
        input_path,
        *,
        to_fisheye: bool,
        start_sec: float | None,
        end_sec: float | None,
        fov: float = 180.0,
    ):
        import cupy as cp
        from gpu_engine import nv12_kernels, probe, v360_lut
        from gpu_engine.pynv_io import PyNvThreadedSerialDecoder

        meta = probe.probe_video(input_path)
        bd = 10 if meta.bit_depth > 8 else 8
        fps = meta.source_fps or 30.0
        start_idx = max(0, int(round((start_sec or 0.0) * fps)))
        end_idx = int(round(end_sec * fps)) if end_sec is not None else None
        dec = PyNvThreadedSerialDecoder(Path(input_path), bit_depth=bd, start_frame=start_idx)
        try:
            total = len(dec)
            stop_idx = min(end_idx if end_idx is not None else total, total)
            info = dec.info
            eye_w = info.width // 2
            eye_h = info.height
            lut_y = v360_lut.make_lut("heq2fisheye", eye_w, eye_h, fov) if to_fisheye else None
            lut_c = v360_lut.make_lut("heq2fisheye", eye_w // 2, eye_h // 2, fov) if to_fisheye else None
            for i in range(start_idx, stop_idx):
                frame = dec.frame_at(i)
                # Plan C: remove whole-device synchronization for the same reason as _iter_gpu_bgr_frames.
                y_plane, uv_plane = frame.y_uv_cupy()
                if to_fisheye:
                    ly = y_plane[:, :eye_w]
                    ry = y_plane[:, eye_w:eye_w * 2]
                    luv = uv_plane[:, :eye_w // 2, :]
                    ruv = uv_plane[:, eye_w // 2:eye_w, :]
                    ly = nv12_kernels.remap_y(ly, lut_y, eye_w, eye_h)
                    ry = nv12_kernels.remap_y(ry, lut_y, eye_w, eye_h)
                    luv = nv12_kernels.remap_uv(luv, lut_c, eye_w // 2, eye_h // 2)
                    ruv = nv12_kernels.remap_uv(ruv, lut_c, eye_w // 2, eye_h // 2)
                    y_plane = nv12_kernels.hstack_planes(ly, ry)
                    uv_plane = nv12_kernels.hstack_planes(luv, ruv)
                yield self._planes_to_torch_bgr(y_plane, uv_plane, bit_depth=bd), getattr(frame, "pts", i - start_idx)
        finally:
            dec.stop()

    def iter_restored_gpu_frames(
        self,
        input_path,
        *,
        crop_mode: str,
        to_fisheye: bool,
        start_sec: float | None = None,
        end_sec: float | None = None,
        max_clip_length: int = 180,
        log_callback=None,
        cancel_token=None,
    ):
        from gpu_engine.fallback import OperationCancelled
        from gpu_engine.native_mosaic.progress import vram_suffix
        from lada.restorationpipeline.frame_restorer import FrameRestorer
        from lada.utils.threading_utils import STOP_MARKER, ErrorMarker

        frame_source_factory, lada_meta = self._make_gpu_bgr_frame_source_factory(
            input_path,
            crop_mode=crop_mode,
            to_fisheye=to_fisheye,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        restore_limits = _native_restore_limits(
            lada_meta,
            max_clip_length,
            torch_module=self.torch,
            device=self.device,
        )
        if log_callback and restore_limits.reason:
            log_callback(f"[native-stream] VRAM guard: {restore_limits.reason}")
        try:
            warm = getattr(self.restoration_model, "warmup_graph", None)
            if callable(warm):
                warm(int(restore_limits.max_clip_length))
                self.torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            if log_callback:
                log_callback(f"[native-stream] graph warmup skipped: {type(exc).__name__}: {exc}")
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
        frame_restorer = FrameRestorer(
            self.device, input_path, restore_limits.max_clip_length, self.restoration_name,
            self.detection_model, self.restoration_model, self.pad_mode,
            video_meta_data=lada_meta, frame_source_factory=frame_source_factory,
            progress_log_callback=log_callback,
            detector_batch_size=restore_limits.detector_batch_size,
            frame_queue_mb=restore_limits.frame_queue_mb,
            clip_queue_mb=restore_limits.clip_queue_mb,
            detector_queue_size=restore_limits.detector_queue_size,
        )

        n = 0
        try:
            frame_restorer.start()
            for elem in frame_restorer:
                if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                    raise OperationCancelled("cancelled by user")
                if elem is STOP_MARKER:
                    raise RuntimeError("native frame restorer stopped")
                if isinstance(elem, ErrorMarker):
                    _log_error_marker(elem, log_callback or (lambda _m: None))
                    raise RuntimeError(f"native frame restorer error: {elem}")
                n += 1
                if log_callback and n % 30 == 0:
                    total = getattr(lada_meta, "frames_count", 0) or 0
                    suffix = f"/{total}" if total else ""
                    log_callback(f"[native-stream] {crop_mode} restored {n}{suffix} frames{vram_suffix()}")
                yield elem
        finally:
            frame_restorer.stop()

    def _prepare_restored_nv12(self, bgr_frame, *, from_fisheye: bool, out_w: int, out_h: int,
                               profile=None):
        import cupy as cp
        from gpu_engine import nv12_kernels, v360_lut
        from gpu_engine.native_mosaic import _gpu_ops

        _gpu_ops.wait_decode_event(bgr_frame)
        y_plane, uv_plane = self._torch_bgr_to_nv12_cupy(bgr_frame, profile=profile)
        if from_fisheye:
            lut_y = v360_lut.make_lut("fisheye2heq", out_w, out_h)
            lut_c = v360_lut.make_lut("fisheye2heq", out_w // 2, out_h // 2)
            y_plane = nv12_kernels.remap_y(y_plane, lut_y, out_w, out_h)
            uv_plane = nv12_kernels.remap_uv(uv_plane, lut_c, out_w // 2, out_h // 2)
        if profile:
            profile.increment("sync_count")
            profile.increment("prepare_nv12_sync_count")
        with profile.section("sync.prepare_nv12") if profile else nullcontext():
            cp.cuda.get_current_stream().synchronize()
        return y_plane, uv_plane

    def _iter_fisheye_delta_reference_frames(
        self,
        input_path,
        *,
        crop_mode: str,
        start_sec: float | None,
        end_sec: float | None,
        fov: float = 180.0,
    ):
        """Second source decode for fisheye delta write-back.

        Yields (source_heq_bgr, source_fisheye_bgr) torch pairs per frame.  The
        fisheye variant repeats the exact remap/convert op sequence of the
        restorer's frame source (_iter_gpu_bgr_frames*, to_fisheye=True), so
        pixels the restorer passed through cancel to a zero delta bit-exactly.
        """
        from gpu_engine import nv12_kernels, probe, v360_lut
        from gpu_engine.pynv_io import PyNvThreadedSerialDecoder

        meta = probe.probe_video(input_path)
        bd = 10 if meta.bit_depth > 8 else 8
        fps = meta.source_fps or 30.0
        start_idx = max(0, int(round((start_sec or 0.0) * fps)))
        end_idx = int(round(end_sec * fps)) if end_sec is not None else None
        dec = PyNvThreadedSerialDecoder(Path(input_path), bit_depth=bd, start_frame=start_idx)
        try:
            total = len(dec)
            stop_idx = min(end_idx if end_idx is not None else total, total)
            info = dec.info
            if crop_mode == "sbs":
                eye_w, eye_h = info.width // 2, info.height
            else:
                _cx, _cy, eye_w, eye_h = self._crop_region(crop_mode, info.width, info.height)
            lut_y = v360_lut.make_lut("heq2fisheye", eye_w, eye_h, fov)
            lut_c = v360_lut.make_lut("heq2fisheye", eye_w // 2, eye_h // 2, fov)
            for i in range(start_idx, stop_idx):
                frame = dec.frame_at(i)
                y_plane, uv_plane = frame.y_uv_cupy()
                if crop_mode == "sbs":
                    y_src = y_plane[:, :eye_w * 2]
                    uv_src = uv_plane[:, :eye_w, :]
                    ly = nv12_kernels.remap_y(y_plane[:, :eye_w], lut_y, eye_w, eye_h)
                    ry = nv12_kernels.remap_y(y_plane[:, eye_w:eye_w * 2], lut_y, eye_w, eye_h)
                    luv = nv12_kernels.remap_uv(uv_plane[:, :eye_w // 2, :], lut_c, eye_w // 2, eye_h // 2)
                    ruv = nv12_kernels.remap_uv(uv_plane[:, eye_w // 2:eye_w, :], lut_c, eye_w // 2, eye_h // 2)
                    fish_y = nv12_kernels.hstack_planes(ly, ry)
                    fish_uv = nv12_kernels.hstack_planes(luv, ruv)
                else:
                    x, y0, _w, _h = self._crop_region(crop_mode, info.width, info.height)
                    y_src = y_plane[y0:y0 + eye_h, x:x + eye_w]
                    uv_src = uv_plane[y0 // 2:(y0 + eye_h) // 2, x // 2:(x + eye_w) // 2, :]
                    fish_y = nv12_kernels.remap_y(y_src, lut_y, eye_w, eye_h)
                    fish_uv = nv12_kernels.remap_uv(uv_src, lut_c, eye_w // 2, eye_h // 2)
                yield (
                    self._planes_to_torch_bgr(y_src, uv_src, bit_depth=bd),
                    self._planes_to_torch_bgr(fish_y, fish_uv, bit_depth=bd),
                )
        finally:
            dec.stop()

    def _prepare_restored_nv12_sbs(self, bgr_frame, *, from_fisheye: bool, eye_w: int, eye_h: int):
        import cupy as cp
        from gpu_engine import nv12_kernels, v360_lut

        y_plane, uv_plane = self._torch_bgr_to_nv12_cupy(bgr_frame)
        if from_fisheye:
            lut_y = v360_lut.make_lut("fisheye2heq", eye_w, eye_h)
            lut_c = v360_lut.make_lut("fisheye2heq", eye_w // 2, eye_h // 2)
            ly = y_plane[:, :eye_w]
            ry = y_plane[:, eye_w:eye_w * 2]
            luv = uv_plane[:, :eye_w // 2, :]
            ruv = uv_plane[:, eye_w // 2:eye_w, :]
            ly = nv12_kernels.remap_y(ly, lut_y, eye_w, eye_h)
            ry = nv12_kernels.remap_y(ry, lut_y, eye_w, eye_h)
            luv = nv12_kernels.remap_uv(luv, lut_c, eye_w // 2, eye_h // 2)
            ruv = nv12_kernels.remap_uv(ruv, lut_c, eye_w // 2, eye_h // 2)
            y_plane = nv12_kernels.hstack_planes(ly, ry)
            uv_plane = nv12_kernels.hstack_planes(luv, ruv)
        cp.cuda.get_current_stream().synchronize()
        return y_plane, uv_plane

    def _encode_restored_stream(
        self,
        input_path,
        output_path,
        *,
        eye_modes: tuple[str, ...],
        use_fisheye: bool,
        start_sec: float | None = None,
        end_sec: float | None = None,
        bitrate_bps: int | None = None,
        log_callback=None,
        cancel_token=None,
    ) -> bool:
        import cupy as cp
        from gpu_engine import mux, probe, runtime
        from gpu_engine.fallback import OperationCancelled
        from gpu_engine.files import (
            _EncodeSink, _Progress, _encoder_kwargs, _log_encoder_settings,
            _media_temp_path,
            _pack_planes, _resolve_bitrate,
        )
        from gpu_engine.pynv_io import PyNvEncoderSession

        src_meta, decision = probe.route(input_path)
        if not decision.is_gpu:
            raise RuntimeError(f"native stream path requires PyNv-safe source: {decision.reason}")
        if src_meta.is_hdr or src_meta.is_bt2020:
            raise RuntimeError("native stream path is 8-bit SDR only")

        fps = src_meta.source_fps or 30.0
        eye_w = src_meta.width // 2
        eye_h = src_meta.height
        if len(eye_modes) == 2:
            out_w, out_h = eye_w * 2, eye_h
        else:
            out_w, out_h = eye_w, eye_h

        bitrate = _resolve_bitrate(out_w, out_h, fps, bitrate_bps, src_meta.bitrate_bps)
        enc_kwargs = _encoder_kwargs(src_meta, bitrate)
        _log_encoder_settings("native stream restore", out_w, out_h, 8, enc_kwargs, log_callback)
        enc = PyNvEncoderSession(
            out_w, out_h, bit_depth=8, codec="hevc",
            **enc_kwargs,
        )

        raw_path = _media_temp_path(output_path, "native_stream")

        sbs_mode = len(eye_modes) == 2
        use_delta = bool(use_fisheye) and fisheye_delta.enabled()
        reference_frames = None
        if use_delta:
            reference_frames = self._iter_fisheye_delta_reference_frames(
                input_path,
                crop_mode="sbs" if sbs_mode else eye_modes[0],
                start_sec=start_sec,
                end_sec=end_sec,
            )
            if log_callback:
                log_callback(
                    "[native-stream] fisheye delta write-back: untouched pixels keep "
                    "the source projection; only restored regions are reprojected"
                )
        if sbs_mode:
            iters = [
                self.iter_restored_gpu_frames(
                    input_path,
                    crop_mode="sbs",
                    to_fisheye=use_fisheye,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    log_callback=log_callback,
                    cancel_token=cancel_token,
                )
            ]
        else:
            iters = [
                self.iter_restored_gpu_frames(
                    input_path,
                    crop_mode=mode,
                    to_fisheye=use_fisheye,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    log_callback=log_callback,
                    cancel_token=cancel_token,
                )
                for mode in eye_modes
            ]
        written = 0
        total = 0
        if src_meta.nb_frames:
            start_idx = int(round((start_sec or 0.0) * fps))
            end_idx = int(round(end_sec * fps)) if end_sec is not None else src_meta.nb_frames
            total = max(0, min(end_idx, src_meta.nb_frames) - start_idx)
        from gpu_engine.native_mosaic.progress import native_progress_interval_s, native_progress_min_pct

        prog = _Progress(
            total,
            log_callback,
            min_interval=native_progress_interval_s(),
            min_pct=native_progress_min_pct(),
            prefix="[native]",
        )
        try:
            with open(raw_path, "wb") as f:
                sink = _EncodeSink(enc, f)
                while True:
                    if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                        raise OperationCancelled("cancelled by user")
                    try:
                        frame = next(iters[0])[0]
                    except StopIteration:
                        break
                    if use_delta:
                        from gpu_engine.native_mosaic import _gpu_ops

                        _gpu_ops.wait_decode_event(frame)
                        source_heq, source_fish = fisheye_delta.next_reference(reference_frames)
                        frame = fisheye_delta.apply_delta_frame(
                            self.torch, self.device, source_heq, source_fish,
                            frame.to(device=self.device), sbs=sbs_mode,
                        )
                    output_from_fisheye = use_fisheye and not use_delta
                    if sbs_mode:
                        y_plane, uv_plane = self._prepare_restored_nv12_sbs(
                            frame, from_fisheye=output_from_fisheye, eye_w=eye_w, eye_h=eye_h
                        )
                    else:
                        y_plane, uv_plane = self._prepare_restored_nv12(
                            frame, from_fisheye=output_from_fisheye, out_w=eye_w, out_h=eye_h
                        )
                    app = _pack_planes(y_plane, uv_plane, 8)
                    sink.feed(app, force_idr=(written == 0))
                    written += 1
                    prog.update(written)
                prog.finish(written)
                sink.flush()
        except Exception:
            try:
                if raw_path and raw_path.exists():
                    raw_path.unlink()
            except OSError:
                pass
            raise
        finally:
            for it in iters:
                close = getattr(it, "close", None)
                if callable(close):
                    close()
            if reference_frames is not None:
                close = getattr(reference_frames, "close", None)
                if callable(close):
                    close()
            runtime.free_memory_pool()
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass

        audio_duration = (end_sec - (start_sec or 0.0)) if end_sec is not None else None
        mux.mux_hevc_with_audio(
            raw_path,
            output_path,
            fps=fps,
            color=src_meta.color,
            audio_source=input_path,
            audio_start_sec=start_sec,
            audio_duration=audio_duration,
            log_callback=log_callback,
        )
        try:
            raw_path.unlink()
        except OSError:
            pass
        if log_callback:
            log_callback(f"[native-stream] encoded {written} frames -> {output_path}")
        return True

    def restore_sbs_stream(self, input_path, output_path, *, use_fisheye: bool,
                           start_sec: float | None = None, end_sec: float | None = None,
                           bitrate_bps: int | None = None, log_callback=None,
                           cancel_token=None) -> bool:
        return self._encode_restored_stream(
            input_path, output_path,
            eye_modes=("left", "right"),
            use_fisheye=use_fisheye,
            start_sec=start_sec,
            end_sec=end_sec,
            bitrate_bps=bitrate_bps,
            log_callback=log_callback,
            cancel_token=cancel_token,
        )

    def restore_single_eye_stream(self, input_path, output_path, *, eye_mode: str,
                                  use_fisheye: bool, start_sec: float | None = None,
                                  end_sec: float | None = None,
                                  bitrate_bps: int | None = None,
                                  log_callback=None, cancel_token=None) -> bool:
        return self._encode_restored_stream(
            input_path, output_path,
            eye_modes=(eye_mode,),
            use_fisheye=use_fisheye,
            start_sec=start_sec,
            end_sec=end_sec,
            bitrate_bps=bitrate_bps,
            log_callback=log_callback,
            cancel_token=cancel_token,
        )
