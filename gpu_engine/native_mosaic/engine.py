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
from fractions import Fraction
from pathlib import Path


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
        from .models_cfg import detection_model_path, restoration_model_path

        self.torch = torch
        self.device = torch.device("cuda")
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
        self._patch_detection_gpu_preprocess()

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

    def _resolve_encoder(self):
        """Choose an nvenc HEVC encoding preset, falling back to reasonable hevc_nvenc options if the CSV preset is unavailable."""
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
        from lada.restorationpipeline.frame_restorer import FrameRestorer
        from lada.utils import video_utils

        def _log(m):
            if log_callback:
                log_callback(m)

        meta = video_utils.get_video_meta_data(input_path)

        def _make_restorer():
            return FrameRestorer(
                self.device, input_path, max_clip_length, self.restoration_name,
                self.detection_model, self.restoration_model, self.pad_mode,
            )

        gpu_ok = False
        src_meta = None
        try:
            from gpu_engine import probe
            src_meta, decision = probe.route(input_path)
            gpu_ok = bool(decision.is_gpu and not src_meta.is_hdr and not src_meta.is_bt2020)
            if not gpu_ok:
                _log(f"[native] GPU NVENC unavailable ({decision.reason or 'non-SDR/10-bit'}); using VideoWriter")
                if not produce_mp4:
                    _log("[native] raw HEVC restore requested but GPU NVENC path is unavailable; writing mp4")
        except Exception as e:
            _log(f"[native] probe failed, using VideoWriter: {type(e).__name__}: {e}")
            gpu_ok = False

        if gpu_ok:
            try:
                return self._restore_file_gpu_nvenc(
                    _make_restorer(), input_path, output_path, meta, src_meta,
                    bitrate_bps=bitrate_bps,
                    log_callback=log_callback, cancel_token=cancel_token,
                    produce_mp4=produce_mp4,
                    sidecar_metadata=sidecar_metadata,
                )
            except _GpuEncodeSetupError as e:
                _log(f"[native] GPU NVENC setup failed ({e}); fallback to VideoWriter")

        return self._restore_file_videowriter(
            _make_restorer(), input_path, output_path, meta,
            log_callback=log_callback, cancel_token=cancel_token,
        )

    def _restore_file_gpu_nvenc(self, frame_restorer, input_path, output_path, meta, src_meta,
                                *, bitrate_bps: int | None = None,
                                log_callback=None, cancel_token=None,
                                produce_mp4: bool = True,
                                sidecar_metadata: dict | None = None) -> bool:
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
        prog = _Progress(total, log_callback)
        written = 0
        success = True
        try:
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
                        _log("[native] frame restorer stopped prematurely")
                        break
                    restored_frame, _pts = elem
                    y_plane, uv_plane = self._prepare_restored_nv12(
                        restored_frame, from_fisheye=False, out_w=out_w, out_h=out_h
                    )
                    app = _pack_planes(y_plane, uv_plane, 8)
                    sink.feed(app, force_idr=(written == 0))
                    written += 1
                    prog.update(written)
                if success:
                    prog.finish(written)
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
                                  *, log_callback=None, cancel_token=None) -> bool:
        """Fallback to the original Lada VideoWriter encode path for HDR/10-bit/non-PyNv-safe sources."""
        from gpu_engine.files import _media_temp_path
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
        try:
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
                        _log("[native] frame restorer stopped prematurely")
                        break
                    restored_frame, restored_pts = elem
                    writer.write(restored_frame, restored_pts, bgr2rgb=True)
                    n += 1
                    if log_callback and (n % 30 == 0):
                        pct = f" ({100 * n / total:.0f}%)" if total else ""
                        _log(f"[native] restored {n}{('/' + str(total)) if total else ''} frames{pct}")
        except Exception as e:
            success = False
            _log(f"[native] error: {type(e).__name__}: {e}")
        finally:
            frame_restorer.stop()

        if success:
            _log("[native] muxing audio...")
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
        hw, hh = w // 2, h // 2
        return {
            "left": (0, 0, hw, h),
            "right": (hw, 0, hw, h),
            "top": (0, 0, w, hh),
            "bottom": (0, hh, w, hh),
        }[mode]

    @staticmethod
    def _cupy_to_torch(arr):
        import cupy as cp
        import torch

        arr = cp.ascontiguousarray(arr)
        cp.cuda.get_current_stream().synchronize()
        try:
            return torch.utils.dlpack.from_dlpack(arr)
        except TypeError:
            return torch.utils.dlpack.from_dlpack(arr.toDlpack())

    @staticmethod
    def _torch_to_cupy(tensor):
        import cupy as cp

        if not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=False)
        tensor = tensor.contiguous()
        try:
            import torch
            torch.cuda.current_stream(tensor.device).synchronize()
        except Exception:
            pass
        try:
            return cp.from_dlpack(tensor)
        except TypeError:
            return cp.from_dlpack(tensor.detach())

    def _planes_to_torch_bgr(self, y_plane, uv_plane, bit_depth: int = 8):
        """Convert NV12/P010 CuPy planes to BGR8 torch CUDA tensors.

        CuPy RawKernel fuses chroma upsampling with YUV->BGR conversion to avoid
        large temporary tensors from torch repeat_interleave on 4K/8K eye frames.
        """
        from gpu_engine import nv12_kernels

        bgr = nv12_kernels.nv12_to_bgr(y_plane, uv_plane, bit_depth=bit_depth)
        return self._cupy_to_torch(bgr).to(device=self.device)

    def _torch_bgr_to_nv12_cupy(self, bgr_frame):
        """Convert BGR8 torch CUDA tensors to NV12 CuPy planes."""
        from gpu_engine import nv12_kernels

        bgr = bgr_frame.to(device=self.device, dtype=self.torch.uint8)
        return nv12_kernels.bgr_to_nv12(self._torch_to_cupy(bgr))

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
    ):
        from gpu_engine import probe

        meta = probe.probe_video(input_path)
        fps = meta.source_fps or 30.0
        start_idx = max(0, int(round((start_sec or 0.0) * fps)))
        end_idx = int(round(end_sec * fps)) if end_sec is not None else None
        if crop_mode == "sbs":
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
            x, y0, out_w, out_h = self._crop_region(crop_mode, info.width, info.height)
            lut_y = v360_lut.make_lut("heq2fisheye", out_w, out_h, fov) if to_fisheye else None
            lut_c = v360_lut.make_lut("heq2fisheye", out_w // 2, out_h // 2, fov) if to_fisheye else None
            for i in range(start_idx, stop_idx):
                frame = dec.frame_at(i)
                # Plan C key fix: remove per-frame whole-device synchronization
                # with cp.cuda.Device().synchronize(). It waited for another
                # thread's full BasicVSR++ clip (~10-14s) before returning,
                # serializing the multithreaded pipeline. The measured fused
                # path was 0.69fps even though the frame source itself reached
                # 156fps. ThreadedDecoder buffers decode ahead with buffer_size=32,
                # so fetched frame memory is ready. Later CuPy geometry/color work
                # runs on the current stream, and _planes_to_torch_bgr ->
                # _cupy_to_torch synchronizes the current stream before yield,
                # ensuring this frame's CuPy reads finish before PyNv batch memory
                # is reused by the next frame.
                y_plane, uv_plane = frame.y_uv_cupy()
                y_plane = y_plane[y0:y0 + out_h, x:x + out_w]
                uv_plane = uv_plane[y0 // 2:(y0 + out_h) // 2, x // 2:(x + out_w) // 2, :]
                if to_fisheye:
                    y_plane = nv12_kernels.remap_y(y_plane, lut_y, out_w, out_h)
                    uv_plane = nv12_kernels.remap_uv(uv_plane, lut_c, out_w // 2, out_h // 2)
                yield self._planes_to_torch_bgr(y_plane, uv_plane, bit_depth=bd), getattr(frame, "pts", i - start_idx)
        finally:
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
        from lada.restorationpipeline.frame_restorer import FrameRestorer
        from lada.utils.threading_utils import STOP_MARKER, ErrorMarker

        frame_source_factory, lada_meta = self._make_gpu_bgr_frame_source_factory(
            input_path,
            crop_mode=crop_mode,
            to_fisheye=to_fisheye,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        frame_restorer = FrameRestorer(
            self.device, input_path, max_clip_length, self.restoration_name,
            self.detection_model, self.restoration_model, self.pad_mode,
            video_meta_data=lada_meta, frame_source_factory=frame_source_factory,
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
                    raise RuntimeError(f"native frame restorer error: {elem}")
                n += 1
                if log_callback and n % 30 == 0:
                    total = getattr(lada_meta, "frames_count", 0) or 0
                    suffix = f"/{total}" if total else ""
                    log_callback(f"[native-stream] {crop_mode} restored {n}{suffix} frames")
                yield elem
        finally:
            frame_restorer.stop()

    def _prepare_restored_nv12(self, bgr_frame, *, from_fisheye: bool, out_w: int, out_h: int):
        import cupy as cp
        from gpu_engine import nv12_kernels, v360_lut

        y_plane, uv_plane = self._torch_bgr_to_nv12_cupy(bgr_frame)
        if from_fisheye:
            lut_y = v360_lut.make_lut("fisheye2heq", out_w, out_h)
            lut_c = v360_lut.make_lut("fisheye2heq", out_w // 2, out_h // 2)
            y_plane = nv12_kernels.remap_y(y_plane, lut_y, out_w, out_h)
            uv_plane = nv12_kernels.remap_uv(uv_plane, lut_c, out_w // 2, out_h // 2)
        cp.cuda.get_current_stream().synchronize()
        return y_plane, uv_plane

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
        prog = _Progress(total, log_callback)
        try:
            with open(raw_path, "wb") as f:
                sink = _EncodeSink(enc, f)
                while True:
                    if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                        raise OperationCancelled("cancelled by user")
                    try:
                        if sbs_mode:
                            frame = next(iters[0])[0]
                            y_plane, uv_plane = self._prepare_restored_nv12_sbs(
                                frame, from_fisheye=use_fisheye, eye_w=eye_w, eye_h=eye_h
                            )
                        else:
                            frame = next(iters[0])[0]
                            y_plane, uv_plane = self._prepare_restored_nv12(
                                frame, from_fisheye=use_fisheye, out_w=eye_w, out_h=eye_h
                            )
                    except StopIteration:
                        break
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
