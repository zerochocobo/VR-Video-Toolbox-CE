"""PyNvVideoCodec decode/encode adapter layer.

Expose GPU NV12/P016 planes decoded by NVDEC as CuPy ndarrays through the CUDA
Array Interface, and wrap CuPy-composed results into AppFrame objects accepted
by the NVENC encoder. PyNv-specific assumptions are kept in this layer; other
modules only see CuPy arrays.

Adapted from reference/PTMediaServer/pipeline/pynv_io.py, with server
configuration coupling removed and these additions:
  - GpuP016AppFrame    : 10-bit P016 encoder input wrapper
  - PyNvEncoderSession : unified NV12 / P016 encoder wrapper that writes raw HEVC bitstreams
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

GpuFrame = Union["GpuNv12Frame", "GpuP016Frame"]


def cuda_device_summary(gpu_id: int = 0) -> str:
    """Return a compact CUDA device summary for diagnostic logs."""
    try:
        import cupy as cp

        gpu = int(gpu_id)
        props = cp.cuda.runtime.getDeviceProperties(gpu)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        major = int(props.get("major", 0))
        minor = int(props.get("minor", 0))
        with cp.cuda.Device(gpu):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        driver = cp.cuda.runtime.driverGetVersion()
        runtime = cp.cuda.runtime.runtimeGetVersion()
        return (
            f"gpu_id={gpu} name={name} cc={major}.{minor} "
            f"vram={total_bytes / (1024 ** 3):.1f}GB free={free_bytes / (1024 ** 3):.1f}GB "
            f"driver={driver} runtime={runtime}"
        )
    except Exception as exc:
        return f"gpu_id={gpu_id} unavailable: {type(exc).__name__}: {exc}"


def _hidden_subprocess_kwargs() -> dict:
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        return {"startupinfo": startupinfo}
    return {}


@functools.lru_cache(maxsize=64)
def _ffprobe_keyframe_times_cached(path: str, mtime_ns: int, size: int) -> tuple[float, ...]:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_frames",
        "-show_entries", "frame=pts_time,best_effort_timestamp_time",
        "-of", "json",
        path,
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **_hidden_subprocess_kwargs())
        data = json.loads(raw)
    except Exception:
        return ()
    out: list[float] = []
    for frame in data.get("frames", []):
        value = frame.get("pts_time") or frame.get("best_effort_timestamp_time")
        try:
            ts = float(value)
        except Exception:
            continue
        if ts >= 0:
            out.append(ts)
    return tuple(sorted(set(out)))


def _keyframe_times_for_path(src: Path) -> tuple[float, ...]:
    try:
        path = Path(src).resolve()
        stat = path.stat()
    except OSError:
        return ()
    return _ffprobe_keyframe_times_cached(str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _frame_timestamp_int(frame: dict, *names: str) -> int | None:
    for name in names:
        value = frame.get(name)
        if value in (None, "", "N/A"):
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=64)
def _ffprobe_first_frame_pts_cached(path: str, mtime_ns: int, size: int) -> int | None:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-read_intervals", "%+#1",
        "-show_frames",
        "-show_entries", "frame=pts,best_effort_timestamp",
        "-of", "json",
        path,
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **_hidden_subprocess_kwargs())
        data = json.loads(raw)
    except Exception:
        return None
    for frame in data.get("frames", []):
        pts = _frame_timestamp_int(frame, "best_effort_timestamp", "pts")
        if pts is not None and pts >= 0:
            return pts
    return None


def _first_frame_pts_for_path(src: Path) -> int | None:
    try:
        path = Path(src).resolve()
        stat = path.stat()
    except OSError:
        return None
    return _ffprobe_first_frame_pts_cached(str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _pts_match_tolerance_from_probe(
    probe: "PyNvSimpleDecoder",
    frame_idx: int,
    expected_pts: int,
    total_frames: int,
) -> int:
    """Allow small timestamp residuals without accepting a whole-frame mismatch."""
    neighbors: list[int] = []
    idx = int(frame_idx)
    total = int(total_frames)
    if idx + 1 < total:
        neighbors.append(idx + 1)
    if idx - 1 >= 0:
        neighbors.append(idx - 1)

    steps: list[int] = []
    for neighbor in neighbors:
        try:
            frame = probe.frame_at(neighbor)
            pts = int(getattr(frame, "pts", -1))
        except Exception:
            continue
        step = abs(pts - int(expected_pts))
        if step > 0:
            steps.append(step)
    if not steps:
        return 0
    frame_step = min(steps)
    if frame_step < 10:
        return 0
    return max(1, int(round(frame_step * 0.10)))


def _threaded_decoder_preroll_frame(
    src: Path,
    target_frame: int,
    fps: float,
    total_frames: int,
    *,
    index_at_time=None,
) -> int:
    """Return the keyframe frame that ThreadedDecoder should start from.

    PyNv ThreadedDecoder can return frames from the previous keyframe when
    started at an arbitrary inter frame. We start at the preceding keyframe and
    let frame_at(target_frame) discard the preroll frames explicitly.
    """
    target = max(0, int(target_frame))
    total = max(0, int(total_frames))
    if target <= 0:
        return 0
    if total > 0:
        target = min(target, total - 1)
    keyframes = _keyframe_times_for_path(Path(src))
    if not keyframes:
        return target
    fps = float(fps or 0.0)
    if fps <= 0:
        return target

    best: int | None = None
    for ts in keyframes:
        try:
            idx = int(index_at_time(float(ts))) if index_at_time is not None else int(round(float(ts) * fps))
        except Exception:
            idx = int(round(float(ts) * fps))
        if total > 0:
            idx = max(0, min(idx, total - 1))
        else:
            idx = max(0, idx)
        if idx <= target:
            best = idx
        else:
            break
    return target if best is None else best


@dataclass(frozen=True)
class CudaPlane:
    """One CUDA Array Interface plane belonging to a PyNvVideoCodec DecodedFrame."""

    view: Any
    owner: Any
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    ptr: int
    readonly: bool

    @classmethod
    def from_view(cls, view: Any, owner: Any) -> "CudaPlane":
        cai = getattr(view, "__cuda_array_interface__", None)
        if not cai:
            raise TypeError(f"object does not expose CUDA Array Interface: {type(view)!r}")
        data = cai.get("data")
        if not data or not isinstance(data, tuple):
            raise TypeError(f"invalid CUDA Array Interface data field: {data!r}")
        return cls(
            view=view,
            owner=owner,
            shape=tuple(cai["shape"]),
            strides=tuple(cai.get("strides") or ()),
            dtype=str(cai["typestr"]),
            ptr=int(data[0]),
            readonly=bool(data[1]),
        )

    @classmethod
    def from_cupy_array(cls, arr: Any) -> "CudaPlane":
        return cls.from_view(arr, arr)

    @property
    def nbytes(self) -> int:
        if not self.shape:
            return 0
        if self.strides:
            return max(1, self.shape[0] * self.row_stride_bytes)
        n = 1
        for dim in self.shape:
            n *= dim
        return n * self.itemsize

    @property
    def itemsize(self) -> int:
        if self.dtype in {"|u1", "uint8", "u1"}:
            return 1
        if self.dtype in {"<u2", ">u2", "|u2", "uint16", "u2"}:
            return 2
        if self.dtype and self.dtype[-1:].isdigit():
            try:
                return int(self.dtype[-1])
            except Exception:
                pass
        return 1

    @property
    def row_stride_bytes(self) -> int:
        if self.strides:
            stride = int(self.strides[0])
            # PyNvVideoCodec reports P016/P010 plane strides as element counts,
            # while CUDA Array Interface normally uses byte strides.
            if self._strides_are_elements():
                return stride * self.itemsize
            return stride
        width = int(self.shape[1]) if len(self.shape) > 1 else int(self.shape[0] if self.shape else 0)
        return width * self.itemsize

    def _strides_are_elements(self) -> bool:
        return bool(self.itemsize > 1 and self.strides and any(int(s) % self.itemsize for s in self.strides))

    @property
    def cupy_strides(self) -> tuple[int, ...] | None:
        if not self.strides:
            return None
        if self.itemsize <= 1:
            return self.strides
        if self._strides_are_elements():
            return tuple(int(s) * self.itemsize for s in self.strides)
        return self.strides

    def as_cupy(self, dtype=None):
        """Return a zero-copy CuPy ndarray view of this plane."""
        import cupy as cp

        cp_dtype = dtype or (cp.uint16 if self.itemsize == 2 else cp.uint8)
        mem = cp.cuda.UnownedMemory(self.ptr, self.nbytes, self.owner)
        mp = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray(self.shape, dtype=cp_dtype, memptr=mp, strides=self.cupy_strides)


@dataclass(frozen=True)
class GpuNv12Frame:
    """GPU-resident NV12 (8-bit) frame decoded by PyNvVideoCodec."""

    owner: Any
    y: CudaPlane
    uv: CudaPlane
    width: int
    height: int
    pts: int
    bit_depth: int = 8

    @classmethod
    def from_decoded_frame(cls, frame: Any, width: int, height: int) -> "GpuNv12Frame":
        planes = frame.cuda()
        if len(planes) < 2:
            raise RuntimeError(f"expected at least 2 NV12 planes, got {len(planes)}")
        return cls(
            owner=frame,
            y=CudaPlane.from_view(planes[0], frame),
            uv=CudaPlane.from_view(planes[1], frame),
            width=width,
            height=height,
            pts=int(frame.getPTS()),
        )

    def y_uv_cupy(self):
        """Return CuPy views (Y[h,w], UV[h//2,w//2,2]) with interleaved Cb/Cr UV."""
        import cupy as cp

        h, w = int(self.height), int(self.width)
        y = self.y.as_cupy(cp.uint8).reshape(h, w)
        uv = self.uv.as_cupy(cp.uint8).reshape(h // 2, w // 2, 2)
        return y, uv

    def owned_copy(self) -> "GpuNv12Frame":
        """Copy PyNv-owned planes into CuPy-owned device memory.

        ThreadedDecoder batch frames are valid only for a short lifetime. When a
        frame goes through multiple CUDA processing steps, it must be copied out
        first instead of continuing to read PyNv-managed batch memory.
        """
        import cupy as cp

        h, w = int(self.height), int(self.width)
        if tuple(self.y.shape[:2]) != (h, w):
            raise RuntimeError(f"unexpected NV12 Y plane shape: frame={w}x{h} y_shape={self.y.shape}")
        uv_shape = tuple(self.uv.shape)
        if uv_shape not in {(h // 2, w), (h // 2, w // 2, 2)}:
            raise RuntimeError(f"unexpected NV12 UV plane shape: frame={w}x{h} uv_shape={self.uv.shape}")
        cp.cuda.Device().synchronize()
        y_src = self.y.as_cupy(cp.uint8).reshape(h, w)
        uv_src = self.uv.as_cupy(cp.uint8).reshape(h // 2, w)
        y = cp.ascontiguousarray(y_src)
        uv = cp.ascontiguousarray(uv_src)
        cp.cuda.get_current_stream().synchronize()
        owner = (y, uv)
        return GpuNv12Frame(
            owner=owner,
            y=CudaPlane.from_cupy_array(y),
            uv=CudaPlane.from_cupy_array(uv),
            width=w,
            height=h,
            pts=self.pts,
        )


@dataclass(frozen=True)
class GpuP016Frame:
    """GPU-resident 10-bit 4:2:0 frame decoded into P016/P010 planes."""

    owner: Any
    y: CudaPlane
    uv: CudaPlane
    width: int
    height: int
    pts: int
    bit_depth: int = 10

    @classmethod
    def from_decoded_frame(cls, frame: Any, width: int, height: int) -> "GpuP016Frame":
        planes = frame.cuda()
        if len(planes) < 2:
            raise RuntimeError(f"expected at least 2 P016 planes, got {len(planes)}")
        y = CudaPlane.from_view(planes[0], frame)
        uv = CudaPlane.from_view(planes[1], frame)
        if y.itemsize < 2 or uv.itemsize < 2:
            raise RuntimeError(f"expected uint16 P016 planes, got y={y.dtype} uv={uv.dtype}")
        return cls(
            owner=frame,
            y=y,
            uv=uv,
            width=width,
            height=height,
            pts=int(frame.getPTS()),
        )

    def y_uv_cupy(self):
        """Return CuPy uint16 views (Y[h,w], UV[h//2,w//2,2]) with interleaved Cb/Cr UV."""
        import cupy as cp

        h, w = int(self.height), int(self.width)
        y = self.y.as_cupy(cp.uint16).reshape(h, w)
        uv = self.uv.as_cupy(cp.uint16).reshape(h // 2, w // 2, 2)
        return y, uv

    def owned_copy(self) -> "GpuP016Frame":
        """Copy PyNv-owned 16-bit planes into CuPy-owned device memory."""
        import cupy as cp

        h, w = int(self.height), int(self.width)
        if tuple(self.y.shape[:2]) != (h, w):
            raise RuntimeError(f"unexpected P016 Y plane shape: frame={w}x{h} y_shape={self.y.shape}")
        uv_shape = tuple(self.uv.shape)
        if uv_shape not in {(h // 2, w), (h // 2, w // 2, 2)}:
            raise RuntimeError(f"unexpected P016 UV plane shape: frame={w}x{h} uv_shape={self.uv.shape}")
        cp.cuda.Device().synchronize()
        y_src = self.y.as_cupy(cp.uint16).reshape(h, w)
        uv_src = self.uv.as_cupy(cp.uint16).reshape(h // 2, w)
        y = cp.ascontiguousarray(y_src)
        uv = cp.ascontiguousarray(uv_src)
        cp.cuda.get_current_stream().synchronize()
        owner = (y, uv)
        return GpuP016Frame(
            owner=owner,
            y=CudaPlane.from_cupy_array(y),
            uv=CudaPlane.from_cupy_array(uv),
            width=w,
            height=h,
            pts=self.pts,
        )


@dataclass(frozen=True)
class PyNvVideoInfo:
    """Basic video metadata reported by PyNvVideoCodec.SimpleDecoder."""
    width: int
    height: int
    fps: float
    duration: float
    codec_name: str
    bitrate: float
    num_frames: int
    bit_depth: int = 8


class PyNvSimpleDecoder:
    """Thin wrapper around PyNvVideoCodec.SimpleDecoder for random access and sequential reads."""

    def __init__(self, src: Path, gpu_id: int = 0, bit_depth: int = 8):
        import PyNvVideoCodec as nvc

        self.src = Path(src).resolve()
        self.gpu_id = gpu_id
        self.bit_depth = int(bit_depth or 8)
        self._decoder = nvc.SimpleDecoder(
            str(self.src),
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.NATIVE,
        )
        meta = self._decoder.get_stream_metadata()
        self.info = PyNvVideoInfo(
            width=int(meta.width),
            height=int(meta.height),
            fps=float(meta.average_fps),
            duration=float(meta.duration),
            codec_name=str(meta.codec_name),
            bitrate=float(meta.bitrate),
            num_frames=int(meta.num_frames),
            bit_depth=self.bit_depth,
        )

    def __len__(self) -> int:
        return len(self._decoder)

    def index_at_time(self, seconds: float) -> int:
        """Map seconds to the nearest frame index."""
        try:
            return int(self._decoder.get_index_from_time_in_seconds(float(seconds)))
        except Exception:
            fps = self.info.fps or 30.0
            return max(0, int(round(float(seconds) * fps)))

    def frame_at(self, index: int) -> GpuFrame:
        frame = self._decoder[index]
        if self.bit_depth > 8:
            return GpuP016Frame.from_decoded_frame(frame, self.info.width, self.info.height)
        return GpuNv12Frame.from_decoded_frame(frame, self.info.width, self.info.height)

    def iter_frames(self, start: int = 0, stop: int | None = None):
        """Yield frames from the [start, stop) interval in order."""
        n = len(self)
        end = n if stop is None else min(int(stop), n)
        for i in range(int(start), end):
            yield self.frame_at(i)

    def stop(self) -> None:
        stop = getattr(self._decoder, "stop", None)
        if callable(stop):
            try:
                stop()
            except AttributeError:
                pass


class PyNvThreadedSerialDecoder:
    """Sequential ThreadedDecoder wrapper for monotonically increasing source-frame access.

    ThreadedDecoder batch frames are valid only until the next get_batch_frames()
    call. This wrapper never caches frames across threads, and callers must
    consume the returned frame before calling frame_at() again. It is much faster
    than SimpleDecoder random access; on 8K tests, decode stopped being the
    bottleneck. Adapted from reference/PTMediaServer.
    """

    def __init__(
        self,
        src: Path,
        gpu_id: int = 0,
        bit_depth: int = 8,
        start_frame: int = 0,
        batch_size: int = 8,
        buffer_size: int = 32,
    ):
        import PyNvVideoCodec as nvc

        self.src = Path(src).resolve()
        self.gpu_id = int(gpu_id)
        self.bit_depth = int(bit_depth or 8)
        self.batch_size = max(1, int(batch_size))
        self.buffer_size = max(1, int(buffer_size))
        self.start_frame = max(0, int(start_frame))
        # Self-check: when start_frame > 0 we record the PTS that a fresh random-
        # access decoder reports for that exact frame. ThreadedDecoder itself is
        # started from a keyframe preroll point; after frame_at() discards preroll
        # frames, this PTS check catches any remaining seek/state mismatch before
        # wrong content is encoded downstream.
        self._expected_first_pts: int | None = None
        self._pts_origin_delta = 0
        self._threaded_pts_delta = 0
        self._pts_match_tolerance = 0
        self._preroll_pts_to_frame: dict[int, int] = {}
        probe = PyNvSimpleDecoder(self.src, gpu_id=self.gpu_id, bit_depth=self.bit_depth)
        try:
            self.info = probe.info
            self._len = len(probe)
            self._decode_start_frame = self.start_frame
            try:
                simple_origin = int(getattr(probe.frame_at(0), "pts", -1))
            except Exception:
                simple_origin = -1
            stream_origin = _first_frame_pts_for_path(self.src)
            if simple_origin >= 0 and stream_origin is not None:
                self._pts_origin_delta = int(stream_origin) - simple_origin
                self._threaded_pts_delta = self._pts_origin_delta
            if 0 < self.start_frame < self._len:
                try:
                    probe_frame = probe.frame_at(self.start_frame)
                    pts_value = int(getattr(probe_frame, "pts", -1))
                    if pts_value >= 0:
                        self._expected_first_pts = pts_value
                        self._pts_match_tolerance = _pts_match_tolerance_from_probe(
                            probe,
                            self.start_frame,
                            pts_value,
                            self._len,
                        )
                except Exception:
                    self._expected_first_pts = None
                    self._pts_match_tolerance = 0
                self._decode_start_frame = _threaded_decoder_preroll_frame(
                    self.src,
                    self.start_frame,
                    self.info.fps,
                    self._len,
                    index_at_time=probe.index_at_time,
                )
                if self._decode_start_frame < self.start_frame:
                    probe_end = min(self._len, self._decode_start_frame + 32)
                    for frame_idx in range(self._decode_start_frame, probe_end):
                        try:
                            preroll_frame = probe.frame_at(frame_idx)
                            pts_value = int(getattr(preroll_frame, "pts", -1))
                        except Exception:
                            continue
                        if pts_value >= 0:
                            self._preroll_pts_to_frame.setdefault(pts_value, frame_idx)
        finally:
            probe.stop()
        self._decoder = nvc.ThreadedDecoder(
            str(self.src),
            self.buffer_size,
            gpu_id=self.gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.NATIVE,
            start_frame=self._decode_start_frame,
        )
        self._batch: list = []
        self._batch_pos = 0
        self._batch_start_idx = self._decode_start_frame
        self._next_source_idx = self._decode_start_frame
        self._ended = False
        self._first_frame_verified = (self.start_frame == 0)
        self._initial_batch_calibrated = False

    def __len__(self) -> int:
        return self._len

    def frame_at(self, index: int) -> GpuFrame:
        if self._ended:
            raise RuntimeError("ThreadedDecoder has already ended")
        target = int(index)
        if target < self._next_source_idx:
            raise ValueError(
                f"serial decoder only supports monotonic access: "
                f"target={target} next={self._next_source_idx}"
            )
        while True:
            if self._batch_pos >= len(self._batch):
                self._batch = []
                self._batch_pos = 0
                self._batch_start_idx = self._next_source_idx
                batch = self._decoder.get_batch_frames(self.batch_size)
                if not batch:
                    raise RuntimeError(f"ThreadedDecoder returned no frames at idx={self._next_source_idx}")
                self._batch = list(batch)
                self._calibrate_initial_batch()
            current = self._batch_start_idx + self._batch_pos
            raw = self._batch[self._batch_pos]
            self._batch_pos += 1
            self._next_source_idx = current + 1
            if current < target:
                continue
            if current > target:
                raise RuntimeError(f"ThreadedDecoder skipped target: target={target} current={current}")
            if self.bit_depth > 8:
                frame = GpuP016Frame.from_decoded_frame(raw, self.info.width, self.info.height)
            else:
                frame = GpuNv12Frame.from_decoded_frame(raw, self.info.width, self.info.height)
            if not self._first_frame_verified:
                self._verify_first_frame_pts(frame)
                self._first_frame_verified = True
            return frame

    def _calibrate_initial_batch(self) -> None:
        if self._initial_batch_calibrated:
            return
        self._initial_batch_calibrated = True
        if not self._batch or self._batch_pos != 0 or not self._preroll_pts_to_frame:
            return
        raw = self._batch[0]
        get_pts = getattr(raw, "getPTS", None)
        if not callable(get_pts):
            return
        try:
            actual_pts = int(get_pts())
        except Exception:
            return
        actual_frame = None
        if self._pts_origin_delta:
            actual_frame = self._preroll_pts_to_frame.get(actual_pts - self._pts_origin_delta)
            if actual_frame is not None:
                self._threaded_pts_delta = self._pts_origin_delta
        if actual_frame is None:
            actual_frame = self._preroll_pts_to_frame.get(actual_pts)
            if actual_frame is not None:
                self._threaded_pts_delta = 0
        if actual_frame is None:
            return
        if actual_frame < self._decode_start_frame:
            return
        self._batch_start_idx = actual_frame
        self._next_source_idx = actual_frame

    def _verify_first_frame_pts(self, frame: GpuFrame) -> None:
        """Compare the first delivered frame's PTS with the SimpleDecoder probe.

        Mismatch means the decoded target frame does not match the random-access
        probe. Raise so the caller surfaces the bad content instead of encoding
        garbage downstream.
        """
        if self._expected_first_pts is None:
            return
        actual = int(getattr(frame, "pts", -1))
        if actual < 0:
            return
        if actual == self._expected_first_pts:
            return
        normalized = actual - self._threaded_pts_delta
        if normalized == self._expected_first_pts:
            return
        normalized_delta = normalized - self._expected_first_pts
        if self._pts_match_tolerance > 0 and abs(normalized_delta) <= self._pts_match_tolerance:
            return
        # PTS diff can hint at the keyframe distance (90kHz timebase: 1s = 90000;
        # ms timebase: 1s = 1000). Surface the raw delta and let logs disambiguate.
        delta = actual - self._expected_first_pts
        raise RuntimeError(
            "PyNvThreadedSerialDecoder NVDEC seek check failed: "
            f"start_frame={self.start_frame}, expected_pts={self._expected_first_pts}, "
            f"got_pts={actual} (delta={delta}, normalized_delta={normalized_delta}, "
            f"pts_origin_delta={self._threaded_pts_delta}, pts_tolerance={self._pts_match_tolerance}), "
            f"decode_start_frame="
            f"{getattr(self, '_decode_start_frame', self.start_frame)}. "
            "Likely decoder seek/pre-roll mismatch or concurrent NVDEC/CUDA "
            "pollution; serialize NVDEC use and keep threaded decode starts "
            "keyframe-aligned."
        )

    def iter_frames(self, start: int = 0, stop: int | None = None):
        n = len(self)
        end = n if stop is None else min(int(stop), n)
        for i in range(int(start), end):
            yield self.frame_at(i)

    def stop(self) -> None:
        if self._ended:
            return
        self._batch = []
        self._batch_pos = 0
        end = getattr(self._decoder, "end", None)
        if callable(end):
            end()
        self._ended = True


class CudaArrayView:
    """Expose a CuPy array slice to PyNv through CUDA Array Interface.

    typestr_override: PyNv encoders only accept PyNvVideoCodec's own typestr
    values: "|u1" for 8-bit and "|u2" for 10-bit P010, byte-order independent.
    CuPy uint16 defaults to "<u2" and is rejected, so this parameter overrides
    the CAI typestr when PyNv expects a different value.
    """

    def __init__(self, arr: Any, typestr_override: str | None = None):
        self.arr = arr
        self.typestr_override = typestr_override

    @property
    def __cuda_array_interface__(self):
        cai = dict(self.arr.__cuda_array_interface__)
        cai["shape"] = tuple(cai["shape"])
        if cai.get("strides") is not None:
            cai["strides"] = tuple(cai["strides"])
        if self.typestr_override:
            cai["typestr"] = self.typestr_override
        return cai


class GpuNv12AppFrame:
    """GPU NV12 input wrapper accepted by the PyNvVideoCodec encoder.

    nv12_dev is a contiguous uint8 CuPy array shaped (h*3//2, w).
    """

    def __init__(self, nv12_dev: Any, width: int, height: int):
        self.nv12_dev = nv12_dev
        self.width = int(width)
        self.height = int(height)
        self.y = CudaArrayView(nv12_dev[: self.height, :].reshape(self.height, self.width, 1))
        self.uv = CudaArrayView(nv12_dev[self.height :, :].reshape(self.height // 2, self.width // 2, 2))

    def cuda(self):
        return [self.y, self.uv]


class GpuP016AppFrame:
    """GPU P010 (10-bit) input wrapper accepted by the PyNvVideoCodec encoder.

    p016_dev is a contiguous uint16 CuPy array shaped (h*3//2, w), with Y on top
    and interleaved UV below.

    The PyNv encoder expects the exact same plane layout as its decoded P016
    frames: Y=(h,w,1), UV=(h/2,w/2,2), and typestr "|u2" (uint16). Use uint16
    views directly and override typestr.
    """

    def __init__(self, p016_dev: Any, width: int, height: int):
        import cupy as cp

        self.p016_dev = cp.ascontiguousarray(p016_dev)
        self.width = int(width)
        self.height = int(height)
        h, w = self.height, self.width
        self.y = CudaArrayView(self.p016_dev[:h, :].reshape(h, w, 1), typestr_override="|u2")
        self.uv = CudaArrayView(self.p016_dev[h:, :].reshape(h // 2, w // 2, 2), typestr_override="|u2")

    def cuda(self):
        return [self.y, self.uv]


def make_app_frame(packed_dev: Any, width: int, height: int, bit_depth: int):
    """Choose an NV12 or P016 AppFrame based on bit depth."""
    if bit_depth > 8:
        return GpuP016AppFrame(packed_dev, width, height)
    return GpuNv12AppFrame(packed_dev, width, height)


class PyNvEncoderSession:
    """Unified NVENC encoder wrapper that consumes GPU NV12/P016 frames and emits raw HEVC bitstreams.

    Usage:
        enc = PyNvEncoderSession(w, h, bit_depth=10, fps=59.94, **enc_kwargs)
        for app_frame in frames:
            data = enc.encode(app_frame, force_idr=(i == 0))
            if data: raw.write(data)
        tail = enc.flush()
        if tail: raw.write(tail)
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        bit_depth: int = 8,
        codec: str = "hevc",
        use_cuda_graph: bool | None = None,
        **enc_kwargs: str,
    ):
        import PyNvVideoCodec as nvc

        self._nvc = nvc
        self.width = int(width)
        self.height = int(height)
        self.bit_depth = int(bit_depth or 8)
        # NVENC's 10-bit input format name is "P010", matching the NVDEC P016
        # surface layout: a 16-bit container with 10 valid bits in the high bits.
        # "P016" is not a valid encoder input format name.
        self.fmt = "P010" if self.bit_depth > 8 else "NV12"
        kwargs = {"codec": codec, **{k: str(v) for k, v in enc_kwargs.items()}}
        if use_cuda_graph is None:
            raw = os.environ.get("VRVT_NVENC_CUDA_GRAPH")
            use_cuda_graph = str(raw or "").strip().lower() in {"1", "true", "yes", "on"}
        self._enc = nvc.CreateEncoder(self.width, self.height, self.fmt, bool(use_cuda_graph), **kwargs)
        self._frame_index = 0

    def encode(self, app_frame: Any, *, force_idr: bool = False) -> bytes:
        nvc = self._nvc
        flags = 0
        if force_idr or self._frame_index == 0:
            flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
        self._frame_index += 1
        bs = self._enc.Encode(app_frame, flags) if flags else self._enc.Encode(app_frame)
        return bytes(bs) if bs else b""

    def flush(self) -> bytes:
        end = getattr(self._enc, "EndEncode", None)
        if callable(end):
            tail = end()
            return bytes(tail) if tail else b""
        return b""
