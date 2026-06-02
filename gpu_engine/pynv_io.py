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
        probe = PyNvSimpleDecoder(self.src, gpu_id=self.gpu_id, bit_depth=self.bit_depth)
        try:
            self.info = probe.info
            self._len = len(probe)
        finally:
            probe.stop()
        self._decoder = nvc.ThreadedDecoder(
            str(self.src),
            self.buffer_size,
            gpu_id=self.gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.NATIVE,
            start_frame=self.start_frame,
        )
        self._batch: list = []
        self._batch_pos = 0
        self._batch_start_idx = self.start_frame
        self._next_source_idx = self.start_frame
        self._ended = False

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
            current = self._batch_start_idx + self._batch_pos
            raw = self._batch[self._batch_pos]
            self._batch_pos += 1
            self._next_source_idx = current + 1
            if current < target:
                continue
            if current > target:
                raise RuntimeError(f"ThreadedDecoder skipped target: target={target} current={current}")
            if self.bit_depth > 8:
                return GpuP016Frame.from_decoded_frame(raw, self.info.width, self.info.height)
            return GpuNv12Frame.from_decoded_frame(raw, self.info.width, self.info.height)

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
        self._enc = nvc.CreateEncoder(self.width, self.height, self.fmt, False, **kwargs)
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
