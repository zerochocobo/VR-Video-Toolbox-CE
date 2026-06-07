"""Torch CUDA image ops used by the vendored Lada pipeline."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


CV2_INTER_NEAREST = 0
CV2_INTER_LINEAR = 1


def is_cuda_hwc_tensor(value) -> bool:
    return isinstance(value, torch.Tensor) and value.is_cuda and value.ndim == 3


def attach_decode_event(tensor: torch.Tensor, event) -> torch.Tensor:
    try:
        tensor._decode_event = event
    except Exception:
        pass
    return tensor


def wait_decode_event(value) -> None:
    event = getattr(value, "_decode_event", None)
    if event is None or not isinstance(value, torch.Tensor) or not value.is_cuda:
        return
    stream = torch.cuda.current_stream(value.device)
    stream.wait_event(event)
    try:
        value.record_stream(stream)
        value._decode_event = None
    except Exception:
        pass


def wait_decode_events(values) -> None:
    for value in values:
        wait_decode_event(value)


def crop_to_box_gpu(tensor_hwc: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    t, l, b, r = box
    return tensor_hwc[t:b + 1, l:r + 1, :]


def resize_hwc_gpu(tensor_hwc: torch.Tensor, size: tuple[int, int],
                   interpolation: int = CV2_INTER_LINEAR) -> torch.Tensor:
    """Resize an HWC CUDA tensor with torch interpolation."""
    if not is_cuda_hwc_tensor(tensor_hwc):
        raise TypeError(f"expected CUDA HWC tensor, got {type(tensor_hwc)!r}")
    new_h, new_w = int(size[0]), int(size[1])
    if tuple(tensor_hwc.shape[:2]) == (new_h, new_w):
        return tensor_hwc

    if interpolation == CV2_INTER_LINEAR:
        mode = "bilinear"
        kwargs = {"align_corners": False}
    elif interpolation == CV2_INTER_NEAREST:
        mode = "nearest"
        kwargs = {}
    else:
        raise NotImplementedError(f"unsupported interpolation: {interpolation}")

    dtype = tensor_hwc.dtype
    nchw = tensor_hwc.permute(2, 0, 1).unsqueeze(0)
    work = nchw.float() if not torch.is_floating_point(nchw) else nchw
    resized = F.interpolate(work, size=(new_h, new_w), mode=mode, **kwargs)
    if dtype == torch.uint8:
        resized = resized.round_().clamp_(0, 255).to(dtype=dtype)
    elif resized.dtype != dtype:
        resized = resized.to(dtype=dtype)
    return resized.squeeze(0).permute(1, 2, 0).contiguous()


def _reflect_pad_chw(chw: torch.Tensor, padding: Sequence[int]) -> torch.Tensor:
    remaining = [int(x) for x in padding]
    if any(x < 0 for x in remaining):
        raise ValueError(f"negative padding is not supported: {padding}")
    out = chw
    while any(remaining):
        h, w = out.shape[-2:]
        limits = [max(0, w - 1), max(0, w - 1), max(0, h - 1), max(0, h - 1)]
        step = [min(p, lim) for p, lim in zip(remaining, limits)]
        if not any(step):
            raise ValueError(f"reflect padding requires spatial dimensions > 1, got {tuple(out.shape[-2:])}")
        out = F.pad(out, tuple(step), mode="reflect")
        remaining = [p - s for p, s in zip(remaining, step)]
    return out


def pad_hwc_gpu(tensor_hwc: torch.Tensor, max_height: int, max_width: int,
                mode: str = "zero") -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad an HWC CUDA tensor and return Lada's (top,bottom,left,right) pad tuple."""
    if not is_cuda_hwc_tensor(tensor_hwc):
        raise TypeError(f"expected CUDA HWC tensor, got {type(tensor_hwc)!r}")
    height, width = tensor_hwc.shape[:2]
    if int(height) == int(max_height) and int(width) == int(max_width):
        return tensor_hwc, (0, 0, 0, 0)

    pad_h = int(max_height) - int(height)
    pad_w = int(max_width) - int(width)
    pad_top = math.ceil(pad_h / 2)
    pad_bottom = math.floor(pad_h / 2)
    pad_left = math.ceil(pad_w / 2)
    pad_right = math.floor(pad_w / 2)
    pad = (pad_top, pad_bottom, pad_left, pad_right)

    chw = tensor_hwc.permute(2, 0, 1)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    if mode == "zero":
        padded = F.pad(chw, padding, mode="constant", value=0)
    elif mode == "reflect":
        padded = _reflect_pad_chw(chw, padding)
    else:
        raise NotImplementedError(f"unsupported pad mode: {mode}")
    padded = padded.permute(1, 2, 0).contiguous()
    assert padded.shape[:2] == (int(max_height), int(max_width))
    return padded, pad
