"""Opportunistic CUDA tensor offload to pinned host memory.

The native mosaic pipeline can accumulate restored clips while the frame
composer catches up. This module provides a conservative offload boundary for
those already-produced tensors: only when VRAM usage crosses a high watermark,
clip tensors are copied D->H into pinned memory and restored on demand.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _enabled() -> bool:
    return str(os.environ.get("VRVT_VRAM_OFFLOAD", "1")).strip().lower() not in {"0", "false", "no", "off"}


def should_offload(high_watermark: float | None = None) -> bool:
    if not _enabled():
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return False
        used_ratio = 1.0 - (float(free_bytes) / float(total_bytes))
        threshold = _cfg_float("VRVT_VRAM_OFFLOAD_HIGH", 0.80) if high_watermark is None else float(high_watermark)
        return used_ratio >= threshold
    except Exception:
        return False


@dataclass
class OffloadedTensor:
    cpu_tensor: Any
    original_device: Any
    ready_event: Any = None

    @classmethod
    def from_tensor(cls, tensor):
        import torch

        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return tensor
        stream = torch.cuda.current_stream(tensor.device)
        cpu_tensor = torch.empty_strided(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        cpu_tensor.copy_(tensor, non_blocking=True)
        tensor.record_stream(stream)
        event = torch.cuda.Event()
        event.record(stream)
        return cls(cpu_tensor=cpu_tensor, original_device=tensor.device, ready_event=event)

    def to_device(self, device=None):
        if self.ready_event is not None:
            self.ready_event.synchronize()
        target = device or self.original_device
        return self.cpu_tensor.to(target, non_blocking=True)


def restore_tensor(value, device=None):
    if isinstance(value, OffloadedTensor):
        return value.to_device(device=device)
    return value


def offload_tensor(value):
    if isinstance(value, OffloadedTensor):
        return value
    return OffloadedTensor.from_tensor(value)


def maybe_offload_clip(clip, *, high_watermark: float | None = None) -> bool:
    """Offload frames/masks in a restored Clip when VRAM pressure is high."""
    if not should_offload(high_watermark):
        return False
    changed = False
    frames = getattr(clip, "frames", None)
    if isinstance(frames, list):
        clip.frames = [offload_tensor(frame) for frame in frames]
        changed = True
    masks = getattr(clip, "masks", None)
    if isinstance(masks, list):
        clip.masks = [offload_tensor(mask) for mask in masks]
        changed = True
    return changed
