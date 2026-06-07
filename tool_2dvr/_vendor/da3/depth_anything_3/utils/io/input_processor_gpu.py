from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _nearest_multiple(value: int, patch: int = PATCH_SIZE) -> int:
    down = (value // patch) * patch
    up = down + patch
    return max(patch, up if abs(up - value) <= abs(value - down) else down)


def _floor_multiple(value: int, patch: int = PATCH_SIZE) -> int:
    return max(patch, (value // patch) * patch)


def _resize_shape(height: int, width: int, target_size: int, method: str) -> tuple[int, int]:
    if method in ("upper_bound_resize", "upper_bound_crop"):
        bound = max(height, width)
    elif method in ("lower_bound_resize", "lower_bound_crop"):
        bound = min(height, width)
    else:
        raise ValueError(f"Unsupported process_res_method: {method}")

    scale = float(target_size) / float(bound) if bound else 1.0
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    if method.endswith("resize"):
        return _nearest_multiple(out_h), _nearest_multiple(out_w)
    if method.endswith("crop"):
        return _floor_multiple(out_h), _floor_multiple(out_w)
    raise ValueError(f"Unsupported process_res_method: {method}")


def gpu_preprocess(
    frames_rgb,
    *,
    device: torch.device,
    target_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> torch.Tensor:
    """Preprocess same-sized RGB uint8 frames directly on CUDA.

    Accepts either:
      - ``list[np.ndarray]`` of CPU RGB uint8 frames (legacy ffmpeg pipe path), or
      - ``torch.Tensor`` shape ``(B, H, W, 3)`` uint8 already resident on CUDA
        (PyNv decode path, no CPU round-trip).

    Returns a DA3-ready tensor with shape ``(1, N, 3, H, W)``, float32.
    """
    if isinstance(frames_rgb, torch.Tensor):
        x = frames_rgb
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B,H,W,3) uint8 tensor, got {tuple(x.shape)}")
        height, width = int(x.shape[1]), int(x.shape[2])
        out_h, out_w = _resize_shape(height, width, int(target_res), process_res_method)
        if x.device != device:
            x = x.to(device=device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32).div_(255.0)
    else:
        if not frames_rgb:
            raise ValueError("gpu_preprocess requires at least one frame")
        first = frames_rgb[0]
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB frames, got {first.shape}")
        height, width = int(first.shape[0]), int(first.shape[1])
        if any(frame.shape[:2] != (height, width) or frame.ndim != 3 or frame.shape[2] != 3 for frame in frames_rgb):
            raise ValueError("gpu_preprocess requires a same-sized RGB frame batch")
        out_h, out_w = _resize_shape(height, width, int(target_res), process_res_method)
        arr = np.ascontiguousarray(np.stack(frames_rgb, axis=0))
        x = torch.from_numpy(arr).to(device=device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32).div_(255.0)

    if (height, width) != (out_h, out_w):
        mode = "bicubic" if out_h > height or out_w > width else "area"
        if mode == "bicubic":
            x = F.interpolate(x, size=(out_h, out_w), mode=mode, align_corners=False)
        else:
            x = F.interpolate(x, size=(out_h, out_w), mode=mode)

    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x.sub_(mean).div_(std)
    return x.unsqueeze(0)
