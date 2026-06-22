"""CUDA-graph-/TensorRT-friendly modulated deformable convolution (DCNv2).

``torchvision.ops.deform_conv2d`` is a fused native op that is NOT safe to
capture inside a CUDA graph (it triggers a Windows native fast-fail,
STATUS_STACK_BUFFER_OVERRUN 0xc0000409 during capture/replay) and cannot be
expressed by ONNX / TensorRT without a custom DCNv2 plugin. This module
re-implements the exact same math using only standard, capturable aten ops
(``arange`` / ``grid_sample`` / ``conv2d`` / reshape), so the BasicVSR++
restoration model can run under CUDA Graphs, ``torch.compile`` and (later)
in-memory TensorRT without any custom plugin.

The implementation matches the semantics of ``torchvision.ops.deform_conv2d``:
zero padding outside the input, bilinear sampling, ``align_corners`` pixel
coordinates, modulation mask applied to the sampled values. Offset channel
layout follows torchvision: ``((group * kH * kW + (i * kW + j)) * 2 + {0:y,1:x})``;
mask layout: ``group * kH * kW + (i * kW + j)``.

Set ``VRVT_NATIVE_DCN=0`` to fall back to the torchvision op (A/B testing).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

_FALSE = {"0", "false", "no", "off"}


def native_dcn_enabled() -> bool:
    return str(os.environ.get("VRVT_NATIVE_DCN", "1")).strip().lower() not in _FALSE


def deform_conv2d_native(
    input: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch modulated deformable convolution.

    Args:
        input:  (B, C_in, H_in, W_in)
        offset: (B, 2 * dg * kH * kW, H_out, W_out)
        weight: (C_out, C_in // groups, kH, kW)
        bias:   (C_out,) or None
        mask:   (B, dg * kH * kW, H_out, W_out) or None (unmodulated)

    Returns:
        (B, C_out, H_out, W_out)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    b, c_in, h_in, w_in = input.shape
    c_out, c_in_pg, kh, kw = weight.shape
    groups = c_in // c_in_pg
    n_pos = kh * kw
    _, _, h_out, w_out = offset.shape
    dg = offset.shape[1] // (2 * n_pos)

    device, dtype = input.device, input.dtype

    # Base sampling locations (independent of batch): for each kernel tap p and
    # output pixel (oy, ox), the undeformed source coordinate.
    oy = torch.arange(h_out, device=device, dtype=dtype) * stride[0] - padding[0]
    ox = torch.arange(w_out, device=device, dtype=dtype) * stride[1] - padding[1]
    ky = torch.arange(kh, device=device, dtype=dtype) * dilation[0]
    kx = torch.arange(kw, device=device, dtype=dtype) * dilation[1]

    kyy = ky.view(kh, 1).expand(kh, kw).reshape(n_pos)  # (n_pos,)
    kxx = kx.view(1, kw).expand(kh, kw).reshape(n_pos)  # (n_pos,)

    # (n_pos, H_out, W_out): source y depends on output row, x on output col.
    base_y = oy.view(1, h_out, 1).expand(n_pos, h_out, w_out) + kyy.view(n_pos, 1, 1)
    base_x = ox.view(1, 1, w_out).expand(n_pos, h_out, w_out) + kxx.view(n_pos, 1, 1)

    # Split offset into per-(group, tap) y/x deltas, matching torchvision layout.
    off = offset.view(b, dg, n_pos, 2, h_out, w_out)
    off_y = off[:, :, :, 0]  # (B, dg, n_pos, H_out, W_out)
    off_x = off[:, :, :, 1]

    sample_y = base_y.view(1, 1, n_pos, h_out, w_out) + off_y
    sample_x = base_x.view(1, 1, n_pos, h_out, w_out) + off_x

    # Normalize to grid_sample coordinates ([-1, 1], align_corners=True).
    norm_y = sample_y * (2.0 / max(h_in - 1, 1)) - 1.0
    norm_x = sample_x * (2.0 / max(w_in - 1, 1)) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)  # (B, dg, n_pos, H_out, W_out, 2)
    grid = grid.reshape(b * dg * n_pos, h_out, w_out, 2)

    # Expand input per deform group and tap so a single grid_sample covers all.
    c_pg = c_in // dg
    inp = input.view(b, dg, c_pg, h_in, w_in)
    inp = inp.unsqueeze(2).expand(b, dg, n_pos, c_pg, h_in, w_in)
    inp = inp.reshape(b * dg * n_pos, c_pg, h_in, w_in)

    sampled = F.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (B*dg*n_pos, c_pg, H_out, W_out)

    if mask is not None:
        m = mask.view(b, dg, n_pos, 1, h_out, w_out).reshape(b * dg * n_pos, 1, h_out, w_out)
        sampled = sampled * m

    # Reassemble into deformable columns: (B, C_in, n_pos, H_out, W_out) with
    # channel order [group, c_pg] == input channel order.
    sampled = sampled.view(b, dg, n_pos, c_pg, h_out, w_out)
    sampled = sampled.permute(0, 1, 3, 2, 4, 5)  # (B, dg, c_pg, n_pos, H, W)
    cols = sampled.reshape(b, c_in * n_pos, h_out, w_out)

    # deform conv == 1x1 conv over the sampled columns.
    # weight (C_out, C_in_pg, kH, kW) -> (C_out, C_in_pg * n_pos, 1, 1)
    w2 = weight.reshape(c_out, c_in_pg * n_pos, 1, 1)
    if groups == 1:
        out = F.conv2d(cols, w2, bias)
    else:
        # cols channel order is (c_in, n_pos); grouped conv expects contiguous
        # per-group channel blocks, which already holds since c_in is grouped.
        out = F.conv2d(cols, w2, bias, groups=groups)
    return out
