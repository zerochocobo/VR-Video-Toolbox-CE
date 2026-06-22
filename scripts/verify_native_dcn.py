"""Validate _deform_conv_native against torchvision.ops.deform_conv2d.

Checks numerical parity on BasicVSR++ DCN shapes (fp32 + fp16) and that the
native implementation is CUDA-graph capturable (the whole point of the rewrite).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision

from gpu_engine.native_mosaic._deform_conv_native import deform_conv2d_native

dev = torch.device("cuda")
torch.manual_seed(0)

# BasicVSR++ SecondOrderDeformableAlignment params (mid_channels=64).
MID = 64
IN_CH = 2 * MID        # 128
OUT_CH = MID           # 64
K = 3
PAD = 1
DG = 16                # deform_groups
H = W = 64             # feature resolution for 256x256 clip
B = 1                  # loop body runs per-frame


def make_inputs(dtype):
    x = torch.randn(B, IN_CH, H, W, device=dev, dtype=dtype)
    n_pos = K * K
    offset = (torch.rand(B, 2 * DG * n_pos, H, W, device=dev, dtype=dtype) - 0.5) * 6.0
    mask = torch.sigmoid(torch.randn(B, DG * n_pos, H, W, device=dev, dtype=dtype))
    weight = torch.randn(OUT_CH, IN_CH, K, K, device=dev, dtype=dtype) * 0.05
    bias = torch.randn(OUT_CH, device=dev, dtype=dtype) * 0.1
    return x, offset, mask, weight, bias


def compare(dtype, tol):
    x, offset, mask, weight, bias = make_inputs(dtype)
    ref = torchvision.ops.deform_conv2d(x, offset, weight, bias, (1, 1), (PAD, PAD), (1, 1), mask)
    out = deform_conv2d_native(x, offset, weight, bias, 1, PAD, 1, mask)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    print(f"[{str(dtype).split('.')[-1]}] max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} "
          f"max_rel={rel.max():.3e} ref_absmean={ref.float().abs().mean():.3e}")
    return diff.max().item()


def test_cuda_graph():
    dtype = torch.float16
    x, offset, mask, weight, bias = make_inputs(dtype)
    # static buffers
    sx, soff, smask = x.clone(), offset.clone(), mask.clone()

    def run():
        return deform_conv2d_native(sx, soff, weight, bias, 1, PAD, 1, smask)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = run()
    # replay with new data
    sx.copy_(x); soff.copy_(offset); smask.copy_(mask)
    g.replay()
    torch.cuda.synchronize()
    ref = deform_conv2d_native(x, offset, weight, bias, 1, PAD, 1, mask)
    d = (static_out.float() - ref.float()).abs().max().item()
    print(f"[cuda_graph] replay max_abs_vs_eager={d:.3e}")
    return d


if __name__ == "__main__":
    print("== numerical parity vs torchvision ==")
    d32 = compare(torch.float32, 1e-4)
    d16 = compare(torch.float16, 1e-2)
    print("== cuda graph capture ==")
    dg = test_cuda_graph()
    ok = d32 < 5e-3 and dg < 5e-2
    print("RESULT:", "PASS" if ok else "CHECK", f"(fp32 max_abs={d32:.3e}, graph={dg:.3e})")
