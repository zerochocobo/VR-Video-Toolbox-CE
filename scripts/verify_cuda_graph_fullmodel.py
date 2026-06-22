"""Phase A1 gate: capture the whole BasicVSR++ forward in a CUDA graph.

With native DCN this used to crash (0xc0000409). Verifies capture succeeds and
replay matches eager, mirroring CudaGraphRunner's capture logic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

os.environ.setdefault("VRVT_NATIVE_DCN", "1")

import gpu_engine.native_mosaic as nm
nm._prepare()
from lada.models.basicvsrpp.inference import load_model  # noqa: E402
from gpu_engine.native_mosaic import _torch_tuning  # noqa: E402

CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
T, SIZE = 12, 256


def main():
    model = load_model(None, CKPT, DEV, fp16=True)
    clip = torch.rand(1, T, 3, SIZE, SIZE, device=DEV, dtype=torch.float16)
    clip = _torch_tuning.to_channels_last_5d(clip)

    with torch.inference_mode():
        eager = model(inputs=clip).float().clone()

    static_in = torch.empty_strided(clip.shape, clip.stride(), dtype=clip.dtype, device=DEV)
    static_in.copy_(clip)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.inference_mode(False), torch.no_grad():
            for _ in range(3):
                model(inputs=static_in)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(False), torch.no_grad():
        with torch.cuda.graph(g):
            static_out = model(inputs=static_in)
    print("[capture] OK")

    # replay with fresh data
    torch.manual_seed(1)
    clip2 = _torch_tuning.to_channels_last_5d(
        torch.rand(1, T, 3, SIZE, SIZE, device=DEV, dtype=torch.float16))
    with torch.inference_mode():
        eager2 = model(inputs=clip2).float().clone()
    static_in.copy_(clip2)
    g.replay()
    torch.cuda.synchronize()

    diff = (static_out.float() - eager2).abs()
    mse = (diff ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    print(f"[replay] max_abs={diff.max():.3e} PSNR_vs_eager={psnr:.2f} dB")
    print("RESULT:", "PASS" if psnr > 50 else "CHECK")


if __name__ == "__main__":
    main()
