"""Validate that padding a clip to a bucket length (repeat-last) and trimming
the output back does not perturb the bidirectional BasicVSR++ result."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

os.environ.setdefault("VRVT_NATIVE_DCN", "1")
import gpu_engine.native_mosaic as nm
nm._prepare()
from lada.models.basicvsrpp.inference import load_model  # noqa: E402

CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
SIZE = 256


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    model = load_model(None, CKPT, DEV, fp16=True)
    for T, bucket in [(23, 32), (5, 8), (47, 64), (100, 128)]:
        torch.manual_seed(T)
        clip = torch.rand(1, T, 3, SIZE, SIZE, device=DEV, dtype=torch.float16)
        with torch.inference_mode():
            ref = model(inputs=clip).float()
            pad = clip[:, -1:].expand(1, bucket - T, 3, SIZE, SIZE)
            padded = torch.cat([clip, pad], dim=1)
            out = model(inputs=padded).float()[:, :T]
        print(f"T={T:3d} -> bucket {bucket:3d}: PSNR(trimmed vs true)={psnr(out, ref):.2f} dB  "
              f"max_abs={ (out-ref).abs().max():.3e}")


if __name__ == "__main__":
    main()
