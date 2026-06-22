"""End-to-end parity: full BasicVSR++ restoration with torchvision DCN vs native.

Runs the real checkpoint over a synthetic clip both ways and reports PSNR /
max-abs. This is the decisive gate for the deform_conv replacement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

import gpu_engine.native_mosaic as nm
nm._prepare()  # adds _vendor to sys.path, warms runtime

from lada.models.basicvsrpp.inference import load_model  # noqa: E402

CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
T = 12
SIZE = 256


def run(fp16: bool):
    dtype = torch.float16 if fp16 else torch.float32
    torch.manual_seed(0)
    clip = torch.rand(1, T, 3, SIZE, SIZE, device=DEV, dtype=dtype)

    outs = {}
    for native in ("1", "0"):
        os.environ["VRVT_NATIVE_DCN"] = native
        model = load_model(None, CKPT, DEV, fp16)
        with torch.inference_mode():
            outs[native] = model(inputs=clip).float()
        del model
        torch.cuda.empty_cache()

    a, b = outs["1"], outs["0"]  # native, torchvision
    diff = (a - b).abs()
    mse = (diff ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    print(f"[{'fp16' if fp16 else 'fp32'}] native-vs-torchvision  "
          f"max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} PSNR={psnr:.2f} dB")
    return psnr


if __name__ == "__main__":
    p32 = run(False)
    p16 = run(True)
    # >50 dB is visually indistinguishable; fp16 will be lower due to rounding.
    ok = p32 > 50 and p16 > 40
    print("RESULT:", "PASS" if ok else "CHECK", f"(fp32 PSNR={p32:.1f}, fp16 PSNR={p16:.1f})")
