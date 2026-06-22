"""Phase A3 experiment: in-memory torch_tensorrt (no .engine files) for the
BasicVSR++ generator, vs A1 manual CUDA graph and eager. Fixed T (whole-clip
unrolled). Reports build time, ms/clip, and PSNR vs eager."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, math
import torch

os.environ.setdefault("VRVT_NATIVE_DCN", "1")
import gpu_engine.native_mosaic as nm
nm._prepare()
from lada.models.basicvsrpp.inference import load_model  # noqa: E402
from gpu_engine.native_mosaic import _torch_tuning  # noqa: E402

CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
SIZE = 256
T = int(os.environ.get("BENCH_T", "30"))
ITERS = 30


def clip(seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(1, T, 3, SIZE, SIZE, generator=g, device=DEV, dtype=torch.float16)


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def time_call(fn, x):
    with torch.inference_mode():
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def main():
    import torch_tensorrt
    print("torch_tensorrt", torch_tensorrt.__version__)
    x = clip(1)
    model = load_model(None, CKPT, DEV, fp16=True)
    gen = model.generator_ema

    with torch.inference_mode():
        eager_out = gen(x).float().clone()
    dt_eager = time_call(lambda z: gen(z), x)
    print(f"eager            {dt_eager*1e3:8.2f} ms/clip  {T/dt_eager:7.1f} fps")

    free, _ = torch.cuda.mem_get_info()
    t0 = time.perf_counter()
    try:
        trt = torch_tensorrt.compile(
            gen, ir="dynamo", inputs=[x],
            enabled_precisions={torch.float16},
            workspace_size=int(free * 0.8),
            min_block_size=1, truncate_double=True,
            use_python_runtime=False,
            cache_built_engines=False, reuse_cached_engines=False,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"A3 compile FAILED: {type(e).__name__}: {e}")
        return
    build_s = time.perf_counter() - t0
    with torch.inference_mode():
        trt_out = trt(x).float()
    dt_trt = time_call(lambda z: trt(z), x)
    print(f"trt in-memory    {dt_trt*1e3:8.2f} ms/clip  {T/dt_trt:7.1f} fps  "
          f"(build {build_s:.0f}s, PSNR vs eager {psnr(trt_out, eager_out):.1f} dB)")
    print(f"\nA3 trt vs eager: {dt_eager/dt_trt:.2f}x   (A1 manual-graph was ~178 ms / 168 fps)")


if __name__ == "__main__":
    main()
