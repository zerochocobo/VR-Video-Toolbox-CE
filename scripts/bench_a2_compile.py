"""Phase A2 experiment: torch.compile (inductor fusion) UNDER the manual CUDA
graph vs A1 (manual graph only). Reports compile time, ms/clip, correctness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, math
import torch

os.environ.setdefault("VRVT_NATIVE_DCN", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.join(os.getcwd(), "runtime_cache", "inductor"))
import gpu_engine.native_mosaic as nm
nm._prepare()
from lada.models.basicvsrpp.inference import load_model  # noqa: E402
from gpu_engine.native_mosaic import _torch_tuning  # noqa: E402

CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
SIZE, T, ITERS = 256, 30, 30


def clip(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return _torch_tuning.to_channels_last_5d(
        torch.rand(1, T, 3, SIZE, SIZE, generator=g, device=DEV, dtype=torch.float16))


def graph_capture(model, x):
    static = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=DEV)
    static.copy_(x)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.inference_mode(False), torch.no_grad():
            for _ in range(3):
                model(inputs=static)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(False), torch.no_grad():
        with torch.cuda.graph(g):
            out = model(inputs=static)
    return g, static, out


def bench(g):
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def main():
    x = clip(1)
    # A1: manual graph only
    m1 = load_model(None, CKPT, DEV, fp16=True)
    g1, _, out1 = graph_capture(m1, x)
    dt1 = bench(g1)
    print(f"A1 manual-graph        {dt1*1e3:7.2f} ms/clip  {T/dt1:6.1f} fps")
    del m1, g1; torch.cuda.empty_cache()

    # A2: compile generator_ema with inductor (no cudagraph), then manual graph
    m2 = load_model(None, CKPT, DEV, fp16=True)
    t0 = time.perf_counter()
    try:
        m2.generator_ema = torch.compile(m2.generator_ema, fullgraph=False, dynamic=False)
        g2, _, out2 = graph_capture(m2, x)  # triggers compile during warmup
    except Exception as e:
        print(f"A2 compile FAILED: {type(e).__name__}: {e}")
        return
    compile_s = time.perf_counter() - t0
    dt2 = bench(g2)
    psnr = (lambda a, b: (lambda mse: float('inf') if mse == 0 else 10*math.log10(1/mse))(((a-b)**2).mean().item()))(out1.float(), out2.float())
    print(f"A2 compile+graph       {dt2*1e3:7.2f} ms/clip  {T/dt2:6.1f} fps  "
          f"(compile {compile_s:.0f}s, PSNR vs A1 {psnr:.1f} dB)")
    print(f"\nA2 vs A1: {dt1/dt2:.2f}x")


if __name__ == "__main__":
    main()
