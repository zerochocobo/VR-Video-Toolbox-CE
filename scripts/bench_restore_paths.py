"""Benchmark restoration forward: torchvision-eager (current prod) vs
native-eager vs native+CUDA-graph. Reports ms/clip and fps (frames/s)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch

import gpu_engine.native_mosaic as nm
nm._prepare()
from lada.models.basicvsrpp.inference import load_model  # noqa: E402
from gpu_engine.native_mosaic import _torch_tuning  # noqa: E402

CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
SIZE = 256
T = int(os.environ.get("BENCH_T", "30"))
ITERS = 30


def _clip():
    return _torch_tuning.to_channels_last_5d(
        torch.rand(1, T, 3, SIZE, SIZE, device=DEV, dtype=torch.float16))


def time_eager(native: str):
    os.environ["VRVT_NATIVE_DCN"] = native
    model = load_model(None, CKPT, DEV, fp16=True)
    clip = _clip()
    with torch.inference_mode():
        for _ in range(5):
            model(inputs=clip)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            model(inputs=clip)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / ITERS
    del model; torch.cuda.empty_cache()
    return dt


def time_graph():
    os.environ["VRVT_NATIVE_DCN"] = "1"
    model = load_model(None, CKPT, DEV, fp16=True)
    clip = _clip()
    static_in = torch.empty_strided(clip.shape, clip.stride(), dtype=clip.dtype, device=DEV)
    static_in.copy_(clip)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.inference_mode(False), torch.no_grad():
            for _ in range(3):
                model(inputs=static_in)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(False), torch.no_grad():
        with torch.cuda.graph(g):
            model(inputs=static_in)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        g.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    del model; torch.cuda.empty_cache()
    return dt


def show(name, dt):
    print(f"{name:28s} {dt*1e3:8.2f} ms/clip   {T/dt:8.1f} frames/s")


if __name__ == "__main__":
    print(f"T={T} frames, size={SIZE}, fp16, RTX 5060 Ti")
    tv = time_eager("0"); show("torchvision-eager (prod)", tv)
    nv = time_eager("1"); show("native-eager", nv)
    gr = time_graph();    show("native + CUDA graph", gr)
    print(f"\nspeedup native+graph vs prod: {tv/gr:.2f}x")
