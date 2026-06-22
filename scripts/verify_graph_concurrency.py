"""Reproduce the production 0xc0000409: CUDA-graph capture while ANOTHER thread
runs GPU work. Default capture_error_mode='global' aborts (fast-fail) on
concurrent cross-thread GPU activity; 'thread_local' tolerates it.

Usage: python verify_graph_concurrency.py [global|thread_local]
Exit 0 = capture+replay OK; nonzero / hard crash = the bug.
"""
import sys, os, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

os.environ.setdefault("VRVT_NATIVE_DCN", "1")
import gpu_engine.native_mosaic as nm
nm._prepare()
from lada.models.basicvsrpp.inference import load_model  # noqa: E402
from gpu_engine.native_mosaic import _torch_tuning  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "global"
CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
T, SIZE = 12, 256


def gpu_noise(stop_evt):
    """Mimic the concurrent YOLO detection thread hammering the GPU."""
    a = torch.randn(2048, 2048, device=DEV)
    while not stop_evt.is_set():
        (a @ a).relu_()
    torch.cuda.synchronize()


def main():
    print(f"capture_error_mode = {MODE}")
    model = load_model(None, CKPT, DEV, fp16=True)
    x = _torch_tuning.to_channels_last_5d(
        torch.rand(1, T, 3, SIZE, SIZE, device=DEV, dtype=torch.float16))
    static = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=DEV)
    static.copy_(x)

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.inference_mode(False), torch.no_grad():
            for _ in range(3):
                model(inputs=static)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

    stop = threading.Event()
    noise = threading.Thread(target=gpu_noise, args=(stop,), daemon=True)
    noise.start()
    time.sleep(0.3)  # ensure the other thread is actively launching kernels

    g = torch.cuda.CUDAGraph()
    try:
        with torch.inference_mode(False), torch.no_grad():
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, stream=s, capture_error_mode=MODE):
                    out = model(inputs=static)
        torch.cuda.synchronize()
        print("[capture] OK")
        g.replay(); torch.cuda.synchronize()
        print("[replay] OK")
        print("RESULT: PASS")
    finally:
        stop.set()
        noise.join(timeout=2)


if __name__ == "__main__":
    main()
