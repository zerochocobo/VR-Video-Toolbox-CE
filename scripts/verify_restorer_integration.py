"""Integration test for the warmup-gated CUDA-graph model.

Verifies:
  - warmup_graph captures the warmed clip length (single-threaded),
  - that length REPLAYS at restore time (speedup path active), bit-exact,
  - an UNWARMED length runs eager at restore time (no runtime capture), bit-exact,
  - zero capture failures, and no capture happens during restore().
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

os.environ.setdefault("VRVT_NATIVE_DCN", "1")
os.environ.setdefault("VRVT_CUDA_GRAPH", "1")
import gpu_engine.native_mosaic as nm
nm._prepare()
from lada.models.basicvsrpp.inference import load_model  # noqa: E402
from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer  # noqa: E402

CKPT = os.path.join("models", "lada_mosaic_restoration_model_generic_v1.2.pth")
DEV = torch.device("cuda")
SIZE = 256
WARM_T = 30
UNWARM_T = 23


def make_clip(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randint(0, 256, (SIZE, SIZE, 3), generator=g, device=DEV, dtype=torch.uint8)
            for _ in range(T)]


def eager_reference(model, clip):
    # Pure eager (graph disabled) reference.
    os.environ["VRVT_CUDA_GRAPH"] = "0"
    r = BasicvsrppMosaicRestorer(model, DEV, fp16=True)
    out = torch.stack(r.restore(clip)).float()
    os.environ["VRVT_CUDA_GRAPH"] = "1"
    return out


def main():
    model = load_model(None, CKPT, DEV, fp16=True)

    ref_warm = eager_reference(model, make_clip(WARM_T, 1))
    ref_unwarm = eager_reference(model, make_clip(UNWARM_T, 2))

    r = BasicvsrppMosaicRestorer(model, DEV, fp16=True)
    runner = r._graph_runner
    if not runner.enabled:
        print("RESULT: SKIP (cuda graph disabled)"); return

    # Single-threaded warmup of WARM_T only.
    r.warmup_graph(WARM_T)
    caps_after_warmup = runner.captures
    print(f"after warmup: captures={caps_after_warmup} (expect 1)")

    # Restore warmed length -> should replay, no new capture.
    out_warm = torch.stack(r.restore(make_clip(WARM_T, 1))).float()
    caps_after_warm_restore = runner.captures
    reps_after_warm = runner.replays

    # Restore unwarmed length -> eager, no capture.
    out_unwarm = torch.stack(r.restore(make_clip(UNWARM_T, 2))).float()
    caps_after_unwarm = runner.captures

    d_warm = (out_warm - ref_warm).abs().max().item()
    d_unwarm = (out_unwarm - ref_unwarm).abs().max().item()
    print(f"warmed   T={WARM_T}: replays={reps_after_warm} max_abs_vs_eager={d_warm:.1f}")
    print(f"unwarmed T={UNWARM_T}: captures_total={caps_after_unwarm} max_abs_vs_eager={d_unwarm:.1f}")
    print(f"capture_failures={runner.capture_failures}")

    ok = (
        caps_after_warmup == 1
        and caps_after_warm_restore == 1            # no capture during restore
        and reps_after_warm >= 1                    # warmed length replayed
        and caps_after_unwarm == 1                  # unwarmed did NOT capture
        and runner.capture_failures == 0
        and d_warm == 0 and d_unwarm == 0           # both bit-exact vs eager
    )
    print("RESULT:", "PASS" if ok else "CHECK")


if __name__ == "__main__":
    main()
