from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from gpu_engine import vram_offload
from gpu_engine._profile import DecodeProfile
from gpu_engine.files import _Progress
from gpu_engine.native_mosaic import _gpu_ops, _torch_tuning
from gpu_engine.native_mosaic._cuda_graph_runner import CudaGraphRunner
from gpu_engine.native_mosaic.engine import NativeMosaicEngine, _gpu_frame_source_decision


def test_decode_profile_writes_sections_and_counters(tmp_path):
    out = tmp_path / "profile.json"
    profile = DecodeProfile(enabled=True, output_path=out)

    profile.metadata(frame_source="gpu_passthrough")
    profile.increment("frames_decoded", 2)
    with profile.section("decode.frame_at"):
        pass

    assert profile.write() == out
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["metadata"]["frame_source"] == "gpu_passthrough"
    assert data["counters"]["frames_decoded"] == 2
    assert data["sections"]["decode.frame_at"]["count"] == 1


def test_native_mosaic_passthrough_crop_region_returns_full_frame():
    assert NativeMosaicEngine._crop_region("passthrough", 3840, 2160) == (0, 0, 3840, 2160)


def test_gpu_frame_source_decision_defaults_to_gpu(monkeypatch):
    monkeypatch.delenv("VRVT_NATIVE_FORCE_CPU_FRAME_SOURCE", raising=False)
    monkeypatch.delenv("VRVT_GPU_FRAME_SOURCE_MIN_PIXELS", raising=False)

    ok, reason, pixels, min_pixels = _gpu_frame_source_decision(SimpleNamespace(width=1376, height=800))

    assert ok is True
    assert pixels == 1376 * 800
    assert min_pixels == 0
    assert "eligible" in reason


def test_gpu_frame_source_decision_can_be_forced_to_cpu(monkeypatch):
    monkeypatch.setenv("VRVT_NATIVE_FORCE_CPU_FRAME_SOURCE", "1")

    ok, reason, _pixels, _min_pixels = _gpu_frame_source_decision(SimpleNamespace(width=8192, height=4096))

    assert ok is False
    assert "VRVT_NATIVE_FORCE_CPU_FRAME_SOURCE" in reason


def test_gpu_frame_source_decision_threshold_can_be_enabled(monkeypatch):
    monkeypatch.delenv("VRVT_NATIVE_FORCE_CPU_FRAME_SOURCE", raising=False)
    monkeypatch.setenv("VRVT_GPU_FRAME_SOURCE_MIN_PIXELS", "2000000")

    ok, reason, _pixels, min_pixels = _gpu_frame_source_decision(SimpleNamespace(width=1376, height=800))

    assert ok is False
    assert min_pixels == 2_000_000
    assert "input too small" in reason


def test_vram_offload_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VRVT_VRAM_OFFLOAD", "0")

    class Clip:
        frames = [object()]
        masks = [object()]

    assert vram_offload.should_offload() is False
    assert vram_offload.maybe_offload_clip(Clip()) is False


def test_progress_fps_uses_adjacent_samples(monkeypatch):
    now = {"value": 0.0}
    logs: list[str] = []

    import gpu_engine.files as files

    monkeypatch.setattr(files.time, "perf_counter", lambda: now["value"])
    progress = _Progress(
        1000,
        logs.append,
        min_interval=0.0,
        min_pct=0.0,
        fps_smoothing_sec=0.001,
    )

    now["value"] = 10.0
    progress.update(1)
    now["value"] = 20.0
    progress.update(101)
    now["value"] = 30.0
    progress.update(201)

    assert "10.0 fps" in logs[1]
    assert "10.0 fps" in logs[2]


def test_inference_tuning_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VRVT_INFERENCE_TUNING", "0")

    assert _torch_tuning.inference_tuning_enabled() is False
    assert _torch_tuning.channels_last_enabled() is False
    assert _torch_tuning.apply_inference_tuning() == {"enabled": False}


def test_cuda_graph_cache_key_includes_shape_dtype_stride_and_device():
    contiguous = torch.zeros((1, 4, 3, 8, 8), dtype=torch.float32)
    sliced = torch.zeros((1, 8, 3, 8, 8), dtype=torch.float32)[:, ::2]

    key_contiguous = CudaGraphRunner.cache_key(contiguous)
    key_sliced = CudaGraphRunner.cache_key(sliced)

    assert key_contiguous[0] == tuple(contiguous.shape)
    assert key_contiguous[1] == str(contiguous.dtype)
    assert key_contiguous[2] == tuple(contiguous.stride())
    assert key_contiguous != key_sliced


def test_cuda_graph_runner_disabled_uses_eager_model():
    class Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, inputs):
            self.calls += 1
            return inputs + 1

    model = Model()
    runner = CudaGraphRunner(model, torch.device("cpu"), enabled=False)
    x = torch.zeros((1, 2), dtype=torch.float32)

    y = runner(x)

    assert model.calls == 1
    assert torch.equal(y, torch.ones_like(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_ops_resize_and_pad_hwc_cuda_tensor():
    img = torch.arange(4 * 4 * 3, device="cuda", dtype=torch.uint8).reshape(4, 4, 3)

    resized = _gpu_ops.resize_hwc_gpu(img, (2, 2))
    padded, pad = _gpu_ops.pad_hwc_gpu(resized, 4, 4, mode="zero")

    assert resized.shape == (2, 2, 3)
    assert padded.shape == (4, 4, 3)
    assert pad == (1, 1, 1, 1)
