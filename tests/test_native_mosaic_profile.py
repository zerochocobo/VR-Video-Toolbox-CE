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
from gpu_engine.native_mosaic.engine import NativeMosaicEngine, _gpu_frame_source_decision, _native_restore_limits


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


def test_native_restore_limits_guard_large_frames_on_16gb_gpu(monkeypatch):
    monkeypatch.delenv("VRVT_NATIVE_VRAM_GUARD", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_MAX_CLIP_LENGTH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_BATCH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_FRAME_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_CLIP_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_QUEUE", raising=False)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return 1 * 1024 ** 3, 16 * 1024 ** 3

    limits = _native_restore_limits(
        SimpleNamespace(width=4096, height=4096),
        180,
        torch_module=SimpleNamespace(cuda=FakeCuda),
    )

    assert limits.max_clip_length == 64
    assert limits.detector_batch_size == 1
    assert limits.frame_queue_mb == 128
    assert limits.clip_queue_mb == 128
    assert limits.detector_queue_size == 2
    assert "large restore frame" in limits.reason


def test_native_restore_limits_guard_4k_frames_on_8gb_gpu(monkeypatch):
    monkeypatch.delenv("VRVT_NATIVE_VRAM_GUARD", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_MAX_CLIP_LENGTH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_BATCH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_FRAME_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_CLIP_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_QUEUE", raising=False)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return 1 * 1024 ** 3, 8 * 1024 ** 3

    limits = _native_restore_limits(
        SimpleNamespace(width=3840, height=2160),
        180,
        torch_module=SimpleNamespace(cuda=FakeCuda),
    )

    assert limits.max_clip_length == 24
    assert limits.detector_batch_size == 1
    assert limits.frame_queue_mb == 64
    assert limits.clip_queue_mb == 64
    assert limits.detector_queue_size == 1
    assert "threshold 6000000" in limits.reason


def test_native_restore_limits_keep_defaults_for_small_frames(monkeypatch):
    monkeypatch.delenv("VRVT_NATIVE_VRAM_GUARD", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_MAX_CLIP_LENGTH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_BATCH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_FRAME_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_CLIP_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_QUEUE", raising=False)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return 1 * 1024 ** 3, 16 * 1024 ** 3

    limits = _native_restore_limits(
        SimpleNamespace(width=1920, height=1080),
        180,
        torch_module=SimpleNamespace(cuda=FakeCuda),
    )

    assert limits.max_clip_length == 180
    assert limits.detector_batch_size == 4
    assert limits.frame_queue_mb == 512
    assert limits.clip_queue_mb == 512
    assert limits.detector_queue_size == 8
    assert limits.reason == ""


def test_native_restore_limits_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VRVT_NATIVE_VRAM_GUARD", "0")
    monkeypatch.delenv("VRVT_NATIVE_MAX_CLIP_LENGTH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_BATCH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_FRAME_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_CLIP_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_QUEUE", raising=False)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return 1 * 1024 ** 3, 16 * 1024 ** 3

    limits = _native_restore_limits(
        SimpleNamespace(width=4096, height=4096),
        180,
        torch_module=SimpleNamespace(cuda=FakeCuda),
    )

    assert limits.max_clip_length == 180
    assert limits.detector_batch_size == 4
    assert limits.reason == ""


def test_native_restore_limits_env_clip_cannot_raise_guarded_clip(monkeypatch):
    monkeypatch.delenv("VRVT_NATIVE_VRAM_GUARD", raising=False)
    monkeypatch.setenv("VRVT_NATIVE_MAX_CLIP_LENGTH", "120")
    monkeypatch.delenv("VRVT_NATIVE_DETECT_BATCH", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_FRAME_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_CLIP_QUEUE_MB", raising=False)
    monkeypatch.delenv("VRVT_NATIVE_DETECT_QUEUE", raising=False)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return 1 * 1024 ** 3, 16 * 1024 ** 3

    limits = _native_restore_limits(
        SimpleNamespace(width=4096, height=4096),
        180,
        torch_module=SimpleNamespace(cuda=FakeCuda),
    )

    assert limits.max_clip_length == 64
    assert "120 (max_clip_length 64->64)" in limits.reason


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
    monkeypatch.setattr(files.runtime, "format_vram_usage", lambda **_kwargs: "")
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


def test_progress_includes_vram_suffix(monkeypatch):
    logs: list[str] = []

    import gpu_engine.files as files

    monkeypatch.setattr(files.runtime, "format_vram_usage", lambda **_kwargs: " | VRAM 1.0/8.0 GiB")
    progress = _Progress(10, logs.append, min_interval=0.0, min_pct=0.0)

    progress.update(1)

    assert logs
    assert "VRAM 1.0/8.0 GiB" in logs[-1]


def test_progress_prefix_can_be_native(monkeypatch):
    logs: list[str] = []

    import gpu_engine.files as files

    monkeypatch.setattr(files.runtime, "format_vram_usage", lambda **_kwargs: "")
    progress = _Progress(10, logs.append, min_interval=0.0, min_pct=0.0, prefix="[native]")

    progress.update(1)

    assert logs[-1].startswith("[native] ")


def test_gpu_progress_does_not_log_percent_bursts_before_interval(monkeypatch):
    now = {"value": 100.0}
    logs: list[str] = []

    import gpu_engine.files as files

    monkeypatch.setattr(files.time, "perf_counter", lambda: now["value"])
    monkeypatch.setattr(files.runtime, "format_vram_usage", lambda **_kwargs: "")
    progress = _Progress(300, logs.append, min_interval=5.0, min_pct=5.0)

    progress.update(1)
    progress.update(16)
    progress.update(31)
    progress.update(46)
    assert len(logs) == 1

    now["value"] += 4.0
    progress.update(62)
    assert len(logs) == 1

    now["value"] += 1.0
    progress.update(77)
    assert len(logs) == 2


def test_native_stage_progress_uses_slower_default_throttle(monkeypatch):
    logs: list[str] = []
    now = {"value": 100.0}

    import gpu_engine.native_mosaic.progress as native_progress

    monkeypatch.setattr(native_progress.time, "perf_counter", lambda: now["value"])
    monkeypatch.setattr(native_progress, "vram_suffix", lambda: "")
    progress = native_progress.NativeStageProgress("FrameRestorer compose", logs.append, total=100)

    progress.update(5)
    progress.update(10)
    progress.update(15)
    assert len(logs) == 0

    progress.update(20)
    assert len(logs) == 1

    now["value"] += 4.0
    progress.update(25)
    assert len(logs) == 1

    now["value"] += 1.0
    progress.update(26)
    assert len(logs) == 2


def test_vram_query_subprocess_is_hidden_on_windows():
    import subprocess
    import sys

    from gpu_engine import runtime

    kwargs = runtime._hidden_subprocess_kwargs(0.8)

    if sys.platform.startswith("win"):
        assert kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW
        assert kwargs.get("startupinfo") is not None
    else:
        assert "creationflags" not in kwargs


def test_vram_query_returns_cached_value_while_refresh_in_progress(monkeypatch):
    from gpu_engine import runtime

    runtime._vram_cache.clear()
    runtime._vram_refreshing.clear()
    cached = runtime.VramUsage(used_mib=512, total_mib=8192, device_index=0, smi_id="0")
    runtime._vram_cache[(0, "0")] = (0.0, cached)
    runtime._vram_refreshing.add((0, "0"))

    def fail_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called while refresh is in progress")

    monkeypatch.setattr(runtime.subprocess, "run", fail_run)

    assert runtime.query_vram_usage(device_index=0, min_interval_s=0.0) is cached

    runtime._vram_cache.clear()
    runtime._vram_refreshing.clear()


def test_vram_query_maps_cuda_visible_devices_to_nvidia_smi_id(monkeypatch):
    from types import SimpleNamespace

    from gpu_engine import runtime

    runtime._vram_cache.clear()
    runtime._vram_refreshing.clear()
    commands: list[list[str]] = []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    monkeypatch.setattr(runtime, "_nvidia_smi_path", lambda: "nvidia-smi")

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="1024, 8192\n")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    usage = runtime.query_vram_usage(device_index=1, min_interval_s=0.0, force=True)

    assert usage is not None
    assert usage.device_index == 1
    assert usage.smi_id == "4"
    assert commands[-1][1] == "--id=4"
    runtime._vram_cache.clear()
    runtime._vram_refreshing.clear()


def test_vram_cache_is_keyed_by_resolved_device(monkeypatch):
    from types import SimpleNamespace

    from gpu_engine import runtime

    runtime._vram_cache.clear()
    runtime._vram_refreshing.clear()
    commands: list[list[str]] = []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    monkeypatch.setattr(runtime, "_nvidia_smi_path", lambda: "nvidia-smi")

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        used = "200" if cmd[1] == "--id=4" else "100"
        return SimpleNamespace(returncode=0, stdout=f"{used}, 8192\n")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    first = runtime.query_vram_usage(device_index=0, min_interval_s=999.0)
    second = runtime.query_vram_usage(device_index=1, min_interval_s=999.0)

    assert first is not None and first.used_mib == 100
    assert second is not None and second.used_mib == 200
    assert [cmd[1] for cmd in commands] == ["--id=2", "--id=4"]
    runtime._vram_cache.clear()
    runtime._vram_refreshing.clear()


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
