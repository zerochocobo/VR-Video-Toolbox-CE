import numpy as np
import pytest

from tool_2dvr import logic


def test_parse_time_to_seconds():
    assert logic.parse_time_to_seconds("") is None
    assert logic.parse_time_to_seconds("12.5") == pytest.approx(12.5)
    assert logic.parse_time_to_seconds("02:03") == pytest.approx(123.0)
    assert logic.parse_time_to_seconds("1:02:03.5") == pytest.approx(3723.5)
    with pytest.raises(ValueError):
        logic.parse_time_to_seconds("1:2:3:4")


def test_output_path_uses_projection_and_segment(tmp_path):
    src = tmp_path / "movie.mp4"
    out_dir = tmp_path / "out"
    path = logic.output_path(str(src), str(out_dir), "fisheye", "00:01", "00:02")
    assert path.endswith("movie_S0001_E0002_2dvr_fisheye_LR_180_SBS.mp4")

    flat_path = logic.output_path(str(src), str(out_dir), logic.PROJECTION_FLAT_3D, "00:01", "00:02")
    assert flat_path.endswith("movie_S0001_E0002_2dvr_flat3d_LR_SBS.mp4")


def test_default_projection_is_flat_3d():
    assert logic.DEFAULT_PROJECTION == logic.PROJECTION_FLAT_3D


def test_default_flat_fov_is_80_degrees():
    assert logic.DEFAULT_FLAT_FOV_DEG == pytest.approx(80.0)


def test_da3_vendor_root_uses_correct_vendor_dir():
    root = logic.da3_vendor_root()
    assert root.parts[-2:] == ("_vendor", "da3")
    assert (root / "depth_anything_3").exists()


def test_extract_depth_accepts_da3_prediction_object():
    class PredictionLike:
        def __init__(self, depth):
            self.depth = depth

    depth = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    extracted = logic._extract_depth(PredictionLike(depth))
    assert extracted.shape == (3, 4)
    np.testing.assert_array_equal(extracted, depth[0])


def test_extract_depths_preserves_da3_batch():
    class PredictionLike:
        def __init__(self, depth):
            self.depth = depth

    depth = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    extracted = logic._extract_depths(PredictionLike(depth))
    assert extracted.shape == (2, 3, 4)
    np.testing.assert_array_equal(extracted, depth)


def test_depth_batch_size_env(monkeypatch):
    monkeypatch.setenv("TOOL_2DVR_BATCH_SIZE", "3")
    assert logic.resolve_depth_batch_size() == 3
    monkeypatch.setenv("TOOL_2DVR_BATCH_SIZE", "bad")
    assert logic.resolve_depth_batch_size() == logic.DEFAULT_DEPTH_BATCH_SIZE


def test_encode_cmd_uses_realtime_nvenc_settings(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_NVENC_PRESET", raising=False)
    cmd = logic._build_encode_cmd("tmp.mp4", 7680, 2160, 60.0)

    assert cmd[cmd.index("-preset") + 1] in {"p1", "p2", "p3"}
    assert cmd[cmd.index("-cq") + 1] == "20"
    assert cmd[cmd.index("-rc") + 1] == "vbr"
    assert cmd[cmd.index("-colorspace") + 1] == "bt709"
    assert cmd[cmd.index("-color_primaries") + 1] == "bt709"
    assert cmd[cmd.index("-color_trc") + 1] == "bt709"
    assert cmd[cmd.index("-color_range") + 1] == "tv"
    assert cmd[cmd.index("-bsf:v") + 1] == (
        "hevc_metadata=colour_primaries=1:"
        "transfer_characteristics=1:"
        "matrix_coefficients=1:"
        "video_full_range_flag=0"
    )
    assert "-multipass" not in cmd
    assert "-spatial_aq" not in cmd
    assert "-temporal_aq" not in cmd


def test_encode_cmd_tags_sd_as_bt601():
    cmd = logic._build_encode_cmd("tmp.mp4", 720, 576, 25.0, input_pix_fmt="yuv420p")

    assert cmd[cmd.index("-colorspace") + 1] == "smpte170m"
    assert cmd[cmd.index("-color_primaries") + 1] == "smpte170m"
    assert cmd[cmd.index("-color_trc") + 1] == "smpte170m"
    assert cmd[cmd.index("-color_range") + 1] == "tv"
    assert cmd[cmd.index("-bsf:v") + 1] == (
        "hevc_metadata=colour_primaries=6:"
        "transfer_characteristics=6:"
        "matrix_coefficients=6:"
        "video_full_range_flag=0"
    )


def test_rgb_batch_to_yuv420p_uses_bt709_for_hd_red():
    height, width = 578, 2
    rgb = np.zeros((1, height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    flat = logic._rgb_batch_to_yuv420p(rgb)[0]
    y_size = height * width
    uv_size = (height // 2) * (width // 2)

    np.testing.assert_array_equal(np.unique(flat[:y_size]), np.array([63], dtype=np.uint8))
    np.testing.assert_array_equal(np.unique(flat[y_size:y_size + uv_size]), np.array([102], dtype=np.uint8))
    np.testing.assert_array_equal(np.unique(flat[y_size + uv_size:]), np.array([240], dtype=np.uint8))


def test_rgb_batch_to_yuv420p_uses_bt601_for_sd_red():
    height, width = 576, 2
    rgb = np.zeros((1, height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    flat = logic._rgb_batch_to_yuv420p(rgb)[0]
    y_size = height * width
    uv_size = (height // 2) * (width // 2)

    np.testing.assert_array_equal(np.unique(flat[:y_size]), np.array([81], dtype=np.uint8))
    np.testing.assert_array_equal(np.unique(flat[y_size:y_size + uv_size]), np.array([90], dtype=np.uint8))
    np.testing.assert_array_equal(np.unique(flat[y_size + uv_size:]), np.array([240], dtype=np.uint8))


def test_cuda_decode_cmd_uses_nvdec_download_filter():
    cmd = logic._build_decode_cmd("in.mp4", 1.0, 2.0, backend="cuda")

    assert "-hwaccel" in cmd
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert "-hwaccel_output_format" in cmd
    assert "scale_cuda=format=nv12,hwdownload,format=nv12,format=rgb24" in cmd
    assert cmd[-1] == "pipe:1"


def test_decode_backend_is_conservative(monkeypatch):
    monkeypatch.setattr(logic, "_ffmpeg_supports_cuda_hwaccel", lambda: True)
    assert logic._resolve_decode_backend(logic.VideoInfo(3840, 2160, 60.0, 10.0, "hevc")) == "cuda"
    assert logic._resolve_decode_backend(logic.VideoInfo(3840, 2160, 60.0, 10.0, "av1")) == "cpu"
    monkeypatch.setenv("TOOL_2DVR_NVDEC", "0")
    assert logic._resolve_decode_backend(logic.VideoInfo(3840, 2160, 60.0, 10.0, "hevc")) == "cpu"


def test_gpu_preprocess_uses_da3_patch_aligned_shape():
    torch = pytest.importorskip("torch")
    logic._add_da3_import_paths()
    from depth_anything_3.utils.io.input_processor_gpu import gpu_preprocess

    frames = [
        np.zeros((63, 112, 3), dtype=np.uint8),
        np.full((63, 112, 3), 255, dtype=np.uint8),
    ]
    out = gpu_preprocess(frames, device=torch.device("cpu"), target_res=50)

    assert tuple(out.shape) == (1, 2, 3, 28, 56)
    assert out.dtype == torch.float32


def test_hole_fill_mode_env(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_HOLE_FILL", raising=False)
    assert logic.resolve_hole_fill_mode(None) == logic.DEFAULT_HOLE_FILL_MODE
    assert logic.resolve_hole_fill_mode("background") == "background"
    assert logic.resolve_hole_fill_mode("e2fgvi") == "e2fgvi"
    assert logic.resolve_hole_fill_mode("inverse_warp") == "inverse_warp"
    assert logic.resolve_hole_fill_mode("bad") == logic.DEFAULT_HOLE_FILL_MODE
    monkeypatch.setenv("TOOL_2DVR_HOLE_FILL", "none")
    assert logic.resolve_hole_fill_mode(None) == "none"


def test_inverse_warp_compat_env_defaults_off(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_FAST_WARP", raising=False)
    assert logic.inverse_warp_compat_enabled() is False
    monkeypatch.setenv("TOOL_2DVR_FAST_WARP", "1")
    assert logic.inverse_warp_compat_enabled() is True


def test_stabilize_mode_env(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_STABILIZE", raising=False)
    assert logic.resolve_stabilize_mode(None) == "auto"
    assert logic.resolve_stabilize_mode("off") == "off"
    assert logic.resolve_stabilize_mode("full") == "full"
    assert logic.resolve_stabilize_mode("0") == "off"
    assert logic.resolve_stabilize_mode("1") == "auto"
    assert logic.resolve_stabilize_mode("bad") == "auto"
    monkeypatch.setenv("TOOL_2DVR_STABILIZE", "off")
    assert logic.resolve_stabilize_mode(None) == "off"


def test_e2fgvi_chunk_size_env(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_E2FGVI_CHUNK", raising=False)
    assert logic.resolve_e2fgvi_chunk_size() == logic.DEFAULT_E2FGVI_CHUNK_SIZE
    monkeypatch.setenv("TOOL_2DVR_E2FGVI_CHUNK", "1")
    assert logic.resolve_e2fgvi_chunk_size() == 2
    monkeypatch.setenv("TOOL_2DVR_E2FGVI_CHUNK", "16")
    assert logic.resolve_e2fgvi_chunk_size() == 16
    monkeypatch.setenv("TOOL_2DVR_E2FGVI_CHUNK", "bad")
    assert logic.resolve_e2fgvi_chunk_size() == logic.DEFAULT_E2FGVI_CHUNK_SIZE


def test_debug_output_stem_defaults_under_debug_output(monkeypatch, tmp_path):
    monkeypatch.delenv("TOOL_2DVR_DEBUG_DIR", raising=False)
    stem = logic.debug_output_stem(str(tmp_path / "movie_2dvr_flat3d_LR_SBS.mp4"))
    parts = stem.parts
    assert "debug_output" in parts
    assert "tool_2dvr" in parts
    assert stem.name == "movie_2dvr_flat3d_LR_SBS"

    custom = tmp_path / "dbg"
    monkeypatch.setenv("TOOL_2DVR_DEBUG_DIR", str(custom))
    assert logic.debug_output_stem("out.mp4").parent == custom / "out"


def test_normalize_near_uses_inverse_depth_order():
    depth = np.linspace(1.0, 10.0, 100, dtype=np.float32).reshape(10, 10)
    near = logic._normalize_near(depth)
    assert near[0, 0] > near[-1, -1]
    assert near[0, 0] == pytest.approx(1.0)
    assert near[-1, -1] == pytest.approx(0.0)


def test_debug_eye_env(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_DEBUG_EYE", raising=False)
    assert logic.debug_eye_enabled() is False
    monkeypatch.setenv("TOOL_2DVR_DEBUG_EYE", "1")
    assert logic.debug_eye_enabled() is True
    monkeypatch.setenv("TOOL_2DVR_DEBUG_EYE", "0")
    assert logic.debug_eye_enabled() is False


def test_max_disparity_scales_with_width_and_eye_distance():
    assert logic._max_disparity_pixels(1000, 65.0) == pytest.approx(35.0)
    assert logic._max_disparity_pixels(1000, 130.0) == pytest.approx(70.0)
    assert logic._max_disparity_pixels(4000, 65.0) == pytest.approx(96.0)


def test_stereo_pair_shifts_near_object_with_correct_eye_sign():
    h, w = 5, 20
    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    frame[:, 8, :] = np.array([255, 0, 0], dtype=np.uint8)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[:, 8] = 1.0

    left, right = logic.make_stereo_pair(frame, depth, 65.0)
    left_red = left[2, :, 0].astype(np.int16) - left[2, :, 1].astype(np.int16)
    right_red = right[2, :, 0].astype(np.int16) - right[2, :, 1].astype(np.int16)

    assert int(np.argmax(left_red)) > 8
    assert int(np.argmax(right_red)) < 8


def test_forward_stereo_marks_and_fills_disocclusion_holes():
    h, w = 5, 20
    frame = np.full((h, w, 3), 80, dtype=np.uint8)
    frame[:, 8, :] = np.array([240, 20, 20], dtype=np.uint8)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[:, 8] = 1.0

    result = logic._make_stereo_result(frame, depth, 65.0)

    assert result.left_holes.any()
    assert result.right_holes.any()
    assert not np.all(result.left[result.left_holes] == 0)
    assert not np.all(result.right[result.right_holes] == 0)


def test_hole_fill_none_leaves_forward_holes_unfilled():
    h, w = 5, 20
    frame = np.full((h, w, 3), 80, dtype=np.uint8)
    frame[:, 8, :] = np.array([240, 20, 20], dtype=np.uint8)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[:, 8] = 1.0

    result = logic._make_stereo_result(frame, depth, 65.0, hole_fill_mode="none")

    assert result.left_holes.any()
    assert np.all(result.left[result.left_holes] == 0)
    assert np.all(result.right[result.right_holes] == 0)


def test_inverse_warp_does_not_emit_forward_holes():
    h, w = 6, 20
    frame = np.full((h, w, 3), 80, dtype=np.uint8)
    frame[:, 8, :] = np.array([240, 20, 20], dtype=np.uint8)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[:, 8] = 1.0

    result = logic._make_stereo_result(frame, depth, 65.0, hole_fill_mode="inverse_warp")

    assert not result.left_holes.any()
    assert not result.right_holes.any()
    np.testing.assert_array_equal(result.left, result.left_before_fill)
    np.testing.assert_array_equal(result.right, result.right_before_fill)


def test_e2fgvi_alpha_keeps_mask_core_opaque():
    from tool_2dvr.e2fgvi_backend import _alpha_from_mask

    mask = np.zeros((9, 9), dtype=bool)
    mask[:, 4] = True
    alpha = _alpha_from_mask(mask)

    assert alpha.shape == (9, 9, 1)
    assert np.all(alpha[:, 4, 0] == 1.0)
    assert np.any((alpha[:, 3, 0] > 0.0) & (alpha[:, 3, 0] < 1.0))


def test_e2fgvi_mask_downscale_keeps_thin_holes():
    from tool_2dvr.e2fgvi_backend import _resize_masks

    mask = np.zeros((1, 64, 64), dtype=bool)
    mask[0, :, 1] = True

    resized = _resize_masks(mask, (16, 16), dilation=0)

    assert resized.shape == (1, 16, 16)
    assert resized.any()


def test_e2fgvi_cleanup_removes_residual_black_line():
    from tool_2dvr.e2fgvi_backend import _cleanup_residual_black_fill

    fill = np.full((12, 12, 3), 120, dtype=np.uint8)
    fill[:, 6, :] = 0
    mask = np.zeros((12, 12), dtype=bool)
    mask[:, 6] = True

    cleaned = _cleanup_residual_black_fill(fill, mask, threshold=12)

    assert cleaned[:, 6, :].mean() > 40


def test_render_sbs_frame_shapes():
    h, w = 6, 10
    grad = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
    frame = np.repeat(np.repeat(grad, h, axis=0), 3, axis=2)
    depth = np.linspace(1.0, 10.0, h * w, dtype=np.float32).reshape(h, w)

    heq_map = logic.make_projection_map(w, h, "hequirect")
    heq = logic.render_sbs_frame(frame, depth, "hequirect", 65.0, heq_map)
    assert heq.shape == (heq_map.out_h, heq_map.out_w * 2, 3)

    fish_map = logic.make_projection_map(w, h, "fisheye")
    fish = logic.render_sbs_frame(frame, depth, "fisheye", 65.0, fish_map)
    assert fish.shape == (fish_map.out_h, fish_map.out_w * 2, 3)


def test_flat_vr_projection_size_scales_with_fov():
    heq80 = logic.make_projection_map(720, 1280, "hequirect", 80.0)
    fish80 = logic.make_projection_map(720, 1280, "fisheye", 80.0)
    heq120 = logic.make_projection_map(720, 1280, "hequirect", 120.0)

    assert heq80.out_w == 2880
    assert heq80.out_h == 2880
    assert fish80.out_w == 2880
    assert fish80.out_h == 2880
    assert heq120.out_w == 1920
    assert heq120.out_h == 1920


def test_hequirect_projection_uses_flat_fov_camera_model():
    pmap = logic.make_projection_map(80, 80, "hequirect", 80.0)
    center = pmap.out_w // 2

    assert pmap.out_w == 180
    assert pmap.out_h == 180
    assert pmap.mask[center, center]
    assert pmap.map_x[center, center] == pytest.approx(40.4, abs=0.6)
    assert pmap.map_y[center, center] == pytest.approx(40.4, abs=0.6)
    assert pmap.mask[center, 50]
    assert pmap.map_x[center, 50] == pytest.approx(0.7, abs=0.8)
    assert not pmap.mask[0, center]
    assert not pmap.mask[center, 0]


def test_fisheye_projection_uses_flat_fov_camera_model():
    pmap = logic.make_projection_map(80, 80, "fisheye", 80.0)
    center = pmap.out_w // 2

    assert pmap.out_w == 180
    assert pmap.out_h == 180
    assert pmap.mask[center, center]
    assert pmap.map_x[center, center] == pytest.approx(40.4, abs=0.6)
    assert pmap.map_y[center, center] == pytest.approx(40.4, abs=0.6)
    assert pmap.mask[center, 50]
    assert pmap.map_x[center, 50] == pytest.approx(0.7, abs=0.8)
    assert pmap.mask[50, center]
    assert pmap.map_y[50, center] == pytest.approx(0.7, abs=0.8)
    assert not pmap.mask[0, center]
    assert not pmap.mask[center, 0]


def test_flat3d_projection_is_identity_sbs():
    h, w = 6, 10
    frame = np.full((h, w, 3), 64, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    pmap = logic.make_projection_map(w, h, logic.PROJECTION_FLAT_3D)

    assert pmap.out_w == w
    assert pmap.out_h == h
    assert pmap.mask.all()
    assert pmap.map_x[0, 0] == pytest.approx(0.0)
    assert pmap.map_x[0, -1] == pytest.approx(w - 1)
    assert pmap.map_y[0, 0] == pytest.approx(0.0)
    assert pmap.map_y[-1, 0] == pytest.approx(h - 1)

    sbs = logic.render_sbs_frame(frame, depth, logic.PROJECTION_FLAT_3D, 65.0, pmap)
    assert sbs.shape == (h, w * 2, 3)


def test_cuda_soft_shift_keeps_forward_warp_by_default(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    monkeypatch.delenv("TOOL_2DVR_FAST_WARP", raising=False)
    pmap = logic.make_projection_map(4, 4, logic.PROJECTION_FLAT_3D)

    renderer = logic.TorchStereoRenderer(4, 4, pmap, 65.0, "soft_shift")
    assert renderer.inverse_warp is False
    assert "inverse_warp" not in renderer.backend

    inverse_renderer = logic.TorchStereoRenderer(4, 4, pmap, 65.0, "inverse_warp")
    assert inverse_renderer.inverse_warp is True
    assert inverse_renderer.backend == "torch_cuda_inverse_warp"

    monkeypatch.setenv("TOOL_2DVR_FAST_WARP", "1")
    compat_renderer = logic.TorchStereoRenderer(4, 4, pmap, 65.0, "soft_shift")
    assert compat_renderer.inverse_warp is True
    assert compat_renderer.backend == "torch_cuda_inverse_warp"


def test_cuda_rgb_to_yuv420p_matches_cpu_bt709():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    height, width = 578, 2
    rgb = np.zeros((1, height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    pmap = logic.make_projection_map(width, height, logic.PROJECTION_FLAT_3D)
    renderer = logic.TorchStereoRenderer(width, height, pmap, 65.0, "soft_shift")

    gpu = renderer._rgb_u8_to_yuv420p_flat(torch.from_numpy(rgb).to(renderer.device)).cpu().numpy()

    np.testing.assert_array_equal(gpu, logic._rgb_batch_to_yuv420p(rgb))


def test_cuda_shift_fill_matches_iterative_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    pmap = logic.make_projection_map(8, 2, logic.PROJECTION_FLAT_3D)
    renderer = logic.TorchStereoRenderer(8, 2, pmap, 65.0, "soft_shift")
    image = torch.arange(1 * 2 * 8 * 3, device=renderer.device, dtype=torch.float32).view(1, 2, 8, 3)
    holes = torch.tensor(
        [[[False, True, True, False, True, False, True, True],
          [True, True, False, True, False, False, True, False]]],
        device=renderer.device,
        dtype=torch.bool,
    )

    for direction in (-1, 1):
        optimized = renderer._shift_fill_holes(image, holes, direction)
        reference = renderer._shift_fill_holes_iterative(image, holes, direction)
        torch.testing.assert_close(optimized, reference)


def test_cuda_stabilize_identity_env_matches_off(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    monkeypatch.setenv("TOOL_2DVR_NORM_ALPHA", "1")
    monkeypatch.setenv("TOOL_2DVR_DEPTH_BETA", "1")
    pmap = logic.make_projection_map(12, 6, logic.PROJECTION_FLAT_3D)
    frames = np.zeros((2, 6, 12, 3), dtype=np.uint8)
    frames[..., 0] = np.arange(12, dtype=np.uint8)[None, None, :]
    depths = torch.linspace(1.0, 4.0, 2 * 6 * 12, device="cuda", dtype=torch.float32).view(2, 6, 12)

    off = logic.TorchStereoRenderer(12, 6, pmap, 65.0, "soft_shift", stabilize_mode="off")
    auto = logic.TorchStereoRenderer(12, 6, pmap, 65.0, "soft_shift", stabilize_mode="auto")

    assert auto.temporal_identity is True
    assert auto.subpixel_splat_enabled is False
    torch.testing.assert_close(auto._render_batch_tensor(frames, depths), off._render_batch_tensor(frames, depths))


def test_cuda_depth_ema_scan_smooths_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    monkeypatch.setenv("TOOL_2DVR_DEPTH_BETA", "0.5")
    monkeypatch.setenv("TOOL_2DVR_ADAPTIVE_BETA", "0")
    pmap = logic.make_projection_map(4, 4, logic.PROJECTION_FLAT_3D)
    renderer = logic.TorchStereoRenderer(4, 4, pmap, 65.0, "soft_shift", stabilize_mode="auto")
    depth = torch.stack(
        [
            torch.ones((1, 4, 4), device=renderer.device),
            torch.full((1, 4, 4), 2.0, device=renderer.device),
        ],
        dim=0,
    )

    smoothed = renderer._apply_depth_ema(depth)

    torch.testing.assert_close(smoothed[0], torch.ones_like(smoothed[0]))
    torch.testing.assert_close(smoothed[1], torch.full_like(smoothed[1], 1.5))
    torch.testing.assert_close(renderer.depth_ema, torch.full_like(renderer.depth_ema, 1.5))


def test_cuda_subpixel_splat_changes_fractional_forward_warp(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    monkeypatch.delenv("TOOL_2DVR_NORM_ALPHA", raising=False)
    monkeypatch.delenv("TOOL_2DVR_DEPTH_BETA", raising=False)
    pmap = logic.make_projection_map(8, 2, logic.PROJECTION_FLAT_3D)
    off = logic.TorchStereoRenderer(8, 2, pmap, 65.0, "soft_shift", stabilize_mode="off")
    auto = logic.TorchStereoRenderer(8, 2, pmap, 65.0, "soft_shift", stabilize_mode="auto")
    frame = torch.zeros((1, 2, 8, 3), device=auto.device)
    frame[:, :, :, 0] = torch.arange(8, device=auto.device).view(1, 1, 8)
    near = torch.full((1, 2, 8), 0.5, device=auto.device)

    off_eye, _, _ = off._forward_warp_eye(frame, near, max_shift=3.0, eye_sign=1.0)
    auto_eye, _, _ = auto._forward_warp_eye(frame, near, max_shift=3.0, eye_sign=1.0)

    assert auto.subpixel_splat_enabled is True
    assert not torch.equal(auto_eye, off_eye)


def test_cuda_subpixel_splat_auto_threshold_and_full_override(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    monkeypatch.setenv("TOOL_2DVR_SUBPIXEL_MAX_PIXELS", "1")
    monkeypatch.delenv("TOOL_2DVR_SUBPIXEL_SPLAT", raising=False)
    pmap = logic.make_projection_map(8, 2, logic.PROJECTION_FLAT_3D)

    auto = logic.TorchStereoRenderer(8, 2, pmap, 65.0, "soft_shift", stabilize_mode="auto")
    full = logic.TorchStereoRenderer(8, 2, pmap, 65.0, "soft_shift", stabilize_mode="full")

    assert auto.subpixel_splat_enabled is False
    assert full.subpixel_splat_enabled is True


# --- Stage-2 PyNv backend tests ---


def test_resolve_backend_env(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_BACKEND", raising=False)
    assert logic.resolve_backend(None) == logic.BACKEND_AUTO
    assert logic.resolve_backend("pynv") == logic.BACKEND_PYNV
    assert logic.resolve_backend("ffmpeg") == logic.BACKEND_FFMPEG
    assert logic.resolve_backend("auto") == logic.BACKEND_AUTO
    assert logic.resolve_backend("1") == logic.BACKEND_PYNV
    assert logic.resolve_backend("0") == logic.BACKEND_FFMPEG
    assert logic.resolve_backend("garbage") == logic.BACKEND_AUTO
    monkeypatch.setenv("TOOL_2DVR_BACKEND", "pynv")
    assert logic.resolve_backend(None) == logic.BACKEND_PYNV


def test_pynv_supports_codec_whitelist():
    assert logic.pynv_supports_codec("hevc")
    assert logic.pynv_supports_codec("HEVC")
    assert logic.pynv_supports_codec("h264")
    assert logic.pynv_supports_codec("h265")  # alias for hevc
    assert logic.pynv_supports_codec("av1")
    assert not logic.pynv_supports_codec("prores")
    assert not logic.pynv_supports_codec("")


def test_pynv_should_use_falls_back_when_codec_unsupported(monkeypatch):
    monkeypatch.setattr(logic, "_pynv_available", lambda: True)
    info = logic.VideoInfo(3840, 2160, 60.0, 10.0, "prores")
    logs: list[str] = []
    assert not logic._pynv_should_use(info, logic.BACKEND_PYNV, log_callback=logs.append)
    assert any("codec" in m.lower() for m in logs)


def test_pynv_should_use_skips_when_backend_ffmpeg(monkeypatch):
    monkeypatch.setattr(logic, "_pynv_available", lambda: True)
    info = logic.VideoInfo(3840, 2160, 60.0, 10.0, "hevc")
    assert not logic._pynv_should_use(info, logic.BACKEND_FFMPEG)


def test_pynv_should_use_skips_when_pynv_unavailable(monkeypatch):
    monkeypatch.setattr(logic, "_pynv_available", lambda: False)
    info = logic.VideoInfo(3840, 2160, 60.0, 10.0, "hevc")
    assert not logic._pynv_should_use(info, logic.BACKEND_AUTO)


def test_cuda_torch_rgb_to_nv12_round_trip_solid_color():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Solid color images survive 4:2:0 subsampling losslessly modulo integer rounding.
    # Use HD height to exercise the bt709 branch.
    rgb = torch.full((720, 1280, 3), 0, dtype=torch.uint8, device="cuda")
    rgb[..., 0] = 200
    rgb[..., 1] = 100
    rgb[..., 2] = 50

    packed = logic._torch_rgb_uint8_to_nv12_packed(rgb)
    assert tuple(packed.shape) == (720 * 3 // 2, 1280)
    assert packed.dtype == torch.uint8

    y = packed[:720]
    uv_row = packed[720:]
    uv_pair = uv_row.reshape(720 // 2, 1280 // 2, 2)
    rgb_back = logic._torch_nv12_to_rgb_uint8(y, uv_pair, width=1280, height=720)
    diff = (rgb_back.to(torch.float32) - rgb.to(torch.float32)).abs()
    assert float(diff.max()) <= 1.0


def test_cuda_torch_rgb_to_nv12_round_trip_smooth_gradient():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Smooth gradient: 4:2:0 chroma loss is small.
    h, w = 720, 1280
    xs = torch.linspace(0.0, 255.0, w, device="cuda")
    ys = torch.linspace(0.0, 255.0, h, device="cuda")
    grid_x = xs.view(1, w).expand(h, w)
    grid_y = ys.view(h, 1).expand(h, w)
    r = grid_x
    g = grid_y
    b = (grid_x + grid_y) * 0.5
    rgb = torch.stack((r, g, b), dim=-1).clamp(0.0, 255.0).round().to(torch.uint8)

    packed = logic._torch_rgb_uint8_to_nv12_packed(rgb)
    y = packed[:h]
    uv_pair = packed[h:].reshape(h // 2, w // 2, 2)
    rgb_back = logic._torch_nv12_to_rgb_uint8(y, uv_pair, width=w, height=h)
    diff = (rgb_back.to(torch.float32) - rgb.to(torch.float32)).abs()
    # Smooth gradient + 4:2:0 chroma typically stays within ~2 LSB mean.
    assert float(diff.mean()) < 2.5


def test_cuda_torch_nv12_packed_layout_bt709():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # h > 576 forces bt709 limited-range path.
    h, w = 720, 24
    rgb = torch.zeros((h, w, 3), dtype=torch.uint8, device="cuda")
    rgb[..., 1] = 200  # solid green
    packed = logic._torch_rgb_uint8_to_nv12_packed(rgb)
    assert packed.shape == (h * 3 // 2, w)
    # bt709 Y for (0,200,0) = 16 + 0.6142*200 ~ 138.84
    y_mean = float(packed[:h].to(torch.float32).mean())
    assert 130.0 <= y_mean <= 145.0
    # bt709 Cb for green is < 128, Cr for green is < 128.
    uv_row = packed[h:]
    cb = uv_row[..., 0::2].to(torch.float32)
    cr = uv_row[..., 1::2].to(torch.float32)
    assert float(cb.mean()) < 128.0
    assert float(cr.mean()) < 128.0


def test_cuda_torch_nv12_packed_layout_bt601():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # h <= 576 forces bt601 limited-range path.
    h, w = 16, 24
    rgb = torch.zeros((h, w, 3), dtype=torch.uint8, device="cuda")
    rgb[..., 1] = 200
    packed = logic._torch_rgb_uint8_to_nv12_packed(rgb)
    # bt601 Y for (0,200,0) = 16 + 0.5041*200 ~ 116.83
    y_mean = float(packed[:h].to(torch.float32).mean())
    assert 110.0 <= y_mean <= 125.0


def test_cuda_gpu_preprocess_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    logic._add_da3_import_paths()
    from depth_anything_3.utils.io.input_processor_gpu import gpu_preprocess

    rng = np.random.default_rng(11)
    rgb = rng.integers(0, 256, size=(2, 224, 336, 3), dtype=np.uint8)
    tensor_input = torch.from_numpy(rgb).to("cuda")
    list_input = [rgb[i] for i in range(rgb.shape[0])]
    target_res = 224

    from_tensor = gpu_preprocess(tensor_input, device=torch.device("cuda"), target_res=target_res)
    from_list = gpu_preprocess(list_input, device=torch.device("cuda"), target_res=target_res)

    assert from_tensor.shape == from_list.shape
    torch.testing.assert_close(from_tensor, from_list, rtol=1e-5, atol=1e-5)
