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


def test_hole_fill_mode_env(monkeypatch):
    monkeypatch.delenv("TOOL_2DVR_HOLE_FILL", raising=False)
    assert logic.resolve_hole_fill_mode(None) == logic.DEFAULT_HOLE_FILL_MODE
    assert logic.resolve_hole_fill_mode("background") == "background"
    assert logic.resolve_hole_fill_mode("e2fgvi") == "e2fgvi"
    assert logic.resolve_hole_fill_mode("bad") == logic.DEFAULT_HOLE_FILL_MODE
    monkeypatch.setenv("TOOL_2DVR_HOLE_FILL", "none")
    assert logic.resolve_hole_fill_mode(None) == "none"


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
