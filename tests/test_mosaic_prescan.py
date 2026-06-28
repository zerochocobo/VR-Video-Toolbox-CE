from __future__ import annotations

import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine.probe import VideoMetadata
from utils.keyframe_cutter import align_segments
from utils import mosaic_prescan


class MosaicPrescanAggregationTests(unittest.TestCase):
    def test_detector_imgsz_defaults_to_2048_for_large_sources(self) -> None:
        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda _key, default: default):
            self.assertEqual(mosaic_prescan._resolve_detector_imgsz(4096, 4096), 2048)

    def test_extract_boxes_can_apply_fine_conf_filter(self) -> None:
        class ArrayLike:
            def __init__(self, data):
                self._data = data

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._data

        class Boxes:
            xyxy = ArrayLike([[1.0, 2.0, 10.0, 20.0], [30.0, 40.0, 80.0, 90.0]])
            conf = ArrayLike([0.55, 0.70])

            def __len__(self):
                return 2

        class Result:
            boxes = Boxes()
            masks = None
            orig_shape = (100, 100, 3)

        boxes, debug = mosaic_prescan._extract_boxes_with_debug(Result(), min_conf=0.60)

        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][4], 0.70, places=5)
        self.assertFalse(debug[0]["accepted"])
        self.assertIn("low_conf", debug[0]["reject_reason"])

    def test_large_boxes_are_not_rejected_by_area(self) -> None:
        self.assertIsNone(mosaic_prescan._box_limit_reason((0.0, 0.0, 4096.0, 4096.0), 4096, 4096))

    def test_sbs_boxes_are_split_to_single_eye_coordinates(self) -> None:
        split = mosaic_prescan._split_sbs_boxes_to_eye(
            [
                (1900.0, 100.0, 2200.0, 300.0, 0.90),
                (2300.0, 50.0, 2600.0, 200.0, 0.80),
                (100.0, 10.0, 300.0, 100.0, 0.70),
            ],
            frame_w=4096,
            frame_h=2048,
        )

        self.assertEqual(
            split["left"],
            [
                (1900.0, 100.0, 2048.0, 300.0, 0.90),
                (100.0, 10.0, 300.0, 100.0, 0.70),
            ],
        )
        self.assertEqual(
            split["right"],
            [
                (0.0, 100.0, 152.0, 300.0, 0.90),
                (252.0, 50.0, 552.0, 200.0, 0.80),
            ],
        )

    def test_detector_work_size_keeps_configured_yolo_resolution_without_8k_bgr(self) -> None:
        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda key, default=None: 2048 if key == "pre_extract_yolo_imgsz" else default):
            self.assertEqual(mosaic_prescan._detector_work_size(8192, 4096), (2048, 1024))
            self.assertEqual(mosaic_prescan._detector_work_size(4096, 4096), (2048, 2048))

    def test_detector_resized_boxes_scale_back_to_source_space(self) -> None:
        boxes = [(475.0, 25.0, 550.0, 75.0, 0.90)]
        debug = [{
            "raw_box_xyxy": [475.0, 25.0, 550.0, 75.0],
            "mask_box_xyxy": [475.0, 25.0, 550.0, 75.0],
            "used_box_xyxy": [475.0, 25.0, 550.0, 75.0],
            "accepted": True,
        }]

        scaled_boxes, scaled_debug = mosaic_prescan._scale_boxes_and_debug(boxes, debug, 4.0, 4.0)

        self.assertEqual(scaled_boxes, [(1900.0, 100.0, 2200.0, 300.0, 0.90)])
        self.assertEqual(scaled_debug[0]["used_box_xyxy"], [1900.0, 100.0, 2200.0, 300.0])

    def test_detector_batch_oom_fallback_splits_and_preserves_order(self) -> None:
        class Detector:
            def preprocess(self, frames):
                if len(frames) > 2:
                    raise RuntimeError("CUDA out of memory")
                return frames

            def inference_and_postprocess(self, preprocessed, frames):
                return [f"result-{frame}" for frame in frames]

        messages = []
        results = mosaic_prescan._run_detector_batch(Detector(), [1, 2, 3, 4, 5], log_callback=messages.append)

        self.assertEqual(results, ["result-1", "result-2", "result-3", "result-4", "result-5"])
        self.assertTrue(any("detector batch OOM" in item for item in messages))

    def test_detector_batch_box_only_uses_lightweight_postprocess(self) -> None:
        class Detector:
            def __init__(self):
                self.full_called = False
                self.box_called = False

            def preprocess(self, frames):
                return [f"pre-{frame}" for frame in frames]

            def inference_and_postprocess_boxes(self, preprocessed, frames):
                self.box_called = True
                assert preprocessed == ["pre-a", "pre-b"]
                return [f"box-{frame}" for frame in frames]

            def inference_and_postprocess(self, _preprocessed, _frames):
                self.full_called = True
                return []

        detector = Detector()
        results = mosaic_prescan._run_detector_batch(detector, ["a", "b"], boxes_only=True)

        self.assertEqual(results, ["box-a", "box-b"])
        self.assertTrue(detector.box_called)
        self.assertFalse(detector.full_called)

    def test_low_conf_far_boxes_do_not_expand_segment_rect_to_full_frame(self) -> None:
        hits = []
        for idx in range(120):
            t = idx * 0.5
            x_shift = min(idx, 60) * 3.0
            boxes = [(1800.0 - x_shift, 3500.0, 2400.0, 3800.0, 0.88)]
            if idx in {21, 76, 77}:
                boxes.append((2118.0, 6.0, 2883.0, 150.0, 0.30))
            hits.append({"t": t, "boxes": boxes})

        meta = VideoMetadata(
            path="synthetic.mp4",
            width=4096,
            height=4096,
            duration=60.125532,
            nb_frames=3607,
            source_fps=59.94,
        )

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda _key, default: default):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertGreaterEqual(segment.y, 3200)
        self.assertLess(segment.h, 1000)
        self.assertLess(segment.w, 1600)

    def test_scene_cut_forces_segment_boundary(self) -> None:
        # One continuous mosaic in a fixed spot; a scene cut mid-way must split
        # the timeline in two even though the crop area never grows.
        hits = [{"t": idx * 0.5, "boxes": [(500.0, 500.0, 900.0, 900.0, 0.9)]} for idx in range(40)]
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=20.0, source_fps=30.0)

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda _key, default: default):
            no_cut = mosaic_prescan._aggregate_hits(hits, meta)
            with_cut = mosaic_prescan._aggregate_hits(hits, meta, scene_cuts=[10.0])

        self.assertEqual(len(no_cut), 1)
        self.assertEqual(len(with_cut), 2)
        self.assertLessEqual(with_cut[0].end_s, 10.5)
        self.assertGreaterEqual(with_cut[1].start_s, 9.5)

    def test_scene_cut_outside_any_group_is_ignored(self) -> None:
        hits = [{"t": idx * 0.5, "boxes": [(500.0, 500.0, 900.0, 900.0, 0.9)]} for idx in range(20)]
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=60.0, source_fps=30.0)

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda _key, default: default):
            segs = mosaic_prescan._aggregate_hits(hits, meta, scene_cuts=[40.0])

        self.assertEqual(len(segs), 1)

    def test_position_jump_splits_continuous_mosaic_run(self) -> None:
        hits = []
        for idx in range(24):
            if idx < 12:
                box = (500.0, 2200.0, 1000.0, 3300.0, 0.91)
            else:
                box = (3000.0, 2200.0, 3500.0, 3300.0, 0.92)
            hits.append({"t": idx * 0.5, "boxes": [box]})
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=14.0, source_fps=30.0)
        cfg = {
            "pre_extract_segment_max_area_ratio": 0.30,
            "pre_extract_segment_jump_min_dur_s": 1.5,
            "pre_extract_segment_min_dur_s": 10.0,
        }

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda key, default: cfg.get(key, default)):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        self.assertEqual(len(segments), 2)
        self.assertLessEqual(segments[0].end_s, 6.0)
        self.assertGreaterEqual(segments[1].start_s, 5.5)
        self.assertLess(segments[0].x, 1000)
        self.assertGreater(segments[1].x, 2500)

    def test_position_jump_splits_even_when_area_stays_under_cap(self) -> None:
        # A small mosaic (union never exceeds the area cap) that teleports to a
        # nearby-but-clearly-different spot must still split: the jump trigger is
        # independent of the area cap, gated only by ``jump_min_dur_s``.
        hits = []
        for idx in range(100):
            if idx < 62:  # ~30.5s at position A
                box = (200.0, 200.0, 600.0, 600.0, 0.91)
            else:         # teleport to B (center moved > jump_center_ratio*frame)
                box = (1000.0, 200.0, 1400.0, 600.0, 0.92)
            hits.append({"t": idx * 0.5, "boxes": [box]})
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=52.0, source_fps=30.0)
        cfg = {
            "pre_extract_segment_max_area_ratio": 0.30,
            "pre_extract_segment_jump_min_dur_s": 30.0,
            "pre_extract_segment_jump_center_ratio": 0.18,
            "pre_extract_segment_jump_min_overlap": 0.10,
            "pre_extract_segment_min_dur_s": 10.0,
        }

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda key, default: cfg.get(key, default)):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        self.assertEqual(len(segments), 2)
        segs = sorted(segments, key=lambda s: s.start_s)
        # A *time* split: the two windows are sequential, not two spatial clusters
        # both spanning the whole run. (Union stays under the area cap throughout,
        # so this split can only have come from the jump rule.)
        self.assertLess(segs[0].end_s, 35.0)
        self.assertAlmostEqual(segs[0].end_s, segs[1].start_s, delta=0.6)
        eye_area = 4096 * 4096
        self.assertLess(max((seg.w * seg.h) / eye_area for seg in segments), 0.30)

    def test_position_jump_holds_until_min_duration(self) -> None:
        # The same teleport, but it happens before ``jump_min_dur_s`` elapses and
        # the union stays under the area cap -> no split (avoids tiny segments).
        hits = []
        for idx in range(100):
            if idx < 20:  # ~9.5s at position A, shorter than jump_min_dur_s
                box = (200.0, 200.0, 600.0, 600.0, 0.91)
            else:
                box = (1000.0, 200.0, 1400.0, 600.0, 0.92)
            hits.append({"t": idx * 0.5, "boxes": [box]})
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=52.0, source_fps=30.0)
        cfg = {
            "pre_extract_segment_max_area_ratio": 0.30,
            "pre_extract_segment_jump_min_dur_s": 30.0,
            "pre_extract_segment_min_dur_s": 10.0,
        }

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda key, default: cfg.get(key, default)):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        # No *time* split: A and B still become two spatial clusters, but both
        # span the same (full) window -- proving the jump did not cut early.
        self.assertEqual(len(segments), 2)
        segs = sorted(segments, key=lambda s: s.start_s)
        self.assertAlmostEqual(segs[0].start_s, segs[1].start_s, delta=0.5)
        self.assertAlmostEqual(segs[0].end_s, segs[1].end_s, delta=0.5)

    def test_two_simultaneous_small_mosaics_are_not_time_fragmented(self) -> None:
        # Two small mosaics present at the same time but far apart: their combined
        # bounding box spans most of the frame (> area cap), but each crop is tiny.
        # The area cap is measured per spatial cluster, so this must NOT be sliced
        # into many short windows -- it stays one window, split spatially into two
        # long segments (the IPVR-385 over-fragmentation regression).
        hits = []
        for idx in range(160):  # 80s, both mosaics every sample
            tl = (100.0, 100.0, 500.0, 500.0, 0.90)
            br = (3500.0, 3500.0, 3900.0, 3900.0, 0.91)
            hits.append({"t": idx * 0.5, "boxes": [tl, br]})
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=82.0, source_fps=30.0)
        cfg = {
            "pre_extract_segment_max_area_ratio": 0.30,
            "pre_extract_segment_jump_min_dur_s": 30.0,
            "pre_extract_segment_min_dur_s": 10.0,
        }

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda key, default: cfg.get(key, default)):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        # Exactly two segments (one per region), both spanning the full run, both
        # with a tiny crop -- no time fragmentation from the huge combined union.
        self.assertEqual(len(segments), 2)
        segs = sorted(segments, key=lambda s: s.start_s)
        self.assertAlmostEqual(segs[0].start_s, segs[1].start_s, delta=0.5)
        self.assertAlmostEqual(segs[0].end_s, segs[1].end_s, delta=0.5)
        self.assertGreater(segs[0].end_s - segs[0].start_s, 60.0)
        eye_area = 4096 * 4096
        self.assertLess(max((s.w * s.h) / eye_area for s in segments), 0.10)

    def test_area_growth_without_gap_splits_continuous_mosaic_run(self) -> None:
        hits = []
        for idx in range(60):
            x = 200.0 + idx * 55.0
            hits.append({"t": idx * 0.5, "boxes": [(x, 2200.0, x + 600.0, 3300.0, 0.90)]})
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=32.0, source_fps=30.0)
        cfg = {
            "pre_extract_segment_max_area_ratio": 0.30,
            "pre_extract_segment_cut_continuous_area": True,
            "pre_extract_segment_min_dur_s": 5.0,
        }

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda key, default: cfg.get(key, default)):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        self.assertGreaterEqual(len(segments), 2)
        eye_area = 4096 * 4096
        self.assertLess(max((seg.w * seg.h) / eye_area for seg in segments), 0.40)

    def test_large_stable_mosaic_is_not_fragmented(self) -> None:
        # A single mosaic whose own expanded footprint already exceeds the area
        # cap must stay one continuous segment: splitting cannot shrink the crop,
        # it would only add model warmups and break temporal restoration.
        box = (200.0, 200.0, 2600.0, 2600.0, 0.92)
        hits = [{"t": idx * 0.5, "boxes": [box]} for idx in range(120)]
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=62.0, source_fps=30.0)
        cfg = {
            "pre_extract_segment_max_area_ratio": 0.30,
            "pre_extract_segment_cut_continuous_area": True,
            "pre_extract_segment_min_dur_s": 10.0,
        }

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda key, default: cfg.get(key, default)):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        # Single spatial cluster, single time window -> exactly one segment.
        self.assertEqual(len(segments), 1)
        eye_area = 4096 * 4096
        self.assertGreater((segments[0].w * segments[0].h) / eye_area, 0.30)

    def test_same_time_far_regions_produce_multiple_segments(self) -> None:
        hits = []
        for idx in range(8):
            hits.append({
                "t": idx * 0.5,
                "boxes": [
                    (300.0, 300.0, 700.0, 700.0, 0.91),
                    (3000.0, 3100.0, 3400.0, 3500.0, 0.89),
                ],
            })
        meta = VideoMetadata(path="synthetic.mp4", width=4096, height=4096, duration=6.0, source_fps=30.0)

        with patch.object(mosaic_prescan, "_cfg", side_effect=lambda _key, default: default):
            segments = mosaic_prescan._aggregate_hits(hits, meta)

        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0].x + segments[0].w, segments[1].x)

    def test_time_and_rect_overlap_segments_are_merged(self) -> None:
        segments = [
            mosaic_prescan.MosaicSegment(0, 0.0, 10.0, 0.0, 10.0, 100, 100, 512, 512, 0.91),
            mosaic_prescan.MosaicSegment(1, 6.0, 12.0, 6.0, 12.0, 500, 200, 512, 512, 0.89),
            mosaic_prescan.MosaicSegment(2, 20.0, 30.0, 20.0, 30.0, 500, 200, 512, 512, 0.88),
        ]

        merged = mosaic_prescan._merge_overlapping_segments(segments)

        self.assertEqual(len(merged), 2)
        self.assertEqual((merged[0].start_s, merged[0].end_s), (0.0, 12.0))
        self.assertEqual((merged[0].x, merged[0].y), (100, 100))
        self.assertEqual((merged[0].w, merged[0].h), (912, 612))
        self.assertEqual(merged[0].conf_max, 0.91)

    def test_keyframe_alignment_keeps_overlapping_time_disjoint_rects(self) -> None:
        segments = [
            mosaic_prescan.MosaicSegment(0, 0.0, 5.0, 0.0, 5.0, 256, 256, 512, 512, 0.91),
            mosaic_prescan.MosaicSegment(1, 0.0, 5.0, 0.0, 5.0, 3000, 3000, 512, 512, 0.89),
        ]

        aligned = align_segments(segments, [0.0, 2.0, 4.0, 6.0], duration=6.0)

        self.assertEqual(len(aligned), 2)
        self.assertEqual([seg.seg_id for seg in aligned], [0, 1])

    def test_source_keyframe_scan_uses_left_eye_original_size_without_scale(self) -> None:
        class Stdout:
            def __init__(self, frame: bytes):
                self._chunks = [frame, b""]

            def read(self, _n):
                return self._chunks.pop(0)

            def close(self):
                pass

        class Proc:
            def __init__(self, frame: bytes):
                self.stdout = Stdout(frame)
                self.returncode = 0

            def wait(self):
                pass

        class Detector:
            def preprocess(self, frames):
                return frames

            def inference_and_postprocess(self, preprocessed, frames):
                class Result:
                    boxes = None
                    orig_shape = (4, 4, 3)

                return [Result() for _ in frames]

        meta = VideoMetadata(path="source.mp4", width=8, height=4, duration=1.0, source_fps=30.0)
        seen = {}

        def fake_popen(cmd, **_kwargs):
            seen["cmd"] = cmd
            return Proc(bytes(4 * 4 * 3))

        fake_torch = types.SimpleNamespace(from_numpy=lambda frame: frame)

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0]),
            patch("utils.mosaic_prescan.probe.probe_video", return_value=meta),
            patch("utils.mosaic_prescan.shutil.which", return_value="ffmpeg"),
            patch("utils.mosaic_prescan.subprocess.Popen", side_effect=fake_popen),
            patch("utils.mosaic_prescan._get_detector", return_value=Detector()) as get_detector,
        ):
            hits, out_meta, _debug = mosaic_prescan._scan_hits_keyframes_lowres("source.mp4")

        self.assertEqual(hits, [])
        self.assertEqual((out_meta.width, out_meta.height), (4, 4))
        self.assertIn("crop=4:4:0:0", seen["cmd"])
        self.assertNotIn("scale=4:4", seen["cmd"])
        self.assertEqual(get_detector.call_args.kwargs["frame_w"], 4)
        self.assertEqual(get_detector.call_args.kwargs["frame_h"], 4)

    def test_keyframe_backend_cpu_uses_ffmpeg_scan(self) -> None:
        meta = VideoMetadata(path="source.mp4", width=4, height=4, duration=1.0, source_fps=30.0)

        def cfg(key, default=None):
            if key == "pre_extract_keyframe_scan_backend":
                return "cpu"
            return default

        with (
            patch("utils.mosaic_prescan._cfg", side_effect=cfg),
            patch("utils.mosaic_prescan._scan_hits_keyframes_lowres", return_value=([], meta, [])) as scan_cpu,
            patch("utils.mosaic_prescan._scan_hits_keyframes_gpu") as scan_gpu,
        ):
            hits, out_meta, debug = mosaic_prescan._scan_hits_keyframes("source.mp4")

        self.assertEqual(hits, [])
        self.assertEqual(out_meta, meta)
        self.assertEqual(debug, [])
        scan_cpu.assert_called_once()
        scan_gpu.assert_not_called()

    def test_keyframe_backend_auto_falls_back_to_cpu_on_gpu_error(self) -> None:
        meta = VideoMetadata(path="source.mp4", codec_name="hevc", pix_fmt="nv12", width=4, height=4, duration=1.0, source_fps=30.0)
        messages = []

        def cfg(key, default=None):
            if key == "pre_extract_keyframe_scan_backend":
                return "auto"
            return default

        with (
            patch("utils.mosaic_prescan._cfg", side_effect=cfg),
            patch("utils.mosaic_prescan.probe.route", return_value=(meta, mosaic_prescan.probe.BackendDecision("gpu_nv12", "ok"))),
            patch("utils.mosaic_prescan._scan_hits_keyframes_gpu", side_effect=RuntimeError("decoder failed")) as scan_gpu,
            patch("utils.mosaic_prescan._scan_hits_keyframes_lowres", return_value=([], meta, [])) as scan_cpu,
        ):
            mosaic_prescan._scan_hits_keyframes("source.mp4", log_callback=messages.append)

        scan_gpu.assert_called_once()
        scan_cpu.assert_called_once()
        self.assertTrue(any("GPU keyframe scan failed; falling back to CPU" in item for item in messages))

    def test_keyframe_backend_gpu_uses_gpu_scan(self) -> None:
        meta = VideoMetadata(path="source.mp4", width=4, height=4, duration=1.0, source_fps=30.0)

        def cfg(key, default=None):
            if key == "pre_extract_keyframe_scan_backend":
                return "gpu"
            return default

        with (
            patch("utils.mosaic_prescan._cfg", side_effect=cfg),
            patch("utils.mosaic_prescan._scan_hits_keyframes_gpu", return_value=([], meta, [])) as scan_gpu,
            patch("utils.mosaic_prescan._scan_hits_keyframes_lowres") as scan_cpu,
        ):
            hits, out_meta, debug = mosaic_prescan._scan_hits_keyframes("source.mp4")

        self.assertEqual(hits, [])
        self.assertEqual(out_meta, meta)
        self.assertEqual(debug, [])
        scan_gpu.assert_called_once()
        scan_cpu.assert_not_called()

    def test_keyframe_gpu_scan_decodes_key_packets_only(self) -> None:
        import numpy as np

        class FakeFrame:
            def y_uv_cupy(self):
                return np.zeros((4, 8), dtype=np.uint8), np.zeros((2, 4, 2), dtype=np.uint8)

        class FakeDecodedFrame:
            def __init__(self, pts):
                self._pts = int(pts)

            def getPTS(self):
                return self._pts

        class FakePacket:
            def __init__(self, *, bsl=1, key=False, pts=-1):
                self.bsl = int(bsl)
                self.key = bool(key)
                self.pts = int(pts)

        class FakeDemuxer:
            def __init__(self):
                self._packets = [
                    FakePacket(key=True, pts=100),
                    FakePacket(key=False, pts=150),
                    FakePacket(key=True, pts=200),
                    FakePacket(bsl=0, key=False, pts=-1),
                ]

            def GetNvCodecId(self):
                return "hevc"

            def Demux(self):
                pkt = self._packets.pop(0)
                seen["demuxed"].append((pkt.key, pkt.pts))
                return pkt

        class FakeDecoder:
            def Decode(self, pkt):
                if pkt.bsl and not pkt.key:
                    seen["decoded_non_key"] = True
                    return []
                if pkt.bsl:
                    seen["decoded_key_pts"].append(pkt.pts)
                    return [FakeDecodedFrame(100)] if pkt.pts == 200 else []
                return [FakeDecodedFrame(200)]

        class Detector:
            def preprocess(self, frames):
                seen["batch_frames"] = list(frames)
                return frames

            def inference_and_postprocess(self, preprocessed, frames):
                class Result:
                    boxes = None
                    masks = None
                    orig_shape = (4, 4, 3)

                return [Result() for _ in frames]

        seen = {"demuxed": [], "decoded_key_pts": []}
        meta = VideoMetadata(
            path="source.mp4",
            codec_name="hevc",
            pix_fmt="nv12",
            width=8,
            height=4,
            duration=2.0,
            source_fps=30.0,
        )
        fake_nvc = types.SimpleNamespace(
            CreateDemuxer=lambda _path: FakeDemuxer(),
            CreateDecoder=lambda **_kwargs: FakeDecoder(),
            OutputColorType=types.SimpleNamespace(NATIVE=0),
        )

        with (
            patch.dict("sys.modules", {"PyNvVideoCodec": fake_nvc}),
            # Pin batch >= 2 so both decoded frames land in one detector batch;
            # the keyframe scan defaults batch to 1, which would otherwise flush
            # per-frame and depend on ambient config.
            patch("utils.mosaic_prescan._cfg", side_effect=lambda key, default=None: 8 if key == "pre_extract_yolo_batch" else default),
            patch("utils.mosaic_prescan.probe.route", return_value=(meta, mosaic_prescan.probe.BackendDecision("gpu_nv12", "ok"))),
            patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0, 1.0]),
            patch("utils.mosaic_prescan._decoded_frame_to_gpu_frame", return_value=FakeFrame()),
            patch("utils.mosaic_prescan._cupy_to_torch_bgr", side_effect=lambda *_args, **_kwargs: "frame"),
            patch("utils.mosaic_prescan._get_detector", return_value=Detector()) as get_detector,
        ):
            hits, out_meta, _debug = mosaic_prescan._scan_hits_keyframes_gpu("source.mp4")

        self.assertEqual(hits, [])
        self.assertEqual(seen["decoded_key_pts"], [100, 200])
        self.assertNotIn("decoded_non_key", seen)
        self.assertEqual(seen["batch_frames"], ["frame", "frame"])
        self.assertEqual((out_meta.width, out_meta.height), (4, 4))
        self.assertEqual(get_detector.call_args.kwargs["frame_w"], 4)
        self.assertEqual(get_detector.call_args.kwargs["frame_h"], 4)

    def test_fine_empty_scan_cache_skips_second_gpu_transform_scan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "clip.mp4"
            model = root / "model.pt"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            meta = VideoMetadata(path=str(video), width=8, height=4, duration=1.0, source_fps=30.0)

            def cfg(key, default=None):
                values = {
                    "pre_extract_empty_scan_cache": True,
                    "pre_extract_save_detection_debug": False,
                    "pre_extract_sample_stride_s": 0.5,
                    "pre_extract_yolo_imgsz": 2048,
                    "pre_extract_use_mask_boxes": True,
                }
                return values.get(key, default)

            with (
                patch("utils.mosaic_prescan._cfg", side_effect=cfg),
                patch("utils.mosaic_prescan._model_path", return_value=str(model)),
                patch("utils.mosaic_prescan._scan_hits_gpu_transform", return_value=([], meta, [], [])) as scan_gpu,
            ):
                first = mosaic_prescan.scan_segments_gpu_transform(
                    video,
                    crop_mode="left",
                    min_conf=0.5,
                )
                second = mosaic_prescan.scan_segments_gpu_transform(
                    video,
                    crop_mode="left",
                    min_conf=0.5,
                )

            self.assertEqual(first, [])
            self.assertEqual(second, [])
            scan_gpu.assert_called_once()

    def test_fine_empty_scan_cache_key_includes_min_conf(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "clip.mp4"
            model = root / "model.pt"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            meta = VideoMetadata(path=str(video), width=8, height=4, duration=1.0, source_fps=30.0)

            def cfg(key, default=None):
                values = {
                    "pre_extract_empty_scan_cache": True,
                    "pre_extract_save_detection_debug": False,
                    "pre_extract_sample_stride_s": 0.5,
                    "pre_extract_yolo_imgsz": 2048,
                    "pre_extract_use_mask_boxes": True,
                }
                return values.get(key, default)

            with (
                patch("utils.mosaic_prescan._cfg", side_effect=cfg),
                patch("utils.mosaic_prescan._model_path", return_value=str(model)),
                patch("utils.mosaic_prescan._scan_hits_gpu_transform", return_value=([], meta, [], [])) as scan_gpu,
            ):
                mosaic_prescan.scan_segments_gpu_transform(video, crop_mode="left", min_conf=0.5)
                mosaic_prescan.scan_segments_gpu_transform(video, crop_mode="left", min_conf=0.6)

            self.assertEqual(scan_gpu.call_count, 2)


if __name__ == "__main__":
    unittest.main()
