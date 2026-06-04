from __future__ import annotations

import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
