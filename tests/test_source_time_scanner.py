from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine.fallback import OperationCancelled
from gpu_engine.probe import VideoMetadata
from one_click import logic
from utils.keyframe_cutter import TimelineEntry
from utils import source_time_scanner
from utils.mosaic_prescan import MosaicSegment
from utils.source_time_scanner import TimeInterval


class SourceTimeScannerTests(unittest.TestCase):
    def test_source_scan_merges_spatial_segments_into_time_interval(self) -> None:
        segments = [
            MosaicSegment(0, 1.0, 5.0, 1.0, 5.0, 100, 100, 512, 512, 0.91),
            MosaicSegment(1, 1.5, 6.0, 1.5, 6.0, 3000, 3000, 512, 512, 0.88),
        ]

        config = {
            "source_scan_strategy": "keyframes",
            "source_scan_merge_gap_s": 0.05,
            "source_scan_min_segment_s": 0.0,
            "source_scan_head_tail_pad_s": 0.0,
            "source_scan_max_segment_s": 0.0,
        }
        with (
            patch("utils.mosaic_prescan.scan_segments", return_value=segments),
            patch("utils.app_config.get", side_effect=lambda key, default=None: config.get(key, default)),
            patch("gpu_engine.probe.probe_video", return_value=VideoMetadata(path="source.mp4", width=4096, height=4096, duration=60.0, source_fps=30.0)),
        ):
            intervals = source_time_scanner.scan_source_time_segments("source.mp4")

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].start_s, 1.0)
        self.assertEqual(intervals[0].end_s, 6.0)
        self.assertEqual(intervals[0].conf_max, 0.91)

    def test_source_scan_coarse_merge_pads_and_combines_short_segments(self) -> None:
        intervals = [
            TimeInterval(100.0, 104.0, 0.72),
            TimeInterval(125.0, 130.0, 0.91),
        ]
        config = {
            "source_scan_merge_gap_s": 30.0,
            "source_scan_min_segment_s": 30.0,
            "source_scan_head_tail_pad_s": 5.0,
            "source_scan_max_segment_s": 0.0,
        }
        with patch("utils.app_config.get", side_effect=lambda key, default=None: config.get(key, default)):
            merged = source_time_scanner._coarse_merge(intervals, duration_s=300.0)

        self.assertEqual(len(merged), 1)
        self.assertLessEqual(merged[0].start_s, 95.0)
        self.assertGreaterEqual(merged[0].end_s, 135.0)
        self.assertEqual(merged[0].conf_max, 0.91)

    def test_source_scan_uses_keyframe_strategy_from_config(self) -> None:
        messages = []
        with (
            patch("utils.app_config.get", return_value="keyframes"),
            patch("utils.mosaic_prescan.scan_segments", return_value=[]) as scan_segments,
        ):
            intervals = source_time_scanner.scan_source_time_segments("source.mp4", log_callback=messages.append)

        self.assertEqual(intervals, [])
        self.assertEqual(scan_segments.call_args.kwargs["scan_strategy"], "keyframes")
        self.assertIn("[source-scan] scan strategy: keyframes", messages)

    def test_source_scan_sbs_helper_respects_fisheye_choice_and_interval(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "source.mp4"
            out = root / "tmp" / "mosaic_seg000.restored.mp4"
            src.write_bytes(b"source")
            out.parent.mkdir()

            with (
                patch("one_click.logic._pre_extract_supported", return_value=True),
                patch("one_click.logic.split_video_dual_fisheye") as split_fish,
                patch("one_click.logic.split_video_dual") as split_plain,
                patch("one_click.logic._process_pre_extract_or_lada") as process_lada,
                patch("one_click.logic.merge_videos_fisheye") as merge_fish,
            ):
                logic._process_sbs_clip_to_output(
                    str(src),
                    str(out),
                    use_fisheye=True,
                    pre_extract_inner=True,
                    keep_intermediate=False,
                    original_bitrate=1000000,
                    keep_original_bitrate=True,
                    start_time="10.000000",
                    end_time="20.000000",
                    work_dir=str(out.parent),
                    work_stem="mosaic_seg000",
                    split_keep_audio=False,
                )

            split_plain.assert_not_called()
            split_fish.assert_called_once()
            self.assertEqual(split_fish.call_args.args[:5], (
                str(src),
                str(out.parent / "mosaic_seg000_L_fisheye.mp4"),
                str(out.parent / "mosaic_seg000_R_fisheye.mp4"),
                "10.000000",
                "20.000000",
            ))
            self.assertFalse(split_fish.call_args.kwargs["keep_audio"])
            self.assertEqual(process_lada.call_count, 2)
            merge_fish.assert_called_once()

    def test_source_scan_single_eye_helper_respects_non_fisheye_choice_and_interval(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "source.mp4"
            out = root / "tmp" / "mosaic_seg000.restored.mp4"
            src.write_bytes(b"source")
            out.parent.mkdir()

            with (
                patch("one_click.logic._pre_extract_supported", return_value=True),
                patch("one_click.logic.split_video") as split_plain,
                patch("one_click.logic.split_video_fisheye") as split_fish,
                patch("one_click.logic._process_pre_extract_or_lada") as process_lada,
            ):
                logic._process_single_eye_clip_to_output(
                    str(src),
                    str(out),
                    eye_mode=2,
                    use_fisheye=False,
                    pre_extract_inner=True,
                    keep_intermediate=False,
                    final_bitrate_kbps=500,
                    start_time="10.000000",
                    end_time="20.000000",
                    work_dir=str(out.parent),
                    work_stem="mosaic_seg000",
                    split_keep_audio=False,
                )

            split_fish.assert_not_called()
            split_plain.assert_called_once()
            self.assertEqual(split_plain.call_args.args[:5], (
                str(src),
                str(out.parent / "mosaic_seg000_R.mp4"),
                "crop=iw/2:ih:iw/2:0",
                "10.000000",
                "20.000000",
            ))
            self.assertNotIn("final_bitrate_kbps", split_plain.call_args.kwargs)
            self.assertFalse(split_plain.call_args.kwargs["keep_audio"])
            process_lada.assert_called_once()

    def test_pair_eye_segments_skips_one_sided_groups(self) -> None:
        left = [
            MosaicSegment(0, 10.0, 20.0, 10.0, 20.0, 100, 100, 512, 512, 0.91),
            MosaicSegment(1, 40.0, 45.0, 40.0, 45.0, 120, 120, 512, 512, 0.55),
        ]
        right = [
            MosaicSegment(0, 10.0, 20.0, 10.0, 20.0, 200, 100, 512, 512, 0.90),
        ]

        paired_left, paired_right = logic._pair_eye_segments_by_time(left, right)

        self.assertEqual(len(paired_left), 1)
        self.assertEqual(len(paired_right), 1)
        self.assertAlmostEqual(paired_left[0].start_s, 10.0)
        self.assertAlmostEqual(paired_right[0].end_s, 20.0)

    def test_pair_eye_segments_keeps_high_confidence_one_sided_groups(self) -> None:
        left = [
            MosaicSegment(0, 10.0, 20.0, 10.0, 20.0, 100, 100, 512, 512, 0.91),
            MosaicSegment(1, 40.0, 45.0, 40.0, 45.0, 120, 120, 512, 512, 0.88),
        ]
        right = [
            MosaicSegment(0, 10.0, 20.0, 10.0, 20.0, 200, 100, 512, 512, 0.90),
        ]

        paired_left, paired_right = logic._pair_eye_segments_by_time(left, right)

        self.assertEqual(len(paired_left), 2)
        self.assertEqual(len(paired_right), 1)
        self.assertAlmostEqual(paired_left[0].start_s, 10.0)
        self.assertAlmostEqual(paired_left[1].start_s, 40.0)
        self.assertAlmostEqual(paired_left[1].end_s, 45.0)

    def test_pair_eye_segments_reindexes_by_start_time_after_score_matching(self) -> None:
        left = [
            MosaicSegment(0, 100.0, 110.0, 100.0, 110.0, 100, 100, 512, 512, 0.95),
            MosaicSegment(1, 10.0, 20.0, 10.0, 20.0, 0, 100, 512, 512, 0.80),
        ]
        right = [
            MosaicSegment(0, 100.0, 110.0, 100.0, 110.0, 100, 100, 512, 512, 0.94),
            MosaicSegment(1, 10.0, 20.0, 10.0, 20.0, 300, 100, 512, 512, 0.79),
        ]

        paired_left, paired_right = logic._pair_eye_segments_by_time(left, right)

        self.assertEqual([seg.seg_id for seg in paired_left], [0, 1])
        self.assertEqual([seg.seg_id for seg in paired_right], [0, 1])
        self.assertEqual([seg.start_s for seg in paired_left], [10.0, 100.0])
        self.assertEqual([seg.start_s for seg in paired_right], [10.0, 100.0])

    def test_pair_eye_segments_reuses_spanning_segment_for_non_overlapping_windows(self) -> None:
        left = [
            MosaicSegment(0, 10.0, 20.0, 10.0, 20.0, 100, 2400, 1200, 600, 0.90),
            MosaicSegment(1, 30.0, 40.0, 30.0, 40.0, 120, 2400, 1200, 600, 0.91),
        ]
        right = [
            MosaicSegment(0, 10.0, 40.0, 10.0, 40.0, 0, 2380, 1400, 640, 0.92),
        ]

        paired_left, paired_right = logic._pair_eye_segments_by_time(left, right)

        self.assertEqual(len(paired_left), 2)
        self.assertEqual(len(paired_right), 1)
        self.assertEqual([seg.start_s for seg in paired_left], [10.0, 30.0])
        self.assertEqual([seg.end_s for seg in paired_left], [20.0, 40.0])
        self.assertEqual([seg.start_s for seg in paired_right], [10.0])
        self.assertEqual([seg.end_s for seg in paired_right], [40.0])

    def test_pair_eye_segments_requires_spatial_overlap_inside_same_time_group(self) -> None:
        left = [
            MosaicSegment(0, 5.5, 175.2, 5.5, 175.2, 1072, 2288, 2208, 1808, 0.82),
            MosaicSegment(1, 5.5, 175.2, 5.5, 175.2, 2032, 0, 2064, 1056, 0.55),
            MosaicSegment(2, 192.2, 400.1, 192.2, 400.1, 1072, 2768, 2064, 1328, 0.94),
        ]
        right = [
            MosaicSegment(0, 5.5, 175.2, 5.5, 175.2, 1152, 2272, 2080, 1824, 0.83),
            MosaicSegment(1, 192.2, 400.1, 192.2, 400.1, 1104, 2496, 1904, 1600, 0.93),
        ]

        paired_left, paired_right = logic._pair_eye_segments_by_time(left, right)

        self.assertEqual([seg.seg_id for seg in paired_left], [0, 1])
        self.assertEqual([seg.seg_id for seg in paired_right], [0, 1])
        self.assertEqual([seg.y for seg in paired_left], [2288, 2768])
        self.assertEqual([seg.y for seg in paired_right], [2272, 2496])

    def test_fine_segments_are_clamped_to_single_eye_before_decode(self) -> None:
        segments = [MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 1900, 100, 300, 200, 0.91)]

        clamped = logic._clamp_eye_segments_for_decode("left", segments, eye_w=2048, eye_h=2048)

        self.assertEqual(len(clamped), 1)
        self.assertEqual((clamped[0].x, clamped[0].y, clamped[0].w, clamped[0].h), (1900, 100, 148, 200))

    def test_source_scan_uses_paired_pre_extract_for_sbs_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "source.mp4"
            final = Path(raw) / "source_sbs.restored.mp4"
            tmp_dir = Path(raw) / "source_sbs.restored_scan_tmp"
            mosaic = tmp_dir / "mosaic_seg000.mp4"
            restored = tmp_dir / "mosaic_seg000.restored.mp4"
            src.write_bytes(b"fake video")
            tmp_dir.mkdir()
            mosaic.write_bytes(b"mosaic")
            restored.write_bytes(b"restored")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=60.0, source_fps=30.0)
            timeline = [TimelineEntry(0.0, 60.0, mosaic, "mosaic", 0.91)]

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.source_time_scanner.scan_source_time_segments", return_value=[TimeInterval(0.0, 60.0, 0.91)]),
                patch("one_click.logic.get_video_bitrate", return_value=1000000),
                patch("utils.keyframe_cutter.cut_source_by_intervals", return_value=timeline),
                patch("one_click.logic._process_sbs_paired_pre_extract_clip", return_value=logic.PreExtractResult.OK) as paired,
                patch("one_click.logic._process_sbs_clip_to_output") as full_eye,
                patch("gpu_engine.files.replace_timeline_segments_gpu") as gpu_merge,
                patch("utils.sbs_concat.concat_timeline") as concat_timeline,
            ):
                result = logic._run_source_scan_branch(
                    str(src),
                    str(final),
                    use_fisheye=True,
                    pre_extract_inner=True,
                    keep_intermediate=False,
                    keep_original_bitrate=False,
                    fine_conf=0.5,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            paired.assert_called_once()
            self.assertEqual(paired.call_args.args[:2], (str(mosaic), str(restored)))
            self.assertTrue(paired.call_args.kwargs["use_fisheye"])
            self.assertEqual(paired.call_args.kwargs["fine_conf"], 0.5)
            full_eye.assert_not_called()
            gpu_merge.assert_called_once()
            self.assertEqual(gpu_merge.call_args.args[:3], (str(src), str(final), timeline))
            concat_timeline.assert_not_called()

    def test_source_scan_bypass_crop_falls_back_only_for_current_timeline_entry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "source.mp4"
            final = Path(raw) / "source_sbs.restored.mp4"
            tmp_dir = Path(raw) / "source_sbs.restored_scan_tmp"
            mosaic0 = tmp_dir / "mosaic_seg000.mp4"
            mosaic1 = tmp_dir / "mosaic_seg001.mp4"
            restored0 = tmp_dir / "mosaic_seg000.restored.mp4"
            restored1 = tmp_dir / "mosaic_seg001.restored.mp4"
            src.write_bytes(b"fake video")
            tmp_dir.mkdir()
            mosaic0.write_bytes(b"mosaic0")
            mosaic1.write_bytes(b"mosaic1")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=80.0, source_fps=30.0)
            timeline = [
                TimelineEntry(0.0, 30.0, mosaic0, "mosaic", 0.91),
                TimelineEntry(40.0, 70.0, mosaic1, "mosaic", 0.90),
            ]
            cfg = {"source_scan_final_merge_mode": "gpu"}

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.source_time_scanner.scan_source_time_segments", return_value=[
                    TimeInterval(0.0, 30.0, 0.91),
                    TimeInterval(40.0, 70.0, 0.90),
                ]),
                patch("one_click.logic.get_video_bitrate", return_value=1000000),
                patch("utils.keyframe_cutter.cut_source_by_intervals", return_value=timeline),
                patch(
                    "one_click.logic._process_sbs_paired_pre_extract_clip",
                    side_effect=[logic.PreExtractResult.BYPASS_CROP, logic.PreExtractResult.OK],
                ) as paired,
                patch("one_click.logic._process_sbs_clip_to_output") as full_eye,
                patch("gpu_engine.files.replace_timeline_segments_gpu") as gpu_merge,
            ):
                result = logic._run_source_scan_branch(
                    str(src),
                    str(final),
                    use_fisheye=False,
                    pre_extract_inner=True,
                    keep_intermediate=False,
                    keep_original_bitrate=False,
                    fine_conf=0.5,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            self.assertEqual(paired.call_count, 2)
            self.assertEqual(paired.call_args_list[0].args[:2], (str(mosaic0), str(restored0)))
            self.assertEqual(paired.call_args_list[1].args[:2], (str(mosaic1), str(restored1)))
            full_eye.assert_called_once()
            self.assertEqual(full_eye.call_args.args[:2], (str(mosaic0), str(restored0)))
            self.assertFalse(full_eye.call_args.kwargs["pre_extract_inner"])
            gpu_merge.assert_called_once()

    def test_pre_extract_cancel_does_not_fallback_to_full_restore(self) -> None:
        with (
            patch("one_click.logic._run_pre_extract_branch", return_value=logic.PreExtractResult.CANCELLED),
            patch("one_click.logic.process_lada") as process_lada,
        ):
            with self.assertRaises(OperationCancelled):
                logic._process_pre_extract_or_lada("base.mp4", "out.mp4", True)

        process_lada.assert_not_called()

    def test_source_scan_paired_cancel_does_not_fallback_to_full_eye_restore(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "source.mp4"
            final = Path(raw) / "source_sbs.restored.mp4"
            tmp_dir = Path(raw) / "source_sbs.restored_scan_tmp"
            mosaic = tmp_dir / "mosaic_seg000.mp4"
            restored = tmp_dir / "mosaic_seg000.restored.mp4"
            src.write_bytes(b"fake video")
            tmp_dir.mkdir()
            mosaic.write_bytes(b"mosaic")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=60.0, source_fps=30.0)
            timeline = [TimelineEntry(0.0, 60.0, mosaic, "mosaic", 0.91)]

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.source_time_scanner.scan_source_time_segments", return_value=[TimeInterval(0.0, 60.0, 0.91)]),
                patch("one_click.logic.get_video_bitrate", return_value=1000000),
                patch("utils.keyframe_cutter.cut_source_by_intervals", return_value=timeline),
                patch("one_click.logic._process_sbs_paired_pre_extract_clip", return_value=logic.PreExtractResult.CANCELLED),
                patch("one_click.logic._process_sbs_clip_to_output") as full_eye,
            ):
                with self.assertRaises(OperationCancelled):
                    logic._run_source_scan_branch(
                        str(src),
                        str(final),
                        use_fisheye=False,
                        pre_extract_inner=True,
                        keep_intermediate=False,
                        keep_original_bitrate=False,
                    )

            full_eye.assert_not_called()

    def test_paired_fisheye_path_patches_in_memory_without_full_fisheye_base(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "mosaic_seg000.mp4"
            out = root / "mosaic_seg000.restored.mp4"
            base.write_bytes(b"base")
            meta = VideoMetadata(path=str(base), width=4096, height=4096, duration=20.0, source_fps=30.0, bitrate_bps=2000000)
            left = [MosaicSegment(0, 1.0, 4.0, 1.0, 4.0, 100, 200, 512, 512, 0.91)]
            right = [MosaicSegment(0, 1.1, 3.9, 1.1, 3.9, 140, 220, 512, 512, 0.90)]
            cfg = {
                "pre_extract_fine_yolo_conf": 0.40,
                "pre_extract_pair_merge_gap_s": 0.75,
                "pre_extract_extract_group_max": 8,
                "pre_extract_keep_segments": False,
            }

            def fake_extract_multi(_src, tasks, **_kwargs):
                for task in tasks:
                    Path(task["dst"]).write_bytes(b"segment")

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.mosaic_prescan.scan_segments_gpu_transform", side_effect=[left, right]) as scan_gpu,
                patch("utils.mosaic_prescan.scan_segments_gpu_transform_pair") as scan_pair,
                patch("gpu_engine.files.extract_transformed_rect_clip") as extract_rect,
                patch("gpu_engine.files.extract_multi_rect_clip", side_effect=fake_extract_multi) as extract_multi,
                patch("one_click.logic.process_lada") as process_lada,
                patch("gpu_engine.files.paste_fisheye_eye_rects_to_sbs_gpu") as fish_patch,
                patch("gpu_engine.files.vr_projection") as vr_projection,
                patch("utils.segment_paster.paste_segments_gpu_or_fallback") as paste_fallback,
            ):
                result = logic._process_sbs_paired_pre_extract_clip(
                    str(base),
                    str(out),
                    use_fisheye=True,
                    keep_intermediate=False,
                    original_bitrate=2000000,
                    keep_original_bitrate=True,
                    fine_conf=0.6,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            scan_pair.assert_not_called()
            self.assertEqual([call.kwargs["crop_mode"] for call in scan_gpu.call_args_list], ["left", "right"])
            self.assertEqual([call.kwargs["to_fisheye"] for call in scan_gpu.call_args_list], [True, True])
            self.assertEqual([call.kwargs["min_conf"] for call in scan_gpu.call_args_list], [0.6, 0.6])
            extract_rect.assert_not_called()
            extract_multi.assert_called_once()
            # Intermediate stage is decoupled from keep_original_bitrate and
            # always reserves a 2x headroom for downstream re-encodes.
            # The area-scaled target (62500 bps) is dominated by the baseline
            # quality floor (0.015 * 512*512 * 30 fps), which intermediate stage
            # always applies regardless of keep_original_bitrate.
            area_scaled_target = int(2_000_000 * (512 * 512) / (4096 * 4096) * 2.0)
            baseline_target = int(0.015 * 512 * 512 * 30)
            expected_rect_bitrate = max(area_scaled_target, baseline_target)
            self.assertFalse(extract_multi.call_args.kwargs["keep_audio"])
            multi_tasks = extract_multi.call_args.args[1]
            self.assertEqual(len(multi_tasks), 2)
            for task in multi_tasks:
                self.assertEqual(task["bitrate_bps"], expected_rect_bitrate)
            self.assertEqual(process_lada.call_count, 2)
            for call in process_lada.call_args_list:
                self.assertTrue(call.kwargs["produce_mp4"])
                self.assertIn("rect", call.kwargs["sidecar_metadata"])
                self.assertIn("time", call.kwargs["sidecar_metadata"])
            extract_names = [Path(task["dst"]).name for task in multi_tasks]
            self.assertEqual(
                extract_names,
                [
                    "mosaic_seg000_L_fisheye.f00000033-00000117.r100_200_512x512.mp4",
                    "mosaic_seg000_R_fisheye.f00000033-00000117.r140_220_512x512.mp4",
                ],
            )
            vr_projection.assert_not_called()
            paste_fallback.assert_not_called()
            fish_patch.assert_called_once()
            self.assertEqual(fish_patch.call_args.args[:2], (str(base.resolve()), str(out.resolve())))
            paste_segments = fish_patch.call_args.args[2]
            self.assertEqual(len(paste_segments), 2)
            self.assertEqual(paste_segments[0].x, 100)
            self.assertEqual(paste_segments[1].x, 2048 + 140)
            self.assertFalse(any("_fisheye_sbs" in str(arg) for arg in fish_patch.call_args.args))
            self.assertFalse(fish_patch.call_args.kwargs["keep_audio"])
            self.assertEqual(fish_patch.call_args.kwargs["bitrate_bps"], 2000000)

    def test_paired_pre_extract_group_limit_falls_back_to_single_rect_extracts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "mosaic_seg000.mp4"
            out = root / "mosaic_seg000.restored.mp4"
            base.write_bytes(b"base")
            meta = VideoMetadata(path=str(base), width=4096, height=4096, duration=20.0, source_fps=30.0, bitrate_bps=2000000)
            left = [MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 100, 200, 512, 512, 0.91)]
            right = [MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 140, 220, 512, 512, 0.90)]
            cfg = {
                "pre_extract_extract_group_max": 1,
                "pre_extract_keep_segments": False,
            }

            def fake_extract_rect(_src, dst, **_kwargs):
                Path(dst).write_bytes(b"segment")

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.mosaic_prescan.scan_segments_gpu_transform_pair", return_value=(left, right)),
                patch("gpu_engine.files.extract_transformed_rect_clip", side_effect=fake_extract_rect) as extract_rect,
                patch("gpu_engine.files.extract_multi_rect_clip") as extract_multi,
                patch("one_click.logic.process_lada"),
                patch("utils.segment_paster.paste_segments_gpu_or_fallback") as paste_fallback,
            ):
                result = logic._process_sbs_paired_pre_extract_clip(
                    str(base),
                    str(out),
                    use_fisheye=False,
                    keep_intermediate=False,
                    original_bitrate=2000000,
                    keep_original_bitrate=True,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            self.assertEqual(extract_rect.call_count, 2)
            extract_multi.assert_not_called()
            paste_fallback.assert_called_once()

    def test_paired_pre_extract_processes_high_confidence_unmatched_eye_segment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "mosaic_seg000.mp4"
            out = root / "mosaic_seg000.restored.mp4"
            base.write_bytes(b"base")
            meta = VideoMetadata(path=str(base), width=4096, height=4096, duration=20.0, source_fps=30.0, bitrate_bps=2000000)
            left = [MosaicSegment(0, 3.0, 5.0, 3.0, 5.0, 100, 200, 512, 512, 0.88)]
            cfg = {
                "pre_extract_keep_segments": False,
                "pre_extract_pair_keep_unmatched_conf": 0.60,
            }

            def fake_extract_rect(_src, dst, **_kwargs):
                Path(dst).write_bytes(b"segment")

            def fake_process_lada(_src, dst, **_kwargs):
                Path(dst).write_bytes(b"restored")
                return str(dst)

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.mosaic_prescan.scan_segments_gpu_transform_pair", return_value=(left, [])),
                patch("gpu_engine.files.extract_transformed_rect_clip", side_effect=fake_extract_rect) as extract_rect,
                patch("gpu_engine.files.extract_multi_rect_clip") as extract_multi,
                patch("one_click.logic.process_lada", side_effect=fake_process_lada) as process_lada,
                patch("utils.segment_paster.paste_segments_gpu_or_fallback") as paste_fallback,
            ):
                result = logic._process_sbs_paired_pre_extract_clip(
                    str(base),
                    str(out),
                    use_fisheye=False,
                    keep_intermediate=False,
                    original_bitrate=2000000,
                    keep_original_bitrate=True,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            extract_rect.assert_called_once()
            extract_multi.assert_not_called()
            process_lada.assert_called_once()
            paste_fallback.assert_called_once()
            paste_segments = paste_fallback.call_args.args[2]
            self.assertEqual(len(paste_segments), 1)
            self.assertEqual(paste_segments[0].x, 100)

    def test_paired_pre_extract_bypasses_crop_for_large_single_eye_area(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "mosaic_seg000.mp4"
            out = root / "mosaic_seg000.restored.mp4"
            base.write_bytes(b"base")
            meta = VideoMetadata(path=str(base), width=4096, height=4096, duration=20.0, source_fps=30.0, bitrate_bps=2000000)
            left = [MosaicSegment(0, 1.0, 4.0, 1.0, 4.0, 0, 0, 2048, 2048, 0.91)]
            right = [MosaicSegment(0, 1.0, 4.0, 1.0, 4.0, 140, 220, 512, 512, 0.90)]
            cfg = {
                "pre_extract_bypass_crop_area_ratio": 1.0 / 3.0,
                "pre_extract_pair_keep_unmatched_conf": 0.60,
            }
            messages: list[str] = []

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.mosaic_prescan.scan_segments_gpu_transform_pair", return_value=(left, right)),
                patch("gpu_engine.files.extract_transformed_rect_clip") as extract_rect,
                patch("gpu_engine.files.extract_multi_rect_clip") as extract_multi,
                patch("one_click.logic.process_lada") as process_lada,
                patch("utils.segment_paster.paste_segments_gpu_or_fallback") as paste_fallback,
            ):
                result = logic._process_sbs_paired_pre_extract_clip(
                    str(base),
                    str(out),
                    use_fisheye=False,
                    keep_intermediate=False,
                    original_bitrate=2000000,
                    keep_original_bitrate=True,
                    log_callback=messages.append,
                    fine_conf=0.6,
                )

            self.assertEqual(result, logic.PreExtractResult.BYPASS_CROP)
            extract_rect.assert_not_called()
            extract_multi.assert_not_called()
            process_lada.assert_not_called()
            paste_fallback.assert_not_called()
            self.assertTrue(any("paired fine crop bypassed" in msg for msg in messages))

    def test_paired_pre_extract_pipelines_next_group_extract_during_restore(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "mosaic_seg000.mp4"
            out = root / "mosaic_seg000.restored.mp4"
            base.write_bytes(b"base")
            meta = VideoMetadata(path=str(base), width=4096, height=4096, duration=20.0, source_fps=30.0, bitrate_bps=2000000)
            left = [
                MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 100, 200, 512, 512, 0.91),
                MosaicSegment(1, 5.0, 6.0, 5.0, 6.0, 120, 220, 512, 512, 0.92),
            ]
            right = [
                MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 140, 220, 512, 512, 0.90),
                MosaicSegment(1, 5.0, 6.0, 5.0, 6.0, 160, 240, 512, 512, 0.93),
            ]
            cfg = {
                "pre_extract_extract_group_max": 8,
                "pre_extract_pipeline_enabled": True,
                "pre_extract_keep_segments": False,
            }
            first_restore_started = threading.Event()
            second_extract_started = threading.Event()
            extract_count = {"value": 0}
            restore_count = {"value": 0}

            def fake_extract_multi(_src, tasks, **_kwargs):
                extract_count["value"] += 1
                if extract_count["value"] == 2:
                    self.assertTrue(first_restore_started.wait(2.0), "second extract ran before restore could start")
                    second_extract_started.set()
                for task in tasks:
                    Path(task["dst"]).write_bytes(b"segment")

            def fake_process_lada(_src, dst, **_kwargs):
                restore_count["value"] += 1
                if restore_count["value"] == 1:
                    first_restore_started.set()
                    self.assertTrue(
                        second_extract_started.wait(2.0),
                        "next extract group did not run while first restore was active",
                    )
                Path(dst).write_bytes(b"restored")
                return str(dst)

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.mosaic_prescan.scan_segments_gpu_transform_pair", return_value=(left, right)),
                patch("gpu_engine.files.extract_transformed_rect_clip") as extract_rect,
                patch("gpu_engine.files.extract_multi_rect_clip", side_effect=fake_extract_multi) as extract_multi,
                patch("one_click.logic.process_lada", side_effect=fake_process_lada) as process_lada,
                patch("utils.segment_paster.paste_segments_gpu_or_fallback") as paste_fallback,
            ):
                result = logic._process_sbs_paired_pre_extract_clip(
                    str(base),
                    str(out),
                    use_fisheye=False,
                    keep_intermediate=False,
                    original_bitrate=2000000,
                    keep_original_bitrate=True,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            extract_rect.assert_not_called()
            self.assertEqual(extract_multi.call_count, 2)
            self.assertEqual(process_lada.call_count, 4)
            paste_fallback.assert_called_once()

    def test_paired_pre_extract_pipeline_propagates_extract_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "mosaic_seg000.mp4"
            out = root / "mosaic_seg000.restored.mp4"
            base.write_bytes(b"base")
            meta = VideoMetadata(path=str(base), width=4096, height=4096, duration=20.0, source_fps=30.0, bitrate_bps=2000000)
            left = [
                MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 100, 200, 512, 512, 0.91),
                MosaicSegment(1, 5.0, 6.0, 5.0, 6.0, 120, 220, 512, 512, 0.92),
            ]
            right = [
                MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 140, 220, 512, 512, 0.90),
                MosaicSegment(1, 5.0, 6.0, 5.0, 6.0, 160, 240, 512, 512, 0.93),
            ]
            cfg = {
                "pre_extract_extract_group_max": 8,
                "pre_extract_pipeline_enabled": True,
                "pre_extract_keep_segments": False,
            }
            extract_count = {"value": 0}

            def fake_extract_multi(_src, tasks, **_kwargs):
                extract_count["value"] += 1
                if extract_count["value"] == 2:
                    raise RuntimeError("extract failed")
                for task in tasks:
                    Path(task["dst"]).write_bytes(b"segment")

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.mosaic_prescan.scan_segments_gpu_transform_pair", return_value=(left, right)),
                patch("gpu_engine.files.extract_multi_rect_clip", side_effect=fake_extract_multi),
                patch("one_click.logic.process_lada", side_effect=lambda _src, dst, **_kwargs: str(dst)),
                patch("utils.segment_paster.paste_segments_gpu_or_fallback") as paste_fallback,
            ):
                with self.assertRaisesRegex(RuntimeError, "extract failed"):
                    logic._process_sbs_paired_pre_extract_clip(
                        str(base),
                        str(out),
                        use_fisheye=False,
                        keep_intermediate=False,
                        original_bitrate=2000000,
                        keep_original_bitrate=True,
                    )

            paste_fallback.assert_not_called()

    def test_paired_pre_extract_uses_frame_key_cache_and_removes_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "mosaic_seg000.mp4"
            out = root / "mosaic_seg000.restored.mp4"
            base.write_bytes(b"base")
            meta = VideoMetadata(path=str(base), width=4096, height=4096, duration=20.0, source_fps=30.0, bitrate_bps=2000000)
            left = [MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 100, 200, 512, 512, 0.91)]
            right = [MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 140, 220, 512, 512, 0.90)]
            expected_files = [
                root / "mosaic_seg000_L.f00000030-00000060.r100_200_512x512.mp4",
                root / "mosaic_seg000_L.f00000030-00000060.r100_200_512x512.restored.mp4",
                root / "mosaic_seg000_R.f00000030-00000060.r140_220_512x512.mp4",
                root / "mosaic_seg000_R.f00000030-00000060.r140_220_512x512.restored.mp4",
            ]
            for path in expected_files:
                path.write_bytes(b"cache")
            orphans = [
                root / "mosaic_seg000_L.seg000.mp4",
                root / "mosaic_seg000_R.seg000.restored.mp4",
                root / "mosaic_seg000_L.f00000030-00000060.r999_999_16x16.mp4",
            ]
            for path in orphans:
                path.write_bytes(b"old")

            with (
                patch("utils.app_config.get", side_effect=lambda key, default=None: default),
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.mosaic_prescan.scan_segments_gpu_transform_pair", return_value=(left, right)),
                patch("gpu_engine.files.extract_transformed_rect_clip") as extract_rect,
                patch("gpu_engine.files.extract_multi_rect_clip") as extract_multi,
                patch("one_click.logic.process_lada") as process_lada,
                patch("utils.segment_paster.paste_segments_gpu_or_fallback") as paste_fallback,
            ):
                result = logic._process_sbs_paired_pre_extract_clip(
                    str(base),
                    str(out),
                    use_fisheye=False,
                    keep_intermediate=True,
                    original_bitrate=2000000,
                    keep_original_bitrate=True,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            extract_rect.assert_not_called()
            extract_multi.assert_not_called()
            process_lada.assert_not_called()
            paste_fallback.assert_called_once()
            for path in expected_files:
                self.assertTrue(path.exists(), path.name)
            for path in orphans:
                self.assertFalse(path.exists(), path.name)

    def test_source_scan_no_mosaic_skips_without_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "source.mp4"
            final = Path(raw) / "source_sbs.restored.mp4"
            detections = Path(raw) / "source.detections.jsonl"
            src.write_bytes(b"fake video")
            detections.write_text("{}\n", encoding="utf-8")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=60.0, source_fps=30.0)

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.source_time_scanner.scan_source_time_segments", return_value=[]),
            ):
                result = logic._run_source_scan_branch(
                    str(src),
                    str(final),
                    use_fisheye=False,
                    pre_extract_inner=False,
                    keep_intermediate=False,
                    keep_original_bitrate=False,
                )

            self.assertEqual(result, logic.PreExtractResult.NO_MOSAIC)
            self.assertFalse(final.exists())
            self.assertFalse((Path(raw) / "source_sbs.restored.source_intervals.json").exists())
            self.assertFalse(detections.exists())

    def test_source_scan_uses_keyframe_cut_and_gpu_timeline_merge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "source.mp4"
            final = Path(raw) / "source_sbs.restored.mp4"
            tmp_dir = Path(raw) / "source_sbs.restored_scan_tmp"
            mosaic = tmp_dir / "mosaic_seg000.mp4"
            src.write_bytes(b"fake video")
            tmp_dir.mkdir()
            mosaic.write_bytes(b"mosaic")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=60.0, source_fps=30.0)
            timeline = [
                TimelineEntry(0.0, 60.0, mosaic, "mosaic", 0.91),
            ]

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.source_time_scanner.scan_source_time_segments", return_value=[TimeInterval(0.0, 60.0, 0.91)]),
                patch("one_click.logic.get_video_bitrate", return_value=1000000),
                patch("one_click.logic._process_sbs_clip_to_output") as process_source,
                patch("utils.keyframe_cutter.cut_source_by_intervals", return_value=timeline) as cut_source,
                patch("utils.sbs_concat.concat_timeline") as concat_timeline,
                patch("utils.sbs_concat.concat_timeline_hevc_fast") as fast_merge,
                patch("gpu_engine.files.replace_timeline_segments_gpu") as gpu_merge,
            ):
                result = logic._run_source_scan_branch(
                    str(src),
                    str(final),
                    use_fisheye=False,
                    pre_extract_inner=False,
                    keep_intermediate=False,
                    keep_original_bitrate=False,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            process_source.assert_called_once()
            self.assertEqual(process_source.call_args.args[:2], (str(mosaic), str(mosaic.with_name("mosaic_seg000.restored.mp4"))))
            cut_source.assert_called_once()
            concat_timeline.assert_not_called()
            fast_merge.assert_called_once()
            self.assertEqual(fast_merge.call_args.args[:2], (timeline, str(final)))
            self.assertEqual(fast_merge.call_args.kwargs["source_src"], str(src))
            self.assertEqual(fast_merge.call_args.kwargs["audio_source"], str(src))
            gpu_merge.assert_not_called()
            # Materialize the gap in SBS mode to avoid IDR drift from fast HEVC merge
            # seeking the same source file twice.
            self.assertTrue(cut_source.call_args.kwargs["materialize_gaps"])
            self.assertTrue(cut_source.call_args.kwargs["materialize_mosaic"])
            self.assertEqual(timeline[0].path, mosaic.with_name("mosaic_seg000.restored.mp4"))
            self.assertIsNone(timeline[0].inpoint_s)
            self.assertIsNone(timeline[0].outpoint_s)

    def test_source_scan_gpu_timeline_merge_uses_original_bitrate_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "source.mp4"
            final = Path(raw) / "source_sbs.restored.mp4"
            tmp_dir = Path(raw) / "source_sbs.restored_scan_tmp"
            mosaic = tmp_dir / "mosaic_seg000.mp4"
            src.write_bytes(b"fake video")
            tmp_dir.mkdir()
            mosaic.write_bytes(b"mosaic")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=60.0, source_fps=30.0)
            timeline = [TimelineEntry(0.0, 60.0, mosaic, "mosaic", 0.91)]

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.source_time_scanner.scan_source_time_segments", return_value=[TimeInterval(0.0, 60.0, 0.91)]),
                patch("one_click.logic.get_video_bitrate", return_value=1000000),
                patch("one_click.logic._process_sbs_clip_to_output"),
                patch("utils.keyframe_cutter.cut_source_by_intervals", return_value=timeline),
                patch("utils.sbs_concat.concat_timeline_hevc_fast", side_effect=RuntimeError("fast failed")),
                patch("gpu_engine.files.replace_timeline_segments_gpu") as gpu_merge,
            ):
                result = logic._run_source_scan_branch(
                    str(src),
                    str(final),
                    use_fisheye=False,
                    pre_extract_inner=False,
                    keep_intermediate=False,
                    keep_original_bitrate=True,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            gpu_merge.assert_called_once()
            self.assertEqual(gpu_merge.call_args.kwargs["bitrate_bps"], 1000000)

    def test_single_eye_source_scan_crops_gap_entries_before_concat(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "source.mp4"
            final = Path(raw) / "source_left.restored.mp4"
            tmp = Path(raw) / "source_left.restored_scan_tmp"
            mosaic = tmp / "mosaic_seg000.mp4"
            src.write_bytes(b"fake video")
            tmp.mkdir()
            mosaic.write_bytes(b"mosaic")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=60.0, source_fps=30.0)
            timeline = [
                TimelineEntry(0.0, 10.0, src, "gap", inpoint_s=0.0, outpoint_s=10.0),
                TimelineEntry(10.0, 20.0, mosaic, "mosaic", 0.91),
            ]

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.source_time_scanner.scan_source_time_segments", return_value=[TimeInterval(10.0, 20.0, 0.91)]),
                patch("one_click.logic.get_video_bitrate", return_value=1000000),
                patch("one_click.logic._process_single_eye_clip_to_output") as process_eye,
                patch("utils.keyframe_cutter.cut_source_by_intervals", return_value=timeline) as cut_source,
                patch("gpu_engine.files.extract_clip") as extract_clip,
                patch("utils.sbs_concat.concat_timeline") as concat_timeline,
            ):
                result = logic._run_source_scan_branch(
                    str(src),
                    str(final),
                    use_fisheye=False,
                    pre_extract_inner=True,
                    keep_intermediate=False,
                    keep_original_bitrate=True,
                    mode="single_eye",
                    eye_mode=1,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            process_eye.assert_called_once()
            self.assertFalse(cut_source.call_args.kwargs["materialize_gaps"])
            self.assertTrue(cut_source.call_args.kwargs["materialize_mosaic"])
            self.assertEqual(process_eye.call_args.args[:2], (str(mosaic), str(mosaic.with_name("mosaic_seg000.restored.mp4"))))
            extract_clip.assert_called_once()
            self.assertEqual(extract_clip.call_args.args[:2], (src, tmp / "gap_seg000_left.mp4"))
            self.assertEqual(extract_clip.call_args.kwargs["crop_mode"], "left")
            self.assertEqual(extract_clip.call_args.kwargs["start_sec"], 0.0)
            self.assertEqual(extract_clip.call_args.kwargs["end_sec"], 10.0)
            self.assertEqual(extract_clip.call_args.kwargs["bitrate_bps"], 500000)
            self.assertFalse(extract_clip.call_args.kwargs["keep_audio"])
            concat_timeline.assert_called_once()
            self.assertEqual(concat_timeline.call_args.kwargs["audio_source"], str(src))
            self.assertEqual(timeline[0].path, Path(raw) / "source_left.restored_scan_tmp" / "gap_seg000_left.mp4")
            self.assertEqual(timeline[1].path, mosaic.with_name("mosaic_seg000.restored.mp4"))
            self.assertIsNone(timeline[0].inpoint_s)
            self.assertIsNone(timeline[1].inpoint_s)


if __name__ == "__main__":
    unittest.main()
