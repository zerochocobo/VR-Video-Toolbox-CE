import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine import pynv_io


class PyNvThreadedDecoderPrerollTests(unittest.TestCase):
    def test_preroll_uses_previous_keyframe_frame(self) -> None:
        def index_at_time(seconds: float) -> int:
            mapping = {
                0.0: 0,
                222.255: 13320,
                227.260: 13622,
            }
            return mapping[round(seconds, 3)]

        with patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=(0.0, 222.255, 227.260)):
            self.assertEqual(
                pynv_io._threaded_decoder_preroll_frame(
                    Path("clip.mp4"),
                    13620,
                    60000 / 1001,
                    14762,
                    index_at_time=index_at_time,
                ),
                13320,
            )

    def test_preroll_keeps_exact_keyframe_start(self) -> None:
        with patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=(0.0, 5.0, 10.0)):
            self.assertEqual(
                pynv_io._threaded_decoder_preroll_frame(
                    Path("clip.mp4"),
                    150,
                    30.0,
                    1000,
                    index_at_time=lambda seconds: int(round(seconds * 30.0)),
                ),
                150,
            )

    def test_preroll_falls_back_to_target_without_keyframes(self) -> None:
        with patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=()):
            self.assertEqual(
                pynv_io._threaded_decoder_preroll_frame(Path("clip.mp4"), 152, 30.0, 1000),
                152,
            )

    def test_decoder_constructor_starts_threaded_decoder_at_preroll_frame(self) -> None:
        class FakeMeta:
            width = 8192
            height = 4096
            average_fps = 60000 / 1001
            duration = 246.279
            codec_name = "hevc"
            bitrate = 33000000
            num_frames = 14762

        class FakeSimpleDecoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_stream_metadata(self):
                return FakeMeta()

            def __len__(self):
                return 14762

            def __getitem__(self, _index):
                raise RuntimeError("pts probe is not needed for this test")

            def get_index_from_time_in_seconds(self, seconds: float) -> int:
                mapping = {
                    0.0: 0,
                    222.255: 13320,
                    227.260: 13622,
                }
                return mapping[round(float(seconds), 3)]

            def stop(self):
                pass

        class FakeThreadedDecoder:
            created_start_frames: list[int] = []

            def __init__(self, _path, _buffer_size, **kwargs):
                self.created_start_frames.append(int(kwargs["start_frame"]))

            def end(self):
                pass

        fake_nvc = types.SimpleNamespace(
            SimpleDecoder=FakeSimpleDecoder,
            ThreadedDecoder=FakeThreadedDecoder,
            OutputColorType=types.SimpleNamespace(NATIVE=object()),
        )

        with (
            patch.dict(sys.modules, {"PyNvVideoCodec": fake_nvc}),
            patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=(0.0, 222.255, 227.260)),
        ):
            dec = pynv_io.PyNvThreadedSerialDecoder(Path("clip.mp4"), start_frame=13620)
            try:
                self.assertEqual(dec._decode_start_frame, 13320)
                self.assertEqual(FakeThreadedDecoder.created_start_frames[-1], 13320)
                self.assertEqual(dec._next_source_idx, 13320)
            finally:
                dec.stop()

    def test_decoder_calibrates_preroll_batch_when_threaded_decoder_starts_after_keyframe(self) -> None:
        class FakeDecodedFrame:
            def __init__(self, index: int):
                self.index = int(index)

            def getPTS(self) -> int:
                return self.index * 1001

        class FakeMeta:
            width = 8192
            height = 4096
            average_fps = 60000 / 1001
            duration = 246.279
            codec_name = "hevc"
            bitrate = 33000000
            num_frames = 14762

        class FakeSimpleDecoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_stream_metadata(self):
                return FakeMeta()

            def __len__(self):
                return 14762

            def __getitem__(self, index):
                return FakeDecodedFrame(index)

            def get_index_from_time_in_seconds(self, seconds: float) -> int:
                mapping = {
                    0.0: 0,
                    35.068: 2102,
                    40.073: 2402,
                }
                return mapping[round(float(seconds), 3)]

            def stop(self):
                pass

        class FakeThreadedDecoder:
            def __init__(self, _path, _buffer_size, **kwargs):
                self.next_index = int(kwargs["start_frame"]) + 2

            def get_batch_frames(self, batch_size: int):
                batch = [FakeDecodedFrame(self.next_index + offset) for offset in range(int(batch_size))]
                self.next_index += int(batch_size)
                return batch

            def end(self):
                pass

        fake_nvc = types.SimpleNamespace(
            SimpleDecoder=FakeSimpleDecoder,
            ThreadedDecoder=FakeThreadedDecoder,
            OutputColorType=types.SimpleNamespace(NATIVE=object()),
        )

        with (
            patch.dict(sys.modules, {"PyNvVideoCodec": fake_nvc}),
            patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=(0.0, 35.068, 40.073)),
            patch(
                "gpu_engine.pynv_io.GpuNv12Frame.from_decoded_frame",
                side_effect=lambda frame, _w, _h: types.SimpleNamespace(pts=frame.getPTS()),
            ),
        ):
            dec = pynv_io.PyNvThreadedSerialDecoder(Path("clip.mp4"), start_frame=2400, batch_size=64)
            try:
                frame = dec.frame_at(2400)
                self.assertEqual(dec._decode_start_frame, 2102)
                self.assertEqual(frame.pts, 2400 * 1001)
            finally:
                dec.stop()

    def test_decoder_accepts_threaded_pts_shifted_by_container_origin(self) -> None:
        origin_delta = 2970
        target_frame = 1770

        class FakeDecodedFrame:
            def __init__(self, index: int, pts: int):
                self.index = int(index)
                self.pts = int(pts)

            def getPTS(self) -> int:
                return self.pts

        class FakeMeta:
            width = 4096
            height = 2048
            average_fps = 30.0
            duration = 128.0
            codec_name = "h264"
            bitrate = 18000000
            num_frames = 3842

        class FakeSimpleDecoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_stream_metadata(self):
                return FakeMeta()

            def __len__(self):
                return 3842

            def __getitem__(self, index):
                return FakeDecodedFrame(index, int(index) * 1000)

            def get_index_from_time_in_seconds(self, seconds: float) -> int:
                mapping = {
                    0.0: 0,
                    51.8: 1554,
                    60.0: 1800,
                }
                return mapping[round(float(seconds), 3)]

            def stop(self):
                pass

        class FakeThreadedDecoder:
            def __init__(self, _path, _buffer_size, **kwargs):
                self.next_index = int(kwargs["start_frame"])

            def get_batch_frames(self, batch_size: int):
                batch = [
                    FakeDecodedFrame(index, index * 1000 + origin_delta)
                    for index in range(self.next_index, self.next_index + int(batch_size))
                ]
                self.next_index += int(batch_size)
                return batch

            def end(self):
                pass

        fake_nvc = types.SimpleNamespace(
            SimpleDecoder=FakeSimpleDecoder,
            ThreadedDecoder=FakeThreadedDecoder,
            OutputColorType=types.SimpleNamespace(NATIVE=object()),
        )

        with (
            patch.dict(sys.modules, {"PyNvVideoCodec": fake_nvc}),
            patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=(0.0, 51.8, 60.0)),
            patch("gpu_engine.pynv_io._first_frame_pts_for_path", return_value=origin_delta),
            patch(
                "gpu_engine.pynv_io.GpuNv12Frame.from_decoded_frame",
                side_effect=lambda frame, _w, _h: types.SimpleNamespace(index=frame.index, pts=frame.getPTS()),
            ),
        ):
            dec = pynv_io.PyNvThreadedSerialDecoder(Path("clip.mp4"), start_frame=target_frame, batch_size=64)
            try:
                frame = dec.frame_at(target_frame)
                self.assertEqual(dec._decode_start_frame, 1554)
                self.assertEqual(dec._threaded_pts_delta, origin_delta)
                self.assertEqual(frame.index, target_frame)
                self.assertEqual(frame.pts, target_frame * 1000 + origin_delta)
            finally:
                dec.stop()

    def test_decoder_accepts_small_pts_residual_after_origin_normalization(self) -> None:
        origin_delta = 2970
        pts_step = 1501
        residual = 86
        target_frame = 2700

        class FakeDecodedFrame:
            def __init__(self, index: int, pts: int):
                self.index = int(index)
                self.pts = int(pts)

            def getPTS(self) -> int:
                return self.pts

        class FakeMeta:
            width = 4096
            height = 2048
            average_fps = 60000 / 1001
            duration = 64.0
            codec_name = "h264"
            bitrate = 30000000
            num_frames = 3842

        class FakeSimpleDecoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_stream_metadata(self):
                return FakeMeta()

            def __len__(self):
                return 3842

            def __getitem__(self, index):
                return FakeDecodedFrame(index, int(index) * pts_step)

            def get_index_from_time_in_seconds(self, seconds: float) -> int:
                mapping = {
                    0.0: 0,
                    41.942: 2514,
                    50.05: 3000,
                }
                return mapping[round(float(seconds), 3)]

            def stop(self):
                pass

        class FakeThreadedDecoder:
            def __init__(self, _path, _buffer_size, **kwargs):
                self.next_index = int(kwargs["start_frame"])

            def get_batch_frames(self, batch_size: int):
                batch = []
                for index in range(self.next_index, self.next_index + int(batch_size)):
                    pts = index * pts_step + origin_delta
                    if index == target_frame:
                        pts += residual
                    batch.append(FakeDecodedFrame(index, pts))
                self.next_index += int(batch_size)
                return batch

            def end(self):
                pass

        fake_nvc = types.SimpleNamespace(
            SimpleDecoder=FakeSimpleDecoder,
            ThreadedDecoder=FakeThreadedDecoder,
            OutputColorType=types.SimpleNamespace(NATIVE=object()),
        )

        with (
            patch.dict(sys.modules, {"PyNvVideoCodec": fake_nvc}),
            patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=(0.0, 41.942, 50.05)),
            patch("gpu_engine.pynv_io._first_frame_pts_for_path", return_value=origin_delta),
            patch(
                "gpu_engine.pynv_io.GpuNv12Frame.from_decoded_frame",
                side_effect=lambda frame, _w, _h: types.SimpleNamespace(index=frame.index, pts=frame.getPTS()),
            ),
        ):
            dec = pynv_io.PyNvThreadedSerialDecoder(Path("clip.mp4"), start_frame=target_frame, batch_size=64)
            try:
                frame = dec.frame_at(target_frame)
                self.assertEqual(dec._decode_start_frame, 2514)
                self.assertEqual(dec._pts_match_tolerance, 150)
                self.assertEqual(frame.index, target_frame)
                self.assertEqual(frame.pts, target_frame * pts_step + origin_delta + residual)
            finally:
                dec.stop()

    def test_decoder_rejects_pts_mismatch_after_origin_normalization(self) -> None:
        origin_delta = 2970
        target_frame = 1770

        class FakeDecodedFrame:
            def __init__(self, index: int, pts: int):
                self.index = int(index)
                self.pts = int(pts)

            def getPTS(self) -> int:
                return self.pts

        class FakeMeta:
            width = 4096
            height = 2048
            average_fps = 30.0
            duration = 128.0
            codec_name = "h264"
            bitrate = 18000000
            num_frames = 3842

        class FakeSimpleDecoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_stream_metadata(self):
                return FakeMeta()

            def __len__(self):
                return 3842

            def __getitem__(self, index):
                return FakeDecodedFrame(index, int(index) * 1000)

            def get_index_from_time_in_seconds(self, seconds: float) -> int:
                mapping = {
                    0.0: 0,
                    51.8: 1554,
                    60.0: 1800,
                }
                return mapping[round(float(seconds), 3)]

            def stop(self):
                pass

        class FakeThreadedDecoder:
            def __init__(self, _path, _buffer_size, **kwargs):
                self.next_index = int(kwargs["start_frame"])

            def get_batch_frames(self, batch_size: int):
                batch = []
                for index in range(self.next_index, self.next_index + int(batch_size)):
                    pts = index * 1000 + origin_delta
                    if index == target_frame:
                        pts += 123
                    batch.append(FakeDecodedFrame(index, pts))
                self.next_index += int(batch_size)
                return batch

            def end(self):
                pass

        fake_nvc = types.SimpleNamespace(
            SimpleDecoder=FakeSimpleDecoder,
            ThreadedDecoder=FakeThreadedDecoder,
            OutputColorType=types.SimpleNamespace(NATIVE=object()),
        )

        with (
            patch.dict(sys.modules, {"PyNvVideoCodec": fake_nvc}),
            patch("gpu_engine.pynv_io._keyframe_times_for_path", return_value=(0.0, 51.8, 60.0)),
            patch("gpu_engine.pynv_io._first_frame_pts_for_path", return_value=origin_delta),
            patch(
                "gpu_engine.pynv_io.GpuNv12Frame.from_decoded_frame",
                side_effect=lambda frame, _w, _h: types.SimpleNamespace(index=frame.index, pts=frame.getPTS()),
            ),
        ):
            dec = pynv_io.PyNvThreadedSerialDecoder(Path("clip.mp4"), start_frame=target_frame, batch_size=64)
            try:
                with self.assertRaisesRegex(RuntimeError, "normalized_delta=123"):
                    dec.frame_at(target_frame)
            finally:
                dec.stop()


if __name__ == "__main__":
    unittest.main()
