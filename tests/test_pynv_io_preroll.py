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


if __name__ == "__main__":
    unittest.main()
