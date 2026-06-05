from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gpu_engine.fallback import OperationCancelled
from gpu_engine import files as gpu_files
from gpu_engine.probe import VideoMetadata


class ExtractMultiRectTests(unittest.TestCase):
    def test_multi_rect_extract_decodes_shared_window_once(self) -> None:
        with patch.dict(
            "sys.modules",
            {
                "cupy": types.SimpleNamespace(
                    cuda=types.SimpleNamespace(
                        Device=lambda: types.SimpleNamespace(synchronize=lambda: None)
                    )
                )
            },
        ):
            with self.subTest("shared decoder"):
                self._run_shared_decoder_case()

    def _run_shared_decoder_case(self) -> None:
        import tempfile

        frame_indices: list[int] = []
        stopped = {"value": False}

        class FakeFrame:
            def y_uv_cupy(self):
                y = np.zeros((4, 8), dtype=np.uint8)
                uv = np.zeros((2, 4, 2), dtype=np.uint8)
                return y, uv

        class FakeDecoder:
            def __init__(self, _src, bit_depth=8, start_frame=0):
                self.info = types.SimpleNamespace(width=8, height=4, fps=30.0)
                self.start_frame = start_frame

            def __len__(self):
                return 3

            def frame_at(self, index):
                frame_indices.append(index)
                return FakeFrame()

            def stop(self):
                stopped["value"] = True

        class FakeEncoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def encode(self, _app, force_idr=False):
                return b"frame"

            def flush(self):
                return b"tail"

        def fake_mux(raw_hevc, out_path, **_kwargs):
            self.assertTrue(Path(raw_hevc).exists())
            Path(out_path).write_bytes(b"mp4")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "base.mp4"
            src.write_bytes(b"base")
            out_l = root / "left.mp4"
            out_r = root / "right.mp4"
            meta = VideoMetadata(path=str(src), width=8, height=4, source_fps=30.0, bitrate_bps=3000)

            with (
                patch("gpu_engine.files.probe.probe_video", return_value=meta),
                patch("gpu_engine.files.PyNvThreadedSerialDecoder", FakeDecoder),
                patch("gpu_engine.files.PyNvEncoderSession", FakeEncoder),
                patch("gpu_engine.files._pack_planes", return_value="app"),
                patch("gpu_engine.files.runtime.free_memory_pool"),
                patch("gpu_engine.files.mux.mux_hevc_with_audio", side_effect=fake_mux),
            ):
                outs = gpu_files.extract_multi_rect_clip(
                    src,
                    [
                        {"dst": out_l, "crop_mode": "left", "rect": (0, 0, 2, 2), "bitrate_bps": 1000},
                        {"dst": out_r, "crop_mode": "right", "rect": (0, 0, 2, 2), "bitrate_bps": 1000},
                    ],
                    start_sec=0.0,
                    end_sec=0.1,
                    keep_audio=False,
                )

        self.assertEqual([Path(p).name for p in outs], ["left.mp4", "right.mp4"])
        self.assertEqual(frame_indices, [0, 1, 2])
        self.assertTrue(stopped["value"])

    def test_multi_rect_extract_cleans_raws_on_cancel(self) -> None:
        with patch.dict(
            "sys.modules",
            {
                "cupy": types.SimpleNamespace(
                    cuda=types.SimpleNamespace(
                        Device=lambda: types.SimpleNamespace(synchronize=lambda: None)
                    )
                )
            },
        ):
            self._run_cancel_cleanup_case()

    def _run_cancel_cleanup_case(self) -> None:
        import tempfile

        class FakeFrame:
            def y_uv_cupy(self):
                return np.zeros((4, 8), dtype=np.uint8), np.zeros((2, 4, 2), dtype=np.uint8)

        class FakeDecoder:
            def __init__(self, _src, bit_depth=8, start_frame=0):
                self.info = types.SimpleNamespace(width=8, height=4, fps=30.0)

            def __len__(self):
                return 2

            def frame_at(self, _index):
                return FakeFrame()

            def stop(self):
                pass

        class FakeEncoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def encode(self, _app, force_idr=False):
                return b"frame"

            def flush(self):
                return b"tail"

        token = gpu_files.CancelToken()
        token.kill()

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "base.mp4"
            src.write_bytes(b"base")
            out_l = root / "left.mp4"
            out_r = root / "right.mp4"
            raw_paths: list[Path] = []
            meta = VideoMetadata(path=str(src), width=8, height=4, source_fps=30.0, bitrate_bps=3000)

            def fake_temp(dst, label, suffix=".raw.hevc"):
                path = Path(dst).with_name(f"{Path(dst).stem}.{label}{suffix}")
                raw_paths.append(path)
                return path

            with (
                patch("gpu_engine.files.probe.probe_video", return_value=meta),
                patch("gpu_engine.files.PyNvThreadedSerialDecoder", FakeDecoder),
                patch("gpu_engine.files.PyNvEncoderSession", FakeEncoder),
                patch("gpu_engine.files._pack_planes", return_value="app"),
                patch("gpu_engine.files._media_temp_path", side_effect=fake_temp),
                patch("gpu_engine.files.runtime.free_memory_pool"),
            ):
                with self.assertRaises(OperationCancelled):
                    gpu_files.extract_multi_rect_clip(
                        src,
                        [
                            {"dst": out_l, "crop_mode": "left", "rect": (0, 0, 2, 2), "bitrate_bps": 1000},
                            {"dst": out_r, "crop_mode": "right", "rect": (0, 0, 2, 2), "bitrate_bps": 1000},
                        ],
                        start_sec=0.0,
                        end_sec=0.1,
                        cancel_token=token,
                    )

            self.assertEqual(len(raw_paths), 2)
            self.assertFalse(any(path.exists() for path in raw_paths))
            self.assertFalse(out_l.exists())
            self.assertFalse(out_r.exists())


if __name__ == "__main__":
    unittest.main()
