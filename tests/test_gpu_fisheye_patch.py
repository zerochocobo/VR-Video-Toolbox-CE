from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gpu_engine import files as gpu_files
from gpu_engine.probe import VideoMetadata


class FisheyePatchRawWrapTests(unittest.TestCase):
    def test_fisheye_patch_wraps_raw_hevc_segment_before_pynv_decode(self) -> None:
        fake_cupy = types.SimpleNamespace(
            cuda=types.SimpleNamespace(Device=lambda: types.SimpleNamespace(synchronize=lambda: None)),
            float32=np.float32,
            uint8=np.uint8,
            uint16=np.uint16,
            rint=np.rint,
        )
        decoder_paths: list[Path] = []

        class FakeFrame:
            def __init__(self, width: int, height: int):
                self.width = int(width)
                self.height = int(height)

            def y_uv_cupy(self):
                y = np.zeros((self.height, self.width), dtype=np.uint8)
                uv = np.zeros((self.height // 2, self.width // 2, 2), dtype=np.uint8)
                return y, uv

        class FakeDecoder:
            def __init__(self, path, bit_depth=8, start_frame=0, **_kwargs):
                p = Path(path)
                decoder_paths.append(p)
                if len(decoder_paths) == 1:
                    self.info = types.SimpleNamespace(width=8, height=4, fps=30.0)
                    self.width = 8
                    self.height = 4
                else:
                    self.info = types.SimpleNamespace(width=2, height=2, fps=30.0)
                    self.width = 2
                    self.height = 2

            def __len__(self):
                return 1

            def frame_at(self, _index):
                return FakeFrame(self.width, self.height)

            def stop(self):
                pass

        class FakeEncoder:
            def __init__(self, *_args, **_kwargs):
                pass

            def encode(self, _app, force_idr=False):
                return b"frame"

            def flush(self):
                return b"tail"

        def fake_mux(_raw_hevc, out_path, **_kwargs):
            Path(out_path).write_bytes(b"mp4")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = root / "base.mp4"
            dst = root / "out.mp4"
            restored_raw = root / "seg.restored.hevc"
            base.write_bytes(b"base")
            restored_raw.write_bytes(b"hevc")
            base_meta = VideoMetadata(path=str(base), width=8, height=4, source_fps=30.0, bitrate_bps=1000)
            seg_meta = VideoMetadata(path=str(restored_raw), width=2, height=2, source_fps=30.0, bitrate_bps=0)
            decision = types.SimpleNamespace(is_gpu=True, reason="")
            seg = types.SimpleNamespace(
                seg_id=0,
                path=restored_raw,
                base_frame_start=0,
                base_frame_end=1,
                x=0,
                y=0,
                w=2,
                h=2,
            )

            with (
                patch.dict("sys.modules", {"cupy": fake_cupy}),
                patch("gpu_engine.files.probe.route", return_value=(base_meta, decision)),
                patch("gpu_engine.files.restored_sidecar.metadata_from_sidecar", return_value=seg_meta),
                patch("gpu_engine.files.restored_sidecar.frame_count_from_sidecar", return_value=1),
                patch("gpu_engine.files.PyNvThreadedSerialDecoder", FakeDecoder),
                patch("gpu_engine.files.PyNvEncoderSession", FakeEncoder),
                patch("gpu_engine.files.v360_lut.make_lut", return_value=None),
                patch("gpu_engine.files.nv12_kernels.remap_y", side_effect=lambda arr, *_args: arr.copy()),
                patch("gpu_engine.files.nv12_kernels.remap_uv", side_effect=lambda arr, *_args: arr.copy()),
                patch("gpu_engine.files.nv12_kernels.hstack_planes", side_effect=lambda a, b: np.hstack([a, b])),
                patch(
                    "gpu_engine.files._make_alpha_mask",
                    side_effect=lambda w, h, _px: np.ones((h, w), dtype=np.float32),
                ),
                patch("gpu_engine.files._pack_planes", return_value="app"),
                patch("gpu_engine.files.mux.mux_hevc_with_audio", side_effect=fake_mux),
                patch("gpu_engine.files.runtime.free_memory_pool"),
            ):
                gpu_files.paste_fisheye_eye_rects_to_sbs_gpu(
                    base,
                    dst,
                    [seg],
                    keep_audio=False,
                    feather_px=0,
                )

            self.assertEqual(decoder_paths[0], base)
            self.assertEqual(decoder_paths[1].suffix.lower(), ".mp4")
            self.assertNotEqual(decoder_paths[1], restored_raw)
            self.assertFalse(list(root.glob("*.pynvwrap.*.mp4")))


if __name__ == "__main__":
    unittest.main()
