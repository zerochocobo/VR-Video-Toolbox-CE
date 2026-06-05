from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gpu_engine import restored_sidecar
from gpu_engine.probe import ColorMetadata


class RestoredSidecarTests(unittest.TestCase):
    def test_write_load_and_metadata_from_restored_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            hevc = root / "seg.restored.hevc"
            hevc.write_bytes(b"hevc")

            sidecar = restored_sidecar.write_restored_sidecar(
                hevc,
                width=512,
                height=256,
                bit_depth=8,
                fps=30000 / 1001,
                frame_count=300,
                color=ColorMetadata("tv", "bt709", "bt709", "bt709"),
                source="source.mp4",
                rect={"x": 10, "y": 20, "w": 512, "h": 256},
                time_range={"start_frame": 30, "end_frame": 330},
                encoder="hevc_nvenc P4 vbr 1000kbps",
            )

            self.assertEqual(sidecar, root / "seg.restored.json")
            data = restored_sidecar.load_restored_sidecar(hevc)
            self.assertEqual(data["format_version"], 1)
            self.assertEqual(data["fps_num"], 30000)
            self.assertEqual(data["fps_den"], 1001)
            self.assertEqual(data["rect"]["x"], 10)
            meta = restored_sidecar.metadata_from_sidecar(hevc)
            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertEqual((meta.width, meta.height, meta.nb_frames), (512, 256, 300))
            self.assertAlmostEqual(meta.source_fps, 30000 / 1001, places=6)
            self.assertEqual(meta.color.color_space, "bt709")


if __name__ == "__main__":
    unittest.main()
