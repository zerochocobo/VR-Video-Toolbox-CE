from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock, patch

from one_click import logic
from gpu_engine import files as gpu_files
from utils import mosaic_prescan


class GpuStageCleanupTests(unittest.TestCase):
    def test_paste_decoder_queue_defaults_to_eight_frames(self) -> None:
        with patch("gpu_engine.files._cfg", return_value=8):
            self.assertEqual(
                gpu_files._memory_bounded_decoder_kwargs(),
                {"batch_size": 8, "buffer_size": 8},
            )

    def test_paste_decoder_queue_clamps_batch_to_small_configured_buffer(self) -> None:
        with patch("gpu_engine.files._cfg", return_value=4):
            self.assertEqual(
                gpu_files._memory_bounded_decoder_kwargs(),
                {"batch_size": 4, "buffer_size": 4},
            )

    def test_post_split_cleanup_uses_only_already_loaded_modules(self) -> None:
        runtime = types.SimpleNamespace(free_memory_pool=Mock())
        modules = dict(sys.modules)
        modules.pop("torch", None)
        modules.pop("cupy", None)
        modules["gpu_engine.runtime"] = runtime

        with patch.object(logic.sys, "modules", modules):
            logic._cleanup_gpu_after_split()

        runtime.free_memory_pool.assert_called_once_with()
        self.assertNotIn("torch", modules)
        self.assertNotIn("cupy", modules)

    def test_post_split_cleanup_does_not_empty_uninitialized_torch(self) -> None:
        cuda = types.SimpleNamespace(
            is_initialized=Mock(return_value=False),
            empty_cache=Mock(),
        )
        modules = dict(sys.modules)
        modules["torch"] = types.SimpleNamespace(cuda=cuda)
        modules.pop("cupy", None)
        modules.pop("gpu_engine.runtime", None)

        with patch.object(logic.sys, "modules", modules):
            logic._cleanup_gpu_after_split()

        cuda.is_initialized.assert_called_once_with()
        cuda.empty_cache.assert_not_called()

    def test_detector_release_does_not_import_torch_for_cleanup(self) -> None:
        modules = dict(sys.modules)
        modules.pop("torch", None)
        old_detector = mosaic_prescan._DETECTOR
        old_config = mosaic_prescan._DETECTOR_CONFIG
        try:
            mosaic_prescan._DETECTOR = object()
            mosaic_prescan._DETECTOR_CONFIG = ("model", 2048, 0.2)
            with patch.object(mosaic_prescan.sys, "modules", modules):
                mosaic_prescan.release_detector()
        finally:
            mosaic_prescan._DETECTOR = old_detector
            mosaic_prescan._DETECTOR_CONFIG = old_config

        self.assertNotIn("torch", modules)


if __name__ == "__main__":
    unittest.main()
