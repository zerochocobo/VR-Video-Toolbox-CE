from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from one_click import logic
from utils import app_config, engine_runner


class EngineRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cache = dict(app_config._cache)

    def tearDown(self) -> None:
        app_config._cache = self._old_cache

    def test_lada_encoder_options_use_configured_gpu_preset(self) -> None:
        app_config._cache = {
            "engine": "lada",
            "gpu_encode_preset": "P4",
            "custom_args_lada": "",
        }

        opts = engine_runner.build_lada_encoder_options(cq=18)
        cmd = engine_runner.build_engine_cmd("in.mp4", "out.mp4", encoder_options=opts)

        self.assertEqual(opts, " -cq 18 -preset p4")
        self.assertIn("--encoder-options", cmd)
        self.assertEqual(cmd[cmd.index("--encoder-options") + 1], " -cq 18 -preset p4")

    def test_process_lada_external_cli_uses_configured_gpu_preset(self) -> None:
        app_config._cache = {
            "engine": "lada",
            "gpu_encode_preset": "P5",
            "custom_args_lada": "",
        }

        with patch.object(logic, "run_process") as run_process:
            logic.process_lada("in.mp4", "out.mp4")

        cmd = run_process.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--encoder-options") + 1], " -cq 18 -preset p5")

    def test_process_lada_native_raw_restore_returns_hevc_path(self) -> None:
        app_config._cache = {
            "engine": "native_gpu",
        }

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            out = root / "seg.restored.mp4"
            raw_out = root / "seg.restored.hevc"

            def fake_restore(_input, output, **kwargs):
                self.assertFalse(kwargs["produce_mp4"])
                self.assertEqual(output, str(out))
                raw_out.write_bytes(b"hevc")
                return True

            with patch("gpu_engine.native_mosaic.restore_file", side_effect=fake_restore):
                actual = logic.process_lada("in.mp4", str(out), produce_mp4=False)

        self.assertEqual(actual, str(raw_out))

    def test_one_click_native_dependency_check_is_lightweight(self) -> None:
        app_config._cache = {
            "engine": "native_gpu",
        }

        with (
            patch("one_click.logic.shutil.which", return_value="tool"),
            patch("gpu_engine.native_mosaic.unavailable_reason", return_value=None) as unavailable_reason,
        ):
            missing = logic.check_dependencies()

        self.assertEqual(missing, [])
        unavailable_reason.assert_called_once_with(runtime_check=False)

    def test_native_unavailable_reason_lightweight_does_not_prepare_gpu(self) -> None:
        from gpu_engine import native_mosaic

        with (
            patch.object(native_mosaic, "_prepare", side_effect=AssertionError("warmup should be deferred")),
            patch("gpu_engine.native_mosaic.importlib.util.find_spec", return_value=object()),
            patch("gpu_engine.native_mosaic.os.path.isfile", return_value=True),
        ):
            reason = native_mosaic.unavailable_reason(runtime_check=False)

        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
