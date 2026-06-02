from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
