from __future__ import annotations

import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from one_click import logic
from utils import app_config, engine_runner


HIGHEST_LADA_OPTS = " -rc vbr -cq 18 -preset p7 -tune hq -spatial_aq 1 -aq-strength 6 -multipass fullres"
BALANCED_LADA_OPTS = " -rc vbr -cq 18 -preset p4 -tune hq -spatial_aq 1 -aq-strength 6 -multipass fullres"
FAST_LADA_OPTS = " -rc vbr -cq 18 -preset p1 -tune hq -spatial_aq 1 -aq-strength 6 -multipass fullres"
ULTRA_FAST_LADA_OPTS = " -rc vbr -cq 18 -preset p1 -tune hq -spatial_aq 1 -aq-strength 6"


class EngineRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cache = dict(app_config._cache)

    def tearDown(self) -> None:
        app_config._cache = self._old_cache

    def test_lada_encoder_options_use_shared_encode_profile_with_aq(self) -> None:
        app_config._cache = {
            "engine": "lada",
            "gpu_encode_profile": "highest_quality",
            "custom_args_lada": "",
        }

        opts = engine_runner.build_lada_encoder_options(cq=18)
        cmd = engine_runner.build_engine_cmd("in.mp4", "out.mp4", encoder_options=opts)

        self.assertEqual(opts, HIGHEST_LADA_OPTS)
        self.assertIn("--encoder-options", cmd)
        self.assertEqual(cmd[cmd.index("--encoder-options") + 1], HIGHEST_LADA_OPTS)

    def test_lada_encoder_options_remain_key_value_pairs_for_pyav_fallback(self) -> None:
        app_config._cache = {
            "engine": "lada",
            "gpu_encode_profile": "highest_quality",
            "custom_args_lada": "",
        }

        tokens = shlex.split(engine_runner.build_lada_encoder_options(cq=18))

        self.assertEqual(len(tokens) % 2, 0)
        self.assertTrue(all(token.startswith("-") for token in tokens[::2]))

    def test_balanced_high_quality_is_the_default_profile(self) -> None:
        app_config._cache = dict(app_config._DEFAULTS)
        app_config._cache["engine"] = "lada"

        opts = engine_runner.build_lada_encoder_options(cq=18)

        self.assertEqual(opts, BALANCED_LADA_OPTS)

    def test_legacy_maximum_quality_profile_stays_on_balanced_p4(self) -> None:
        app_config._cache = {
            "engine": "lada",
            "gpu_encode_profile": "maximum_quality",
            "gpu_encode_preset": "P7",
            "custom_args_lada": "",
        }

        opts = engine_runner.build_lada_encoder_options(cq=18)

        self.assertEqual(opts, BALANCED_LADA_OPTS)
        self.assertNotIn("-preset p7", opts)

    def test_ultra_fast_normal_profile_keeps_aq_but_disables_multipass(self) -> None:
        app_config._cache = {
            "engine": "lada",
            "gpu_encode_profile": "ultra_fast_normal",
            "custom_args_lada": "",
        }

        opts = engine_runner.build_lada_encoder_options(cq=18)

        self.assertEqual(opts, ULTRA_FAST_LADA_OPTS)
        self.assertNotIn("-multipass", opts)

    def test_process_lada_external_cli_uses_profile_not_stale_raw_encode_config(self) -> None:
        app_config._cache = {
            "engine": "lada",
            "gpu_encode_profile": "fast_quality",
            "gpu_encode_preset": "P5",
            "gpu_encode_multipass": "off",
            "gpu_encode_aq": False,
            "gpu_encode_temporal_aq": True,
            "custom_args_lada": "",
        }

        with patch.object(logic, "run_process") as run_process:
            logic.process_lada("in.mp4", "out.mp4")

        cmd = run_process.call_args.args[0]
        opts = cmd[cmd.index("--encoder-options") + 1]
        self.assertEqual(opts, FAST_LADA_OPTS)
        self.assertIn("-spatial_aq", opts)
        self.assertIn("-multipass", opts)

    def test_jasna_command_and_log_disclose_profile_limit(self) -> None:
        app_config._cache = {
            "engine": "jasna",
            "gpu_encode_profile": "highest_quality",
            "custom_args_jasna": "",
        }
        logs: list[str] = []

        with patch.object(logic, "run_process") as run_process:
            logic.process_lada("in.mp4", "out.mp4", log_callback=logs.append)

        cmd = run_process.call_args.args[0]
        self.assertIn("--encoder-settings", cmd)
        self.assertEqual(cmd[cmd.index("--encoder-settings") + 1], "cq=18")
        self.assertNotIn("-preset", cmd)
        self.assertTrue(any("Jasna CLI only receives cq" in line for line in logs))

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

    def test_native_mosaic_release_engine_drops_cached_singleton(self) -> None:
        from gpu_engine import native_mosaic

        class DummyEngine:
            def __init__(self) -> None:
                self.released = False

            def release(self) -> None:
                self.released = True

        dummy = DummyEngine()
        native_mosaic._engine = dummy
        try:
            self.assertTrue(native_mosaic.release_engine())
            self.assertIsNone(native_mosaic._engine)
            self.assertTrue(dummy.released)
        finally:
            native_mosaic._engine = None

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
