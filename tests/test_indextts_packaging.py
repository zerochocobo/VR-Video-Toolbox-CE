from __future__ import annotations

import builtins
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import build_exe


class IndexTTSPackagingTests(unittest.TestCase):
    REQUIRED_TEXT_DATA = (
        Path("contractions/data/contractions_dict.json"),
        Path("contractions/data/leftovers_dict.json"),
        Path("contractions/data/slang_dict.json"),
        Path("wetext/fsts/zh/tn/tagger.fst"),
        Path("wetext/fsts/zh/tn/verbalizer.fst"),
        Path("wetext/fsts/en/tn/tagger.fst"),
        Path("wetext/fsts/en/tn/verbalizer.fst"),
        Path("textstat/resources/en/easy_words.txt"),
        Path("pyphen/dictionaries/hyph_en_US.dic"),
    )

    def _create_complete_runtime(self, internal: Path) -> None:
        unidic_dir = internal / "unidic_lite" / "dicdir"
        unidic_dir.mkdir(parents=True)
        for name in build_exe.INDEXTTS_UNIDIC_REQUIRED_FILES:
            (unidic_dir / name).write_bytes(b"test")

        fugashi_dir = internal / "fugashi"
        fugashi_dir.mkdir()
        (fugashi_dir / "fugashi.cp312-win_amd64.pyd").write_bytes(b"test")

        for relative_path in self.REQUIRED_TEXT_DATA:
            path = internal / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")

        kaldifst_dir = internal / "kaldifst" / "lib"
        kaldifst_dir.mkdir(parents=True)
        (kaldifst_dir / "_kaldifst.cp312-win_amd64.pyd").write_bytes(b"test")

        sentencepiece_dir = internal / "sentencepiece"
        sentencepiece_dir.mkdir()
        (sentencepiece_dir / "_sentencepiece.cp312-win_amd64.pyd").write_bytes(b"test")

    def test_spec_collects_text_runtime_as_required_dependencies(self) -> None:
        spec = Path("VR_Video_Toolbox.spec").read_text(encoding="utf-8")

        for package in (
            "fugashi", "unidic_lite", "wetext", "contractions", "kaldifst",
            "sentencepiece", "textstat", "pyphen", "soundfile",
        ):
            self.assertIn(f'"{package}"', spec)
        self.assertIn("Required packaged audio/IndexTTS dependency is missing", spec)

    def test_verify_accepts_soundfile_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            internal = Path(temp_dir)
            data_dir = internal / "_soundfile_data"
            data_dir.mkdir()
            (data_dir / "libsndfile_x64.dll").write_bytes(b"test")

            with patch.object(build_exe, "INTERNAL", internal):
                build_exe.verify_soundfile_runtime()

    def test_verify_rejects_missing_soundfile_dll(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(build_exe, "INTERNAL", Path(temp_dir)):
                with self.assertRaisesRegex(build_exe.BuildError, "libsndfile"):
                    build_exe.verify_soundfile_runtime()

    def test_verify_accepts_complete_text_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            internal = Path(temp_dir)
            self._create_complete_runtime(internal)

            with patch.object(build_exe, "INTERNAL", internal):
                build_exe.verify_indextts_text_runtime()

    def test_verify_rejects_missing_unidic_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            internal = Path(temp_dir)
            self._create_complete_runtime(internal)
            (internal / "unidic_lite" / "dicdir" / "version").unlink()

            with patch.object(build_exe, "INTERNAL", internal):
                with self.assertRaisesRegex(build_exe.BuildError, "version"):
                    build_exe.verify_indextts_text_runtime()

    def test_verify_rejects_missing_fugashi_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            internal = Path(temp_dir)
            self._create_complete_runtime(internal)
            next((internal / "fugashi").glob("fugashi*.pyd")).unlink()

            with patch.object(build_exe, "INTERNAL", internal):
                with self.assertRaisesRegex(build_exe.BuildError, "fugashi"):
                    build_exe.verify_indextts_text_runtime()

    def test_verify_rejects_missing_contractions_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            internal = Path(temp_dir)
            self._create_complete_runtime(internal)
            (internal / "contractions" / "data" / "contractions_dict.json").unlink()

            with patch.object(build_exe, "INTERNAL", internal):
                with self.assertRaisesRegex(build_exe.BuildError, "contractions_dict.json"):
                    build_exe.verify_indextts_text_runtime()

    def test_verify_rejects_missing_wetext_fst(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            internal = Path(temp_dir)
            self._create_complete_runtime(internal)
            (internal / "wetext" / "fsts" / "zh" / "tn" / "tagger.fst").unlink()

            with patch.object(build_exe, "INTERNAL", internal):
                with self.assertRaisesRegex(build_exe.BuildError, "tagger.fst"):
                    build_exe.verify_indextts_text_runtime()

    def test_running_packaged_app_blocks_build_before_clean(self) -> None:
        running = ["VR_DLNA_Server PID 456", "VR_Video_Toolbox PID 123"]
        with patch.object(build_exe, "running_dist_app_pids", return_value=running):
            with self.assertRaisesRegex(build_exe.BuildError, "VR_Video_Toolbox PID 123"):
                build_exe.ensure_dist_app_not_running()

    def test_running_app_guard_covers_both_packaged_exes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dist = root / build_exe.APP_NAME
            dlna_dist = root / build_exe.DLNA_NAME
            main_exe = dist / f"{build_exe.APP_NAME}.exe"
            merged_dlna_exe = dist / f"{build_exe.DLNA_NAME}.exe"
            intermediate_dlna_exe = dlna_dist / f"{build_exe.DLNA_NAME}.exe"

            class FakeProcess:
                def __init__(self, pid: int, exe: Path):
                    self.info = {"pid": pid, "exe": str(exe)}

            fake_psutil = types.SimpleNamespace(
                Error=RuntimeError,
                process_iter=lambda _attrs: [
                    FakeProcess(101, main_exe),
                    FakeProcess(102, merged_dlna_exe),
                    FakeProcess(103, intermediate_dlna_exe),
                ],
            )
            with (
                patch.dict(sys.modules, {"psutil": fake_psutil}),
                patch.object(build_exe, "DIST", dist),
                patch.object(build_exe, "DLNA_DIST", dlna_dist),
            ):
                running = build_exe.running_dist_app_pids()

        self.assertEqual(
            running,
            [
                "VR_Video_Toolbox PID 101",
                "vr_dlna_server PID 102",
                "vr_dlna_server PID 103",
            ],
        )

    def test_running_app_guard_is_tied_to_clean_not_to_main_build(self) -> None:
        """--skip-main must not silently disable the guard: clean() still deletes dist."""
        source = Path("build_exe.py").read_text(encoding="utf-8")
        guard = source.index("ensure_dist_app_not_running()\n            clean()")
        skip_clean = source.rindex("if not args.skip_clean:", 0, guard)

        self.assertNotIn("args.skip_main", source[skip_clean:guard])

    def test_missing_psutil_fails_instead_of_skipping_the_guard(self) -> None:
        real_import = builtins.__import__

        def deny_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", deny_psutil):
            with self.assertRaisesRegex(build_exe.BuildError, "psutil is required"):
                build_exe.running_dist_app_pids()

    def test_build_strategy_never_preserves_or_restores_models_in_dist(self) -> None:
        script = Path("build_exe.py").read_text(encoding="utf-8")
        spec = Path("VR_Video_Toolbox.spec").read_text(encoding="utf-8")

        self.assertNotIn("preserve_external_models", script)
        self.assertNotIn("restore_external_models", script)
        self.assertIn('if (DIST / "models").exists():', script)
        self.assertNotIn('(\"models\", \"models\")', spec)


if __name__ == "__main__":
    unittest.main()
