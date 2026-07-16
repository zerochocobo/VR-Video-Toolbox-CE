from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from tool_dlna import main as dlna_main


class _TimeoutThenExitProcess:
    pid = 12345

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired("vr_dlna_server.exe", timeout)
        return 0

    def kill(self):
        self.killed = True


class _Button:
    def __init__(self) -> None:
        self.kwargs = None

    def config(self, **kwargs):
        self.kwargs = kwargs


class _LauncherFake:
    def __init__(self, process=None) -> None:
        self.dlna_process = process
        self.btn_dlna_toggle = _Button()
        self.refresh_dlna_ui_calls = 0

    def get_configured_dlna_dirs(self):
        return [r"C:\Videos"]

    def show_dlna_config_dialog(self, require_dirs=False):
        return True

    def get_startupinfo(self):
        return None

    def _refresh_dlna_ui(self):
        self.refresh_dlna_ui_calls += 1


class PackagingRuntimeTests(unittest.TestCase):
    def test_dlna_executable_name_is_lowercase(self) -> None:
        self.assertEqual(main.DLNA_SERVER_EXE_NAME, "vr_dlna_server.exe")

    def test_dlna_server_path_uses_launcher_exe_dir(self) -> None:
        with patch.object(main, "_exe_dir", r"C:\VRToolbox"):
            self.assertEqual(main.get_dlna_server_exe_path(), r"C:\VRToolbox\vr_dlna_server.exe")

    def test_runtime_work_dir_uses_exe_dir_when_frozen(self) -> None:
        with patch.object(main.sys, "frozen", True, create=True), patch.object(main, "_exe_dir", r"C:\VRToolbox"):
            self.assertEqual(main.get_runtime_work_dir(), r"C:\VRToolbox")

    def test_terminate_process_tree_uses_taskkill_after_timeout_on_windows(self) -> None:
        process = _TimeoutThenExitProcess()

        with (
            patch.object(main.sys, "platform", "win32"),
            patch.object(main, "build_hidden_startupinfo", return_value=None),
            patch("main.subprocess.run") as run,
        ):
            main.terminate_process_tree(process, timeout=0.01)

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["taskkill", "/PID", "12345", "/T", "/F"])

    def test_packaged_close_can_kill_leftover_dlna_exe(self) -> None:
        with (
            patch.object(main.sys, "platform", "win32"),
            patch.object(main.sys, "frozen", True, create=True),
            patch.object(main, "build_hidden_startupinfo", return_value=None),
            patch("main.subprocess.run") as run,
        ):
            main.terminate_packaged_dlna_server_instances()

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["taskkill", "/IM", "vr_dlna_server.exe", "/T", "/F"])

    def test_packaged_stop_uses_image_cleanup_for_onefile_children(self) -> None:
        launcher = _LauncherFake(process=SimpleNamespace())

        with (
            patch.object(main, "is_packaged_mode", return_value=True),
            patch.object(main, "terminate_packaged_dlna_server_instances") as cleanup,
            patch.object(main, "terminate_process_tree") as terminate_tree,
        ):
            main.VRVideoToolboxLauncher.toggle_dlna_server(launcher)

        cleanup.assert_called_once()
        terminate_tree.assert_not_called()
        self.assertIsNone(launcher.dlna_process)
        self.assertEqual(launcher.refresh_dlna_ui_calls, 1)

    def test_packaged_start_cleans_stale_dlna_before_launching_exe(self) -> None:
        launcher = _LauncherFake(process=None)
        popen_process = SimpleNamespace(pid=777)

        with (
            patch.object(main.sys, "frozen", True, create=True),
            patch.object(main, "terminate_packaged_dlna_server_instances") as cleanup,
            patch.object(main, "get_dlna_server_exe_path", return_value=r"C:\VRToolbox\vr_dlna_server.exe"),
            patch.object(main.os.path, "exists", return_value=True),
            patch.object(main, "get_runtime_work_dir", return_value=r"C:\VRToolbox"),
            patch("main.subprocess.Popen", return_value=popen_process) as popen,
        ):
            main.VRVideoToolboxLauncher.toggle_dlna_server(launcher)

        cleanup.assert_called_once()
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], [r"C:\VRToolbox\vr_dlna_server.exe"])
        self.assertEqual(popen.call_args.kwargs["cwd"], r"C:\VRToolbox")
        self.assertIs(launcher.dlna_process, popen_process)

    def test_dlna_runtime_base_dir_uses_real_exe_parent_when_frozen(self) -> None:
        with (
            patch.object(dlna_main.sys, "frozen", True, create=True),
            patch.object(dlna_main.sys, "executable", r"C:\VRToolbox\vr_dlna_server.exe"),
        ):
            self.assertEqual(dlna_main.get_runtime_base_dir(), Path(r"C:\VRToolbox"))

    def test_spec_builds_onefile_exes_with_i18n_and_lowercase_dlna(self) -> None:
        spec = Path("VR_Video_Toolbox.spec").read_text(encoding="utf-8")

        self.assertIn('"i18n"', spec)
        self.assertIn('name="vr_dlna_server"', spec)
        self.assertIn("main_a.binaries", spec)
        self.assertIn("main_a.datas", spec)
        self.assertIn("dlna_a.binaries", spec)
        self.assertIn("dlna_a.datas", spec)
        self.assertNotIn("COLLECT(", spec)
        self.assertNotIn("exclude_binaries=True", spec)
        self.assertNotIn("release_readme", spec)

    def test_upx_build_script_uses_onefile_spec(self) -> None:
        script = Path("build_upxexe.bat").read_text(encoding="utf-8")

        self.assertIn("VR_Video_Toolbox.spec", script)
        self.assertIn("VR_TOOLBOX_USE_UPX=1", script)
        self.assertNotIn("--onefile", script)


if __name__ == "__main__":
    unittest.main()
