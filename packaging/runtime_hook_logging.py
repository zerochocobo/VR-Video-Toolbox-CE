"""PyInstaller runtime hook: redirect stdout/stderr to runtime_cache\\logs in frozen+windowed mode.

When the main exe is built with console=False, sys.stdout / sys.stderr are None
on Windows, and any print()/traceback would raise AttributeError. Redirect both
streams to a rotating log file next to the exe so existing print() calls keep
working without a console window.
"""
import os
import sys
from datetime import datetime


def _setup() -> None:
    if not getattr(sys, "frozen", False):
        return
    # Only redirect when running headless (sys.stdout is None under --noconsole).
    if sys.stdout is not None and sys.stderr is not None:
        return

    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    log_dir = os.path.join(exe_dir, "runtime_cache", "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        return

    stamp = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"vr_toolbox_{stamp}.log")
    try:
        stream = open(log_path, "a", encoding="utf-8", buffering=1)
    except OSError:
        return

    stream.write(f"\n===== VR Video Toolbox launched at {datetime.now().isoformat()} =====\n")
    stream.flush()
    sys.stdout = stream
    sys.stderr = stream


_setup()
