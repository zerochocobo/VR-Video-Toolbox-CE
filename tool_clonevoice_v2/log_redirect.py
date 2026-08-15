"""Route library stdout/stderr (incl. tqdm progress bars) into the GUI log.

Model code we don't own (OmniVoice "Loading weights", faster-whisper, the
vendored bandit handler's "Rank 0:" bar, torch warnings) writes progress to
``sys.stderr``. In a windowed PyInstaller build there is no console, so that
output is lost. :func:`redirect_stdio` temporarily swaps ``sys.stdout`` /
``sys.stderr`` for a writer that forwards complete lines to a callback, while
collapsing ``\\r``-updated progress bars onto a single (replaceable) line.
"""
from __future__ import annotations

import contextlib
import sys
import time
from typing import Callable

# emit(text, is_progress): is_progress lines are carriage-return updates that
# should replace the previous progress line rather than pile up.
Emit = Callable[[str, bool], None]


class LogWriter:
    def __init__(self, emit: Emit, min_progress_interval: float = 0.12) -> None:
        self.emit = emit
        self.min_progress_interval = min_progress_interval
        self._buf = ""
        self._last_progress_at = 0.0

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while self._buf:
            i_n = self._buf.find("\n")
            i_r = self._buf.find("\r")
            if i_n == -1 and i_r == -1:
                break
            if i_r == -1 or (i_n != -1 and i_n < i_r):
                idx, is_progress = i_n, False
            else:
                idx, is_progress = i_r, True
            line = self._buf[:idx].rstrip("\r\n")
            self._buf = self._buf[idx + 1 :]
            if not line.strip():
                continue
            if is_progress:
                now = time.monotonic()
                if now - self._last_progress_at < self.min_progress_interval:
                    continue
                self._last_progress_at = now
            try:
                self.emit(line, is_progress)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:  # pragma: no cover - file-like protocol
        pass


@contextlib.contextmanager
def redirect_stdio(emit: Emit):
    writer = LogWriter(emit)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = writer
    sys.stderr = writer
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
