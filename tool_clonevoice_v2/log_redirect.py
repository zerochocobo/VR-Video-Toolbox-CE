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
import re
import sys
import time
from typing import Callable

# emit(text, is_progress): is_progress lines are carriage-return updates that
# should replace the previous progress line rather than pile up.
Emit = Callable[[str, bool], None]

# Per-sentence chatter from the vendored IndexTTS: 14 lines for every line of
# dialogue, ~1100 of the 1254 lines a single 80-line title produced. None of it
# says anything the pipeline's own logging does not, and the GUI log holds a
# limited number of lines, so real progress was being pushed out of the window.
_VENDOR_NOISE_PREFIXES = (
    ">> ",
    "torch.Size(",
    "Use the specified emotion vector",
    "origin text:",
    # One line per sentence restating text the pipeline already logs.
    "text after normalization:",
)

# tqdm bars. Two problems, both visible on hnvr-174: tqdm closes a bar by
# writing a newline, so the final "100%|" frame arrives as an ordinary line and
# piles up instead of replacing anything; and the vendored IndexTTS opens a
# 25-step bar for every sentence, which on a 352-line title is 704 lines of
# nothing. A bar carrying a description ("Span 112/195: 100%|...") is the only
# progress signal that stage has, so it is kept -- but as a *replaceable*
# progress line. A bare bar has no description and is dropped outright.
_PROGRESS_BAR_RE = re.compile(r"^(?P<desc>.*?)\s*\d{1,3}%\|")


def _is_vendor_noise(line: str) -> bool:
    text = line.strip()
    if any(text.startswith(prefix) for prefix in _VENDOR_NOISE_PREFIXES):
        return True
    match = _PROGRESS_BAR_RE.match(text)
    return match is not None and not match.group("desc").strip()


def _is_progress_bar(line: str) -> bool:
    """A described tqdm frame, which should replace the previous one."""
    match = _PROGRESS_BAR_RE.match(line.strip())
    return match is not None and bool(match.group("desc").strip())


class LogWriter:
    def __init__(self, emit: Emit, min_progress_interval: float = 0.12,
                 drop_vendor_noise: bool = True) -> None:
        self.emit = emit
        self.min_progress_interval = min_progress_interval
        self.drop_vendor_noise = drop_vendor_noise
        self.dropped = 0
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
            if self.drop_vendor_noise and _is_vendor_noise(line):
                self.dropped += 1
                continue
            # A closed tqdm bar ends with a newline, so without this its final
            # frame would be appended as a permanent line.
            if _is_progress_bar(line):
                is_progress = True
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
def redirect_stdio(emit: Emit, drop_vendor_noise: bool = True):
    writer = LogWriter(emit, drop_vendor_noise=drop_vendor_noise)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = writer
    sys.stderr = writer
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
