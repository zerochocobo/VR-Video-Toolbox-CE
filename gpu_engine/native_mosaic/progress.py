from __future__ import annotations

import threading
import time

from gpu_engine import runtime


def _cfg(key: str, default):
    try:
        from utils import app_config

        value = app_config.get(key, default)
        return default if value is None else value
    except Exception:
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    value = _cfg(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def vram_suffix() -> str:
    if not _cfg_bool("progress_log_vram", True):
        return ""
    interval_s = float(_cfg("progress_vram_query_interval_s", 5.0) or 0.0)
    return runtime.format_vram_usage(min_interval_s=max(0.0, interval_s))


def native_progress_interval_s() -> float:
    return max(0.0, float(_cfg("progress_native_log_interval_s", 5.0) or 0.0))


def native_progress_min_pct() -> float:
    return max(0.0, float(_cfg("progress_native_log_min_pct", 20.0) or 0.0))


class NativeStageProgress:
    def __init__(
        self,
        label: str,
        log_callback=None,
        *,
        total: int = 0,
        unit: str = "frames",
        min_interval: float | None = None,
        min_pct: float | None = None,
    ):
        self.label = str(label)
        self.log = log_callback
        self.total = max(0, int(total or 0))
        self.unit = str(unit)
        self.min_interval = (
            native_progress_interval_s()
            if min_interval is None
            else max(0.0, float(min_interval))
        )
        self.min_pct = (
            native_progress_min_pct()
            if min_pct is None
            else max(0.0, float(min_pct))
        )
        self.t0 = time.perf_counter()
        self._last_t = self.t0
        self._last_pct = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0, int(sec))
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def update(self, done: int, *, force: bool = False, extra: str = "") -> None:
        if not self.log:
            return
        now = time.perf_counter()
        done = max(0, int(done))
        pct = (100.0 * done / self.total) if self.total else 0.0
        with self._lock:
            time_due = (now - self._last_t) >= self.min_interval
            pct_due = self.total > 0 and (pct - self._last_pct) >= self.min_pct
            if not force and not (time_due or pct_due):
                return
            self._last_t = now
            self._last_pct = pct
        elapsed = max(0.001, now - self.t0)
        rate = done / elapsed
        if self.total:
            progress = f"{done}/{self.total} ({pct:.1f}%)"
        else:
            progress = f"{done} {self.unit}"
        extra_text = f" | {extra}" if extra else ""
        self.log(
            f"[native] {self.label}: {progress} | {rate:.1f} {self.unit}/s | "
            f"elapsed {self._fmt(elapsed)}{extra_text}{vram_suffix()}"
        )
