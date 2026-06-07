"""Lightweight profiling helpers for GPU decode/restoration paths."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on", "profile"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_active_profile: "DecodeProfile | None" = None
_active_profile_lock = threading.Lock()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_profile_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "runtime_cache" / f"profile_{stamp}.json"


def set_active_profile(profile: "DecodeProfile | None") -> None:
    global _active_profile
    with _active_profile_lock:
        _active_profile = profile


def get_active_profile() -> "DecodeProfile | None":
    with _active_profile_lock:
        return _active_profile


class DecodeProfile:
    """Thread-safe section/counter profiler.

    The disabled instance keeps call sites simple while adding only a cheap
    branch in the default path. CUDA timings intentionally synchronize only
    when profiling is enabled.
    """

    def __init__(self, *, enabled: bool, output_path: str | os.PathLike[str] | None = None):
        self.enabled = bool(enabled)
        self.output_path = Path(output_path) if output_path else None
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._sections: dict[str, dict[str, float | int]] = {}
        self._counters: dict[str, int] = {}
        self._metadata: dict[str, Any] = {}

    @classmethod
    def from_env_or_argv(cls) -> "DecodeProfile":
        flag = os.environ.get("VRVT_PROFILE_DECODE")
        argv_enabled = "--profile-decode" in sys.argv
        output_path = os.environ.get("VRVT_PROFILE_DECODE_PATH")
        enabled = argv_enabled

        if flag is not None:
            text = str(flag).strip()
            low = text.lower()
            if low in _FALSE_VALUES:
                enabled = False
            elif low in _TRUE_VALUES:
                enabled = True
            else:
                enabled = True
                output_path = text

        return cls(enabled=enabled, output_path=output_path)

    def metadata(self, **values: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._metadata.update(values)

    def increment(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + int(amount)

    @contextmanager
    def section(self, name: str, *, torch_module=None, cuda: bool = False):
        if not self.enabled:
            yield
            return

        start_event = None
        end_event = None
        if cuda and torch_module is not None:
            try:
                if torch_module.cuda.is_available():
                    start_event = torch_module.cuda.Event(enable_timing=True)
                    end_event = torch_module.cuda.Event(enable_timing=True)
                    start_event.record()
            except Exception:
                start_event = None
                end_event = None

        t0 = time.perf_counter()
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            cuda_ms = 0.0
            if start_event is not None and end_event is not None:
                try:
                    end_event.record()
                    end_event.synchronize()
                    cuda_ms = float(start_event.elapsed_time(end_event))
                except Exception:
                    cuda_ms = 0.0
            with self._lock:
                item = self._sections.setdefault(
                    name,
                    {"count": 0, "wall_ms": 0.0, "cuda_ms": 0.0},
                )
                item["count"] = int(item["count"]) + 1
                item["wall_ms"] = float(item["wall_ms"]) + wall_ms
                item["cuda_ms"] = float(item["cuda_ms"]) + cuda_ms

    def write(self, log_callback=None) -> Path | None:
        if not self.enabled:
            return None
        output_path = self.output_path or _default_profile_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": self.created_at,
            "elapsed_ms": (time.perf_counter() - self._t0) * 1000.0,
            "metadata": self._metadata,
            "sections": self._sections,
            "counters": self._counters,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if log_callback:
            log_callback(f"[profile] wrote {output_path}")
        return output_path
