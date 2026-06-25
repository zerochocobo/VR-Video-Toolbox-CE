"""Real-time DLNA SI audio mixing streams.

The service exposes a virtual MP4 stream where the original video is copied and
the first audio track is mixed with the sibling ``.si.wav`` file on demand.
DLNA directory entries use the separate MPEG-TS live iterator because common VR
players handle it more reliably than fragmented MP4 for live playback.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_dlna import content_directory
from tool_dlna.firewall import hidden_subprocess_kwargs
from tool_dlna.media_library import safe_resolve_path
from tool_si import logic as si_logic


AUDIO_BITRATE_BPS = 192_000
SIZE_OVERHEAD_FACTOR = 1.05
MIN_ESTIMATED_SIZE = 64 * 1024
DEFAULT_CHUNK_SIZE = 64 * 1024
DEFAULT_REUSE_TOLERANCE_BYTES = 1024 * 1024
DEFAULT_SEEK_COOLDOWN_SECONDS = 0.2

log = logging.getLogger("vrtoolbox.dlna")


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_choice(value: object, choices: tuple[Any, ...], default: Any) -> Any:
    if value in choices:
        return value
    for choice in choices:
        try:
            if type(choice)(value) == choice:
                return choice
        except (TypeError, ValueError):
            continue
    return default


@dataclass(frozen=True)
class SIMixConfig:
    enabled: bool = False
    mix_channel: str = "both"
    original_volume_percent: int = si_logic.DEFAULT_ORIGINAL_VOLUME_PERCENT
    si_volume_percent: int = 100
    si_delay_seconds: float = si_logic.DEFAULT_SI_DELAY_SECONDS
    duck_original: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _coerce_bool(self.enabled, False))
        object.__setattr__(
            self,
            "mix_channel",
            _coerce_choice(str(self.mix_channel).strip().lower(), si_logic.SI_MIX_CHANNELS, "both"),
        )
        object.__setattr__(
            self,
            "original_volume_percent",
            _coerce_choice(
                self.original_volume_percent,
                si_logic.ORIGINAL_VOLUME_CHOICES,
                si_logic.DEFAULT_ORIGINAL_VOLUME_PERCENT,
            ),
        )
        object.__setattr__(
            self,
            "si_volume_percent",
            _coerce_choice(self.si_volume_percent, si_logic.SI_VOLUME_CHOICES, 100),
        )
        object.__setattr__(
            self,
            "si_delay_seconds",
            _coerce_choice(
                round(float(self.si_delay_seconds), 1) if self.si_delay_seconds is not None else None,
                si_logic.SI_DELAY_SECONDS_CHOICES,
                si_logic.DEFAULT_SI_DELAY_SECONDS,
            ),
        )
        object.__setattr__(self, "duck_original", _coerce_bool(self.duck_original, True))

    @classmethod
    def from_app_config(cls, getter: Callable[..., Any]) -> "SIMixConfig":
        def read(key: str, default: Any) -> Any:
            try:
                return getter(key, default)
            except TypeError:
                value = getter(key)
                return default if value is None else value

        return cls(
            enabled=read("dlna_si_enabled", True),
            mix_channel=read("dlna_si_mix_channel", "both"),
            original_volume_percent=read("dlna_si_original_volume_percent", si_logic.DEFAULT_ORIGINAL_VOLUME_PERCENT),
            si_volume_percent=read("dlna_si_volume_percent", 100),
            si_delay_seconds=read("dlna_si_delay_seconds", si_logic.DEFAULT_SI_DELAY_SECONDS),
            duck_original=read("dlna_si_duck_original", True),
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SIMixConfig":
        source = data or {}
        return cls(
            enabled=source.get("enabled", source.get("dlna_si_enabled", True)),
            mix_channel=source.get("mix_channel", source.get("dlna_si_mix_channel", "both")),
            original_volume_percent=source.get(
                "original_volume_percent",
                source.get("dlna_si_original_volume_percent", si_logic.DEFAULT_ORIGINAL_VOLUME_PERCENT),
            ),
            si_volume_percent=source.get(
                "si_volume_percent",
                source.get("dlna_si_volume_percent", 100),
            ),
            si_delay_seconds=source.get(
                "si_delay_seconds",
                source.get("dlna_si_delay_seconds", si_logic.DEFAULT_SI_DELAY_SECONDS),
            ),
            duck_original=source.get("duck_original", source.get("dlna_si_duck_original", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mix_channel": self.mix_channel,
            "original_volume_percent": self.original_volume_percent,
            "si_volume_percent": self.si_volume_percent,
            "si_delay_seconds": self.si_delay_seconds,
            "duck_original": self.duck_original,
        }

    def filter_string(self) -> str:
        return si_logic.build_si_mix_filter(
            self.mix_channel,
            self.original_volume_percent,
            self.si_volume_percent,
            self.si_delay_seconds,
            duck_original=self.duck_original,
        )


class ConfigHolder:
    def __init__(self, config: SIMixConfig | None = None) -> None:
        self._config = config or SIMixConfig()
        self._lock = threading.RLock()

    def get(self) -> SIMixConfig:
        with self._lock:
            return self._config

    def set(self, config: SIMixConfig) -> None:
        with self._lock:
            self._config = config


def parse_range_header(value: str | None) -> tuple[int, int | None]:
    """Parse a single HTTP bytes range, falling back to a full stream."""
    header = (value or "").strip()
    if not header.lower().startswith("bytes="):
        return 0, None
    spec = header[6:].split(",", 1)[0].strip()
    if not spec or spec.startswith("-") or "-" not in spec:
        return 0, None
    start_text, end_text = spec.split("-", 1)
    try:
        start = int(start_text)
        end = int(end_text) if end_text.strip() else None
    except ValueError:
        return 0, None
    if start < 0 or (end is not None and end < start):
        return 0, None
    return start, end


class LiveStreamSession:
    """One active ffmpeg stdout pipe for a virtual SI stream."""

    def __init__(
        self,
        video: Path,
        si_wav: Path,
        config: SIMixConfig,
        start_time: float,
        estimated_total: int,
        start_byte: int = 0,
    ) -> None:
        self.video = video
        self.si_wav = si_wav
        self.config = config
        self.estimated_total = max(1, int(estimated_total))
        self.start_time = max(0.0, float(start_time))
        self.byte_cursor = max(0, int(start_byte))
        self.last_used = time.monotonic()
        self.proc: subprocess.Popen[bytes] | None = None
        self.lock = threading.Lock()
        self._closed = False
        self._start_ffmpeg(self.start_time)

    def _start_ffmpeg(self, start_time: float) -> None:
        seek = f"{max(0.0, start_time):.3f}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            seek,
            "-i",
            str(self.video),
            "-ss",
            seek,
            "-i",
            str(self.si_wav),
            "-filter_complex",
            self.config.filter_string(),
            "-map",
            "0:v",
            "-c:v",
            "copy",
            "-map",
            "[si_track]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **hidden_subprocess_kwargs(),
        )
        log.info("Started SI ffmpeg stream video=%s si=%s seek=%s", self.video, self.si_wav, seek)

    def is_usable(self) -> bool:
        proc = self.proc
        return not self._closed and proc is not None and proc.stdout is not None and proc.poll() is None

    def read(self, n: int) -> bytes:
        with self.lock:
            if not self.is_usable() or self.proc is None or self.proc.stdout is None:
                return b""
            chunk = self.proc.stdout.read(max(1, int(n)))
            if chunk:
                self.byte_cursor += len(chunk)
                self.last_used = time.monotonic()
            return chunk

    def discard(self, n: int) -> int:
        remaining = max(0, int(n))
        discarded = 0
        while remaining > 0:
            chunk = self.read(min(DEFAULT_CHUNK_SIZE, remaining))
            if not chunk:
                break
            discarded += len(chunk)
            remaining -= len(chunk)
        return discarded

    def close(self) -> None:
        with self.lock:
            self._closed = True
            proc = self.proc
            self.proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception as exc:
            log.warning("Failed to terminate SI ffmpeg stream for %s: %s", self.video, exc)
        for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass


class SIStreamService:
    """Coordinates active live SI sessions and config hot reloads."""

    def __init__(
        self,
        media_library: Any = None,
        config_holder: ConfigHolder | None = None,
        *,
        session_factory: Callable[..., LiveStreamSession] = LiveStreamSession,
        reuse_tolerance_bytes: int = DEFAULT_REUSE_TOLERANCE_BYTES,
        seek_cooldown_seconds: float = DEFAULT_SEEK_COOLDOWN_SECONDS,
    ) -> None:
        self.media_library = media_library
        self._config_holder = config_holder or ConfigHolder()
        self._session_factory = session_factory
        self._reuse_tolerance_bytes = max(0, int(reuse_tolerance_bytes))
        self._seek_cooldown_seconds = max(0.0, float(seek_cooldown_seconds))
        self._sessions: dict[str, Any] = {}
        self._last_start_at: dict[str, float] = {}
        self._estimate_cache: dict[str, tuple[float, int]] = {}
        self._sessions_lock = threading.Lock()

    def current_config(self) -> SIMixConfig:
        return self._config_holder.get()

    def has_si_source(self, video: Path) -> Path | None:
        video = Path(video)
        sibling = video.with_suffix(".si.wav")
        if sibling.is_file():
            return sibling
        target_name = f"{video.stem}.si.wav".casefold()
        try:
            for child in video.parent.iterdir():
                if child.name.casefold() == target_name and child.is_file():
                    return child
        except OSError:
            pass
        return None

    def estimate_output_size(self, video: Path) -> int:
        video = safe_resolve_path(Path(video))
        try:
            mtime = video.stat().st_mtime
        except OSError:
            mtime = 0.0
        key = str(video)
        cached = self._estimate_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        meta = content_directory.probe_cached(video)
        try:
            file_size = int(meta.get("size") or video.stat().st_size or 0)
        except OSError:
            file_size = int(meta.get("size") or 0)
        video_size = int(meta.get("video_size") or file_size or 0)
        duration = max(0.0, float(meta.get("duration") or 0.0))
        audio_size = int(AUDIO_BITRATE_BPS * duration / 8) if duration > 0 else 0
        estimated = int((video_size + audio_size) * SIZE_OVERHEAD_FACTOR)
        estimated = max(estimated, file_size, MIN_ESTIMATED_SIZE)
        self._estimate_cache[key] = (mtime, estimated)
        return estimated

    def _duration(self, video: Path) -> float:
        meta = content_directory.probe_cached(video)
        try:
            return max(0.0, float(meta.get("duration") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _start_time_for_range(self, video: Path, range_start: int, total_size: int) -> float:
        duration = self._duration(video)
        if duration <= 0 or total_size <= 0:
            return 0.0
        ratio = min(1.0, max(0.0, range_start / total_size))
        return ratio * duration

    def _session_key(self, video: Path, client_id: str | None = None) -> str:
        base = str(safe_resolve_path(Path(video)))
        # The DLNA handler passes client IP as client_id, so clients sharing one IP also share a session.
        normalized_client = str(client_id or "").strip()
        return f"{base}\0{normalized_client}" if normalized_client else base

    def _can_reuse(self, session: Any, config: SIMixConfig, si_wav: Path, range_start: int) -> bool:
        if getattr(session, "config", None) != config:
            return False
        if Path(getattr(session, "si_wav", "")) != si_wav:
            return False
        if hasattr(session, "is_usable") and not session.is_usable():
            return False
        cursor = int(getattr(session, "byte_cursor", 0))
        return cursor <= range_start <= cursor + self._reuse_tolerance_bytes

    def _close_session(self, session: Any) -> None:
        try:
            session.close()
        except Exception as exc:
            log.warning("Failed to close SI stream session: %s", exc)

    def _get_or_start_session(
        self,
        video: Path,
        si_wav: Path,
        config: SIMixConfig,
        range_start: int,
        total_size: int,
        client_id: str | None,
    ) -> Any:
        key = self._session_key(video, client_id)
        with self._sessions_lock:
            session = self._sessions.get(key)
            if session is not None and self._can_reuse(session, config, si_wav, range_start):
                return session
            if session is not None:
                self._close_session(session)
                self._sessions.pop(key, None)

            last_start = self._last_start_at.get(key, 0.0)
            wait_seconds = self._seek_cooldown_seconds - (time.monotonic() - last_start)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            start_time = self._start_time_for_range(video, range_start, total_size)
            session = self._session_factory(
                video=video,
                si_wav=si_wav,
                config=config,
                start_time=start_time,
                estimated_total=total_size,
                start_byte=range_start,
            )
            self._sessions[key] = session
            self._last_start_at[key] = time.monotonic()
            return session

    def _drop_session_if_current(
        self,
        video: Path,
        session: Any,
        *,
        client_id: str | None,
        close: bool,
    ) -> None:
        key = self._session_key(video, client_id)
        with self._sessions_lock:
            if self._sessions.get(key) is not session:
                return
            self._sessions.pop(key, None)
        if close:
            self._close_session(session)

    def open_stream(
        self,
        video: Path,
        range_start: int = 0,
        range_end: int | None = None,
        *,
        client_id: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> tuple[Iterator[bytes], int, int, int]:
        video = safe_resolve_path(Path(video))
        config = self.current_config()
        if not config.enabled:
            raise FileNotFoundError("SI streaming is disabled")
        si_wav = self.has_si_source(video)
        if si_wav is None:
            raise FileNotFoundError("No sibling SI WAV file")

        total_size = self.estimate_output_size(video)
        safe_start = min(max(0, int(range_start)), max(0, total_size - 1))
        safe_end = min(int(range_end), total_size - 1) if range_end is not None else total_size - 1
        if safe_end < safe_start:
            safe_end = total_size - 1
        content_length = max(0, safe_end - safe_start + 1)
        status_code = 206 if safe_start > 0 or range_end is not None else 200
        session = self._get_or_start_session(video, si_wav, config, safe_start, total_size, client_id)

        def chunks() -> Iterator[bytes]:
            remaining = content_length
            saw_eof = False
            try:
                cursor = int(getattr(session, "byte_cursor", safe_start))
                if cursor < safe_start and hasattr(session, "discard"):
                    session.discard(safe_start - cursor)
                while remaining > 0:
                    chunk = session.read(min(chunk_size, remaining))
                    if not chunk:
                        saw_eof = True
                        break
                    remaining -= len(chunk)
                    yield chunk
            finally:
                # DLNA players open a NEW TCP connection for each Range request and
                # close it as soon as they have enough bytes. That means GeneratorExit
                # fires here with remaining > 0 on every healthy sequential read.
                # Killing the session here would force a fresh ffmpeg restart (with a
                # fresh moov atom) on every Range request, which corrupts playback.
                # Only drop the session when ffmpeg actually reached EOF; idle ffmpegs
                # self-throttle on a full stdout pipe until the next reader consumes it.
                if saw_eof:
                    self._drop_session_if_current(video, session, client_id=client_id, close=True)

        return chunks(), content_length, total_size, status_code

    def reload_config(self, new_config: SIMixConfig) -> None:
        self._config_holder.set(new_config)
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            self._close_session(session)
        log.info("Reloaded DLNA SI config: %s", new_config.as_dict())

    def shutdown(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            self._close_session(session)


def iter_si_mpegts(
    video: Path,
    si_wav: Path,
    config: SIMixConfig,
    start_time: float,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[bytes]:
    """Yield a realtime MPEG-TS SI mix stream from ``start_time`` seconds."""
    seek = f"{max(0.0, float(start_time or 0.0)):.3f}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        seek,
        "-i",
        str(video),
        "-ss",
        seek,
        "-i",
        str(si_wav),
        "-filter_complex",
        config.filter_string(),
        "-map",
        "0:v",
        "-c:v",
        "copy",
        "-map",
        "[si_track]",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-muxpreload",
        "0",
        "-muxdelay",
        "0",
        "-f",
        "mpegts",
        "pipe:1",
    ]
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )
        log.info("Started SI MPEG-TS stream video=%s si=%s seek=%s", video, si_wav, seek)
        if proc.stdout is None:
            return
        read_size = max(1, int(chunk_size))
        while True:
            try:
                chunk = proc.stdout.read(read_size)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            yield chunk
    finally:
        if proc is not None:
            for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
                try:
                    if pipe is not None:
                        pipe.close()
                except Exception:
                    pass
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
            except Exception as exc:
                log.warning("Failed to terminate SI MPEG-TS stream for %s: %s", video, exc)
