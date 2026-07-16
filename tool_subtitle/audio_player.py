from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes

import numpy as np


class AudioPlayerError(RuntimeError):
    pass


if os.name == "nt":
    HWAVEOUT = wintypes.HANDLE
    MMSYSERR_NOERROR = 0
    WAVE_MAPPER = -1
    CALLBACK_NULL = 0
    WHDR_DONE = 0x00000001
    TIME_SAMPLES = 0x0002

    class WAVEFORMATEX(ctypes.Structure):
        _fields_ = [
            ("wFormatTag", wintypes.WORD),
            ("nChannels", wintypes.WORD),
            ("nSamplesPerSec", wintypes.DWORD),
            ("nAvgBytesPerSec", wintypes.DWORD),
            ("nBlockAlign", wintypes.WORD),
            ("wBitsPerSample", wintypes.WORD),
            ("cbSize", wintypes.WORD),
        ]

    class WAVEHDR(ctypes.Structure):
        _fields_ = [
            ("lpData", ctypes.c_void_p),
            ("dwBufferLength", wintypes.DWORD),
            ("dwBytesRecorded", wintypes.DWORD),
            ("dwUser", ctypes.c_size_t),
            ("dwFlags", wintypes.DWORD),
            ("dwLoops", wintypes.DWORD),
            ("lpNext", ctypes.c_void_p),
            ("reserved", ctypes.c_size_t),
        ]

    class MMTIME_VALUE(ctypes.Union):
        # The native union also contains an 8-byte SMPTE structure. Keeping the
        # correct union size is required or waveOutGetPosition returns MMSYSERR_ERROR.
        _fields_ = [("ms", wintypes.DWORD), ("sample", wintypes.DWORD), ("cb", wintypes.DWORD),
                    ("ticks", wintypes.DWORD), ("smpte", ctypes.c_ubyte * 8)]

    class MMTIME(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("wType", wintypes.UINT), ("u", MMTIME_VALUE)]


class WinMMAudioPlayer:
    """Small PCM player backed by Windows waveOut; no external player is required."""

    def __init__(self, samples: np.ndarray, sample_rate: int):
        if os.name != "nt":
            raise AudioPlayerError("WinMM playback is only available on Windows")
        mono = np.asarray(samples)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        if np.issubdtype(mono.dtype, np.floating):
            mono = np.clip(mono, -1.0, 1.0)
            mono = (mono * 32767.0).astype(np.int16)
        else:
            mono = mono.astype(np.int16, copy=False)
        self.samples = np.ascontiguousarray(mono)
        self.sample_rate = int(sample_rate)
        self.duration = len(self.samples) / self.sample_rate if self.sample_rate else 0.0
        self._winmm = ctypes.WinDLL("winmm")
        self._winmm.waveOutOpen.argtypes = [ctypes.POINTER(HWAVEOUT), ctypes.c_size_t,
                                            ctypes.POINTER(WAVEFORMATEX), ctypes.c_size_t,
                                            ctypes.c_size_t, wintypes.DWORD]
        self._winmm.waveOutOpen.restype = wintypes.UINT
        for name in ("waveOutPrepareHeader", "waveOutWrite", "waveOutUnprepareHeader"):
            function = getattr(self._winmm, name)
            function.argtypes = [HWAVEOUT, ctypes.POINTER(WAVEHDR), wintypes.UINT]
            function.restype = wintypes.UINT
        for name in ("waveOutPause", "waveOutRestart", "waveOutReset", "waveOutClose"):
            function = getattr(self._winmm, name)
            function.argtypes = [HWAVEOUT]
            function.restype = wintypes.UINT
        self._winmm.waveOutGetPosition.argtypes = [HWAVEOUT, ctypes.POINTER(MMTIME), wintypes.UINT]
        self._winmm.waveOutGetPosition.restype = wintypes.UINT
        self._handle = HWAVEOUT()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._start_sample = 0
        self._paused = False
        self._playing = False

    def _check(self, result: int, operation: str) -> None:
        if result != MMSYSERR_NOERROR:
            raise AudioPlayerError(f"{operation} failed (WinMM error {result})")

    def play(self, position: float = 0.0) -> None:
        self.stop()
        start = max(0, min(len(self.samples), int(position * self.sample_rate)))
        if start >= len(self.samples):
            return
        with self._lock:
            fmt = WAVEFORMATEX(1, 1, self.sample_rate, self.sample_rate * 2, 2, 16, 0)
            self._handle = HWAVEOUT()
            self._check(
                self._winmm.waveOutOpen(ctypes.byref(self._handle), ctypes.c_size_t(WAVE_MAPPER),
                                        ctypes.byref(fmt), 0, 0, CALLBACK_NULL),
                "waveOutOpen",
            )
            self._start_sample = start
            self._paused = False
            self._playing = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._stream, args=(start,), daemon=True)
            self._thread.start()

    def _stream(self, start: int) -> None:
        chunk_samples = max(self.sample_rate // 2, 4096)
        cursor = start
        active: list[tuple[WAVEHDR, ctypes.Array]] = []
        try:
            while not self._stop_event.is_set() and (cursor < len(self.samples) or active):
                while cursor < len(self.samples) and len(active) < 4 and not self._stop_event.is_set():
                    end = min(len(self.samples), cursor + chunk_samples)
                    payload = self.samples[cursor:end].tobytes()
                    buffer = ctypes.create_string_buffer(payload)
                    header = WAVEHDR(ctypes.cast(buffer, ctypes.c_void_p), len(payload), 0, 0, 0, 0, None, 0)
                    self._check(self._winmm.waveOutPrepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header)),
                                "waveOutPrepareHeader")
                    self._check(self._winmm.waveOutWrite(self._handle, ctypes.byref(header), ctypes.sizeof(header)),
                                "waveOutWrite")
                    active.append((header, buffer))
                    cursor = end
                remaining: list[tuple[WAVEHDR, ctypes.Array]] = []
                for header, buffer in active:
                    if header.dwFlags & WHDR_DONE:
                        self._winmm.waveOutUnprepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
                    else:
                        remaining.append((header, buffer))
                active = remaining
                time.sleep(0.01)
        finally:
            handle = self._handle
            if handle:
                self._winmm.waveOutReset(handle)
                for header, _buffer in active:
                    self._winmm.waveOutUnprepareHeader(handle, ctypes.byref(header), ctypes.sizeof(header))
                self._winmm.waveOutClose(handle)
            with self._lock:
                self._handle = HWAVEOUT()
                self._playing = False
                self._paused = False

    def pause(self) -> None:
        with self._lock:
            if self._playing and not self._paused and self._handle:
                self._check(self._winmm.waveOutPause(self._handle), "waveOutPause")
                self._paused = True

    def resume(self) -> None:
        with self._lock:
            if self._playing and self._paused and self._handle:
                self._check(self._winmm.waveOutRestart(self._handle), "waveOutRestart")
                self._paused = False

    def stop(self) -> None:
        thread = self._thread
        self._stop_event.set()
        with self._lock:
            if self._handle:
                self._winmm.waveOutReset(self._handle)
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    def seek(self, position: float, continue_playing: bool = True) -> None:
        if continue_playing:
            self.play(position)
        else:
            self.stop()
            self._start_sample = max(0, min(len(self.samples), int(position * self.sample_rate)))

    def position(self) -> float:
        with self._lock:
            if not self._playing or not self._handle:
                return self._start_sample / self.sample_rate
            value = MMTIME()
            value.wType = TIME_SAMPLES
            result = self._winmm.waveOutGetPosition(self._handle, ctypes.byref(value), ctypes.sizeof(value))
            if result != MMSYSERR_NOERROR:
                return self._start_sample / self.sample_rate
            return min(self.duration, (self._start_sample + int(value.sample)) / self.sample_rate)

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_paused(self) -> bool:
        return self._paused

    def close(self) -> None:
        self.stop()
