from __future__ import annotations

import bisect
import re
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf

from tool_subtitle.audio_player import AudioPlayerError, WinMMAudioPlayer
from tool_subtitle import logic
from utils import i18n


def tr(key: str) -> str:
    return i18n.translate("subtitle", key)


@dataclass(frozen=True)
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str


_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
_WAVEFORM_MIN_GAIN = 2.0
_WAVEFORM_MAX_GAIN = 12.0
_WAVEFORM_TARGET_PEAK = 0.88


def _parse_time(value: str) -> float:
    match = _TIME_RE.search(value.strip())
    if not match:
        raise ValueError(value)
    hours, minutes, seconds, fraction = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(fraction.ljust(3, "0")[:3]) / 1000


def parse_srt(path: str | Path) -> list[SubtitleEntry]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    entries: list[SubtitleEntry] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = block.splitlines()
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        try:
            start_text, end_text = lines[timing_index].split("-->", 1)
            start, end = _parse_time(start_text), _parse_time(end_text)
        except (ValueError, IndexError):
            continue
        number = len(entries) + 1
        if timing_index and lines[timing_index - 1].strip().isdigit():
            number = int(lines[timing_index - 1].strip())
        body = "\n".join(line.strip() for line in lines[timing_index + 1:]).strip()
        entries.append(SubtitleEntry(number, start, max(start, end), body))
    return sorted(entries, key=lambda item: (item.start, item.end, item.index))


def build_peak_envelope(samples: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if len(mono) == 0 or buckets <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    buckets = min(buckets, len(mono))
    # Vectorized fixed-size reduction is substantially faster than a Python
    # loop over up to 200,000 display buckets on long recordings.
    edges = np.linspace(0, len(mono), buckets + 1, dtype=np.int64)
    low = np.minimum.reduceat(mono, edges[:-1]).astype(np.float32, copy=False)
    high = np.maximum.reduceat(mono, edges[:-1]).astype(np.float32, copy=False)
    return low, high


def waveform_display_gain(low: np.ndarray, high: np.ndarray) -> float:
    """Calculate display-only gain so quiet speech remains visible."""
    low_values = np.asarray(low, dtype=np.float32)
    high_values = np.asarray(high, dtype=np.float32)
    if not len(low_values) or not len(high_values):
        return _WAVEFORM_MIN_GAIN
    peaks = np.maximum(np.abs(low_values), np.abs(high_values))
    reference_peak = float(np.percentile(peaks, 99.0))
    if not np.isfinite(reference_peak) or reference_peak <= 1e-6:
        return _WAVEFORM_MIN_GAIN
    return float(np.clip(
        _WAVEFORM_TARGET_PEAK / reference_peak,
        _WAVEFORM_MIN_GAIN,
        _WAVEFORM_MAX_GAIN,
    ))


def find_debug_sessions(base_dir: str | Path | None) -> list[Path]:
    if not base_dir:
        return []
    root = Path(base_dir)
    if not root.exists():
        return []
    sessions: list[Path] = []
    # filedialog.askdirectory commonly returns the debug directory itself.
    # Path.rglob() only yields descendants, so explicitly accept the selected
    # root when it already contains a retained debug WAV.
    if root.is_dir() and any(root.glob("*.wav")):
        sessions.append(root)
    sessions.extend(
        path for path in root.rglob("*_debug")
        if path.is_dir() and any(path.glob("*.wav")) and path != root
    )
    sessions.extend(
        path for path in root.rglob("*.clone")
        if path.is_dir() and any(path.glob("*.wav")) and path != root
    )
    return sorted(sessions, key=lambda path: path.stat().st_mtime, reverse=True)


def find_media_subtitles(media_path: str | Path) -> list[Path]:
    media = Path(media_path)
    excluded = (".raw.srt", ".vad.srt", ".removed.srt")
    candidates = [
        path for path in media.parent.glob(f"{media.stem}*.srt")
        if path.is_file() and not path.name.lower().endswith(excluded)
        and (path.stem == media.stem or path.name.startswith(f"{media.stem}."))
    ]
    def priority(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if name == f"{media.stem.lower()}.srt":
            return (0, name)
        if name == f"{media.stem.lower()}.jp.srt":
            return (1, name)
        return (2, name)
    return sorted(candidates, key=priority)


class SubtitleDebugAnalyzer:
    TRACK_SUFFIXES = {"final": ".jp.srt", "raw": ".raw.srt", "vad": ".vad.srt", "removed": ".removed.srt"}

    def __init__(self, parent: tk.Misc, base_dir: str = ""):
        self.window = tk.Toplevel(parent)
        self.window.title(tr("debug_analyzer_title"))
        self.window.geometry("1200x780")
        self.window.minsize(800, 560)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        # An empty generation directory must not default to the repository
        # root: recursively scanning cwd on the Tk main thread can make the
        # window appear frozen. The user can choose a debug directory from the
        # analyzer itself.
        self.base_dir = Path(base_dir) if base_dir else None
        self.audio: np.ndarray | None = None
        self.sample_rate = 0
        self.duration = 0.0
        self.entries: list[SubtitleEntry] = []
        self.starts: list[float] = []
        self.player: WinMMAudioPlayer | None = None
        self.position_sec = 0.0
        self.pixels_per_second = 80.0
        self.wave_low = np.zeros(0, dtype=np.float32)
        self.wave_high = np.zeros(0, dtype=np.float32)
        self.wave_gain = _WAVEFORM_MIN_GAIN
        self._load_token = 0
        self._changing_selection = False
        self._redraw_job = None
        self._drag_origin: tuple[int, int] | None = None
        self._dragging = False
        self.mode = "debug"
        self.media_path: Path | None = None
        self.track_paths: dict[str, Path] = {}
        self.wav_generating = False
        self.wav_stop_event = threading.Event()
        self._build_ui()
        self._populate_sessions()
        self.window.after(40, self._tick)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.window, padding=8)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text=tr("debug_session")).pack(side="left")
        self.session_var = tk.StringVar()
        self.session_box = ttk.Combobox(toolbar, textvariable=self.session_var, state="readonly", width=55)
        self.session_box.pack(side="left", padx=6, fill="x", expand=True)
        self.session_box.bind("<<ComboboxSelected>>", lambda _e: self._load_selected_session())
        ttk.Button(toolbar, text=tr("debug_refresh"), command=self._populate_sessions).pack(side="left", padx=3)
        ttk.Button(toolbar, text=tr("debug_open_folder"), command=self._choose_folder).pack(side="left", padx=3)
        ttk.Button(toolbar, text=tr("debug_select_video"), command=self._choose_media).pack(side="left", padx=3)

        trackbar = ttk.Frame(self.window, padding=(8, 0, 8, 6))
        trackbar.pack(fill="x")
        ttk.Label(trackbar, text=tr("debug_track")).pack(side="left")
        self.track_var = tk.StringVar(value="final")
        self.track_box = ttk.Combobox(trackbar, textvariable=self.track_var, state="readonly", width=18,
                                     values=("final", "raw", "vad", "removed"))
        self.track_box.pack(side="left", padx=6)
        self.track_box.bind("<<ComboboxSelected>>", lambda _e: self._load_track())
        self.play_button = ttk.Button(trackbar, text=tr("debug_play"), command=self._toggle_play)
        self.play_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(trackbar, text=tr("debug_stop"), command=self._stop)
        self.stop_button.pack(side="left", padx=3)
        self.status_var = tk.StringVar(value="")
        ttk.Label(trackbar, textvariable=self.status_var).pack(side="left", padx=10)
        self.time_label = ttk.Label(trackbar, text="00:00:00.000 / 00:00:00.000")
        self.time_label.pack(side="right")

        progress_frame = ttk.Frame(self.window, padding=(8, 0, 8, 6))
        progress_frame.pack(fill="x")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Scale(progress_frame, variable=self.progress_var, from_=0.0, to=1.0)
        self.progress.pack(fill="x", expand=True)
        self.progress.bind("<ButtonRelease-1>", lambda _e: self._seek(self.progress_var.get(), center=True))

        pane = ttk.Panedwindow(self.window, orient="vertical")
        pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        list_frame = ttk.Frame(pane)
        wave_frame = ttk.Frame(pane)
        pane.add(list_frame, weight=2)
        pane.add(wave_frame, weight=3)

        columns = ("index", "start", "end", "duration", "text")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        widths = (55, 105, 105, 80, 700)
        for column, width in zip(columns, widths):
            self.tree.heading(column, text=tr(f"debug_col_{column}"))
            self.tree.column(column, width=width, stretch=column == "text")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._select_entry)

        self.canvas = tk.Canvas(wave_frame, bg="#101418", highlightthickness=0, height=300)
        xscroll = ttk.Scrollbar(wave_frame, orient="horizontal", command=self._scroll_canvas)
        self.canvas.pack(fill="both", expand=True)
        xscroll.pack(fill="x")
        self.canvas.configure(xscrollcommand=xscroll.set)
        self.canvas.bind("<Configure>", lambda _e: self._schedule_redraw())
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.subtitle_var = tk.StringVar(value="")
        ttk.Label(wave_frame, textvariable=self.subtitle_var, anchor="center", font=("Arial", 13, "bold"),
                  wraplength=1000).pack(fill="x", pady=(8, 0))

    def _choose_folder(self) -> None:
        chosen = filedialog.askdirectory(parent=self.window, initialdir=str(self.base_dir or Path.cwd()))
        if chosen:
            self.base_dir = Path(chosen)
            self._populate_sessions()

    def _choose_media(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self.window,
            initialdir=str(self.base_dir or Path.cwd()),
            filetypes=[(tr("debug_video_files"), "*.mp4 *.mkv"), (tr("debug_all_files"), "*.*")],
        )
        if chosen:
            self._load_media(Path(chosen))

    def _load_media(self, media: Path) -> None:
        self.mode = "media"
        self.media_path = media
        self._load_token += 1
        if self.player:
            self.player.close()
        self.player = None
        self.audio = None
        self.wave_low = np.zeros(0, dtype=np.float32)
        self.wave_high = np.zeros(0, dtype=np.float32)
        self.wave_gain = _WAVEFORM_MIN_GAIN
        self.duration = 0.0
        self.base_dir = media.parent
        self.session_var.set(str(media))
        subtitles = find_media_subtitles(media)
        self.track_paths = {path.name: path for path in subtitles}
        labels = list(self.track_paths)
        self.track_box.configure(values=labels)
        self.track_var.set(labels[0] if labels else "")
        self._load_track()
        wav = logic.analysis_wav_path(media)
        self.wav_path = wav
        self.session = wav.parent
        if wav.exists():
            self._start_audio_load(wav)
        else:
            self._set_subtitle_only()
            self.window.after(10, self._offer_generate_wav)

    def _populate_sessions(self) -> None:
        self.sessions = find_debug_sessions(self.base_dir) if self.base_dir else []
        labels = [str(path) for path in self.sessions]
        self.session_box.configure(values=labels)
        if labels:
            self.session_var.set(labels[0])
            self._load_selected_session()
        else:
            self.session_var.set("")

    def _load_selected_session(self) -> None:
        value = self.session_var.get()
        if not value:
            return
        session = Path(value)
        self.mode = "clone" if session.name.lower().endswith(".clone") else "debug"
        self.media_path = None
        self.track_paths = {}
        if self.mode == "clone":
            clone_tracks = sorted(session.glob("*.srt"), key=lambda path: (0 if path.name == "source.srt" else 1, path.name.lower()))
            self.track_paths = {path.name: path for path in clone_tracks}
            values = list(self.track_paths) or ["source.srt"]
            self.track_box.configure(values=values)
            self.track_var.set(values[0])
        else:
            self.track_box.configure(values=("final", "raw", "vad", "removed"))
            if self.track_var.get() not in ("final", "raw", "vad", "removed"):
                self.track_var.set("final")
        wavs = sorted(session.glob("*.wav"), key=lambda path: (0 if path.name == "audio16k.wav" else 1, path.name.lower()))
        if not wavs:
            messagebox.showwarning(tr("debug_analyzer_title"), tr("debug_no_wav"), parent=self.window)
            return
        self.session = session
        self.wav_path = wavs[0]
        self._start_audio_load(self.wav_path)
        self._load_track()

    def _start_audio_load(self, path: Path) -> None:
        self._load_token += 1
        token = self._load_token
        self.status_var.set(tr("debug_loading_wav"))
        threading.Thread(target=self._load_audio_worker, args=(token, path), daemon=True).start()

    def _load_audio_worker(self, token: int, path: Path) -> None:
        try:
            data, rate = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = np.ascontiguousarray(data)
            low, high = build_peak_envelope(data, max(2000, min(200000, int(len(data) / max(1, rate) * 100))))
            self.window.after(0, lambda: self._accept_audio(token, data, rate, low, high))
        except Exception as exc:
            self.window.after(0, lambda: messagebox.showerror(tr("debug_analyzer_title"), str(exc), parent=self.window))

    def _accept_audio(self, token: int, data: np.ndarray, rate: int, low: np.ndarray, high: np.ndarray) -> None:
        if token != self._load_token:
            return
        if self.player:
            self.player.close()
        self.audio, self.sample_rate = data, int(rate)
        self.duration = len(data) / rate if rate else 0.0
        self.progress.configure(to=max(0.001, self.duration))
        self.wave_low, self.wave_high = low, high
        self.wave_gain = waveform_display_gain(low, high)
        try:
            self.player = WinMMAudioPlayer(data, rate)
        except AudioPlayerError as exc:
            self.player = None
            messagebox.showwarning(tr("debug_analyzer_title"), str(exc), parent=self.window)
        self.position_sec = 0.0
        self.status_var.set("")
        self.play_button.configure(state="normal")
        self.stop_button.configure(state="normal")
        # Start with a local timeline window instead of squeezing the complete
        # recording into one screen. The user can pan horizontally or press Fit All.
        self.pixels_per_second = max(30.0, self.canvas.winfo_width() / 30.0)
        self.canvas.xview_moveto(0)
        self._schedule_redraw()

    def _track_path(self) -> Path | None:
        if self.mode == "clone":
            return self.track_paths.get(self.track_var.get())
        if self.mode == "media":
            return self.track_paths.get(self.track_var.get())
        if not hasattr(self, "session"):
            return None
        stem = self.wav_path.stem
        kind = self.track_var.get()
        if kind == "final":
            candidates = [self.session.parent / f"{stem}.jp.srt", self.session.parent / f"{stem}.srt",
                          self.session / f"{stem}.srt"]
        else:
            candidates = [self.session / f"{stem}{self.TRACK_SUFFIXES[kind]}"]
        return next((path for path in candidates if path.exists()), None)

    def _load_track(self) -> None:
        path = self._track_path()
        self.entries = parse_srt(path) if path else []
        self.starts = [entry.start for entry in self.entries]
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row, entry in enumerate(self.entries):
            self.tree.insert("", "end", iid=str(row), values=(entry.index, self._format(entry.start),
                             self._format(entry.end), f"{entry.end-entry.start:.3f}", entry.text.replace("\n", " / ")))
        if self.audio is None and self.entries:
            self.duration = max(self.duration, max(entry.end for entry in self.entries))
            self.progress.configure(to=max(0.001, self.duration))
        self._schedule_redraw()

    def _set_subtitle_only(self) -> None:
        if self.player:
            self.player.close()
        self.player = None
        self.audio = None
        self.sample_rate = 0
        self.wave_low = np.zeros(0, dtype=np.float32)
        self.wave_high = np.zeros(0, dtype=np.float32)
        self.wave_gain = _WAVEFORM_MIN_GAIN
        self.duration = max((entry.end for entry in self.entries), default=1.0)
        self.progress.configure(to=max(0.001, self.duration))
        self.play_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status_var.set(tr("debug_subtitle_only"))
        self._schedule_redraw()

    def _offer_generate_wav(self) -> None:
        if self.mode != "media" or not self.media_path or self.wav_generating or self.audio is not None:
            return
        if messagebox.askyesno(tr("debug_analyzer_title"), tr("debug_missing_wav_prompt"), parent=self.window):
            self._generate_wav()

    def _generate_wav(self) -> None:
        if not self.media_path or self.wav_generating:
            return
        self.wav_generating = True
        self.wav_stop_event.clear()
        self.status_var.set(tr("debug_generating_wav"))
        media = self.media_path
        token = self._load_token = self._load_token + 1
        def worker():
            try:
                result = logic.generate_analysis_wav(media, lambda _msg: None, self.wav_stop_event)
                self.window.after(0, lambda: self._finish_wav_generation(token, result))
            except Exception as exc:
                self.window.after(0, lambda: self._finish_wav_generation(token, None, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_wav_generation(self, token: int, result: str | None, error: str = "") -> None:
        self.wav_generating = False
        if token != self._load_token:
            return
        if result:
            self.wav_path = Path(result)
            self._start_audio_load(self.wav_path)
        else:
            self.status_var.set(tr("debug_subtitle_only"))
            if error:
                messagebox.showerror(tr("debug_analyzer_title"), error, parent=self.window)
            else:
                messagebox.showwarning(tr("debug_analyzer_title"), tr("debug_wav_failed"), parent=self.window)

    @staticmethod
    def _format(value: float) -> str:
        millis = int(max(0, value) * 1000)
        return f"{millis//3600000:02d}:{millis//60000%60:02d}:{millis//1000%60:02d}.{millis%1000:03d}"

    def _select_entry(self, _event=None) -> None:
        if self._changing_selection:
            return
        selected = self.tree.selection()
        if selected:
            entry = self.entries[int(selected[0])]
            self._seek(entry.start, center=True)

    def _seek(self, value: float, center: bool = False) -> None:
        self.position_sec = max(0.0, min(self.duration, value))
        if self.player and self.player.is_playing:
            self.player.seek(self.position_sec)
        if center:
            width = max(1, int(self.canvas.winfo_width()))
            total = max(width, int(self.duration * self.pixels_per_second))
            left = max(0, self.position_sec * self.pixels_per_second - width / 2)
            self.canvas.xview_moveto(left / total)
        self._update_active_entry()
        self.progress_var.set(self.position_sec)
        self._schedule_redraw()

    def _canvas_press(self, event) -> None:
        self._drag_origin = (event.x, event.y)
        self._dragging = False
        self.canvas.scan_mark(event.x, event.y)

    def _canvas_drag(self, event) -> None:
        if self._drag_origin is None:
            return
        if abs(event.x - self._drag_origin[0]) >= 4:
            self._dragging = True
        if self._dragging:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self._schedule_redraw()

    def _canvas_release(self, event) -> None:
        if not self._dragging:
            if self.audio is None and self.mode == "media":
                self._offer_generate_wav()
            else:
                x = self.canvas.canvasx(event.x)
                self._seek(x / self.pixels_per_second)
        self._drag_origin = None
        self._dragging = False

    def _toggle_play(self) -> None:
        if not self.player:
            return
        if self.player.is_playing:
            if self.player.is_paused:
                self.player.resume()
            else:
                self.player.pause()
        else:
            self.player.play(self.position_sec)

    def _stop(self) -> None:
        if self.player:
            self.position_sec = self.player.position()
            self.player.stop()
        self._schedule_redraw()

    def _fit(self) -> None:
        width = max(100, self.canvas.winfo_width() or 1000)
        self.pixels_per_second = max(0.2, width / max(1.0, self.duration))
        self.canvas.xview_moveto(0)
        self._schedule_redraw()

    def _zoom(self, factor: float) -> None:
        self.pixels_per_second = max(0.2, min(1000.0, self.pixels_per_second * factor))
        self._schedule_redraw()

    def _mousewheel(self, event) -> None:
        if event.state & 0x0004:
            self._zoom(1.25 if event.delta > 0 else 0.8)
        else:
            self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
            self._schedule_redraw()

    def _scroll_canvas(self, *args) -> None:
        self.canvas.xview(*args)
        self._schedule_redraw()

    def _schedule_redraw(self) -> None:
        if self._redraw_job is None and self.window.winfo_exists():
            self._redraw_job = self.window.after_idle(self._redraw)

    def _redraw(self) -> None:
        self._redraw_job = None
        canvas = self.canvas
        canvas.delete("static")
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        total_width = max(width, int(self.duration * self.pixels_per_second))
        canvas.configure(scrollregion=(0, 0, total_width, height))
        left = canvas.canvasx(0)
        right = canvas.canvasx(width)
        wave_top, wave_bottom = 30, max(80, int(height * 0.68))
        center = (wave_top + wave_bottom) / 2
        amp = (wave_bottom - wave_top) * 0.46
        if len(self.wave_low) and self.duration:
            first = max(0, int(left / total_width * len(self.wave_low)))
            last = min(len(self.wave_low), int(right / total_width * len(self.wave_low)) + 2)
            for i in range(first, last):
                x = i / len(self.wave_low) * total_width
                high = float(np.clip(self.wave_high[i] * self.wave_gain, -1.0, 1.0))
                low = float(np.clip(self.wave_low[i] * self.wave_gain, -1.0, 1.0))
                canvas.create_line(x, center-high*amp, x, center-low*amp,
                                   fill="#58b7ff", tags="static")
        elif self.mode == "media":
            canvas.create_text((left+right)/2, center, text=tr("debug_waveform_missing_hint"),
                               fill="#c7d0d9", font=("Arial", 11), tags="static")
        canvas.create_line(left, center, right, center, fill="#52606b", tags="static")
        interval = next((v for v in (1, 2, 5, 10, 30, 60, 300, 600) if v*self.pixels_per_second >= 70), 1800)
        tick = int(left / self.pixels_per_second / interval) * interval
        while tick <= right / self.pixels_per_second:
            x = tick * self.pixels_per_second
            canvas.create_line(x, 0, x, 18, fill="#8b98a5", tags="static")
            canvas.create_text(x+3, 18, text=self._format(tick), anchor="nw", fill="#c7d0d9",
                               font=("Arial", 8), tags="static")
            tick += interval
        lane_top = wave_bottom + 8
        # Only visit subtitles that can intersect the visible time range. This
        # avoids scanning thousands of entries on every pan/zoom operation.
        visible_start = max(0.0, left / self.pixels_per_second)
        visible_end = right / self.pixels_per_second
        first_row = max(0, bisect.bisect_left(self.starts, visible_start) - 1)
        last_row = bisect.bisect_right(self.starts, visible_end)
        for row in range(first_row, last_row):
            entry = self.entries[row]
            x1, x2 = entry.start*self.pixels_per_second, max(entry.start*self.pixels_per_second+3, entry.end*self.pixels_per_second)
            if x2 < left or x1 > right:
                continue
            # A subtitle range is a fenced interval: contrasting fill plus
            # explicit left/right boundary rails make its timing unambiguous.
            canvas.create_rectangle(x1, wave_top, x2, height-8, fill="#284b3a", outline="",
                                    stipple="gray25", tags="static")
            canvas.create_rectangle(x1, lane_top, x2, height-8, fill="#284b3a", outline="", tags="static")
            canvas.create_line(x1, wave_top-4, x1, height-3, fill="#8dffad", width=2, tags="static")
            canvas.create_line(x2, wave_top-4, x2, height-3, fill="#8dffad", width=2, tags="static")
            if x2 - x1 > 45:
                canvas.create_text((x1+x2)/2, (lane_top+height-8)/2, text=str(entry.index),
                                   fill="#d7ffe1", font=("Arial", 8), tags="static")
        self._update_cursor()

    def _update_cursor(self) -> None:
        height = max(1, self.canvas.winfo_height())
        cursor = self.position_sec * self.pixels_per_second
        if self.canvas.find_withtag("play_cursor"):
            self.canvas.coords("play_cursor", cursor, 0, cursor, height)
        else:
            self.canvas.create_line(cursor, 0, cursor, height, fill="#ff5252", width=2,
                                    tags="play_cursor")
        self.canvas.tag_raise("play_cursor")

    def _update_active_entry(self) -> None:
        active = None
        if self.entries:
            index = bisect.bisect_right(self.starts, self.position_sec) - 1
            if index >= 0 and self.entries[index].start <= self.position_sec <= self.entries[index].end:
                active = index
        if active is None:
            self.subtitle_var.set("")
            return
        entry = self.entries[active]
        self.subtitle_var.set(entry.text)
        iid = str(active)
        if self.tree.selection() != (iid,):
            self._changing_selection = True
            try:
                self.tree.selection_set(iid)
                self.tree.see(iid)
            finally:
                self._changing_selection = False

    def _tick(self) -> None:
        if not self.window.winfo_exists():
            return
        if self.player and self.player.is_playing:
            self.position_sec = self.player.position()
            self._update_active_entry()
            self.progress_var.set(self.position_sec)
            left = self.canvas.canvasx(0)
            right = self.canvas.canvasx(self.canvas.winfo_width())
            cursor = self.position_sec * self.pixels_per_second
            if cursor > right - 30 or cursor < left:
                total = max(1.0, self.duration * self.pixels_per_second)
                self.canvas.xview_moveto(max(0.0, cursor - 60) / total)
                self._schedule_redraw()
            else:
                self._update_cursor()
        self.time_label.configure(text=f"{self._format(self.position_sec)} / {self._format(self.duration)}")
        self.window.after(40, self._tick)

    def close(self) -> None:
        self._load_token += 1
        self.wav_stop_event.set()
        if self.player:
            self.player.close()
        self.window.destroy()
