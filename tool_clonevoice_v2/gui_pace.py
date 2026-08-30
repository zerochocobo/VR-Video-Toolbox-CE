"""Side-by-side view of what was said and what the dub says in its place.

Built because the pacing question cannot be answered by arithmetic. IndexTTS
renders about twice the length the translated text predicts -- it fills lines
out with interjections -- so how much of a source line the dub actually covers
is only knowable by measuring the finished track. This tab measures it, shows
both texts against each other, and plays either side of the comparison.
"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from utils import i18n, ui_theme
from tool_clonevoice_v2 import logic, pace, proofread


def get_text(key: str) -> str:
    return i18n.translate("clonevoice", key)


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")
GENERATED_MP4_SUFFIXES = ("_si.mp4", "_dub.mp4")

# Coverage below which a line is called out. Above it the uncovered remainder
# is a fraction of a second, which nobody hears as the original still talking.
WARN_COVERAGE = 0.70
BAD_COVERAGE = 0.50

# Seconds of context drawn around a line. The lead-in shows whether the line
# really starts where the transcript says; the tail is where the uncovered
# original lives, so it is the wider of the two.
WINDOW_LEAD = 0.6
WINDOW_TAIL = 2.5
WAVE_HEIGHT = 150
PLAYHEAD_INTERVAL_MS = 60


def _is_generated_output(name: str) -> bool:
    return name.lower().endswith(GENERATED_MP4_SUFFIXES)


def _fmt_seconds(value: float) -> str:
    minutes, seconds = divmod(max(0.0, float(value)), 60.0)
    return f"{int(minutes):d}:{seconds:05.2f}"


def _fmt_total(value: float) -> str:
    if value >= 60.0:
        return f"{value / 60.0:.1f}min"
    return f"{value:.0f}s"


class PaceTab:
    """The tab body. Owns its own thread for analysis and its own playback."""

    def __init__(self, frame: ttk.Frame):
        self.frame = frame
        self.videos: list[str] = []
        self.rows: list[dict] = []
        self.current_video: str | None = None
        self._thread: threading.Thread | None = None
        self._tick_job = None
        self.players: dict = {}
        self.lanes: dict = {}
        self.window = None
        self.active: str | None = None
        self._build()

    # --- construction -----------------------------------------------------

    def _build(self) -> None:
        frame = self.frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=3)
        frame.rowconfigure(6, weight=2)

        picker = ttk.Frame(frame)
        picker.grid(row=0, column=0, sticky="ew")
        picker.columnconfigure(1, weight=1)
        ttk.Label(picker, text=get_text("lbl_pace_input")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.input_var = tk.StringVar()
        ttk.Entry(picker, textvariable=self.input_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(picker, text=get_text("btn_browse"), command=self._browse).grid(
            row=0, column=2, sticky="ew", padx=(6, 0)
        )
        self.btn_analyze = ttk.Button(picker, text=get_text("btn_pace_analyze"), command=self._analyze)
        self.btn_analyze.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        ttk.Label(picker, text=get_text("lbl_pace_video")).grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 0)
        )
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(picker, textvariable=self.video_var, state="readonly")
        self.video_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        self.video_combo.bind("<<ComboboxSelected>>", lambda _e: self._analyze_selected())

        ttk.Label(
            frame, text=get_text("pace_note"), foreground="dim gray",
            wraplength=760, justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 4))

        self.summary_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.summary_var, wraplength=760, justify="left").grid(
            row=2, column=0, sticky="ew", pady=(0, 6)
        )

        table = ttk.Frame(frame)
        table.grid(row=3, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ("id", "time", "slot", "clone", "coverage", "speaker", "src", "tgt")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse")
        widths = {
            "id": 50, "time": 90, "slot": 70, "clone": 70,
            "coverage": 70, "speaker": 90, "src": 260, "tgt": 260,
        }
        for column in columns:
            self.tree.heading(column, text=get_text(f"col_pace_{column}"))
            anchor = "w" if column in ("src", "tgt", "speaker") else "e"
            self.tree.column(column, width=widths[column], anchor=anchor,
                             stretch=column in ("src", "tgt"))
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        palette = ui_theme.get_palette()
        self.tree.tag_configure("bad", foreground=getattr(palette, "DANGER", "#c0392b"))
        self.tree.tag_configure("warn", foreground=getattr(palette, "WARNING", "#b8860b"))

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, sticky="ew", pady=(8, 4))
        self.btn_play_source = ttk.Button(
            buttons, text=get_text("btn_pace_play_source"), command=self._play_source
        )
        self.btn_play_source.pack(side="left", padx=(0, 4))
        self.btn_play_clone = ttk.Button(
            buttons, text=get_text("btn_pace_play_clone"), command=self._play_clone
        )
        self.btn_play_clone.pack(side="left", padx=(0, 4))
        # Disabled while nothing is playing. The launcher counts any enabled
        # button labelled "stop" as a task in flight and refuses to go back to
        # the menu, so an always-live stop button reports work that never ran.
        self.btn_stop = ttk.Button(
            buttons, text=get_text("btn_pace_stop"), command=self._stop, state="disabled"
        )
        self.btn_stop.pack(side="left")
        self.playing_var = tk.StringVar()
        ttk.Label(buttons, textvariable=self.playing_var, foreground="dim gray").pack(
            side="left", padx=(12, 0)
        )

        texts = ttk.Frame(frame)
        texts.grid(row=5, column=0, sticky="nsew")
        texts.columnconfigure(0, weight=1)
        texts.columnconfigure(1, weight=1)
        texts.rowconfigure(1, weight=1)
        ttk.Label(texts, text=get_text("lbl_pace_src_text")).grid(row=0, column=0, sticky="w")
        ttk.Label(texts, text=get_text("lbl_pace_tgt_text")).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.src_text = tk.Text(texts, height=4, wrap="word")
        self.src_text.grid(row=1, column=0, sticky="nsew")
        self.tgt_text = tk.Text(texts, height=4, wrap="word")
        self.tgt_text.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        for widget in (self.src_text, self.tgt_text):
            widget.configure(state="disabled")

        wave = ttk.Frame(frame)
        wave.grid(row=6, column=0, sticky="nsew", pady=(8, 0))
        wave.columnconfigure(0, weight=1)
        wave.rowconfigure(1, weight=1)
        ttk.Label(wave, text=get_text("lbl_pace_wave")).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(
            wave, height=WAVE_HEIGHT, highlightthickness=1,
            highlightbackground=self._palette_colour("CARD_BORDER", "#cccccc"),
            background=self._palette_colour("CARD_BG", "#ffffff"),
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._draw_wave())
        self.canvas.bind("<Button-1>", self._canvas_seek)

    # --- input ------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.input_var.set(path)

    def _scan(self, base: str) -> list[str]:
        target = Path(base)
        if target.is_file():
            return [str(target)]
        found: list[str] = []
        for root, _dirs, files in os.walk(target):
            for name in files:
                if name.lower().endswith(VIDEO_EXTENSIONS) and not _is_generated_output(name):
                    found.append(os.path.join(root, name))
        return sorted(found, key=lambda item: item.lower())

    def _analyze(self) -> None:
        base = self.input_var.get().strip()
        if not base or not Path(base).exists():
            messagebox.showerror("Error", get_text("err_no_dir"))
            return
        videos = self._scan(base)
        if not videos:
            messagebox.showinfo("Info", get_text("msg_pace_no_videos"))
            return
        self.videos = videos
        names = [Path(video).name for video in videos]
        self.video_combo.configure(values=names)
        self.video_var.set(names[0])
        self._analyze_selected()

    def _analyze_selected(self) -> None:
        name = self.video_var.get()
        video = next((v for v in self.videos if Path(v).name == name), None)
        if video is None or (self._thread is not None and self._thread.is_alive()):
            return
        self.current_video = video
        self.btn_analyze.configure(state="disabled")
        self.summary_var.set("...")

        def worker():
            try:
                result = pace.analyze_video(video)
            except Exception as exc:  # surfaced in the UI thread
                self.frame.after(0, lambda: self._failed(exc))
                return
            self.frame.after(0, lambda: self._loaded(result))

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _failed(self, exc: Exception) -> None:
        self.btn_analyze.configure(state="normal")
        self.summary_var.set("")
        self.rows = []
        self.tree.delete(*self.tree.get_children())
        self._load_wave(None)
        messagebox.showerror("Error", get_text("err_pace_failed").format(exc))

    # --- display ----------------------------------------------------------

    def _loaded(self, result: dict) -> None:
        self.btn_analyze.configure(state="normal")
        self.rows = result["rows"]
        self.summary_var.set(self._summary_line(result["summary"]))
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(self.rows):
            coverage = row["coverage"]
            tags = ()
            if coverage is not None:
                if coverage < BAD_COVERAGE:
                    tags = ("bad",)
                elif coverage < WARN_COVERAGE:
                    tags = ("warn",)
            self.tree.insert(
                "", "end", iid=str(index), tags=tags,
                values=(
                    row["id"],
                    _fmt_seconds(row["start"]),
                    f"{row['slot']:.2f}",
                    "-" if row["clone"] is None else f"{row['clone']:.2f}",
                    "-" if coverage is None else f"{coverage * 100:.0f}%",
                    row["speaker"],
                    row["src_text"],
                    row["tgt_text"],
                ),
            )
        if self.rows:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self._on_select()
        else:
            self._on_select()

    def _summary_line(self, summary: dict) -> str:
        ratio = summary.get("median_char_ratio")
        ratio_text = "-" if ratio is None else f"{ratio:.2f}"
        if summary.get("median_coverage") is None:
            return get_text("msg_pace_no_clone").format(
                summary["lines"], summary["translated"], ratio_text
            )
        return get_text("msg_pace_summary").format(
            summary["lines"],
            summary["translated"],
            summary["measured"],
            f"{summary['median_coverage'] * 100:.0f}%",
            summary["under_exposed"],
            summary["under_half"],
            _fmt_total(summary["exposed_total"]),
            ratio_text,
        )

    def _row(self) -> dict | None:
        selection = self.tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def _show_row(self) -> None:
        row = self._row()
        for widget, key in ((self.src_text, "src_text"), (self.tgt_text, "tgt_text")):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            if row is not None:
                widget.insert("1.0", row.get(key, ""))
            widget.configure(state="disabled")

    def _on_select(self) -> None:
        row = self._row()
        self._show_row()
        self._load_wave(row)

    # --- waveform ---------------------------------------------------------

    def _palette_colour(self, name: str, fallback: str) -> str:
        return str(getattr(ui_theme.get_palette(), name, fallback) or fallback)

    def _load_wave(self, row: dict | None) -> None:
        """Read both tracks over this line's window and prepare to draw them.

        Windowed rather than whole-track: an 8K part runs half an hour, and
        every question this tab answers is about one line's edges.
        """
        self._release_players()
        self.window = None
        self.lanes = {}
        if row is None or self.current_video is None:
            self._draw_wave()
            return
        start = float(row["start"])
        end = float(row["end"])
        window_start = max(0.0, start - WINDOW_LEAD)
        window_end = max(end, start + float(row.get("clone") or 0.0)) + WINDOW_TAIL
        sources = [("src", Path(logic.clone_dir(self.current_video)) / logic.AUDIO16K_NAME)]
        cloned = proofread.cloned_track_path(self.current_video)
        if cloned is not None:
            sources.append(("clone", Path(cloned)))
        for key, path in sources:
            if not Path(path).is_file():
                continue
            try:
                samples, sr, real_start = pace.read_window(path, window_start, window_end)
            except Exception:
                continue
            if samples.size == 0:
                continue
            self.lanes[key] = {
                "samples": samples,
                "sr": sr,
                "start": real_start,
                "end": real_start + samples.size / float(sr or 1),
            }
        self.window = (window_start, window_end, start, end)
        self._draw_wave()

    def _player_for(self, key: str):
        """Built on demand: opening a waveOut device per row selection would
        open one for every line scrolled past."""
        existing = self.players.get(key)
        if existing is not None:
            return existing
        lane = self.lanes.get(key)
        if lane is None:
            return None
        try:
            from tool_subtitle.audio_player import WinMMAudioPlayer

            player = WinMMAudioPlayer(lane["samples"], lane["sr"])
        except Exception:
            return None
        self.players[key] = player
        return player

    def _release_players(self) -> None:
        for player in self.players.values():
            try:
                player.close()
            except Exception:
                pass
        self.players = {}
        self.active = None
        if getattr(self, "btn_stop", None) is not None:
            self.btn_stop.configure(state="disabled")
            self.playing_var.set("")

    def _x_of(self, seconds: float, width: int) -> float:
        window_start, window_end = self.window[0], self.window[1]
        span = max(1e-6, window_end - window_start)
        return (seconds - window_start) / span * width

    def _draw_wave(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        if self.window is None or not self.lanes:
            canvas.create_text(
                width / 2, height / 2, text=get_text("msg_pace_wave_empty"),
                fill=self._palette_colour("MUTED_FG", "#888888"),
            )
            return
        _ws, _we, start, end = self.window
        row = self._row()

        # The slot the dub had to fill, then the part of it the dub never
        # covered -- which is exactly the stretch where the ducked original is
        # still talking on its own.
        canvas.create_rectangle(
            self._x_of(start, width), 0, self._x_of(end, width), height,
            fill=self._palette_colour("SIDEBAR_BG", "#eef1f4"), outline="",
        )
        clone_end = None
        if row is not None and row.get("clone"):
            clone_end = start + float(row["clone"])
            if clone_end < end:
                canvas.create_rectangle(
                    self._x_of(clone_end, width), 0, self._x_of(end, width), height,
                    fill="#f2d5d5", outline="",
                )

        lane_height = height / 2.0
        for key, index in (("src", 0), ("clone", 1)):
            top = index * lane_height
            middle = top + lane_height / 2.0
            canvas.create_line(0, middle, width, middle,
                               fill=self._palette_colour("CARD_BORDER", "#d5d5d5"))
            canvas.create_text(
                4, top + 9, anchor="w", text=get_text("lbl_pace_wave_" + key),
                fill=self._palette_colour("MUTED_FG", "#888888"), font=("Arial", 8),
            )
            lane = self.lanes.get(key)
            if lane is None:
                continue
            left = int(max(0, self._x_of(lane["start"], width)))
            right = int(min(width, self._x_of(lane["end"], width)))
            buckets = max(1, right - left)
            low, high = pace.peak_envelope(lane["samples"], buckets)
            gain = pace.display_gain(low, high)
            limit = lane_height / 2.0 - 8
            scale = limit * gain
            colour = "#2c7fb8" if key == "src" else "#d95f0e"
            for offset in range(min(buckets, int(low.size))):
                x = left + offset
                y_low = middle - max(-limit, min(limit, float(low[offset]) * scale))
                y_high = middle - max(-limit, min(limit, float(high[offset]) * scale))
                canvas.create_line(x, y_low, x, y_high, fill=colour)

        for seconds in (start, end):
            canvas.create_line(self._x_of(seconds, width), 0,
                               self._x_of(seconds, width), height,
                               fill="#2ca25f", width=1)
        if clone_end is not None:
            canvas.create_line(self._x_of(clone_end, width), 0,
                               self._x_of(clone_end, width), height,
                               fill="#c0392b", width=1, dash=(3, 2))
        self._draw_playhead()

    def _draw_playhead(self) -> None:
        self.canvas.delete("playhead")
        if self.window is None or self.active is None:
            return
        player = self.players.get(self.active)
        lane = self.lanes.get(self.active)
        if player is None or lane is None:
            return
        width = max(1, self.canvas.winfo_width())
        x = self._x_of(lane["start"] + player.position(), width)
        self.canvas.create_line(
            x, 0, x, max(1, self.canvas.winfo_height()),
            fill="#111111", width=1, tags="playhead",
        )

    def _canvas_seek(self, event) -> None:
        if self.window is None or self.active is None:
            return
        player = self.players.get(self.active)
        lane = self.lanes.get(self.active)
        if player is None or lane is None:
            return
        window_start, window_end = self.window[0], self.window[1]
        span = max(1e-6, window_end - window_start)
        seconds = window_start + (event.x / max(1, self.canvas.winfo_width())) * span
        player.seek(max(0.0, seconds - lane["start"]), continue_playing=player.is_playing)
        self._draw_playhead()

    # --- playback ---------------------------------------------------------

    def _tick(self) -> None:
        """One timer drives both the playhead and the stop button, so the
        button follows what is really playing instead of a guessed length."""
        self._tick_job = None
        if self.active is None:
            return
        player = self.players.get(self.active)
        if player is not None and player.is_playing:
            self._draw_playhead()
            self._schedule_tick()
            return
        self._idle()

    def _schedule_tick(self) -> None:
        if self._tick_job is None and self.frame.winfo_exists():
            self._tick_job = self.frame.after(PLAYHEAD_INTERVAL_MS, self._tick)

    def _idle(self) -> None:
        self.active = None
        self.btn_stop.configure(state="disabled")
        self.playing_var.set("")
        self.canvas.delete("playhead")

    def _stop(self) -> None:
        for player in self.players.values():
            try:
                player.stop()
            except Exception:
                pass
        self._idle()

    def _play(self, key: str) -> None:
        if self._row() is None:
            return
        if key not in self.lanes:
            messagebox.showinfo("Info", get_text("msg_pace_no_clone_yet"))
            return
        self._stop()
        player = self._player_for(key)
        if player is None:
            messagebox.showerror("Error", get_text("err_pace_play_failed").format("waveOut"))
            return
        try:
            player.play(0.0)
        except Exception as exc:
            messagebox.showerror("Error", get_text("err_pace_play_failed").format(exc))
            return
        self.active = key
        self.btn_stop.configure(state="normal")
        self.playing_var.set(get_text("lbl_pace_wave_" + key))
        self._schedule_tick()

    def _play_source(self) -> None:
        self._play("src")

    def _play_clone(self) -> None:
        self._play("clone")

    def destroy(self) -> None:
        self._release_players()
