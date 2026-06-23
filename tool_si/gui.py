from __future__ import annotations

import os
import importlib.util
import shutil
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from tool_si import logic
from utils import app_config, i18n


def get_text(key: str) -> str:
    return i18n.translate("si", key)


_SUPPRESSED_LOG_MESSAGES = {
    "FlashAttention2 is not installed; using PyTorch SDPA attention",
    "FlashAttention2 is not installed; using PyTorch SDPA attention.",
}


def _should_show_log_message(message: object) -> bool:
    return str(message).strip() not in _SUPPRESSED_LOG_MESSAGES


class SimultaneousInterpretationApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text("title"))

        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_root = os.path.join(base_dir, "models")

        self.stop_single_event = threading.Event()
        self.stop_batch_event = threading.Event()
        self.stop_mix_event = threading.Event()
        self.stop_batch_mix_event = threading.Event()
        self.single_thread: threading.Thread | None = None
        self.batch_thread: threading.Thread | None = None
        self.mix_thread: threading.Thread | None = None
        self.batch_mix_thread: threading.Thread | None = None
        self.mix_process = None
        self.batch_mix_process = None
        self._download_button_packed = False
        self._model_frame_packed = False

        self._setup_ui()
        self._check_dependencies()
        self._refresh_model_status()

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 2))
        ttk.Label(header_frame, text=get_text("title"), font=("Arial", 14, "bold")).pack(side="left")
        if self.on_return:
            ttk.Button(header_frame, text=get_text("btn_return"), command=self.on_return).pack(side="right")
        ttk.Label(
            main_frame,
            text=get_text("lbl_dlna_si_note"),
            font=("Arial", 9),
            foreground="dim gray",
            wraplength=760,
            justify="left",
        ).pack(fill="x", pady=(0, 10))

        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 8], font=("Arial", 10, "bold"))

        self.model_frame = ttk.LabelFrame(main_frame, text=get_text("grp_model"), padding=10)
        self.model_status_var = tk.StringVar()
        ttk.Label(self.model_frame, textvariable=self.model_status_var).pack(side="left", fill="x", expand=True)
        self.btn_download_model = ttk.Button(
            self.model_frame,
            text=get_text("btn_download_model"),
            command=self.download_model,
        )
        self.btn_download_model.pack(side="right", padx=(8, 0))
        self._download_button_packed = True

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="x", pady=(0, 8))

        single_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(single_tab, text=get_text("tab_single"))
        self._setup_single_frame(single_tab)

        batch_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(batch_tab, text=get_text("tab_batch"))
        self._setup_batch_frame(batch_tab)

        mix_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(mix_tab, text=get_text("tab_mix_video_audio"))
        self._setup_mix_frame(mix_tab)

        batch_mix_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(batch_mix_tab, text=get_text("tab_batch_mix_video_audio"))
        self._setup_batch_mix_frame(batch_mix_tab)

        log_frame = ttk.LabelFrame(main_frame, text=get_text("lbl_log"), padding=10)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=12, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _setup_single_frame(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text=get_text("lbl_srt_file")).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.single_srt_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.single_srt_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text=get_text("btn_browse"), command=self.browse_single_srt).grid(
            row=0, column=2, sticky="ew", padx=(6, 0), pady=3
        )

        ttk.Label(parent, text=get_text("lbl_output_file")).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.single_output_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.single_output_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text=get_text("btn_browse"), command=self.browse_single_output).grid(
            row=1, column=2, sticky="ew", padx=(6, 0), pady=3
        )

        options_frame = ttk.Frame(parent)
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.single_language_var = tk.StringVar()
        self.single_speaker_var = tk.StringVar()
        self.single_speaker_note_var = tk.StringVar()
        self.single_language_combo, self.single_speaker_combo = self._create_language_speaker_controls(
            options_frame,
            self.single_language_var,
            self.single_speaker_var,
            self.single_speaker_note_var,
            self._on_single_language_change,
        )

        time_frame = ttk.Frame(parent)
        time_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        ttk.Label(time_frame, text=get_text("lbl_process_time")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(time_frame, text=get_text("lbl_start_time")).grid(row=0, column=1, sticky="w", padx=(0, 4))
        self.single_start_time_var = tk.StringVar(value="00:00:00")
        ttk.Entry(time_frame, textvariable=self.single_start_time_var, width=10).grid(
            row=0, column=2, sticky="w", padx=(0, 12)
        )
        self.single_duration_var = tk.StringVar(value=get_text("opt_duration_30s"))
        self.single_duration_combo = ttk.Combobox(
            time_frame,
            textvariable=self.single_duration_var,
            values=list(self._duration_display_map().keys()),
            width=16,
            state="readonly",
        )
        self.single_duration_combo.grid(row=0, column=3, sticky="w")
        self.single_duration_combo.bind("<<ComboboxSelected>>", self._update_single_time_controls)
        self.single_custom_minutes_label = ttk.Label(time_frame, text=get_text("lbl_custom_minutes"))
        self.single_custom_minutes_var = tk.StringVar(value="5")
        self.single_custom_minutes_entry = ttk.Entry(time_frame, textvariable=self.single_custom_minutes_var, width=6)
        self.single_end_time_label = ttk.Label(time_frame, text=get_text("lbl_end_time"))
        self.single_end_time_var = tk.StringVar(value="00:00:30")
        self.single_end_time_entry = ttk.Entry(time_frame, textvariable=self.single_end_time_var, width=10)
        self._update_single_time_controls()

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        self.btn_single_start = ttk.Button(button_frame, text=get_text("btn_start_single"), command=self.run_single)
        self.btn_single_start.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.btn_single_stop = ttk.Button(
            button_frame,
            text=get_text("btn_stop"),
            command=self.stop_single,
            state="disabled",
        )
        self.btn_single_stop.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _setup_batch_frame(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text=get_text("lbl_input_dir")).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.batch_dir_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.batch_dir_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text=get_text("btn_browse"), command=self.browse_batch_dir).grid(
            row=0, column=2, sticky="ew", padx=(6, 0), pady=3
        )

        self.batch_recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text=get_text("opt_paired_video_srt"),
            variable=self.batch_recursive_var,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 2))

        options_frame = ttk.Frame(parent)
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.batch_language_var = tk.StringVar()
        self.batch_speaker_var = tk.StringVar()
        self.batch_speaker_note_var = tk.StringVar()
        self.batch_language_combo, self.batch_speaker_combo = self._create_language_speaker_controls(
            options_frame,
            self.batch_language_var,
            self.batch_speaker_var,
            self.batch_speaker_note_var,
            self._on_batch_language_change,
        )

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        self.btn_batch_start = ttk.Button(button_frame, text=get_text("btn_start_batch"), command=self.run_batch)
        self.btn_batch_start.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.btn_batch_stop = ttk.Button(
            button_frame,
            text=get_text("btn_stop"),
            command=self.stop_batch,
            state="disabled",
        )
        self.btn_batch_stop.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _setup_mix_frame(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text=get_text("lbl_video_file")).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.mix_video_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.mix_video_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text=get_text("btn_browse"), command=self.browse_mix_video).grid(
            row=0, column=2, sticky="ew", padx=(6, 0), pady=3
        )

        ttk.Label(parent, text=get_text("lbl_si_wav_file")).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.mix_si_audio_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.mix_si_audio_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text=get_text("btn_browse"), command=self.browse_mix_si_audio).grid(
            row=1, column=2, sticky="ew", padx=(6, 0), pady=3
        )

        ttk.Label(parent, text=get_text("lbl_mix_output_file")).grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
        self.mix_output_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.mix_output_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text=get_text("btn_browse"), command=self.browse_mix_output).grid(
            row=2, column=2, sticky="ew", padx=(6, 0), pady=3
        )

        options_frame = ttk.Frame(parent)
        options_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.mix_add_independent_track_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text=get_text("chk_add_independent_track"),
            variable=self.mix_add_independent_track_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 18), pady=2)
        self.mix_duck_original_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text=get_text("chk_duck_original_when_si"),
            variable=self.mix_duck_original_var,
        ).grid(row=0, column=2, columnspan=2, sticky="w", padx=(0, 18), pady=2)

        ttk.Label(options_frame, text=get_text("lbl_mix_channel")).grid(row=0, column=4, sticky="w", padx=(0, 6), pady=2)
        self.mix_channel_var = tk.StringVar(value=get_text("opt_channel_left"))
        self.mix_channel_combo = ttk.Combobox(
            options_frame,
            textvariable=self.mix_channel_var,
            values=list(self._mix_channel_display_map().keys()),
            width=16,
            state="readonly",
        )
        self.mix_channel_combo.grid(row=0, column=5, sticky="w", pady=2)

        ttk.Label(options_frame, text=get_text("lbl_original_volume")).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        self.mix_original_volume_var = tk.StringVar(value="100%")
        ttk.Combobox(
            options_frame,
            textvariable=self.mix_original_volume_var,
            values=list(self._original_volume_display_map().keys()),
            width=8,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=(0, 18), pady=2)

        ttk.Label(options_frame, text=get_text("lbl_si_volume")).grid(row=1, column=2, sticky="w", padx=(0, 6), pady=2)
        self.mix_si_volume_var = tk.StringVar(value="50%")
        ttk.Combobox(
            options_frame,
            textvariable=self.mix_si_volume_var,
            values=list(self._si_volume_display_map().keys()),
            width=8,
            state="readonly",
        ).grid(row=1, column=3, sticky="w", padx=(0, 18), pady=2)

        ttk.Label(options_frame, text=get_text("lbl_si_delay")).grid(row=1, column=4, sticky="w", padx=(0, 6), pady=2)
        self.mix_si_delay_var = tk.StringVar(value="1s")
        ttk.Combobox(
            options_frame,
            textvariable=self.mix_si_delay_var,
            values=list(self._si_delay_display_map().keys()),
            width=8,
            state="readonly",
        ).grid(row=1, column=5, sticky="w", pady=2)

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        self.btn_mix_start = ttk.Button(button_frame, text=get_text("btn_start_mix"), command=self.run_mix)
        self.btn_mix_start.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.btn_mix_stop = ttk.Button(
            button_frame,
            text=get_text("btn_stop"),
            command=self.stop_mix,
            state="disabled",
        )
        self.btn_mix_stop.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _setup_batch_mix_frame(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text=get_text("lbl_input_dir")).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.batch_mix_dir_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.batch_mix_dir_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text=get_text("btn_browse"), command=self.browse_batch_mix_dir).grid(
            row=0, column=2, sticky="ew", padx=(6, 0), pady=3
        )

        self.batch_mix_recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text=get_text("opt_paired_video_si_wav"),
            variable=self.batch_mix_recursive_var,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 2))

        options_frame = ttk.Frame(parent)
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.batch_mix_add_independent_track_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text=get_text("chk_add_independent_track"),
            variable=self.batch_mix_add_independent_track_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 18), pady=2)
        self.batch_mix_duck_original_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text=get_text("chk_duck_original_when_si"),
            variable=self.batch_mix_duck_original_var,
        ).grid(row=0, column=2, columnspan=2, sticky="w", padx=(0, 18), pady=2)

        ttk.Label(options_frame, text=get_text("lbl_mix_channel")).grid(row=0, column=4, sticky="w", padx=(0, 6), pady=2)
        self.batch_mix_channel_var = tk.StringVar(value=get_text("opt_channel_left"))
        ttk.Combobox(
            options_frame,
            textvariable=self.batch_mix_channel_var,
            values=list(self._mix_channel_display_map().keys()),
            width=16,
            state="readonly",
        ).grid(row=0, column=5, sticky="w", pady=2)

        ttk.Label(options_frame, text=get_text("lbl_original_volume")).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        self.batch_mix_original_volume_var = tk.StringVar(value="100%")
        ttk.Combobox(
            options_frame,
            textvariable=self.batch_mix_original_volume_var,
            values=list(self._original_volume_display_map().keys()),
            width=8,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=(0, 18), pady=2)

        ttk.Label(options_frame, text=get_text("lbl_si_volume")).grid(row=1, column=2, sticky="w", padx=(0, 6), pady=2)
        self.batch_mix_si_volume_var = tk.StringVar(value="50%")
        ttk.Combobox(
            options_frame,
            textvariable=self.batch_mix_si_volume_var,
            values=list(self._si_volume_display_map().keys()),
            width=8,
            state="readonly",
        ).grid(row=1, column=3, sticky="w", padx=(0, 18), pady=2)

        ttk.Label(options_frame, text=get_text("lbl_si_delay")).grid(row=1, column=4, sticky="w", padx=(0, 6), pady=2)
        self.batch_mix_si_delay_var = tk.StringVar(value="1s")
        ttk.Combobox(
            options_frame,
            textvariable=self.batch_mix_si_delay_var,
            values=list(self._si_delay_display_map().keys()),
            width=8,
            state="readonly",
        ).grid(row=1, column=5, sticky="w", pady=2)

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        self.btn_batch_mix_start = ttk.Button(
            button_frame,
            text=get_text("btn_start_batch_mix"),
            command=self.run_batch_mix,
        )
        self.btn_batch_mix_start.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.btn_batch_mix_stop = ttk.Button(
            button_frame,
            text=get_text("btn_stop"),
            command=self.stop_batch_mix,
            state="disabled",
        )
        self.btn_batch_mix_stop.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _language_display_map(self) -> dict[str, str]:
        return {
            get_text(f"lang_{logic.LANGUAGE_CODES[language]}"): language
            for language in logic.SUPPORTED_LANGUAGES
        }

    def _language_to_display(self, language: str) -> str:
        for display, value in self._language_display_map().items():
            if value == language:
                return display
        return language

    def _create_language_speaker_controls(self, parent, language_var, speaker_var, speaker_note_var, language_callback):
        parent.columnconfigure(1, weight=0)
        parent.columnconfigure(3, weight=0)
        parent.columnconfigure(4, weight=1)
        ttk.Label(parent, text=get_text("lbl_language")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        language_combo = ttk.Combobox(
            parent,
            textvariable=language_var,
            values=list(self._language_display_map().keys()),
            width=18,
            state="readonly",
        )
        language_combo.grid(row=0, column=1, sticky="w", padx=(0, 18))
        language_combo.bind("<<ComboboxSelected>>", language_callback)

        ttk.Label(parent, text=get_text("lbl_speaker")).grid(row=0, column=2, sticky="w", padx=(0, 6))
        speaker_combo = ttk.Combobox(parent, textvariable=speaker_var, values=list(logic.ALL_SPEAKERS), width=18, state="readonly")
        speaker_combo.grid(row=0, column=3, sticky="w", padx=(0, 10))
        speaker_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_speaker_note(speaker_var, speaker_note_var),
        )
        ttk.Label(
            parent,
            textvariable=speaker_note_var,
            foreground="#555555",
            wraplength=520,
            justify="left",
        ).grid(row=0, column=4, sticky="w")

        default_language = logic.default_tts_language(app_config.get_language())
        language_var.set(self._language_to_display(default_language))
        self._refresh_speaker_combo(language_var, speaker_var, speaker_combo, speaker_note_var)
        return language_combo, speaker_combo

    def _selected_language(self, language_var) -> str:
        return self._language_display_map().get(language_var.get(), logic.default_tts_language(app_config.get_language()))

    def _duration_display_map(self) -> dict[str, float | str | None]:
        return {
            get_text("opt_duration_15s"): 15.0,
            get_text("opt_duration_30s"): 30.0,
            get_text("opt_duration_60s"): 60.0,
            get_text("opt_duration_custom_minutes"): "custom_minutes",
            get_text("opt_duration_until_time"): "until_time",
            get_text("opt_duration_all"): None,
        }

    def _mix_channel_display_map(self) -> dict[str, str]:
        return {
            get_text("opt_channel_left"): "left",
            get_text("opt_channel_right"): "right",
        }

    def _original_volume_display_map(self) -> dict[str, int]:
        return {f"{value}%": value for value in logic.ORIGINAL_VOLUME_CHOICES}

    def _si_volume_display_map(self) -> dict[str, int]:
        return {f"{value}%": value for value in logic.SI_VOLUME_CHOICES}

    def _si_delay_display_map(self) -> dict[str, float]:
        return {f"{value:g}s": value for value in logic.SI_DELAY_SECONDS_CHOICES}

    def _update_single_time_controls(self, _event=None):
        selected = self._duration_display_map().get(self.single_duration_var.get(), 30.0)
        if selected == "custom_minutes":
            self.single_custom_minutes_label.grid(row=0, column=4, sticky="w", padx=(10, 4))
            self.single_custom_minutes_entry.grid(row=0, column=5, sticky="w")
        else:
            self.single_custom_minutes_label.grid_remove()
            self.single_custom_minutes_entry.grid_remove()

        if selected == "until_time":
            self.single_end_time_label.grid(row=0, column=4, sticky="w", padx=(10, 4))
            self.single_end_time_entry.grid(row=0, column=5, sticky="w")
        else:
            self.single_end_time_label.grid_remove()
            self.single_end_time_entry.grid_remove()

    def _parse_time_seconds(self, value: str) -> float:
        raw = (value or "").strip().replace(",", ".")
        if not raw:
            raise ValueError("empty time")
        parts = raw.split(":")
        if len(parts) > 3:
            raise ValueError(f"invalid time: {value}")
        try:
            numbers = [float(part.strip()) for part in parts]
        except ValueError as exc:
            raise ValueError(f"invalid time: {value}") from exc
        if any(number < 0 for number in numbers):
            raise ValueError(f"invalid time: {value}")
        if len(numbers) == 1:
            return numbers[0]
        if len(numbers) == 2:
            minutes, seconds = numbers
            return minutes * 60.0 + seconds
        hours, minutes, seconds = numbers
        return hours * 3600.0 + minutes * 60.0 + seconds

    def _selected_single_time_window(self) -> tuple[float, float | None]:
        start_seconds = self._parse_time_seconds(self.single_start_time_var.get())
        selected = self._duration_display_map().get(self.single_duration_var.get(), 30.0)
        if selected == "custom_minutes":
            try:
                minutes = float((self.single_custom_minutes_var.get() or "").strip().replace(",", "."))
            except ValueError as exc:
                raise ValueError("invalid custom minutes") from exc
            if minutes <= 0:
                raise ValueError("invalid custom minutes")
            return start_seconds, minutes * 60.0
        if selected == "until_time":
            end_seconds = self._parse_time_seconds(self.single_end_time_var.get())
            if end_seconds <= start_seconds:
                raise ValueError("end time must be greater than start time")
            return start_seconds, end_seconds - start_seconds
        if selected is None:
            return start_seconds, None
        return start_seconds, float(selected)

    def _selected_mix_channel(self) -> str:
        return self._mix_channel_display_map().get(self.mix_channel_var.get(), "left")

    def _selected_original_volume(self) -> int:
        return self._original_volume_display_map().get(
            self.mix_original_volume_var.get(), logic.DEFAULT_ORIGINAL_VOLUME_PERCENT
        )

    def _selected_si_volume(self) -> int:
        return self._si_volume_display_map().get(self.mix_si_volume_var.get(), logic.DEFAULT_SI_VOLUME_PERCENT)

    def _selected_si_delay(self) -> float:
        return self._si_delay_display_map().get(self.mix_si_delay_var.get(), logic.DEFAULT_SI_DELAY_SECONDS)

    def _selected_batch_mix_channel(self) -> str:
        return self._mix_channel_display_map().get(self.batch_mix_channel_var.get(), "left")

    def _selected_batch_mix_original_volume(self) -> int:
        return self._original_volume_display_map().get(
            self.batch_mix_original_volume_var.get(), logic.DEFAULT_ORIGINAL_VOLUME_PERCENT
        )

    def _selected_batch_mix_si_volume(self) -> int:
        return self._si_volume_display_map().get(
            self.batch_mix_si_volume_var.get(), logic.DEFAULT_SI_VOLUME_PERCENT
        )

    def _selected_batch_mix_si_delay(self) -> float:
        return self._si_delay_display_map().get(
            self.batch_mix_si_delay_var.get(), logic.DEFAULT_SI_DELAY_SECONDS
        )

    def _speaker_note_text(self, speaker: str) -> str:
        key = logic.speaker_note_key(speaker)
        if not key:
            return ""
        text = get_text(key)
        return "" if text == key else text

    def _update_speaker_note(self, speaker_var, speaker_note_var) -> None:
        speaker_note_var.set(self._speaker_note_text(speaker_var.get()))

    def _refresh_speaker_combo(self, language_var, speaker_var, speaker_combo, speaker_note_var=None):
        language = self._selected_language(language_var)
        speakers = logic.speakers_for_language(language)
        speaker_combo.config(values=list(speakers))
        if speaker_var.get() not in speakers:
            speaker_var.set(logic.default_speaker_for_language(language))
        if speaker_note_var is not None:
            self._update_speaker_note(speaker_var, speaker_note_var)

    def _on_single_language_change(self, _event=None):
        self._refresh_speaker_combo(
            self.single_language_var,
            self.single_speaker_var,
            self.single_speaker_combo,
            self.single_speaker_note_var,
        )

    def _on_batch_language_change(self, _event=None):
        self._refresh_speaker_combo(
            self.batch_language_var,
            self.batch_speaker_var,
            self.batch_speaker_combo,
            self.batch_speaker_note_var,
        )

    def log(self, message: str):
        if not _should_show_log_message(message):
            return

        def _log():
            self.log_text.config(state="normal")
            self.log_text.insert("end", str(message) + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")

        self.root.after(0, _log)

    def _check_dependencies(self):
        missing = []
        if not shutil.which("ffmpeg"):
            missing.append("ffmpeg")
        for module_name, label in (
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
            ("huggingface_hub", "huggingface_hub"),
        ):
            if importlib.util.find_spec(module_name) is None:
                missing.append(label)
        if missing:
            self.log(get_text("warn_missing_deps").format(", ".join(missing)))

    def _refresh_model_status(self):
        model_dir = logic.get_model_dir(self.models_root)
        if logic.check_model_files(self.models_root):
            self.model_status_var.set("")
            if self._model_frame_packed:
                self.model_frame.pack_forget()
                self._model_frame_packed = False
            if self._download_button_packed:
                self.btn_download_model.pack_forget()
                self._download_button_packed = False
        else:
            self.model_status_var.set(get_text("model_missing").format(model_dir))
            if not self._model_frame_packed:
                self.model_frame.pack(fill="x", pady=(0, 8), before=self.notebook)
                self._model_frame_packed = True
            if not self._download_button_packed:
                self.btn_download_model.pack(side="right", padx=(8, 0))
                self._download_button_packed = True
            self.btn_download_model.config(state="normal")

    def download_model(self):
        def task():
            self.root.after(0, lambda: self.btn_download_model.config(state="disabled"))
            ok = logic.download_model(self.models_root, self.log)
            def finish():
                self._refresh_model_status()
                if not ok:
                    self.btn_download_model.config(state="normal")
                    self.log(get_text("msg_download_failed"))
            self.root.after(0, finish)

        threading.Thread(target=task, daemon=True).start()

    def browse_single_srt(self):
        path = filedialog.askopenfilename(filetypes=[("SRT", "*.srt"), ("All Files", "*.*")])
        if path:
            self.single_srt_var.set(path)
            if not self.single_output_var.get().strip():
                self.single_output_var.set(logic.default_output_path(path))

    def browse_single_output(self):
        initial = self.single_output_var.get().strip() or (
            logic.default_output_path(self.single_srt_var.get()) if self.single_srt_var.get() else ""
        )
        path = filedialog.asksaveasfilename(
            initialfile=os.path.basename(initial) if initial else "",
            initialdir=os.path.dirname(initial) if initial else "",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("All Files", "*.*")],
        )
        if path:
            self.single_output_var.set(path)

    def browse_batch_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.batch_dir_var.set(path)

    def browse_batch_mix_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.batch_mix_dir_var.set(path)

    def browse_mix_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("MP4", "*.mp4"), ("Video Files", "*.mp4 *.mkv *.mov"), ("All Files", "*.*")]
        )
        if path:
            self.mix_video_var.set(path)
            self.mix_si_audio_var.set(logic.default_si_audio_path(path))
            self.mix_output_var.set(logic.default_si_mix_output_path(path))

    def browse_mix_si_audio(self):
        initial = self.mix_si_audio_var.get().strip()
        path = filedialog.askopenfilename(
            initialfile=os.path.basename(initial) if initial else "",
            initialdir=os.path.dirname(initial) if initial else "",
            filetypes=[("WAV", "*.wav"), ("All Files", "*.*")],
        )
        if path:
            self.mix_si_audio_var.set(path)

    def browse_mix_output(self):
        initial = self.mix_output_var.get().strip() or (
            logic.default_si_mix_output_path(self.mix_video_var.get()) if self.mix_video_var.get() else ""
        )
        path = filedialog.asksaveasfilename(
            initialfile=os.path.basename(initial) if initial else "",
            initialdir=os.path.dirname(initial) if initial else "",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All Files", "*.*")],
        )
        if path:
            self.mix_output_var.set(path)

    def run_single(self):
        srt_path = self.single_srt_var.get().strip()
        if not srt_path or not os.path.isfile(srt_path):
            messagebox.showerror("Error", get_text("err_srt_file"))
            return
        output_path = self.single_output_var.get().strip() or logic.default_output_path(srt_path)
        language = self._selected_language(self.single_language_var)
        speaker = self.single_speaker_var.get()
        try:
            start_seconds, duration_seconds = self._selected_single_time_window()
        except ValueError:
            messagebox.showerror("Error", get_text("err_time_range"))
            return

        def task():
            start_time = time.time()
            self.log(get_text("msg_start_single"))
            self.stop_single_event.clear()
            try:
                output = logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language=language,
                    speaker=speaker,
                    models_root=self.models_root,
                    log_callback=self.log,
                    stop_event=self.stop_single_event,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                )
                if not self.stop_single_event.is_set():
                    self.log(get_text("msg_done").format(output))
            except Exception as exc:
                self.log(f"Error: {exc}")
            finally:
                self._log_elapsed(start_time)
                self.root.after(0, lambda: self.btn_single_start.config(state="normal"))
                self.root.after(0, lambda: self.btn_single_stop.config(state="disabled"))
                self.root.after(0, self._refresh_model_status)

        self.btn_single_start.config(state="disabled")
        self.btn_single_stop.config(state="normal")
        self.single_thread = threading.Thread(target=task, daemon=True)
        self.single_thread.start()

    def stop_single(self):
        self.stop_single_event.set()
        self.log(get_text("msg_stop_wait"))
        self.btn_single_stop.config(state="disabled")

    def run_batch(self):
        base_dir = self.batch_dir_var.get().strip()
        if not base_dir or not os.path.isdir(base_dir):
            messagebox.showerror("Error", get_text("err_dir"))
            return
        language = self._selected_language(self.batch_language_var)
        speaker = self.batch_speaker_var.get()

        def task():
            start_time = time.time()
            self.log(get_text("msg_start_batch"))
            self.stop_batch_event.clear()
            try:
                outputs = logic.batch_subtitle_to_audio(
                    base_dir=base_dir,
                    language=language,
                    speaker=speaker,
                    models_root=self.models_root,
                    log_callback=self.log,
                    stop_event=self.stop_batch_event,
                    recursive=self.batch_recursive_var.get(),
                )
                if not self.stop_batch_event.is_set():
                    self.log(get_text("msg_batch_done").format(len(outputs)))
            except Exception as exc:
                self.log(f"Error: {exc}")
            finally:
                self._log_elapsed(start_time)
                self.root.after(0, lambda: self.btn_batch_start.config(state="normal"))
                self.root.after(0, lambda: self.btn_batch_stop.config(state="disabled"))
                self.root.after(0, self._refresh_model_status)

        self.btn_batch_start.config(state="disabled")
        self.btn_batch_stop.config(state="normal")
        self.batch_thread = threading.Thread(target=task, daemon=True)
        self.batch_thread.start()

    def stop_batch(self):
        self.stop_batch_event.set()
        self.log(get_text("msg_stop_wait"))
        self.btn_batch_stop.config(state="disabled")

    def run_mix(self):
        video_path = self.mix_video_var.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showerror("Error", get_text("err_video_file"))
            return
        si_audio_path = self.mix_si_audio_var.get().strip() or logic.default_si_audio_path(video_path)
        if not si_audio_path or not os.path.isfile(si_audio_path):
            messagebox.showerror("Error", get_text("err_si_wav_file"))
            return
        output_path = self.mix_output_var.get().strip() or logic.default_si_mix_output_path(video_path)
        if os.path.abspath(video_path) == os.path.abspath(output_path):
            messagebox.showerror("Error", get_text("err_mix_output_file"))
            return
        if os.path.exists(output_path) and not messagebox.askyesno(
            get_text("msg_mix_overwrite_title"),
            get_text("msg_mix_overwrite").format(output_path),
        ):
            return
        mix_channel = self._selected_mix_channel()
        original_volume = self._selected_original_volume()
        si_volume = self._selected_si_volume()
        si_delay = self._selected_si_delay()
        add_independent_track = self.mix_add_independent_track_var.get()
        duck_original = self.mix_duck_original_var.get()

        def task():
            start_time = time.time()
            self.log(get_text("msg_start_mix"))
            self.stop_mix_event.clear()
            try:
                output = logic.mix_si_audio_track(
                    video_path=video_path,
                    si_audio_path=si_audio_path,
                    output_path=output_path,
                    mix_channel=mix_channel,
                    original_volume_percent=original_volume,
                    si_volume_percent=si_volume,
                    si_delay_seconds=si_delay,
                    add_independent_track=add_independent_track,
                    duck_original=duck_original,
                    log_callback=self.log,
                    stop_event=self.stop_mix_event,
                    process_callback=self._set_mix_process,
                )
                if not self.stop_mix_event.is_set():
                    self.log(get_text("msg_mix_done").format(output))
            except Exception as exc:
                self.log(f"Error: {exc}")
            finally:
                self._log_elapsed(start_time)
                self.root.after(0, lambda: self.btn_mix_start.config(state="normal"))
                self.root.after(0, lambda: self.btn_mix_stop.config(state="disabled"))

        self.btn_mix_start.config(state="disabled")
        self.btn_mix_stop.config(state="normal")
        self.mix_thread = threading.Thread(target=task, daemon=True)
        self.mix_thread.start()

    def _set_mix_process(self, process):
        self.mix_process = process

    def stop_mix(self):
        self.stop_mix_event.set()
        self.log(get_text("msg_mix_stop_wait"))
        process = self.mix_process
        if process is not None:
            logic._terminate_process(process)
        self.btn_mix_stop.config(state="disabled")

    def run_batch_mix(self):
        base_dir = self.batch_mix_dir_var.get().strip()
        if not base_dir or not os.path.isdir(base_dir):
            messagebox.showerror("Error", get_text("err_dir"))
            return
        mix_channel = self._selected_batch_mix_channel()
        original_volume = self._selected_batch_mix_original_volume()
        si_volume = self._selected_batch_mix_si_volume()
        si_delay = self._selected_batch_mix_si_delay()
        add_independent_track = self.batch_mix_add_independent_track_var.get()
        duck_original = self.batch_mix_duck_original_var.get()
        recursive = self.batch_mix_recursive_var.get()

        def task():
            start_time = time.time()
            self.log(get_text("msg_start_batch_mix"))
            self.stop_batch_mix_event.clear()
            try:
                outputs = logic.batch_mix_si_audio_tracks(
                    base_dir=base_dir,
                    mix_channel=mix_channel,
                    original_volume_percent=original_volume,
                    si_volume_percent=si_volume,
                    si_delay_seconds=si_delay,
                    add_independent_track=add_independent_track,
                    duck_original=duck_original,
                    log_callback=self.log,
                    stop_event=self.stop_batch_mix_event,
                    recursive=recursive,
                    process_callback=self._set_batch_mix_process,
                )
                if not self.stop_batch_mix_event.is_set():
                    self.log(get_text("msg_batch_mix_done").format(len(outputs)))
            except Exception as exc:
                self.log(f"Error: {exc}")
            finally:
                self._log_elapsed(start_time)
                self.root.after(0, lambda: self.btn_batch_mix_start.config(state="normal"))
                self.root.after(0, lambda: self.btn_batch_mix_stop.config(state="disabled"))

        self.btn_batch_mix_start.config(state="disabled")
        self.btn_batch_mix_stop.config(state="normal")
        self.batch_mix_thread = threading.Thread(target=task, daemon=True)
        self.batch_mix_thread.start()

    def _set_batch_mix_process(self, process):
        self.batch_mix_process = process

    def stop_batch_mix(self):
        self.stop_batch_mix_event.set()
        self.log(get_text("msg_mix_stop_wait"))
        process = self.batch_mix_process
        if process is not None:
            logic._terminate_process(process)
        self.btn_batch_mix_stop.config(state="disabled")

    def _log_elapsed(self, start_time: float):
        elapsed = time.time() - start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self.log(get_text("msg_elapsed").format(h, m, s))

    def has_running_tasks(self) -> bool:
        return bool(
            (self.single_thread and self.single_thread.is_alive())
            or (self.batch_thread and self.batch_thread.is_alive())
            or (self.mix_thread and self.mix_thread.is_alive())
            or (self.batch_mix_thread and self.batch_mix_thread.is_alive())
        )

    def stop_running_tasks(self) -> bool:
        stopped = False
        if self.single_thread and self.single_thread.is_alive():
            self.stop_single()
            stopped = True
        if self.batch_thread and self.batch_thread.is_alive():
            self.stop_batch()
            stopped = True
        if self.mix_thread and self.mix_thread.is_alive():
            self.stop_mix()
            stopped = True
        if self.batch_mix_thread and self.batch_mix_thread.is_alive():
            self.stop_batch_mix()
            stopped = True
        return stopped
