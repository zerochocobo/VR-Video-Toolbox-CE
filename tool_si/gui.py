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
        self.single_thread: threading.Thread | None = None
        self.batch_thread: threading.Thread | None = None
        self._download_button_packed = False

        self._setup_ui()
        self._check_dependencies()
        self._refresh_model_status()

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(header_frame, text=get_text("title"), font=("Arial", 14, "bold")).pack(side="left")
        if self.on_return:
            ttk.Button(header_frame, text=get_text("btn_return"), command=self.on_return).pack(side="right")

        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 8], font=("Arial", 10, "bold"))

        model_frame = ttk.LabelFrame(main_frame, text=get_text("grp_model"), padding=10)
        model_frame.pack(fill="x", pady=(0, 8))
        self.model_status_var = tk.StringVar()
        ttk.Label(model_frame, textvariable=self.model_status_var).pack(side="left", fill="x", expand=True)
        self.btn_download_model = ttk.Button(
            model_frame,
            text=get_text("btn_download_model"),
            command=self.download_model,
        )
        self.btn_download_model.pack(side="right", padx=(8, 0))
        self._download_button_packed = True

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="x", pady=(0, 8))

        single_tab = ttk.Frame(notebook, padding=10)
        notebook.add(single_tab, text=get_text("tab_single"))
        self._setup_single_frame(single_tab)

        batch_tab = ttk.Frame(notebook, padding=10)
        notebook.add(batch_tab, text=get_text("tab_batch"))
        self._setup_batch_frame(batch_tab)

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
        self.single_language_combo, self.single_speaker_combo = self._create_language_speaker_controls(
            options_frame,
            self.single_language_var,
            self.single_speaker_var,
            self._on_single_language_change,
        )

        test_lines_frame = ttk.Frame(parent)
        test_lines_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        ttk.Label(test_lines_frame, text=get_text("lbl_test_lines")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.single_test_lines_var = tk.StringVar(value=get_text("opt_test_lines_10"))
        self.single_test_lines_combo = ttk.Combobox(
            test_lines_frame,
            textvariable=self.single_test_lines_var,
            values=list(self._test_line_display_map().keys()),
            width=12,
            state="readonly",
        )
        self.single_test_lines_combo.grid(row=0, column=1, sticky="w")

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

        self.batch_mode_var = tk.StringVar(value="paired_video_srt")
        ttk.Radiobutton(
            parent,
            text=get_text("opt_paired_video_srt"),
            variable=self.batch_mode_var,
            value="paired_video_srt",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 2))

        options_frame = ttk.Frame(parent)
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.batch_language_var = tk.StringVar()
        self.batch_speaker_var = tk.StringVar()
        self.batch_language_combo, self.batch_speaker_combo = self._create_language_speaker_controls(
            options_frame,
            self.batch_language_var,
            self.batch_speaker_var,
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

    def _create_language_speaker_controls(self, parent, language_var, speaker_var, language_callback):
        parent.columnconfigure(1, weight=0)
        parent.columnconfigure(3, weight=0)
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
        speaker_combo.grid(row=0, column=3, sticky="w")

        default_language = logic.default_tts_language(app_config.get_language())
        language_var.set(self._language_to_display(default_language))
        self._refresh_speaker_combo(language_var, speaker_var, speaker_combo)
        return language_combo, speaker_combo

    def _selected_language(self, language_var) -> str:
        return self._language_display_map().get(language_var.get(), logic.default_tts_language(app_config.get_language()))

    def _test_line_display_map(self) -> dict[str, int | None]:
        return {
            get_text("opt_test_lines_10"): 10,
            get_text("opt_test_lines_20"): 20,
            get_text("opt_test_lines_40"): 40,
            get_text("opt_test_lines_all"): None,
        }

    def _selected_single_test_line_limit(self) -> int | None:
        return self._test_line_display_map().get(self.single_test_lines_var.get(), 10)

    def _refresh_speaker_combo(self, language_var, speaker_var, speaker_combo):
        language = self._selected_language(language_var)
        speakers = logic.speakers_for_language(language)
        speaker_combo.config(values=list(speakers))
        if speaker_var.get() not in speakers:
            speaker_var.set(logic.default_speaker_for_language(language))

    def _on_single_language_change(self, _event=None):
        self._refresh_speaker_combo(self.single_language_var, self.single_speaker_var, self.single_speaker_combo)

    def _on_batch_language_change(self, _event=None):
        self._refresh_speaker_combo(self.batch_language_var, self.batch_speaker_var, self.batch_speaker_combo)

    def log(self, message: str):
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
        ):
            if importlib.util.find_spec(module_name) is None:
                missing.append(label)
        if missing:
            self.log(get_text("warn_missing_deps").format(", ".join(missing)))

    def _refresh_model_status(self):
        model_dir = logic.get_model_dir(self.models_root)
        if logic.check_model_files(self.models_root):
            self.model_status_var.set(get_text("model_ready").format(model_dir))
            if self._download_button_packed:
                self.btn_download_model.pack_forget()
                self._download_button_packed = False
        else:
            self.model_status_var.set(get_text("model_missing").format(model_dir))
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

    def run_single(self):
        srt_path = self.single_srt_var.get().strip()
        if not srt_path or not os.path.isfile(srt_path):
            messagebox.showerror("Error", get_text("err_srt_file"))
            return
        output_path = self.single_output_var.get().strip() or logic.default_output_path(srt_path)
        language = self._selected_language(self.single_language_var)
        speaker = self.single_speaker_var.get()
        max_entries = self._selected_single_test_line_limit()

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
                    max_entries=max_entries,
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
        )

    def stop_running_tasks(self) -> bool:
        stopped = False
        if self.single_thread and self.single_thread.is_alive():
            self.stop_single()
            stopped = True
        if self.batch_thread and self.batch_thread.is_alive():
            self.stop_batch()
            stopped = True
        return stopped
