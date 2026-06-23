from __future__ import annotations

import gc
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from utils import i18n
from tool_clonevoice.log_redirect import redirect_stdio


def get_text(key: str) -> str:
    return i18n.translate("clonevoice", key)


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")
GENERATED_MP4_SUFFIXES = ("_si.mp4", "_dub.mp4")


def _is_generated_output_mp4(path: str) -> bool:
    return os.path.basename(path).lower().endswith(GENERATED_MP4_SUFFIXES)


class ClonevoiceToolsApp:
    """Voice clone translation tool (P0 skeleton).

    Two tabs: voice clone translation (main pipeline) and audio remix. The
    pipeline stages are not implemented yet; see the dev plan in summary/.
    """

    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text("title"))

        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_root = os.path.join(base_dir, "models")

        self.stop_event = threading.Event()
        self.run_thread: threading.Thread | None = None

        self._setup_ui()
        self._refresh_asr_model_status()
        self._refresh_voice_model_status()
        self._start_backend_warmup()

    def _start_backend_warmup(self):
        """Preload heavy backends in the background so the first run isn't cold.

        Keep the launcher-to-window transition responsive: the packaged build
        can spend a long time cold-importing torch/transformers. Start that
        best-effort warmup after Tk has had a chance to paint this window.
        """
        def _run():
            try:
                import torch  # noqa: F401  -- the expensive cold import
                import transformers  # noqa: F401
            except Exception:
                pass

        def _start():
            threading.Thread(target=_run, name="clonevoice-warmup", daemon=True).start()

        try:
            self.root.after(1500, _start)
        except tk.TclError:
            _start()

    # --- UI ---
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

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)

        self.tab_clone = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_clone, text=get_text("tab_clone"))
        self._setup_clone_tab(self.tab_clone)

        self.tab_mix = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_mix, text=get_text("tab_mix_video_audio"))
        self._setup_single_mix_tab(self.tab_mix)

    def _setup_clone_tab(self, frame):
        info = ttk.Label(frame, text=get_text("lbl_info"), wraplength=760, justify="left", foreground="dim gray")
        info.pack(fill="x", pady=(0, 8))

        input_mode_frame = ttk.Frame(frame)
        input_mode_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(input_mode_frame, text=get_text("lbl_input_mode")).pack(side="left", padx=(0, 8))
        self.clone_input_mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            input_mode_frame, text=get_text("opt_single_file"),
            variable=self.clone_input_mode_var, value="single",
            command=self._on_clone_input_mode_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            input_mode_frame, text=get_text("opt_batch_dir"),
            variable=self.clone_input_mode_var, value="batch",
            command=self._on_clone_input_mode_change,
        ).pack(side="left")

        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill="x", pady=(0, 6))
        self.clone_input_label_var = tk.StringVar(value=get_text("lbl_input_video"))
        ttk.Label(dir_frame, textvariable=self.clone_input_label_var).pack(side="left", padx=(0, 8))
        self.input_video_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.input_video_var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(dir_frame, text=get_text("btn_browse"), command=self._browse_video).pack(side="left")

        options_frame = ttk.Frame(frame)
        options_frame.pack(fill="x", pady=(0, 6))

        row1 = ttk.Frame(options_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text=get_text("lbl_src_lang"), width=12).pack(side="left")
        self._lang_map = {
            get_text("opt_lang_auto"): None,
            get_text("opt_lang_ja"): "ja",
            get_text("opt_lang_en"): "en",
            get_text("opt_lang_zh"): "zh",
        }
        self.src_lang_var = tk.StringVar(value=get_text("opt_lang_auto"))
        ttk.Combobox(row1, textvariable=self.src_lang_var, values=list(self._lang_map.keys()), state="readonly", width=14).pack(side="left", padx=(0, 16))
        ttk.Label(row1, text=get_text("lbl_model"), width=10).pack(side="left")
        self._model_map = {
            "large-v3": "large-v3",
            "large-v2": "large-v2",
            get_text("opt_model_kotoba"): "kotoba",
        }
        self.model_var = tk.StringVar(value="large-v3")
        self.model_combo = ttk.Combobox(row1, textvariable=self.model_var, values=list(self._model_map.keys()), state="readonly", width=18)
        self.model_combo.pack(side="left", padx=(0, 8))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_asr_model_status())

        self.asr_model_status_var = tk.StringVar()
        self.asr_download_button_packed = True
        self.btn_download_asr_model = ttk.Button(row1, text=get_text("btn_download_model"), command=self._download_asr_model)
        self.btn_download_asr_model.pack(side="left")

        self.asr_model_status_row = ttk.Frame(options_frame)
        self.asr_model_status_packed = True
        self.asr_model_status_row.pack(fill="x", pady=(0, 2))
        ttk.Label(self.asr_model_status_row, textvariable=self.asr_model_status_var, foreground="dim gray").pack(side="left")

        self.voice_model_frame = ttk.LabelFrame(options_frame, text=get_text("grp_voice_models"), padding=6)
        self.voice_model_frame_packed = True
        self.voice_model_frame.pack(fill="x", pady=(2, 4))
        self.ov_row = ttk.Frame(self.voice_model_frame)
        self.ov_row.pack(fill="x", pady=1)
        self.ov_row_packed = True
        self.omnivoice_status_var = tk.StringVar()
        ttk.Label(self.ov_row, textvariable=self.omnivoice_status_var, foreground="dim gray").pack(side="left", fill="x", expand=True)
        self.btn_download_omnivoice = ttk.Button(
            self.ov_row, text=get_text("btn_download_model"), command=lambda: self._download_voice_model("omnivoice")
        )
        self.btn_download_omnivoice.pack(side="right", padx=(8, 0))
        self.omnivoice_download_button_packed = True

        self.ecapa_row = ttk.Frame(self.voice_model_frame)
        self.ecapa_row.pack(fill="x", pady=1)
        self.ecapa_row_packed = True
        self.ecapa_status_var = tk.StringVar()
        ttk.Label(self.ecapa_row, textvariable=self.ecapa_status_var, foreground="dim gray").pack(side="left", fill="x", expand=True)
        self.btn_download_ecapa = ttk.Button(
            self.ecapa_row, text=get_text("btn_download_model"), command=lambda: self._download_voice_model("ecapa")
        )
        self.btn_download_ecapa.pack(side="right", padx=(8, 0))
        self.ecapa_download_button_packed = True

        row2 = ttk.Frame(options_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text=get_text("lbl_diarize"), width=12).pack(side="left")
        self._diar_map = {
            get_text("opt_diar_auto"): "auto",
            get_text("opt_diar_single"): "single",
            get_text("opt_diar_pyannote"): "pyannote",
        }
        self.diar_var = tk.StringVar(value=get_text("opt_diar_auto"))
        ttk.Combobox(row2, textvariable=self.diar_var, values=list(self._diar_map.keys()), state="readonly", width=16).pack(side="left", padx=(0, 16))
        ttk.Label(row2, text=get_text("lbl_num_speakers"), width=10).pack(side="left")
        self._num_map = {get_text("opt_num_auto"): None, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
        self.num_spk_var = tk.StringVar(value=get_text("opt_num_auto"))
        ttk.Combobox(row2, textvariable=self.num_spk_var, values=list(self._num_map.keys()), state="readonly", width=14).pack(side="left", padx=(0, 16))
        ttk.Label(row2, text=get_text("lbl_denoise"), width=10).pack(side="left")
        self._denoise_map = {
            get_text("opt_denoise_none"): "none",
            get_text("opt_denoise_mild"): "mild",
            get_text("opt_denoise_balanced"): "balanced",
            get_text("opt_denoise_strong"): "strong",
        }
        self.denoise_var = tk.StringVar(value=get_text("opt_denoise_none"))
        ttk.Combobox(row2, textvariable=self.denoise_var, values=list(self._denoise_map.keys()), state="readonly", width=12).pack(side="left")

        row3 = ttk.Frame(options_frame)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text=get_text("lbl_target_lang"), width=12).pack(side="left")
        self._tgt_map = {
            get_text("opt_lang_zh"): "Chinese",
            get_text("opt_lang_en"): "English",
            get_text("opt_lang_ko"): "Korean",
            get_text("opt_lang_th"): "Thai",
            get_text("opt_lang_de"): "German",
            get_text("opt_lang_fr"): "French",
            get_text("opt_lang_es"): "Spanish",
            get_text("opt_lang_pt"): "Portuguese",
            get_text("opt_lang_it"): "Italian",
            get_text("opt_lang_ru"): "Russian",
        }
        self.tgt_lang_var = tk.StringVar(value=get_text("opt_lang_zh"))
        self.tgt_lang_combo = ttk.Combobox(
            row3,
            textvariable=self.tgt_lang_var,
            values=list(self._tgt_map.keys()),
            state="readonly",
            width=16,
        )
        self.tgt_lang_combo.pack(side="left")
        self.btn_trans_config = ttk.Button(row3, text=get_text("btn_trans_config"), command=self._open_translate_config)
        self.btn_trans_config.pack(side="left", padx=(12, 0))

        row4 = ttk.Frame(options_frame)
        row4.pack(fill="x", pady=2)
        ttk.Label(row4, text=get_text("lbl_loudness_mode"), width=12).pack(side="left")
        self._loudness_mode_map = {
            get_text("opt_loudness_flat"): "flat",
            get_text("opt_loudness_sentence"): "sentence",
            get_text("opt_loudness_envelope"): "envelope",
        }
        self.loudness_mode_var = tk.StringVar(value=get_text("opt_loudness_envelope"))
        ttk.Combobox(
            row4, textvariable=self.loudness_mode_var, values=list(self._loudness_mode_map.keys()),
            state="readonly", width=16,
        ).pack(side="left", padx=(0, 16))
        self.loudness_mode_var.trace_add("write", lambda *_: self._on_loudness_mode_change())

        self._envelope_alpha_map = {
            get_text("opt_envelope_strong"): 0.6,
            get_text("opt_envelope_normal"): 0.3,
        }
        self.envelope_strength_label = ttk.Label(row4, text=get_text("lbl_envelope_strength"))
        self.envelope_strength_label.pack(side="left", padx=(0, 6))
        self.envelope_alpha_var = tk.StringVar(value=get_text("opt_envelope_strong"))
        self.envelope_alpha_combo = ttk.Combobox(
            row4, textvariable=self.envelope_alpha_var, values=list(self._envelope_alpha_map.keys()),
            state="readonly", width=8,
        )
        self.envelope_alpha_combo.pack(side="left")
        self._on_loudness_mode_change()

        opt_frame = ttk.Frame(frame)
        opt_frame.pack(fill="x", pady=(0, 6))
        self.keep_intermediate_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text=get_text("chk_keep_intermediate"), variable=self.keep_intermediate_var).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(opt_frame, text=get_text("chk_skip_existing"), variable=self.skip_existing_var).pack(side="left")

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(2, 6))
        self.btn_start = ttk.Button(btn_frame, text=get_text("btn_start"), command=self._run)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_stop = ttk.Button(btn_frame, text=get_text("btn_stop"), command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(5, 0))

        log_frame = ttk.LabelFrame(frame, text=get_text("lbl_log"), padding=6)
        log_frame.pack(fill="both", expand=True, pady=(0, 2))
        self.clone_log = tk.Text(log_frame, height=12, state="disabled")
        self.clone_log.pack(fill="both", expand=True)

    def _setup_single_mix_tab(self, frame):
        from tool_si import logic as sl

        self.single_mix_stop_event = threading.Event()
        self.single_mix_proc = None
        si = lambda k: i18n.translate("si", k)

        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=get_text("single_mix_info"), wraplength=760, justify="left", foreground="dim gray").grid(
            row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8)
        )

        mix_input_mode_frame = ttk.Frame(frame)
        mix_input_mode_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(mix_input_mode_frame, text=get_text("lbl_input_mode")).pack(side="left", padx=(0, 8))
        self.mix_input_mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            mix_input_mode_frame, text=get_text("opt_single_file"),
            variable=self.mix_input_mode_var, value="single",
            command=self._on_mix_input_mode_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            mix_input_mode_frame, text=get_text("opt_batch_dir"),
            variable=self.mix_input_mode_var, value="batch",
            command=self._on_mix_input_mode_change,
        ).pack(side="left")

        self.single_mix_video_row = ttk.Frame(frame)
        self.single_mix_video_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)
        self.single_mix_video_row.columnconfigure(1, weight=1)
        ttk.Label(self.single_mix_video_row, text=si("lbl_video_file")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.single_mix_video_var = tk.StringVar()
        ttk.Entry(self.single_mix_video_row, textvariable=self.single_mix_video_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(self.single_mix_video_row, text=get_text("btn_browse"), command=self._browse_single_mix_video).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.single_mix_audio_row = ttk.Frame(frame)
        self.single_mix_audio_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=3)
        self.single_mix_audio_row.columnconfigure(1, weight=1)
        ttk.Label(self.single_mix_audio_row, text=si("lbl_si_wav_file")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.single_mix_audio_var = tk.StringVar()
        ttk.Entry(self.single_mix_audio_row, textvariable=self.single_mix_audio_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(self.single_mix_audio_row, text=get_text("btn_browse"), command=self._browse_single_mix_audio).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.single_mix_output_row = ttk.Frame(frame)
        self.single_mix_output_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=3)
        self.single_mix_output_row.columnconfigure(1, weight=1)
        ttk.Label(self.single_mix_output_row, text=si("lbl_mix_output_file")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.single_mix_output_var = tk.StringVar()
        ttk.Entry(self.single_mix_output_row, textvariable=self.single_mix_output_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(self.single_mix_output_row, text=get_text("btn_browse"), command=self._browse_single_mix_output).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.mix_batch_dir_row = ttk.Frame(frame)
        self.mix_batch_dir_row.columnconfigure(1, weight=1)
        ttk.Label(self.mix_batch_dir_row, text=si("lbl_input_dir")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.mix_dir_var = tk.StringVar()
        ttk.Entry(self.mix_batch_dir_row, textvariable=self.mix_dir_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(self.mix_batch_dir_row, text=get_text("btn_browse"), command=self._browse_mix_dir).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        mode_frame = ttk.Frame(frame)
        mode_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Label(mode_frame, text=get_text("lbl_mode")).pack(side="left", padx=(0, 8))
        self.single_mix_mode_var = tk.StringVar(value="si")
        ttk.Radiobutton(
            mode_frame, text=get_text("opt_mode_si"), variable=self.single_mix_mode_var,
            value="si", command=self._on_single_mix_mode_change,
        ).pack(side="left", padx=(0, 12))
        dub_ok = self._dubbing_available()
        ttk.Radiobutton(
            mode_frame, text=get_text("opt_mode_dub") if dub_ok else get_text("opt_mode_dub_wip"),
            variable=self.single_mix_mode_var, value="dub",
            state="normal" if dub_ok else "disabled",
            command=self._on_single_mix_mode_change,
        ).pack(side="left")

        options = ttk.Frame(frame)
        self.single_mix_opts_frame = options
        self._single_mix_channel_map = {
            si("opt_channel_left"): "left",
            si("opt_channel_right"): "right",
            get_text("opt_channel_both"): "both",
        }
        self.single_mix_si_option_widgets = []
        self.single_mix_channel_label = ttk.Label(options, text=si("lbl_mix_channel"))
        self.single_mix_channel_label.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        self.single_mix_channel_var = tk.StringVar(value=si("opt_channel_left"))
        self.single_mix_channel_combo = ttk.Combobox(options, textvariable=self.single_mix_channel_var, values=list(self._single_mix_channel_map), width=16, state="readonly")
        self.single_mix_channel_combo.grid(row=0, column=1, sticky="w", pady=2)
        self.single_mix_origvol_label = ttk.Label(options, text=si("lbl_original_volume"))
        self.single_mix_origvol_label.grid(row=0, column=2, sticky="w", padx=(12, 6), pady=2)
        self.single_mix_origvol_var = tk.StringVar(value="100%")
        self.single_mix_origvol_combo = ttk.Combobox(options, textvariable=self.single_mix_origvol_var, values=[f"{v}%" for v in sl.ORIGINAL_VOLUME_CHOICES], width=8, state="readonly")
        self.single_mix_origvol_combo.grid(row=0, column=3, sticky="w", pady=2)
        self.single_mix_sivol_label = ttk.Label(options, text=si("lbl_si_volume"))
        self.single_mix_sivol_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        self.single_mix_sivol_var = tk.StringVar(value="50%")
        self.single_mix_sivol_combo = ttk.Combobox(options, textvariable=self.single_mix_sivol_var, values=[f"{v}%" for v in sl.SI_VOLUME_CHOICES], width=8, state="readonly")
        self.single_mix_sivol_combo.grid(row=1, column=1, sticky="w", pady=2)
        self.single_mix_delay_label = ttk.Label(options, text=si("lbl_si_delay"))
        self.single_mix_delay_label.grid(row=1, column=2, sticky="w", padx=(12, 6), pady=2)
        self.single_mix_delay_var = tk.StringVar(value="1s")
        self.single_mix_delay_combo = ttk.Combobox(options, textvariable=self.single_mix_delay_var, values=[f"{v:g}s" for v in sl.SI_DELAY_SECONDS_CHOICES], width=8, state="readonly")
        self.single_mix_delay_combo.grid(row=1, column=3, sticky="w", pady=2)
        self.single_mix_indep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text=si("chk_add_independent_track"), variable=self.single_mix_indep_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        self.single_mix_duck_var = tk.BooleanVar(value=True)
        self.single_mix_duck_check = ttk.Checkbutton(options, text=si("chk_duck_original_when_si"), variable=self.single_mix_duck_var)
        self.single_mix_duck_check.grid(row=2, column=2, columnspan=2, sticky="w", pady=2)
        self.single_mix_si_option_widgets.extend([
            self.single_mix_channel_label,
            self.single_mix_channel_combo,
            self.single_mix_origvol_label,
            self.single_mix_origvol_combo,
            self.single_mix_sivol_label,
            self.single_mix_sivol_combo,
            self.single_mix_delay_label,
            self.single_mix_delay_combo,
            self.single_mix_duck_check,
        ])

        btn_frame = ttk.Frame(frame)
        self._single_mix_btn_frame = btn_frame
        options.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        btn_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(2, 6))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        self.single_mix_btn_start = ttk.Button(btn_frame, text=get_text("btn_start_mix"), command=self._run_single_mix)
        self.single_mix_btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.single_mix_btn_stop = ttk.Button(btn_frame, text=get_text("btn_stop"), command=self._stop_single_mix, state="disabled")
        self.single_mix_btn_stop.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        log_frame = ttk.LabelFrame(frame, text=get_text("lbl_log"), padding=6)
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(8, weight=1)
        self.single_mix_log = tk.Text(log_frame, height=10, state="disabled")
        self.single_mix_log.pack(fill="both", expand=True)
        self.mix_batch_dir_row.grid_remove()

    def _selected_model_key(self) -> str:
        return self._model_map.get(self.model_var.get(), self.model_var.get())

    def _refresh_asr_model_status(self):
        from tool_clonevoice import whisperx_backend as wx

        if not hasattr(self, "asr_model_status_var"):
            return
        model_key = self._selected_model_key()
        model_dir = wx.model_dir(model_key, self.models_root)
        if wx.check_model_files(model_key, self.models_root):
            self.asr_model_status_var.set("")
            if self.asr_model_status_packed:
                self.asr_model_status_row.pack_forget()
                self.asr_model_status_packed = False
            if self.asr_download_button_packed:
                self.btn_download_asr_model.pack_forget()
                self.asr_download_button_packed = False
        else:
            self.asr_model_status_var.set(get_text("model_missing").format(model_dir))
            if not self.asr_model_status_packed:
                self.asr_model_status_row.pack(fill="x", pady=(0, 2))
                self.asr_model_status_packed = True
            if not self.asr_download_button_packed:
                self.btn_download_asr_model.pack(side="left")
                self.asr_download_button_packed = True
            self.btn_download_asr_model.config(state="normal")

    def _voice_model_specs(self):
        from tool_clonevoice import model_downloads as md

        return {
            "omnivoice": (
                md.OMNIVOICE_SPEC,
                self.omnivoice_status_var,
                self.btn_download_omnivoice,
                "omnivoice_download_button_packed",
                self.ov_row,
                "ov_row_packed",
            ),
            "ecapa": (
                md.ECAPA_SPEC,
                self.ecapa_status_var,
                self.btn_download_ecapa,
                "ecapa_download_button_packed",
                self.ecapa_row,
                "ecapa_row_packed",
            ),
        }

    def _refresh_voice_model_status(self):
        from tool_clonevoice import model_downloads as md

        if not hasattr(self, "omnivoice_status_var"):
            return
        any_missing = False
        for _key, (spec, status_var, button, flag_name, row, row_flag_name) in self._voice_model_specs().items():
            model_dir = md.model_dir(self.models_root, spec)
            if md.check_model_files(self.models_root, spec):
                status_var.set("")
                if getattr(self, flag_name):
                    button.pack_forget()
                    setattr(self, flag_name, False)
                if getattr(self, row_flag_name):
                    row.pack_forget()
                    setattr(self, row_flag_name, False)
            else:
                any_missing = True
                status_var.set(get_text("voice_model_missing").format(spec.label, model_dir))
                if not getattr(self, row_flag_name):
                    row.pack(fill="x", pady=1)
                    setattr(self, row_flag_name, True)
                if not getattr(self, flag_name):
                    button.pack(side="right", padx=(8, 0))
                    setattr(self, flag_name, True)
                button.config(state="normal")
        if any_missing:
            if not self.voice_model_frame_packed:
                self.voice_model_frame.pack(fill="x", pady=(2, 4))
                self.voice_model_frame_packed = True
        elif self.voice_model_frame_packed:
            self.voice_model_frame.pack_forget()
            self.voice_model_frame_packed = False

    def _confirm_model_download(self, model_label: str, total_bytes: int | None, files: list[tuple[str, int | None]]) -> bool:
        from tool_clonevoice import model_downloads as md

        size_text = md.format_bytes(total_bytes)
        lines = [get_text("confirm_download_model").format(model_label, size_text)]
        if files:
            lines.append("")
            lines.append(get_text("confirm_download_files"))
            for filename, size in files[:8]:
                lines.append(f"- {filename} ({md.format_bytes(size)})")
            if len(files) > 8:
                lines.append(f"- ... +{len(files) - 8}")
        return messagebox.askyesno(get_text("confirm_download_title"), "\n".join(lines))

    def _query_and_confirm_download(self, model_label: str, plan_func) -> bool:
        holder: dict[str, object] = {}
        files, total = plan_func()
        event = threading.Event()

        def ask():
            holder["ok"] = self._confirm_model_download(model_label, total, files)
            event.set()

        self.root.after(0, ask)
        event.wait()
        return bool(holder.get("ok"))

    def _download_asr_model(self):
        from tool_clonevoice import whisperx_backend as wx

        model_key = self._selected_model_key()
        model_label = self.model_var.get()

        def task():
            self.root.after(0, lambda: self.btn_download_asr_model.config(state="disabled"))
            self.log(self.clone_log, get_text("msg_check_download_size").format(model_label))
            ok_to_download = self._query_and_confirm_download(
                model_label,
                lambda: wx.remote_file_plan(model_key, lambda m: self.log(self.clone_log, m)),
            )
            if not ok_to_download:
                self.log(self.clone_log, get_text("msg_download_cancelled"))
                self.root.after(0, lambda: self.btn_download_asr_model.config(state="normal"))
                return
            ok = wx.download_model(model_key, self.models_root, lambda m: self.log(self.clone_log, m))

            def finish():
                self._refresh_asr_model_status()
                if not ok:
                    self.btn_download_asr_model.config(state="normal")
                    self.log(self.clone_log, get_text("msg_download_failed"))

            self.root.after(0, finish)

        threading.Thread(target=task, daemon=True).start()

    def _download_voice_model(self, key: str):
        from tool_clonevoice import model_downloads as md

        spec, _status_var, button, _flag_name = self._voice_model_specs()[key]

        def task():
            self.root.after(0, lambda: button.config(state="disabled"))
            self.log(self.clone_log, get_text("msg_check_download_size").format(spec.label))
            ok_to_download = self._query_and_confirm_download(
                spec.label,
                lambda: md.remote_file_plan(spec, lambda m: self.log(self.clone_log, m)),
            )
            if not ok_to_download:
                self.log(self.clone_log, get_text("msg_download_cancelled"))
                self.root.after(0, lambda: button.config(state="normal"))
                return
            ok = md.download_model(self.models_root, spec, lambda m: self.log(self.clone_log, m))

            def finish():
                self._refresh_voice_model_status()
                if not ok:
                    button.config(state="normal")
                    self.log(self.clone_log, get_text("msg_download_failed"))

            self.root.after(0, finish)

        threading.Thread(target=task, daemon=True).start()

    def _dubbing_available(self) -> bool:
        # Mirror tool_clonevoice.separate.is_available() without importing it.
        # Importing separate pulls in torch/torchaudio/Bandit/librosa and can
        # make the launcher click feel frozen in the packaged build.
        bandit_dir = os.path.join(self.models_root, "bandit-v2")
        return (
            os.path.isfile(os.path.join(bandit_dir, "checkpoint-multi.slim.pt"))
            or os.path.isfile(os.path.join(bandit_dir, "checkpoint-multi.ckpt"))
        )

    def _browse_mix_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.mix_dir_var.set(d)

    def _on_mix_input_mode_change(self):
        if self.mix_input_mode_var.get() == "batch":
            self.single_mix_video_row.grid_remove()
            self.single_mix_audio_row.grid_remove()
            self.single_mix_output_row.grid_remove()
            self.mix_batch_dir_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)
        else:
            self.mix_batch_dir_row.grid_remove()
            self.single_mix_video_row.grid()
            self.single_mix_audio_row.grid()
            self.single_mix_output_row.grid()

    def _browse_single_mix_video(self):
        from tool_si import logic as sl

        path = filedialog.askopenfilename(
            filetypes=[("MP4", "*.mp4"), ("Video Files", "*.mp4 *.mkv *.mov"), ("All Files", "*.*")]
        )
        if path:
            self.single_mix_video_var.set(path)
            self.single_mix_audio_var.set(sl.default_si_audio_path(path))
            self.single_mix_output_var.set(sl.default_si_mix_output_path(path))

    def _browse_single_mix_audio(self):
        initial = self.single_mix_audio_var.get().strip()
        path = filedialog.askopenfilename(
            initialfile=os.path.basename(initial) if initial else "",
            initialdir=os.path.dirname(initial) if initial else "",
            filetypes=[("WAV", "*.wav"), ("All Files", "*.*")],
        )
        if path:
            self.single_mix_audio_var.set(path)

    def _browse_single_mix_output(self):
        from tool_si import logic as sl

        initial = self.single_mix_output_var.get().strip() or (
            sl.default_si_mix_output_path(self.single_mix_video_var.get()) if self.single_mix_video_var.get() else ""
        )
        path = filedialog.asksaveasfilename(
            initialfile=os.path.basename(initial) if initial else "",
            initialdir=os.path.dirname(initial) if initial else "",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All Files", "*.*")],
        )
        if path:
            self.single_mix_output_var.set(path)

    def _run_single_mix(self):
        from tool_si import logic as sl
        from tool_clonevoice import dubbing as dub

        dubbing = self.single_mix_mode_var.get() == "dub"
        batch_mode = self.mix_input_mode_var.get() == "batch"
        if batch_mode:
            base = self.mix_dir_var.get().strip()
            if not base or not os.path.isdir(base):
                messagebox.showerror("Error", get_text("err_no_dir"))
                return
        else:
            video_path = self.single_mix_video_var.get().strip()
            if not video_path or not os.path.isfile(video_path):
                messagebox.showerror("Error", i18n.translate("si", "err_video_file"))
                return
            si_audio_path = self.single_mix_audio_var.get().strip() or sl.default_si_audio_path(video_path)
            if not si_audio_path or not os.path.isfile(si_audio_path):
                messagebox.showerror("Error", i18n.translate("si", "err_si_wav_file"))
                return
            default_output = (
                dub.default_dub_output_path(video_path)
                if dubbing
                else sl.default_si_mix_output_path(video_path)
            )
            output_path = self.single_mix_output_var.get().strip() or default_output
            if os.path.abspath(video_path) == os.path.abspath(output_path):
                messagebox.showerror("Error", i18n.translate("si", "err_mix_output_file"))
                return
            if os.path.exists(output_path) and not messagebox.askyesno(
                i18n.translate("si", "msg_mix_overwrite_title"),
                i18n.translate("si", "msg_mix_overwrite").format(output_path),
            ):
                return

        channel = "both" if dubbing else self._single_mix_channel_map.get(self.single_mix_channel_var.get(), "left")
        orig_vol = int(self.single_mix_origvol_var.get().rstrip("%"))
        si_vol = int(self.single_mix_sivol_var.get().rstrip("%"))
        delay = 0.0 if dubbing else float(self.single_mix_delay_var.get().rstrip("s"))
        indep = self.single_mix_indep_var.get()
        duck = False if dubbing else self.single_mix_duck_var.get()

        self.single_mix_stop_event.clear()
        self.single_mix_btn_start.config(state="disabled")
        self.single_mix_btn_stop.config(state="normal")

        def task():
            try:
                if dubbing:
                    self._run_dub_task(batch_mode, base if batch_mode else video_path,
                                       None if batch_mode else si_audio_path,
                                       None if batch_mode else output_path,
                                       indep)
                    return
                if batch_mode:
                    sl.batch_mix_si_audio_tracks(
                        base_dir=base,
                        mix_channel=channel,
                        original_volume_percent=orig_vol,
                        si_volume_percent=si_vol,
                        si_delay_seconds=delay,
                        add_independent_track=indep,
                        duck_original=duck,
                        log_callback=lambda m: self.log(self.single_mix_log, m),
                        stop_event=self.single_mix_stop_event,
                        process_callback=lambda p: setattr(self, "single_mix_proc", p),
                    )
                    if not self.single_mix_stop_event.is_set():
                        self.log(self.single_mix_log, get_text("msg_mix_done"))
                    return
                output = sl.mix_si_audio_track(
                    video_path=video_path,
                    si_audio_path=si_audio_path,
                    output_path=output_path,
                    mix_channel=channel,
                    original_volume_percent=orig_vol,
                    si_volume_percent=si_vol,
                    si_delay_seconds=delay,
                    add_independent_track=indep,
                    duck_original=duck,
                    log_callback=lambda m: self.log(self.single_mix_log, m),
                    stop_event=self.single_mix_stop_event,
                    process_callback=lambda p: setattr(self, "single_mix_proc", p),
                )
                if not self.single_mix_stop_event.is_set():
                    self.log(self.single_mix_log, get_text("msg_mix_done").format(output))
            except Exception as e:
                self.log(self.single_mix_log, f"Error: {e}")
            finally:
                self.root.after(0, lambda: self.single_mix_btn_start.config(state="normal"))
                self.root.after(0, lambda: self.single_mix_btn_stop.config(state="disabled"))

        self.single_mix_thread = threading.Thread(target=task, daemon=True)
        self.single_mix_thread.start()

    def _run_dub_task(self, batch_mode, target, si_audio_path, output_path, add_independent_track):
        """Dubbing mode: separate original dialogue (bandit-v2) and replace it
        with the cloned voice. The separator stays resident in VRAM across a
        batch (no OmniVoice runs concurrently here)."""
        from tool_clonevoice import dubbing as dub

        log = lambda m: self.log(self.single_mix_log, m)
        proc = lambda p: setattr(self, "single_mix_proc", p)
        # Dubbing fixes both channels / 100% voice / 0 s delay; the cloned track
        # is already aligned to the source timeline, so no delay or channel split.
        # Route bandit-v2's progress bar (tqdm to stderr) into the GUI log.
        with redirect_stdio(self._make_log_emitter(self.single_mix_log)):
            if batch_mode:
                dub.batch_dub_videos(
                    base_dir=target,
                    models_root=self.models_root,
                    background_volume_percent=100,
                    voice_volume_percent=100,
                    add_independent_track=add_independent_track,
                    log_callback=log,
                    stop_event=self.single_mix_stop_event,
                    process_callback=proc,
                )
                if not self.single_mix_stop_event.is_set():
                    self.log(self.single_mix_log, get_text("msg_mix_done"))
                return
            separator = dub.BanditSeparator(self.models_root, log=log)
            try:
                output = dub.dub_video(
                    video_path=target,
                    si_audio_path=si_audio_path,
                    output_path=output_path,
                    separator=separator,
                    background_volume_percent=100,
                    voice_volume_percent=100,
                    add_independent_track=add_independent_track,
                    log_callback=log,
                    stop_event=self.single_mix_stop_event,
                    process_callback=proc,
                )
            finally:
                separator.close()
            if not self.single_mix_stop_event.is_set():
                self.log(self.single_mix_log, get_text("msg_mix_done").format(output))

    def _stop_single_mix(self):
        from tool_si import logic as sl

        self.single_mix_stop_event.set()
        try:
            if self.single_mix_proc is not None:
                sl._terminate_process(self.single_mix_proc)
        except Exception:
            pass
        self.single_mix_btn_stop.config(state="disabled")

    def _on_loudness_mode_change(self):
        mode = self._loudness_mode_map.get(self.loudness_mode_var.get(), "envelope")
        if mode == "envelope":
            self.envelope_strength_label.pack(side="left", padx=(0, 6))
            self.envelope_alpha_combo.pack(side="left")
        else:
            self.envelope_strength_label.pack_forget()
            self.envelope_alpha_combo.pack_forget()

    def _selected_target_language(self) -> str:
        return self._tgt_map.get(self.tgt_lang_var.get(), "Chinese")

    def _on_single_mix_mode_change(self):
        from tool_si import logic as sl
        from tool_clonevoice import dubbing as dub

        dubbing = self.single_mix_mode_var.get() == "dub"
        if dubbing:
            # Dubbing replaces the dialogue and always uses both channels /
            # 100% voice / 0 s delay. Snapshot the SI settings so toggling back
            # restores them; keep the options visible so independent-track mode
            # remains available.
            self._si_mode_saved = (
                self.single_mix_channel_var.get(),
                self.single_mix_sivol_var.get(),
                self.single_mix_delay_var.get(),
            )
            self.single_mix_channel_var.set(get_text("opt_channel_both"))
            self.single_mix_sivol_var.set("100%")
            self.single_mix_delay_var.set("0s")
            for widget in self.single_mix_si_option_widgets:
                widget.grid_remove()
        else:
            saved = getattr(self, "_si_mode_saved", None)
            if saved is not None:
                self.single_mix_channel_var.set(saved[0])
                self.single_mix_sivol_var.set(saved[1])
                self.single_mix_delay_var.set(saved[2])
                self._si_mode_saved = None
            for widget in self.single_mix_si_option_widgets:
                widget.grid()
            self.single_mix_opts_frame.grid()

        # Swap the auto-managed output suffix (_SI <-> _DUB) unless the user
        # typed a custom path.
        video = self.single_mix_video_var.get().strip()
        if video:
            si_default = sl.default_si_mix_output_path(video)
            dub_default = dub.default_dub_output_path(video)
            current = self.single_mix_output_var.get().strip()
            if current in ("", si_default, dub_default):
                self.single_mix_output_var.set(dub_default if dubbing else si_default)

    # --- helpers ---
    def log(self, text_widget, message):
        def _log():
            text_widget.config(state="normal")
            text_widget.insert("end", message + "\n")
            text_widget.see("end")
            text_widget.config(state="disabled")
            text_widget._last_was_progress = False
        self.root.after(0, _log)

    def _make_log_emitter(self, text_widget):
        """Return an ``emit(text, is_progress)`` for log_redirect: progress
        (carriage-return) lines replace the previous progress line instead of
        piling up; normal lines append."""
        def emit(text, is_progress):
            def _do():
                text_widget.config(state="normal")
                if is_progress and getattr(text_widget, "_last_was_progress", False):
                    text_widget.delete("end-2l linestart", "end-1l linestart")
                text_widget.insert("end", text + "\n")
                text_widget.see("end")
                text_widget.config(state="disabled")
                text_widget._last_was_progress = is_progress
            self.root.after(0, _do)
        return emit

    def _on_clone_input_mode_change(self):
        if self.clone_input_mode_var.get() == "batch":
            self.clone_input_label_var.set(get_text("lbl_input_dir"))
        else:
            self.clone_input_label_var.set(get_text("lbl_input_video"))
        self.input_video_var.set("")

    def _browse_video(self):
        if self.clone_input_mode_var.get() == "batch":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"), ("All files", "*.*")]
            )
        if path:
            self.input_video_var.set(path)

    def _scan_clone_batch_videos(self, base_dir: str) -> list[str]:
        videos = []
        for root, _dirs, files in os.walk(base_dir):
            for name in files:
                if name.lower().endswith(VIDEO_EXTENSIONS) and not _is_generated_output_mp4(name):
                    videos.append(os.path.join(root, name))
        return sorted(videos, key=lambda p: p.lower())

    def _translation_api_key_configured(self) -> bool:
        try:
            import keyring

            api_key = (
                keyring.get_password("VR_Video_Toolbox", "deepseek_api_key")
                or keyring.get_password("VR_Mosaic_Removal", "deepseek_api_key")
            )
        except Exception:
            api_key = None
        return bool((api_key or "").strip())

    def _run(self):
        input_path = self.input_video_var.get().strip()
        batch_mode = self.clone_input_mode_var.get() == "batch"
        if batch_mode:
            if not input_path or not os.path.isdir(input_path):
                messagebox.showerror("Error", get_text("err_no_dir"))
                return
            videos = self._scan_clone_batch_videos(input_path)
            if not videos:
                messagebox.showerror("Error", get_text("err_no_batch_videos"))
                return
        elif not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", get_text("err_no_video"))
            return
        else:
            videos = [input_path]

        if not self._translation_api_key_configured():
            messagebox.showerror("Error", get_text("err_no_translation_api_key"))
            return
        target_language = self._selected_target_language()
        if not target_language:
            messagebox.showerror("Error", get_text("err_no_target_lang"))
            return

        from tool_clonevoice import logic

        self.stop_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        model_key = self._selected_model_key()
        language = self._lang_map.get(self.src_lang_var.get())
        backend = self._diar_map.get(self.diar_var.get(), "auto")
        num_speakers = self._num_map.get(self.num_spk_var.get())
        denoise = self._denoise_map.get(self.denoise_var.get(), "none")
        loudness_mode = self._loudness_mode_map.get(self.loudness_mode_var.get(), "envelope")
        envelope_alpha = self._envelope_alpha_map.get(self.envelope_alpha_var.get(), 0.6)
        keep_intermediate = self.keep_intermediate_var.get()
        skip_existing = self.skip_existing_var.get()

        def task():
            holder: list = []

            def release_holder_on_main_thread():
                cleaned = threading.Event()

                def cleanup_models():
                    logic.release_model_holder(holder)
                    cleaned.set()

                self.root.after(0, cleanup_models)
                cleaned.wait()

            # Route model progress bars (tqdm to stderr) into the GUI log so the
            # packaged (console-less) app still shows progress.
            _redir = redirect_stdio(self._make_log_emitter(self.clone_log))
            _redir.__enter__()
            try:
                if batch_mode:
                    self.log(self.clone_log, get_text("msg_batch_start").format(len(videos)))
                outputs = []
                for index, video in enumerate(videos, 1):
                    if self.stop_event.is_set():
                        raise RuntimeError("Stopped by user.")
                    if batch_mode:
                        self.log(self.clone_log, get_text("msg_batch_item").format(index, len(videos), video))
                    out = logic.run_full(
                        video,
                        model_key=model_key,
                        language=language,
                        diarize_backend=backend,
                        num_speakers=num_speakers,
                        target_language=target_language,
                        denoise=denoise,
                        loudness_mode=loudness_mode,
                        envelope_alpha=envelope_alpha,
                        models_root=self.models_root,
                        keep_intermediate=keep_intermediate,
                        skip_existing=skip_existing,
                        log=lambda m: self.log(self.clone_log, m),
                        stop_event=self.stop_event,
                        model_holder=holder,
                    )
                    outputs.append(out)
                    if batch_mode and index < len(videos):
                        release_holder_on_main_thread()
                if not self.stop_event.is_set():
                    if batch_mode:
                        self.log(self.clone_log, get_text("msg_batch_clone_done").format(len(outputs)))
                    else:
                        self.log(self.clone_log, get_text("msg_done"))
            except RuntimeError as e:
                if "Stopped by user" in str(e):
                    self.log(self.clone_log, get_text("msg_stopped"))
                else:
                    self.log(self.clone_log, f"Error: {e}")
            except Exception as e:
                self.log(self.clone_log, f"Error: {e}")
            finally:
                _redir.__exit__(None, None, None)
                # Release native (CTranslate2/torch) models on the main thread to
                # avoid a background-thread C++ destructor crash.
                release_holder_on_main_thread()
                self.root.after(0, lambda: self.btn_start.config(state="normal"))
                self.root.after(0, lambda: self.btn_stop.config(state="disabled"))

        self.run_thread = threading.Thread(target=task, daemon=True)
        self.run_thread.start()

    def _open_translate_config(self):
        """AI-translation API config dialog (shared with tool_subtitle)."""
        from tool_subtitle import logic as tsl
        from tool_subtitle.gui import _load_keyring
        st = lambda k: i18n.translate("subtitle", k)
        mt = lambda k: i18n.translate("main", k)
        cfg = tsl.load_trans_config()

        dlg = tk.Toplevel(self.root)
        dlg.title(get_text("title_trans_config"))
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        def _row(label, var, show=None):
            f = ttk.Frame(dlg)
            f.pack(fill="x", padx=12, pady=4)
            ttk.Label(f, text=label, width=16).pack(side="left")
            ttk.Entry(f, textvariable=var, show=show, width=40).pack(side="left", fill="x", expand=True)

        url_var = tk.StringVar(value=cfg.get("api_base_url", ""))
        model_var = tk.StringVar(value=cfg.get("model_name", ""))
        key_var = tk.StringVar(value="")
        tokens_var = tk.StringVar(value=str(cfg.get("tokens_per_chunk", 500000)))
        adult_var = tk.BooleanVar(value=bool(cfg.get("adult_content", True)))

        _row(st("lbl_api_url"), url_var)
        _row(st("lbl_model_name"), model_var)
        _row(st("lbl_api_key"), key_var, show="*")
        _row(st("lbl_tokens"), tokens_var)
        ttk.Checkbutton(dlg, text=st("chk_adult_content"), variable=adult_var).pack(anchor="w", padx=14, pady=2)

        try:
            keyring = _load_keyring()
            saved_key = (
                keyring.get_password("VR_Video_Toolbox", "deepseek_api_key")
                or keyring.get_password("VR_Mosaic_Removal", "deepseek_api_key")
                or ""
            )
        except Exception:
            keyring = None
            saved_key = ""
        if saved_key:
            key_var.set(saved_key)

        status = ttk.Label(dlg, text="配置完成" if saved_key else "", foreground="gray")
        status.pack(fill="x", padx=14, pady=(4, 0))

        def _test():
            api_key = key_var.get().strip()
            if not api_key:
                status.config(text=st("err_no_api_key"), foreground="red")
                return
            status.config(text="...", foreground="gray")

            def work():
                try:
                    client = tsl.LLMClient(url_var.get().strip(), api_key, model_var.get().strip(), temperature=0.3)
                    client.complete("Reply with OK.")
                    if keyring:
                        keyring.set_password("VR_Video_Toolbox", "deepseek_api_key", api_key)
                    self.root.after(0, lambda: status.config(text="OK ✓", foreground="green"))
                except Exception as e:
                    msg = str(e)[:60]
                    self.root.after(0, lambda: status.config(text=f"FAIL: {msg}", foreground="red"))

            threading.Thread(target=work, daemon=True).start()

        def _delete_key():
            if keyring:
                for svc in ("VR_Video_Toolbox", "VR_Mosaic_Removal"):
                    try:
                        keyring.delete_password(svc, "deepseek_api_key")
                    except Exception:
                        pass
            key_var.set("")
            status.config(text="-", foreground="gray")

        def _save():
            cfg["api_base_url"] = url_var.get().strip()
            cfg["model_name"] = model_var.get().strip()
            try:
                cfg["tokens_per_chunk"] = int(tokens_var.get())
            except ValueError:
                pass
            cfg["adult_content"] = adult_var.get()
            cfg["dubbing_optimized"] = True
            tsl.save_trans_config(cfg)
            k = key_var.get().strip()
            if k and keyring:
                keyring.set_password("VR_Video_Toolbox", "deepseek_api_key", k)
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill="x", padx=12, pady=10)
        ttk.Button(bf, text=st("btn_test_api"), command=_test).pack(side="left", padx=2)
        ttk.Button(bf, text=st("btn_delete_key"), command=_delete_key).pack(side="left", padx=2)
        ttk.Button(bf, text=mt("btn_save"), command=_save).pack(side="right", padx=2)
        ttk.Button(bf, text=mt("btn_cancel"), command=dlg.destroy).pack(side="right", padx=2)
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

    def _stop(self):
        self.stop_event.set()
        self.btn_stop.config(state="disabled")
