from __future__ import annotations

import gc
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from utils import i18n
from tool_clonevoice.gui_candidate_panel import CandidateBasisPanel
from tool_clonevoice.gui_proofread import build_proofread_panel
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

        self.tab_single_clone = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_single_clone, text=get_text("tab_single_clone"))
        self._setup_single_clone_tab(self.tab_single_clone)

        self.tab_multi_clone = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_multi_clone, text=get_text("tab_multi_clone"))
        self._setup_multi_clone_tab(self.tab_multi_clone)

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
            input_mode_frame, text=get_text("opt_shared_dir"),
            variable=self.clone_input_mode_var, value="shared",
            command=self._on_clone_input_mode_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            input_mode_frame, text=get_text("opt_batch_dir_independent"),
            variable=self.clone_input_mode_var, value="batch",
            command=self._on_clone_input_mode_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            input_mode_frame, text=get_text("opt_shared_batch_dir"),
            variable=self.clone_input_mode_var, value="shared_batch",
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

        # VAD sensitivity on its own row (the model row above is already full).
        self.vad_sensitivity_map = {
            get_text("opt_vad_standard"): "standard",
            get_text("opt_vad_high"): "high",
            get_text("opt_vad_max"): "max",
        }
        # Denoise + sensitivity share one row: the two must be tuned together
        # (high sensitivity without denoise feeds noise to the decoder).
        clone_vad_row = ttk.Frame(options_frame)
        clone_vad_row.pack(fill="x", pady=2)
        ttk.Label(clone_vad_row, text=get_text("lbl_denoise"), width=12).pack(side="left")
        self._denoise_map = {
            get_text("opt_denoise_none"): "none",
            get_text("opt_denoise_mild"): "mild",
            get_text("opt_denoise_balanced"): "balanced",
            get_text("opt_denoise_strong"): "strong",
        }
        self.denoise_var = tk.StringVar(value=get_text("opt_denoise_mild"))
        ttk.Combobox(clone_vad_row, textvariable=self.denoise_var, values=list(self._denoise_map.keys()), state="readonly", width=12).pack(side="left", padx=(0, 16))
        ttk.Label(clone_vad_row, text=get_text("lbl_vad_sensitivity"), width=12).pack(side="left")
        self.clone_vad_var = tk.StringVar(value=get_text("opt_vad_high"))
        ttk.Combobox(
            clone_vad_row, textvariable=self.clone_vad_var,
            values=list(self.vad_sensitivity_map.keys()),
            state="readonly", width=16,
        ).pack(side="left")

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
        self._num_map = {
            get_text("opt_num_auto"): None,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
        }
        self.num_spk_var = tk.StringVar(value=get_text("opt_num_auto"))
        ttk.Combobox(row2, textvariable=self.num_spk_var, values=list(self._num_map.keys()), state="readonly", width=14).pack(side="left")

        correction_row = ttk.Frame(options_frame)
        correction_row.pack(fill="x", pady=2)
        self.source_correction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            correction_row, text=get_text("chk_source_correction"),
            variable=self.source_correction_var,
        ).pack(side="left")

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

        self._tempo_fit_map = {
            get_text("opt_tempo_fit_off"): "off",
            get_text("opt_tempo_fit_moderate"): "moderate",
            get_text("opt_tempo_fit_strong"): "strong",
        }
        ttk.Label(row4, text=get_text("lbl_tempo_fit")).pack(side="left", padx=(16, 6))
        self.tempo_fit_var = tk.StringVar(value=get_text("opt_tempo_fit_moderate"))
        ttk.Combobox(
            row4, textvariable=self.tempo_fit_var, values=list(self._tempo_fit_map.keys()),
            state="readonly", width=8,
        ).pack(side="left")
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

    def _setup_single_clone_tab(self, frame):
        self.single_clone_stop_event = threading.Event()
        self.single_clone_thread: threading.Thread | None = None
        self.single_clone_candidates: list[dict] = []
        self.single_clone_candidate_iid_to_index: dict[str, int] = {}
        self.single_clone_videos: list[str] = []
        self.single_clone_basis_applied = False
        self.single_clone_basis_source_kind = ""
        self.single_clone_basis_meta: dict = {}
        self.single_clone_step_index = 0

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(3, weight=1)

        ttk.Label(
            frame,
            text=get_text("single_clone_note"),
            wraplength=760,
            justify="left",
            foreground="dim gray",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        step_bar = ttk.Frame(frame)
        step_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        step_bar.columnconfigure(1, weight=1)
        self.single_clone_step_title_var = tk.StringVar()
        ttk.Label(step_bar, textvariable=self.single_clone_step_title_var, font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        nav = ttk.Frame(step_bar)
        nav.grid(row=0, column=2, sticky="e")
        self.single_clone_btn_prev = ttk.Button(nav, text=get_text("btn_prev_step"), command=lambda: self._show_single_clone_step(self.single_clone_step_index - 1))
        self.single_clone_btn_prev.pack(side="left", padx=(0, 4))
        self.single_clone_btn_next = ttk.Button(nav, text=get_text("btn_next_step"), command=lambda: self._show_single_clone_step(self.single_clone_step_index + 1))
        self.single_clone_btn_next.pack(side="left", padx=(0, 4))
        self.single_clone_btn_stop = ttk.Button(nav, text=get_text("btn_stop"), command=self._stop_single_clone, state="disabled")
        self.single_clone_btn_stop.pack(side="left")

        content = ttk.Frame(frame)
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.single_clone_step_frames: list[ttk.Frame] = []
        self.single_clone_step_names = [
            get_text("step_1_transcribe"),
            get_text("step_2_select_basis"),
            get_text("step_3_confirm_basis"),
            get_text("step_4_translate_clone"),
        ]

        step1 = ttk.Frame(content, padding=8)
        step1.columnconfigure(1, weight=1)
        self.single_clone_step_frames.append(step1)
        step1_label_width = 16
        step1_secondary_label_width = 10

        mode_row = ttk.Frame(step1)
        mode_row.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        ttk.Label(mode_row, text=get_text("lbl_input_mode"), width=step1_label_width).pack(side="left", padx=(0, 6))
        self.single_clone_input_mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode_row,
            text=get_text("opt_single_file"),
            variable=self.single_clone_input_mode_var,
            value="single",
            command=self._on_single_clone_input_mode_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            mode_row,
            text=get_text("opt_shared_dir"),
            variable=self.single_clone_input_mode_var,
            value="batch",
            command=self._on_single_clone_input_mode_change,
        ).pack(side="left")

        self.single_clone_input_label_var = tk.StringVar(value=get_text("lbl_input_video"))
        ttk.Label(step1, textvariable=self.single_clone_input_label_var, width=step1_label_width).grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.single_clone_input_var = tk.StringVar()
        ttk.Entry(step1, textvariable=self.single_clone_input_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)
        self.single_clone_btn_browse = ttk.Button(step1, text=get_text("btn_browse"), command=self._browse_single_clone_input)
        self.single_clone_btn_browse.grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=3)

        self.single_clone_lang_map = {
            get_text("opt_lang_auto"): None,
            get_text("opt_lang_ja"): "ja",
            get_text("opt_lang_en"): "en",
            get_text("opt_lang_zh"): "zh",
        }
        self.single_clone_model_map = {
            "large-v3": "large-v3",
            "large-v2": "large-v2",
            get_text("opt_model_kotoba"): "kotoba",
        }
        self.single_clone_denoise_map = {
            get_text("opt_denoise_none"): "none",
            get_text("opt_denoise_mild"): "mild",
            get_text("opt_denoise_balanced"): "balanced",
            get_text("opt_denoise_strong"): "strong",
        }
        self.single_clone_tgt_map = {
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
        self.single_clone_loudness_mode_map = {
            get_text("opt_loudness_flat"): "flat",
            get_text("opt_loudness_sentence"): "sentence",
            get_text("opt_loudness_envelope"): "envelope",
        }
        self.single_clone_envelope_alpha_map = {
            get_text("opt_envelope_strong"): 0.6,
            get_text("opt_envelope_normal"): 0.3,
        }

        single_clone_options_row = ttk.Frame(step1)
        single_clone_options_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Label(single_clone_options_row, text=get_text("lbl_src_lang"), width=step1_label_width).pack(
            side="left", padx=(0, 6)
        )
        self.single_clone_src_lang_var = tk.StringVar(value=get_text("opt_lang_auto"))
        ttk.Combobox(
            single_clone_options_row,
            textvariable=self.single_clone_src_lang_var,
            values=list(self.single_clone_lang_map.keys()),
            state="readonly",
            width=14,
        ).pack(side="left", padx=(0, 16))

        ttk.Label(single_clone_options_row, text=get_text("lbl_denoise"), width=step1_secondary_label_width).pack(
            side="left", padx=(0, 6)
        )
        self.single_clone_denoise_var = tk.StringVar(value=get_text("opt_denoise_mild"))
        ttk.Combobox(
            single_clone_options_row,
            textvariable=self.single_clone_denoise_var,
            values=list(self.single_clone_denoise_map.keys()),
            state="readonly",
            width=14,
        ).pack(side="left", padx=(0, 16))

        self.vad_sensitivity_map = {
            get_text("opt_vad_standard"): "standard",
            get_text("opt_vad_high"): "high",
            get_text("opt_vad_max"): "max",
        }
        ttk.Label(single_clone_options_row, text=get_text("lbl_vad_sensitivity")).pack(side="left", padx=(0, 6))
        self.single_clone_vad_var = tk.StringVar(value=get_text("opt_vad_high"))
        ttk.Combobox(
            single_clone_options_row, textvariable=self.single_clone_vad_var,
            values=list(self.vad_sensitivity_map.keys()),
            state="readonly", width=14,
        ).pack(side="left")

        ttk.Label(step1, text=get_text("lbl_model"), width=step1_label_width).grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.single_clone_model_var = tk.StringVar(value="large-v3")
        self.single_clone_model_combo = ttk.Combobox(
            step1,
            textvariable=self.single_clone_model_var,
            values=list(self.single_clone_model_map.keys()),
            state="readonly",
            width=16,
        )
        self.single_clone_model_combo.grid(row=3, column=1, sticky="w", pady=3)
        self.single_clone_model_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_single_asr_model_status())
        self.btn_download_single_asr_model = ttk.Button(
            step1,
            text=get_text("btn_download_model"),
            command=self._download_single_asr_model,
        )
        self.btn_download_single_asr_model.grid(row=3, column=2, sticky="w", padx=(6, 0), pady=3)
        self.single_asr_model_status_var = tk.StringVar()
        self.single_asr_model_status_label = ttk.Label(step1, textvariable=self.single_asr_model_status_var, foreground="dim gray")
        self.single_asr_model_status_label.grid(
            row=4, column=1, columnspan=3, sticky="ew", pady=(0, 3)
        )

        self.single_clone_source_correction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            step1,
            text=get_text("chk_source_correction"),
            variable=self.single_clone_source_correction_var,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=3)

        single_clone_target_row = ttk.Frame(step1)
        single_clone_target_row.grid(row=6, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Label(single_clone_target_row, text=get_text("lbl_single_clone_target_lang"), width=step1_label_width).pack(
            side="left", padx=(0, 6)
        )
        self.single_clone_tgt_lang_var = tk.StringVar(value=get_text("opt_lang_zh"))
        self.single_clone_tgt_lang_combo = ttk.Combobox(
            single_clone_target_row,
            textvariable=self.single_clone_tgt_lang_var,
            values=list(self.single_clone_tgt_map.keys()),
            state="readonly",
            width=16,
        )
        self.single_clone_tgt_lang_combo.pack(side="left")
        self.single_clone_tgt_lang_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_single_clone_target_label())
        self.single_clone_btn_trans_config = ttk.Button(
            single_clone_target_row,
            text=get_text("btn_trans_config"),
            command=self._open_single_translate_config,
            width=16,
        )
        self.single_clone_btn_trans_config.pack(side="left", padx=(8, 8))
        self.single_translate_status_var = tk.StringVar()
        ttk.Label(single_clone_target_row, textvariable=self.single_translate_status_var, foreground="dim gray").pack(
            side="left"
        )

        self.single_clone_btn_transcribe = ttk.Button(step1, text=get_text("btn_start_transcribe"), command=self._single_clone_run_transcribe)
        self.single_clone_btn_transcribe.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        step2 = ttk.Frame(content, padding=8)
        step2.columnconfigure(0, weight=1)
        step2.rowconfigure(1, weight=1)
        self.single_clone_step_frames.append(step2)

        cand_buttons = ttk.Frame(step2)
        cand_buttons.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(cand_buttons, text=get_text("lbl_candidate_limit")).pack(side="left", padx=(0, 6))
        self.single_clone_candidate_limit_var = tk.StringVar(value="12")
        ttk.Spinbox(cand_buttons, from_=1, to=50, textvariable=self.single_clone_candidate_limit_var, width=6).pack(
            side="left", padx=(0, 8)
        )

        self.single_clone_btn_collect_generate = ttk.Button(
            cand_buttons,
            text=get_text("btn_collect_generate_samples"),
            command=self._single_clone_collect_generate_samples,
        )
        self.single_clone_btn_collect_generate.pack(side="left", padx=(0, 4))

        self.single_clone_post_candidate_controls = ttk.Frame(cand_buttons)
        self.single_clone_post_candidate_controls.pack(side="left")
        self.single_clone_btn_use_candidate = ttk.Button(
            self.single_clone_post_candidate_controls,
            text=get_text("btn_use_as_speaker1"),
            command=self._single_clone_use_selected_candidate,
        )
        self.single_clone_btn_use_candidate.grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.single_clone_btn_upload_design = ttk.Button(
            self.single_clone_post_candidate_controls,
            text=get_text("btn_upload_or_design_basis"),
            command=lambda: self._show_single_clone_step(2),
        )
        self.single_clone_btn_upload_design.grid(row=0, column=1, sticky="w")

        self.single_clone_tree_frame = ttk.Frame(step2)
        self.single_clone_tree_frame.grid(row=1, column=0, sticky="nsew")
        self.single_clone_tree_frame.columnconfigure(0, weight=1)
        self.single_clone_tree_frame.rowconfigure(0, weight=1)
        self.single_clone_candidate_panel = CandidateBasisPanel(
            self.single_clone_tree_frame,
            get_text=get_text,
            play_wav=lambda path, label: self._single_clone_play_wav(path, label),
            include_video=True,
            height=9,
        )
        self.single_clone_candidate_panel.grid(row=0, column=0, sticky="nsew")
        self.single_clone_candidate_tree = self.single_clone_candidate_panel.tree
        self._set_single_clone_candidate_actions_visible(False)

        step3 = ttk.Frame(content, padding=8)
        step3.columnconfigure(1, weight=1)
        self.single_clone_step_frames.append(step3)

        ttk.Label(step3, text=get_text("lbl_speaker1_wav")).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.single_clone_basis_wav_var = tk.StringVar()
        ttk.Entry(step3, textvariable=self.single_clone_basis_wav_var).grid(row=0, column=1, sticky="ew", pady=3)
        self.single_clone_btn_import_wav = ttk.Button(step3, text=get_text("btn_import_wav"), command=self._single_clone_import_wav)
        self.single_clone_btn_import_wav.grid(row=0, column=2, sticky="ew", padx=(6, 0), pady=3)
        self.single_clone_btn_play_wav = ttk.Button(
            step3,
            text=get_text("btn_play_inline"),
            command=self._single_clone_play_basis_wav,
        )
        self.single_clone_btn_play_wav.grid(row=0, column=3, sticky="ew", padx=(6, 0), pady=3)
        self.single_clone_btn_play_wav.grid_remove()
        self.single_clone_basis_wav_var.trace_add("write", lambda *_: self._update_single_clone_basis_play_button())

        ttk.Label(step3, text=get_text("lbl_speaker1_txt")).grid(row=1, column=0, sticky="nw", padx=(0, 6), pady=3)
        self.single_clone_basis_txt_path_var = tk.StringVar()
        self.single_clone_basis_text = tk.Text(step3, height=4, wrap="word")
        self.single_clone_basis_text.grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)
        ttk.Label(
            step3,
            text=get_text("note_speaker1_text_match"),
            foreground="dim gray",
            wraplength=620,
            justify="left",
        ).grid(row=2, column=1, columnspan=3, sticky="ew", pady=(0, 6))

        ttk.Label(step3, text=get_text("lbl_voice_design_instruct")).grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
        self.single_clone_btn_voice_design = ttk.Button(step3, text=get_text("btn_voice_design"), command=self._single_clone_voice_design)
        self.single_clone_btn_voice_design.grid(row=3, column=1, sticky="w", pady=3)

        ttk.Label(step3, text=get_text("lbl_basis_status")).grid(row=4, column=0, sticky="w", padx=(0, 6), pady=3)
        self.single_clone_basis_status_var = tk.StringVar()
        ttk.Label(step3, textvariable=self.single_clone_basis_status_var, foreground="dim gray").grid(
            row=4, column=1, columnspan=3, sticky="ew", pady=3
        )
        self.single_clone_btn_apply_basis = ttk.Button(step3, text=get_text("btn_use_as_speaker1"), command=self._single_clone_apply_basis)
        self.single_clone_btn_apply_basis.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        step4 = ttk.Frame(content, padding=8)
        step4.columnconfigure(0, weight=1)
        self.single_clone_step_frames.append(step4)
        step4_synthesis_options = ttk.Frame(step4)
        step4_synthesis_options.grid(row=1, column=0, sticky="ew", pady=(4, 6))
        ttk.Label(step4_synthesis_options, text=get_text("lbl_loudness_mode")).pack(side="left", padx=(0, 6))
        self.single_clone_loudness_mode_var = tk.StringVar(value=get_text("opt_loudness_envelope"))
        ttk.Combobox(
            step4_synthesis_options,
            textvariable=self.single_clone_loudness_mode_var,
            values=list(self.single_clone_loudness_mode_map.keys()),
            state="readonly",
            width=16,
        ).pack(side="left", padx=(0, 18))
        ttk.Label(step4_synthesis_options, text=get_text("lbl_envelope_strength")).pack(side="left", padx=(0, 6))
        self.single_clone_envelope_alpha_var = tk.StringVar(value=get_text("opt_envelope_strong"))
        ttk.Combobox(
            step4_synthesis_options,
            textvariable=self.single_clone_envelope_alpha_var,
            values=list(self.single_clone_envelope_alpha_map.keys()),
            state="readonly",
            width=10,
        ).pack(side="left")
        self.single_clone_tempo_fit_map = {
            get_text("opt_tempo_fit_off"): "off",
            get_text("opt_tempo_fit_moderate"): "moderate",
            get_text("opt_tempo_fit_strong"): "strong",
        }
        ttk.Label(step4_synthesis_options, text=get_text("lbl_tempo_fit")).pack(side="left", padx=(18, 6))
        self.single_clone_tempo_fit_var = tk.StringVar(value=get_text("opt_tempo_fit_moderate"))
        ttk.Combobox(
            step4_synthesis_options,
            textvariable=self.single_clone_tempo_fit_var,
            values=list(self.single_clone_tempo_fit_map.keys()),
            state="readonly",
            width=8,
        ).pack(side="left")

        step4_options = ttk.Frame(step4)
        step4_options.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        self.single_clone_skip_existing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            step4_options,
            text=get_text("chk_single_skip_existing_si"),
            variable=self.single_clone_skip_existing_var,
        ).pack(side="left")
        (
            self.single_clone_proofread_panel,
            self._refresh_single_clone_proofread_panel,
            self.single_clone_proofread_buttons,
        ) = build_proofread_panel(
            step4,
            app=self,
            get_videos=self._single_clone_scan_videos_silent,
            run_async=self._single_clone_run_async,
            log_widget=lambda: self.single_clone_log,
            show_speaker=False,
            get_target_language=self._selected_single_target_language,
            get_stop_event=lambda: self.single_clone_stop_event,
            get_source_correction=lambda: self.single_clone_source_correction_var.get(),
        )
        self.single_clone_proofread_panel.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        self.single_clone_btn_translate_clone = ttk.Button(
            step4,
            text=get_text("btn_start_single_clone"),
            command=self._single_clone_translate_clone,
        )
        self.single_clone_btn_translate_clone.grid(row=3, column=0, sticky="ew")

        log_frame = ttk.LabelFrame(frame, text=get_text("lbl_log"), padding=6)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        self.single_clone_log = tk.Text(log_frame, height=8, state="disabled")
        self.single_clone_log.pack(fill="both", expand=True)

        self.single_clone_action_buttons = [
            self.single_clone_btn_prev,
            self.single_clone_btn_next,
            self.single_clone_btn_browse,
            self.btn_download_single_asr_model,
            self.single_clone_btn_transcribe,
            self.single_clone_btn_collect_generate,
            self.single_clone_btn_use_candidate,
            self.single_clone_btn_upload_design,
            self.single_clone_btn_import_wav,
            self.single_clone_btn_play_wav,
            self.single_clone_btn_voice_design,
            self.single_clone_btn_apply_basis,
            self.single_clone_btn_trans_config,
            *self.single_clone_proofread_buttons,
            self.single_clone_btn_translate_clone,
        ]
        self._refresh_single_clone_target_label()
        self._refresh_single_asr_model_status()
        self._refresh_single_translate_config_status()
        self._show_single_clone_step(0)

    def _setup_multi_clone_tab(self, frame):
        self.multi_clone_stop_event = threading.Event()
        self.multi_clone_thread: threading.Thread | None = None
        self.multi_clone_video = ""
        self.multi_clone_videos: list[str] = []
        self.multi_clone_speakers: list[dict] = []
        self.multi_clone_skipped: set[str] = set()
        self.multi_clone_basis: dict[str, dict] = {}
        self.multi_clone_speaker_iid_to_id: dict[str, str] = {}
        self.multi_clone_step_index = 0

        self.multi_clone_lang_map = {
            get_text("opt_lang_auto"): None,
            get_text("opt_lang_ja"): "ja",
            get_text("opt_lang_en"): "en",
            get_text("opt_lang_zh"): "zh",
        }
        self.multi_clone_model_map = {
            "large-v3": "large-v3",
            "large-v2": "large-v2",
            get_text("opt_model_kotoba"): "kotoba",
        }
        self.multi_clone_denoise_map = {
            get_text("opt_denoise_none"): "none",
            get_text("opt_denoise_mild"): "mild",
            get_text("opt_denoise_balanced"): "balanced",
            get_text("opt_denoise_strong"): "strong",
        }
        self.multi_clone_num_map = {
            get_text("opt_num_auto"): None,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
        }
        self.multi_clone_tgt_map = {
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
        self.multi_clone_loudness_mode_map = {
            get_text("opt_loudness_flat"): "flat",
            get_text("opt_loudness_sentence"): "sentence",
            get_text("opt_loudness_envelope"): "envelope",
        }
        self.multi_clone_envelope_alpha_map = {
            get_text("opt_envelope_strong"): 0.6,
            get_text("opt_envelope_normal"): 0.3,
        }

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(3, weight=1)

        ttk.Label(
            frame,
            text=get_text("multi_clone_note"),
            wraplength=760,
            justify="left",
            foreground="dim gray",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        step_bar = ttk.Frame(frame)
        step_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        step_bar.columnconfigure(1, weight=1)
        self.multi_clone_step_title_var = tk.StringVar()
        ttk.Label(step_bar, textvariable=self.multi_clone_step_title_var, font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        nav = ttk.Frame(step_bar)
        nav.grid(row=0, column=2, sticky="e")
        self.multi_clone_btn_prev = ttk.Button(
            nav,
            text=get_text("btn_prev_step"),
            command=lambda: self._show_multi_clone_step(self.multi_clone_step_index - 1),
        )
        self.multi_clone_btn_prev.pack(side="left", padx=(0, 4))
        self.multi_clone_btn_next = ttk.Button(
            nav,
            text=get_text("btn_next_step"),
            command=lambda: self._show_multi_clone_step(self.multi_clone_step_index + 1),
        )
        self.multi_clone_btn_next.pack(side="left", padx=(0, 4))
        self.multi_clone_btn_stop = ttk.Button(nav, text=get_text("btn_stop"), command=self._stop_multi_clone, state="disabled")
        self.multi_clone_btn_stop.pack(side="left")

        content = ttk.Frame(frame)
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.multi_clone_step_frames: list[ttk.Frame] = []
        self.multi_clone_step_names = [
            get_text("multi_step_1_transcribe"),
            get_text("multi_step_2_basis"),
            get_text("multi_step_3_export"),
        ]

        step1 = ttk.Frame(content, padding=8)
        step1.columnconfigure(1, weight=1)
        self.multi_clone_step_frames.append(step1)
        label_width = 16
        secondary_width = 10

        mode_row = ttk.Frame(step1)
        mode_row.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        ttk.Label(mode_row, text=get_text("lbl_input_mode"), width=label_width).pack(side="left", padx=(0, 6))
        self.multi_clone_input_mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode_row,
            text=get_text("opt_single_file"),
            variable=self.multi_clone_input_mode_var,
            value="single",
            command=self._on_multi_clone_input_mode_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            mode_row,
            text=get_text("opt_shared_dir"),
            variable=self.multi_clone_input_mode_var,
            value="batch",
            command=self._on_multi_clone_input_mode_change,
        ).pack(side="left")

        self.multi_clone_input_label_var = tk.StringVar(value=get_text("lbl_input_video"))
        ttk.Label(step1, textvariable=self.multi_clone_input_label_var, width=label_width).grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.multi_clone_input_var = tk.StringVar()
        ttk.Entry(step1, textvariable=self.multi_clone_input_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)
        self.multi_clone_btn_browse = ttk.Button(step1, text=get_text("btn_browse"), command=self._browse_multi_clone_video)
        self.multi_clone_btn_browse.grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=3)

        lang_row = ttk.Frame(step1)
        lang_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Label(lang_row, text=get_text("lbl_src_lang"), width=label_width).pack(side="left", padx=(0, 6))
        self.multi_clone_src_lang_var = tk.StringVar(value=get_text("opt_lang_auto"))
        ttk.Combobox(
            lang_row,
            textvariable=self.multi_clone_src_lang_var,
            values=list(self.multi_clone_lang_map.keys()),
            state="readonly",
            width=14,
        ).pack(side="left", padx=(0, 16))
        ttk.Label(lang_row, text=get_text("lbl_denoise"), width=secondary_width).pack(side="left", padx=(0, 6))
        self.multi_clone_denoise_var = tk.StringVar(value=get_text("opt_denoise_mild"))
        ttk.Combobox(
            lang_row,
            textvariable=self.multi_clone_denoise_var,
            values=list(self.multi_clone_denoise_map.keys()),
            state="readonly",
            width=14,
        ).pack(side="left", padx=(0, 16))

        ttk.Label(lang_row, text=get_text("lbl_vad_sensitivity")).pack(side="left", padx=(0, 6))
        self.multi_clone_vad_var = tk.StringVar(value=get_text("opt_vad_high"))
        ttk.Combobox(
            lang_row, textvariable=self.multi_clone_vad_var,
            values=list(self.vad_sensitivity_map.keys()),
            state="readonly", width=14,
        ).pack(side="left")

        model_row = ttk.Frame(step1)
        model_row.grid(row=3, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Label(model_row, text=get_text("lbl_model"), width=label_width).pack(side="left", padx=(0, 6))
        self.multi_clone_model_var = tk.StringVar(value="large-v3")
        self.multi_clone_model_combo = ttk.Combobox(
            model_row,
            textvariable=self.multi_clone_model_var,
            values=list(self.multi_clone_model_map.keys()),
            state="readonly",
            width=16,
        )
        self.multi_clone_model_combo.pack(side="left")
        self.multi_clone_model_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_multi_asr_model_status())
        self.btn_download_multi_asr_model = ttk.Button(
            model_row,
            text=get_text("btn_download_model"),
            command=self._download_multi_asr_model,
        )
        self.btn_download_multi_asr_model.pack(side="left", padx=(6, 16))
        self.multi_clone_num_label = ttk.Label(model_row, text=get_text("lbl_num_speakers"), width=secondary_width)
        self.multi_clone_num_label.pack(side="left", padx=(0, 6))
        self.multi_clone_num_var = tk.StringVar(value=get_text("opt_num_auto"))
        self.multi_clone_num_combo = ttk.Combobox(
            model_row,
            textvariable=self.multi_clone_num_var,
            values=list(self.multi_clone_num_map.keys()),
            state="readonly",
            width=8,
        )
        self.multi_clone_num_combo.pack(side="left")
        self.multi_asr_model_status_var = tk.StringVar()
        self.multi_asr_model_status_label = ttk.Label(step1, textvariable=self.multi_asr_model_status_var, foreground="dim gray")
        self.multi_asr_model_status_label.grid(row=4, column=1, columnspan=3, sticky="ew", pady=(0, 3))

        self.multi_clone_source_correction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            step1,
            text=get_text("chk_source_correction"),
            variable=self.multi_clone_source_correction_var,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=3)

        target_row = ttk.Frame(step1)
        target_row.grid(row=6, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Label(target_row, text=get_text("lbl_single_clone_target_lang"), width=label_width).pack(side="left", padx=(0, 6))
        self.multi_clone_tgt_lang_var = tk.StringVar(value=get_text("opt_lang_zh"))
        ttk.Combobox(
            target_row,
            textvariable=self.multi_clone_tgt_lang_var,
            values=list(self.multi_clone_tgt_map.keys()),
            state="readonly",
            width=16,
        ).pack(side="left")
        self.multi_clone_btn_trans_config = ttk.Button(
            target_row,
            text=get_text("btn_trans_config"),
            command=self._open_multi_translate_config,
            width=16,
        )
        self.multi_clone_btn_trans_config.pack(side="left", padx=(8, 8))
        self.multi_translate_status_var = tk.StringVar()
        ttk.Label(target_row, textvariable=self.multi_translate_status_var, foreground="dim gray").pack(side="left")

        self.multi_clone_btn_transcribe = ttk.Button(
            step1,
            text=get_text("btn_start_multi_transcribe"),
            command=self._multi_clone_run_transcribe,
        )
        self.multi_clone_btn_transcribe.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        step2 = ttk.Frame(content, padding=8)
        step2.columnconfigure(0, weight=1)
        step2.rowconfigure(2, weight=1)
        self.multi_clone_step_frames.append(step2)

        speaker_controls = ttk.Frame(step2)
        speaker_controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.multi_clone_candidate_limit_var = tk.StringVar(value="12")
        self.multi_clone_btn_export_basis = ttk.Button(
            speaker_controls,
            text=get_text("btn_export_basis"),
            command=self._multi_clone_export_selected_basis,
        )
        self.multi_clone_btn_export_basis.pack(side="right", padx=(4, 0))
        ttk.Label(
            speaker_controls,
            text=get_text("multi_step2_hint"),
            foreground="dim gray",
        ).pack(side="left", fill="x", expand=True)

        speaker_actions = ttk.Frame(step2)
        speaker_actions.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.multi_clone_btn_select_basis = ttk.Button(
            speaker_actions, text=get_text("btn_select_basis"),
            command=lambda: self._multi_speaker_action(self._open_multi_candidate_dialog),
        )
        self.multi_clone_btn_select_basis.pack(side="left", padx=(0, 4))
        self.multi_clone_btn_import_basis = ttk.Button(
            speaker_actions, text=get_text("btn_import_wav"),
            command=lambda: self._multi_speaker_action(self._open_multi_import_basis_dialog),
        )
        self.multi_clone_btn_import_basis.pack(side="left", padx=(0, 4))
        self.multi_clone_btn_design_basis = ttk.Button(
            speaker_actions, text=get_text("btn_voice_design"),
            command=lambda: self._multi_speaker_action(self._open_multi_voice_design_dialog),
        )
        self.multi_clone_btn_design_basis.pack(side="left", padx=(0, 4))
        self.multi_clone_btn_play_basis = ttk.Button(
            speaker_actions, text=get_text("btn_play_selected_basis"),
            command=lambda: self._multi_speaker_action(self._multi_clone_play_selected_basis),
        )
        self.multi_clone_btn_play_basis.pack(side="left", padx=(0, 4))
        self.multi_clone_btn_toggle_skip = ttk.Button(
            speaker_actions, text=get_text("multi_skip_off"),
            command=lambda: self._multi_speaker_action(self._multi_clone_toggle_skip_speaker),
        )
        self.multi_clone_btn_toggle_skip.pack(side="left", padx=(0, 4))
        self.multi_clone_speaker_action_buttons = [
            self.multi_clone_btn_select_basis, self.multi_clone_btn_import_basis,
            self.multi_clone_btn_design_basis, self.multi_clone_btn_play_basis,
            self.multi_clone_btn_toggle_skip,
        ]

        speaker_frame = ttk.Frame(step2)
        speaker_frame.grid(row=2, column=0, sticky="nsew")
        speaker_frame.columnconfigure(0, weight=1)
        speaker_frame.rowconfigure(0, weight=1)
        columns = ("speaker", "dur", "segments", "status", "text")
        self.multi_clone_speaker_tree = ttk.Treeview(speaker_frame, columns=columns, show="headings", height=10)
        for col, width, anchor in (
            ("speaker", 110, "center"),
            ("dur", 90, "center"),
            ("segments", 80, "center"),
            ("status", 150, "w"),
            ("text", 340, "w"),
        ):
            self.multi_clone_speaker_tree.heading(col, text=get_text(f"multi_col_{col}"))
            self.multi_clone_speaker_tree.column(col, width=width, anchor=anchor, stretch=(col == "text"))
        self.multi_clone_speaker_tree.grid(row=0, column=0, sticky="nsew")
        self.multi_clone_speaker_tree.bind("<<TreeviewSelect>>", lambda _e: self._refresh_multi_speaker_action_buttons())
        speaker_scroll = ttk.Scrollbar(speaker_frame, orient="vertical", command=self.multi_clone_speaker_tree.yview)
        speaker_scroll.grid(row=0, column=1, sticky="ns")
        self.multi_clone_speaker_tree.configure(yscrollcommand=speaker_scroll.set)

        step3 = ttk.Frame(content, padding=8)
        step3.columnconfigure(0, weight=1)
        self.multi_clone_step_frames.append(step3)
        synthesis_options = ttk.Frame(step3)
        synthesis_options.grid(row=1, column=0, sticky="ew", pady=(4, 6))
        ttk.Label(synthesis_options, text=get_text("lbl_loudness_mode")).pack(side="left", padx=(0, 6))
        self.multi_clone_loudness_mode_var = tk.StringVar(value=get_text("opt_loudness_envelope"))
        ttk.Combobox(
            synthesis_options,
            textvariable=self.multi_clone_loudness_mode_var,
            values=list(self.multi_clone_loudness_mode_map.keys()),
            state="readonly",
            width=16,
        ).pack(side="left", padx=(0, 18))
        ttk.Label(synthesis_options, text=get_text("lbl_envelope_strength")).pack(side="left", padx=(0, 6))
        self.multi_clone_envelope_alpha_var = tk.StringVar(value=get_text("opt_envelope_strong"))
        ttk.Combobox(
            synthesis_options,
            textvariable=self.multi_clone_envelope_alpha_var,
            values=list(self.multi_clone_envelope_alpha_map.keys()),
            state="readonly",
            width=10,
        ).pack(side="left")
        self.multi_clone_tempo_fit_map = {
            get_text("opt_tempo_fit_off"): "off",
            get_text("opt_tempo_fit_moderate"): "moderate",
            get_text("opt_tempo_fit_strong"): "strong",
        }
        ttk.Label(synthesis_options, text=get_text("lbl_tempo_fit")).pack(side="left", padx=(18, 6))
        self.multi_clone_tempo_fit_var = tk.StringVar(value=get_text("opt_tempo_fit_moderate"))
        ttk.Combobox(
            synthesis_options,
            textvariable=self.multi_clone_tempo_fit_var,
            values=list(self.multi_clone_tempo_fit_map.keys()),
            state="readonly",
            width=8,
        ).pack(side="left")

        step3_options = ttk.Frame(step3)
        step3_options.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        self.multi_clone_skip_existing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            step3_options,
            text=get_text("chk_single_skip_existing_si"),
            variable=self.multi_clone_skip_existing_var,
        ).pack(side="left")
        (
            self.multi_clone_proofread_panel,
            self._refresh_multi_clone_proofread_panel,
            self.multi_clone_proofread_buttons,
        ) = build_proofread_panel(
            step3,
            app=self,
            get_videos=self._multi_clone_current_or_scanned_videos_silent,
            run_async=self._multi_clone_run_async,
            log_widget=lambda: self.multi_clone_log,
            show_speaker=True,
            get_target_language=self._selected_multi_target_language,
            get_stop_event=lambda: self.multi_clone_stop_event,
            get_source_correction=lambda: self.multi_clone_source_correction_var.get(),
        )
        self.multi_clone_proofread_panel.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        self.multi_clone_btn_start_export = ttk.Button(
            step3,
            text=get_text("btn_start_multi_clone"),
            command=self._multi_clone_translate_clone,
        )
        self.multi_clone_btn_start_export.grid(row=3, column=0, sticky="ew")

        log_frame = ttk.LabelFrame(frame, text=get_text("lbl_log"), padding=6)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        self.multi_clone_log = tk.Text(log_frame, height=8, state="disabled")
        self.multi_clone_log.pack(fill="both", expand=True)

        self.multi_clone_action_buttons = [
            self.multi_clone_btn_prev,
            self.multi_clone_btn_next,
            self.multi_clone_btn_browse,
            self.btn_download_multi_asr_model,
            self.multi_clone_btn_trans_config,
            self.multi_clone_btn_transcribe,
            self.multi_clone_btn_export_basis,
            *self.multi_clone_proofread_buttons,
            self.multi_clone_btn_start_export,
        ]
        self._refresh_multi_asr_model_status()
        self._refresh_multi_translate_config_status()
        self._refresh_multi_clone_speakers()
        self._show_multi_clone_step(0)

    def _setup_single_mix_tab(self, frame):
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

        self.mix_batch_dir_row = ttk.Frame(frame)
        self.mix_batch_dir_row.columnconfigure(1, weight=1)
        ttk.Label(self.mix_batch_dir_row, text=si("lbl_input_dir")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.mix_dir_var = tk.StringVar()
        ttk.Entry(self.mix_batch_dir_row, textvariable=self.mix_dir_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(self.mix_batch_dir_row, text=get_text("btn_browse"), command=self._browse_mix_dir).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        mode_frame = ttk.Frame(frame)
        mode_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 6))
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
        self._single_mix_duck_preset_map = {
            si("opt_duck_preset_normal"): "normal",
            si("opt_duck_preset_strong"): "strong",
            si("opt_duck_preset_strongest"): "strongest",
        }
        self.single_mix_channel_var = tk.StringVar(value=get_text("opt_channel_both"))
        self.single_mix_origvol_var = tk.StringVar(value="100%")
        self.single_mix_sivol_var = tk.StringVar(value="120%")
        self.single_mix_delay_var = tk.StringVar(value="0s")
        self.single_mix_duck_var = tk.BooleanVar(value=True)
        self.single_mix_duck_preset_label = ttk.Label(options, text=get_text("lbl_original_duck_preset"))
        self.single_mix_duck_preset_label.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        self.single_mix_duck_preset_var = tk.StringVar(value=si("opt_duck_preset_strong"))
        self.single_mix_duck_preset_combo = ttk.Combobox(
            options,
            textvariable=self.single_mix_duck_preset_var,
            values=list(self._single_mix_duck_preset_map),
            width=8,
            state="readonly",
        )
        self.single_mix_duck_preset_combo.grid(row=0, column=1, sticky="w", pady=2)
        self.single_mix_sivol_label = ttk.Label(options, text=get_text("lbl_dub_volume"))
        self.single_mix_sivol_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        self.single_mix_sivol_combo = ttk.Combobox(
            options,
            textvariable=self.single_mix_sivol_var,
            values=[f"{v}%" for v in (80, 90, 100, 110, 120, 130)],
            width=8,
            state="readonly",
        )
        self.single_mix_sivol_combo.grid(row=1, column=1, sticky="w", pady=2)
        self.single_mix_sivol_hint = ttk.Label(
            options, text=get_text("dub_volume_hint"), foreground="dim gray"
        )
        self.single_mix_sivol_hint.grid(row=1, column=2, columnspan=4, sticky="w", padx=(12, 0), pady=2)
        self.single_mix_duck_key_var = tk.BooleanVar(value=True)
        self.single_mix_duck_key_check = ttk.Checkbutton(
            options,
            text=get_text("chk_use_duck_key"),
            variable=self.single_mix_duck_key_var,
        )
        self.single_mix_duck_key_check.grid(row=2, column=0, columnspan=6, sticky="w", pady=2)
        self.single_mix_indep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text=si("chk_add_independent_track"), variable=self.single_mix_indep_var).grid(
            row=3, column=0, columnspan=6, sticky="w", pady=2
        )
        self.single_mix_si_option_widgets = [
            self.single_mix_duck_preset_label,
            self.single_mix_duck_preset_combo,
            self.single_mix_sivol_label,
            self.single_mix_sivol_combo,
            self.single_mix_sivol_hint,
            self.single_mix_duck_key_check,
        ]
        self._update_single_mix_duck_preset_state()

        btn_frame = ttk.Frame(frame)
        self._single_mix_btn_frame = btn_frame
        options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        btn_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(2, 6))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        self.single_mix_btn_start = ttk.Button(btn_frame, text=get_text("btn_start_mix"), command=self._run_single_mix)
        self.single_mix_btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.single_mix_btn_stop = ttk.Button(btn_frame, text=get_text("btn_stop"), command=self._stop_single_mix, state="disabled")
        self.single_mix_btn_stop.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        log_frame = ttk.LabelFrame(frame, text=get_text("lbl_log"), padding=6)
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(6, weight=1)
        self.single_mix_log = tk.Text(log_frame, height=10, state="disabled")
        self.single_mix_log.pack(fill="both", expand=True)
        self.mix_batch_dir_row.grid_remove()

    def _show_single_clone_step(self, index: int):
        index = max(0, min(index, len(self.single_clone_step_frames) - 1))
        if index == 1 and not self.single_clone_candidates:
            self._single_clone_load_existing_candidates()
        if index >= 2:
            self._single_clone_restore_basis_state()
        if self.single_clone_step_index == 2 and index > 2 and not self._single_clone_basis_text_value():
            messagebox.showerror("Error", get_text("err_speaker1_text_required"))
            return
        self.single_clone_step_index = index
        for frame in self.single_clone_step_frames:
            frame.grid_remove()
        self.single_clone_step_frames[index].grid(row=0, column=0, sticky="nsew")
        self.single_clone_step_title_var.set(f"{index + 1}/{len(self.single_clone_step_frames)} {self.single_clone_step_names[index]}")
        if index == len(self.single_clone_step_frames) - 1 and hasattr(self, "_refresh_single_clone_proofread_panel"):
            self._refresh_single_clone_proofread_panel()
        if getattr(self, "single_clone_busy", False):
            return
        self.single_clone_btn_prev.config(state="normal" if index > 0 else "disabled")
        self.single_clone_btn_next.config(state="normal" if index < len(self.single_clone_step_frames) - 1 else "disabled")
        self._refresh_single_asr_model_status()
        self._refresh_single_translate_config_status()

    def _set_single_clone_busy(self, busy: bool):
        self.single_clone_busy = busy
        state = "disabled" if busy else "normal"
        for button in getattr(self, "single_clone_action_buttons", []):
            button.config(state=state)
        self.single_clone_btn_stop.config(state="normal" if busy else "disabled")
        if not busy:
            self._show_single_clone_step(self.single_clone_step_index)

    def _on_single_clone_input_mode_change(self):
        if self.single_clone_input_mode_var.get() == "batch":
            self.single_clone_input_label_var.set(get_text("lbl_input_dir"))
        else:
            self.single_clone_input_label_var.set(get_text("lbl_input_video"))
        self.single_clone_input_var.set("")
        self.single_clone_videos = []
        self.single_clone_candidates = []
        self._refresh_single_clone_candidates()
        self._set_single_clone_candidate_actions_visible(False)

    def _browse_single_clone_input(self):
        if self.single_clone_input_mode_var.get() == "batch":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"), ("All files", "*.*")]
            )
        if path:
            self.single_clone_input_var.set(path)
            self.single_clone_videos = []
            self.single_clone_candidates = []
            self._refresh_single_clone_candidates()
            self._set_single_clone_candidate_actions_visible(False)

    def _selected_single_model_key(self) -> str:
        return self.single_clone_model_map.get(self.single_clone_model_var.get(), self.single_clone_model_var.get())

    def _single_asr_model_available(self) -> bool:
        from tool_clonevoice import whisperx_backend as wx

        return wx.check_model_files(self._selected_single_model_key(), self.models_root)

    def _refresh_single_asr_model_status(self):
        from tool_clonevoice import whisperx_backend as wx

        if not hasattr(self, "single_asr_model_status_var"):
            return
        model_key = self._selected_single_model_key()
        model_dir = wx.model_dir(model_key, self.models_root)
        if wx.check_model_files(model_key, self.models_root):
            self.single_asr_model_status_var.set("")
            if hasattr(self, "btn_download_single_asr_model"):
                self.btn_download_single_asr_model.grid_remove()
            if hasattr(self, "single_asr_model_status_label"):
                self.single_asr_model_status_label.grid_remove()
        else:
            self.single_asr_model_status_var.set(get_text("model_missing").format(model_dir))
            if hasattr(self, "single_asr_model_status_label"):
                self.single_asr_model_status_label.grid()
            if hasattr(self, "btn_download_single_asr_model"):
                self.btn_download_single_asr_model.grid()
                self.btn_download_single_asr_model.config(state="normal")

    def _download_single_asr_model(self, after_success=None):
        from tool_clonevoice import whisperx_backend as wx

        model_key = self._selected_single_model_key()
        model_label = self.single_clone_model_var.get()

        def task():
            self.root.after(0, lambda: self.btn_download_single_asr_model.config(state="disabled"))
            self.log(self.single_clone_log, get_text("msg_check_download_size").format(model_label))
            ok_to_download = self._query_and_confirm_download(
                model_label,
                lambda: wx.remote_file_plan(model_key, lambda m: self.log(self.single_clone_log, m)),
            )
            if not ok_to_download:
                self.log(self.single_clone_log, get_text("msg_download_cancelled"))
                self.root.after(0, lambda: self.btn_download_single_asr_model.config(state="normal"))
                return
            ok = wx.download_model(model_key, self.models_root, lambda m: self.log(self.single_clone_log, m))

            def finish():
                self._refresh_single_asr_model_status()
                if ok:
                    if after_success:
                        after_success()
                else:
                    self.btn_download_single_asr_model.config(state="normal")
                    self.log(self.single_clone_log, get_text("msg_download_failed"))

            self.root.after(0, finish)

        threading.Thread(target=task, daemon=True).start()

    def _selected_single_target_language(self) -> str:
        return self.single_clone_tgt_map.get(self.single_clone_tgt_lang_var.get(), "Chinese")

    def _refresh_single_clone_target_label(self):
        if hasattr(self, "single_clone_target_status_var"):
            self.single_clone_target_status_var.set(
                get_text("msg_single_clone_target_lang").format(self.single_clone_tgt_lang_var.get())
            )

    def _single_clone_candidate_limit(self) -> int:
        try:
            return max(1, int(self.single_clone_candidate_limit_var.get()))
        except ValueError:
            return 12

    def _single_clone_scan_videos(self) -> list[str]:
        from tool_clonevoice import single_clone as sc

        input_path = self.single_clone_input_var.get().strip()
        batch_mode = self.single_clone_input_mode_var.get() == "batch"
        if batch_mode:
            if not input_path or not os.path.isdir(input_path):
                messagebox.showerror("Error", get_text("err_no_dir"))
                return []
        elif not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", get_text("err_no_video"))
            return []
        videos = sc.scan_videos(input_path, batch=batch_mode)
        if batch_mode and not videos:
            messagebox.showerror("Error", get_text("err_no_batch_videos"))
            return []
        if not batch_mode and not videos:
            messagebox.showerror("Error", get_text("err_no_video"))
            return []
        self.single_clone_videos = videos
        return videos

    def _single_clone_scan_videos_silent(self) -> list[str]:
        if self.single_clone_videos:
            return list(self.single_clone_videos)
        from tool_clonevoice import single_clone as sc

        input_path = self.single_clone_input_var.get().strip()
        batch_mode = self.single_clone_input_mode_var.get() == "batch"
        if batch_mode:
            if not input_path or not os.path.isdir(input_path):
                return []
        elif not input_path or not os.path.isfile(input_path):
            return []
        videos = sc.scan_videos(input_path, batch=batch_mode)
        self.single_clone_videos = videos
        return videos

    def _single_clone_restore_basis_state(self, videos: list[str] | None = None) -> bool:
        if getattr(self, "single_clone_basis_applied", False):
            return True
        if (
            (self.single_clone_basis_wav_var.get() or "").strip()
            and self._single_clone_basis_text_value()
        ):
            return False
        videos = list(videos or self._single_clone_scan_videos_silent())
        if not videos:
            return False
        from tool_clonevoice import logic
        from tool_clonevoice import single_clone as sc

        first_wav = ""
        first_text = ""
        for video in videos:
            manifest = logic.load_manifest(video) or {}
            info = (manifest.get("speakers") or {}).get(sc.SPEAKER_ID) or {}
            ref_audio = info.get("ref_audio") or ""
            ref_text = (info.get("ref_text") or "").strip()
            if not ref_audio or not ref_text:
                return False
            ref_path = logic.clone_dir(video) / ref_audio
            if not ref_path.is_file():
                return False
            if not first_wav:
                first_wav = str(ref_path)
                first_text = ref_text
        self.single_clone_basis_wav_var.set(first_wav)
        self.single_clone_basis_txt_path_var.set("")
        self._single_clone_set_basis_text(first_text)
        self.single_clone_basis_source_kind = "existing_manifest"
        self.single_clone_basis_meta = {}
        self.single_clone_basis_applied = True
        self.single_clone_basis_status_var.set(get_text("msg_single_basis_applied").format(len(videos)))
        return True

    def _selected_single_candidate(self) -> dict | None:
        return self.single_clone_candidate_panel.selected_candidate()

    def _refresh_single_clone_candidates(self):
        if not hasattr(self, "single_clone_candidate_panel"):
            return
        self.single_clone_candidate_panel.set_candidates(self.single_clone_candidates)

    def _single_clone_load_existing_candidates(self) -> bool:
        from tool_clonevoice import single_clone as sc

        videos = self._single_clone_scan_videos_silent()
        if not videos:
            return False
        loaded = sc.load_existing_candidates_for_videos(
            videos,
            total=self._single_clone_candidate_limit(),
            log=lambda m: self.log(self.single_clone_log, m),
        )
        if not loaded:
            return False
        self.single_clone_candidates = loaded
        self._refresh_single_clone_candidates()
        self._set_single_clone_candidate_actions_visible(True)
        self.log(self.single_clone_log, get_text("msg_single_candidates_done").format(len(loaded)))
        return True

    def _set_single_clone_candidate_actions_visible(self, visible: bool):
        if not hasattr(self, "single_clone_btn_use_candidate"):
            return
        if visible:
            self.single_clone_btn_use_candidate.grid()
        else:
            self.single_clone_btn_use_candidate.grid_remove()

    def _single_clone_on_candidate_tree_click(self, event):
        region = self.single_clone_candidate_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.single_clone_candidate_tree.identify_row(event.y)
        col = self.single_clone_candidate_tree.identify_column(event.x)
        if not row_id:
            return
        self.single_clone_candidate_tree.selection_set(row_id)
        idx = self.single_clone_candidate_iid_to_index.get(row_id)
        if idx is None or idx >= len(self.single_clone_candidates):
            return
        cand = self.single_clone_candidates[idx]
        if col == "#1":
            self._single_clone_play_wav(cand.get("source_audio") or "", get_text("col_play_source"))
        elif col == "#2":
            self._single_clone_play_wav(cand.get("translated_audio") or "", get_text("col_play_translation"))
        elif col == "#3":
            self._single_clone_play_wav(cand.get("target_sample_audio") or "", get_text("col_play_sample"))

    def _single_clone_set_basis_text(self, text: str):
        self.single_clone_basis_text.delete("1.0", "end")
        self.single_clone_basis_text.insert("1.0", text or "")

    def _single_clone_basis_text_value(self) -> str:
        return self.single_clone_basis_text.get("1.0", "end").strip()

    def _update_single_clone_basis_play_button(self):
        if not hasattr(self, "single_clone_btn_play_wav"):
            return
        if (self.single_clone_basis_wav_var.get() or "").strip():
            self.single_clone_btn_play_wav.grid()
            state = "disabled" if getattr(self, "single_clone_busy", False) else "normal"
            self.single_clone_btn_play_wav.config(state=state)
        else:
            self.single_clone_btn_play_wav.grid_remove()

    def _single_clone_set_basis(self, wav_path: str, basis_text: str, *, source_kind: str, meta: dict | None = None, txt_path: str = ""):
        self.single_clone_basis_wav_var.set(wav_path)
        self.single_clone_basis_txt_path_var.set(txt_path)
        self._single_clone_set_basis_text(basis_text)
        self.single_clone_basis_source_kind = source_kind
        self.single_clone_basis_meta = dict(meta or {})
        self.single_clone_basis_applied = False
        self.single_clone_basis_status_var.set(get_text("msg_single_basis_ready").format(wav_path))
        self.log(self.single_clone_log, get_text("msg_single_basis_ready").format(wav_path))

    def _single_clone_play_basis_wav(self):
        label = get_text("lbl_speaker1_wav").rstrip("：: ")
        self._single_clone_play_wav(self.single_clone_basis_wav_var.get(), label)

    def _single_clone_play_wav(self, path: str, label: str):
        path = (path or "").strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", get_text("err_no_speaker1"))
            return
        try:
            import winsound

            winsound.PlaySound(None, 0)
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.log(self.single_clone_log, get_text("msg_single_playing_audio").format(label, path))
        except Exception as exc:
            self.log(self.single_clone_log, f"Error: {exc}")

    def _load_single_clone_omnivoice_model(self, holder: list):
        from tool_clonevoice import omnivoice_backend as ov

        model = ov.load_model(
            self.models_root,
            ov.resolve_device(),
            lambda m: self.log(self.single_clone_log, m),
        )
        holder.append(model)
        return model

    def _single_clone_run_async(self, worker, done=None):
        if self.single_clone_thread is not None and self.single_clone_thread.is_alive():
            return
        self.single_clone_stop_event.clear()
        self._set_single_clone_busy(True)

        def task():
            from tool_clonevoice import logic

            holder: list = []
            result = None
            success = False

            def release_holder_on_main_thread():
                cleaned = threading.Event()

                def cleanup_models():
                    logic.release_model_holder(holder)
                    cleaned.set()

                self.root.after(0, cleanup_models)
                cleaned.wait()

            _redir = redirect_stdio(self._make_log_emitter(self.single_clone_log))
            _redir.__enter__()
            try:
                result = worker(holder, release_holder_on_main_thread)
                success = True
            except RuntimeError as exc:
                if "Stopped by user" in str(exc):
                    self.log(self.single_clone_log, get_text("msg_stopped"))
                else:
                    self.log(self.single_clone_log, f"Error: {exc}")
            except Exception as exc:
                self.log(self.single_clone_log, f"Error: {exc}")
            finally:
                _redir.__exit__(None, None, None)

                def finish():
                    try:
                        logic.release_model_holder(holder)
                        if success and done is not None and not self.single_clone_stop_event.is_set():
                            try:
                                done(result)
                            except Exception as exc:
                                self.log(self.single_clone_log, f"Error: {exc}")
                    finally:
                        self._set_single_clone_busy(False)

                self.root.after(0, finish)

        self.single_clone_thread = threading.Thread(target=task, daemon=True)
        self.single_clone_thread.start()

    def _single_clone_run_transcribe(self):
        if not self._translation_api_configured():
            self._refresh_single_translate_config_status()
            messagebox.showerror("Error", get_text("err_no_translation_api_key"))
            return
        if not self._single_asr_model_available():
            self._download_single_asr_model(after_success=self._single_clone_run_transcribe)
            return
        videos = self._single_clone_scan_videos()
        if not videos:
            return
        target_language = self._selected_single_target_language()
        model_key = self._selected_single_model_key()
        language = self.single_clone_lang_map.get(self.single_clone_src_lang_var.get())
        denoise = self.single_clone_denoise_map.get(self.single_clone_denoise_var.get(), "none")
        source_correction = self.single_clone_source_correction_var.get()
        vad_sensitivity = self.vad_sensitivity_map.get(self.single_clone_vad_var.get(), "high")

        def worker(holder, release_holder):
            from tool_clonevoice import single_clone as sc

            self.log(self.single_clone_log, get_text("msg_single_transcribe_start").format(len(videos)))
            for index, video in enumerate(videos, 1):
                if self.single_clone_stop_event.is_set():
                    raise RuntimeError("Stopped by user.")
                self.log(self.single_clone_log, get_text("msg_batch_item").format(index, len(videos), video))
                sc.run_single_transcribe(
                    video,
                    model_key=model_key,
                    language=language,
                    target_language=target_language,
                    models_root=self.models_root,
                    denoise=denoise,
                    vad_sensitivity=vad_sensitivity,
                    log=lambda m: self.log(self.single_clone_log, m),
                    stop_event=self.single_clone_stop_event,
                    model_holder=holder,
                )
                release_holder()
            self.log(self.single_clone_log, get_text("msg_single_translate_after_transcribe"))
            sc.ensure_translated_for_videos(
                videos,
                target_language=target_language,
                source_correction=source_correction,
                log=lambda m: self.log(self.single_clone_log, m),
                stop_event=self.single_clone_stop_event,
            )
            return videos

        def done(result):
            self.single_clone_videos = list(result or [])
            self._show_single_clone_step(1)

        self._single_clone_run_async(worker, done)

    def _single_clone_collect_generate_samples(self):
        from tool_clonevoice import single_clone as sc

        videos = self.single_clone_videos or self._single_clone_scan_videos()
        if not videos:
            return
        display = self._single_clone_candidate_limit()
        pool = max(1, display * sc.CANDIDATE_POOL_FACTOR)
        per_video = pool if len(videos) == 1 else max(1, min(pool, (pool + len(videos) - 1) // len(videos) + 2))
        target_language = self._selected_single_target_language()
        existing_candidates = list(self.single_clone_candidates or [])

        def worker(holder, release_holder):
            from tool_clonevoice import single_clone as sc

            disk_existing = sc.load_all_existing_candidates_for_videos(
                videos,
                log=lambda m: self.log(self.single_clone_log, m),
            )
            candidates = sc.collect_candidates_with_existing_for_videos(
                videos,
                disk_existing or existing_candidates,
                per_video=per_video,
                total=pool,
                log=lambda m: self.log(self.single_clone_log, m),
            )
            if not candidates:
                self.log(self.single_clone_log, get_text("msg_single_no_candidates"))
                return candidates
            self.log(self.single_clone_log, get_text("msg_single_prepare_samples"))
            jobs = []
            missing_target = [
                cand for cand in candidates
                if not (
                    cand.get("target_sample_audio")
                    and os.path.isfile(cand.get("target_sample_audio"))
                    and (cand.get("target_sample_text") or "").strip()
                )
            ]
            if missing_target:
                model = self._load_single_clone_omnivoice_model(holder)
                try:
                    for idx, cand in enumerate(candidates, 1):
                        if cand not in missing_target:
                            continue
                        if self.single_clone_stop_event.is_set():
                            raise RuntimeError("Stopped by user.")
                        label = get_text("msg_single_candidate_label").format(idx, len(candidates), cand.get("id") or "")
                        self.log(self.single_clone_log, get_text("msg_single_candidate_generating").format(label))
                        jobs.append(sc.build_candidate_target_sample_job(
                            cand,
                            model=model,
                            target_language=target_language,
                            log_label=label,
                            log=lambda m: self.log(self.single_clone_log, m),
                            stop_event=self.single_clone_stop_event,
                        ))
                finally:
                    del model
                    release_holder()
            if jobs:
                sc.finish_candidate_target_sample_jobs(
                    jobs,
                    models_root=self.models_root,
                    log=lambda m: self.log(self.single_clone_log, m),
                )
            preview_missing = [
                cand for cand in candidates
                if (cand.get("tgt_text") or "").strip()
                and cand.get("target_sample_audio")
                and os.path.isfile(cand.get("target_sample_audio"))
                and not (cand.get("translated_audio") and os.path.isfile(cand.get("translated_audio")))
            ]
            if preview_missing:
                model = self._load_single_clone_omnivoice_model(holder)
                try:
                    sc.generate_candidate_translated_previews_with_model(
                        candidates,
                        model=model,
                        target_language=target_language,
                        label_func=lambda i, n, c: get_text("msg_single_candidate_label").format(i, n, c.get("id") or ""),
                        log=lambda m: self.log(self.single_clone_log, m),
                        stop_event=self.single_clone_stop_event,
                    )
                finally:
                    del model
                    release_holder()
            score_missing = [
                cand for cand in candidates
                if cand.get("ecapa_similarity") is None
                and cand.get("source_audio")
                and (
                    (cand.get("translated_audio") and os.path.isfile(cand.get("translated_audio")))
                    or (cand.get("target_sample_audio") and os.path.isfile(cand.get("target_sample_audio")))
                )
            ]
            if score_missing:
                sc.score_candidate_similarities(
                    score_missing,
                    models_root=self.models_root,
                    log=lambda m: self.log(self.single_clone_log, m),
                )
            candidates.sort(
                key=lambda c: float(c.get("ecapa_similarity") if c.get("ecapa_similarity") is not None else -999.0),
                reverse=True,
            )
            # Evaluated a 2x pool by similarity; show the best `display`.
            candidates = candidates[:display]
            for idx, cand in enumerate(candidates, 1):
                cand["global_rank"] = idx
            return candidates

        def done(result):
            self.single_clone_candidates = list(result or [])
            self._refresh_single_clone_candidates()
            self._set_single_clone_candidate_actions_visible(bool(self.single_clone_candidates))
            self.log(self.single_clone_log, get_text("msg_single_candidates_done").format(len(self.single_clone_candidates)))

        self._single_clone_run_async(worker, done)

    def _single_clone_basis_from_candidate(self, cand: dict):
        wav_path = cand.get("target_sample_audio") or ""
        text = cand.get("target_sample_text") or ""
        if not wav_path or not os.path.isfile(wav_path):
            raise FileNotFoundError("Target-language sample is missing.")
        meta = {
            "candidate_id": cand.get("id"),
            "candidate_video": cand.get("video"),
            "source_audio": cand.get("source_audio"),
            "source_score": cand.get("score"),
            "ecapa_similarity": cand.get("ecapa_similarity"),
        }
        self._single_clone_set_basis(wav_path, text, source_kind="candidate_target_sample", meta=meta)
        self._show_single_clone_step(2)

    def _single_clone_use_selected_candidate(self):
        cand = self._selected_single_candidate()
        if cand is None:
            messagebox.showerror("Error", get_text("err_no_candidate_selected"))
            return
        if cand.get("target_sample_audio") and os.path.isfile(cand.get("target_sample_audio")):
            self._single_clone_basis_from_candidate(cand)
            return
        target_language = self._selected_single_target_language()

        def worker(holder, release_holder):
            from tool_clonevoice import single_clone as sc

            model = self._load_single_clone_omnivoice_model(holder)
            try:
                job = sc.build_candidate_target_sample_job(
                    cand,
                    model=model,
                    target_language=target_language,
                    log_label=get_text("msg_single_candidate_label").format(1, 1, cand.get("id") or ""),
                    log=lambda m: self.log(self.single_clone_log, m),
                    stop_event=self.single_clone_stop_event,
                )
            finally:
                del model
                release_holder()
            result = sc.finish_candidate_target_sample_jobs(
                [job],
                models_root=self.models_root,
                log=lambda m: self.log(self.single_clone_log, m),
            )[0]
            model = self._load_single_clone_omnivoice_model(holder)
            try:
                sc.generate_candidate_translated_previews_with_model(
                    [result],
                    model=model,
                    target_language=target_language,
                    label_func=lambda i, n, c: get_text("msg_single_candidate_label").format(i, n, c.get("id") or ""),
                    log=lambda m: self.log(self.single_clone_log, m),
                    stop_event=self.single_clone_stop_event,
                )
            finally:
                del model
                release_holder()
            scored = sc.score_candidate_similarities(
                [result],
                models_root=self.models_root,
                log=lambda m: self.log(self.single_clone_log, m),
            )
            return scored[0] if scored else result

        def done(result):
            self._refresh_single_clone_candidates()
            self._single_clone_basis_from_candidate(result)

        self._single_clone_run_async(worker, done)

    def _single_clone_import_wav(self):
        path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        self.single_clone_basis_wav_var.set(path)
        self.single_clone_basis_source_kind = "user_import"
        self.single_clone_basis_meta = {"import_wav": path}
        self.single_clone_basis_applied = False

    def _single_clone_import_txt(self):
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8-sig").strip()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        self.single_clone_basis_txt_path_var.set(path)
        self._single_clone_set_basis_text(text)
        self.single_clone_basis_source_kind = self.single_clone_basis_source_kind or "user_import"
        self.single_clone_basis_meta.update({"import_txt": path})
        self.single_clone_basis_applied = False

    def _single_clone_voice_design_groups(self):
        return [
            (
                get_text("grp_voice_design_gender"),
                [("男 / male", "male"), ("女 / female", "female")],
            ),
            (
                get_text("grp_voice_design_age"),
                [
                    ("儿童 / child", "child"),
                    ("少年 / teenager", "teenager"),
                    ("青年 / young adult", "young adult"),
                    ("中年 / middle-aged", "middle-aged"),
                    ("老年 / elderly", "elderly"),
                ],
            ),
            (
                get_text("grp_voice_design_pitch"),
                [
                    ("极低音调 / very low pitch", "very low pitch"),
                    ("低音调 / low pitch", "low pitch"),
                    ("中音调 / moderate pitch", "moderate pitch"),
                    ("高音调 / high pitch", "high pitch"),
                    ("极高音调 / very high pitch", "very high pitch"),
                ],
            ),
            (
                get_text("grp_voice_design_style"),
                [("耳语 / whisper", "whisper")],
            ),
            (
                get_text("grp_voice_design_english_accent"),
                [
                    ("american accent", "american accent"),
                    ("british accent", "british accent"),
                    ("australian accent", "australian accent"),
                    ("canadian accent", "canadian accent"),
                    ("indian accent", "indian accent"),
                    ("chinese accent", "chinese accent"),
                    ("korean accent", "korean accent"),
                    ("japanese accent", "japanese accent"),
                    ("portuguese accent", "portuguese accent"),
                    ("russian accent", "russian accent"),
                ],
            ),
            (
                get_text("grp_voice_design_chinese_dialect"),
                [
                    ("河南话", "河南话"),
                    ("陕西话", "陕西话"),
                    ("四川话", "四川话"),
                    ("贵州话", "贵州话"),
                    ("云南话", "云南话"),
                    ("桂林话", "桂林话"),
                    ("济南话", "济南话"),
                    ("石家庄话", "石家庄话"),
                    ("甘肃话", "甘肃话"),
                    ("宁夏话", "宁夏话"),
                    ("青岛话", "青岛话"),
                    ("东北话", "东北话"),
                ],
            ),
        ]

    def _single_clone_voice_design(self):
        input_path = self.single_clone_input_var.get().strip()
        batch_mode = self.single_clone_input_mode_var.get() == "batch"
        videos = self._single_clone_scan_videos()
        if not videos:
            return
        target_language = self._selected_single_target_language()

        dlg = tk.Toplevel(self.root)
        dlg.title(get_text("dlg_voice_design_title"))
        dlg.transient(self.root)
        dlg.columnconfigure(0, weight=1)
        none_label = get_text("opt_voice_design_none")
        selections = []

        for row, (group_label, options) in enumerate(self._single_clone_voice_design_groups()):
            group = ttk.LabelFrame(dlg, text=group_label, padding=6)
            group.grid(row=row, column=0, sticky="ew", padx=10, pady=(8 if row == 0 else 2, 2))
            group.columnconfigure(0, weight=1)
            var = tk.StringVar(value=none_label)
            display_to_value = {none_label: ""}
            for display, value in options:
                display_to_value[display] = value
            ttk.Combobox(
                group,
                textvariable=var,
                values=list(display_to_value.keys()),
                state="readonly",
                width=34,
            ).grid(row=0, column=0, sticky="ew")
            selections.append((var, display_to_value))

        status_var = tk.StringVar()
        ttk.Label(dlg, textvariable=status_var, foreground="dim gray", wraplength=460).grid(
            row=len(selections), column=0, sticky="ew", padx=10, pady=(6, 2)
        )
        buttons = ttk.Frame(dlg)
        buttons.grid(row=len(selections) + 1, column=0, sticky="e", padx=10, pady=(8, 10))
        preview_state = {"wav": "", "text": "", "instruct": ""}

        def current_instruct() -> str:
            attrs = []
            for var, display_to_value in selections:
                value = display_to_value.get(var.get(), "")
                if value:
                    attrs.append(value)
            return ", ".join(attrs)

        def apply_preview():
            wav_path = preview_state.get("wav") or ""
            text = preview_state.get("text") or ""
            instruct = preview_state.get("instruct") or ""
            if not wav_path or not text:
                return False
            self._single_clone_set_basis(
                wav_path,
                text,
                source_kind="voice_design",
                meta={"instruct": instruct},
            )
            return True

        def generate_design(*, save_after: bool):
            instruct = current_instruct()
            if not instruct:
                messagebox.showerror("Error", get_text("err_voice_design_instruct"))
                return
            if (
                not save_after
                and preview_state.get("instruct") == instruct
                and os.path.isfile(preview_state.get("wav") or "")
            ):
                self._single_clone_play_wav(preview_state["wav"], get_text("btn_preview_voice_design"))
                return
            status_var.set(get_text("msg_voice_design_generating").format(instruct))

            def worker(holder, _release_holder):
                from tool_clonevoice import single_clone as sc

                model = self._load_single_clone_omnivoice_model(holder)
                try:
                    return sc.generate_voice_design_basis_with_model(
                        input_path,
                        batch=batch_mode,
                        model=model,
                        target_language=target_language,
                        instruct=instruct,
                        log=lambda m: self.log(self.single_clone_log, m),
                        stop_event=self.single_clone_stop_event,
                    )
                finally:
                    del model

            def done(result):
                wav_path, text = result
                preview_state.update({"wav": wav_path, "text": text, "instruct": instruct})
                if not dlg.winfo_exists():
                    return
                status_var.set(get_text("msg_voice_design_preview_ready").format(wav_path))
                if save_after:
                    if apply_preview() and dlg.winfo_exists():
                        dlg.destroy()
                else:
                    self._single_clone_play_wav(wav_path, get_text("btn_preview_voice_design"))

            self._single_clone_run_async(worker, done)

        def save_design():
            instruct = current_instruct()
            if not instruct:
                messagebox.showerror("Error", get_text("err_voice_design_instruct"))
                return
            if (
                preview_state.get("instruct") == instruct
                and os.path.isfile(preview_state.get("wav") or "")
                and preview_state.get("text")
            ):
                if apply_preview():
                    dlg.destroy()
            else:
                generate_design(save_after=True)

        ttk.Button(
            buttons,
            text=get_text("btn_preview_voice_design"),
            command=lambda: generate_design(save_after=False),
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            buttons,
            text=get_text("btn_save_use_voice_design"),
            command=save_design,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text=get_text("btn_cancel"), command=dlg.destroy).pack(side="left")
        dlg.grab_set()

    def _single_clone_apply_basis(self):
        videos = self.single_clone_videos or self._single_clone_scan_videos()
        if not videos:
            return
        wav_path = self.single_clone_basis_wav_var.get().strip()
        text = self._single_clone_basis_text_value()
        if not wav_path or not os.path.isfile(wav_path):
            messagebox.showerror("Error", get_text("err_no_speaker1"))
            return
        if not text:
            messagebox.showerror("Error", get_text("err_speaker1_text_required"))
            return
        input_path = self.single_clone_input_var.get().strip()
        batch_mode = self.single_clone_input_mode_var.get() == "batch"
        target_language = self._selected_single_target_language()
        source_kind = self.single_clone_basis_source_kind or "user_import"
        meta = dict(self.single_clone_basis_meta)

        def worker(_holder, _release_holder):
            from tool_clonevoice import single_clone as sc

            return sc.save_speaker1_basis(
                videos,
                basis_wav=wav_path,
                basis_text=text,
                target_language=target_language,
                visible_target=input_path,
                batch=batch_mode,
                source_kind=source_kind,
                meta=meta,
                log=lambda m: self.log(self.single_clone_log, m),
            )

        def done(result):
            visible_wav, visible_txt = result
            self.single_clone_basis_wav_var.set(visible_wav)
            self.single_clone_basis_txt_path_var.set(visible_txt)
            self.single_clone_basis_applied = True
            self.single_clone_basis_status_var.set(get_text("msg_single_basis_applied").format(len(videos)))
            self.log(self.single_clone_log, get_text("msg_single_basis_applied").format(len(videos)))
            self._show_single_clone_step(3)

        self._single_clone_run_async(worker, done)

    def _single_clone_translate_clone(self):
        videos = self.single_clone_videos or self._single_clone_scan_videos()
        if not videos:
            return
        self._single_clone_restore_basis_state(videos)
        if not getattr(self, "single_clone_basis_applied", False):
            messagebox.showerror("Error", get_text("err_no_speaker1"))
            return
        if not self._translation_api_configured():
            self._refresh_single_translate_config_status()
            messagebox.showerror("Error", get_text("err_no_translation_api_key"))
            return
        target_language = self._selected_single_target_language()
        loudness_mode = self.single_clone_loudness_mode_map.get(self.single_clone_loudness_mode_var.get(), "envelope")
        envelope_alpha = self.single_clone_envelope_alpha_map.get(self.single_clone_envelope_alpha_var.get(), 0.6)
        tempo_fit = self.single_clone_tempo_fit_map.get(self.single_clone_tempo_fit_var.get(), "moderate")
        skip_existing = self.single_clone_skip_existing_var.get()
        source_correction = self.single_clone_source_correction_var.get()

        def worker(holder, _release_holder):
            from tool_clonevoice import single_clone as sc

            return sc.translate_and_synthesize(
                videos,
                target_language=target_language,
                models_root=self.models_root,
                source_correction=source_correction,
                loudness_mode=loudness_mode,
                envelope_alpha=envelope_alpha,
                tempo_fit=tempo_fit,
                skip_existing=skip_existing,
                log=lambda m: self.log(self.single_clone_log, m),
                stop_event=self.single_clone_stop_event,
                model_holder=holder,
            )

        def done(result):
            result = result or {}
            written = len(result.get("written") or [])
            skipped = len(result.get("skipped") or [])
            if written == 0 and skipped:
                self.log(self.single_clone_log, get_text("msg_single_all_skipped").format(skipped))
            else:
                self.log(self.single_clone_log, get_text("msg_single_done").format(written, skipped))

        self._single_clone_run_async(worker, done)

    def _stop_single_clone(self):
        self.single_clone_stop_event.set()
        self.single_clone_btn_stop.config(state="disabled")

    def _refresh_single_translate_config_status(self):
        if not hasattr(self, "single_translate_status_var"):
            return
        configured = self._translation_api_configured()
        self.single_translate_status_var.set(
            get_text("msg_translation_api_configured")
            if configured
            else get_text("msg_translation_api_not_configured")
        )
        if hasattr(self, "single_clone_btn_translate_clone") and not getattr(self, "single_clone_busy", False):
            self.single_clone_btn_translate_clone.config(state="normal" if configured else "disabled")
        if hasattr(self, "single_clone_btn_transcribe") and not getattr(self, "single_clone_busy", False):
            self.single_clone_btn_transcribe.config(state="normal" if configured else "disabled")

    def _open_single_translate_config(self):
        dlg = self._open_translate_config()
        if dlg is not None:
            dlg.bind(
                "<Destroy>",
                lambda event: self.root.after(0, self._refresh_single_translate_config_status)
                if event.widget is dlg else None,
                add="+",
            )

    def _show_multi_clone_step(self, index: int):
        index = max(0, min(index, len(self.multi_clone_step_frames) - 1))
        if index >= 1:
            self._multi_clone_restore_manifest_state()
        if index > 1 and not self._multi_clone_ready_for_export():
            messagebox.showerror("Error", get_text("err_not_all_speakers_ready"))
            return
        self.multi_clone_step_index = index
        for frame in self.multi_clone_step_frames:
            frame.grid_remove()
        self.multi_clone_step_frames[index].grid(row=0, column=0, sticky="nsew")
        self.multi_clone_step_title_var.set(f"{index + 1}/{len(self.multi_clone_step_frames)} {self.multi_clone_step_names[index]}")
        if index == len(self.multi_clone_step_frames) - 1 and hasattr(self, "_refresh_multi_clone_proofread_panel"):
            self._refresh_multi_clone_proofread_panel()
        if getattr(self, "multi_clone_busy", False):
            return
        self.multi_clone_btn_prev.config(state="normal" if index > 0 else "disabled")
        next_state = "normal" if index < len(self.multi_clone_step_frames) - 1 else "disabled"
        if index == 1 and not self._multi_clone_ready_for_export():
            next_state = "disabled"
        self.multi_clone_btn_next.config(state=next_state)
        self._refresh_multi_asr_model_status()
        self._refresh_multi_translate_config_status()

    def _set_multi_clone_busy(self, busy: bool):
        self.multi_clone_busy = busy
        state = "disabled" if busy else "normal"
        for button in getattr(self, "multi_clone_action_buttons", []):
            button.config(state=state)
        self.multi_clone_btn_stop.config(state="normal" if busy else "disabled")
        self._refresh_multi_speaker_action_buttons()
        if not busy:
            self._show_multi_clone_step(self.multi_clone_step_index)

    def _on_multi_clone_input_mode_change(self):
        if self.multi_clone_input_mode_var.get() == "batch":
            self.multi_clone_input_label_var.set(get_text("lbl_input_dir"))
            if hasattr(self, "multi_clone_num_label"):
                self.multi_clone_num_label.config(text=get_text("lbl_global_num_speakers"))
        else:
            self.multi_clone_input_label_var.set(get_text("lbl_input_video"))
            if hasattr(self, "multi_clone_num_label"):
                self.multi_clone_num_label.config(text=get_text("lbl_num_speakers"))
        self.multi_clone_input_var.set("")
        self.multi_clone_video = ""
        self.multi_clone_videos = []
        self.multi_clone_speakers = []
        self.multi_clone_skipped = set()
        self.multi_clone_basis = {}
        self._refresh_multi_clone_speakers()

    def _browse_multi_clone_video(self):
        if self.multi_clone_input_mode_var.get() == "batch":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"), ("All files", "*.*")]
            )
        if path:
            self.multi_clone_input_var.set(path)
            self.multi_clone_video = ""
            self.multi_clone_videos = []
            self.multi_clone_speakers = []
            self.multi_clone_skipped = set()
            self.multi_clone_basis = {}
            self._refresh_multi_clone_speakers()

    def _selected_multi_model_key(self) -> str:
        return self.multi_clone_model_map.get(self.multi_clone_model_var.get(), self.multi_clone_model_var.get())

    def _multi_asr_model_available(self) -> bool:
        from tool_clonevoice import whisperx_backend as wx

        return wx.check_model_files(self._selected_multi_model_key(), self.models_root)

    def _refresh_multi_asr_model_status(self):
        from tool_clonevoice import whisperx_backend as wx

        if not hasattr(self, "multi_asr_model_status_var"):
            return
        model_key = self._selected_multi_model_key()
        model_dir = wx.model_dir(model_key, self.models_root)
        if wx.check_model_files(model_key, self.models_root):
            self.multi_asr_model_status_var.set("")
            if hasattr(self, "btn_download_multi_asr_model"):
                self.btn_download_multi_asr_model.pack_forget()
            if hasattr(self, "multi_asr_model_status_label"):
                self.multi_asr_model_status_label.grid_remove()
        else:
            self.multi_asr_model_status_var.set(get_text("model_missing").format(model_dir))
            if hasattr(self, "multi_asr_model_status_label"):
                self.multi_asr_model_status_label.grid()
            if hasattr(self, "btn_download_multi_asr_model"):
                if not self.btn_download_multi_asr_model.winfo_manager():
                    self.btn_download_multi_asr_model.pack(
                        side="left",
                        padx=(6, 16),
                        before=getattr(self, "multi_clone_num_label", None),
                    )
                self.btn_download_multi_asr_model.config(state="normal")

    def _download_multi_asr_model(self, after_success=None):
        from tool_clonevoice import whisperx_backend as wx

        model_key = self._selected_multi_model_key()
        model_label = self.multi_clone_model_var.get()

        def task():
            self.root.after(0, lambda: self.btn_download_multi_asr_model.config(state="disabled"))
            self.log(self.multi_clone_log, get_text("msg_check_download_size").format(model_label))
            ok_to_download = self._query_and_confirm_download(
                model_label,
                lambda: wx.remote_file_plan(model_key, lambda m: self.log(self.multi_clone_log, m)),
            )
            if not ok_to_download:
                self.log(self.multi_clone_log, get_text("msg_download_cancelled"))
                self.root.after(0, lambda: self.btn_download_multi_asr_model.config(state="normal"))
                return
            ok = wx.download_model(model_key, self.models_root, lambda m: self.log(self.multi_clone_log, m))

            def finish():
                self._refresh_multi_asr_model_status()
                if ok:
                    if after_success:
                        after_success()
                else:
                    self.btn_download_multi_asr_model.config(state="normal")
                    self.log(self.multi_clone_log, get_text("msg_download_failed"))

            self.root.after(0, finish)

        threading.Thread(target=task, daemon=True).start()

    def _selected_multi_target_language(self) -> str:
        return self.multi_clone_tgt_map.get(self.multi_clone_tgt_lang_var.get(), "Chinese")

    def _multi_clone_candidate_limit(self) -> int:
        try:
            return max(1, int(self.multi_clone_candidate_limit_var.get()))
        except ValueError:
            return 12

    def _multi_clone_scan_videos(self) -> list[str]:
        from tool_clonevoice import single_clone as sc

        input_path = self.multi_clone_input_var.get().strip()
        batch_mode = self.multi_clone_input_mode_var.get() == "batch"
        if batch_mode:
            if not input_path or not os.path.isdir(input_path):
                messagebox.showerror("Error", get_text("err_no_dir"))
                return []
        elif not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", get_text("err_no_video"))
            return []
        videos = sc.scan_videos(input_path, batch=batch_mode)
        if batch_mode and not videos:
            messagebox.showerror("Error", get_text("err_no_batch_videos"))
            return []
        if not batch_mode and not videos:
            messagebox.showerror("Error", get_text("err_no_video"))
            return []
        self.multi_clone_videos = videos
        self.multi_clone_video = videos[0] if videos else ""
        return videos

    def _multi_clone_current_videos(self) -> list[str]:
        if self.multi_clone_videos:
            return list(self.multi_clone_videos)
        if self.multi_clone_video:
            return [self.multi_clone_video]
        return []

    def _multi_clone_current_or_scanned_videos_silent(self) -> list[str]:
        videos = self._multi_clone_current_videos()
        if videos:
            return videos
        from tool_clonevoice import single_clone as sc

        input_path = self.multi_clone_input_var.get().strip()
        batch_mode = self.multi_clone_input_mode_var.get() == "batch"
        if batch_mode:
            if not input_path or not os.path.isdir(input_path):
                return []
        elif not input_path or not os.path.isfile(input_path):
            return []
        videos = sc.scan_videos(input_path, batch=batch_mode)
        self.multi_clone_videos = videos
        self.multi_clone_video = videos[0] if videos else ""
        return videos

    def _multi_clone_restore_manifest_state(self) -> bool:
        videos = self._multi_clone_current_or_scanned_videos_silent()
        if not videos:
            return False
        from tool_clonevoice import multi_clone as mc

        speakers = mc.list_global_speakers(videos)
        if not speakers:
            return False
        if not self.multi_clone_speakers:
            self.multi_clone_speakers = speakers
        self._multi_clone_load_manifest_state()
        self._refresh_multi_clone_speakers()
        return True

    def _multi_clone_basis_source_video(self, speaker: str) -> str:
        videos = self._multi_clone_current_videos()
        if not videos:
            return ""
        try:
            from tool_clonevoice import multi_clone as mc

            matched = mc.videos_with_speaker(videos, speaker)
            return matched[0] if matched else videos[0]
        except Exception:
            return videos[0]

    def _refresh_multi_translate_config_status(self):
        if not hasattr(self, "multi_translate_status_var"):
            return
        configured = self._translation_api_configured()
        self.multi_translate_status_var.set(
            get_text("msg_translation_api_configured")
            if configured
            else get_text("msg_translation_api_not_configured")
        )
        if hasattr(self, "multi_clone_btn_transcribe") and not getattr(self, "multi_clone_busy", False):
            self.multi_clone_btn_transcribe.config(state="normal" if configured else "disabled")
        if hasattr(self, "multi_clone_btn_start_export") and not getattr(self, "multi_clone_busy", False):
            self.multi_clone_btn_start_export.config(state="normal" if configured else "disabled")

    def _open_multi_translate_config(self):
        dlg = self._open_translate_config()
        if dlg is not None:
            dlg.bind(
                "<Destroy>",
                lambda event: self.root.after(0, self._refresh_multi_translate_config_status)
                if event.widget is dlg else None,
                add="+",
            )

    def _load_multi_clone_omnivoice_model(self, holder: list):
        from tool_clonevoice import omnivoice_backend as ov

        model = ov.load_model(
            self.models_root,
            ov.resolve_device(),
            lambda m: self.log(self.multi_clone_log, m),
        )
        holder.append(model)
        return model

    def _multi_clone_run_async(self, worker, done=None):
        if self.multi_clone_thread is not None and self.multi_clone_thread.is_alive():
            return
        self.multi_clone_stop_event.clear()
        self._set_multi_clone_busy(True)

        def task():
            from tool_clonevoice import logic

            holder: list = []
            result = None
            success = False

            def release_holder_on_main_thread():
                cleaned = threading.Event()

                def cleanup_models():
                    logic.release_model_holder(holder)
                    cleaned.set()

                self.root.after(0, cleanup_models)
                cleaned.wait()

            _redir = redirect_stdio(self._make_log_emitter(self.multi_clone_log))
            _redir.__enter__()
            try:
                result = worker(holder, release_holder_on_main_thread)
                success = True
            except RuntimeError as exc:
                if "Stopped by user" in str(exc):
                    self.log(self.multi_clone_log, get_text("msg_stopped"))
                else:
                    self.log(self.multi_clone_log, f"Error: {exc}")
            except Exception as exc:
                self.log(self.multi_clone_log, f"Error: {exc}")
            finally:
                _redir.__exit__(None, None, None)

                def finish():
                    try:
                        logic.release_model_holder(holder)
                        if success and done is not None and not self.multi_clone_stop_event.is_set():
                            try:
                                done(result)
                            except Exception as exc:
                                self.log(self.multi_clone_log, f"Error: {exc}")
                    finally:
                        self._set_multi_clone_busy(False)

                self.root.after(0, finish)

        self.multi_clone_thread = threading.Thread(target=task, daemon=True)
        self.multi_clone_thread.start()

    def _multi_clone_run_transcribe(self):
        if not self._translation_api_configured():
            self._refresh_multi_translate_config_status()
            messagebox.showerror("Error", get_text("err_no_translation_api_key"))
            return
        num_speaker_label = self.multi_clone_num_var.get()
        if num_speaker_label not in self.multi_clone_num_map:
            messagebox.showerror("Error", get_text("err_multi_num_speakers_required"))
            return
        num_speakers = self.multi_clone_num_map.get(num_speaker_label)
        if not self._multi_asr_model_available():
            self._download_multi_asr_model(after_success=self._multi_clone_run_transcribe)
            return
        videos = self._multi_clone_scan_videos()
        if not videos:
            return
        batch_mode = self.multi_clone_input_mode_var.get() == "batch"
        if batch_mode:
            from tool_clonevoice import multi_clone as mc

            total_seconds = mc.estimate_total_video_duration(
                videos,
                log=lambda m: self.log(self.multi_clone_log, m),
            )
            if total_seconds > mc.GLOBAL_DIARIZE_WARN_SECONDS:
                total_minutes = total_seconds / 60.0
                if not messagebox.askyesno(
                    "Warning",
                    get_text("warn_multi_batch_total_duration").format(f"{total_minutes:.1f}"),
                ):
                    return
        target_language = self._selected_multi_target_language()
        model_key = self._selected_multi_model_key()
        language = self.multi_clone_lang_map.get(self.multi_clone_src_lang_var.get())
        denoise = self.multi_clone_denoise_map.get(self.multi_clone_denoise_var.get(), "none")
        source_correction = self.multi_clone_source_correction_var.get()
        vad_sensitivity = self.vad_sensitivity_map.get(self.multi_clone_vad_var.get(), "high")
        diarize_backend = "pyannote"

        def worker(holder, release_holder):
            from tool_clonevoice import multi_clone as mc
            from tool_clonevoice import single_clone as sc

            turns_by_video = {}
            if batch_mode:
                speaker_note = num_speakers if num_speakers is not None else get_text("opt_num_auto")
                self.log(self.multi_clone_log, get_text("msg_multi_global_prescan_start").format(len(videos), speaker_note))
                turns_by_video = mc.prescan_global_diarize(
                    videos,
                    models_root=self.models_root,
                    diarize_backend=diarize_backend,
                    num_speakers=num_speakers,
                    denoise=denoise,
                    log=lambda m: self.log(self.multi_clone_log, m),
                    stop_event=self.multi_clone_stop_event,
                )
            self.log(self.multi_clone_log, get_text("msg_multi_transcribe_start").format(len(videos)))
            for index, video in enumerate(videos, 1):
                if self.multi_clone_stop_event.is_set():
                    raise RuntimeError("Stopped by user.")
                self.log(self.multi_clone_log, get_text("msg_batch_item").format(index, len(videos), video))
                mc.run_multi_transcribe(
                    video,
                    model_key=model_key,
                    language=language,
                    target_language=target_language,
                    models_root=self.models_root,
                    diarize_backend=diarize_backend,
                    num_speakers=num_speakers,
                    denoise=denoise,
                    vad_sensitivity=vad_sensitivity,
                    precomputed_turns=turns_by_video.get(str(Path(video))) if batch_mode else None,
                    log=lambda m: self.log(self.multi_clone_log, m),
                    stop_event=self.multi_clone_stop_event,
                    model_holder=holder,
                )
                release_holder()
            self.log(self.multi_clone_log, get_text("msg_multi_translate_after_transcribe"))
            sc.ensure_translated_for_videos(
                videos,
                target_language=target_language,
                source_correction=source_correction,
                log=lambda m: self.log(self.multi_clone_log, m),
                stop_event=self.multi_clone_stop_event,
            )
            speakers = mc.list_global_speakers(videos)
            if not speakers:
                raise RuntimeError(get_text("err_no_multi_speakers"))
            return {"videos": videos, "speakers": speakers}

        def done(result):
            result = result or {}
            self.multi_clone_videos = list(result.get("videos") or videos)
            self.multi_clone_video = self.multi_clone_videos[0] if self.multi_clone_videos else ""
            self.multi_clone_speakers = list(result.get("speakers") or [])
            self._multi_clone_load_manifest_state()
            self._refresh_multi_clone_speakers()
            self.log(self.multi_clone_log, get_text("msg_multi_speakers_ready").format(len(self.multi_clone_speakers)))
            self._show_multi_clone_step(1)

        self._multi_clone_run_async(worker, done)

    def _multi_clone_load_manifest_state(self):
        videos = self._multi_clone_current_videos()
        if not videos:
            return
        from tool_clonevoice import logic

        self.multi_clone_basis = {}
        self.multi_clone_skipped = set()
        for video in videos:
            manifest = logic.load_manifest(video) or {}
            for spk, info in (manifest.get("speakers") or {}).items():
                if (info or {}).get("skip_synthesis"):
                    self.multi_clone_skipped.add(str(spk))
                ref_audio = (info or {}).get("ref_audio") or ""
                ref_text = (info or {}).get("ref_text") or ""
                if ref_audio and ref_text and str(spk) not in self.multi_clone_basis:
                    path = logic.clone_dir(video) / ref_audio
                    if path.is_file():
                        self.multi_clone_basis[str(spk)] = {
                            "wav": str(path),
                            "text": ref_text,
                            "source": (info or {}).get("source") or (info or {}).get("ref_kind") or "",
                        }

    def _multi_clone_ready_for_export(self) -> bool:
        videos = self._multi_clone_current_videos()
        if not videos or not self.multi_clone_speakers:
            return False
        from tool_clonevoice import multi_clone as mc

        return mc.all_videos_have_basis(videos, skipped=self.multi_clone_skipped)

    def _multi_clone_all_speakers_skipped(self) -> bool:
        speaker_ids = {str(item.get("speaker") or "") for item in self.multi_clone_speakers}
        speaker_ids.discard("")
        return bool(speaker_ids) and speaker_ids.issubset(self.multi_clone_skipped)

    def _multi_clone_speaker_basis_info(self, speaker: str) -> dict:
        if speaker in self.multi_clone_basis:
            return self.multi_clone_basis[speaker]
        videos = self._multi_clone_current_videos()
        if not videos:
            return {}
        from tool_clonevoice import logic

        for video in videos:
            manifest = logic.load_manifest(video) or {}
            info = (manifest.get("speakers") or {}).get(speaker) or {}
            ref_audio = info.get("ref_audio") or ""
            ref_text = info.get("ref_text") or ""
            if ref_audio and ref_text:
                path = logic.clone_dir(video) / ref_audio
                if path.is_file():
                    return {
                        "wav": str(path),
                        "text": ref_text,
                        "source": info.get("source") or info.get("ref_kind") or "",
                    }
        return {}

    def _refresh_multi_clone_speakers(self):
        if not hasattr(self, "multi_clone_speaker_tree"):
            return
        prev_speaker = self._selected_multi_speaker_id()
        self.multi_clone_speaker_tree.delete(*self.multi_clone_speaker_tree.get_children())
        self.multi_clone_speaker_iid_to_id = {}
        reselect_iid = ""
        for idx, item in enumerate(self.multi_clone_speakers):
            speaker = str(item.get("speaker") or "")
            iid = str(idx)
            self.multi_clone_speaker_iid_to_id[iid] = speaker
            if speaker and speaker == prev_speaker:
                reselect_iid = iid
            skipped = speaker in self.multi_clone_skipped
            basis = self._multi_clone_speaker_basis_info(speaker)
            if skipped:
                status = get_text("multi_status_skipped")
                text = ""
            elif basis:
                status = get_text("multi_status_basis_set")
                text = (basis.get("text") or "")[:80]
            else:
                status = get_text("multi_status_no_basis")
                text = ""
            self.multi_clone_speaker_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    speaker,
                    f"{float(item.get('total_dur') or 0.0):.1f}s",
                    str(int(item.get("seg_count") or 0)),
                    status,
                    text,
                ),
            )
        if reselect_iid:
            self.multi_clone_speaker_tree.selection_set(reselect_iid)
        self._refresh_multi_speaker_action_buttons()
        if hasattr(self, "multi_clone_btn_next") and not getattr(self, "multi_clone_busy", False):
            next_state = "normal" if self.multi_clone_step_index < len(self.multi_clone_step_frames) - 1 else "disabled"
            if self.multi_clone_step_index == 1 and not self._multi_clone_ready_for_export():
                next_state = "disabled"
            self.multi_clone_btn_next.config(state=next_state)

    def _refresh_multi_speaker_action_buttons(self):
        if not hasattr(self, "multi_clone_speaker_action_buttons"):
            return
        speaker = self._selected_multi_speaker_id()
        busy = bool(getattr(self, "multi_clone_busy", False))
        enabled = bool(speaker) and not busy
        for button in self.multi_clone_speaker_action_buttons:
            button.config(state="normal" if enabled else "disabled")
        if speaker:
            skipped = speaker in self.multi_clone_skipped
            self.multi_clone_btn_toggle_skip.config(
                text=get_text("multi_skip_on") if skipped else get_text("multi_skip_off")
            )
            basis = self._multi_clone_speaker_basis_info(speaker)
            can_play = bool(basis) and not skipped and not busy
            self.multi_clone_btn_play_basis.config(state="normal" if can_play else "disabled")
        else:
            self.multi_clone_btn_toggle_skip.config(text=get_text("multi_skip_off"))

    def _multi_speaker_action(self, action):
        speaker = self._selected_multi_speaker_id()
        if not speaker:
            messagebox.showinfo("", get_text("msg_select_speaker_first"))
            return
        action(speaker)

    def _multi_clone_play_selected_basis(self, speaker: str):
        basis = self._multi_clone_speaker_basis_info(speaker)
        self._multi_clone_play_wav(basis.get("wav") or "", speaker)

    def _selected_multi_speaker_id(self) -> str:
        selection = self.multi_clone_speaker_tree.selection()
        if not selection:
            return ""
        return self.multi_clone_speaker_iid_to_id.get(selection[0], "")

    def _multi_clone_export_selected_basis(self):
        speaker = self._selected_multi_speaker_id()
        if not speaker:
            messagebox.showerror("Error", get_text("err_no_speaker_selected"))
            return
        basis = self._multi_clone_speaker_basis_info(speaker)
        if not basis:
            messagebox.showerror("Error", get_text("err_no_basis_to_export"))
            return
        target_dir = filedialog.askdirectory(title=get_text("title_export_basis_dir"))
        if not target_dir:
            return
        try:
            from tool_clonevoice import multi_clone as mc

            source_video = self._multi_clone_basis_source_video(speaker)
            wav_path, txt_path, _meta_path = mc.export_speaker_basis(
                source_video,
                speaker,
                target_dir,
                log=lambda m: self.log(self.multi_clone_log, m),
            )
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        self.log(self.multi_clone_log, get_text("msg_multi_basis_exported").format(speaker, wav_path, txt_path))

    def _multi_clone_toggle_skip_speaker(self, speaker: str):
        if speaker in self.multi_clone_skipped:
            self.multi_clone_skipped.remove(speaker)
            skipped = False
        else:
            self.multi_clone_skipped.add(speaker)
            skipped = True
        videos = self._multi_clone_current_videos()
        if videos:
            from tool_clonevoice import multi_clone as mc

            for video in mc.videos_with_speaker(videos, speaker):
                mc.set_speaker_skipped(
                    video,
                    speaker,
                    skipped=skipped,
                    log=lambda m: self.log(self.multi_clone_log, m),
                )
        self._refresh_multi_clone_speakers()

    def _multi_clone_play_wav(self, path: str, label: str):
        path = (path or "").strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", get_text("err_no_basis_audio"))
            return
        try:
            import winsound

            winsound.PlaySound(None, 0)
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.log(self.multi_clone_log, get_text("msg_single_playing_audio").format(label, path))
        except Exception as exc:
            self.log(self.multi_clone_log, f"Error: {exc}")

    def _multi_clone_apply_basis(
        self,
        speaker: str,
        wav_path: str,
        text: str,
        *,
        source_kind: str,
        meta: dict | None = None,
    ) -> str:
        from tool_clonevoice import multi_clone as mc

        videos = self._multi_clone_current_videos()
        saved = mc.save_speaker_basis_for_videos(
            videos,
            speaker,
            basis_wav=wav_path,
            basis_text=text,
            target_language=self._selected_multi_target_language(),
            source_kind=source_kind,
            meta=meta or {},
            log=lambda m: self.log(self.multi_clone_log, m),
        )
        self.multi_clone_skipped.discard(speaker)
        for video, _saved_wav, _saved_txt in saved:
            mc.set_speaker_skipped(video, speaker, skipped=False, log=lambda m: self.log(self.multi_clone_log, m))
        saved_wav = saved[0][1]
        self.multi_clone_basis[speaker] = {"wav": saved_wav, "text": text, "source": source_kind}
        self.log(self.multi_clone_log, get_text("msg_multi_basis_applied").format(speaker, saved_wav))
        self._refresh_multi_clone_speakers()
        return saved_wav

    def _open_multi_import_basis_dialog(self, speaker: str):
        if not self._multi_clone_current_videos():
            messagebox.showerror("Error", get_text("err_no_video"))
            return
        if speaker in self.multi_clone_skipped:
            messagebox.showerror("Error", get_text("err_speaker_skipped"))
            return

        dlg = tk.Toplevel(self.root)
        dlg.title(get_text("dlg_multi_import_basis_title").format(speaker))
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.columnconfigure(1, weight=1)
        wav_var = tk.StringVar()
        status_var = tk.StringVar()

        ttk.Label(dlg, text=get_text("lbl_basis_wav")).grid(row=0, column=0, sticky="w", padx=(10, 6), pady=(10, 4))
        ttk.Entry(dlg, textvariable=wav_var, width=54).grid(row=0, column=1, sticky="ew", pady=(10, 4))

        def browse_wav():
            path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav"), ("All files", "*.*")])
            if not path:
                return
            wav_var.set(path)
            # Auto-fill from a same-name .txt sidecar (e.g. an exported basis),
            # only when the user has not typed text yet. Still editable.
            sidecar = Path(path).with_suffix(".txt")
            if sidecar.is_file() and not text_widget.get("1.0", "end").strip():
                try:
                    sidecar_text = sidecar.read_text(encoding="utf-8-sig").strip()
                except Exception:
                    sidecar_text = ""
                if sidecar_text:
                    text_widget.delete("1.0", "end")
                    text_widget.insert("1.0", sidecar_text)
                    status_var.set(get_text("msg_basis_text_autofilled").format(sidecar.name))

        ttk.Button(dlg, text=get_text("btn_browse"), command=browse_wav).grid(
            row=0, column=2, sticky="ew", padx=(6, 10), pady=(10, 4)
        )
        ttk.Label(dlg, text=get_text("lbl_basis_text")).grid(row=1, column=0, sticky="nw", padx=(10, 6), pady=4)
        text_widget = tk.Text(dlg, height=5, wrap="word", width=54)
        text_widget.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 10), pady=4)
        ttk.Label(
            dlg,
            text=get_text("note_multi_basis_text_match"),
            foreground="dim gray",
            wraplength=560,
            justify="left",
        ).grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 10), pady=(0, 4))
        ttk.Label(dlg, textvariable=status_var, foreground="dim gray").grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 4)
        )
        buttons = ttk.Frame(dlg)
        buttons.grid(row=4, column=0, columnspan=3, sticky="e", padx=10, pady=(6, 10))

        def save_import():
            wav_path = wav_var.get().strip()
            basis_text = text_widget.get("1.0", "end").strip()
            if not wav_path or not os.path.isfile(wav_path):
                messagebox.showerror("Error", get_text("err_no_basis_audio"))
                return
            if not basis_text:
                messagebox.showerror("Error", get_text("err_basis_text_required"))
                return
            try:
                import soundfile as sf

                info = sf.info(wav_path)
            except Exception as exc:
                messagebox.showerror("Error", get_text("err_bad_wav").format(exc))
                return
            self._multi_clone_apply_basis(
                speaker,
                wav_path,
                basis_text,
                source_kind="user_import",
                meta={
                    "import_wav": wav_path,
                    "samplerate": int(getattr(info, "samplerate", 0) or 0),
                    "channels": int(getattr(info, "channels", 0) or 0),
                    "duration": float(getattr(info, "duration", 0.0) or 0.0),
                },
            )
            dlg.destroy()

        ttk.Button(buttons, text=get_text("btn_save_use_basis"), command=save_import).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text=get_text("btn_cancel"), command=dlg.destroy).pack(side="left")
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

    def _open_multi_voice_design_dialog(self, speaker: str):
        if not self._multi_clone_current_videos():
            messagebox.showerror("Error", get_text("err_no_video"))
            return
        if speaker in self.multi_clone_skipped:
            messagebox.showerror("Error", get_text("err_speaker_skipped"))
            return

        dlg = tk.Toplevel(self.root)
        dlg.title(get_text("dlg_multi_voice_design_title").format(speaker))
        dlg.transient(self.root)
        dlg.columnconfigure(0, weight=1)
        none_label = get_text("opt_voice_design_none")
        selections = []

        for row, (group_label, options) in enumerate(self._single_clone_voice_design_groups()):
            group = ttk.LabelFrame(dlg, text=group_label, padding=6)
            group.grid(row=row, column=0, sticky="ew", padx=10, pady=(8 if row == 0 else 2, 2))
            group.columnconfigure(0, weight=1)
            var = tk.StringVar(value=none_label)
            display_to_value = {none_label: ""}
            for display, value in options:
                display_to_value[display] = value
            ttk.Combobox(
                group,
                textvariable=var,
                values=list(display_to_value.keys()),
                state="readonly",
                width=34,
            ).grid(row=0, column=0, sticky="ew")
            selections.append((var, display_to_value))

        status_var = tk.StringVar()
        ttk.Label(dlg, textvariable=status_var, foreground="dim gray", wraplength=460).grid(
            row=len(selections), column=0, sticky="ew", padx=10, pady=(6, 2)
        )
        buttons = ttk.Frame(dlg)
        buttons.grid(row=len(selections) + 1, column=0, sticky="e", padx=10, pady=(8, 10))
        preview_state = {"wav": "", "text": "", "instruct": ""}

        def current_instruct() -> str:
            attrs = []
            for var, display_to_value in selections:
                value = display_to_value.get(var.get(), "")
                if value:
                    attrs.append(value)
            return ", ".join(attrs)

        def apply_preview() -> bool:
            wav_path = preview_state.get("wav") or ""
            text = preview_state.get("text") or ""
            instruct = preview_state.get("instruct") or ""
            if not wav_path or not text:
                return False
            self._multi_clone_apply_basis(
                speaker,
                wav_path,
                text,
                source_kind="voice_design",
                meta={"instruct": instruct},
            )
            return True

        def generate_design(*, save_after: bool):
            instruct = current_instruct()
            if not instruct:
                messagebox.showerror("Error", get_text("err_voice_design_instruct"))
                return
            if (
                not save_after
                and preview_state.get("instruct") == instruct
                and os.path.isfile(preview_state.get("wav") or "")
            ):
                self._multi_clone_play_wav(preview_state["wav"], get_text("btn_preview_voice_design"))
                return
            status_var.set(get_text("msg_voice_design_generating").format(instruct))

            def worker(holder, _release_holder):
                from tool_clonevoice import multi_clone as mc

                model = self._load_multi_clone_omnivoice_model(holder)
                try:
                    return mc.generate_voice_design_basis_with_model(
                        self._multi_clone_basis_source_video(speaker),
                        speaker,
                        model=model,
                        target_language=self._selected_multi_target_language(),
                        instruct=instruct,
                        log=lambda m: self.log(self.multi_clone_log, m),
                        stop_event=self.multi_clone_stop_event,
                    )
                finally:
                    del model

            def done(result):
                wav_path, text = result
                preview_state.update({"wav": wav_path, "text": text, "instruct": instruct})
                if not dlg.winfo_exists():
                    return
                status_var.set(get_text("msg_voice_design_preview_ready").format(wav_path))
                if save_after:
                    if apply_preview() and dlg.winfo_exists():
                        dlg.destroy()
                else:
                    self._multi_clone_play_wav(wav_path, get_text("btn_preview_voice_design"))

            self._multi_clone_run_async(worker, done)

        def save_design():
            instruct = current_instruct()
            if not instruct:
                messagebox.showerror("Error", get_text("err_voice_design_instruct"))
                return
            if (
                preview_state.get("instruct") == instruct
                and os.path.isfile(preview_state.get("wav") or "")
                and preview_state.get("text")
            ):
                if apply_preview():
                    dlg.destroy()
            else:
                generate_design(save_after=True)

        ttk.Button(
            buttons,
            text=get_text("btn_preview_voice_design"),
            command=lambda: generate_design(save_after=False),
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            buttons,
            text=get_text("btn_save_use_voice_design"),
            command=save_design,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text=get_text("btn_stop"), command=self._stop_multi_clone).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text=get_text("btn_cancel"), command=dlg.destroy).pack(side="left")
        dlg.grab_set()

    def _open_multi_candidate_dialog(self, speaker: str):
        videos = self._multi_clone_current_videos()
        if not videos:
            messagebox.showerror("Error", get_text("err_no_video"))
            return
        if speaker in self.multi_clone_skipped:
            messagebox.showerror("Error", get_text("err_speaker_skipped"))
            return

        dlg = tk.Toplevel(self.root)
        dlg.title(get_text("dlg_multi_select_basis_title").format(speaker))
        dlg.transient(self.root)
        dlg.geometry("980x260")
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(dlg)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        ttk.Label(toolbar, text=get_text("multi_candidate_dialog_hint"), foreground="dim gray").pack(
            side="left", fill="x", expand=True
        )

        panel = CandidateBasisPanel(
            dlg,
            get_text=get_text,
            play_wav=lambda path, label: self._multi_clone_play_wav(path, label),
            include_video=len(videos) > 1,
            height=5,
        )
        panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 8))

        status_var = tk.StringVar()
        ttk.Label(dlg, textvariable=status_var, foreground="dim gray", wraplength=760).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4)
        )
        buttons = ttk.Frame(dlg)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 10))
        buttons.columnconfigure(0, weight=1)

        candidates: list[dict] = []

        def refresh_tree():
            panel.set_candidates(candidates)
            status_var.set(get_text("msg_multi_candidates_done").format(speaker, len(candidates)) if candidates else "")

        def selected_candidate() -> dict | None:
            return panel.selected_candidate()

        def collect_generate():
            from tool_clonevoice import multi_clone as mc

            collect_button.config(state="disabled")
            collect_limit_spinbox.config(state="disabled")
            display = self._multi_clone_candidate_limit()
            pool = max(1, display * mc.CANDIDATE_POOL_FACTOR)
            target_language = self._selected_multi_target_language()
            existing_candidates = list(candidates or [])

            def worker(holder, release_holder):
                from tool_clonevoice import multi_clone as mc
                from tool_clonevoice import single_clone as sc

                per_video = pool if len(videos) == 1 else max(1, min(pool, (pool + len(videos) - 1) // len(videos) + 2))
                disk_existing = mc.load_all_existing_speaker_candidates_for_videos(
                    videos,
                    speaker,
                    log=lambda m: self.log(self.multi_clone_log, m),
                )
                collected = mc.collect_speaker_candidates_with_existing_for_videos(
                    videos,
                    speaker,
                    disk_existing or existing_candidates,
                    per_video=per_video,
                    total=pool,
                    log=lambda m: self.log(self.multi_clone_log, m),
                )
                if not collected:
                    self.log(self.multi_clone_log, get_text("msg_multi_no_candidates").format(speaker))
                    return collected
                self.log(self.multi_clone_log, get_text("msg_single_prepare_samples"))
                jobs = []
                missing_target = [
                    cand for cand in collected
                    if not (
                        cand.get("target_sample_audio")
                        and os.path.isfile(cand.get("target_sample_audio"))
                        and (cand.get("target_sample_text") or "").strip()
                    )
                ]
                if missing_target:
                    model = self._load_multi_clone_omnivoice_model(holder)
                    try:
                        for idx, cand in enumerate(collected, 1):
                            if cand not in missing_target:
                                continue
                            if self.multi_clone_stop_event.is_set():
                                raise RuntimeError("Stopped by user.")
                            label = get_text("msg_multi_candidate_label").format(speaker, idx, len(collected), cand.get("id") or "")
                            self.log(self.multi_clone_log, get_text("msg_single_candidate_generating").format(label))
                            jobs.append(
                                sc.build_candidate_target_sample_job(
                                    cand,
                                    model=model,
                                    target_language=target_language,
                                    log_label=label,
                                    log=lambda m: self.log(self.multi_clone_log, m),
                                    stop_event=self.multi_clone_stop_event,
                                )
                            )
                    finally:
                        del model
                        release_holder()
                if jobs:
                    sc.finish_candidate_target_sample_jobs(
                        jobs,
                        models_root=self.models_root,
                        log=lambda m: self.log(self.multi_clone_log, m),
                    )
                preview_missing = [
                    cand for cand in collected
                    if (cand.get("tgt_text") or "").strip()
                    and cand.get("target_sample_audio")
                    and os.path.isfile(cand.get("target_sample_audio"))
                    and not (cand.get("translated_audio") and os.path.isfile(cand.get("translated_audio")))
                ]
                if preview_missing:
                    model = self._load_multi_clone_omnivoice_model(holder)
                    try:
                        sc.generate_candidate_translated_previews_with_model(
                            collected,
                            model=model,
                            target_language=target_language,
                            label_func=lambda i, n, c: get_text("msg_multi_candidate_label").format(speaker, i, n, c.get("id") or ""),
                            log=lambda m: self.log(self.multi_clone_log, m),
                            stop_event=self.multi_clone_stop_event,
                        )
                    finally:
                        del model
                        release_holder()
                score_missing = [
                    cand for cand in collected
                    if cand.get("ecapa_similarity") is None
                    and cand.get("source_audio")
                    and (
                        (cand.get("translated_audio") and os.path.isfile(cand.get("translated_audio")))
                        or (cand.get("target_sample_audio") and os.path.isfile(cand.get("target_sample_audio")))
                    )
                ]
                if score_missing:
                    sc.score_candidate_similarities(
                        score_missing,
                        models_root=self.models_root,
                        log=lambda m: self.log(self.multi_clone_log, m),
                    )
                collected.sort(
                    key=lambda c: float(c.get("ecapa_similarity") if c.get("ecapa_similarity") is not None else -999.0),
                    reverse=True,
                )
                # Evaluated a 2x pool by similarity; show the best `display`.
                collected = collected[:display]
                for idx, cand in enumerate(collected, 1):
                    cand["global_rank"] = idx
                return collected

            def done(result):
                if not dlg.winfo_exists():
                    return
                candidates[:] = list(result or [])
                refresh_tree()

            self._multi_clone_run_async(worker, done)

            def enable_when_idle():
                if not dlg.winfo_exists():
                    return
                if self.multi_clone_thread is not None and self.multi_clone_thread.is_alive():
                    self.root.after(300, enable_when_idle)
                    return
                collect_button.config(state="normal")
                collect_limit_spinbox.config(state="normal")

            self.root.after(300, enable_when_idle)

        def adopt_selected():
            cand = selected_candidate()
            if cand is None:
                messagebox.showerror("Error", get_text("err_no_candidate_selected"))
                return
            wav_path = cand.get("target_sample_audio") or ""
            text = cand.get("target_sample_text") or ""
            if not wav_path or not os.path.isfile(wav_path) or not text:
                messagebox.showerror("Error", get_text("err_no_basis_audio"))
                return
            meta = {
                "candidate_id": cand.get("id"),
                "source_audio": cand.get("source_audio"),
                "source_score": cand.get("score"),
                "ecapa_similarity": cand.get("ecapa_similarity"),
            }
            self._multi_clone_apply_basis(
                speaker,
                wav_path,
                text,
                source_kind="candidate_target_sample",
                meta=meta,
            )
            dlg.destroy()

        collect_group = ttk.Frame(buttons)
        collect_group.grid(row=0, column=0, sticky="w")
        ttk.Label(collect_group, text=get_text("lbl_candidate_limit")).pack(side="left", padx=(0, 6))
        collect_limit_spinbox = ttk.Spinbox(
            collect_group,
            from_=1,
            to=50,
            textvariable=self.multi_clone_candidate_limit_var,
            width=6,
        )
        collect_limit_spinbox.pack(side="left", padx=(0, 8))
        collect_button = ttk.Button(collect_group, text=get_text("btn_collect_generate_samples"), command=collect_generate)
        collect_button.pack(side="left")
        ttk.Button(collect_group, text=get_text("btn_stop"), command=self._stop_multi_clone).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text=get_text("btn_adopt_basis"), command=adopt_selected).grid(
            row=0, column=1, sticky="e", padx=(8, 4)
        )
        ttk.Button(buttons, text=get_text("btn_cancel"), command=dlg.destroy).grid(row=0, column=2, sticky="e")
        try:
            from tool_clonevoice import multi_clone as mc

            loaded = mc.load_existing_speaker_candidates_for_videos(
                videos,
                speaker,
                total=self._multi_clone_candidate_limit(),
                log=lambda m: self.log(self.multi_clone_log, m),
            )
            if loaded:
                candidates[:] = loaded
                refresh_tree()
        except Exception as exc:
            self.log(self.multi_clone_log, f"[multi] existing candidates load skipped: {exc}")
        dlg.grab_set()

    def _multi_clone_translate_clone(self):
        videos = self._multi_clone_current_videos() or self._multi_clone_scan_videos()
        if not videos:
            return
        self._multi_clone_restore_manifest_state()
        if not self._multi_clone_ready_for_export():
            messagebox.showerror("Error", get_text("err_not_all_speakers_ready"))
            return
        if self._multi_clone_all_speakers_skipped():
            messagebox.showerror("Error", get_text("err_all_speakers_skipped"))
            return
        if not self._translation_api_configured():
            self._refresh_multi_translate_config_status()
            messagebox.showerror("Error", get_text("err_no_translation_api_key"))
            return
        target_language = self._selected_multi_target_language()
        loudness_mode = self.multi_clone_loudness_mode_map.get(self.multi_clone_loudness_mode_var.get(), "envelope")
        envelope_alpha = self.multi_clone_envelope_alpha_map.get(self.multi_clone_envelope_alpha_var.get(), 0.6)
        tempo_fit = self.multi_clone_tempo_fit_map.get(self.multi_clone_tempo_fit_var.get(), "moderate")
        skip_existing = self.multi_clone_skip_existing_var.get()
        skipped = set(self.multi_clone_skipped)
        source_correction = self.multi_clone_source_correction_var.get()

        def worker(holder, _release_holder):
            from tool_clonevoice import multi_clone as mc
            from tool_clonevoice import single_clone as sc

            mc.set_skipped_speakers_for_videos(
                videos,
                skipped,
                log=lambda m: self.log(self.multi_clone_log, m),
            )
            return sc.translate_and_synthesize(
                videos,
                target_language=target_language,
                models_root=self.models_root,
                source_correction=source_correction,
                loudness_mode=loudness_mode,
                envelope_alpha=envelope_alpha,
                tempo_fit=tempo_fit,
                skip_existing=skip_existing,
                log=lambda m: self.log(self.multi_clone_log, m),
                stop_event=self.multi_clone_stop_event,
                model_holder=holder,
            )

        def done(result):
            result = result or {}
            written = len(result.get("written") or [])
            skipped_count = len(result.get("skipped") or [])
            if written == 0 and skipped_count:
                self.log(self.multi_clone_log, get_text("msg_single_all_skipped").format(skipped_count))
            else:
                self.log(self.multi_clone_log, get_text("msg_multi_done").format(written, skipped_count))

        self._multi_clone_run_async(worker, done)

    def _stop_multi_clone(self):
        self.multi_clone_stop_event.set()
        self.multi_clone_btn_stop.config(state="disabled")

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
            self.mix_batch_dir_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)
        else:
            self.mix_batch_dir_row.grid_remove()
            self.single_mix_video_row.grid()

    def _browse_single_mix_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("MP4", "*.mp4"), ("Video Files", "*.mp4 *.mkv *.mov"), ("All Files", "*.*")]
        )
        if path:
            self.single_mix_video_var.set(path)

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
            si_audio_path = sl.default_si_audio_path(video_path)
            if not si_audio_path or not os.path.isfile(si_audio_path):
                messagebox.showerror("Error", i18n.translate("si", "err_si_wav_file"))
                return
            output_path = (
                dub.default_dub_output_path(video_path)
                if dubbing
                else sl.default_si_mix_output_path(video_path)
            )
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
        duck_preset = self._single_mix_duck_preset_map.get(
            self.single_mix_duck_preset_var.get(),
            sl.DEFAULT_DUCK_PRESET,
        )
        use_duck_key = False if dubbing else self.single_mix_duck_key_var.get()

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
                        duck_preset=duck_preset,
                        use_duck_key=use_duck_key,
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
                    duck_preset=duck_preset,
                    use_duck_key=use_duck_key,
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

    def _update_single_mix_duck_preset_state(self):
        if hasattr(self, "single_mix_duck_preset_combo"):
            self.single_mix_duck_preset_combo.config(state="readonly")

    def _selected_target_language(self) -> str:
        return self._tgt_map.get(self.tgt_lang_var.get(), "Chinese")

    def _on_single_mix_mode_change(self):
        dubbing = self.single_mix_mode_var.get() == "dub"
        if dubbing:
            self.single_mix_channel_var.set(get_text("opt_channel_both"))
            self.single_mix_delay_var.set("0s")
            for widget in self.single_mix_si_option_widgets:
                widget.grid_remove()
        else:
            for widget in self.single_mix_si_option_widgets:
                widget.grid()
            self.single_mix_opts_frame.grid()

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
        if self.clone_input_mode_var.get() in ("batch", "shared", "shared_batch"):
            self.clone_input_label_var.set(get_text("lbl_input_dir"))
        else:
            self.clone_input_label_var.set(get_text("lbl_input_video"))
        self.input_video_var.set("")

    def _browse_video(self):
        if self.clone_input_mode_var.get() in ("batch", "shared", "shared_batch"):
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

    def _list_shared_subfolders(self, base_dir: str) -> list[tuple[str, list[str]]]:
        """Immediate subfolders that contain videos; each is one shared-basis group.

        Used by the per-subfolder batch mode: the parent holds several folders,
        each folder being one 'same people' set that shares a voice basis.
        """
        groups: list[tuple[str, list[str]]] = []
        try:
            names = sorted(os.listdir(base_dir), key=lambda s: s.lower())
        except OSError:
            return groups
        for name in names:
            sub = os.path.join(base_dir, name)
            if os.path.isdir(sub):
                vids = self._scan_clone_batch_videos(sub)
                if vids:
                    groups.append((sub, vids))
        return groups

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

    def _translation_api_configured(self) -> bool:
        if not self._translation_api_key_configured():
            return False
        try:
            from tool_subtitle import logic as tsl

            cfg = tsl.load_trans_config()
            return bool((cfg.get("api_base_url") or "").strip() and (cfg.get("model_name") or "").strip())
        except Exception:
            return False

    def _run(self):
        input_path = self.input_video_var.get().strip()
        mode = self.clone_input_mode_var.get()
        shared_mode = mode == "shared"
        shared_batch_mode = mode == "shared_batch"
        batch_mode = mode == "batch"
        shared_groups: list[tuple[str, list[str]]] = []
        if batch_mode or shared_mode or shared_batch_mode:
            if not input_path or not os.path.isdir(input_path):
                messagebox.showerror("Error", get_text("err_no_dir"))
                return
            if shared_batch_mode:
                shared_groups = self._list_shared_subfolders(input_path)
                if not shared_groups:
                    messagebox.showerror("Error", get_text("err_no_shared_subfolders"))
                    return
                videos = [v for _folder, group in shared_groups for v in group]
            else:
                videos = self._scan_clone_batch_videos(input_path)
                if not videos:
                    messagebox.showerror("Error", get_text("err_no_batch_videos"))
                    return
        elif not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", get_text("err_no_video"))
            return
        else:
            videos = [input_path]

        if not self._translation_api_configured():
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
        tempo_fit = self._tempo_fit_map.get(self.tempo_fit_var.get(), "moderate")
        keep_intermediate = self.keep_intermediate_var.get()
        skip_existing = self.skip_existing_var.get()
        source_correction = self.source_correction_var.get()
        vad_sensitivity = self.vad_sensitivity_map.get(self.clone_vad_var.get(), "high")

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
                if shared_batch_mode:
                    for gi, (folder, group_videos) in enumerate(shared_groups, 1):
                        if self.stop_event.is_set():
                            raise RuntimeError("Stopped by user.")
                        self.log(self.clone_log, get_text("msg_shared_batch_group").format(
                            gi, len(shared_groups), folder, len(group_videos)))
                        self._run_shared_folder(
                            group_videos, holder, release_holder_on_main_thread,
                            model_key=model_key, language=language, backend=backend,
                            num_speakers=num_speakers, target_language=target_language,
                            denoise=denoise, loudness_mode=loudness_mode,
                            envelope_alpha=envelope_alpha, tempo_fit=tempo_fit,
                            skip_existing=skip_existing, source_correction=source_correction,
                            vad_sensitivity=vad_sensitivity,
                        )
                    return
                if shared_mode:
                    self._run_shared_folder(
                        videos, holder, release_holder_on_main_thread,
                        model_key=model_key, language=language, backend=backend,
                        num_speakers=num_speakers, target_language=target_language,
                        denoise=denoise, loudness_mode=loudness_mode,
                        envelope_alpha=envelope_alpha, tempo_fit=tempo_fit,
                        skip_existing=skip_existing, source_correction=source_correction,
                        vad_sensitivity=vad_sensitivity,
                    )
                    return
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
                        tempo_fit=tempo_fit,
                        models_root=self.models_root,
                        keep_intermediate=keep_intermediate,
                        skip_existing=skip_existing,
                        source_correction=source_correction,
                        vad_sensitivity=vad_sensitivity,
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

    def _run_shared_folder(self, videos, holder, release_holder, *, model_key, language,
                           backend, num_speakers, target_language, denoise, loudness_mode,
                           envelope_alpha, tempo_fit, skip_existing, source_correction=None,
                           vad_sensitivity="high"):
        """One-click 'same folder' mode: global-diarize the folder, auto-pick one
        shared reference per speaker, then translate + synthesize every video so
        the whole folder clones each speaker from the same voice."""
        from pathlib import Path
        from tool_clonevoice import logic
        from tool_clonevoice import multi_clone as mc
        from tool_si import logic as si

        diar_backend = backend if backend != "auto" else "pyannote"
        clog = lambda m: self.log(self.clone_log, m)

        speaker_note = num_speakers if num_speakers is not None else get_text("opt_num_auto")
        self.log(self.clone_log, get_text("msg_multi_global_prescan_start").format(len(videos), speaker_note))
        turns_by_video = mc.prescan_global_diarize(
            videos, models_root=self.models_root, diarize_backend=diar_backend,
            num_speakers=num_speakers, denoise=denoise, log=clog, stop_event=self.stop_event,
        )
        for index, video in enumerate(videos, 1):
            if self.stop_event.is_set():
                raise RuntimeError("Stopped by user.")
            self.log(self.clone_log, get_text("msg_batch_item").format(index, len(videos), video))
            mc.run_multi_transcribe(
                video, model_key=model_key, language=language, target_language=target_language,
                models_root=self.models_root, diarize_backend=diar_backend, num_speakers=num_speakers,
                denoise=denoise, vad_sensitivity=vad_sensitivity,
                precomputed_turns=turns_by_video.get(str(Path(video))),
                log=clog, stop_event=self.stop_event, model_holder=holder,
            )
            release_holder()

        self.log(self.clone_log, get_text("msg_shared_refs"))
        mc.extract_shared_references(videos, log=clog, stop_event=self.stop_event)

        written = 0
        for index, video in enumerate(videos, 1):
            if self.stop_event.is_set():
                raise RuntimeError("Stopped by user.")
            out_path = si.default_si_audio_path(video)
            if skip_existing and Path(out_path).exists():
                self.log(self.clone_log, get_text("msg_shared_skip_existing").format(out_path))
                continue
            self.log(self.clone_log, get_text("msg_batch_item").format(index, len(videos), video))
            logic.run_translate(
                video, target_language=target_language, source_correction=source_correction,
                log=clog, stop_event=self.stop_event,
            )
            logic.run_synthesize(
                video, models_root=self.models_root, text_field="tgt_text", language=target_language,
                loudness_mode=loudness_mode, envelope_alpha=envelope_alpha, tempo_fit=tempo_fit,
                log=clog, stop_event=self.stop_event, model_holder=holder,
            )
            release_holder()
            written += 1
        if not self.stop_event.is_set():
            self.log(self.clone_log, get_text("msg_batch_clone_done").format(written))

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
        return dlg

    def _stop(self):
        self.stop_event.set()
        self.btn_stop.config(state="disabled")
