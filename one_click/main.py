import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import locale
import time
from utils import app_config, i18n

# Import logic module - use try/except to handle both direct run and import from main
try:
    from . import logic
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import logic


# --- i18n Setup ---


def get_text(key):
    return i18n.translate('one_click', key)

_FINE_CONF_VALUES = ("0.3（误检多）", "0.4", "0.5", "0.6", "0.7", "0.8（漏检多）")
_DEFAULT_FINE_CONF = "0.5"


class VRMosaicOneClickApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text('title'))
        
        # Main Frame to hold everything
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(header_frame, text=get_text('title'), font=('Arial', 14, 'bold')).pack(side='left')
        
        if self.on_return:
            ttk.Button(header_frame, text=get_text('btn_return'), command=self.on_return).pack(side='right')
            
            # Clear any existing menu
            empty_menu = tk.Menu(self.root)
            self.root.config(menu=empty_menu)

        
        # Global quality/speed setting for NVENC presets, applied to GPU encoding in all tabs.
        self.create_settings_bar(main_frame)

        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 8], font=('Arial', 10, 'bold'))

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(expand=True, fill='both')
        
        self.create_single_auto_tab()
        self.create_single_eye_tab()
        self.create_batch_auto_tab()
        self.create_batch_eye_tab()
        self.create_merge_tab()
        
        # Check dependencies
        missing = logic.check_dependencies()
        if missing:
            self.log_to_all(get_text('warn_dep').format(', '.join(missing)))
            self.log_to_all(get_text('warn_path'))

    def _tr(self, zh, en, ja):
        """Small three-language text helper to avoid changing i18n JSON."""
        lang = app_config.get_language()
        return {'zh': zh, 'en': en, 'ja': ja}.get(lang, zh)

    def _grid_pre_extract_check(self, parent, enabled_variable, conf_variable, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        ttk.Checkbutton(frame, text=get_text('opt_pre_extract'), variable=enabled_variable).pack(side='left')
        ttk.Combobox(
            frame,
            textvariable=conf_variable,
            state='readonly',
            width=5,
            values=_FINE_CONF_VALUES,
        ).pack(side='left', padx=(8, 0))
        return frame

    def _selected_fine_conf(self, variable):
        try:
            text = str(variable.get()).strip()
            for value in ("0.3", "0.4", "0.5", "0.6", "0.7", "0.8"):
                if text.startswith(value):
                    return float(value)
            return float(text)
        except (tk.TclError, TypeError, ValueError):
            return float(_DEFAULT_FINE_CONF)

    def create_settings_bar(self, parent):
        """Quality/speed dropdown mapped to NVENC encode presets P4-P7.

        NVENC preset semantics are counterintuitive: P7 is slowest/highest
        quality, while P4 is faster with slightly lower quality.
        """
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(0, 8))

        label = self._tr('画质 / 速度：', 'Quality / Speed:', '画質 / 速度：')
        ttk.Label(bar, text=label).pack(side='left', padx=(0, 5))

        # (display text, preset value), ordered from higher quality to faster; include NVENC preset codes for advanced users.
        opts = [
            (self._tr('最高画质（最慢）', 'Best quality (slowest)', '最高画質（最も遅い）') + ' [P7]', 'P7'),
            (self._tr('高画质', 'High quality', '高画質') + ' [P6]', 'P6'),
            (self._tr('均衡', 'Balanced', 'バランス') + ' [P5]', 'P5'),
            (self._tr('快速（画质略低）', 'Fast (lower quality)', '高速（画質やや低）') + ' [P4]', 'P4'),
        ]
        self._preset_disp_to_val = {d: v for d, v in opts}
        self._preset_val_to_disp = {v: d for d, v in opts}

        cur = str(app_config.get('gpu_encode_preset', 'P7') or 'P7').upper()
        if cur not in self._preset_val_to_disp:
            cur = 'P7'

        self.preset_var = tk.StringVar(value=self._preset_val_to_disp[cur])
        combo = ttk.Combobox(bar, textvariable=self.preset_var, state='readonly',
                             width=26, values=[d for d, _ in opts])
        combo.pack(side='left')
        combo.bind('<<ComboboxSelected>>', self._on_preset_change)

        hint = self._tr('（仅影响显卡编码，越快画质越低）',
                        '(GPU encode only; faster = lower quality)',
                        '（GPUエンコードのみ。速いほど画質低）')
        ttk.Label(bar, text=hint, foreground='gray').pack(side='left', padx=8)

    def _on_preset_change(self, event=None):
        val = self._preset_disp_to_val.get(self.preset_var.get(), 'P7')
        app_config.set('gpu_encode_preset', val)
        self.log_to_all(self._tr(f'画质/速度已设为 {val}',
                                 f'Quality/Speed set to {val}',
                                 f'画質/速度を {val} に設定'))

    def log_to_widget(self, widget, message):
        def _do():
            widget.insert(tk.END, message + "\n")
            widget.see(tk.END)
        self.root.after(0, _do)

    def log_to_all(self, message):
        for log_widget in [self.log_s_auto, self.log_s_eye, self.log_b_auto, self.log_b_eye, self.log_merge]:
            self.log_to_widget(log_widget, message)
            
    def browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.webm")])
        if path: var.set(path)
        
    def browse_dir(self, var):
        path = filedialog.askdirectory()
        if path: var.set(path)

    def validate_time_input(self, P):
        # Allow empty or digits and colon
        if P == "": return True
        return all(c in "0123456789:" for c in P)

    def validate_time_logic(self, start_time, end_time=None):
        # 1. Format Check
        def check_format(t):
            if not t: return True 
            parts = t.split(':')
            if len(parts) > 3: return False
            try:
                for p in parts:
                    float(p)
            except ValueError:
                return False
            return True

        if not check_format(start_time):
            return False, get_text('lbl_start') + " " + get_text('err_invalid_format')
        if end_time and not check_format(end_time):
            return False, get_text('lbl_end') + " " + get_text('err_invalid_format')

        # 2. Logic Check (Start < End)
        def to_sec(t):
            if not t: return 0
            parts = list(map(float, t.split(':')))
            if len(parts) == 1: return parts[0]
            if len(parts) == 2: return parts[0]*60 + parts[1]
            if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
            return 0

        s_sec = to_sec(start_time)
        e_sec = to_sec(end_time) if end_time else 0
        
        if end_time and e_sec > 0 and s_sec >= e_sec:
             return False, get_text('err_time_order')
             
        return True, ""

    # --- Tab 1: Single File (Auto) ---
    def create_single_auto_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_s_auto'))
        
        vcmd = (self.root.register(self.validate_time_input), '%P')
        
        ttk.Label(tab, text=get_text('lbl_input')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.s_auto_input = tk.StringVar()
        
        input_frame = ttk.Frame(tab)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.s_auto_input, width=50).pack(side='left')
        ttk.Button(input_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.s_auto_input)).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_start')).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.s_auto_start = tk.StringVar()
        ttk.Entry(tab, textvariable=self.s_auto_start, validate='key', validatecommand=vcmd).grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_end')).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.s_auto_end = tk.StringVar()
        end_frame_s_auto = ttk.Frame(tab)
        end_frame_s_auto.grid(row=2, column=1, sticky='w', padx=5)
        ttk.Entry(end_frame_s_auto, textvariable=self.s_auto_end, validate='key', validatecommand=vcmd).pack(side='left')
        ttk.Label(end_frame_s_auto, text=get_text('lbl_end_hint'), foreground='gray').pack(side='left', padx=(8, 0))

        self.s_auto_fisheye = tk.BooleanVar()
        self.s_auto_pre_extract = tk.BooleanVar(value=False)
        self.s_auto_fine_conf = tk.StringVar(value=_DEFAULT_FINE_CONF)
        self.s_auto_keep = tk.BooleanVar()
        self.s_auto_keep_bitrate = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(tab, text=get_text('opt_fisheye'), variable=self.s_auto_fisheye).grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        self._grid_pre_extract_check(tab, self.s_auto_pre_extract, self.s_auto_fine_conf, 4)
        ttk.Checkbutton(tab, text=get_text('chk_keep_inter'), variable=self.s_auto_keep).grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        ttk.Checkbutton(tab, text=get_text('chk_keep_bitrate'), variable=self.s_auto_keep_bitrate).grid(row=6, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=7, column=1, pady=20, sticky='w', padx=5)
        
        self.btn_s_auto = ttk.Button(btn_frame, text=get_text('btn_start'), command=self.run_s_auto)
        self.btn_s_auto.pack(side='left', padx=5)
        
        self.btn_stop_s_auto = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_s_auto, state='disabled')
        self.btn_stop_s_auto.pack(side='left', padx=5)
        
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=8, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(8, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_s_auto = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_s_auto.pack(fill='both', expand=True)
        
        self.proc_s_auto = None
        self.stop_s_auto_requested = False

    def run_s_auto(self):
        valid, msg = self.validate_time_logic(self.s_auto_start.get(), self.s_auto_end.get())
        if not valid:
            messagebox.showerror("Error", msg)
            return

        self.btn_s_auto.config(state='disabled')
        self.btn_stop_s_auto.config(state='normal')
        self.stop_s_auto_requested = False
        threading.Thread(target=self._s_auto_thread, daemon=True).start()

    def stop_s_auto(self):
        self.stop_s_auto_requested = True
        proc = self.proc_s_auto
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_s_auto = None
            self.log_to_widget(self.log_s_auto, get_text('msg_stop'))

    def _s_auto_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.proc_s_auto = p
            if self.stop_s_auto_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_single_file_pipeline(
                self.s_auto_input.get(),
                self.s_auto_start.get(),
                self.s_auto_end.get(),
                self.s_auto_fisheye.get(),
                self.s_auto_keep.get(),
                self.s_auto_keep_bitrate.get(),
                lambda msg: self.log_to_widget(self.log_s_auto, msg),
                _on_proc,
                pre_extract=self.s_auto_pre_extract.get(),
                source_scan=True,
                fine_conf=self._selected_fine_conf(self.s_auto_fine_conf)
            )
        except Exception as e:
            self.log_to_widget(self.log_s_auto, f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_to_widget(self.log_s_auto, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_s_auto.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop_s_auto.config(state='disabled'))
            self.proc_s_auto = None

    # --- Tab 2: Single File (One Eye) ---
    def create_single_eye_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_s_eye'))
        
        vcmd = (self.root.register(self.validate_time_input), '%P')
        
        ttk.Label(tab, text=get_text('lbl_input')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.s_eye_input = tk.StringVar()
        
        input_frame = ttk.Frame(tab)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.s_eye_input, width=50).pack(side='left')
        ttk.Button(input_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.s_eye_input)).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_eye')).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.s_eye_mode = tk.IntVar(value=1)
        eye_frame = ttk.Frame(tab)
        eye_frame.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Radiobutton(eye_frame, text=get_text('opt_left'), variable=self.s_eye_mode, value=1).pack(side='left', padx=5)
        ttk.Radiobutton(eye_frame, text=get_text('opt_right'), variable=self.s_eye_mode, value=2).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_start')).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.s_eye_start = tk.StringVar()
        ttk.Entry(tab, textvariable=self.s_eye_start, validate='key', validatecommand=vcmd).grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_end')).grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.s_eye_end = tk.StringVar()
        end_frame_s_eye = ttk.Frame(tab)
        end_frame_s_eye.grid(row=3, column=1, sticky='w', padx=5)
        ttk.Entry(end_frame_s_eye, textvariable=self.s_eye_end, validate='key', validatecommand=vcmd).pack(side='left')
        ttk.Label(end_frame_s_eye, text=get_text('lbl_end_hint'), foreground='gray').pack(side='left', padx=(8, 0))

        self.s_eye_fisheye = tk.BooleanVar()
        self.s_eye_pre_extract = tk.BooleanVar(value=False)
        self.s_eye_fine_conf = tk.StringVar(value=_DEFAULT_FINE_CONF)
        self.s_eye_keep = tk.BooleanVar()
        self.s_eye_keep_bitrate = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(tab, text=get_text('opt_fisheye'), variable=self.s_eye_fisheye).grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        self._grid_pre_extract_check(tab, self.s_eye_pre_extract, self.s_eye_fine_conf, 5)
        ttk.Checkbutton(tab, text=get_text('chk_keep_inter'), variable=self.s_eye_keep).grid(row=6, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        ttk.Checkbutton(tab, text=get_text('chk_keep_bitrate'), variable=self.s_eye_keep_bitrate).grid(row=7, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=8, column=1, pady=20, sticky='w', padx=5)
        
        self.btn_s_eye = ttk.Button(btn_frame, text=get_text('btn_start'), command=self.run_s_eye)
        self.btn_s_eye.pack(side='left', padx=5)
        
        self.btn_stop_s_eye = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_s_eye, state='disabled')
        self.btn_stop_s_eye.pack(side='left', padx=5)
        
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=9, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(9, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_s_eye = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_s_eye.pack(fill='both', expand=True)
        
        self.proc_s_eye = None
        self.stop_s_eye_requested = False

    def run_s_eye(self):
        valid, msg = self.validate_time_logic(self.s_eye_start.get(), self.s_eye_end.get())
        if not valid:
            messagebox.showerror("Error", msg)
            return

        self.btn_s_eye.config(state='disabled')
        self.btn_stop_s_eye.config(state='normal')
        self.stop_s_eye_requested = False
        threading.Thread(target=self._s_eye_thread, daemon=True).start()

    def stop_s_eye(self):
        self.stop_s_eye_requested = True
        proc = self.proc_s_eye
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_s_eye = None
            self.log_to_widget(self.log_s_eye, get_text('msg_stop'))

    def _s_eye_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.proc_s_eye = p
            if self.stop_s_eye_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_single_eye_pipeline(
                self.s_eye_input.get(),
                self.s_eye_mode.get(),
                self.s_eye_start.get(),
                self.s_eye_end.get(),
                self.s_eye_fisheye.get(),
                self.s_eye_keep.get(),
                self.s_eye_keep_bitrate.get(),
                lambda msg: self.log_to_widget(self.log_s_eye, msg),
                _on_proc,
                pre_extract=self.s_eye_pre_extract.get(),
                source_scan=True,
                fine_conf=self._selected_fine_conf(self.s_eye_fine_conf)
            )
        except Exception as e:
            self.log_to_widget(self.log_s_eye, f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_to_widget(self.log_s_eye, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_s_eye.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop_s_eye.config(state='disabled'))
            self.proc_s_eye = None

    # --- Tab 3: Batch (Auto) ---
    def create_batch_auto_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_b_auto'))
        
        ttk.Label(tab, text=get_text('lbl_dir')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.b_auto_input = tk.StringVar()
        
        input_frame = ttk.Frame(tab)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.b_auto_input, width=50).pack(side='left')
        ttk.Button(input_frame, text=get_text('btn_browse'), command=lambda: self.browse_dir(self.b_auto_input)).pack(side='left', padx=5)
        
        self.b_auto_fisheye = tk.BooleanVar()
        self.b_auto_pre_extract = tk.BooleanVar(value=False)
        self.b_auto_fine_conf = tk.StringVar(value=_DEFAULT_FINE_CONF)
        self.b_auto_keep_bitrate = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text=get_text('opt_fisheye'), variable=self.b_auto_fisheye).grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        self._grid_pre_extract_check(tab, self.b_auto_pre_extract, self.b_auto_fine_conf, 2)
        ttk.Checkbutton(tab, text=get_text('chk_keep_bitrate'), variable=self.b_auto_keep_bitrate).grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=4, column=1, pady=20, sticky='w', padx=5)
        
        self.btn_b_auto = ttk.Button(btn_frame, text=get_text('btn_batch'), command=self.run_b_auto)
        self.btn_b_auto.pack(side='left', padx=5)
        
        self.btn_stop_b_auto = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_b_auto, state='disabled')
        self.btn_stop_b_auto.pack(side='left', padx=5)
        
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=5, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(5, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_b_auto = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_b_auto.pack(fill='both', expand=True)
        
        self.proc_b_auto = None
        self.stop_b_auto_requested = False

    def run_b_auto(self):
        self.btn_b_auto.config(state='disabled')
        self.btn_stop_b_auto.config(state='normal')
        self.stop_b_auto_requested = False
        threading.Thread(target=self._b_auto_thread, daemon=True).start()

    def stop_b_auto(self):
        self.stop_b_auto_requested = True
        proc = self.proc_b_auto
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_b_auto = None
            self.log_to_widget(self.log_b_auto, get_text('msg_stop'))

    def _b_auto_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.proc_b_auto = p
            if self.stop_b_auto_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_batch_pipeline(
                self.b_auto_input.get(),
                self.b_auto_fisheye.get(),
                self.b_auto_keep_bitrate.get(),
                lambda msg: self.log_to_widget(self.log_b_auto, msg),
                _on_proc,
                pre_extract=self.b_auto_pre_extract.get(),
                source_scan=True,
                fine_conf=self._selected_fine_conf(self.b_auto_fine_conf)
            )
        except Exception as e:
            self.log_to_widget(self.log_b_auto, f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_to_widget(self.log_b_auto, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_b_auto.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop_b_auto.config(state='disabled'))
            self.proc_b_auto = None

    # --- Tab 4: Batch (One Eye) ---
    def create_batch_eye_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_b_eye'))
        
        ttk.Label(tab, text=get_text('lbl_dir')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.b_eye_input = tk.StringVar()
        
        input_frame = ttk.Frame(tab)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.b_eye_input, width=50).pack(side='left')
        ttk.Button(input_frame, text=get_text('btn_browse'), command=lambda: self.browse_dir(self.b_eye_input)).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_eye')).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.b_eye_mode = tk.IntVar(value=1)
        eye_frame = ttk.Frame(tab)
        eye_frame.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Radiobutton(eye_frame, text=get_text('opt_left'), variable=self.b_eye_mode, value=1).pack(side='left', padx=5)
        ttk.Radiobutton(eye_frame, text=get_text('opt_right'), variable=self.b_eye_mode, value=2).pack(side='left', padx=5)
        
        self.b_eye_fisheye = tk.BooleanVar()
        self.b_eye_pre_extract = tk.BooleanVar(value=False)
        self.b_eye_fine_conf = tk.StringVar(value=_DEFAULT_FINE_CONF)
        self.b_eye_keep_bitrate = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text=get_text('opt_fisheye'), variable=self.b_eye_fisheye).grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        self._grid_pre_extract_check(tab, self.b_eye_pre_extract, self.b_eye_fine_conf, 3)
        ttk.Checkbutton(tab, text=get_text('chk_keep_bitrate'), variable=self.b_eye_keep_bitrate).grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=5, column=1, pady=20, sticky='w', padx=5)
        
        self.btn_b_eye = ttk.Button(btn_frame, text=get_text('btn_batch'), command=self.run_b_eye)
        self.btn_b_eye.pack(side='left', padx=5)
        
        self.btn_stop_b_eye = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_b_eye, state='disabled')
        self.btn_stop_b_eye.pack(side='left', padx=5)
        
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=6, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(6, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_b_eye = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_b_eye.pack(fill='both', expand=True)
        
        self.proc_b_eye = None
        self.stop_b_eye_requested = False

    def run_b_eye(self):
        self.btn_b_eye.config(state='disabled')
        self.btn_stop_b_eye.config(state='normal')
        self.stop_b_eye_requested = False
        threading.Thread(target=self._b_eye_thread, daemon=True).start()

    def stop_b_eye(self):
        self.stop_b_eye_requested = True
        proc = self.proc_b_eye
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_b_eye = None
            self.log_to_widget(self.log_b_eye, get_text('msg_stop'))

    def _b_eye_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.proc_b_eye = p
            if self.stop_b_eye_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_batch_eye_pipeline(
                self.b_eye_input.get(),
                self.b_eye_mode.get(),
                self.b_eye_fisheye.get(),
                self.b_eye_keep_bitrate.get(),
                lambda msg: self.log_to_widget(self.log_b_eye, msg),
                _on_proc,
                pre_extract=self.b_eye_pre_extract.get(),
                source_scan=True,
                fine_conf=self._selected_fine_conf(self.b_eye_fine_conf)
            )
        except Exception as e:
            self.log_to_widget(self.log_b_eye, f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_to_widget(self.log_b_eye, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_b_eye.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop_b_eye.config(state='disabled'))
            self.proc_b_eye = None

    # --- Tab 5: Merge Tools ---
    def create_merge_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_merge'))
        
        ttk.Label(tab, text=get_text('lbl_left')).grid(row=0, column=0, sticky='w')
        self.merge_l = tk.StringVar()
        
        l_frame = ttk.Frame(tab)
        l_frame.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Entry(l_frame, textvariable=self.merge_l, width=50).pack(side='left')
        ttk.Button(l_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.merge_l)).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_right')).grid(row=1, column=0, sticky='w')
        self.merge_r = tk.StringVar()
        
        r_frame = ttk.Frame(tab)
        r_frame.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Entry(r_frame, textvariable=self.merge_r, width=50).pack(side='left')
        ttk.Button(r_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.merge_r)).pack(side='left', padx=5)
        
        self.merge_keep_bitrate = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text=get_text('chk_keep_bitrate'), variable=self.merge_keep_bitrate).grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=3, column=1, pady=20, sticky='w', padx=5)
        
        self.btn_merge = ttk.Button(btn_frame, text=get_text('btn_merge'), command=self.run_merge)
        self.btn_merge.pack(side='left', padx=5)
        
        self.btn_stop_merge = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_merge, state='disabled')
        self.btn_stop_merge.pack(side='left', padx=5)
        
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=4, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(4, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_merge = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_merge.pack(fill='both', expand=True)
        
        self.proc_merge = None
        self.stop_merge_requested = False

    def run_merge(self):
        self.btn_merge.config(state='disabled')
        self.btn_stop_merge.config(state='normal')
        self.stop_merge_requested = False
        threading.Thread(target=self._merge_thread, daemon=True).start()

    def stop_merge(self):
        self.stop_merge_requested = True
        proc = self.proc_merge
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_merge = None
            self.log_to_widget(self.log_merge, get_text('msg_stop'))

    def _merge_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.proc_merge = p
            if self.stop_merge_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_merge_tool(
                self.merge_l.get(),
                self.merge_r.get(),
                self.merge_keep_bitrate.get(),
                lambda msg: self.log_to_widget(self.log_merge, msg),
                _on_proc
            )
        except Exception as e:
            self.log_to_widget(self.log_merge, f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_to_widget(self.log_merge, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_merge.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop_merge.config(state='disabled'))
            self.proc_merge = None

if __name__ == "__main__":
    root = tk.Tk()
    app = VRMosaicOneClickApp(root)
    root.mainloop()
