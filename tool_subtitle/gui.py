import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import gc
import threading
import time
import sys

from tool_subtitle import logic
from utils import app_config, i18n

import locale

# --- i18n Setup ---


def get_text(key):
    return i18n.translate('subtitle', key)


def _load_keyring():
    import keyring
    return keyring


class SubtitleToolsApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text('title'))
        
        # Determine models root based on persistent directory, not PyInstaller's temp extraction dir (_MEIPASS)
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.models_root = os.path.join(base_dir, "models")
        
        # Thread control events
        self.stop_event_gen = threading.Event()
        self.proc_srt = None
        self.stop_srt_requested = False
        self.proc_rm = None
        self.stop_rm_requested = False

        self.setup_ui()

        # Check dependencies
        import shutil
        missing = []
        for tool in ["ffmpeg", "ffprobe"]:
            if not shutil.which(tool):
                missing.append(tool)
        if missing:
             self.log(self.gen_log, get_text('warn_dep').format(', '.join(missing)))
             self.log(self.gen_log, get_text('warn_path'))

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(header_frame, text=get_text('title'), font=('Arial', 14, 'bold')).pack(side='left')
        
        if self.on_return:
            ttk.Button(header_frame, text=get_text('btn_return'), command=self.on_return).pack(side='right')

        # Tabs
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 8], font=('Arial', 10, 'bold'))

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Tab 1: Generate Subtitles
        self.tab_generate = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_generate, text=get_text('tab_generate'))
        self.setup_generate_tab()

        # Tab 2: Subtitle Translation
        self.tab_trans = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_trans, text=get_text('tab_trans'))
        self.setup_trans_tab()
        
        # Tab 3: SRT to ASS
        self.tab_ass = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_ass, text=get_text('tab_ass'))
        self.setup_ass_tab()

        # Tab 4: Batch Add Soft Subtitles
        self.tab_srt = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_srt, text=get_text('tab_srt'))
        self.setup_srt_tab()

        # Tab 4.5: Remove Soft Subtitles
        self.tab_rm_sub = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_rm_sub, text=get_text('tab_rm_sub'))
        self.setup_rm_sub_tab()
        
        # Tab 5: Rank Subtitles
        self.tab_rank = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_rank, text=get_text('tab_rank'))
        self.setup_rank_tab()
        
    def log(self, text_widget, message):
        def _log():
            text_widget.config(state='normal')
            text_widget.insert('end', message + "\n")
            text_widget.see('end')
            text_widget.config(state='disabled')
        self.root.after(0, _log)

    # --- Generate Subtitles Tab ---
    def setup_generate_tab(self):
        frame = self.tab_generate
        
        # Directory Selection
        dir_frame = ttk.LabelFrame(frame, text=get_text('lbl_input_dir'), padding=10)
        dir_frame.pack(fill='x', pady=5)
        
        self.gen_dir_path = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.gen_dir_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(dir_frame, text=get_text('btn_browse'), command=self.browse_gen_dir).pack(side='right')

        # Options
        opt_frame = ttk.Frame(frame, padding=5)
        opt_frame.pack(fill='x', pady=5)
        
        self.gen_search_subdirs = tk.BooleanVar(value=True)
        self.gen_skip_if_exists = tk.BooleanVar(value=True)
        
        chk_frame = ttk.Frame(opt_frame)
        chk_frame.pack(fill='x', pady=2)
        ttk.Checkbutton(chk_frame, text=get_text('chk_search_subdirs'), variable=self.gen_search_subdirs).pack(side='left', padx=(0, 20))
        ttk.Checkbutton(chk_frame, text=get_text('chk_skip_if_exists'), variable=self.gen_skip_if_exists).pack(side='left')

        # Denoise Level
        denoise_frame = ttk.Frame(opt_frame)
        denoise_frame.pack(fill='x', pady=(10, 5))
        ttk.Label(denoise_frame, text=get_text('lbl_denoise')).pack(side='left', padx=(0, 10))
        
        self.denoise_mapping = {
            get_text('opt_none'): "none",
            get_text('opt_mild'): "mild",
            get_text('opt_balanced'): "balanced",
            get_text('opt_strong'): "strong"
        }
        self.denoise_var = tk.StringVar(value=get_text('opt_mild'))
        denoise_cb = ttk.Combobox(denoise_frame, textvariable=self.denoise_var, values=list(self.denoise_mapping.keys()), state="readonly", width=30)
        denoise_cb.pack(side='left')

        # Segmentation Model Selection
        seg_model_frame = ttk.Frame(opt_frame)
        seg_model_frame.pack(fill='x', pady=5)
        ttk.Label(seg_model_frame, text=get_text('lbl_seg_model')).pack(side='left', padx=(0, 10))
        
        self.seg_model_var = tk.StringVar(value="whisperSeg")
        ttk.Radiobutton(seg_model_frame, text=get_text('opt_whisperseg'), variable=self.seg_model_var, value="whisperSeg", command=self.check_model_status).pack(side='left', padx=5)

        self.btn_download_seg_model = ttk.Button(seg_model_frame, text=get_text('btn_download'), command=self.download_seg_model)
        
        # Model Selection
        model_frame = ttk.Frame(opt_frame)
        model_frame.pack(fill='x', pady=5)
        ttk.Label(model_frame, text=get_text('lbl_model')).pack(side='left', padx=(0, 10))
        
        self.model_var = tk.StringVar(value="large-v3")
        
        ttk.Radiobutton(model_frame, text=get_text('opt_kotoba'), variable=self.model_var, value="kotoba", command=self.check_model_status).pack(side='left', padx=5)
        ttk.Radiobutton(model_frame, text=get_text('opt_large_v3'), variable=self.model_var, value="large-v3", command=self.check_model_status).pack(side='left', padx=5)
        ttk.Radiobutton(model_frame, text=get_text('opt_large_v2'), variable=self.model_var, value="large-v2", command=self.check_model_status).pack(side='left', padx=5)
        
        self.btn_download_model = ttk.Button(model_frame, text=get_text('btn_download'), command=self.download_current_model)
        self.btn_download_model.pack(side='left', padx=10)
        
        # Initial model check
        self.root.after(100, self.check_model_status)

        # GPU Option
        gpu_frame = ttk.Frame(opt_frame)
        gpu_frame.pack(fill='x', pady=5)
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.chk_gpu = ttk.Checkbutton(gpu_frame, text=get_text('chk_use_gpu'), variable=self.use_gpu_var)
        self.chk_gpu.pack(side='left', padx=5)
        
        def run_check_cuda():
            self.btn_check_cuda.config(state='disabled')
            language = app_config.get_language()
            checking_text = {
                'en': " (Checking...)",
                'zh': " (环境检测中...)",
                'ja': " (確認中...)",
            }.get(language, " (Checking...)")
            self.chk_gpu.config(text=get_text('chk_use_gpu') + checking_text)
            
            def check_cuda():
                has_cuda = False
                missing_dlls = []
                try:
                    import ctranslate2
                    if ctranslate2.get_cuda_device_count() > 0:
                        has_cuda = True
                except Exception:
                    pass
                    
                # If basic CUDA is present, verify dlls on Windows
                if has_cuda and sys.platform.startswith('win'):
                    import ctypes.util
                    import ctranslate2
                    
                    def log_thread(msg):
                        self.root.after(0, lambda m=msg: self.log(self.gen_log, m))

                    def check_any_dll(prefix, suffixes):
                        import ctypes.util
                        for s in suffixes:
                            name = f'{prefix}{s}'
                            log_thread(f"System checking DLL: {name} ...")
                            try:
                                ctypes.CDLL(f'{name}.dll')
                                log_thread(f"  [OK] Loaded via CDLL (local directory).")
                                return True
                            except Exception as e:
                                log_thread(f"  [CDLL Load Error] {e}")
                            
                            path = ctypes.util.find_library(name)
                            if path:
                                log_thread(f"  [OK] Found via find_library (system PATH): {path}")
                                return True
                            else:
                                log_thread(f"  [find_library] Not found in PATH.")
                        return False

                    language = app_config.get_language()
                    if language == 'zh':
                        req_v12 = "(需CUDA 12+)"
                        req_v9 = "(需v9+)"
                    elif language == 'ja':
                        req_v12 = "(CUDA12+必要)"
                        req_v9 = "(v9+必要)"
                    else:
                        req_v12 = "(requires CUDA 12+)"
                        req_v9 = "(requires v9+)"
                    
                    if not check_any_dll('cublas64_', ['12', '13', '14', '15', '16']):
                        missing_dlls.append(f"cuBLAS{req_v12}")
                    
                    has_cudnn = check_any_dll('cudnn_cnn64_', ['9', '10', '11', '12', '13']) or \
                                check_any_dll('cudnn_cnn_infer64_', [ '9', '10', '11', '12', '13'])
                    if not has_cudnn:
                        missing_dlls.append(f"cuDNN{req_v9}")
                    
                    if missing_dlls:
                        has_cuda = False
                
                def update_ui():
                    self.btn_check_cuda.config(state='normal')
                    language = app_config.get_language()
                    if has_cuda:
                        ok_text = {
                            'en': " [CUDA detected]",
                            'zh': " [已检测到可用 CUDA]",
                            'ja': " [CUDA検出]",
                        }.get(language, " [CUDA detected]")
                        self.chk_gpu.config(text=get_text('chk_use_gpu') + ok_text)
                    else:
                        if missing_dlls:
                            if language == 'zh':
                                err_msg = f" [缺少 {'/'.join(missing_dlls)}，可能回退CPU]"
                            elif language == 'ja':
                                err_msg = f" [不足 {'/'.join(missing_dlls)}、CPUへ戻る可能性]"
                            else:
                                err_msg = f" [Missing {'/'.join(missing_dlls)}, may fallback to CPU]"
                            self.chk_gpu.config(text=get_text('chk_use_gpu') + err_msg)
                            self.show_cuda_download_dialog()
                        else:
                            if language == 'zh':
                                err_msg = " [未检测到 NVIDIA 显卡或 CUDA，已回退CPU]"
                            elif language == 'ja':
                                err_msg = " [NVIDIA GPU/CUDAなし、CPUへ]"
                            else:
                                err_msg = " [NVIDIA GPU or CUDA not found, fallback to CPU]"
                            self.chk_gpu.config(text=get_text('chk_use_gpu') + err_msg)
                        
                self.root.after(0, update_ui)
                
            threading.Thread(target=check_cuda, daemon=True).start()

        self.btn_check_cuda = ttk.Button(gpu_frame, text=get_text('btn_check_cuda'), command=run_check_cuda)
        self.btn_check_cuda.pack(side='left', padx=5)

        # Action Buttons
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_start_gen = ttk.Button(btn_frame, text=get_text('btn_start_gen'), command=self.run_gen)
        self.btn_start_gen.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_gen = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_gen, state='disabled')
        self.btn_stop_gen.pack(side='left', fill='x', expand=True, padx=(5, 0))

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.gen_log = tk.Text(log_frame, height=12, state='disabled')
        self.gen_log.pack(fill='both', expand=True)

    def show_cuda_download_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title(get_text('cuda_dialog_title'))
        dialog.geometry("550x320")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')

        content_frame = ttk.Frame(dialog, padding=20)
        content_frame.pack(fill='both', expand=True)

        msg_lbl = ttk.Label(content_frame, text=get_text('cuda_dialog_msg'), wraplength=500, justify='left')
        msg_lbl.pack(fill='both', expand=True, pady=(0, 20))

        btn_frame = ttk.Frame(content_frame)
        btn_frame.pack(fill='x', side='bottom')

        def open_github():
            import webbrowser
            webbrowser.open("https://github.com/Purfview/whisper-standalone-win/releases/tag/libs")

        def open_root():
            if sys.platform.startswith('win'):
                base_dir = os.path.dirname(self.models_root)
                os.startfile(base_dir)

        ttk.Button(btn_frame, text=get_text('btn_open_github'), command=open_github).pack(side='left', padx=(30, 10), fill='x', expand=True)
        ttk.Button(btn_frame, text=get_text('btn_open_root'), command=open_root).pack(side='right', padx=(10, 30), fill='x', expand=True)

    def browse_gen_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.gen_dir_path.set(path)
            
    def check_model_status(self):
        model_key = self.model_var.get()
        seg_model_key = getattr(self, 'seg_model_var', tk.StringVar(value="whisperSeg")).get()
        
        if logic.check_model_files(model_key, self.models_root):
            self.btn_download_model.pack_forget()
        else:
            self.btn_download_model.pack(side='left', padx=10)
            
        if hasattr(self, 'btn_download_seg_model'):
            if logic.check_model_files(seg_model_key, self.models_root):
                self.btn_download_seg_model.pack_forget()
            else:
                self.btn_download_seg_model.pack(side='left', padx=10)
            
    def download_current_model(self):
        model_key = self.model_var.get()
        
        def task():
            self.btn_download_model.config(state='disabled')
            self.btn_start_gen.config(state='disabled')
            
            success = logic.download_model(model_key, self.models_root, lambda msg: self.log(self.gen_log, msg))
            
            self.root.after(0, lambda: self.btn_start_gen.config(state='normal'))
            if success:
                self.root.after(0, self.check_model_status)
            else:
                self.root.after(0, lambda: self.btn_download_model.config(state='normal'))
                
        threading.Thread(target=task, daemon=True).start()

    def download_seg_model(self):
        seg_model_key = getattr(self, 'seg_model_var', tk.StringVar(value="whisperSeg")).get()
        
        def task():
            self.btn_download_seg_model.config(state='disabled')
            self.btn_start_gen.config(state='disabled')
            
            success = logic.download_model(seg_model_key, self.models_root, lambda msg: self.log(self.gen_log, msg))
            
            self.root.after(0, lambda: self.btn_start_gen.config(state='normal'))
            if success:
                self.root.after(0, self.check_model_status)
            else:
                self.root.after(0, lambda: self.btn_download_seg_model.config(state='normal'))
                
        threading.Thread(target=task, daemon=True).start()

    def run_gen(self):
        base_dir = self.gen_dir_path.get()
        if not base_dir or not os.path.exists(base_dir):
            messagebox.showerror("Error", get_text('err_dir'))
            return
            
        denoise_preset = self.denoise_mapping.get(self.denoise_var.get(), "mild")
        model_key = self.model_var.get()
            
        def task():
            start_time = time.time()
            self.log(self.gen_log, get_text('msg_start_gen'))
            self.stop_event_gen.clear()
            # gen_holder lets us move the WhisperModel reference back to the main
            # thread for cleanup, avoiding a CTranslate2 native crash (ucrtbase.dll
            # 0xc0000409) that occurs when the C++ destructor runs on a background thread.
            gen_holder = []
            try:
                logic.batch_generate_srt(
                    base_dir=base_dir,
                    search_subdirs=self.gen_search_subdirs.get(),
                    skip_if_exists=self.gen_skip_if_exists.get(),
                    denoise_preset=denoise_preset,
                    model_key=model_key,
                    models_root=self.models_root,
                    use_gpu=self.use_gpu_var.get(),
                    log_callback=lambda msg: self.log(self.gen_log, msg),
                    stop_event=self.stop_event_gen,
                    gen_holder=gen_holder
                )
                if not self.stop_event_gen.is_set():
                    self.log(self.gen_log, get_text('msg_done'))
            except Exception as e:
                self.log(self.gen_log, f"Error: {e}")
            finally:
                # Schedule model cleanup on the main thread to prevent native crash
                def cleanup_model():
                    gen_holder.clear()
                    gc.collect()
                self.root.after(0, cleanup_model)
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(self.gen_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_start_gen.config(state='normal'))
                self.root.after(0, lambda: self.btn_stop_gen.config(state='disabled'))

        self.btn_start_gen.config(state='disabled')
        self.btn_stop_gen.config(state='normal')
        threading.Thread(target=task, daemon=True).start()

    def stop_gen(self):
        self.stop_event_gen.set()
        self.log(self.gen_log, get_text('msg_stop_wait'))
        self.btn_stop_gen.config(state='disabled')
        # Re-enable start immediately so user can restart or knows it will stop
        self.btn_start_gen.config(state='normal')


    # --- Batch Add SRT Tab ---
    def setup_srt_tab(self):
        frame = self.tab_srt
        
        # Info text
        info_lbl = ttk.Label(frame, text=get_text('lbl_srt_info'), wraplength=700, justify='left', foreground='dim gray')
        info_lbl.pack(fill='x', pady=(0, 10))

        # Directory Selection
        dir_frame = ttk.LabelFrame(frame, text=get_text('lbl_input_dir'), padding=10)
        dir_frame.pack(fill='x', pady=5)
        
        self.srt_dir_path = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.srt_dir_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(dir_frame, text=get_text('btn_browse'), command=self.browse_srt_dir).pack(side='right')

        # Options
        opt_frame = ttk.Frame(frame, padding=5)
        opt_frame.pack(fill='x', pady=5)
        
        self.srt_search_subdirs = tk.BooleanVar(value=True)
        self.srt_replace_original = tk.BooleanVar(value=False)
        self.srt_auto_load = tk.BooleanVar(value=True)
        self.srt_skip_if_has_sub = tk.BooleanVar(value=True)
        self.srt_prefer_ass = tk.BooleanVar(value=False)

        ttk.Checkbutton(opt_frame, text=get_text('chk_search_subdirs_srt'), variable=self.srt_search_subdirs).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_replace_original'), variable=self.srt_replace_original).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_auto_load_srt'), variable=self.srt_auto_load).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_skip_if_has_sub'), variable=self.srt_skip_if_has_sub).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_prefer_ass'), variable=self.srt_prefer_ass).pack(anchor='w', pady=2)

        # Action Buttons
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_start_srt = ttk.Button(btn_frame, text=get_text('btn_start_srt'), command=self.run_srt)
        self.btn_start_srt.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_srt = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_srt, state='disabled')
        self.btn_stop_srt.pack(side='left', fill='x', expand=True, padx=(5, 0))

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.srt_log = tk.Text(log_frame, height=12, state='disabled')
        self.srt_log.pack(fill='both', expand=True)

    def browse_srt_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.srt_dir_path.set(path)

    def run_srt(self):
        base_dir = self.srt_dir_path.get()
        if not base_dir or not os.path.exists(base_dir):
            messagebox.showerror("Error", get_text('err_dir'))
            return
            
        def _on_proc(p):
            self.proc_srt = p
            if self.stop_srt_requested:
                try: p.kill()
                except Exception: pass

        def task():
            start_time = time.time()
            self.log(self.srt_log, get_text('msg_start_srt'))
            try:
                logic.batch_add_srt(
                    base_dir=base_dir,
                    search_subdirs=self.srt_search_subdirs.get(),
                    replace_original=self.srt_replace_original.get(),
                    auto_load_srt=self.srt_auto_load.get(),
                    skip_if_has_sub=self.srt_skip_if_has_sub.get(),
                    prefer_ass=self.srt_prefer_ass.get(),
                    log_callback=lambda msg: self.log(self.srt_log, msg),
                    process_callback=_on_proc
                )
                self.log(self.srt_log, get_text('msg_done'))
            except Exception as e:
                self.log(self.srt_log, f"Error: {e}")
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(self.srt_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_start_srt.config(state='normal'))
                self.root.after(0, lambda: self.btn_stop_srt.config(state='disabled'))
                self.proc_srt = None

        self.btn_start_srt.config(state='disabled')
        self.btn_stop_srt.config(state='normal')
        self.stop_srt_requested = False
        threading.Thread(target=task, daemon=True).start()

    def stop_srt(self):
        self.stop_srt_requested = True
        proc = self.proc_srt
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_srt = None
            self.log(self.srt_log, get_text('msg_stop'))

    # --- Translation Tab ---
    def setup_trans_tab(self):
        frame = ttk.Frame(self.tab_trans, padding=10)
        frame.pack(fill='both', expand=True)

        # Input Directory
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill='x', pady=5)
        ttk.Label(dir_frame, text=get_text('lbl_input_dir')).pack(side='left', padx=(0, 10))
        self.trans_dir_path = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.trans_dir_path).pack(side='left', fill='x', expand=True, padx=(0, 10))
        ttk.Button(dir_frame, text=get_text('btn_browse'), command=self.browse_trans_dir).pack(side='left')

        # Load config
        self.trans_config = logic.load_trans_config()

        # AI Configuration Group
        ai_frame = ttk.LabelFrame(frame, text=get_text('grp_ai_config'), padding=10)
        ai_frame.pack(fill='x', pady=5)

        # API URL & Model Name (Side by Side)
        url_model_frame = ttk.Frame(ai_frame)
        url_model_frame.pack(fill='x', pady=2)
        
        ttk.Label(url_model_frame, text=get_text('lbl_api_url'), width=15).pack(side='left')
        self.api_url_var = tk.StringVar(value=self.trans_config.get('api_base_url', ''))
        ttk.Entry(url_model_frame, textvariable=self.api_url_var).pack(side='left', fill='x', expand=True)

        ttk.Label(url_model_frame, text=get_text('lbl_model_name')).pack(side='left', padx=(20, 5))
        self.model_name_var = tk.StringVar(value=self.trans_config.get('model_name', ''))
        ttk.Entry(url_model_frame, textvariable=self.model_name_var, width=25).pack(side='left')

        # API Key
        key_frame = ttk.Frame(ai_frame)
        key_frame.pack(fill='x', pady=2)
        ttk.Label(key_frame, text=get_text('lbl_api_key'), width=20).pack(side='left')

        self.api_key_var = tk.StringVar(value="")
        self.api_key_entry = ttk.Entry(key_frame, textvariable=self.api_key_var, show="*")
        self.api_key_entry.pack(side='left', fill='x', expand=True)
        
        self.btn_test_api = ttk.Button(key_frame, text=get_text('btn_test_api'), command=self.test_trans_api)
        self.btn_del_key = ttk.Button(key_frame, text=get_text('btn_delete_key'), command=self.delete_trans_key)
        self.btn_test_api.pack(side='left', padx=(10, 0))

        # Max Tokens & Save config button
        token_frame = ttk.Frame(ai_frame)
        token_frame.pack(fill='x', pady=2)
        ttk.Label(token_frame, text=get_text('lbl_tokens'), width=20).pack(side='left')
        self.tokens_var = tk.StringVar(value=str(self.trans_config.get('tokens_per_chunk', '500000')))
        ttk.Entry(token_frame, textvariable=self.tokens_var).pack(side='left')
        
        ttk.Button(token_frame, text=get_text('btn_save_config'), command=self.save_trans_config).pack(side='left', padx=(20, 0))

        # Translation Options Group
        opt_frame = ttk.LabelFrame(frame, text=get_text('grp_trans_opt'), padding=10)
        opt_frame.pack(fill='x', pady=5)

        # Target Language
        lang_frame = ttk.Frame(opt_frame)
        lang_frame.pack(fill='x', pady=2)
        ttk.Label(lang_frame, text=get_text('lbl_target_lang')).pack(side='left', padx=(0, 10))
        
        self.lang_var = tk.StringVar()
        self.lang_custom_var = tk.StringVar()
        self.lang_custom_entry = ttk.Entry(lang_frame, textvariable=self.lang_custom_var)
        
        def on_lang_change(*args):
            if self.lang_var.get() == get_text('opt_lang_other'):
                self.lang_custom_entry.pack(side='left', padx=(10, 0))
            else:
                self.lang_custom_entry.pack_forget()
                
        self.lang_var.trace_add("write", on_lang_change)
        
        langs = [get_text('opt_lang_zh'), get_text('opt_lang_en'), get_text('opt_lang_other')]
        target_lang = self.trans_config.get('target_language', 'Chinese')
        
        if target_lang == 'Chinese' or target_lang == '中文':
            self.lang_var.set(get_text('opt_lang_zh'))
        elif target_lang == 'English' or target_lang == '英文':
            self.lang_var.set(get_text('opt_lang_en'))
        else:
            self.lang_var.set(get_text('opt_lang_other'))
            self.lang_custom_var.set(target_lang)

        ttk.Combobox(lang_frame, textvariable=self.lang_var, values=langs, state="readonly", width=15).pack(side='left')

        # Checkboxes
        self.trans_search_subdirs = tk.BooleanVar(value=True)
        self.trans_skip_if_exists = tk.BooleanVar(value=True)
        self.trans_keep_orig = tk.BooleanVar(value=self.trans_config.get('keep_original', True))
        self.trans_adult_content = tk.BooleanVar(value=self.trans_config.get('adult_content', True))
        
        ttk.Checkbutton(opt_frame, text=get_text('chk_search_subdirs_trans'), variable=self.trans_search_subdirs).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_skip_if_exists_trans'), variable=self.trans_skip_if_exists).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_keep_orig'), variable=self.trans_keep_orig, command=self.on_option_toggled).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_adult_content'), variable=self.trans_adult_content, command=self.on_option_toggled).pack(anchor='w', pady=2)

        # Action Buttons
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_start_trans = ttk.Button(btn_frame, text=get_text('btn_start_trans'), command=self.run_trans)
        self.btn_start_trans.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_trans = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_trans, state='disabled')
        self.btn_stop_trans.pack(side='left', fill='x', expand=True, padx=(5, 0))

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.trans_log = tk.Text(log_frame, height=12, state='disabled')
        self.trans_log.pack(fill='both', expand=True)
        
        self.stop_event_trans = threading.Event()
        self.root.after(0, self.load_saved_trans_key)

    def load_saved_trans_key(self):
        def task():
            try:
                keyring = _load_keyring()
                saved_key = (
                    keyring.get_password("VR_Video_Toolbox", "deepseek_api_key")
                    or keyring.get_password("VR_Mosaic_Removal", "deepseek_api_key")
                    or ""
                )
            except Exception:
                saved_key = ""

            if not saved_key:
                return

            def update_ui():
                try:
                    if not self.api_key_entry.winfo_exists():
                        return
                    if not self.api_key_var.get():
                        self.api_key_var.set(saved_key)
                    self.btn_test_api.pack_forget()
                    if not self.btn_del_key.winfo_ismapped():
                        self.btn_del_key.pack(side='left', padx=(10, 0))
                except tk.TclError:
                    pass

            try:
                self.root.after(0, update_ui)
            except tk.TclError:
                pass

        threading.Thread(target=task, daemon=True).start()

    def browse_trans_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.trans_dir_path.set(path)

    def on_option_toggled(self):
        self.trans_config['keep_original'] = self.trans_keep_orig.get()
        self.trans_config['adult_content'] = self.trans_adult_content.get()
        logic.save_trans_config(self.trans_config)

    def save_trans_config(self):
        # Update config dict
        self.trans_config['api_base_url'] = self.api_url_var.get()
        self.trans_config['model_name'] = self.model_name_var.get()
        self.trans_config['tokens_per_chunk'] = int(self.tokens_var.get())
        
        lang = self.lang_var.get()
        if lang == get_text('opt_lang_other'):
            self.trans_config['target_language'] = self.lang_custom_var.get()
        else:
            self.trans_config['target_language'] = "Chinese" if lang == get_text('opt_lang_zh') else "English"
            
        if logic.save_trans_config(self.trans_config):
            messagebox.showinfo("Success", get_text('msg_config_saved'))
            
    def delete_trans_key(self):
        try:
            keyring = _load_keyring()
            for service_name in ("VR_Video_Toolbox", "VR_Mosaic_Removal"):
                try:
                    keyring.delete_password(service_name, "deepseek_api_key")
                except Exception:
                    pass
            self.api_key_var.set("")
            self.btn_del_key.pack_forget()
            self.btn_test_api.pack(side='left', padx=(10, 0))
            messagebox.showinfo("Success", get_text('msg_key_deleted'))
        except Exception as e:
            messagebox.showerror("Error", get_text('msg_key_del_warn').format(e))

    def test_trans_api(self):
        api_url = self.api_url_var.get()
        model_name = self.model_name_var.get()
        api_key = self.api_key_var.get()
        
        if not api_key:
            messagebox.showerror("Error", get_text('err_no_api_key'))
            return
            
        def test_task():
            try:
                client = logic.LLMClient(api_url, api_key, model_name, temperature=0.5)
                response = client.complete("Say 'Hello' or '你好' only, nothing else.")
                
                # Save the key to keyring only after the test succeeds.
                try:
                    keyring = _load_keyring()
                    keyring.set_password("VR_Video_Toolbox", "deepseek_api_key", api_key)
                    def update_ui_success():
                        self.btn_test_api.pack_forget()
                        self.btn_del_key.pack(side='left', padx=(10, 0))
                        messagebox.showinfo("Success", get_text('msg_api_test_success').format(response))
                    self.root.after(0, update_ui_success)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showwarning("Warning", f"Could not save API Key to keyring: {e}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", get_text('msg_api_test_fail').format(e)))

        threading.Thread(target=test_task, daemon=True).start()

    def run_trans(self):
        base_dir = self.trans_dir_path.get()
        if not base_dir or not os.path.exists(base_dir):
            messagebox.showerror("Error", get_text('err_dir'))
            return
            
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showerror("Error", "Please provide an API Key.")
            return
            
        # Sync current transient options
        lang = self.lang_var.get()
        if lang == get_text('opt_lang_other'):
            self.trans_config['target_language'] = self.lang_custom_var.get()
        else:
            self.trans_config['target_language'] = "Chinese" if lang == get_text('opt_lang_zh') else "English"
        self.trans_config['keep_original'] = self.trans_keep_orig.get()
        self.trans_config['adult_content'] = self.trans_adult_content.get()

        def task():
            start_time = time.time()
            self.log(self.trans_log, get_text('msg_start_trans'))
            self.stop_event_trans.clear()
            try:
                logic.batch_translate_srt(
                    base_dir=base_dir,
                    search_subdirs=self.trans_search_subdirs.get(),
                    skip_if_exists=self.trans_skip_if_exists.get(),
                    api_key=api_key,
                    config=self.trans_config,
                    log_callback=lambda msg: self.log(self.trans_log, msg),
                    stop_event=self.stop_event_trans
                )
                if not self.stop_event_trans.is_set():
                    self.log(self.trans_log, get_text('msg_done'))
            except Exception as e:
                self.log(self.trans_log, f"Error: {e}")
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(self.trans_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_start_trans.config(state='normal'))
                self.root.after(0, lambda: self.btn_stop_trans.config(state='disabled'))

        self.btn_start_trans.config(state='disabled')
        self.btn_stop_trans.config(state='normal')
        threading.Thread(target=task, daemon=True).start()

    def stop_trans(self):
        self.stop_event_trans.set()
        self.log(self.trans_log, get_text('msg_stop'))
        self.btn_stop_trans.config(state='disabled')

    # ===============================
    # SRT to ASS Methods
    # ===============================
    def setup_ass_tab(self):
        frame = self.tab_ass
        
        # Info label
        info_lbl = ttk.Label(frame, text=get_text('lbl_ass_info'), wraplength=700, justify='left', foreground='dim gray')
        info_lbl.pack(fill='x', pady=(0, 10))

        # Input Directory
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill='x', pady=5)
        ttk.Label(dir_frame, text=get_text('lbl_input_dir')).pack(side='left')
        self.ass_dir_path = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.ass_dir_path).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(dir_frame, text=get_text('btn_browse'), command=self.browse_ass_dir).pack(side='left')

        # Alignment
        opt_frame = ttk.LabelFrame(frame, text=get_text('grp_ass_opt'), padding=10)
        opt_frame.pack(fill='x', pady=5)
        
        align_frame = ttk.Frame(opt_frame)
        align_frame.pack(fill='x', pady=2)
        ttk.Label(align_frame, text=get_text('lbl_align'), width=20).pack(side='left')
        self.ass_align_mapping = {
            get_text('opt_align_bot'): 2,
            get_text('opt_align_mid'): 5,
            get_text('opt_align_top'): 8
        }
        self.ass_align_var = tk.StringVar(value=get_text('opt_align_bot'))
        ttk.Combobox(align_frame, textvariable=self.ass_align_var, values=list(self.ass_align_mapping.keys()), state="readonly", width=15).pack(side='left')

        # Font sizes
        size_frame = ttk.Frame(opt_frame)
        size_frame.pack(fill='x', pady=2)
        ttk.Label(size_frame, text=get_text('lbl_cn_size'), width=20).pack(side='left')
        self.ass_cn_size_var = tk.StringVar(value="42")
        ttk.Combobox(size_frame, textvariable=self.ass_cn_size_var, values=["26", "34", "38", "40", "42", "44", "46", "50", "58", "74", "84"], width=15).pack(side='left')

        ttk.Label(size_frame, text=get_text('lbl_jp_size'), width=16).pack(side='left', padx=(20, 0))
        self.ass_jp_size_var = tk.StringVar(value="30")
        ttk.Combobox(size_frame, textvariable=self.ass_jp_size_var, values=["15", "22", "26", "28", "30", "32", "34", "38", "46", "60"], width=15).pack(side='left')

        self.ass_color_mapping = {
            get_text('opt_color_white_black'): ("&H00FFFFFF", "&H00000000"),
            get_text('opt_color_black_white'): ("&H00000000", "&H00FFFFFF"),
            get_text('opt_color_green_black'): ("&H005AFF65", "&H00000000"),
            get_text('opt_color_yellow_black'): ("&H0000FFFF", "&H00000000"),
            get_text('opt_color_red_black'): ("&H000000FF", "&H00000000"),
        }

        # Color presets
        color_frame = ttk.Frame(opt_frame)
        color_frame.pack(fill='x', pady=2)
        ttk.Label(color_frame, text=get_text('lbl_default_color'), width=20).pack(side='left')
        self.ass_default_color_var = tk.StringVar(value=get_text('opt_color_green_black'))
        ttk.Combobox(color_frame, textvariable=self.ass_default_color_var, values=list(self.ass_color_mapping.keys()), state="readonly", width=18).pack(side='left')

        ttk.Label(color_frame, text=get_text('lbl_secondary_color'), width=16).pack(side='left', padx=(20, 0))
        self.ass_secondary_color_var = tk.StringVar(value=get_text('opt_color_white_black'))
        ttk.Combobox(color_frame, textvariable=self.ass_secondary_color_var, values=list(self.ass_color_mapping.keys()), state="readonly", width=18).pack(side='left')

        # Checkboxes
        self.ass_search_subdirs = tk.BooleanVar(value=True)
        self.ass_skip_exists = tk.BooleanVar(value=True)
        self.ass_only_bilingual = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(opt_frame, text=get_text('chk_search_subdirs_ass'), variable=self.ass_search_subdirs).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_skip_ass_exists'), variable=self.ass_skip_exists).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_frame, text=get_text('chk_only_bilingual'), variable=self.ass_only_bilingual).pack(anchor='w', pady=2)

        # Action Buttons
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_start_ass = ttk.Button(btn_frame, text=get_text('btn_start_ass'), command=self.run_ass)
        self.btn_start_ass.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_ass = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_ass, state='disabled')
        self.btn_stop_ass.pack(side='left', fill='x', expand=True, padx=(5, 0))

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.ass_log = tk.Text(log_frame, height=12, state='disabled')
        self.ass_log.pack(fill='both', expand=True)
        
        self.stop_event_ass = threading.Event()

    def browse_ass_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.ass_dir_path.set(path)

    def run_ass(self):
        input_dir = self.ass_dir_path.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", get_text('err_dir'))
            return
            
        align_val = self.ass_align_mapping.get(self.ass_align_var.get(), 2)
        cn_size = int(self.ass_cn_size_var.get())
        jp_size = int(self.ass_jp_size_var.get())
        search_subdirs = self.ass_search_subdirs.get()
        skip_exists = self.ass_skip_exists.get()
        only_bilingual = self.ass_only_bilingual.get()
        default_primary_colour, default_outline_colour = self.ass_color_mapping.get(
            self.ass_default_color_var.get(),
            ("&H005AFF65", "&H00000000")
        )
        secondary_primary_colour, secondary_outline_colour = self.ass_color_mapping.get(
            self.ass_secondary_color_var.get(),
            ("&H00FFFFFF", "&H00000000")
        )

        self.btn_start_ass.config(state='disabled')
        self.btn_stop_ass.config(state='normal')
        self.ass_log.config(state='normal')
        self.ass_log.delete('1.0', tk.END)
        self.ass_log.config(state='disabled')
        self.stop_event_ass.clear()

        def task():
            start_time = time.time()
            self.log(self.ass_log, get_text('msg_start_ass'))
            logic.batch_convert_srt_to_ass(
                input_dir,
                align_val,
                cn_size,
                jp_size,
                search_subdirs,
                skip_exists,
                only_bilingual,
                lambda msg: self.log(self.ass_log, msg),
                self.stop_event_ass,
                default_primary_colour,
                default_outline_colour,
                secondary_primary_colour,
                secondary_outline_colour
            )
            if self.stop_event_ass.is_set():
                self.log(self.ass_log, get_text('msg_stop'))
            else:
                self.log(self.ass_log, get_text('msg_done'))
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log(self.ass_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_start_ass.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop_ass.config(state='disabled'))

        threading.Thread(target=task, daemon=True).start()

    def stop_ass(self):
        self.stop_event_ass.set()
        self.btn_stop_ass.config(state='disabled')

    # ===============================
    # Remove Soft Subtitles Methods
    # ===============================
    def setup_rm_sub_tab(self):
        frame = self.tab_rm_sub
        frame.grid_columnconfigure(1, weight=1)

        # 0. Info / Note label (at top)
        info_lbl = ttk.Label(frame, text=get_text('lbl_rm_sub_note'), wraplength=700, justify='left', foreground='dim gray')
        info_lbl.grid(row=0, column=0, columnspan=2, padx=5, pady=(5, 0), sticky='w')

        # 1. Input Directory
        ttk.Label(frame, text=get_text('lbl_input_dir')).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.rm_base_dir = tk.StringVar()
        
        dir_frame = ttk.Frame(frame)
        dir_frame.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        ttk.Entry(dir_frame, textvariable=self.rm_base_dir).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(dir_frame, text=get_text('btn_browse'), command=lambda: self.rm_base_dir.set(filedialog.askdirectory() or self.rm_base_dir.get())).pack(side='right')

        # 2. Options
        opt_frame = ttk.Frame(frame)
        opt_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        self.rm_search_subdirs = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text=get_text('chk_rm_subdirs'), variable=self.rm_search_subdirs).pack(anchor='w', pady=2)
        
        self.rm_delete_mkv = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text=get_text('chk_delete_mkv'), variable=self.rm_delete_mkv).pack(anchor='w', pady=2)

        # 3. Action Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        self.btn_start_rm = ttk.Button(btn_frame, text=get_text('btn_rm_sub'), command=self.run_rm_sub)
        self.btn_start_rm.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_rm = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_rm_sub, state='disabled')
        self.btn_stop_rm.pack(side='left', fill='x', expand=True, padx=(5, 0))

        # 4. Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'))
        log_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        frame.grid_rowconfigure(4, weight=1)
        
        self.rm_log = tk.Text(log_frame, height=10, state='disabled')
        self.rm_log.pack(fill='both', expand=True)

        self.stop_event_rm = threading.Event()


    def run_rm_sub(self):
        base_dir = self.rm_base_dir.get()
        if not base_dir or not os.path.exists(base_dir):
            messagebox.showerror("Error", get_text('err_dir'))
            return

        def _on_proc(p):
            self.proc_rm = p
            if self.stop_rm_requested or self.stop_event_rm.is_set():
                try: p.kill()
                except Exception: pass

        def task():
            start_time = time.time()
            self.log(self.rm_log, get_text('msg_start_rm'))
            self.stop_event_rm.clear()
            try:
                logic.batch_remove_srt(
                    base_dir=base_dir,
                    search_subdirs=self.rm_search_subdirs.get(),
                    delete_mkv=self.rm_delete_mkv.get(),
                    log_callback=lambda msg: self.log(self.rm_log, msg),
                    process_callback=_on_proc
                )
            except Exception as e:
                self.log(self.rm_log, f"Error: {e}")
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(self.rm_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_start_rm.config(state='normal'))
                self.root.after(0, lambda: self.btn_stop_rm.config(state='disabled'))
                self.proc_rm = None

        self.btn_start_rm.config(state='disabled')
        self.btn_stop_rm.config(state='normal')
        self.stop_rm_requested = False
        threading.Thread(target=task, daemon=True).start()

    def stop_rm_sub(self):
        self.stop_rm_requested = True
        self.stop_event_rm.set()
        proc = self.proc_rm
        if proc:
            try: proc.kill()
            except Exception: pass
        self.log(self.rm_log, get_text('msg_stop_rm'))
        self.btn_stop_rm.config(state='disabled')
        self.btn_start_rm.config(state='normal')

    # ===============================
    # Rank Subtitles Methods
    # ===============================
    def setup_rank_tab(self):

        frame = self.tab_rank
        
        # Info label
        info_lbl = ttk.Label(frame, text=get_text('lbl_rank_info'), wraplength=700, justify='left', foreground='dim gray')
        info_lbl.pack(fill='x', pady=(0, 10))

        # Input Directory
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill='x', pady=5)
        ttk.Label(dir_frame, text=get_text('lbl_input_dir')).pack(side='left')
        self.rank_dir_path = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.rank_dir_path).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(dir_frame, text=get_text('btn_browse'), command=self.browse_rank_dir).pack(side='left')

        # Action Buttons
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_start_rank = ttk.Button(btn_frame, text=get_text('btn_start_rank'), command=self.run_rank)
        self.btn_start_rank.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_rank = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_rank, state='disabled')
        self.btn_stop_rank.pack(side='left', fill='x', expand=True, padx=(5, 0))

        # Table
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill='both', expand=True, pady=5)
        
        cols = ('col_rank', 'col_file', 'col_score', 'col_entries', 'col_span', 'col_cov', 'col_chars', 'col_jp', 'col_dup', 'col_gap')
        self.rank_tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        
        # Define headings and column widths
        widths = {
            'col_rank': 50,
            'col_score': 80,
            'col_entries': 80,
            'col_span': 80,
            'col_cov': 80,
            'col_chars': 100,
            'col_jp': 80,
            'col_dup': 80,
            'col_gap': 80,
            'col_file': 200
        }
        for col in cols:
            self.rank_tree.heading(col, text=get_text(col))
            self.rank_tree.column(col, width=widths.get(col, 100), anchor='center')
            
        self.rank_tree.column('col_file', anchor='w') # filename left-aligned
        
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.rank_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.rank_tree.xview)
        self.rank_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.rank_tree.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='both', pady=5)
        
        self.rank_log = tk.Text(log_frame, height=6, state='disabled')
        self.rank_log.pack(fill='both', expand=True)
        
        self.stop_event_rank = threading.Event()

    def browse_rank_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.rank_dir_path.set(path)

    def run_rank(self):
        input_dir = self.rank_dir_path.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", get_text('err_dir'))
            return
            
        self.btn_start_rank.config(state='disabled')
        self.btn_stop_rank.config(state='normal')
        self.rank_log.config(state='normal')
        self.rank_log.delete('1.0', tk.END)
        self.rank_log.config(state='disabled')
        self.stop_event_rank.clear()
        
        # Clear treeview
        for item in self.rank_tree.get_children():
            self.rank_tree.delete(item)

        def task():
            self.log(self.rank_log, get_text('msg_start_rank'))
            
            def insert_data(rows):
                # Update UI thread
                self.root.after(0, lambda: self._update_rank_tree(rows))
                
            logic.batch_rank_srt(
                input_dir,
                lambda msg: self.log(self.rank_log, msg),
                self.stop_event_rank,
                insert_data
            )
            
            if self.stop_event_rank.is_set():
                self.log(self.rank_log, get_text('msg_stop'))
            else:
                self.log(self.rank_log, get_text('msg_done'))
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log(self.rank_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_start_rank.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop_rank.config(state='disabled'))

        threading.Thread(target=task, daemon=True).start()

    def _update_rank_tree(self, rows):
        for index, row in enumerate(rows, start=1):
            values = (
                index,
                os.path.basename(row['file']),
                row['score'],
                row['entries'],
                row['span_min'],
                row['coverage'],
                row['chars_per_min'],
                row['jp_ratio'],
                row['duplicates'],
                row['large_gaps']
            )
            self.rank_tree.insert('', 'end', values=values)

    def stop_rank(self):
        self.stop_event_rank.set()
        self.btn_stop_rank.config(state='disabled')
