import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import sys
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
    return i18n.translate('split_combine', key)

class VRSplitCombineApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text('title'))
        
        # Configure layout
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill='both', expand=True)

        # Header Frame
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill='x', pady=(0, 10))
        
        # Title (Left)
        ttk.Label(header_frame, text=get_text('title'), font=('Arial', 14, 'bold')).pack(side='left')
        
        # Return Button (Right)
        if self.on_return:
            ttk.Button(header_frame, text=get_text('btn_back'), command=self.go_back).pack(side='right')

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Tab 1: Split
        self.tab_split = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_split, text=get_text('tab_split'))
        self.setup_split_tab()

        # Tab 2: Combine
        self.tab_combine = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_combine, text=get_text('tab_combine'))
        self.setup_combine_tab()

        # Log Area
        log_frame = ttk.LabelFrame(self.main_frame, text=get_text('log_title'), padding=5)
        log_frame.pack(fill='both', expand=True, pady=10)
        self.log_text = tk.Text(log_frame, height=10, state='disabled')
        self.log_text.pack(fill='both', expand=True, side='left')
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Check dependencies
        missing = logic.check_dependencies()
        if missing:
            self.log(get_text('warn_dep').format(', '.join(missing)))
            self.log(get_text('warn_path'))

    def go_back(self):
        if self.on_return:
            self.on_return()
        else:
            self.root.quit()

    def log(self, message):
        def _do():
            self.log_text.config(state='normal')
            self.log_text.insert('end', message + "\n")
            self.log_text.see('end')
            self.log_text.config(state='disabled')
        self.root.after(0, _do)

    def setup_split_tab(self):
        # Input File
        frame_input = ttk.Frame(self.tab_split)
        frame_input.pack(fill='x', pady=5)
        ttk.Label(frame_input, text=get_text('lbl_input_video')).pack(side='left')
        self.split_input_var = tk.StringVar()
        ttk.Entry(frame_input, textvariable=self.split_input_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_input, text=get_text('btn_browse'), command=lambda: self.browse_file(self.split_input_var)).pack(side='left')

        # Output Dir
        frame_output = ttk.Frame(self.tab_split)
        frame_output.pack(fill='x', pady=5)
        ttk.Label(frame_output, text=get_text('lbl_output_dir')).pack(side='left')
        self.split_output_var = tk.StringVar()
        ttk.Entry(frame_output, textvariable=self.split_output_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_output, text=get_text('btn_browse'), command=lambda: self.browse_dir(self.split_output_var)).pack(side='left')

        # Mode Selection
        frame_mode = ttk.LabelFrame(self.tab_split, text=get_text('grp_split_mode'), padding=10)
        frame_mode.pack(fill='x', pady=10)
        
        self.split_mode_var = tk.StringVar(value="left_and_right")
        
        modes = [
            (get_text('mode_left'), "left"),
            (get_text('mode_right'), "right"),
            (get_text('mode_lr_sep'), "left_and_right"),
            (get_text('mode_top'), "top"),
            (get_text('mode_bottom'), "bottom"),
            (get_text('mode_tb_sep'), "top_and_bottom"),
        ]
        
        for text, val in modes:
            ttk.Radiobutton(frame_mode, text=text, variable=self.split_mode_var, value=val).pack(anchor='w')

        # Fisheye Option
        self.split_fisheye_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tab_split, text=get_text('chk_split_fisheye'), variable=self.split_fisheye_var).pack(anchor='w', pady=5)

        # Run Buttons
        btn_frame = ttk.Frame(self.tab_split)
        btn_frame.pack(pady=10)
        
        self.btn_split_start = ttk.Button(btn_frame, text=get_text('btn_start'), command=self.run_split)
        self.btn_split_start.pack(side='left', padx=5)
        
        self.btn_split_stop = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_split, state='disabled')
        self.btn_split_stop.pack(side='left', padx=5)

        self.proc_split = None
        self.stop_split_requested = False

    def setup_combine_tab(self):
        # Input File 1
        frame_in1 = ttk.Frame(self.tab_combine)
        frame_in1.pack(fill='x', pady=5)
        ttk.Label(frame_in1, text=get_text('lbl_input_1')).pack(side='left')
        self.combine_in1_var = tk.StringVar()
        ttk.Entry(frame_in1, textvariable=self.combine_in1_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_in1, text=get_text('btn_browse'), command=lambda: self.browse_file(self.combine_in1_var)).pack(side='left')

        # Input File 2
        frame_in2 = ttk.Frame(self.tab_combine)
        frame_in2.pack(fill='x', pady=5)
        ttk.Label(frame_in2, text=get_text('lbl_input_2')).pack(side='left')
        self.combine_in2_var = tk.StringVar()
        ttk.Entry(frame_in2, textvariable=self.combine_in2_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_in2, text=get_text('btn_browse'), command=lambda: self.browse_file(self.combine_in2_var)).pack(side='left')

        # Optional original source used only as the output bitrate reference.
        frame_ref = ttk.Frame(self.tab_combine)
        frame_ref.pack(fill='x', pady=5)
        ttk.Label(frame_ref, text=get_text('lbl_bitrate_reference')).pack(side='left')
        self.combine_bitrate_ref_var = tk.StringVar()
        ttk.Entry(frame_ref, textvariable=self.combine_bitrate_ref_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_ref, text=get_text('btn_browse'), command=lambda: self.browse_file(self.combine_bitrate_ref_var)).pack(side='left')

        # Output File
        frame_out = ttk.Frame(self.tab_combine)
        frame_out.pack(fill='x', pady=5)
        ttk.Label(frame_out, text=get_text('lbl_output_file')).pack(side='left')
        self.combine_out_var = tk.StringVar()
        ttk.Entry(frame_out, textvariable=self.combine_out_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_out, text=get_text('btn_save'), command=lambda: self.save_file(self.combine_out_var)).pack(side='left')

        # Mode
        frame_mode = ttk.LabelFrame(self.tab_combine, text=get_text('grp_combine_mode'), padding=10)
        frame_mode.pack(fill='x', pady=10)
        self.combine_mode_var = tk.StringVar(value="left_right")
        ttk.Radiobutton(frame_mode, text=get_text('mode_sbs'), variable=self.combine_mode_var, value="left_right").pack(anchor='w')
        ttk.Radiobutton(frame_mode, text=get_text('mode_ou'), variable=self.combine_mode_var, value="top_bottom").pack(anchor='w')

        # Fisheye Option
        self.combine_fisheye_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tab_combine, text=get_text('chk_combine_fisheye'), variable=self.combine_fisheye_var).pack(anchor='w', pady=5)

        # Run Buttons
        btn_frame = ttk.Frame(self.tab_combine)
        btn_frame.pack(pady=10)
        
        self.btn_combine_start = ttk.Button(btn_frame, text=get_text('btn_start'), command=self.run_combine)
        self.btn_combine_start.pack(side='left', padx=5)
        
        self.btn_combine_stop = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_combine, state='disabled')
        self.btn_combine_stop.pack(side='left', padx=5)

        self.proc_combine = None
        self.stop_combine_requested = False

    # --- Helpers ---
    def browse_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*")])
        if f: var.set(f)

    def browse_dir(self, var):
        d = filedialog.askdirectory()
        if d: var.set(d)

    def save_file(self, var):
        f = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if f: var.set(f)

    # --- Actions ---
    def run_split(self):
        input_path = self.split_input_var.get()
        output_dir = self.split_output_var.get()
        mode = self.split_mode_var.get()

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_file'))
            return
        if not output_dir:
            output_dir = os.path.dirname(input_path) # Default to input dir

        self.log(get_text('msg_starting_split').format(mode))
        
        self.btn_split_start.config(state='disabled')
        self.btn_split_stop.config(state='normal')
        self.stop_split_requested = False

        def _on_proc(p):
            self.proc_split = p
            if self.stop_split_requested:
                try: p.kill()
                except Exception: pass

        def task():
            start_time = time.time()
            try:
                logic.split_video(
                    input_path,
                    mode,
                    output_dir,
                    to_fisheye=self.split_fisheye_var.get(),
                    log_callback=self.log,
                    process_callback=_on_proc
                )
                self.log(get_text('msg_task_complete'))
                if not self.stop_split_requested:
                    self.root.after(0, lambda: messagebox.showinfo(get_text('title_success'), get_text('msg_success_split')))
            except Exception as e:
                err = str(e)
                self.log(get_text('msg_error_occurred').format(err))
                if not self.stop_split_requested:
                    self.root.after(0, lambda msg=err: messagebox.showerror(get_text('title_error'), get_text('msg_error_occurred').format(msg)))
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_split_start.config(state='normal'))
                self.root.after(0, lambda: self.btn_split_stop.config(state='disabled'))
                self.proc_split = None

        threading.Thread(target=task, daemon=True).start()

    def stop_split(self):
        self.stop_split_requested = True
        proc = self.proc_split
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_split = None
            self.log(get_text('msg_stop'))

    def run_combine(self):
        in1 = self.combine_in1_var.get()
        in2 = self.combine_in2_var.get()
        out = self.combine_out_var.get()
        mode = self.combine_mode_var.get()
        bitrate_reference = self.combine_bitrate_ref_var.get().strip()

        if not in1 or not os.path.exists(in1):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_1'))
            return
        if not in2 or not os.path.exists(in2):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_2'))
            return
        if bitrate_reference and not os.path.exists(bitrate_reference):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_bitrate_reference'))
            return
        if not out:
             # Auto-generate output filenames
             dirname = os.path.dirname(in1)
             filename = os.path.splitext(os.path.basename(in1))[0]
             # Clean common suffixes to avoid recursiveness
             for s in ["_L", "_R", "_T", "_B", "_l", "_r", "_t", "_b"]:
                 if filename.endswith(s):
                     filename = filename[:-len(s)]
                     break
             
             suffix = "_sbs" if mode == "left_right" else "_ou"
             out = os.path.join(dirname, f"{filename}{suffix}.mp4")

        self.log(get_text('msg_starting_combine').format(mode))

        self.btn_combine_start.config(state='disabled')
        self.btn_combine_stop.config(state='normal')
        self.stop_combine_requested = False

        def _on_proc(p):
            self.proc_combine = p
            if self.stop_combine_requested:
                try: p.kill()
                except Exception: pass

        def task():
            start_time = time.time()
            try:
                logic.combine_video(
                    in1,
                    in2,
                    mode,
                    out,
                    from_fisheye=self.combine_fisheye_var.get(),
                    bitrate_reference_path=bitrate_reference or None,
                    log_callback=self.log,
                    process_callback=_on_proc
                )
                self.log(get_text('msg_task_complete'))
                if not self.stop_combine_requested:
                    self.root.after(0, lambda: messagebox.showinfo(get_text('title_success'), get_text('msg_success_combine')))
            except Exception as e:
                err = str(e)
                self.log(get_text('msg_error_occurred').format(err))
                if not self.stop_combine_requested:
                    self.root.after(0, lambda msg=err: messagebox.showerror(get_text('title_error'), get_text('msg_error_occurred').format(msg)))
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_combine_start.config(state='normal'))
                self.root.after(0, lambda: self.btn_combine_stop.config(state='disabled'))
                self.proc_combine = None

        threading.Thread(target=task, daemon=True).start()

    def stop_combine(self):
        self.stop_combine_requested = True
        proc = self.proc_combine
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_combine = None
            self.log(get_text('msg_stop'))

if __name__ == "__main__":
    root = tk.Tk()
    app = VRSplitCombineApp(root)
    root.mainloop()

