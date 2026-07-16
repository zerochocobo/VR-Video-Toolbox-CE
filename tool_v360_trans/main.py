import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import sys
import locale
import time
from utils import app_config, i18n, ui_theme

# Import logic module - use try/except to handle both direct run and import from main
try:
    from . import logic
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import logic

# --- i18n Setup ---


def get_text(key):
    return i18n.translate('v360_trans', key)

class VRTransApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text('title'))
        ui_theme.apply_theme(self.root)
        
        # Configure layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill='both', expand=True)

        # Full-height left rail: tool title on top, back-to-home pinned at the bottom
        self.notebook = ui_theme.ToolShell(
            self.main_frame,
            title=get_text('title'),
            back_text=get_text('btn_back'),
            on_back=self.go_back if self.on_return else None,
        )
        self.notebook.pack(fill='both', expand=True)

        # Tab 1: Hequirect -> Fisheye
        self.tab_h2f = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_h2f, text=get_text('tab_hequirect2fisheye'), icon=ui_theme.TAB_ICONS['globe'])
        self.setup_conversion_tab(self.tab_h2f, "hequirect2fisheye")

        # Tab 2: Fisheye -> Hequirect
        self.tab_f2h = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_f2h, text=get_text('tab_fisheye2hequirect'), icon=ui_theme.TAB_ICONS['globe_wire'])
        self.setup_conversion_tab(self.tab_f2h, "fisheye2hequirect")

        # Log Area (shared across tabs, in the shell footer)
        log_frame = ttk.LabelFrame(self.notebook.footer(expand=True), text=get_text('log_title'), padding=5)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(4, 10))
        self.log_text = tk.Text(log_frame, height=10, state='disabled')
        self.log_text.pack(fill='both', expand=True, side='left')
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        self.proc = None
        self.stop_requested = False

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
            ui_theme.scroll_text_to_end(self.log_text)
            self.log_text.config(state='disabled')
        self.root.after(0, _do)

    def setup_conversion_tab(self, parent, mode):
        # We need independent vars for each tab to avoid conflict
        # Using a container class or dictionary to store widget refs if needed.
        # But for simplicity, let's attach vars to the parent frame or a dict.
        
        vars_dict = {}
        parent.vars = vars_dict # Attach to frame instance
        
        # Input File
        frame_input = ttk.Frame(parent)
        frame_input.pack(fill='x', pady=5)
        ttk.Label(frame_input, text=get_text('lbl_input_video')).pack(side='left')
        vars_dict['input'] = tk.StringVar()
        ttk.Entry(frame_input, textvariable=vars_dict['input']).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_input, text=get_text('btn_browse'), command=lambda: self.browse_file(vars_dict['input'])).pack(side='left')

        # Output Dir
        frame_output = ttk.Frame(parent)
        frame_output.pack(fill='x', pady=5)
        ttk.Label(frame_output, text=get_text('lbl_output_dir')).pack(side='left')
        vars_dict['output'] = tk.StringVar()
        ttk.Entry(frame_output, textvariable=vars_dict['output']).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_output, text=get_text('btn_browse'), command=lambda: self.browse_dir(vars_dict['output'])).pack(side='left')

        # Dual Screen Option (placed before buttons)
        vars_dict['dual_screen'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text=get_text('chk_dual_screen'), variable=vars_dict['dual_screen']).pack(anchor='w', pady=5)
        
        # Keep Bitrate Option
        vars_dict['keep_bitrate'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text=get_text('chk_keep_bitrate'), variable=vars_dict['keep_bitrate']).pack(anchor='w', pady=5)

        # Run Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=20)
        
        btn_start = ttk.Button(btn_frame, text=get_text('btn_start'), command=lambda: self.run_conversion(mode, vars_dict, btn_start, btn_stop))
        btn_start.pack(side='left', padx=5)
        
        btn_stop = ttk.Button(btn_frame, text=get_text('btn_stop'), command=lambda: self.stop_process(), state='disabled')
        btn_stop.pack(side='left', padx=5)
        
    def browse_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*")])
        if f: var.set(f)

    def browse_dir(self, var):
        d = filedialog.askdirectory()
        if d: var.set(d)

    def run_conversion(self, mode, vars_dict, btn_start, btn_stop):
        input_path = vars_dict['input'].get()
        output_dir = vars_dict['output'].get()

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_file'))
            return
        if not output_dir:
            output_dir = os.path.dirname(input_path) # Default to input dir

        self.log(get_text('msg_starting').format(mode))

        btn_start.config(state='disabled')
        btn_stop.config(state='normal')
        self.stop_requested = False

        def _on_proc(p):
            # If user already pressed stop, kill immediately to win the race
            self.proc = p
            if self.stop_requested:
                try:
                    p.kill()
                except Exception:
                    pass

        def task():
            start_time = time.time()
            try:
                logic.convert_projection(
                    input_path,
                    output_dir,
                    mode,
                    dual_screen=vars_dict['dual_screen'].get(),
                    keep_original_bitrate=vars_dict['keep_bitrate'].get(),
                    log_callback=self.log,
                    process_callback=_on_proc
                )
                self.log(get_text('msg_task_complete'))
                if not self.stop_requested:
                    self.root.after(0, lambda: messagebox.showinfo(get_text('title_success'), get_text('msg_success')))
            except Exception as e:
                err = str(e)
                self.log(get_text('msg_error_occurred').format(err))
                if not self.stop_requested:
                    self.root.after(0, lambda msg=err: messagebox.showerror(get_text('title_error'), get_text('msg_error_occurred').format(msg)))
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: btn_start.config(state='normal'))
                self.root.after(0, lambda: btn_stop.config(state='disabled'))
                self.proc = None

        threading.Thread(target=task, daemon=True).start()

    def stop_process(self):
        self.stop_requested = True
        proc = self.proc
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
            self.proc = None
            self.log(get_text('msg_stop'))

if __name__ == "__main__":
    root = tk.Tk()
    app = VRTransApp(root)
    root.mainloop()
