import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import os
import locale
import time
import sys
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
    return i18n.translate('vr2flat', key)


def _runtime_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _preview_debug_path(filename="preview_flat.jpg"):
    out_dir = os.path.join(_runtime_base_dir(), "debug_output", "tool_vr2flat")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


class VRMosaicApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text('title'))
        ui_theme.apply_theme(self.root)
        
        # Main Frame to hold everything
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True)

        if self.on_return:
            # Clear any existing menu
            empty_menu = tk.Menu(self.root)
            self.root.config(menu=empty_menu)

        # Full-height left rail: tool title on top, back-to-home pinned at the bottom
        self.notebook = ui_theme.ToolShell(
            main_frame,
            title=get_text('title'),
            back_text=get_text('btn_return'),
            on_back=self.on_return,
        )
        self.notebook.pack(expand=True, fill='both')
        
        self.create_extract_tab()
        self.create_locate_tab()
        self.create_process_tab()
        # self.create_merge_tab() # Removed
        
        # Check dependencies
        missing = logic.check_dependencies()
        if missing:
            self.log_to_all(get_text('warn_dep').format(', '.join(missing)))
            self.log_to_all(get_text('warn_path'))

    def log_to_widget(self, widget, message):
        def _do():
            widget.insert(tk.END, message + "\n")
            ui_theme.scroll_text_to_end(widget)
        self.root.after(0, _do)

    def log_to_all(self, message):
        for log_widget in [self.log_ext, self.log_loc, self.log_proc]:
            self.log_to_widget(log_widget, message)

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

    # --- Tab 1: Extract ---
    # --- Tab 1: Extract ---
    def create_extract_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_extract'), icon=ui_theme.TAB_ICONS['extract'])
        
        vcmd = (self.root.register(self.validate_time_input), '%P')
        
        ttk.Label(tab, text=get_text('lbl_input')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.ext_input = tk.StringVar()
        
        # Input Frame
        input_frame = ttk.Frame(tab)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.ext_input, width=50).pack(side='left')
        ttk.Button(input_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.ext_input)).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_eye')).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.ext_eye = tk.IntVar(value=1)
        eye_frame = ttk.Frame(tab)
        eye_frame.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Radiobutton(eye_frame, text=get_text('opt_left'), variable=self.ext_eye, value=1).pack(side='left', padx=5)
        ttk.Radiobutton(eye_frame, text=get_text('opt_right'), variable=self.ext_eye, value=2).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_start')).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.ext_start = tk.StringVar()
        ttk.Entry(tab, textvariable=self.ext_start, validate='key', validatecommand=vcmd).grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_end')).grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.ext_end = tk.StringVar()
        ttk.Entry(tab, textvariable=self.ext_end, validate='key', validatecommand=vcmd).grid(row=3, column=1, sticky='w', padx=5)
        
        # Buttons Frame
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=4, column=1, pady=20, sticky='w', padx=5)
        
        self.btn_extract = ttk.Button(btn_frame, text=get_text('btn_extract'), command=self.run_extract)
        self.btn_extract.pack(side='left', padx=5)
        
        self.btn_stop_ext = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_extract, state='disabled')
        self.btn_stop_ext.pack(side='left', padx=5)

        # Log Area
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=5, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(5, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_ext = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_ext.pack(fill='both', expand=True)

        self.current_process = None
        self.stop_requested = False

    def run_extract(self):
        valid, msg = self.validate_time_logic(self.ext_start.get(), self.ext_end.get())
        if not valid:
            messagebox.showerror("Error", msg)
            return

        self.btn_extract.config(state='disabled')
        self.btn_stop_ext.config(state='normal')
        self.stop_requested = False
        threading.Thread(target=self._extract_thread, daemon=True).start()

    def stop_extract(self):
        self.stop_requested = True
        proc = self.current_process
        if proc:
            self.log_to_widget(self.log_ext, get_text('msg_stop'))
            try: proc.kill()
            except Exception: pass
            self.current_process = None

    def _extract_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.current_process = p
            if self.stop_requested:
                try: p.kill()
                except Exception: pass
        try:
            eye_mode = self.ext_eye.get()
            if eye_mode == 3: # Should not happen
                pass
            else:
                logic.extract_clip(
                    self.ext_input.get(),
                    eye_mode == 1,
                    self.ext_start.get(),
                    self.ext_end.get(),
                    lambda msg: self.log_to_widget(self.log_ext, msg),
                    _on_proc
                )
        except Exception as e:
            self.log_to_widget(self.log_ext, f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_to_widget(self.log_ext, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, self._reset_extract_buttons)

    def _reset_extract_buttons(self):
        self.btn_extract.config(state='normal')
        self.btn_stop_ext.config(state='disabled')
        self.current_process = None

    # --- Tab 2: Locate ---
    def create_locate_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_locate'), icon=ui_theme.TAB_ICONS['locate'])
        
        # Controls
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.pack(fill='x', padx=5, pady=5, anchor='w')
        
        ttk.Label(ctrl_frame, text=get_text('lbl_input')).pack(side='left')
        self.loc_input = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.loc_input, width=40).pack(side='left', padx=5)
        ttk.Button(ctrl_frame, text=get_text('btn_browse'), command=self.load_video_info).pack(side='left')
        
        # Timeline
        self.loc_time = tk.DoubleVar()
        self.loc_slider = ttk.Scale(tab, from_=0, to=100, variable=self.loc_time, orient='horizontal', command=self.on_slider_move)
        self.loc_slider.pack(fill='x', padx=10, pady=5)
        self.loc_time_lbl = ttk.Label(tab, text=get_text('lbl_time') + " 00:00:00")
        self.loc_time_lbl.pack(anchor='w', padx=10)
        
        # Params
        param_frame = ttk.Frame(tab)
        param_frame.pack(fill='x', padx=10, pady=5, anchor='w')
        
        self.loc_yaw = tk.DoubleVar()
        self.loc_pitch = tk.DoubleVar()
        self.loc_fov = tk.DoubleVar(value=100)
        self.loc_w = tk.IntVar()
        self.loc_h = tk.IntVar()
        
        ttk.Label(param_frame, text=get_text('lbl_yaw')).pack(side='left')
        ttk.Entry(param_frame, textvariable=self.loc_yaw, width=8).pack(side='left', padx=5)
        ttk.Label(param_frame, text=get_text('lbl_pitch')).pack(side='left')
        ttk.Entry(param_frame, textvariable=self.loc_pitch, width=8).pack(side='left', padx=5)
        ttk.Label(param_frame, text=get_text('lbl_fov')).pack(side='left')
        ttk.Entry(param_frame, textvariable=self.loc_fov, width=8).pack(side='left', padx=5)
        ttk.Label(param_frame, text=get_text('lbl_width')).pack(side='left')
        ttk.Entry(param_frame, textvariable=self.loc_w, width=6).pack(side='left', padx=5)
        ttk.Label(param_frame, text=get_text('lbl_height')).pack(side='left')
        ttk.Entry(param_frame, textvariable=self.loc_h, width=6).pack(side='left', padx=5)

        # Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill='x', padx=10, pady=5, anchor='w')
        
        self.btn_load_vr = ttk.Button(btn_frame, text=get_text('btn_load_vr'), command=self.load_vr_image)
        self.btn_load_vr.pack(side='left', padx=5)
        self.btn_load_flat = ttk.Button(btn_frame, text=get_text('btn_load_flat'), command=self.load_flat_preview)
        self.btn_load_flat.pack(side='left', padx=5)
        ttk.Button(btn_frame, text=get_text('btn_send'), command=self.send_to_process).pack(side='left', padx=5)
        # ttk.Label(btn_frame, text=get_text('lbl_lada_warn'), foreground='red').pack(side='left', padx=5) # Removed

        # Canvas Area (Split)
        split_frame = ttk.Frame(tab)
        split_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left Canvas (VR)
        self.left_frame = ttk.LabelFrame(split_frame, text=get_text('grp_vr_view'))
        self.left_frame.pack(side='left', fill='both', expand=True, padx=5)
        self.canvas_vr = tk.Canvas(self.left_frame, bg='black', width=400, height=400)
        self.canvas_vr.pack(fill='both', expand=True)
        self.canvas_vr.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas_vr.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas_vr.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Right Canvas (Flat)
        self.right_frame = ttk.LabelFrame(split_frame, text=get_text('grp_flat_view'))
        self.right_frame.pack(side='right', fill='both', expand=True, padx=5)
        self.canvas_flat = tk.Canvas(self.right_frame, bg='black', width=400, height=400)
        self.canvas_flat.pack(fill='both', expand=True)

        # Log Area
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.pack(fill='x', padx=10, pady=5, side='bottom')
        self.log_loc = scrolledtext.ScrolledText(log_frame, height=4, state='normal')
        self.log_loc.pack(fill='both', expand=True)

        self.preview_vr_img = None
        self.tk_vr_img = None
        self.preview_flat_img = None
        self.tk_flat_img = None

    def load_video_info(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.webm")])
        if path:
            self.loc_input.set(path)
            info = logic.get_video_info(path)
            if info:
                self.video_info = info
                self.loc_slider.config(to=info['duration'])
                self.log_to_widget(self.log_loc, get_text('msg_loaded_video').format(info['width'], info['height'], info['duration']))

    def format_time(self, seconds):
        seconds = int(float(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def on_slider_move(self, val):
        self.loc_time_lbl.config(text=f"{get_text('lbl_time')} {self.format_time(val)}")

    def load_vr_image(self):
        if not self.loc_input.get():
            return
        self.btn_load_vr.config(state='disabled')
        self.show_canvas_message(self.canvas_vr, get_text('msg_loading_vr'))
        threading.Thread(target=self._vr_thread, daemon=True).start()

    def _vr_thread(self):
        path = self.loc_input.get()
        if not path: return
        try:
            img = logic.get_vr_frame_image(path, self.loc_time.get())
            self.root.after(0, lambda: self._display_image(img, self.canvas_vr, is_vr=True))
        except Exception as e:
            self.log_to_widget(self.log_loc, f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.btn_load_vr.config(state='normal'))

    def load_flat_preview(self):
        if not self.loc_input.get():
            return
        self.btn_load_flat.config(state='disabled')
        self.show_canvas_message(self.canvas_flat, get_text('msg_loading_flat'))
        threading.Thread(target=self._flat_thread, daemon=True).start()

    def _flat_thread(self):
        path = self.loc_input.get()
        if not path: return
        try:
            tmp = _preview_debug_path()
            flat_w = 0
            flat_h = 0
            if hasattr(self, 'video_info') and self.video_info and hasattr(self, 'sel_ratio_w'):
                flat_w = int(self.video_info['width'] * self.sel_ratio_w)
                flat_h = int(self.video_info['height'] * self.sel_ratio_h)
            elif self.tk_vr_img is not None:
                flat_w = self.tk_vr_img.width() * self.loc_fov.get() 
                flat_h = self.tk_vr_img.height() * self.loc_fov.get()

            success = logic.get_flat_frame(path, self.loc_time.get(), self.loc_yaw.get(), self.loc_pitch.get(), self.loc_fov.get(), flat_w, flat_h, tmp)
            
            if success and os.path.exists(tmp):
                img = Image.open(tmp).copy()
                self.root.after(0, lambda: self._display_image(img, self.canvas_flat, is_vr=False))
        except Exception as e:
            self.log_to_widget(self.log_loc, f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.btn_load_flat.config(state='normal'))

    def show_canvas_message(self, canvas, message):
        canvas.delete("all")
        cw = max(canvas.winfo_width(), 400)
        ch = max(canvas.winfo_height(), 400)
        canvas.create_text(cw//2, ch//2, text=message, fill="#e6e6e6", font=("Arial", 14, "bold"), anchor='center')

    def _display_image(self, img, canvas, is_vr):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10: return # Too small
        
        iw, ih = img.size
        ratio = min(cw/iw, ch/ih)
        nw, nh = int(iw * ratio), int(ih * ratio)
        
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        
        canvas.delete("all")
        canvas.create_image(cw//2, ch//2, image=tk_img, anchor='center')
        
        # Keep reference to avoid GC
        if is_vr:
            self.preview_vr_img = img
            self.tk_vr_img = tk_img
            # Store geometry for click calculation: (x_offset, y_offset, width, height)
            self.vr_img_rect = ((cw - nw) // 2, (ch - nh) // 2, nw, nh)
        else:
            self.preview_flat_img = img
            self.tk_flat_img = tk_img

    def on_canvas_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.canvas_vr.find_withtag("selection"):
            self.canvas_vr.delete("selection")
        self.rect = self.canvas_vr.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2, tags="selection")

    def on_canvas_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas_vr.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_canvas_release(self, event):
        if not hasattr(self, 'vr_img_rect') or not self.vr_img_rect:
            return

        end_x, end_y = (event.x, event.y)
        
        # Calculate center of selection
        center_x = (self.start_x + end_x) / 2
        center_y = (self.start_y + end_y) / 2
        
        # Image geometry
        off_x, off_y, w, h = self.vr_img_rect
        
        # Map to image coordinates
        img_x = center_x - off_x
        img_y = center_y - off_y
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, w))
        img_y = max(0, min(img_y, h))
        
        # Calculate Yaw (-90 to 90)
        # 0 -> -90, w -> 90
        yaw = (img_x / w) * 180 - 90
        
        # Calculate Pitch (-90 to 90)
        # 0 -> 90, h -> -90 (Standard Equirectangular: Top is North Pole +90)
        pitch = 90 - (img_y / h) * 180

        # Calculate D_FOV (10 to 100)
        selection_width = abs(end_x - self.start_x)
        selection_height = abs(end_y - self.start_y)
        
        self.sel_ratio_w = selection_width / w
        self.sel_ratio_h = selection_height / h
        
        d_fov = (selection_width / w) * 180
        if d_fov > 130:
            d_fov = 130
        if d_fov < 10:
            d_fov = 10
        self.loc_fov.set(f"{d_fov:.2f}")
        self.loc_yaw.set(f"{yaw:.2f}")
        self.loc_pitch.set(f"{pitch:.2f}")
        
        # Calculate W/H
        if hasattr(self, 'video_info') and self.video_info:
            flat_w = int(self.video_info['width'] * self.sel_ratio_w)
            flat_h = int(self.video_info['height'] * self.sel_ratio_h)
            
            # Force multiple of 10
            flat_w = round(flat_w / 10) * 10
            flat_h = round(flat_h / 10) * 10
            
            self.loc_w.set(flat_w)
            self.loc_h.set(flat_h)
        
        self.log_to_widget(self.log_loc, get_text('msg_selected_center').format(f'{yaw:.2f}', f'{pitch:.2f}'))

    def send_to_process(self):
        self.proc_input.set(self.loc_input.get())
        self.proc_yaw.set(self.loc_yaw.get())
        self.proc_pitch.set(self.loc_pitch.get())
        self.proc_fov.set(self.loc_fov.get())
        self.proc_w.set(self.loc_w.get())
        self.proc_h.set(self.loc_h.get())
        self.proc_w.set(self.loc_w.get())
        self.proc_h.set(self.loc_h.get())
        self.notebook.select(2) # Switch to Process tab

    # --- Tab 3: Process ---
    def create_process_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_process'), icon=ui_theme.TAB_ICONS['process'])
        
        ttk.Label(tab, text=get_text('lbl_input')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.proc_input = tk.StringVar()
        self.proc_input.trace("w", self.check_other_eye_file)
        
        # Input Frame
        input_frame = ttk.Frame(tab)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.proc_input, width=50).pack(side='left')
        ttk.Button(input_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.proc_input)).pack(side='left', padx=5)
        

        
        # Row 1: Yaw, Pitch
        row1_frame = ttk.Frame(tab)
        row1_frame.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(row1_frame, text=get_text('lbl_yaw')).pack(side='left')
        self.proc_yaw = tk.DoubleVar()
        ttk.Entry(row1_frame, textvariable=self.proc_yaw, width=8).pack(side='left', padx=5)
        
        ttk.Label(row1_frame, text=get_text('lbl_pitch')).pack(side='left')
        self.proc_pitch = tk.DoubleVar()
        ttk.Entry(row1_frame, textvariable=self.proc_pitch, width=8).pack(side='left', padx=5)
        
        # Row 2: FOV, Width, Height
        row2_frame = ttk.Frame(tab)
        row2_frame.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(row2_frame, text=get_text('lbl_fov')).pack(side='left')
        self.proc_fov = tk.DoubleVar()
        ttk.Entry(row2_frame, textvariable=self.proc_fov, width=8).pack(side='left', padx=5)

        self.proc_w = tk.IntVar()
        self.proc_h = tk.IntVar()
        
        ttk.Label(row2_frame, text=get_text('lbl_width')).pack(side='left')
        ttk.Entry(row2_frame, textvariable=self.proc_w, width=8).pack(side='left', padx=5)
        ttk.Label(row2_frame, text=get_text('lbl_height')).pack(side='left')
        ttk.Entry(row2_frame, textvariable=self.proc_h, width=8).pack(side='left', padx=5)
        
        # Buttons Frame
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=20, sticky='w', padx=5)
        
        self.btn_process = ttk.Button(btn_frame, text=get_text('btn_process'), command=self.run_process)
        self.btn_process.pack(side='left', padx=5)
        
        self.btn_stop_proc = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_process, state='disabled')
        self.btn_stop_proc.pack(side='left', padx=5)
        # ttk.Label(btn_frame, text=get_text('lbl_lada_warn'), foreground='red').pack(side='left', padx=5) # Removed

        # Log Area
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=6, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(6, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_proc = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_proc.pack(fill='both', expand=True)

    def check_other_eye_file(self, *args):
        pass # Removed logic

    def run_process(self):
        self.btn_process.config(state='disabled')
        self.btn_stop_proc.config(state='normal')
        self.stop_requested = False
        threading.Thread(target=self._process_thread, daemon=True).start()

    def stop_process(self):
        self.stop_requested = True
        proc = self.current_process
        if proc:
            self.log_to_widget(self.log_proc, get_text('msg_stop'))
            try: proc.kill()
            except Exception: pass
            self.current_process = None

    def _process_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.current_process = p
            if self.stop_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_pipeline(
                self.proc_input.get(),
                self.proc_yaw.get(),
                self.proc_pitch.get(),
                self.proc_fov.get(),
                self.proc_w.get(),
                self.proc_h.get(),
                lambda msg: self.log_to_widget(self.log_proc, msg),
                _on_proc
            )
            
        except Exception as e:
            self.log_to_widget(self.log_proc, f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_to_widget(self.log_proc, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, self._reset_process_buttons)

    def _reset_process_buttons(self):
        self.btn_process.config(state='normal')
        self.btn_stop_proc.config(state='disabled')
        self.current_process = None

    # --- Tab 4: Merge --- REMOVED
    # def create_merge_tab(self):
    #     pass

    def browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.webm")])
        if path: var.set(path)

if __name__ == "__main__":
    root = tk.Tk()
    app = VRMosaicApp(root)
    root.mainloop()
