import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
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
    return i18n.translate('area_rect_crop', key)

class VRMosaicApp:
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

        
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 8], font=('Arial', 10, 'bold'))
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(expand=True, fill='both')
        
        self.create_extract_tab()
        self.create_locate_tab()
        self.create_process_tab()
        self.create_merge_tab()
        
        # Check dependencies
        missing = logic.check_dependencies()
        if missing:
            self.log_to_all(get_text('warn_dep').format(', '.join(missing)))
            self.log_to_all(get_text('warn_path'))

    def log_to_widget(self, widget, message):
        def _do():
            widget.insert(tk.END, message + "\n")
            widget.see(tk.END)
        self.root.after(0, _do)

    def log_to_all(self, message):
        for log_widget in [self.log_ext, self.log_loc, self.log_proc, self.log_merge]:
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
    def create_extract_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_extract'))
        
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
        ttk.Radiobutton(eye_frame, text=get_text('opt_both'), variable=self.ext_eye, value=3).pack(side='left', padx=5)
        
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
        def _on_proc(p):
            self.current_process = p
            if self.stop_requested:
                try: p.kill()
                except Exception: pass
        try:
            eye_mode = self.ext_eye.get()
            if eye_mode == 3: # Both - use dual output in one pass
                self.log_to_widget(self.log_ext, get_text('msg_processing_both'))
                logic.extract_clip_both(
                    self.ext_input.get(),
                    self.ext_start.get(),
                    self.ext_end.get(),
                    lambda msg: self.log_to_widget(self.log_ext, msg),
                    _on_proc
                )
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
            self.root.after(0, self._reset_extract_buttons)

    def _reset_extract_buttons(self):
        self.btn_extract.config(state='normal')
        self.btn_stop_ext.config(state='disabled')
        self.current_process = None

    # --- Tab 2: Locate ---
    def create_locate_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_locate'))
        
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
        param_frame1 = ttk.Frame(tab)
        param_frame1.pack(fill='x', padx=10, pady=5, anchor='w')
        
        self.loc_width = tk.IntVar(value=1920)
        self.loc_height = tk.IntVar(value=1080)
        self.loc_center_x = tk.IntVar(value=0)
        self.loc_center_y = tk.IntVar(value=0)
        
        ttk.Label(param_frame1, text=get_text('lbl_width')).pack(side='left')
        ttk.Entry(param_frame1, textvariable=self.loc_width, width=8).pack(side='left', padx=5)
        ttk.Label(param_frame1, text=get_text('lbl_height')).pack(side='left')
        ttk.Entry(param_frame1, textvariable=self.loc_height, width=8).pack(side='left', padx=5)

        ttk.Label(param_frame1, text=get_text('lbl_center_x')).pack(side='left')
        ttk.Entry(param_frame1, textvariable=self.loc_center_x, width=8).pack(side='left', padx=5)
        ttk.Label(param_frame1, text=get_text('lbl_center_y')).pack(side='left')
        ttk.Entry(param_frame1, textvariable=self.loc_center_y, width=8).pack(side='left', padx=5)

        # Time Params
        time_frame = ttk.Frame(tab)
        time_frame.pack(fill='x', padx=10, pady=5, anchor='w')
        
        self.loc_start = tk.StringVar()
        self.loc_end = tk.StringVar()
        vcmd = (self.root.register(self.validate_time_input), '%P')
        
        ttk.Label(time_frame, text=get_text('lbl_start')).pack(side='left')
        ttk.Entry(time_frame, textvariable=self.loc_start, width=10, validate='key', validatecommand=vcmd).pack(side='left', padx=5)
        ttk.Label(time_frame, text=get_text('lbl_end')).pack(side='left')
        ttk.Entry(time_frame, textvariable=self.loc_end, width=10, validate='key', validatecommand=vcmd).pack(side='left', padx=5)
        ttk.Label(time_frame, text=get_text('lbl_time_comment')).pack(side='left')
        
        # Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill='x', padx=10, pady=5, anchor='w')
        
        self.btn_load_preview = ttk.Button(btn_frame, text=get_text('btn_load_preview'), command=self.load_preview)
        self.btn_load_preview.pack(side='left', padx=5)
        ttk.Button(btn_frame, text=get_text('btn_send'), command=self.send_to_process).pack(side='left', padx=5)

        # Canvas Area
        canvas_frame = ttk.LabelFrame(tab, text=get_text('grp_preview'))
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black', width=800, height=450)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Log Area
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.pack(fill='x', padx=10, pady=5, side='bottom')
        self.log_loc = scrolledtext.ScrolledText(log_frame, height=4, state='normal')
        self.log_loc.pack(fill='both', expand=True)

        self.preview_img = None
        self.tk_img = None
        self.video_info = None

    def load_video_info(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.webm")])
        if path:
            self.loc_input.set(path)
            self.video_info = logic.get_video_info(path)
            if self.video_info:
                self.loc_slider.config(to=self.video_info['duration'])
                self.log_to_widget(self.log_loc, get_text('msg_loaded_video').format(self.video_info['width'], self.video_info['height'], self.video_info['duration']))
                # Set default center
                self.loc_center_x.set(self.video_info['width'] // 2)
                self.loc_center_y.set(self.video_info['height'] // 2)

    def format_time(self, seconds):
        seconds = int(float(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def on_slider_move(self, val):
        self.loc_time_lbl.config(text=f"{get_text('lbl_time')} {self.format_time(val)}")

    def load_preview(self):
        if not self.loc_input.get():
            return
        self.btn_load_preview.config(state='disabled')
        self.show_canvas_message(get_text('msg_loading_preview'))
        threading.Thread(target=self._preview_thread, daemon=True).start()

    def _preview_thread(self):
        path = self.loc_input.get()
        if not path: return
        try:
            img = logic.get_preview_frame_image(path, self.loc_time.get())
            self.root.after(0, lambda: self._display_image(img))
        except Exception as e:
            self.log_to_widget(self.log_loc, f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.btn_load_preview.config(state='normal'))

    def show_canvas_message(self, message):
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 800)
        ch = max(self.canvas.winfo_height(), 450)
        self.canvas.create_text(cw//2, ch//2, text=message, fill="#e6e6e6", font=("Arial", 14, "bold"), anchor='center')

    def _display_image(self, img):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return # Too small
        
        iw, ih = img.size
        ratio = min(cw/iw, ch/ih)
        nw, nh = int(iw * ratio), int(ih * ratio)
        
        img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=tk_img, anchor='center')
        
        # Keep reference
        self.preview_img = img
        self.tk_img = tk_img
        # Store geometry: (x_offset, y_offset, width, height, original_width, original_height)
        self.img_rect = ((cw - nw) // 2, (ch - nh) // 2, nw, nh, iw, ih)

    def on_canvas_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.canvas.find_withtag("selection"):
            self.canvas.delete("selection")
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2, tags="selection")

    def on_canvas_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_canvas_release(self, event):
        if not hasattr(self, 'img_rect') or not self.img_rect:
            return
        
        end_x, end_y = (event.x, event.y)
        
        # Canvas coordinates of selection
        sel_x1 = min(self.start_x, end_x)
        sel_y1 = min(self.start_y, end_y)
        sel_x2 = max(self.start_x, end_x)
        sel_y2 = max(self.start_y, end_y)
        
        # Image geometry
        off_x, off_y, w, h, orig_w, orig_h = self.img_rect
        
        # Map to image coordinates
        img_x1 = sel_x1 - off_x
        img_y1 = sel_y1 - off_y
        img_x2 = sel_x2 - off_x
        img_y2 = sel_y2 - off_y
        
        # Clamp
        img_x1 = max(0, min(img_x1, w))
        img_y1 = max(0, min(img_y1, h))
        img_x2 = max(0, min(img_x2, w))
        img_y2 = max(0, min(img_y2, h))
        
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            return # Invalid selection
            
        # Scale to original video resolution
        scale_x = orig_w / w
        scale_y = orig_h / h
        
        vid_x1 = int(img_x1 * scale_x)
        vid_y1 = int(img_y1 * scale_y)
        vid_x2 = int(img_x2 * scale_x)
        vid_y2 = int(img_y2 * scale_y)
        
        width = vid_x2 - vid_x1
        height = vid_y2 - vid_y1
        center_x = (vid_x1 + vid_x2) // 2
        center_y = (vid_y1 + vid_y2) // 2
        
        # Update inputs
        self.loc_width.set(width)
        self.loc_height.set(height)
        self.loc_center_x.set(center_x)
        self.loc_center_y.set(center_y)
        
        self.log_to_widget(self.log_loc, get_text('msg_selected_area').format(width, height, center_x, center_y))

    def send_to_process(self):
        self.proc_input.set(self.loc_input.get())
        self.proc_width.set(self.loc_width.get())
        self.proc_height.set(self.loc_height.get())
        self.proc_center_x.set(self.loc_center_x.get())
        self.proc_center_y.set(self.loc_center_y.get())
        self.proc_start.set(self.loc_start.get())
        self.proc_end.set(self.loc_end.get())
        self.notebook.select(2) # Switch to Process tab

    # --- Tab 3: Process ---
    def create_process_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_process'))
        
        ttk.Label(tab, text=get_text('lbl_input')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.proc_input = tk.StringVar()
        self.proc_input.trace("w", self.check_other_eye_file)
        
        # Input Frame
        input_frame = ttk.Frame(tab)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.proc_input, width=50).pack(side='left')
        ttk.Button(input_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.proc_input)).pack(side='left', padx=5)
        


        # Params Frame
        param_frame = ttk.Frame(tab)
        param_frame.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(param_frame, text=get_text('lbl_width')).grid(row=0, column=0, sticky='w')
        self.proc_width = tk.IntVar()
        ttk.Entry(param_frame, textvariable=self.proc_width, width=10).grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(param_frame, text=get_text('lbl_height')).grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.proc_height = tk.IntVar()
        ttk.Entry(param_frame, textvariable=self.proc_height, width=10).grid(row=0, column=3, sticky='w', padx=5)
        
        ttk.Label(param_frame, text=get_text('lbl_center_x')).grid(row=1, column=0, sticky='w', pady=5)
        self.proc_center_x = tk.IntVar()
        ttk.Entry(param_frame, textvariable=self.proc_center_x, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(param_frame, text=get_text('lbl_center_y')).grid(row=1, column=2, sticky='w', padx=(20, 0), pady=5)
        self.proc_center_y = tk.IntVar()
        ttk.Entry(param_frame, textvariable=self.proc_center_y, width=10).grid(row=1, column=3, sticky='w', padx=5, pady=5)
        
        # Time Params
        self.proc_start = tk.StringVar()
        self.proc_end = tk.StringVar()
        vcmd = (self.root.register(self.validate_time_input), '%P')

        ttk.Label(param_frame, text=get_text('lbl_start')).grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(param_frame, textvariable=self.proc_start, width=10, validate='key', validatecommand=vcmd).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(param_frame, text=get_text('lbl_end')).grid(row=2, column=2, sticky='w', padx=(20, 0), pady=5)
        ttk.Entry(param_frame, textvariable=self.proc_end, width=10, validate='key', validatecommand=vcmd).grid(row=2, column=3, sticky='w', padx=5, pady=5)

        ttk.Label(param_frame, text=get_text('lbl_time_comment')).grid(row=2, column=4, sticky='w', padx=(20, 0), pady=5)
        
        # Buttons Frame
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20, sticky='w', padx=5)
        
        # Checkbox Frame
        chk_frame = ttk.Frame(btn_frame)
        chk_frame.pack(side='top', anchor='w', pady=2)

        # Overwrite Input
        self.overwrite_input = tk.BooleanVar(value=False)
        ttk.Checkbutton(chk_frame, text=get_text('chk_overwrite'), variable=self.overwrite_input).pack(side='left', padx=(0, 10))

        # Keep Intermediate Files
        self.keep_intermediate = tk.BooleanVar(value=False)
        ttk.Checkbutton(chk_frame, text=get_text('chk_keep_inter'), variable=self.keep_intermediate).pack(side='left')

        # Checkbox for other eye (initially hidden)
        self.proc_other_eye = tk.BooleanVar(value=False)
        self.chk_other_eye = ttk.Checkbutton(chk_frame, text=get_text('chk_process_other'), variable=self.proc_other_eye)
        self.chk_other_eye.pack(side='left', padx=5)
        self.chk_other_eye.pack_forget()
        
        self.btn_process = ttk.Button(btn_frame, text=get_text('btn_process'), command=self.run_process)
        self.btn_process.pack(side='left', padx=5)
        
        self.btn_stop_proc = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_process, state='disabled')
        self.btn_stop_proc.pack(side='left', padx=5)

        # Log Area
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=4, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(4, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_proc = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_proc.pack(fill='both', expand=True)

    def check_other_eye_file(self, *args):
        path = self.proc_input.get()
        if not path or not os.path.exists(path):
            self.chk_other_eye.pack_forget()
            self.proc_other_eye.set(False)
            self.other_eye_file = None
            return

        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        
        other_name = None
        
        # Check patterns
        if "_L_" in name and "_R_" not in name:
             other_name = name.replace("_L_", "_R_")
        elif "_R_" in name and "_L_" not in name:
             other_name = name.replace("_R_", "_L_")
        elif "_L." in filename:
             other_name = filename.replace("_L.", "_R.")
             # Handle extension logic if needed, but simple replace works for full filename
             other_path = os.path.join(directory, other_name)
             if os.path.exists(other_path):
                 self.chk_other_eye.pack(side='left', padx=5)
                 self.proc_other_eye.set(True)
                 self.other_eye_file = other_path
                 return
        elif "_R." in filename:
             other_name = filename.replace("_R.", "_L.")
             other_path = os.path.join(directory, other_name)
             if os.path.exists(other_path):
                 self.chk_other_eye.pack(side='left', padx=5)
                 self.proc_other_eye.set(True)
                 self.other_eye_file = other_path
                 return

        if other_name:
            other_path = os.path.join(directory, other_name + ext)
            if os.path.exists(other_path):
                self.chk_other_eye.pack(side='left', padx=5)
                self.proc_other_eye.set(True)
                self.other_eye_file = other_path
            else:
                self.chk_other_eye.pack_forget()
                self.proc_other_eye.set(False)
                self.other_eye_file = None
        else:
            self.chk_other_eye.pack_forget()
            self.proc_other_eye.set(False)
            self.other_eye_file = None

    def run_process(self):
        valid, msg = self.validate_time_logic(self.proc_start.get(), self.proc_end.get())
        if not valid:
            messagebox.showerror("Error", msg)
            return

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
        def _on_proc(p):
            self.current_process = p
            if self.stop_requested:
                try: p.kill()
                except Exception: pass
        try:
            # Process current file
            self.log_to_widget(self.log_proc, get_text('msg_processing').format(self.proc_input.get()))
            logic.run_pipeline(
                self.proc_input.get(),
                self.proc_width.get(),
                self.proc_height.get(),
                self.proc_center_x.get(),
                self.proc_center_y.get(),
                self.proc_start.get(),
                self.proc_end.get(),
                lambda msg: self.log_to_widget(self.log_proc, msg),
                _on_proc,
                keep_intermediate=self.keep_intermediate.get(),
                overwrite_input=self.overwrite_input.get()
            )

            # Process other eye if checked
            if self.proc_other_eye.get() and self.other_eye_file and not self.stop_requested:
                self.log_to_widget(self.log_proc, get_text('msg_processing_other').format(self.other_eye_file))
                logic.run_pipeline(
                    self.other_eye_file,
                    self.proc_width.get(),
                    self.proc_height.get(),
                    self.proc_center_x.get(),
                    self.proc_center_y.get(),
                    self.proc_start.get(),
                    self.proc_end.get(),
                    lambda msg: self.log_to_widget(self.log_proc, msg),
                    _on_proc,
                    keep_intermediate=self.keep_intermediate.get(),
                    overwrite_input=self.overwrite_input.get()
                )
                
        except Exception as e:
            self.log_to_widget(self.log_proc, f"Error: {e}")
        finally:
            self.root.after(0, self._reset_process_buttons)

    def _reset_process_buttons(self):
        self.btn_process.config(state='normal')
        self.btn_stop_proc.config(state='disabled')
        self.current_process = None

    # --- Tab 4: Merge ---
    def create_merge_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text('tab_merge'))
        
        ttk.Label(tab, text=get_text('lbl_left')).grid(row=0, column=0, sticky='w')
        self.merge_l = tk.StringVar()
        
        # Left Input Frame
        l_frame = ttk.Frame(tab)
        l_frame.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Entry(l_frame, textvariable=self.merge_l, width=50).pack(side='left')
        ttk.Button(l_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.merge_l)).pack(side='left', padx=5)
        
        ttk.Label(tab, text=get_text('lbl_right')).grid(row=1, column=0, sticky='w')
        self.merge_r = tk.StringVar()
        
        # Right Input Frame
        r_frame = ttk.Frame(tab)
        r_frame.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Entry(r_frame, textvariable=self.merge_r, width=50).pack(side='left')
        ttk.Button(r_frame, text=get_text('btn_browse'), command=lambda: self.browse_file(self.merge_r)).pack(side='left', padx=5)
        
        # Buttons Frame
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=2, column=1, pady=20, sticky='w', padx=5)
        
        self.btn_merge = ttk.Button(btn_frame, text=get_text('btn_merge'), command=self.run_merge)
        self.btn_merge.pack(side='left', padx=5)
        
        self.btn_stop_merge = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_merge, state='disabled')
        self.btn_stop_merge.pack(side='left', padx=5)

        # Log Area
        log_frame = ttk.LabelFrame(tab, text=get_text('grp_log'))
        log_frame.grid(row=3, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        tab.grid_rowconfigure(3, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_merge = scrolledtext.ScrolledText(log_frame, height=8, state='normal')
        self.log_merge.pack(fill='both', expand=True)

    def run_merge(self):
        self.btn_merge.config(state='disabled')
        self.btn_stop_merge.config(state='normal')
        self.stop_requested = False
        threading.Thread(target=self._merge_thread, daemon=True).start()

    def stop_merge(self):
        self.stop_requested = True
        proc = self.current_process
        if proc:
            self.log_to_widget(self.log_merge, get_text('msg_stop'))
            try: proc.kill()
            except Exception: pass
            self.current_process = None

    def _merge_thread(self):
        def _on_proc(p):
            self.current_process = p
            if self.stop_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.merge_channels(
                self.merge_l.get(),
                self.merge_r.get(),
                lambda msg: self.log_to_widget(self.log_merge, msg),
                _on_proc
            )
        except Exception as e:
            self.log_to_widget(self.log_merge, f"Error: {e}")
        finally:
            self.root.after(0, self._reset_merge_buttons)

    def _reset_merge_buttons(self):
        self.btn_merge.config(state='normal')
        self.btn_stop_merge.config(state='disabled')
        self.current_process = None

    def browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.webm")])
        if path: var.set(path)

if __name__ == "__main__":
    root = tk.Tk()
    app = VRMosaicApp(root)
    root.mainloop()
