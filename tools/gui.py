import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
import time
from utils import app_config, i18n

# Import logic module - use try/except to handle both direct run and import from main
try:
    from . import logic
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import logic

import locale

# --- i18n Setup ---


def get_text(key):
    return i18n.translate('tools', key)

class VRVideoToolsApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text('title'))
        # self.root.geometry("800x600") # Let it resize naturally or inherit

        self.setup_ui()

        # Check dependencies
        import shutil
        missing = []
        for tool in ["ffmpeg", "ffprobe"]:
            if not shutil.which(tool):
                missing.append(tool)
        if missing:
             self.log_to_all(get_text('warn_dep').format(', '.join(missing)))
             self.log_to_all(get_text('warn_path'))

    def log_to_all(self, message):
        for log_widget in [self.ss_log, self.patch_log, self.zoom_log, self.merge_log, self.cut_log, self.kf_log]:
            self.log(log_widget, message)

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

        # Tab 1: Screenshot
        self.tab_screenshot = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_screenshot, text=get_text('tab_screenshot'))
        self.setup_screenshot_tab()

        # Tab 1.5: Keyframe Extraction
        self.tab_keyframe = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_keyframe, text=get_text('tab_keyframe'))
        self.setup_keyframe_tab()

        # Tab 2: Patcher
        self.tab_patcher = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_patcher, text=get_text('tab_patcher'))
        self.setup_patcher_tab()

        # Tab 3: Zoom View
        self.tab_zoom = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_zoom, text=get_text('tab_zoom'))
        self.setup_zoom_tab()

        # Tab 4: Merge Files
        self.tab_merge = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_merge, text=get_text('tab_merge'))
        self.setup_merge_tab()

        # Tab 5: Quick Safe Cut
        self.tab_cut = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_cut, text=get_text('tab_cut'))
        self.setup_cut_tab()

        # Attach hover tooltips to all notebook tabs
        self._setup_tab_tooltips()

    def _setup_tab_tooltips(self):
        """Show a tooltip with the full tab name when hovering over a truncated notebook tab."""
        tip_win = [None]  # mutable container so inner functions can reassign

        def show_tip(event):
            # Destroy any existing tooltip
            if tip_win[0]:
                tip_win[0].destroy()
                tip_win[0] = None
            # Identify which tab the cursor is over
            try:
                tab_idx = self.notebook.index(f"@{event.x},{event.y}")
            except tk.TclError:
                return
            full_text = self.notebook.tab(tab_idx, 'text')
            # Position just below the cursor
            x = event.x_root + 12
            y = event.y_root + 20
            win = tk.Toplevel(self.notebook)
            win.wm_overrideredirect(True)  # no window decorations
            win.wm_geometry(f"+{x}+{y}")
            tk.Label(
                win, text=full_text,
                background="#ffffc0", foreground="#000000",
                relief='solid', borderwidth=1,
                font=('Arial', 9), padx=4, pady=2
            ).pack()
            tip_win[0] = win

        def hide_tip(event):
            if tip_win[0]:
                tip_win[0].destroy()
                tip_win[0] = None

        self.notebook.bind("<Motion>", show_tip)
        self.notebook.bind("<Leave>",  hide_tip)
        self.notebook.bind("<Button-1>", hide_tip)

    def log(self, text_widget, message):
        def _do():
            text_widget.config(state='normal')
            text_widget.insert('end', message + "\n")
            text_widget.see('end')
            text_widget.config(state='disabled')
        self.root.after(0, _do)

    # --- Screenshot Tab ---
    def setup_screenshot_tab(self):
        frame = self.tab_screenshot
        
        # File Selection
        file_frame = ttk.LabelFrame(frame, text=get_text('lbl_input_video'), padding=10)
        file_frame.pack(fill='x', pady=5)
        
        self.ss_video_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.ss_video_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(file_frame, text=get_text('btn_browse'), command=self.browse_ss_video).pack(side='right')

        # Timestamp
        time_frame = ttk.LabelFrame(frame, text=get_text('lbl_timestamp'), padding=10)
        time_frame.pack(fill='x', pady=5)
        
        self.ss_timestamp = tk.StringVar(value="00:00:05")
        ttk.Entry(time_frame, textvariable=self.ss_timestamp).pack(fill='x')

        # Action
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        self.btn_ss_run = ttk.Button(btn_frame, text=get_text('btn_extract'), command=self.run_screenshot)
        self.btn_ss_run.pack(fill='x')

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.ss_log = tk.Text(log_frame, height=10, state='disabled')
        self.ss_log.pack(fill='both', expand=True)

    def browse_ss_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if path:
            self.ss_video_path.set(path)

    def run_screenshot(self):
        video_path = self.ss_video_path.get()
        timestamp = self.ss_timestamp.get()
        
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", get_text('err_video'))
            return

        # Validation
        valid, msg = self.validate_time_logic(timestamp)
        if not valid:
            messagebox.showerror("Error", msg)
            return

        # Construct output path
        dirname = os.path.dirname(video_path)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        safe_time = timestamp.replace(':', '.')
        output_path = os.path.join(dirname, f"{basename}_T[{safe_time}].jpg")

        def task():
            self.log(self.ss_log, get_text('msg_start_extract'))
            logic.extract_screenshot(video_path, timestamp, output_path, lambda msg: self.log(self.ss_log, msg))
            self.log(self.ss_log, get_text('msg_done'))
            self.root.after(0, lambda: self.btn_ss_run.config(state='normal'))

        self.btn_ss_run.config(state='disabled')
        threading.Thread(target=task, daemon=True).start()

    # --- Patcher Tab ---
    def setup_patcher_tab(self):
        frame = self.tab_patcher
        
        # Main Video
        main_frame = ttk.LabelFrame(frame, text=get_text('lbl_main_video'), padding=10)
        main_frame.pack(fill='x', pady=5)
        
        self.patch_main_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.patch_main_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(main_frame, text=get_text('btn_browse'), command=self.browse_patch_main).pack(side='right')

        # Patch Clip
        clip_frame = ttk.LabelFrame(frame, text=get_text('lbl_patch_clip'), padding=10)
        clip_frame.pack(fill='x', pady=5)
        
        self.patch_clip_path = tk.StringVar()
        ttk.Entry(clip_frame, textvariable=self.patch_clip_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(clip_frame, text=get_text('btn_browse'), command=self.browse_patch_clip).pack(side='right')

        # Start Time
        time_frame = ttk.LabelFrame(frame, text=get_text('lbl_start_time'), padding=10)
        time_frame.pack(fill='x', pady=5)
        
        self.patch_start_time = tk.StringVar()
        ttk.Entry(time_frame, textvariable=self.patch_start_time).pack(fill='x')

        # Keep Bitrate Option
        self.patch_keep_bitrate = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text=get_text('chk_keep_bitrate'), variable=self.patch_keep_bitrate).pack(anchor='w', pady=5)

        # Action
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        self.btn_patch = ttk.Button(btn_frame, text=get_text('btn_patch'), command=self.run_patcher)
        self.btn_patch.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_patch = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_patcher, state='disabled')
        self.btn_stop_patch.pack(side='left', fill='x', expand=True, padx=(5, 0))

        self.proc_patch = None
        self.stop_patch_requested = False

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.patch_log = tk.Text(log_frame, height=10, state='disabled')
        self.patch_log.pack(fill='both', expand=True)

    def browse_patch_main(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if path:
            self.patch_main_path.set(path)

    def browse_patch_clip(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if path:
            self.patch_clip_path.set(path)

    def run_patcher(self):
        main_video = self.patch_main_path.get()
        patch_video = self.patch_clip_path.get()
        start_time = self.patch_start_time.get()
        
        if not main_video or not os.path.exists(main_video):
            messagebox.showerror("Error", get_text('err_main'))
            return
        if not patch_video or not os.path.exists(patch_video):
            messagebox.showerror("Error", get_text('err_patch'))
            return
            
        # Validation
        valid, msg = self.validate_time_logic(start_time)
        if not valid:
            messagebox.showerror("Error", msg)
            return

        # Check resolution match
        main_res = logic.get_video_resolution(main_video)
        patch_res = logic.get_video_resolution(patch_video)

        if main_res[0] and patch_res[0]:
            if main_res != patch_res:
                messagebox.showerror("Error", f"{get_text('err_resolution_mismatch')}\nMain: {main_res}\nPatch: {patch_res}")
                return
        else:
             self.log(self.patch_log, get_text('msg_warn_resolution'))

        # Construct output path
        dirname = os.path.dirname(main_video)
        basename = os.path.splitext(os.path.basename(main_video))[0]
        output_path = os.path.join(dirname, f"{basename}_patched_auto.mp4")

        def _on_proc(p):
            self.proc_patch = p
            if self.stop_patch_requested:
                try: p.kill()
                except Exception: pass

        def task():
            start_time = time.time()
            self.log(self.patch_log, get_text('msg_start_patch'))
            try:
                logic.patch_video(
                    main_video,
                    patch_video,
                    start_time,
                    output_path,
                    self.patch_keep_bitrate.get(),
                    lambda msg: self.log(self.patch_log, msg),
                    _on_proc
                )
            except Exception as e:
                self.log(self.patch_log, f"Error: {e}")
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(self.patch_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_patch.config(state='normal'))
                self.root.after(0, lambda: self.btn_stop_patch.config(state='disabled'))
                self.proc_patch = None

        self.btn_patch.config(state='disabled')
        self.btn_stop_patch.config(state='normal')
        self.stop_patch_requested = False
        threading.Thread(target=task, daemon=True).start()

    def stop_patcher(self):
        self.stop_patch_requested = True
        proc = self.proc_patch
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_patch = None
            self.log(self.patch_log, get_text('msg_stop'))

    # --- Zoom View Tab ---
    def setup_zoom_tab(self):
        frame = self.tab_zoom
        
        # File Selection
        file_frame = ttk.LabelFrame(frame, text=get_text('lbl_input_video'), padding=10)
        file_frame.pack(fill='x', pady=5)
        
        self.zoom_video_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.zoom_video_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(file_frame, text=get_text('btn_browse'), command=self.browse_zoom_video).pack(side='right')

        # Timestamp / Slider
        time_frame = ttk.LabelFrame(frame, text=get_text('lbl_drag_time'), padding=10)
        time_frame.pack(fill='x', pady=5)
        
        self.zoom_time_var = tk.DoubleVar(value=5.0)
        self.zoom_slider = ttk.Scale(time_frame, from_=0, to=100, variable=self.zoom_time_var, orient='horizontal', command=self.on_zoom_slider_move)
        self.zoom_slider.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.zoom_time_lbl = ttk.Label(time_frame, text="00:00:05")
        self.zoom_time_lbl.pack(side='left', padx=(0, 5))
        
        self.btn_zoom_load_frame = ttk.Button(time_frame, text=get_text('btn_load_frame'), command=self.load_zoom_frame)
        self.btn_zoom_load_frame.pack(side='right')

        # Canvas Area (Split)
        split_frame = ttk.Frame(frame)
        split_frame.pack(fill='both', expand=True, pady=5)
        
        # Left Canvas (Original)
        self.zoom_left_frame = ttk.LabelFrame(split_frame, text=get_text('grp_original'))
        self.zoom_left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        self.canvas_orig = tk.Canvas(self.zoom_left_frame, bg='black', width=400, height=400)
        self.canvas_orig.pack(fill='both', expand=True)
        self.canvas_orig.bind("<ButtonPress-1>", self.on_zoom_press)
        self.canvas_orig.bind("<B1-Motion>", self.on_zoom_drag)
        self.canvas_orig.bind("<ButtonRelease-1>", self.on_zoom_release)
        
        # Right Canvas (Zoomed)
        self.zoom_right_frame = ttk.LabelFrame(split_frame, text=get_text('grp_zoomed'))
        self.zoom_right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        self.canvas_zoom = tk.Canvas(self.zoom_right_frame, bg='black', width=400, height=400)
        self.canvas_zoom.pack(fill='both', expand=True)

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='x', pady=5, side='bottom')
        
        self.zoom_log = tk.Text(log_frame, height=4, state='disabled')
        self.zoom_log.pack(fill='both', expand=True)

        self.orig_img = None
        self.tk_orig_img = None
        self.tk_zoom_img = None

    def browse_zoom_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if path:
            self.zoom_video_path.set(path)
            # Get duration
            duration = logic.get_video_duration(path)
            if duration:
                self.zoom_slider.config(to=duration)
                self.log(self.zoom_log, get_text('msg_loaded_video_duration').format(duration))
            else:
                self.log(self.zoom_log, get_text('msg_err_duration'))

    def on_zoom_slider_move(self, val):
        seconds = int(float(val))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        time_str = f"{h:02d}:{m:02d}:{s:02d}"
        self.zoom_time_lbl.config(text=time_str)

    def load_zoom_frame(self):
        video_path = self.zoom_video_path.get()
        timestamp = self.zoom_time_lbl.cget("text") # Use the formatted string
        
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", get_text('err_video'))
            return

        self.btn_zoom_load_frame.config(state='disabled')
        self.show_zoom_canvas_message(self.canvas_orig, get_text('msg_loading_frame'))
        self.canvas_zoom.delete("all")

        def task():
            self.log(self.zoom_log, get_text('msg_start_extract'))
            try:
                img = logic.extract_frame_image(video_path, timestamp)
                self.log(self.zoom_log, get_text('msg_done'))
                self.root.after(0, lambda: self.display_zoom_original(img))
            except Exception as e:
                self.log(self.zoom_log, f"Error: {e}")
                self.log(self.zoom_log, get_text('msg_err_frame'))
            finally:
                self.root.after(0, lambda: self.btn_zoom_load_frame.config(state='normal'))

        threading.Thread(target=task, daemon=True).start()

    def show_zoom_canvas_message(self, canvas, message):
        canvas.delete("all")
        cw = max(canvas.winfo_width(), 400)
        ch = max(canvas.winfo_height(), 400)
        canvas.create_text(cw//2, ch//2, text=message, fill="#e6e6e6", font=("Arial", 14, "bold"), anchor='center')

    def display_zoom_original(self, img):
        self.orig_img = img # Keep original high-res image
        
        cw = self.canvas_orig.winfo_width()
        ch = self.canvas_orig.winfo_height()
        
        iw, ih = img.size
        ratio = min(cw/iw, ch/ih)
        nw, nh = int(iw * ratio), int(ih * ratio)
        
        resized_img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_orig_img = ImageTk.PhotoImage(resized_img)
        
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(cw//2, ch//2, image=self.tk_orig_img, anchor='center')
        
        # Store geometry for click calculation: (x_offset, y_offset, width, height)
        self.orig_img_rect = ((cw - nw) // 2, (ch - nh) // 2, nw, nh)

    def on_zoom_press(self, event):
        self.zoom_start_x = event.x
        self.zoom_start_y = event.y
        if self.canvas_orig.find_withtag("selection"):
            self.canvas_orig.delete("selection")
        self.zoom_rect = self.canvas_orig.create_rectangle(self.zoom_start_x, self.zoom_start_y, self.zoom_start_x, self.zoom_start_y, outline='red', width=2, tags="selection")

    def on_zoom_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas_orig.coords(self.zoom_rect, self.zoom_start_x, self.zoom_start_y, cur_x, cur_y)

    def on_zoom_release(self, event):
        if not self.orig_img:
            return

        end_x, end_y = (event.x, event.y)
        
        # Normalize coordinates (top-left, bottom-right)
        x1 = min(self.zoom_start_x, end_x)
        y1 = min(self.zoom_start_y, end_y)
        x2 = max(self.zoom_start_x, end_x)
        y2 = max(self.zoom_start_y, end_y)
        
        # Minimum selection size
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return

        # Map to image coordinates
        off_x, off_y, w, h = self.orig_img_rect
        
        # Convert canvas coords to displayed image coords
        img_x1 = x1 - off_x
        img_y1 = y1 - off_y
        img_x2 = x2 - off_x
        img_y2 = y2 - off_y
        
        # Clamp to displayed image bounds
        img_x1 = max(0, min(img_x1, w))
        img_y1 = max(0, min(img_y1, h))
        img_x2 = max(0, min(img_x2, w))
        img_y2 = max(0, min(img_y2, h))
        
        if (img_x2 - img_x1) < 1 or (img_y2 - img_y1) < 1:
            return

        # Map to original high-res image coordinates
        orig_w, orig_h = self.orig_img.size
        scale_x = orig_w / w
        scale_y = orig_h / h
        
        crop_x1 = int(img_x1 * scale_x)
        crop_y1 = int(img_y1 * scale_y)
        crop_x2 = int(img_x2 * scale_x)
        crop_y2 = int(img_y2 * scale_y)
        
        # Crop and display
        cropped = self.orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        self.display_zoomed_image(cropped)

    def display_zoomed_image(self, img):
        cw = self.canvas_zoom.winfo_width()
        ch = self.canvas_zoom.winfo_height()
        
        iw, ih = img.size
        ratio = min(cw/iw, ch/ih)
        nw, nh = int(iw * ratio), int(ih * ratio)
        
        resized_img = img.resize((nw, nh), Image.Resampling.NEAREST) # Use NEAREST to see pixels/mosaic clearly
        self.tk_zoom_img = ImageTk.PhotoImage(resized_img)
        
        self.canvas_zoom.delete("all")
        self.canvas_zoom.create_image(cw//2, ch//2, image=self.tk_zoom_img, anchor='center')

    # --- Merge Tab ---
    def setup_merge_tab(self):
        frame = self.tab_merge
        
        # File List Container
        list_frame = ttk.LabelFrame(frame, text=get_text('lbl_file_list'), padding=10)
        list_frame.pack(fill='both', expand=True, pady=5)
        
        # Scrollable Frame for inputs
        canvas = tk.Canvas(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.merge_scroll_frame = ttk.Frame(canvas)
        
        self.merge_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.merge_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.merge_file_vars = []
        self.merge_rows = []
        
        # Initial 2 rows
        self.add_merge_row()
        self.add_merge_row()
        
        # Action Buttons (Add/Remove)
        action_frame = ttk.Frame(frame, padding=5)
        action_frame.pack(fill='x', pady=5)
        
        ttk.Button(action_frame, text=get_text('btn_add_file'), command=self.add_merge_row).pack(side='left', padx=5)
        ttk.Button(action_frame, text=get_text('btn_remove_file'), command=self.remove_merge_row).pack(side='left', padx=5)
        ttk.Label(action_frame, text=get_text('lbl_merge_warning'), foreground='red').pack(side='left', padx=5)
        
        # Merge Action
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_merge = ttk.Button(btn_frame, text=get_text('btn_merge_files'), command=self.run_merge)
        self.btn_merge.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_merge = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_merge, state='disabled')
        self.btn_stop_merge.pack(side='left', fill='x', expand=True, padx=(5, 0))

        self.proc_merge = None
        self.stop_merge_requested = False

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='x', pady=5, side='bottom')
        
        self.merge_log = tk.Text(log_frame, height=8, state='disabled')
        self.merge_log.pack(fill='both', expand=True)

    def add_merge_row(self):
        row_idx = len(self.merge_rows)
        row_frame = ttk.Frame(self.merge_scroll_frame, padding=2)
        row_frame.pack(fill='x', pady=2)
        
        ttk.Label(row_frame, text=f"{row_idx + 1}.").pack(side='left', padx=(0, 5))
        
        var = tk.StringVar()
        entry = ttk.Entry(row_frame, textvariable=var, width=60)
        entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        btn = ttk.Button(row_frame, text=get_text('btn_browse'), command=lambda v=var: self.browse_merge_file(v))
        btn.pack(side='right')
        
        self.merge_file_vars.append(var)
        self.merge_rows.append(row_frame)

    def remove_merge_row(self):
        if len(self.merge_rows) > 2:
            row = self.merge_rows.pop()
            row.destroy()
            self.merge_file_vars.pop()

    def browse_merge_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if path:
            var.set(path)

    def run_merge(self):
        files = [v.get() for v in self.merge_file_vars if v.get().strip()]
        
        if len(files) < 2:
            messagebox.showerror("Error", get_text('err_merge_count'))
            return
            
        for i, f in enumerate(files):
            if not os.path.exists(f):
                messagebox.showerror("Error", get_text('err_merge_file').format(i+1))
                return

        # Output path: merged.mp4 in first file's dir
        first_file = files[0]
        dirname = os.path.dirname(first_file)
        output_path = os.path.join(dirname, "merged.mp4")
        
        def _on_proc(p):
            self.proc_merge = p
            if self.stop_merge_requested:
                try: p.kill()
                except Exception: pass

        def task():
            start_time = time.time()
            self.log(self.merge_log, get_text('msg_start_merge'))
            try:
                logic.merge_files(
                    files,
                    output_path,
                    lambda msg: self.log(self.merge_log, msg),
                    _on_proc
                )
            except Exception as e:
                self.log(self.merge_log, f"Error: {e}")
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(self.merge_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_merge.config(state='normal'))
                self.root.after(0, lambda: self.btn_stop_merge.config(state='disabled'))
                self.proc_merge = None

        self.btn_merge.config(state='disabled')
        self.btn_stop_merge.config(state='normal')
        self.stop_merge_requested = False
        threading.Thread(target=task, daemon=True).start()

    def stop_merge(self):
        self.stop_merge_requested = True
        proc = self.proc_merge
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_merge = None
            self.log(self.merge_log, get_text('msg_stop'))

    # --- Quick Safe Cut Tab ---
    def setup_cut_tab(self):
        frame = self.tab_cut
        
        # File Selection
        file_frame = ttk.LabelFrame(frame, text=get_text('lbl_input_video'), padding=10)
        file_frame.pack(fill='x', pady=5)
        
        self.cut_video_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.cut_video_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(file_frame, text=get_text('btn_browse'), command=self.browse_cut_video).pack(side='right')
        
        self.cut_video_duration = 0.0
        self.lbl_cut_duration = ttk.Label(file_frame, text="")
        self.lbl_cut_duration.pack(side='right', padx=5)

        # Time List Container
        list_frame = ttk.LabelFrame(frame, text=get_text('lbl_time_list'), padding=10)
        list_frame.pack(fill='both', expand=True, pady=5)
        
        # Scrollable Frame for inputs
        canvas = tk.Canvas(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.cut_scroll_frame = ttk.Frame(canvas)
        
        self.cut_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.cut_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.cut_time_vars = []
        self.cut_rows = []
        
        # Initial 1 row
        self.add_cut_row()
        
        # Action Buttons (Add/Remove)
        action_frame = ttk.Frame(frame, padding=5)
        action_frame.pack(fill='x', pady=5)
        
        ttk.Button(action_frame, text=get_text('btn_add_time'), command=self.add_cut_row).pack(side='left', padx=5)
        ttk.Button(action_frame, text=get_text('btn_remove_time'), command=self.remove_cut_row).pack(side='left', padx=5)
        
        # Cut Action
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.pack(fill='x', pady=5)
        
        self.btn_cut = ttk.Button(btn_frame, text=get_text('btn_cut'), command=self.run_cut)
        self.btn_cut.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.btn_stop_cut = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_cut, state='disabled')
        self.btn_stop_cut.pack(side='left', fill='x', expand=True, padx=(5, 0))

        self.proc_cut = None
        self.stop_cut_requested = False

        # Log
        log_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'), padding=10)
        log_frame.pack(fill='x', pady=5, side='bottom')
        
        self.cut_log = tk.Text(log_frame, height=15, state='disabled')
        self.cut_log.pack(fill='both', expand=True)

    def setup_keyframe_tab(self):
        frame = self.tab_keyframe
        
        # Configure Grid
        frame.grid_columnconfigure(1, weight=1)
        
        # 1. Video Selection
        ttk.Label(frame, text=get_text('lbl_input_video')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.kf_video_path = tk.StringVar()
        
        input_frame = ttk.Frame(frame)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Entry(input_frame, textvariable=self.kf_video_path).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(input_frame, text=get_text('btn_browse'), command=self.browse_kf_video).pack(side='right')

        # 2. Eye Selection
        ttk.Label(frame, text=get_text('lbl_eye_select')).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.kf_eye_var = tk.StringVar(value='full')
        
        eye_frame = ttk.Frame(frame)
        eye_frame.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Radiobutton(eye_frame, text=get_text('opt_full'), variable=self.kf_eye_var, value='full').pack(side='left', padx=(0, 10))
        ttk.Radiobutton(eye_frame, text=get_text('opt_left'), variable=self.kf_eye_var, value='left').pack(side='left', padx=(0, 10))
        ttk.Radiobutton(eye_frame, text=get_text('opt_right'), variable=self.kf_eye_var, value='right').pack(side='left')

        # 3. Time Range - Start
        ttk.Label(frame, text=get_text('lbl_start')).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.kf_start_time = tk.StringVar()
        ttk.Entry(frame, textvariable=self.kf_start_time, width=20).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # 4. Time Range - End
        ttk.Label(frame, text=get_text('lbl_end')).grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.kf_end_time = tk.StringVar()
        ttk.Entry(frame, textvariable=self.kf_end_time, width=20).grid(row=3, column=1, padx=5, pady=5, sticky='w')

        # 5. Output Directory
        ttk.Label(frame, text=get_text('lbl_output_dir')).grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.kf_output_dir = tk.StringVar()
        
        out_frame = ttk.Frame(frame)
        out_frame.grid(row=4, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Entry(out_frame, textvariable=self.kf_output_dir).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(out_frame, text=get_text('btn_browse'), command=self.browse_kf_output).pack(side='right')
        
        # Note row
        ttk.Label(frame, text=get_text('lbl_output_dir_note')).grid(row=5, column=1, padx=5, pady=0, sticky='w')

        # 6. Extract Button
        btn_frame = ttk.Frame(frame, padding=10)
        btn_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        
        self.btn_kf_extract = ttk.Button(btn_frame, text=get_text('btn_extract_kf'), command=self.run_keyframe_extraction)
        self.btn_kf_extract.pack(fill='x')
        
        # 6.1 Note Row for Extract
        ttk.Label(frame, text=get_text('lbl_kf_note')).grid(row=7, column=0, columnspan=2, padx=15, pady=0, sticky='w')

        # 7. Log
        log_label_frame = ttk.LabelFrame(frame, text=get_text('lbl_log'))
        log_label_frame.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        
        frame.grid_rowconfigure(8, weight=1)
        
        self.kf_log = tk.Text(log_label_frame, height=10, state='disabled')
        self.kf_log.pack(fill='both', expand=True)

    def browse_kf_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if path:
            self.kf_video_path.set(path)

    def browse_kf_output(self):
        path = filedialog.askdirectory()
        if path:
            self.kf_output_dir.set(path)

    def run_keyframe_extraction(self):
        video_path = self.kf_video_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", get_text('err_video'))
            return
            
        start_t = self.kf_start_time.get()
        end_t = self.kf_end_time.get()
        
        valid, msg = self.validate_time_logic(start_t, end_t)
        if not valid:
            messagebox.showerror("Error", msg)
            return

        def process_thread():
            start_time = time.time()
            self.btn_kf_extract.config(state='disabled')
            
            # 1. Estimate,ignored it will be blocked
                
            # 2. Extract
            out_dir = self.kf_output_dir.get()
            if not out_dir.strip():
                # Default to same dir
                out_dir = os.path.dirname(video_path)
            
            self.log(self.kf_log, get_text('msg_kf_starting'))
            logic.batch_extract_keyframes(
                video_path,
                out_dir,
                start_t,
                end_t,
                self.kf_eye_var.get(),
                lambda msg: self.log(self.kf_log, msg)
            )
            
            self.log(self.kf_log, get_text('msg_done'))
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log(self.kf_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, lambda: self.btn_kf_extract.config(state='normal'))

        threading.Thread(target=process_thread, daemon=True).start()


    def browse_cut_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All Files", "*.*")])
        if path:
            self.cut_video_path.set(path)
            duration = logic.get_video_duration(path)
            if duration:
                self.cut_video_duration = duration
                self.lbl_cut_duration.config(text=f"{get_text('lbl_duration')}: {duration}s")
            else:
                self.cut_video_duration = 0.0
                self.lbl_cut_duration.config(text="")

    def add_cut_row(self):
        row_idx = len(self.cut_rows)
        row_frame = ttk.Frame(self.cut_scroll_frame, padding=2)
        row_frame.pack(fill='x', pady=2)
        
        ttk.Label(row_frame, text=f"{row_idx + 1}.").pack(side='left', padx=(0, 5))
        
        var = tk.StringVar()
        entry = ttk.Entry(row_frame, textvariable=var, width=20)
        entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Label(row_frame, text=get_text('lbl_timestamp')).pack(side='left')
        
        self.cut_time_vars.append(var)
        self.cut_rows.append(row_frame)

    def remove_cut_row(self):
        if len(self.cut_rows) > 1:
            row = self.cut_rows.pop()
            row.destroy()
            self.cut_time_vars.pop()

    def run_cut(self):
        video_path = self.cut_video_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", get_text('err_video'))
            return

        # Collect and Validate Times
        cut_points = []
        last_time_sec = -1.0
        
        for i, var in enumerate(self.cut_time_vars):
            t_str = var.get().strip()
            if not t_str: continue # Skip empty
            
            # Format Check
            valid, msg = self.validate_time_logic(t_str)
            if not valid:
                messagebox.showerror("Error", f"Row {i+1}: {msg}")
                return
            
            # Sequence Check
            t_sec = logic.parse_time_to_seconds(t_str)
            if t_sec is None:
                messagebox.showerror("Error", f"Row {i+1}: {get_text('err_invalid_format')}")
                return
                
            if t_sec <= last_time_sec:
                 messagebox.showerror("Error", get_text('err_time_sequence').format(i+1))
                 return
            
            # Max Duration Check
            if self.cut_video_duration > 0 and t_sec >= self.cut_video_duration:
                 messagebox.showerror("Error", get_text('err_time_max').format(i+1))
                 return

            last_time_sec = t_sec
            cut_points.append(t_str)
            
        if not cut_points:
            messagebox.showerror("Error", get_text('err_cut_count'))
            return

        def _on_proc(p):
            self.proc_cut = p
            if self.stop_cut_requested:
                try: p.kill()
                except Exception: pass

        def task():
            start_time = time.time()
            self.log(self.cut_log, get_text('msg_start_cut'))
            try:
                logic.quick_safe_cut(
                    video_path,
                    cut_points,
                    lambda msg: self.log(self.cut_log, msg),
                    _on_proc
                )
                self.log(self.cut_log, get_text('msg_done'))
            except Exception as e:
                self.log(self.cut_log, f"Error: {e}")
            finally:
                elapsed = time.time() - start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.log(self.cut_log, f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
                self.root.after(0, lambda: self.btn_cut.config(state='normal'))
                self.root.after(0, lambda: self.btn_stop_cut.config(state='disabled'))
                self.proc_cut = None

        self.btn_cut.config(state='disabled')
        self.btn_stop_cut.config(state='normal')
        self.stop_cut_requested = False
        threading.Thread(target=task, daemon=True).start()

    def stop_cut(self):
        self.stop_cut_requested = True
        proc = self.proc_cut
        if proc:
            try: proc.kill()
            except Exception: pass
            self.proc_cut = None
            self.log(self.cut_log, get_text('msg_stop'))
