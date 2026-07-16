import locale
import os
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk
from utils import app_config, i18n, ui_theme

try:
    from . import logic
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import logic


def get_text(key):
    return i18n.translate('subembed', key)


def distance_text(value):
    return get_text("meter").format(value)


class VRSubtitleEmbedApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text("title"))
        ui_theme.apply_theme(self.root)
        self.video_info = None
        self.current_process = None
        self.current_process_2d = None
        self.stop_requested = False
        self.stop_2d_requested = False

        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True)
        if on_return:
            self.root.config(menu=tk.Menu(self.root))

        # Full-height left rail: tool title on top, back-to-home pinned at the bottom
        self.notebook = ui_theme.ToolShell(
            main_frame,
            title=get_text("title"),
            back_text=get_text("btn_return"),
            on_back=on_return,
        )
        self.notebook.pack(fill="both", expand=True)

        self.create_preview_tab()
        self.create_process_tab()
        self.create_2d_tab()

        missing = logic.check_dependencies()
        if missing:
            self.log(get_text("missing_tools").format(", ".join(missing)))

    def create_preview_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text("tab_preview"), icon=ui_theme.TAB_ICONS["preview"])

        top = ttk.Frame(tab)
        top.pack(fill="x", padx=6, pady=4)
        self.video_var = tk.StringVar()
        self.ass_var = tk.StringVar()
        self.time_var = tk.DoubleVar(value=0)
        self.yaw_var = tk.DoubleVar(value=0)
        self.pitch_var = tk.DoubleVar(value=10)
        self.fov_var = tk.DoubleVar(value=90)
        self.alpha_var = tk.StringVar(value="0%")
        self.direction_var = tk.StringVar(value=get_text("horizontal_middle"))
        self.mode_var = tk.StringVar(value="dual")
        self.distance_var = tk.StringVar(value=distance_text(3))

        self._file_row(top, 0, get_text("video"), self.video_var, self.load_video)
        self._file_row(top, 1, get_text("ass"), self.ass_var, lambda: self.browse_file(self.ass_var, "*.ass *.srt"))

        time_frame = ttk.Frame(top)
        time_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(time_frame, text=get_text("time")).pack(side="left")
        self.time_slider = ttk.Scale(time_frame, from_=0, to=100, variable=self.time_var, command=self.on_time_move)
        self.time_slider.pack(side="left", fill="x", expand=True, padx=6)
        self.time_label = ttk.Label(time_frame, text="00:00:00")
        self.time_label.pack(side="left")

        param = ttk.Frame(top)
        param.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(param, text=get_text("subtitle_region")).pack(side="left", padx=(0, 6))
        for label, var, width in (
            ("yaw", self.yaw_var, 7),
            ("pitch", self.pitch_var, 7),
            ("fov", self.fov_var, 7),
        ):
            ttk.Label(param, text=get_text(label)).pack(side="left")
            ttk.Entry(param, textvariable=var, width=width).pack(side="left", padx=(2, 8))
        ttk.Label(param, text=get_text("region_note")).pack(side="left", padx=(2, 0))

        mode_frame = ttk.Frame(top)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(mode_frame, text=get_text("mode")).pack(side="left")
        for value, key in (("dual", "dual"), ("left", "left"), ("right", "right")):
            ttk.Radiobutton(
                mode_frame,
                text=get_text(key),
                variable=self.mode_var,
                value=value,
                command=self.update_preview_distance_visibility,
            ).pack(side="left", padx=5)
        self.preview_distance_frame = ttk.Frame(mode_frame)
        self.preview_distance_frame.pack(side="left", padx=(20, 0))
        ttk.Label(self.preview_distance_frame, text=get_text("distance")).pack(side="left", padx=(0, 2))
        self.distance_combo = ttk.Combobox(
            self.preview_distance_frame,
            textvariable=self.distance_var,
            width=12,
            state="readonly",
            values=[distance_text(i) for i in range(1, 11)],
        )
        self.distance_combo.current(2)
        self.distance_combo.pack(side="left")
        self.update_preview_distance_visibility()
        ttk.Button(mode_frame, text=get_text("mode_help"), command=self.show_mode_help).pack(side="left", padx=(20, 0))

        alpha_frame = ttk.Frame(top)
        alpha_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(alpha_frame, text=get_text("alpha")).pack(side="left")
        self.alpha_combo = ttk.Combobox(
            alpha_frame,
            textvariable=self.alpha_var,
            width=7,
            state="readonly",
            values=[f"{i}%" for i in range(0, 71, 10)],
        )
        self.alpha_combo.current(0)
        self.alpha_combo.pack(side="left", padx=(2, 14))
        ttk.Label(alpha_frame, text=get_text("direction")).pack(side="left")
        self.direction_combo = ttk.Combobox(
            alpha_frame,
            textvariable=self.direction_var,
            width=12,
            state="readonly",
            values=[
                get_text("horizontal_middle"),
                get_text("horizontal_top"),
                get_text("horizontal_bottom"),
                get_text("vertical_left"),
                get_text("vertical_middle"),
                get_text("vertical_right"),
            ],
        )
        self.direction_combo.current(0)
        self.direction_combo.pack(side="left", padx=(2, 8))

        btns = ttk.Frame(top)
        btns.grid(row=6, column=0, columnspan=3, sticky="w", pady=4)
        self.load_btn = ttk.Button(btns, text=get_text("btn_load"), command=self.load_original_frame)
        self.load_btn.pack(side="left", padx=4)
        self.preview_btn = ttk.Button(btns, text=get_text("btn_preview"), command=self.generate_preview)
        self.preview_btn.pack(side="left", padx=4)
        ttk.Button(btns, text=get_text("btn_send"), command=self.send_to_process).pack(side="left", padx=4)

        split = ttk.Frame(tab)
        split.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.LabelFrame(split, text=get_text("original"))
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        right = ttk.LabelFrame(split, text=get_text("preview"))
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.canvas_original = tk.Canvas(left, bg="black", width=420, height=320)
        self.canvas_original.pack(fill="both", expand=True)
        self.canvas_preview = tk.Canvas(right, bg="black", width=420, height=320)
        self.canvas_preview.pack(fill="both", expand=True)
        self.canvas_preview.bind("<Double-Button-1>", self.open_preview_viewer)
        self.canvas_original.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas_original.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas_original.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.orig_img_ref = None
        self.preview_img_ref = None
        self.preview_original_img = None
        self.orig_img_rect = None

        log_frame = ttk.LabelFrame(tab, text=get_text("log"))
        log_frame.pack(fill="x", padx=6, pady=4)
        self.log_preview = scrolledtext.ScrolledText(log_frame, height=4)
        self.log_preview.pack(fill="both", expand=True)

    def create_process_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text("tab_process"), icon=ui_theme.TAB_ICONS["video"])

        self.proc_video_var = tk.StringVar()
        self.proc_ass_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.start_var = tk.StringVar(value="00:00:00")
        self.duration_var = tk.StringVar(value="15")
        self.custom_minutes_var = tk.StringVar(value="5")
        self.proc_yaw_var = tk.DoubleVar(value=0)
        self.proc_pitch_var = tk.DoubleVar(value=10)
        self.proc_fov_var = tk.DoubleVar(value=90)
        self.proc_alpha_var = tk.StringVar(value="0%")
        self.proc_direction_var = tk.StringVar(value=get_text("horizontal_middle"))
        self.proc_mode_var = tk.StringVar(value="dual")
        self.proc_distance_var = tk.StringVar(value=distance_text(10))

        self._file_row(tab, 0, get_text("video"), self.proc_video_var, lambda: self.browse_file(self.proc_video_var, "*.mp4 *.mkv *.webm"))
        self._file_row(tab, 1, get_text("ass"), self.proc_ass_var, lambda: self.browse_file(self.proc_ass_var, "*.ass *.srt"))
        self._file_row(tab, 2, get_text("output"), self.output_var, self.browse_output)

        param_row = ttk.Frame(tab)
        param_row.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(param_row, text=get_text("subtitle_region")).pack(side="left", padx=(0, 6))
        for label, var in (
            ("yaw", self.proc_yaw_var),
            ("pitch", self.proc_pitch_var),
            ("fov", self.proc_fov_var),
        ):
            ttk.Label(param_row, text=get_text(label)).pack(side="left")
            ttk.Entry(param_row, textvariable=var, width=8).pack(side="left", padx=(2, 8))
        ttk.Label(param_row, text=get_text("region_note")).pack(side="left", padx=(2, 0))

        mode_row = ttk.Frame(tab)
        mode_row.grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(mode_row, text=get_text("mode")).pack(side="left")
        for value, key in (("dual", "dual"), ("left", "left"), ("right", "right")):
            ttk.Radiobutton(
                mode_row,
                text=get_text(key),
                variable=self.proc_mode_var,
                value=value,
                command=self.update_process_distance_visibility,
            ).pack(side="left", padx=5)
        self.process_distance_frame = ttk.Frame(mode_row)
        self.process_distance_frame.pack(side="left", padx=(20, 0))
        ttk.Label(self.process_distance_frame, text=get_text("distance")).pack(side="left")
        self.proc_distance_combo = ttk.Combobox(
            self.process_distance_frame,
            textvariable=self.proc_distance_var,
            width=12,
            state="readonly",
            values=[distance_text(i) for i in range(1, 11)],
        )
        self.proc_distance_combo.current(9)
        self.proc_distance_combo.pack(side="left", padx=(2, 0))
        self.update_process_distance_visibility()
        ttk.Button(mode_row, text=get_text("mode_help"), command=self.show_mode_help).pack(side="left", padx=(20, 0))

        alpha_row = ttk.Frame(tab)
        alpha_row.grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(alpha_row, text=get_text("alpha")).pack(side="left")
        self.proc_alpha_combo = ttk.Combobox(
            alpha_row,
            textvariable=self.proc_alpha_var,
            width=7,
            state="readonly",
            values=[f"{i}%" for i in range(0, 71, 10)],
        )
        self.proc_alpha_combo.current(0)
        self.proc_alpha_combo.pack(side="left", padx=(2, 14))
        ttk.Label(alpha_row, text=get_text("direction")).pack(side="left")
        self.proc_direction_combo = ttk.Combobox(
            alpha_row,
            textvariable=self.proc_direction_var,
            width=12,
            state="readonly",
            values=[
                get_text("horizontal_middle"),
                get_text("horizontal_top"),
                get_text("horizontal_bottom"),
                get_text("vertical_left"),
                get_text("vertical_middle"),
                get_text("vertical_right"),
            ],
        )
        self.proc_direction_combo.current(0)
        self.proc_direction_combo.pack(side="left", padx=(2, 8))

        time_row = ttk.Frame(tab)
        time_row.grid(row=6, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(time_row, text=get_text("start")).pack(side="left")
        ttk.Entry(time_row, textvariable=self.start_var, width=10).pack(side="left", padx=(2, 14))
        ttk.Label(time_row, text=get_text("duration")).pack(side="left")
        self.duration_combo = ttk.Combobox(
            time_row,
            textvariable=self.duration_var,
            width=12,
            state="readonly",
        )
        self.duration_combo["values"] = [get_text("s15"), get_text("s30"), get_text("s60"), get_text("custom_minutes"), get_text("full")]
        self.duration_combo.current(0)
        self.duration_combo.pack(side="left", padx=(2, 8))
        self.duration_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_custom_minutes_visibility())
        self.custom_minutes_frame = ttk.Frame(time_row)
        ttk.Label(self.custom_minutes_frame, text=get_text("minutes")).pack(side="left")
        vcmd = (self.root.register(self.validate_digits), "%P")
        ttk.Entry(self.custom_minutes_frame, textvariable=self.custom_minutes_var, width=6, validate="key", validatecommand=vcmd).pack(side="left", padx=(2, 0))
        self.update_custom_minutes_visibility()

        btns = ttk.Frame(tab)
        btns.grid(row=7, column=0, columnspan=3, sticky="w", padx=5, pady=12)
        self.process_btn = ttk.Button(btns, text=get_text("btn_process"), command=self.run_process)
        self.process_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btns, text=get_text("btn_stop"), command=self.stop_process, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        log_frame = ttk.LabelFrame(tab, text=get_text("log"))
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        tab.grid_rowconfigure(8, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.log_process = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_process.pack(fill="both", expand=True)

    def create_2d_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=get_text("tab_2d"), icon=ui_theme.TAB_ICONS["frame"])

        self.flat_video_var = tk.StringVar()
        self.flat_subtitle_var = tk.StringVar()
        self.flat_output_var = tk.StringVar()
        self.flat_start_var = tk.StringVar(value="00:00:00")
        self.flat_duration_var = tk.StringVar(value=get_text("s15"))
        self.flat_custom_minutes_var = tk.StringVar(value="5")

        self._file_row(tab, 0, get_text("video"), self.flat_video_var, self.browse_flat_video)
        self._file_row(tab, 1, get_text("ass"), self.flat_subtitle_var, lambda: self.browse_file(self.flat_subtitle_var, "*.ass *.srt"))
        self._file_row(tab, 2, get_text("output"), self.flat_output_var, self.browse_flat_output)

        time_row = ttk.Frame(tab)
        time_row.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(time_row, text=get_text("start")).pack(side="left")
        ttk.Entry(time_row, textvariable=self.flat_start_var, width=10).pack(side="left", padx=(2, 14))
        ttk.Label(time_row, text=get_text("duration")).pack(side="left")
        self.flat_duration_combo = ttk.Combobox(
            time_row,
            textvariable=self.flat_duration_var,
            width=12,
            state="readonly",
            values=[get_text("s15"), get_text("s30"), get_text("s60"), get_text("custom_minutes"), get_text("full")],
        )
        self.flat_duration_combo.current(0)
        self.flat_duration_combo.pack(side="left", padx=(2, 8))
        self.flat_duration_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_flat_custom_minutes_visibility())
        self.flat_custom_minutes_frame = ttk.Frame(time_row)
        ttk.Label(self.flat_custom_minutes_frame, text=get_text("minutes")).pack(side="left")
        vcmd = (self.root.register(self.validate_digits), "%P")
        ttk.Entry(self.flat_custom_minutes_frame, textvariable=self.flat_custom_minutes_var, width=6, validate="key", validatecommand=vcmd).pack(side="left", padx=(2, 0))
        self.update_flat_custom_minutes_visibility()

        btns = ttk.Frame(tab)
        btns.grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=12)
        self.flat_process_btn = ttk.Button(btns, text=get_text("btn_process"), command=self.run_2d_process)
        self.flat_process_btn.pack(side="left", padx=4)
        self.flat_stop_btn = ttk.Button(btns, text=get_text("btn_stop"), command=self.stop_2d_process, state="disabled")
        self.flat_stop_btn.pack(side="left", padx=4)

        log_frame = ttk.LabelFrame(tab, text=get_text("log"))
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        tab.grid_rowconfigure(5, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        self.flat_log = scrolledtext.ScrolledText(log_frame, height=12)
        self.flat_log.pack(fill="both", expand=True)

    def _file_row(self, parent, row, label, var, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=4)
        ttk.Entry(parent, textvariable=var, width=62).grid(row=row, column=1, sticky="ew", padx=5, pady=4)
        ttk.Button(parent, text=get_text("btn_browse"), command=command).grid(row=row, column=2, padx=5, pady=4)
        parent.columnconfigure(1, weight=1)

    def browse_file(self, var, pattern):
        path = filedialog.askopenfilename(filetypes=[("Supported files", pattern), ("All files", "*.*")])
        if path:
            var.set(path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if path:
            self.output_var.set(path)

    def browse_flat_video(self):
        self.browse_file(self.flat_video_var, "*.mp4 *.mkv *.webm")
        if self.flat_video_var.get() and not self.flat_output_var.get():
            self.flat_output_var.set(logic.default_2d_output_path(self.flat_video_var.get()))

    def browse_flat_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if path:
            self.flat_output_var.set(path)

    def validate_digits(self, value):
        return value == "" or value.isdigit()

    def load_video(self):
        self.browse_file(self.video_var, "*.mp4 *.mkv *.webm")
        if not self.video_var.get():
            return
        info = logic.get_video_info(self.video_var.get())
        if info:
            self.video_info = info
            self.time_slider.config(to=info["duration"])
            self.log(get_text("loaded_video").format(info["width"], info["height"], info["duration"]))

    def on_time_move(self, value):
        seconds = int(float(value))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        self.time_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")

    def update_preview_distance_visibility(self):
        if self.mode_var.get() == "dual":
            self.preview_distance_frame.pack(side="left", padx=(20, 0))
        else:
            self.preview_distance_frame.pack_forget()

    def update_process_distance_visibility(self):
        if self.proc_mode_var.get() == "dual":
            self.process_distance_frame.pack(side="left", padx=(20, 0))
        else:
            self.process_distance_frame.pack_forget()

    def update_custom_minutes_visibility(self):
        if self.duration_combo.get() == get_text("custom_minutes"):
            self.custom_minutes_frame.pack(side="left", padx=(8, 0))
        else:
            self.custom_minutes_frame.pack_forget()

    def update_flat_custom_minutes_visibility(self):
        if self.flat_duration_combo.get() == get_text("custom_minutes"):
            self.flat_custom_minutes_frame.pack(side="left", padx=(8, 0))
        else:
            self.flat_custom_minutes_frame.pack_forget()

    def show_mode_help(self):
        messagebox.showinfo(get_text("mode_help_title"), get_text("mode_help_msg"))

    def get_distance_m(self, var):
        raw = str(var.get()).strip().lower().replace("米", "").replace("m", "")
        try:
            return max(1, min(10, int(float(raw))))
        except ValueError:
            return 10

    def get_transparency_percent(self, var):
        raw = str(var.get()).strip().replace("%", "")
        try:
            return max(0, min(70, int(float(raw))))
        except ValueError:
            return 0

    def subtitle_direction_value(self, var):
        value = str(var.get()).strip().lower()
        if value in ("horizontal_middle", "horizontal", get_text("horizontal_middle").lower()):
            return "horizontal_middle"
        if value in ("horizontal_top", get_text("horizontal_top").lower()):
            return "horizontal_top"
        if value in ("horizontal_bottom", get_text("horizontal_bottom").lower()):
            return "horizontal_bottom"
        if value in ("vertical_left", "vertical", get_text("vertical_left").lower()):
            return "vertical_left"
        if value in ("vertical_middle", get_text("vertical_middle").lower()):
            return "vertical_middle"
        if value in ("vertical_right", get_text("vertical_right").lower()):
            return "vertical_right"
        return "horizontal_middle"

    def load_original_frame(self):
        if not self.video_var.get():
            return
        self.load_btn.config(state="disabled")
        self.show_canvas_message(self.canvas_original, get_text("loading_frame"))
        threading.Thread(target=self._load_original_thread, daemon=True).start()

    def _load_original_thread(self):
        try:
            if not self.video_var.get():
                return
            image = logic.get_left_eye_frame_image(self.video_var.get(), self.time_var.get())
            self.root.after(0, lambda: self.display_pil_image(image, self.canvas_original, "orig"))
        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.load_btn.config(state="normal"))

    def generate_preview(self):
        if not self.video_var.get() or not self.ass_var.get():
            self.log(get_text("select_video_ass"))
            return
        self.preview_btn.config(state="disabled")
        self.show_canvas_message(self.canvas_preview, get_text("generating_preview"))
        self.preview_original_img = None
        threading.Thread(target=self._preview_thread, daemon=True).start()

    def _preview_thread(self):
        try:
            if not self.video_var.get() or not self.ass_var.get():
                self.log(get_text("select_video_ass"))
                return
            out = os.path.join(tempfile.gettempdir(), "vr_subembed_preview.jpg")
            logic.generate_preview(
                self.video_var.get(),
                self.ass_var.get(),
                self.time_var.get(),
                self.fov_var.get(),
                self.yaw_var.get(),
                self.pitch_var.get(),
                self.get_transparency_percent(self.alpha_var),
                self.mode_var.get(),
                self.get_distance_m(self.distance_var),
                self.subtitle_direction_value(self.direction_var),
                out,
                [get_text("preview_test_text")],
            )
            self.root.after(0, lambda: self.display_image(out, self.canvas_preview, "preview"))
            self.log(get_text("preview_done"))
        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.preview_btn.config(state="normal"))

    def show_canvas_message(self, canvas, message):
        canvas.delete("all")
        cw = max(canvas.winfo_width(), 420)
        ch = max(canvas.winfo_height(), 320)
        canvas.create_text(
            cw // 2,
            ch // 2,
            text=message,
            fill="#e6e6e6",
            font=("Arial", 14, "bold"),
            anchor="center",
        )

    def display_image(self, path, canvas, kind):
        img = Image.open(path)
        self.display_pil_image(img, canvas, kind)

    def display_pil_image(self, img, canvas, kind):
        cw = max(canvas.winfo_width(), 10)
        ch = max(canvas.winfo_height(), 10)
        ratio = min(cw / img.width, ch / img.height)
        nw, nh = max(1, int(img.width * ratio)), max(1, int(img.height * ratio))
        shown = img.resize((nw, nh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(shown)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")
        if kind == "orig":
            self.orig_img_ref = tk_img
            self.orig_img_rect = ((cw - nw) // 2, (ch - nh) // 2, nw, nh)
        else:
            self.preview_img_ref = tk_img
            self.preview_original_img = img.copy()

    def open_preview_viewer(self, _event=None):
        if self.preview_original_img is None:
            self.log(get_text("preview_required"))
            return
        ImageZoomWindow(self.root, self.preview_original_img, get_text("zoom_hint"))

    def on_canvas_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.canvas_original.delete("selection")
        self.selection_rect = self.canvas_original.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2, tags="selection"
        )

    def on_canvas_drag(self, event):
        if hasattr(self, "selection_rect"):
            self.canvas_original.coords(self.selection_rect, self.start_x, self.start_y, event.x, event.y)

    def on_canvas_release(self, event):
        if not self.orig_img_rect:
            return
        off_x, off_y, w, h = self.orig_img_rect
        x1 = max(0, min(self.start_x - off_x, w))
        x2 = max(0, min(event.x - off_x, w))
        y1 = max(0, min(self.start_y - off_y, h))
        y2 = max(0, min(event.y - off_y, h))
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        sel_w = max(1, abs(x2 - x1))
        yaw = (center_x / w) * 180 - 90
        pitch = 90 - (center_y / h) * 180
        fov = max(10, min(130, (sel_w / w) * 180))
        self.yaw_var.set(f"{yaw:.2f}")
        self.pitch_var.set(f"{pitch:.2f}")
        self.fov_var.set(f"{fov:.2f}")
        self.log(get_text("selected_region").format(yaw, pitch, fov))

    def send_to_process(self):
        self.proc_video_var.set(self.video_var.get())
        self.proc_ass_var.set(self.ass_var.get())
        self.proc_yaw_var.set(self.yaw_var.get())
        self.proc_pitch_var.set(self.pitch_var.get())
        self.proc_fov_var.set(self.fov_var.get())
        self.proc_alpha_var.set(self.alpha_var.get())
        self.proc_direction_var.set(self.direction_var.get())
        self.proc_mode_var.set(self.mode_var.get())
        self.proc_distance_var.set(self.distance_var.get())
        self.update_process_distance_visibility()
        self.start_var.set("00:00:00")
        if self.video_var.get() and not self.output_var.get():
            self.output_var.set(
                logic.default_output_path(
                    self.video_var.get(), self.mode_var.get(), self.fov_var.get(), self.yaw_var.get(), self.pitch_var.get()
                )
            )
        self.notebook.select(1)

    def format_time(self, seconds):
        seconds = int(float(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def selected_duration(self):
        index = self.duration_combo.current()
        if index == 3:
            minutes = int(self.custom_minutes_var.get() or "0")
            return max(1, minutes) * 60
        if index == 4:
            return None
        return [15, 30, 60][index if index >= 0 else 0]

    def selected_2d_duration(self):
        index = self.flat_duration_combo.current()
        if index == 3:
            minutes = int(self.flat_custom_minutes_var.get() or "0")
            return max(1, minutes) * 60
        if index == 4:
            return None
        return [15, 30, 60][index if index >= 0 else 0]

    def run_process(self):
        if not self.proc_video_var.get() or not self.proc_ass_var.get():
            messagebox.showerror("Error", get_text("select_video_ass"))
            return
        if not self.output_var.get():
            self.output_var.set(
                logic.default_output_path(
                    self.proc_video_var.get(),
                    self.proc_mode_var.get(),
                    self.proc_fov_var.get(),
                    self.proc_yaw_var.get(),
                    self.proc_pitch_var.get(),
                )
            )
        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.stop_requested = False
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.current_process = p
            if self.stop_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_embed(
                self.proc_video_var.get(),
                self.proc_ass_var.get(),
                self.output_var.get(),
                self.start_var.get(),
                self.selected_duration(),
                self.proc_fov_var.get(),
                self.proc_yaw_var.get(),
                self.proc_pitch_var.get(),
                self.get_transparency_percent(self.proc_alpha_var),
                self.proc_mode_var.get(),
                self.get_distance_m(self.proc_distance_var),
                self.subtitle_direction_value(self.proc_direction_var),
                log_callback=self.log_process_msg,
                process_callback=_on_proc,
            )
            self.log_process_msg(get_text("process_done").format(self.output_var.get()))
        except Exception as e:
            self.log_process_msg(f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_process_msg(f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, self.reset_process_buttons)

    def stop_process(self):
        self.stop_requested = True
        proc = self.current_process
        if proc:
            try: proc.kill()
            except Exception: pass
            self.current_process = None
            self.log_process_msg(get_text("process_stopped"))

    def reset_process_buttons(self):
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.current_process = None

    def run_2d_process(self):
        if not self.flat_video_var.get() or not self.flat_subtitle_var.get():
            messagebox.showerror("Error", get_text("select_video_ass"))
            return
        if not self.flat_output_var.get():
            self.flat_output_var.set(logic.default_2d_output_path(self.flat_video_var.get()))
        self.flat_process_btn.config(state="disabled")
        self.flat_stop_btn.config(state="normal")
        self.stop_2d_requested = False
        threading.Thread(target=self._process_2d_thread, daemon=True).start()

    def _process_2d_thread(self):
        start_time = time.time()
        def _on_proc(p):
            self.current_process_2d = p
            if self.stop_2d_requested:
                try: p.kill()
                except Exception: pass
        try:
            logic.run_embed_2d(
                self.flat_video_var.get(),
                self.flat_subtitle_var.get(),
                self.flat_output_var.get(),
                self.flat_start_var.get(),
                self.selected_2d_duration(),
                log_callback=self.log_2d_msg,
                process_callback=_on_proc,
            )
            self.log_2d_msg(get_text("process_done").format(self.flat_output_var.get()))
        except Exception as e:
            self.log_2d_msg(f"Error: {e}")
        finally:
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.log_2d_msg(f"[System] Process completed. Total time elapsed: {h} hours, {m} minutes, {s} seconds.")
            self.root.after(0, self.reset_2d_process_buttons)

    def stop_2d_process(self):
        self.stop_2d_requested = True
        proc = self.current_process_2d
        if proc:
            try: proc.kill()
            except Exception: pass
            self.current_process_2d = None
            self.log_2d_msg(get_text("process_stopped"))

    def reset_2d_process_buttons(self):
        self.flat_process_btn.config(state="normal")
        self.flat_stop_btn.config(state="disabled")
        self.current_process_2d = None

    def log(self, message):
        self.root.after(0, lambda: self._append_log(self.log_preview, message))

    def log_process_msg(self, message):
        self.root.after(0, lambda: self._append_log(self.log_process, message))

    def log_2d_msg(self, message):
        self.root.after(0, lambda: self._append_log(self.flat_log, message))

    def _append_log(self, widget, message):
        widget.insert(tk.END, str(message) + "\n")
        ui_theme.scroll_text_to_end(widget)


class ImageZoomWindow:
    def __init__(self, parent, image, hint):
        self.image = image
        self.scale = 1.0
        self.tk_img = None

        self.window = tk.Toplevel(parent)
        self.window.title(hint)
        self.window.geometry("1100x760")

        frame = ttk.Frame(self.window)
        frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(frame, bg="black")
        xbar = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        hint_label = ttk.Label(self.window, text=hint)
        hint_label.pack(fill="x", padx=8, pady=(2, 6))

        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        self.render()

    def on_mousewheel(self, event):
        direction = 1
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            direction = -1
        factor = 1.15 if direction > 0 else 1 / 1.15
        self.scale = max(0.1, min(8.0, self.scale * factor))
        self.render()

    def render(self):
        w = max(1, int(self.image.width * self.scale))
        h = max(1, int(self.image.height * self.scale))
        shown = self.image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, w, h))


if __name__ == "__main__":
    root = tk.Tk()
    app = VRSubtitleEmbedApp(root)
    root.mainloop()
