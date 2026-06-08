import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from utils import i18n

try:
    from . import logic
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import logic


def get_text(key):
    return i18n.translate("two_d_vr", key)


def _format_seconds_as_time(seconds):
    value = max(0.0, round(float(seconds), 3))
    whole = int(value)
    frac = value - whole
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if frac > 0:
        sec_text = f"{secs + frac:06.3f}".rstrip("0").rstrip(".")
    else:
        sec_text = f"{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{sec_text}"


class TwoDToVRApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text("title"))
        self.proc = None
        self.stop_requested = False

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill="both", expand=True)

        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(header_frame, text=get_text("title"), font=("Arial", 14, "bold")).pack(side="left")
        if self.on_return:
            ttk.Button(header_frame, text=get_text("btn_back"), command=self.go_back).pack(side="right")

        self._create_form()
        self._create_log()
        self._check_dependencies()

    def _create_form(self):
        form = ttk.LabelFrame(self.main_frame, text=get_text("grp_input"), padding=10)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.start_var = tk.StringVar(value="00:00:00")
        self.duration_var = tk.StringVar(value=get_text("opt_duration_30s"))
        self.custom_minutes_var = tk.StringVar(value="5")
        self.until_time_var = tk.StringVar(value="00:00:30")
        self.projection_var = tk.StringVar(value=logic.DEFAULT_PROJECTION)
        self.hole_fill_var = tk.StringVar()
        self.stabilize_var = tk.StringVar()
        self.eye_distance_var = tk.StringVar(value=str(int(logic.DEFAULT_EYE_DISTANCE_MM)))

        ttk.Label(form, text=get_text("lbl_input")).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(form, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(form, text=get_text("btn_browse"), command=self.browse_file).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(form, text=get_text("lbl_output")).grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(form, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(form, text=get_text("btn_browse"), command=self.browse_dir).grid(row=1, column=2, padx=4, pady=4)

        time_frame = ttk.Frame(form)
        time_frame.grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(form, text=get_text("lbl_time")).grid(row=2, column=0, sticky="w", padx=4, pady=4)
        vcmd = (self.root.register(self.validate_time_input), "%P")
        number_vcmd = (self.root.register(self.validate_number_input), "%P")
        ttk.Label(time_frame, text=get_text("lbl_start")).pack(side="left")
        ttk.Entry(time_frame, textvariable=self.start_var, width=12, validate="key", validatecommand=vcmd).pack(side="left", padx=(4, 12))
        ttk.Label(time_frame, text=get_text("lbl_duration")).pack(side="left")
        self.duration_combo = ttk.Combobox(
            time_frame,
            textvariable=self.duration_var,
            values=list(self._duration_display_map().keys()),
            state="readonly",
            width=18,
        )
        self.duration_combo.pack(side="left", padx=(4, 0))
        self.duration_combo.bind("<<ComboboxSelected>>", self._update_time_controls)
        self.custom_minutes_label = ttk.Label(time_frame, text=get_text("lbl_custom_minutes"))
        self.custom_minutes_entry = ttk.Entry(
            time_frame,
            textvariable=self.custom_minutes_var,
            width=6,
            validate="key",
            validatecommand=number_vcmd,
        )
        self.until_time_label = ttk.Label(time_frame, text=get_text("lbl_end_time"))
        self.until_time_entry = ttk.Entry(
            time_frame,
            textvariable=self.until_time_var,
            width=12,
            validate="key",
            validatecommand=vcmd,
        )
        self._update_time_controls()

        projection_frame = ttk.Frame(form)
        projection_frame.grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(form, text=get_text("lbl_projection")).grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(
            projection_frame,
            text=get_text("opt_flat3d"),
            variable=self.projection_var,
            value=logic.PROJECTION_FLAT_3D,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            projection_frame,
            text=get_text("opt_hequirect"),
            variable=self.projection_var,
            value=logic.PROJECTION_HEQUIRECT,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            projection_frame,
            text=get_text("opt_fisheye"),
            variable=self.projection_var,
            value=logic.PROJECTION_FISHEYE,
        ).pack(side="left")

        # Visible UI options are limited to the two production-quality modes.
        # The rest (shift_fill / background / inpaint / e2fgvi / none) remain
        # selectable via the TOOL_2DVR_HOLE_FILL env var for debugging, but are
        # hidden from the GUI because their stereo output is either too slow
        # or visibly inferior to soft_shift / inverse_warp.
        self.hole_fill_options = [
            (get_text("opt_hole_soft_shift"), "soft_shift"),
            (get_text("opt_hole_inverse_warp"), "inverse_warp"),
        ]
        # Keep label->value lookup complete so an env-selected mode still maps
        # back to a display label without raising.
        self.hole_fill_display_to_value = {
            get_text("opt_hole_soft_shift"): "soft_shift",
            get_text("opt_hole_inverse_warp"): "inverse_warp",
            get_text("opt_hole_shift_fill"): "shift_fill",
            get_text("opt_hole_background"): "background",
            get_text("opt_hole_inpaint"): "inpaint",
            get_text("opt_hole_e2fgvi"): "e2fgvi",
            get_text("opt_hole_none"): "none",
        }
        default_hole_label = next(
            label for label, value in self.hole_fill_options
            if value == logic.DEFAULT_HOLE_FILL_MODE
        )
        self.hole_fill_var.set(default_hole_label)
        ttk.Label(form, text=get_text("lbl_hole_fill")).grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            form,
            textvariable=self.hole_fill_var,
            values=[label for label, _ in self.hole_fill_options],
            state="readonly",
            width=24,
        ).grid(row=4, column=1, sticky="w", padx=4, pady=4)

        self.stabilize_options = [
            (get_text("opt_stabilize_auto"), "auto"),
            (get_text("opt_stabilize_off"), "off"),
            (get_text("opt_stabilize_full"), "full"),
        ]
        self.stabilize_display_to_value = {label: value for label, value in self.stabilize_options}
        default_stabilize_label = next(
            label for label, value in self.stabilize_options
            if value == logic.DEFAULT_STABILIZE_MODE
        )
        self.stabilize_var.set(default_stabilize_label)
        ttk.Label(form, text=get_text("lbl_stabilize")).grid(row=5, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            form,
            textvariable=self.stabilize_var,
            values=[label for label, _ in self.stabilize_options],
            state="readonly",
            width=24,
        ).grid(row=5, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(form, text=get_text("lbl_eye_distance")).grid(row=6, column=0, sticky="w", padx=4, pady=4)
        eye_frame = ttk.Frame(form)
        eye_frame.grid(row=6, column=1, sticky="w", padx=4, pady=4)
        ttk.Entry(eye_frame, textvariable=self.eye_distance_var, width=10).pack(side="left")
        ttk.Label(eye_frame, text="mm").pack(side="left", padx=(4, 0))

        ttk.Label(form, text=get_text("lbl_model")).grid(row=7, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(form, text=str(logic.default_da3_dir()), foreground="gray").grid(row=7, column=1, columnspan=2, sticky="w", padx=4, pady=4)

        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill="x", pady=10)
        self.btn_start = ttk.Button(btn_frame, text=get_text("btn_start"), command=self.run_conversion)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(btn_frame, text=get_text("btn_stop"), command=self.stop_process, state="disabled")
        self.btn_stop.pack(side="left", padx=4)

    def _create_log(self):
        log_frame = ttk.LabelFrame(self.main_frame, text=get_text("grp_log"), padding=5)
        log_frame.pack(fill="both", expand=True, pady=(0, 4))
        self.log_text = tk.Text(log_frame, height=12, state="disabled")
        self.log_text.pack(fill="both", expand=True, side="left")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _check_dependencies(self):
        missing = logic.check_dependencies()
        if missing:
            self.log(get_text("warn_dep").format(", ".join(missing)))
            self.log(get_text("warn_model_path").format(logic.default_da3_dir()))

    def go_back(self):
        if self.on_return:
            self.on_return()
        else:
            self.root.quit()

    def browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All files", "*.*")]
        )
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                self.output_var.set(os.path.dirname(path))

    def browse_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def validate_time_input(self, value):
        if value == "":
            return True
        return all(ch in "0123456789:." for ch in value)

    def validate_number_input(self, value):
        if value == "":
            return True
        return all(ch in "0123456789." for ch in value)

    def log(self, message):
        def _do():
            self.log_text.config(state="normal")
            self.log_text.insert("end", str(message) + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")

        self.root.after(0, _do)

    def _read_eye_distance(self):
        try:
            value = float(self.eye_distance_var.get().strip())
        except ValueError as exc:
            raise ValueError(get_text("err_eye_distance")) from exc
        if value <= 0:
            raise ValueError(get_text("err_eye_distance"))
        return value

    def _read_hole_fill_mode(self):
        label = self.hole_fill_var.get()
        return self.hole_fill_display_to_value.get(label, logic.DEFAULT_HOLE_FILL_MODE)

    def _read_stabilize_mode(self):
        label = self.stabilize_var.get()
        return self.stabilize_display_to_value.get(label, logic.DEFAULT_STABILIZE_MODE)

    def _duration_display_map(self):
        return {
            get_text("opt_duration_15s"): 15.0,
            get_text("opt_duration_30s"): 30.0,
            get_text("opt_duration_60s"): 60.0,
            get_text("opt_duration_custom_minutes"): "custom_minutes",
            get_text("opt_duration_until_time"): "until_time",
            get_text("opt_duration_all"): None,
        }

    def _update_time_controls(self, _event=None):
        selected = self._duration_display_map().get(self.duration_var.get(), 30.0)
        if selected == "custom_minutes":
            self.custom_minutes_label.pack(side="left", padx=(10, 4))
            self.custom_minutes_entry.pack(side="left")
        else:
            self.custom_minutes_label.pack_forget()
            self.custom_minutes_entry.pack_forget()

        if selected == "until_time":
            self.until_time_label.pack(side="left", padx=(10, 4))
            self.until_time_entry.pack(side="left")
        else:
            self.until_time_label.pack_forget()
            self.until_time_entry.pack_forget()

    def _read_time_range(self):
        start = self.start_var.get().strip() or "00:00:00"
        try:
            start_sec = logic.parse_time_to_seconds(start)
        except ValueError as exc:
            raise ValueError(get_text("err_time_range")) from exc
        if start_sec is None:
            start_sec = 0.0
            start = "00:00:00"

        selected = self._duration_display_map().get(self.duration_var.get(), 30.0)
        if selected == "custom_minutes":
            try:
                minutes = float((self.custom_minutes_var.get() or "").strip())
            except ValueError as exc:
                raise ValueError(get_text("err_time_range")) from exc
            if minutes <= 0:
                raise ValueError(get_text("err_time_range"))
            return start, _format_seconds_as_time(start_sec + minutes * 60.0)

        if selected == "until_time":
            end = self.until_time_var.get().strip()
            try:
                end_sec = logic.parse_time_to_seconds(end)
            except ValueError as exc:
                raise ValueError(get_text("err_time_range")) from exc
            if end_sec is None or end_sec <= start_sec:
                raise ValueError(get_text("err_time_order"))
            return start, end

        if selected is None:
            if start_sec <= 0:
                return "", ""
            return start, ""

        return start, _format_seconds_as_time(start_sec + float(selected))

    def _validate_form(self):
        input_path = self.input_var.get().strip()
        if not input_path or not os.path.exists(input_path):
            raise ValueError(get_text("err_input"))
        start, end = self._read_time_range()
        start_sec = logic.parse_time_to_seconds(start)
        end_sec = logic.parse_time_to_seconds(end)
        if start_sec is not None and end_sec is not None and start_sec >= end_sec:
            raise ValueError(get_text("err_time_order"))
        return (
            input_path,
            self.output_var.get().strip() or os.path.dirname(input_path),
            start,
            end,
            self._read_eye_distance(),
            self._read_hole_fill_mode(),
            self._read_stabilize_mode(),
        )

    def run_conversion(self):
        try:
            input_path, output_dir, start, end, eye_distance, hole_fill_mode, stabilize_mode = self._validate_form()
        except Exception as exc:
            messagebox.showerror(get_text("title_error"), str(exc))
            return

        self.stop_requested = False
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.log(get_text("msg_start"))

        def _on_proc(proc):
            self.proc = proc
            if self.stop_requested:
                try:
                    proc.kill()
                except Exception:
                    pass

        def task():
            started = time.time()
            try:
                output = logic.convert_2d_to_vr(
                    input_path=input_path,
                    output_dir=output_dir,
                    start_time=start,
                    end_time=end,
                    projection=self.projection_var.get(),
                    eye_distance_mm=eye_distance,
                    hole_fill_mode=hole_fill_mode,
                    stabilize_mode=stabilize_mode,
                    log_callback=self.log,
                    process_callback=_on_proc,
                )
                if not self.stop_requested:
                    self.log(get_text("msg_done").format(output))
                    self.root.after(0, lambda: messagebox.showinfo(get_text("title_success"), get_text("msg_success")))
            except logic.OperationCancelled:
                self.log(get_text("msg_stop"))
            except Exception as exc:
                err = str(exc)
                self.log(get_text("msg_error").format(err))
                if not self.stop_requested:
                    self.root.after(0, lambda msg=err: messagebox.showerror(get_text("title_error"), msg))
            finally:
                elapsed = time.time() - started
                self.log(get_text("msg_elapsed").format(elapsed))
                self.proc = None
                self.root.after(0, lambda: self.btn_start.config(state="normal"))
                self.root.after(0, lambda: self.btn_stop.config(state="disabled"))

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
        self.log(get_text("msg_stop"))


if __name__ == "__main__":
    root = tk.Tk()
    app = TwoDToVRApp(root)
    root.mainloop()
