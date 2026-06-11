import tkinter as tk
from tkinter import ttk
import locale
import sys
import os
import webbrowser

try:
    from utils import app_config, i18n
except ImportError:
    _utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)
    from utils import app_config, i18n

# Ensure bundled submodules win over any same-named packages in the environment.
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# When packaged as --onefile exe (PyInstaller), sys.executable points to the real .exe
# path, NOT the temp extraction dir (_MEIPASS). Prepend the exe's directory to PATH so
# that ffmpeg.exe / lada-cli placed alongside the exe are found by shutil.which().
_exe_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__))
if _exe_dir not in os.environ.get('PATH', '').split(os.pathsep):
    os.environ['PATH'] = _exe_dir + os.pathsep + os.environ.get('PATH', '')

# Python 3.8+ on Windows requires os.add_dll_directory to load DLLs (e.g., ctranslate2 loading cuBLAS)
if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(_exe_dir)
    except Exception:
        pass

import subprocess
from tkinter import filedialog
from tkinter import messagebox

ver_name = "v1.0 beta.3 patch.3 (build 2026-06-11)"
DLNA_SERVER_EXE_NAME = "vr_dlna_server.exe"


def build_hidden_startupinfo():
    if sys.platform != "win32":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE
    return startupinfo

def get_text(key):
    text = i18n.translate('main', key)
    if key == 'title':
        return text.format(version=ver_name)
    return text


def get_home_title():
    return i18n.translate('main', 'title').replace('{version}', '').strip()


def get_release_readme_filename():
    language = app_config.get_language()
    if language == 'zh':
        return "说明.txt"
    if language == 'ja':
        return "readme_ja.txt"
    return "readme.txt"


def get_runtime_work_dir():
    return _exe_dir if getattr(sys, 'frozen', False) else _app_dir


def get_dlna_server_exe_path():
    return os.path.join(_exe_dir, DLNA_SERVER_EXE_NAME)


def is_packaged_mode():
    return bool(getattr(sys, 'frozen', False))


def terminate_process_tree(process, timeout=2):
    if process is None:
        return
    try:
        if process.poll() is not None:
            return
    except Exception:
        pass

    try:
        process.terminate()
        process.wait(timeout=timeout)
        return
    except Exception:
        pass

    pid = getattr(process, "pid", None)
    if sys.platform == "win32" and pid:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
                startupinfo=build_hidden_startupinfo(),
            )
            process.wait(timeout=timeout)
            return
        except Exception:
            pass

    try:
        process.kill()
        process.wait(timeout=timeout)
    except Exception:
        pass


def terminate_packaged_dlna_server_instances():
    if sys.platform != "win32" or not is_packaged_mode():
        return False
    try:
        result = subprocess.run(
            ["taskkill", "/IM", DLNA_SERVER_EXE_NAME, "/T", "/F"],
            capture_output=True,
            timeout=5,
            startupinfo=build_hidden_startupinfo(),
        )
        return result.returncode in (0, 128)
    except Exception:
        return False


def _iter_widget_tree(widget):
    if widget is None:
        return
    try:
        children = widget.winfo_children()
    except Exception:
        return
    for child in children:
        yield child
        yield from _iter_widget_tree(child)


def _widget_is_enabled(widget):
    try:
        if "disabled" in widget.state():
            return False
    except Exception:
        pass
    try:
        return str(widget.cget("state")) != "disabled"
    except Exception:
        return True


def _is_stop_button(widget):
    if not isinstance(widget, (ttk.Button, tk.Button)):
        return False
    try:
        text = str(widget.cget("text")).strip().lower()
    except Exception:
        return False
    return "stop" in text or "停止" in text


def _enabled_stop_buttons(app):
    root = getattr(app, "root", None)
    return [
        widget
        for widget in _iter_widget_tree(root)
        if _is_stop_button(widget) and _widget_is_enabled(widget)
    ]


def _process_handle_is_running(handle):
    if handle is None:
        return False
    poll = getattr(handle, "poll", None)
    if callable(poll):
        try:
            return poll() is None
        except Exception:
            return True
    if any(callable(getattr(handle, name, None)) for name in ("kill", "terminate", "cancel")):
        return not bool(getattr(handle, "cancelled", False))
    return False


def _iter_process_handles(app):
    for name, value in vars(app).items():
        if value is None:
            continue
        if "proc" not in name and "process" not in name:
            continue
        if any(callable(getattr(value, method, None)) for method in ("kill", "terminate", "cancel")):
            yield value


def _app_has_running_tasks(app):
    if app is None:
        return False
    custom = getattr(app, "has_running_tasks", None)
    if callable(custom):
        try:
            return bool(custom())
        except Exception:
            pass
    if _enabled_stop_buttons(app):
        return True
    return any(_process_handle_is_running(handle) for handle in _iter_process_handles(app))


def _stop_process_handle(handle):
    for method in ("kill", "terminate", "cancel"):
        callback = getattr(handle, method, None)
        if callable(callback):
            try:
                callback()
                return True
            except Exception:
                pass
    return False


def _stop_app_running_tasks(app):
    if app is None:
        return False
    custom = getattr(app, "stop_running_tasks", None)
    if callable(custom):
        try:
            return bool(custom())
        except Exception:
            pass

    stopped = False
    for button in _enabled_stop_buttons(app):
        if not _widget_is_enabled(button):
            continue
        try:
            button.invoke()
            stopped = True
        except Exception:
            pass

    for handle in list(_iter_process_handles(app)):
        if _process_handle_is_running(handle):
            stopped = _stop_process_handle(handle) or stopped
    return stopped


class VRVideoToolboxLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title(get_text('title'))
        self.root.geometry("840x620")
        self.app = None
        self.dlna_process = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.create_widgets()
        self.check_dlna_status()

    def create_widgets(self):
        self.root.title(get_text('title'))
        self.main_frame = ttk.Frame(self.root, padding=(12, 8, 12, 4))
        self.main_frame.pack(side='top', fill='x')
        
        ttk.Label(self.main_frame, text=get_home_title(), font=('Arial', 15, 'bold')).pack(pady=(0, 4))
        
        # Buttons
        btn_style = ttk.Style()
        btn_style.configure('Big.TButton', font=('Arial', 11), padding=(6, 6))
        
        mosaic_frame = self.create_group(get_text('grp_mosaic'))
        
        ttk.Button(mosaic_frame, text=get_text('btn_one_click'), style='Big.TButton', command=self.launch_one_click).grid(row=1, column=0, columnspan=2, sticky='ew', padx=4, pady=2)
        
        ttk.Button(mosaic_frame, text=get_text('btn_area_sel_rect'), style='Big.TButton', command=self.launch_area_selection_rect_crop).grid(row=2, column=0, sticky='ew', padx=4, pady=2)
        ttk.Button(mosaic_frame, text=get_text('btn_area_sel'), style='Big.TButton', command=self.launch_area_selection).grid(row=2, column=1, sticky='ew', padx=4, pady=2)

        # Engine selection row.
        engine_row = ttk.Frame(mosaic_frame)
        engine_row.grid(row=3, column=0, columnspan=2, sticky='w', padx=6, pady=(3, 1))
        ttk.Label(engine_row, text=get_text('lbl_engine')).pack(side='left')
        self._engine_var = tk.StringVar(value=app_config.get_engine())
        ttk.Radiobutton(
            engine_row,
            text=get_text('engine_jasna'),
            variable=self._engine_var, value='jasna',
            command=self._on_engine_change,
        ).pack(side='left', padx=(6, 4))
        ttk.Radiobutton(
            engine_row,
            text=get_text('engine_lada'),
            variable=self._engine_var, value='lada',
            command=self._on_engine_change,
        ).pack(side='left', padx=(0, 4))
        _native_label = {'zh': '内置(GPU)', 'ja': '内蔵(GPU)', 'en': 'Built-in(GPU)'}.get(
            app_config.get_language(), '内置(GPU)')
        ttk.Radiobutton(
            engine_row,
            text=_native_label,
            variable=self._engine_var, value='native_gpu',
            command=self._on_engine_change,
        ).pack(side='left', padx=(0, 4))
        # Custom-arguments button and display label.
        self._btn_custom_args = ttk.Button(
            engine_row,
            text=get_text('btn_custom_args'),
            command=self._on_custom_args_click
        )
        self._btn_custom_args.pack(side='left', padx=(4, 4))
        
        self._lbl_custom_args = ttk.Label(engine_row, text="", foreground="gray")
        self._lbl_custom_args.pack(side='left', padx=(0, 4))
        
        self._update_custom_args_display()

        help_row_frame = ttk.Frame(mosaic_frame)
        help_row_frame.grid(row=4, column=0, columnspan=2, sticky='w', padx=6, pady=(1, 0))
        
        ttk.Label(help_row_frame, text=get_text('mode_help_prefix')).pack(side='left')
        link1 = ttk.Label(help_row_frame, text=get_text('mode_help_link'), style="Link.TLabel", cursor="hand2")
        link1.pack(side='left')
        link1.bind("<Button-1>", lambda _e: self.open_release_readme())
        
        ttk.Label(help_row_frame, text="    |    ", foreground="gray").pack(side='left')
        
        ttk.Label(help_row_frame, text=get_text('mosaic_help_prefix')).pack(side='left')
        link2 = ttk.Label(help_row_frame, text=get_text('mosaic_help_link'), style="Link.TLabel", cursor="hand2")
        link2.pack(side='left')
        link2.bind("<Button-1>", lambda _e: self.launch_tools_zoom())
        mosaic_frame.columnconfigure(0, weight=1)
        mosaic_frame.columnconfigure(1, weight=1)
        
        dlna_frame = self.create_group(get_text('grp_dlna'))
        self.btn_dlna_toggle = ttk.Button(
            dlna_frame, 
            text=self.get_dlna_btn_text(), 
            style='Big.TButton', 
            command=self.toggle_dlna_server
        )
        self.btn_dlna_toggle.grid(row=1, column=0, columnspan=2, sticky='ew', padx=4, pady=2)
        
        ttk.Button(
            dlna_frame, 
            text=get_text('btn_dlna_config'), 
            style='Big.TButton', 
            command=self.show_dlna_config_dialog
        ).grid(row=2, column=0, sticky='ew', padx=4, pady=2)
        
        link_frame = ttk.Frame(dlna_frame)
        link_frame.grid(row=2, column=1, sticky='w', padx=10, pady=2)
        link_lbl = ttk.Label(
            link_frame, 
            text=get_text('link_nvidia_server'), 
            style="Link.TLabel", 
            cursor="hand2"
        )
        link_lbl.pack(anchor='w')
        link_lbl.bind("<Button-1>", lambda e: webbrowser.open("https://wapok.com"))
        
        dlna_frame.columnconfigure(0, weight=1)
        dlna_frame.columnconfigure(1, weight=1)

        subtitle_frame = self.create_group(get_text('grp_subtitle'))
        ttk.Button(subtitle_frame, text=get_text('btn_subtitle_tools'), style='Big.TButton', command=self.launch_subtitle_tools).grid(row=1, column=0, sticky='ew', padx=4, pady=2)
        ttk.Button(subtitle_frame, text=get_text('btn_subembed_tools'), style='Big.TButton', command=self.launch_subembed_tools).grid(row=1, column=1, sticky='ew', padx=4, pady=2)
        ttk.Button(subtitle_frame, text=get_text('btn_si_voice'), style='Big.TButton', command=self.launch_si_voice).grid(row=2, column=0, columnspan=2, sticky='ew', padx=4, pady=2)
        subtitle_frame.columnconfigure(0, weight=1)
        subtitle_frame.columnconfigure(1, weight=1)
        
        tools_frame = self.create_group(get_text('grp_other'))
        
        ttk.Button(tools_frame, text=get_text('btn_vr2flat'), style='Big.TButton', command=self.launch_vr2flat).grid(row=1, column=0, sticky='ew', padx=4, pady=2)
        ttk.Button(tools_frame, text=get_text('btn_split_combine'), style='Big.TButton', command=self.launch_split_combine).grid(row=1, column=1, sticky='ew', padx=4, pady=2)
        ttk.Button(tools_frame, text=get_text('btn_v360_trans'), style='Big.TButton', command=self.launch_v360_trans).grid(row=2, column=0, sticky='ew', padx=4, pady=2)
        ttk.Button(tools_frame, text=get_text('btn_tools'), style='Big.TButton', command=self.launch_tools).grid(row=2, column=1, sticky='ew', padx=4, pady=2)
        ttk.Button(tools_frame, text=get_text('btn_2dvr'), style='Big.TButton', command=self.launch_2dvr).grid(row=3, column=0, columnspan=2, sticky='ew', padx=4, pady=2)
        
        tools_frame.columnconfigure(0, weight=1)
        tools_frame.columnconfigure(1, weight=1)
        
        btn_style.configure("Link.TLabel", foreground="blue", font=('Arial', 10, 'underline'))
        self.create_statusbar()

    def create_statusbar(self):
        statusbar = ttk.Frame(self.root, padding=(8, 2))
        statusbar.pack(side='bottom', fill='x')
        statusbar.columnconfigure(0, weight=0)
        statusbar.columnconfigure(1, weight=1)
        statusbar.columnconfigure(2, weight=0)

        language_frame = ttk.Frame(statusbar)
        language_frame.grid(row=0, column=0, sticky='w')
        ttk.Label(language_frame, text=get_text('lbl_language')).pack(side='left')
        current_display = i18n.language_code_to_display().get(app_config.get_language(), 'English')
        self._language_var = tk.StringVar(value=current_display)
        language_combo = ttk.Combobox(
            language_frame,
            textvariable=self._language_var,
            values=list(i18n.language_display_to_code().keys()),
            width=12,
            state='readonly',
        )
        language_combo.pack(side='left', padx=(4, 0))
        language_combo.bind('<<ComboboxSelected>>', self._on_language_change)

        link_lbl = ttk.Label(statusbar, text=get_text('link_opensource'), style="Link.TLabel", cursor="hand2")
        link_lbl.grid(row=0, column=1, sticky='')
        link_lbl.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/zerochocobo/VR-Video-Toolbox-CE"))

        ttk.Label(statusbar, text=ver_name).grid(row=0, column=2, sticky='e')

    def _on_language_change(self, _event=None):
        selected = self._language_var.get()
        language = i18n.language_display_to_code().get(selected, 'en')
        if language == app_config.get_language():
            return
        app_config.set_language(language)
        self.show_launcher()

    def _on_engine_change(self):
        """Persist engine radio changes immediately and update the custom-arguments display."""
        engine = self._engine_var.get()
        app_config.set_engine(engine)
        self._update_custom_args_display()

    def _update_custom_args_display(self):
        engine = self._engine_var.get()
        args = app_config.get_custom_args(engine)
        if args:
            display_text = args if len(args) <= 30 else args[:27] + "..."
            self._lbl_custom_args.config(text=f"[{display_text}]")
        else:
            self._lbl_custom_args.config(text="")

    def _on_custom_args_click(self):
        engine = self._engine_var.get()
        current_args = app_config.get_custom_args(engine)
        
        dialog = tk.Toplevel(self.root)
        dialog.title(get_text('title_custom_args'))
        dialog.geometry("450x120")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=get_text('prompt_custom_args')).pack(padx=10, pady=(10, 5), anchor='w')
        
        entry_var = tk.StringVar(value=current_args)
        entry = ttk.Entry(dialog, textvariable=entry_var, width=60)
        entry.pack(padx=10, pady=5, fill='x')
        entry.select_range(0, 'end')
        entry.focus_set()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=(5, 10))
        
        def on_save():
            new_args = entry_var.get().strip()
            app_config.set_custom_args(engine, new_args)
            self._update_custom_args_display()
            dialog.destroy()
            
        ttk.Button(btn_frame, text="OK", command=on_save, width=10).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).pack(side='left', padx=5)
        dialog.bind('<Return>', lambda e: on_save())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
        self.root.wait_window(dialog)

    def create_group(self, title):
        frame = ttk.Frame(self.main_frame, padding=3)
        frame.pack(fill='x', pady=1)
        ttk.Label(frame, text=title, font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w', padx=4, pady=(0, 1))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        return frame

    def create_link_row(self, parent, row, prefix, link_text, command):
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=0, columnspan=2, sticky='w', padx=6, pady=(3, 0))
        ttk.Label(row_frame, text=prefix).pack(side='left')
        link = ttk.Label(row_frame, text=link_text, style="Link.TLabel", cursor="hand2")
        link.pack(side='left')
        link.bind("<Button-1>", lambda _e: command())

    def open_release_readme(self):
        filename = get_release_readme_filename()
        # In onefile builds, _app_dir points to PyInstaller's temp _MEI folder.
        # The release readme is expected beside the exe; source runs keep it in release_readme.
        candidate_paths = [
            os.path.join(_exe_dir, filename),
            os.path.join(_exe_dir, "release_readme", filename),
            os.path.join(_app_dir, filename),
            os.path.join(_app_dir, "release_readme", filename),
        ]
        path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
        try:
            os.startfile(path)
        except Exception:
            webbrowser.open("file:///" + os.path.abspath(path).replace("\\", "/"))

    def launch_one_click(self):
        from one_click import main as one_click_main

        self.clear_frame()
        # Initialize One-Click App
        self.app = one_click_main.VRMosaicOneClickApp(self.root, on_return=self.request_show_launcher)

    def launch_area_selection(self):
        from area_selection_vr2flat import main as area_selection_main

        self.clear_frame()
        # Initialize Area Selection App
        self.app = area_selection_main.VRMosaicApp(self.root, on_return=self.request_show_launcher)

    def launch_area_selection_rect_crop(self):
        from area_selection_rect_crop import main as area_selection_rect_crop_main

        self.clear_frame()
        # Initialize Area Selection Rect Crop App
        self.app = area_selection_rect_crop_main.VRMosaicApp(self.root, on_return=self.request_show_launcher)

    def launch_vr2flat(self):
        from tool_vr2flat import main as vr2flat_main

        self.clear_frame()
        # Initialize VR2Flat App
        self.app = vr2flat_main.VRMosaicApp(self.root, on_return=self.request_show_launcher)

    def launch_split_combine(self):
        from tool_split_combine import main as split_combine_main

        self.clear_frame()
        # Initialize Split/Combine App
        self.app = split_combine_main.VRSplitCombineApp(self.root, on_return=self.request_show_launcher)

    def launch_v360_trans(self):
        from tool_v360_trans import main as v360_trans_main

        self.clear_frame()
        # Initialize V360 Trans App
        self.app = v360_trans_main.VRTransApp(self.root, on_return=self.request_show_launcher)

    def launch_2dvr(self):
        from tool_2dvr import main as two_d_vr_main

        self.clear_frame()
        self.app = two_d_vr_main.TwoDToVRApp(self.root, on_return=self.request_show_launcher)

    def launch_tools(self):
        from tools import gui as tools_gui

        self.clear_frame()
        # Initialize Tools App
        self.app = tools_gui.VRVideoToolsApp(self.root, on_return=self.request_show_launcher)

    def launch_tools_zoom(self):
        from tools import gui as tools_gui

        self.clear_frame()
        # Initialize Tools App and select Zoom tab (index 2)
        self.app = tools_gui.VRVideoToolsApp(self.root, on_return=self.request_show_launcher)
        self.app.notebook.select(3)
        
    def launch_subtitle_tools(self):
        from tool_subtitle import gui as tool_subtitle_gui

        self.clear_frame()
        self.app = tool_subtitle_gui.SubtitleToolsApp(self.root, on_return=self.request_show_launcher)

    def launch_si_voice(self):
        from tool_si import gui as tool_si_gui

        self.clear_frame()
        self.app = tool_si_gui.SimultaneousInterpretationApp(self.root, on_return=self.request_show_launcher)

    def launch_subembed_tools(self):
        from tool_subembed import main as tool_subembed_main

        self.clear_frame()
        self.app = tool_subembed_main.VRSubtitleEmbedApp(self.root, on_return=self.request_show_launcher)

    def get_startupinfo(self):
        """Build STARTUPINFO to hide black command prompt windows on Windows."""
        return build_hidden_startupinfo()

    def get_dlna_btn_text(self):
        if self.dlna_process is None:
            return get_text('btn_dlna_start')
        return get_text('btn_dlna_stop')

    def check_dlna_status(self):
        if self.dlna_process is not None:
            if self.dlna_process.poll() is not None:
                self.dlna_process = None
                self.btn_dlna_toggle.config(text=self.get_dlna_btn_text())
        self.root.after(1000, self.check_dlna_status)

    def get_configured_dlna_dirs(self):
        raw_dirs = app_config.get('dlna_video_dirs', '') or ''
        return [d.strip() for d in str(raw_dirs).split('|') if d.strip()]

    def toggle_dlna_server(self):
        if self.dlna_process is None:
            if not self.get_configured_dlna_dirs():
                saved = self.show_dlna_config_dialog(require_dirs=True)
                if not saved or not self.get_configured_dlna_dirs():
                    return

            if getattr(sys, 'frozen', False):
                terminate_packaged_dlna_server_instances()
                server_exe = get_dlna_server_exe_path()
                if not os.path.exists(server_exe):
                    messagebox.showerror("Error", f"{DLNA_SERVER_EXE_NAME} not found in {_exe_dir}")
                    return
                cmd = [server_exe]
            else:
                main_py = os.path.join(_app_dir, 'tool_dlna', 'main.py')
                cmd = [sys.executable, main_py]

            try:
                self.dlna_process = subprocess.Popen(
                    cmd,
                    cwd=get_runtime_work_dir(),
                    startupinfo=self.get_startupinfo()
                )
                self.btn_dlna_toggle.config(text=self.get_dlna_btn_text())
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start DLNA Server: {e}")
        else:
            if is_packaged_mode():
                terminate_packaged_dlna_server_instances()
            else:
                terminate_process_tree(self.dlna_process, timeout=2)
            self.dlna_process = None
            self.btn_dlna_toggle.config(text=self.get_dlna_btn_text())

    def on_close(self):
        """Ensure the independent DLNA process is fully terminated upon exiting the launcher."""
        if not self._confirm_stop_current_app_tasks():
            return
        if is_packaged_mode():
            terminate_packaged_dlna_server_instances()
        elif self.dlna_process is not None:
            terminate_process_tree(self.dlna_process, timeout=2)
            self.dlna_process = None
        self.dlna_process = None
        self.root.destroy()

    def show_dlna_config_dialog(self, require_dirs=False):
        """Show dynamic dialog for server config with path scanner."""
        current_name = app_config.get('dlna_server_name', 'VR Video Server')
        current_port = app_config.get('dlna_port', 8090)
        current_auto_sub = app_config.get('dlna_auto_subnotes', True) if app_config.get('dlna_auto_subnotes') is not None else app_config.get('dlna_auto_subtitles', True)
        current_dirs_str = app_config.get('dlna_video_dirs', '') or ''
        current_dirs = [d.strip() for d in str(current_dirs_str).split('|') if d.strip()]

        dialog = tk.Toplevel(self.root)
        dialog.title(get_text('dlna_config_title'))
        dialog.geometry("640x550")
        dialog.minsize(560, 490)
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)
        dialog.rowconfigure(5, weight=1)
        saved = {'ok': False}

        # Name
        ttk.Label(dialog, text=get_text('lbl_dlna_name')).grid(row=0, column=0, sticky='w', padx=(18, 8), pady=(18, 6))
        name_var = tk.StringVar(value=current_name)
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=32)
        name_entry.grid(row=0, column=1, sticky='ew', padx=(8, 18), pady=(18, 6))

        # Port
        ttk.Label(dialog, text=get_text('lbl_dlna_port')).grid(row=1, column=0, sticky='w', padx=(18, 8), pady=6)
        port_var = tk.IntVar(value=current_port)
        port_entry = ttk.Entry(dialog, textvariable=port_var, width=32)
        port_entry.grid(row=1, column=1, sticky='ew', padx=(8, 18), pady=6)
        ttk.Label(
            dialog,
            text=get_text('lbl_dlna_port_note'),
            foreground='gray',
            wraplength=560,
        ).grid(row=2, column=0, columnspan=2, sticky='w', padx=18, pady=(0, 6))

        # Subs
        auto_sub_var = tk.BooleanVar(value=current_auto_sub)
        ttk.Checkbutton(dialog, text=get_text('lbl_dlna_auto_sub'), variable=auto_sub_var).grid(row=3, column=0, columnspan=2, sticky='w', padx=18, pady=6)

        # Path List
        ttk.Label(dialog, text=get_text('lbl_dlna_dirs')).grid(row=4, column=0, columnspan=2, sticky='w', padx=18, pady=(12, 4))

        list_frame = ttk.Frame(dialog)
        list_frame.grid(row=5, column=0, columnspan=2, sticky='nsew', padx=18, pady=4)
        
        listbox = tk.Listbox(list_frame, height=10, font=('Arial', 10))
        listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        scrollbar.pack(side='right', fill='y')
        listbox.config(yscrollcommand=scrollbar.set)
        
        for d in current_dirs:
            listbox.insert('end', d)

        # Add/Del
        btn_dirs_frame = ttk.Frame(dialog)
        btn_dirs_frame.grid(row=6, column=0, columnspan=2, sticky='e', padx=18, pady=(8, 6))
        
        def add_directory():
            chosen = filedialog.askdirectory(parent=dialog)
            if chosen:
                chosen = os.path.normpath(chosen)
                existing = listbox.get(0, 'end')
                if chosen not in existing:
                    listbox.insert('end', chosen)
                    
        def delete_directory():
            selected = listbox.curselection()
            if selected:
                listbox.delete(selected[0])

        ttk.Button(btn_dirs_frame, text=get_text('btn_add_dir'), command=add_directory, width=10).pack(side='left', padx=5)
        ttk.Button(btn_dirs_frame, text=get_text('btn_del_dir'), command=delete_directory, width=10).pack(side='left', padx=5)

        # Bottom Buttons
        bottom_btn_frame = ttk.Frame(dialog)
        bottom_btn_frame.grid(row=7, column=0, columnspan=2, sticky='e', padx=18, pady=(10, 18))

        def save_config():
            name = name_var.get().strip()
            try:
                port = int(port_var.get())
            except ValueError:
                port = 8090

            auto_sub = auto_sub_var.get()
            dirs = [str(d).strip() for d in listbox.get(0, 'end') if str(d).strip()]
            if require_dirs and not dirs:
                messagebox.showwarning(
                    get_text('msg_dlna_dirs_required_title'),
                    get_text('msg_dlna_dirs_required'),
                    parent=dialog,
                )
                return
            dirs_str = "|".join(dirs)

            app_config.set('dlna_server_name', name if name else 'VR Video Server')
            app_config.set('dlna_port', port)
            app_config.set('dlna_auto_subtitles', auto_sub)
            app_config.set('dlna_auto_subnotes', auto_sub)  # Dual save for compat
            # DLNA SI live-mix feature is currently disabled (UI hidden). Force the
            # flag off on every save so any previously-enabled config gets cleared.
            app_config.set('dlna_si_enabled', False)
            app_config.set('dlna_video_dirs', dirs_str)

            saved['ok'] = True
            dialog.destroy()

            if self.dlna_process is not None:
                messagebox.showinfo(
                    "Config Saved",
                    "DLNA Server configurations saved. Please restart the DLNA Server to apply changes.\n\nDLNA服务器配置已保存，重启服务后生效。"
                )

        ttk.Button(bottom_btn_frame, text=get_text('btn_save'), command=save_config, width=12).pack(side='left', padx=(0, 10))
        ttk.Button(bottom_btn_frame, text=get_text('btn_cancel'), command=dialog.destroy, width=12).pack(side='left')

        dialog.bind('<Escape>', lambda e: dialog.destroy())
        self.root.wait_window(dialog)
        return saved['ok']
    
    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def _confirm_stop_current_app_tasks(self):
        if not _app_has_running_tasks(self.app):
            return True
        ok = messagebox.askyesno(
            get_text('confirm_running_title'),
            get_text('confirm_running_message'),
            parent=self.root,
        )
        if not ok:
            return False
        _stop_app_running_tasks(self.app)
        return True

    def request_show_launcher(self):
        if not self._confirm_stop_current_app_tasks():
            return
        self.show_launcher()

    def show_launcher(self):
        self.clear_frame()
        self.app = None
        # Clear any menu bar set by sub-apps
        empty_menu = tk.Menu(self.root)
        self.root.config(menu=empty_menu)
        self.create_widgets()

def _start_gpu_warmup():
    """Warm up the GPU engine in the background to avoid first-task cold-start JIT delay.

    Runs on a separate thread without blocking the UI. Failure does not affect
    startup because tools using backend=auto will fall back to ffmpeg.
    """
    import threading

    def _run():
        try:
            from gpu_engine import runtime
            runtime.warmup(verbose=bool(app_config.get("gpu_log_verbose", False)))
        except Exception as e:
            print(f"[main] GPU warmup skipped: {e}")

    threading.Thread(target=_run, name="gpu-warmup", daemon=True).start()


def _startup_gpu_warmup_enabled() -> bool:
    value = os.environ.get("VRTB_START_GPU_WARMUP", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _selftest_gpu() -> int:
    """GPU self-test: warmup + one CuPy kernel + PyNv probe for diagnostics and packaging validation.

    Usage: VR_Video_Toolbox.exe --selftest-gpu
    """
    try:
        from gpu_engine import runtime, pynv_io
        st = runtime.warmup(verbose=True)
        print(f"[selftest] gpu_available={st.available} reason={st.reason}")
        print(f"[selftest] {pynv_io.cuda_device_summary(0)}")
        if not st.available:
            return 1
        import cupy as cp
        k = cp.RawKernel(r'extern "C" __global__ void inc(unsigned char* d){ d[threadIdx.x]+=1; }', 'inc')
        buf = cp.zeros(8, dtype=cp.uint8)
        k((1,), (8,), (buf,))
        cp.cuda.Stream.null.synchronize()
        ok = int(buf[0]) == 1
        print(f"[selftest] RawKernel JIT {'OK' if ok else 'FAILED'}; nvenc_hevc_10bit={st.nvenc_hevc_10bit}")
        return 0 if ok else 1
    except Exception as e:
        import traceback
        print(f"[selftest] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    if "--selftest-gpu" in sys.argv:
        sys.exit(_selftest_gpu())
    if _startup_gpu_warmup_enabled():
        _start_gpu_warmup()
    root = tk.Tk()
    app = VRVideoToolboxLauncher(root)
    root.mainloop()

VRMosaicLauncher = VRVideoToolboxLauncher
