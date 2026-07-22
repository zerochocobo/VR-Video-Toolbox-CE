import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import locale
import sys
import os
import webbrowser

try:
    from utils import app_config, encode_config, i18n, ui_theme
except ImportError:
    _utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)
    from utils import app_config, encode_config, i18n, ui_theme

# Ensure bundled submodules win over any same-named packages in the environment.
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from tool_dlna import media_library as dlna_media_library

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

ver_name = "v1.6.1 (build 2026-07-22)"
DLNA_SERVER_EXE_NAME = "vr_dlna_server.exe"
TWO_DVR_DOWNLOAD_URL = "https://wapok.com"

# Kept as a source-level compatibility marker for existing launcher checks.
# Runtime navigation icons are resolved by utils.ui_theme.icon_for_title().
NAV_ICONS = {'voice': '\uE720'}

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


def split_dlna_video_dirs(raw_dirs):
    return [d.strip() for d in str(raw_dirs or '').split('|') if d.strip()]


def filter_supported_dlna_video_dirs(raw_dirs):
    return [d for d in split_dlna_video_dirs(raw_dirs) if not dlna_media_library.is_unc_path(d)]


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
        self.palette = ui_theme.apply_theme(root)
        self.root.title(get_text('title'))
        self.root.geometry("840x620")
        self.app = None
        self.dlna_process = None
        self._current_page = 'mosaic'
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.create_widgets()
        self.check_dlna_status()

    def create_widgets(self):
        self.root.title(get_text('title'))
        self.palette = ui_theme.apply_theme(self.root)

        self.create_statusbar()

        body = tk.Frame(self.root, bg=self.palette.HOME_BG)
        body.pack(side='top', fill='both', expand=True)

        self._build_sidebar(body)

        content = tk.Frame(body, bg=self.palette.HOME_BG)
        content.pack(side='left', fill='both', expand=True)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        self._pages = {}
        for key, builder in (
            ('mosaic', self._build_page_mosaic),
            ('subtitle', self._build_page_subtitle),
            ('voice', self._build_page_voice),
            ('dlna', self._build_page_dlna),
            ('other', self._build_page_other),
            ('settings', self._build_page_settings),
        ):
            page = tk.Frame(content, bg=self.palette.HOME_BG)
            page.grid(row=0, column=0, sticky='nsew')
            builder(page)
            self._pages[key] = page

        self._select_page(self._current_page)
        self._refresh_dlna_ui()

    def _build_sidebar(self, parent):
        bar = tk.Frame(parent, bg=self.palette.SIDEBAR_BG, width=190)
        bar.pack(side='left', fill='y')
        bar.pack_propagate(False)

        self._nav_icons_enabled = ui_theme.NAV_ICON_FONT in tkfont.families(self.root)

        # Break the app title before the "(CUDA...)" suffix so it wraps cleanly.
        title = get_home_title()
        if '(' in title:
            name, _, suffix = title.partition('(')
            title = name.strip() + '\n(' + suffix
        tk.Label(
            bar, text=title, bg=self.palette.SIDEBAR_BG, fg=self.palette.CARD_TITLE_FG,
            font=('Arial', 11, 'bold'), wraplength=162, justify='left', anchor='w', padx=14,
        ).pack(fill='x', pady=(14, 10))

        self._nav_items = {}
        for key, text_key in (
            ('mosaic', 'nav_mosaic'),
            ('subtitle', 'nav_subtitle'),
            ('voice', 'nav_voice'),
            ('dlna', 'nav_dlna'),
            ('other', 'nav_other'),
        ):
            self._add_nav_item(bar, key, get_text(text_key), side='top')
        self._add_nav_item(bar, 'settings', get_text('nav_settings'), side='bottom')

    def _add_nav_item(self, bar, key, text, side):
        item = tk.Frame(bar, bg=self.palette.SIDEBAR_BG, cursor='hand2')
        item.pack(side=side, fill='x', pady=(0, 1) if side == 'top' else (1, 8))

        icon = None
        if self._nav_icons_enabled:
            icon = tk.Label(
                item, text=ui_theme.icon_for_title(text), bg=self.palette.SIDEBAR_BG,
                fg=self.palette.SIDEBAR_ITEM_FG, font=(ui_theme.NAV_ICON_FONT, 12), anchor='w',
            )
            icon.pack(side='left', padx=(16, 8), pady=8)
        lbl = tk.Label(
            item, text=text, bg=self.palette.SIDEBAR_BG, fg=self.palette.SIDEBAR_ITEM_FG,
            font=('Arial', 11), anchor='w', padx=0 if icon else 16, pady=8,
        )
        lbl.pack(side='left', fill='x', expand=True)

        widgets = [w for w in (item, icon, lbl) if w is not None]

        def on_enter(_e, k=key):
            if self._current_page != k:
                for w in widgets:
                    w.config(bg=self.palette.SIDEBAR_HOVER_BG)

        def on_leave(_e, k=key):
            if self._current_page != k:
                for w in widgets:
                    w.config(bg=self.palette.SIDEBAR_BG)

        for w in widgets:
            w.bind('<Button-1>', lambda _e, k=key: self._select_page(k))
            w.bind('<Enter>', on_enter)
            w.bind('<Leave>', on_leave)
        self._nav_items[key] = (item, icon, lbl)

    def _select_page(self, key):
        self._current_page = key
        for k, (item, icon, lbl) in self._nav_items.items():
            selected = k == key
            bg = self.palette.SIDEBAR_SEL_BG if selected else self.palette.SIDEBAR_BG
            fg = self.palette.SIDEBAR_SEL_FG if selected else self.palette.SIDEBAR_ITEM_FG
            item.config(bg=bg)
            if icon is not None:
                icon.config(bg=bg, fg=fg)
            lbl.config(bg=bg, fg=fg, font=('Arial', 11, 'bold') if selected else ('Arial', 11))
        self._pages[key].tkraise()

    def _make_card(self, parent, title, desc, command, primary=False, wrap=560):
        bg = self.palette.PRIMARY_BG if primary else self.palette.CARD_BG
        hover_bg = self.palette.PRIMARY_HOVER_BG if primary else self.palette.CARD_HOVER_BG
        border = self.palette.PRIMARY_BORDER if primary else self.palette.CARD_BORDER
        title_fg = self.palette.PRIMARY_TITLE_FG if primary else self.palette.CARD_TITLE_FG
        desc_fg = self.palette.PRIMARY_DESC_FG if primary else self.palette.CARD_DESC_FG

        card = tk.Frame(
            parent, bg=bg, cursor='hand2',
            highlightbackground=border, highlightthickness=2 if primary else 1,
        )
        widgets = [card]
        title_lbl = tk.Label(
            card, text=title, bg=bg, fg=title_fg, anchor='w', justify='left',
            font=('Arial', 12, 'bold'), wraplength=wrap,
        )
        title_lbl.pack(fill='x', padx=12, pady=(8, 0) if desc else (8, 8))
        widgets.append(title_lbl)
        if desc:
            desc_lbl = tk.Label(
                card, text=desc, bg=bg, fg=desc_fg, anchor='w', justify='left',
                font=('Arial', 9), wraplength=wrap,
            )
            desc_lbl.pack(fill='x', padx=12, pady=(1, 8))
            widgets.append(desc_lbl)

        def set_bg(color):
            for w in widgets:
                w.config(bg=color)

        for w in widgets:
            w.bind('<Button-1>', lambda _e: command())
            w.bind('<Enter>', lambda _e: set_bg(hover_bg))
            w.bind('<Leave>', lambda _e: set_bg(bg))
        return card

    def _make_page_inner(self, page):
        inner = tk.Frame(page, bg=self.palette.HOME_BG)
        inner.pack(fill='both', expand=True, padx=16, pady=14)
        return inner

    def _build_page_mosaic(self, page):
        inner = self._make_page_inner(page)

        self._make_card(
            inner, get_text('btn_one_click'), get_text('desc_one_click'),
            self.launch_one_click, primary=True,
        ).pack(fill='x')

        self._make_card(
            inner, get_text('btn_area_sel_rect'), get_text('desc_area_sel_rect'),
            self.launch_area_selection_rect_crop,
        ).pack(fill='x', pady=(10, 0))
        self._make_card(
            inner, get_text('btn_area_sel'), get_text('desc_area_sel'),
            self.launch_area_selection,
        ).pack(fill='x', pady=(8, 0))

        ttk.Separator(inner, orient='horizontal').pack(fill='x', pady=12)

        global_group = ttk.LabelFrame(
            inner,
            text=get_text('grp_global_mosaic_settings'),
            padding=(10, 8),
        )
        global_group.pack(fill='x')

        quality_row = ttk.Frame(global_group)
        quality_row.pack(fill='x')
        ttk.Label(quality_row, text=get_text('lbl_encode_profile')).pack(side='left')
        profile_labels = {
            'highest_quality': get_text('opt_encode_highest_quality'),
            'balanced_high_quality': get_text('opt_encode_balanced_high_quality'),
            'fast_quality': get_text('opt_encode_fast_quality'),
            'ultra_fast_normal': get_text('opt_encode_ultra_fast_normal'),
        }
        profile_options = [(profile_labels[key], key) for key in encode_config.get_profile_keys()]
        current_profile = encode_config.current_encode_profile_key()
        if current_profile is None:
            profile_options.append((get_text('opt_encode_custom'), 'custom'))
            current_profile = 'custom'
        self._encode_profile_display_to_key = {display: key for display, key in profile_options}
        self._encode_profile_key_to_display = {key: display for display, key in profile_options}
        self._encode_profile_var = tk.StringVar(
            value=self._encode_profile_key_to_display[current_profile]
        )
        profile_combo = ttk.Combobox(
            quality_row,
            textvariable=self._encode_profile_var,
            values=[display for display, _key in profile_options],
            width=24,
            state='readonly',
        )
        profile_combo.pack(side='left', padx=(6, 0))
        profile_combo.bind('<<ComboboxSelected>>', self._on_encode_profile_change)
        ttk.Label(
            global_group,
            text=get_text('hint_encode_profile'),
            foreground=self.palette.CARD_DESC_FG,
            wraplength=590,
            justify='left',
        ).pack(fill='x', pady=(4, 7))

        ttk.Separator(global_group, orient='horizontal').pack(fill='x', pady=(0, 7))

        engine_row = ttk.Frame(global_group)
        engine_row.pack(fill='x')
        ttk.Label(engine_row, text=get_text('lbl_engine')).pack(side='left')
        engine_display = {
            'jasna': get_text('engine_jasna'),
            'lada': get_text('engine_lada'),
            'native_gpu': get_text('engine_native'),
        }
        current_engine = app_config.get_engine()
        if current_engine not in engine_display:
            current_engine = 'native_gpu'
        self._engine_var = tk.StringVar(value=current_engine)
        for engine_key, display in engine_display.items():
            ttk.Radiobutton(
                engine_row,
                text=display,
                variable=self._engine_var,
                value=engine_key,
                command=self._on_engine_change,
            ).pack(side='left', padx=(8, 0))

        self._custom_args_frame = ttk.Frame(engine_row)
        self._custom_args_frame.pack(side='left', padx=(14, 0))
        self._btn_custom_args = ttk.Button(
            self._custom_args_frame,
            text=get_text('btn_custom_args'),
            command=self._on_custom_args_click
        )
        self._btn_custom_args.pack(side='left', padx=(0, 4))

        self._lbl_custom_args = ttk.Label(
            self._custom_args_frame,
            text="",
            foreground=self.palette.CARD_DESC_FG,
        )
        self._lbl_custom_args.pack(side='left')

        self._update_custom_args_display()

        # Two help hints on separate rows so long translations don't overflow.
        help_row1 = tk.Frame(inner, bg=self.palette.HOME_BG)
        help_row1.pack(fill='x', pady=(10, 0))
        tk.Label(help_row1, text=get_text('mode_help_prefix'), bg=self.palette.HOME_BG).pack(side='left')
        link1 = ttk.Label(help_row1, text=get_text('mode_help_link'), style="Link.TLabel", cursor="hand2")
        link1.pack(side='left')
        link1.bind("<Button-1>", lambda _e: self.open_release_readme())

        help_row2 = tk.Frame(inner, bg=self.palette.HOME_BG)
        help_row2.pack(fill='x', pady=(3, 0))
        tk.Label(help_row2, text=get_text('mosaic_help_prefix'), bg=self.palette.HOME_BG).pack(side='left')
        link2 = ttk.Label(help_row2, text=get_text('mosaic_help_link'), style="Link.TLabel", cursor="hand2")
        link2.pack(side='left')
        link2.bind("<Button-1>", lambda _e: self.launch_tools_zoom())

    def _build_page_subtitle(self, page):
        inner = self._make_page_inner(page)
        cards = (
            (get_text('btn_subtitle_tools'), get_text('desc_subtitle_tools'), self.launch_subtitle_tools),
            (get_text('btn_subembed_tools'), get_text('desc_subembed_tools'), self.launch_subembed_tools),
            (get_text('btn_subtitle_analyzer'), get_text('desc_subtitle_analyzer'), self.launch_subtitle_analyzer),
        )
        for title, desc, command in cards:
            self._make_card(inner, title, desc, command).pack(fill='x', pady=(0, 8))

    def _build_page_voice(self, page):
        inner = self._make_page_inner(page)
        cards = (
            (get_text('btn_clonevoice'), get_text('desc_clonevoice'), self.launch_clonevoice),
            (get_text('btn_si_voice'), get_text('desc_si_voice'), self.launch_si_voice),
        )
        for title, desc, command in cards:
            self._make_card(inner, title, desc, command).pack(fill='x', pady=(0, 8))

    def _build_page_dlna(self, page):
        inner = self._make_page_inner(page)

        card = tk.Frame(inner, bg=self.palette.CARD_BG, highlightbackground=self.palette.CARD_BORDER, highlightthickness=1)
        card.pack(fill='x')
        self._dlna_dot = tk.Label(card, text='●', bg=self.palette.CARD_BG, fg=self.palette.DOT_STOPPED, font=('Arial', 12))
        self._dlna_dot.pack(side='left', padx=(12, 6), pady=12)
        self.btn_dlna_toggle = ttk.Button(
            card,
            text=get_text('btn_dlna_start_short'),
            style='Big.TButton',
            command=self.toggle_dlna_server
        )
        self.btn_dlna_toggle.pack(side='right', padx=12, pady=10)
        text_frame = tk.Frame(card, bg=self.palette.CARD_BG)
        text_frame.pack(side='left', fill='x', expand=True, pady=10)
        self._dlna_status_lbl = tk.Label(
            text_frame, text='', bg=self.palette.CARD_BG, fg=self.palette.CARD_TITLE_FG,
            font=('Arial', 11, 'bold'), anchor='w',
        )
        self._dlna_status_lbl.pack(fill='x')
        self._dlna_summary_lbl = tk.Label(
            text_frame, text='', bg=self.palette.CARD_BG, fg=self.palette.CARD_DESC_FG,
            font=('Arial', 9), anchor='w',
        )
        self._dlna_summary_lbl.pack(fill='x')

        links_row = tk.Frame(inner, bg=self.palette.HOME_BG)
        links_row.pack(fill='x', pady=(12, 0))
        cfg_link = ttk.Label(links_row, text=get_text('btn_dlna_config'), style="Link.TLabel", cursor="hand2")
        cfg_link.pack(side='left')
        cfg_link.bind("<Button-1>", lambda _e: self.show_dlna_config_dialog())

        tk.Label(links_row, text="    |    ", fg=self.palette.CARD_DESC_FG, bg=self.palette.HOME_BG).pack(side='left')

        nv_link = ttk.Label(links_row, text=get_text('link_nvidia_server'), style="Link.TLabel", cursor="hand2")
        nv_link.pack(side='left')
        nv_link.bind("<Button-1>", lambda _e: webbrowser.open(TWO_DVR_DOWNLOAD_URL))

    def _build_page_other(self, page):
        inner = self._make_page_inner(page)
        items = (
            (get_text('btn_vr2flat'), get_text('desc_vr2flat'), self.launch_vr2flat),
            (get_text('btn_split_combine'), get_text('desc_split_combine'), self.launch_split_combine),
            (get_text('btn_v360_trans'), get_text('desc_v360_trans'), self.launch_v360_trans),
            (get_text('btn_tools'), get_text('desc_tools'), self.launch_tools),
            (get_text('btn_2dvr'), get_text('desc_2dvr'), self.launch_2dvr),
        )
        for title, desc, command in items:
            self._make_card(inner, title, desc, command).pack(fill='x', pady=(0, 8))

    def _build_page_settings(self, page):
        inner = self._make_page_inner(page)

        language_row = tk.Frame(inner, bg=self.palette.HOME_BG)
        language_row.pack(fill='x', pady=(0, 12))
        tk.Label(language_row, text=get_text('lbl_language'), bg=self.palette.HOME_BG).pack(side='left')
        current_display = i18n.language_code_to_display().get(app_config.get_language(), 'English')
        self._language_var = tk.StringVar(value=current_display)
        language_combo = ttk.Combobox(
            language_row,
            textvariable=self._language_var,
            values=list(i18n.language_display_to_code().keys()),
            width=12,
            state='readonly',
        )
        language_combo.pack(side='left', padx=(4, 0))
        language_combo.bind('<<ComboboxSelected>>', self._on_language_change)

        theme_row = tk.Frame(inner, bg=self.palette.HOME_BG)
        theme_row.pack(fill='x', pady=(0, 12))
        tk.Label(theme_row, text=get_text('lbl_ui_theme'), bg=self.palette.HOME_BG).pack(side='left')
        theme_options = {
            get_text('opt_theme_light'): 'light',
            get_text('opt_theme_dark'): 'dark',
        }
        self._theme_by_display = theme_options
        current_theme = app_config.get_ui_theme()
        current_theme_display = next(
            (display for display, value in theme_options.items() if value == current_theme),
            get_text('opt_theme_light'),
        )
        self._theme_var = tk.StringVar(value=current_theme_display)
        theme_combo = ttk.Combobox(
            theme_row,
            textvariable=self._theme_var,
            values=list(theme_options.keys()),
            width=12,
            state='readonly',
        )
        theme_combo.pack(side='left', padx=(4, 0))
        theme_combo.bind('<<ComboboxSelected>>', self._on_theme_change)

        docs_row = tk.Frame(inner, bg=self.palette.HOME_BG)
        docs_row.pack(fill='x')
        tk.Label(docs_row, text=get_text('lbl_docs'), bg=self.palette.HOME_BG).pack(side='left')
        docs_link = ttk.Label(docs_row, text=get_text('mode_help_link'), style="Link.TLabel", cursor="hand2")
        docs_link.pack(side='left', padx=(4, 0))
        docs_link.bind("<Button-1>", lambda _e: self.open_release_readme())

    def create_statusbar(self):
        statusbar = ttk.Frame(self.root, padding=(8, 2))
        statusbar.pack(side='bottom', fill='x')
        statusbar.columnconfigure(0, weight=0)
        statusbar.columnconfigure(1, weight=1)
        statusbar.columnconfigure(2, weight=0)

        self._sb_dlna_lbl = tk.Label(statusbar, text='', fg=self.palette.DOT_STOPPED, font=('Arial', 9))
        self._sb_dlna_lbl.grid(row=0, column=0, sticky='w')

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

    def _on_theme_change(self, _event=None):
        theme = self._theme_by_display.get(self._theme_var.get(), 'light')
        if theme == app_config.get_ui_theme():
            return
        app_config.set_ui_theme(theme)
        self.show_launcher()

    def _selected_engine(self):
        engine = self._engine_var.get()
        return engine if engine in {'jasna', 'lada', 'native_gpu'} else 'native_gpu'

    def _on_engine_change(self):
        """Persist engine radio changes immediately and update the custom-arguments display."""
        app_config.set_engine(self._selected_engine())
        self._update_custom_args_display()

    def _on_encode_profile_change(self, _event=None):
        key = self._encode_profile_display_to_key.get(
            self._encode_profile_var.get(),
            encode_config.DEFAULT_ENCODE_PROFILE,
        )
        if key != 'custom':
            encode_config.apply_encode_profile(key)

    def _update_custom_args_display(self):
        engine = self._selected_engine()
        if engine == 'native_gpu':
            self._custom_args_frame.pack_forget()
            return
        if not self._custom_args_frame.winfo_manager():
            self._custom_args_frame.pack(side='left', padx=(14, 0))
        args = app_config.get_custom_args(engine)
        if args:
            display_text = args if len(args) <= 30 else args[:27] + "..."
            self._lbl_custom_args.config(text=f"[{display_text}]")
        else:
            self._lbl_custom_args.config(text="")

    def _on_custom_args_click(self):
        engine = self._selected_engine()
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
        self.show_2dvr_migration_dialog()

    def show_2dvr_migration_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title(get_text('title_2dvr_migrated'))
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill='both', expand=True)

        ttk.Label(
            frame,
            text=get_text('msg_2dvr_migrated'),
            wraplength=440,
            justify='left',
        ).pack(anchor='w', fill='x')

        link = ttk.Label(
            frame,
            text=get_text('link_2dvr_download'),
            style="Link.TLabel",
            cursor="hand2",
        )
        link.pack(anchor='w', pady=(10, 12))
        link.bind("<Button-1>", lambda _e: webbrowser.open(TWO_DVR_DOWNLOAD_URL))

        ttk.Button(frame, text=get_text('btn_close'), command=dialog.destroy, width=10).pack(anchor='e')
        dialog.bind('<Escape>', lambda _e: dialog.destroy())

        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - dialog.winfo_width()) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - dialog.winfo_height()) // 2)
        dialog.geometry(f"+{x}+{y}")

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

    def launch_subtitle_analyzer(self):
        from tool_subtitle.debug_analyzer import SubtitleDebugAnalyzer

        self.subtitle_analyzer = SubtitleDebugAnalyzer(self.root)

    def launch_si_voice(self):
        from tool_si import gui as tool_si_gui

        self.clear_frame()
        self.app = tool_si_gui.SimultaneousInterpretationApp(self.root, on_return=self.request_show_launcher)

    def launch_clonevoice(self):
        from tool_clonevoice import gui as tool_clonevoice_gui

        self.clear_frame()
        self.app = tool_clonevoice_gui.ClonevoiceToolsApp(self.root, on_return=self.request_show_launcher)

    def launch_subembed_tools(self):
        from tool_subembed import main as tool_subembed_main

        self.clear_frame()
        self.app = tool_subembed_main.VRSubtitleEmbedApp(self.root, on_return=self.request_show_launcher)

    def get_startupinfo(self):
        """Build STARTUPINFO to hide black command prompt windows on Windows."""
        return build_hidden_startupinfo()

    def _refresh_dlna_ui(self):
        """Sync every DLNA status widget (page card + statusbar) with the process state.

        Safe to call while a sub-app owns the root: destroyed widgets are skipped.
        """
        running = self.dlna_process is not None

        def alive(widget):
            try:
                return widget is not None and widget.winfo_exists()
            except tk.TclError:
                return False

        try:
            if alive(getattr(self, 'btn_dlna_toggle', None)):
                self.btn_dlna_toggle.config(
                    text=get_text('btn_dlna_stop_short' if running else 'btn_dlna_start_short'))
            if alive(getattr(self, '_dlna_dot', None)):
                self._dlna_dot.config(fg=self.palette.DOT_RUNNING if running else self.palette.DOT_STOPPED)
            if alive(getattr(self, '_dlna_status_lbl', None)):
                self._dlna_status_lbl.config(
                    text=get_text('dlna_status_running' if running else 'dlna_status_stopped'))
            if alive(getattr(self, '_dlna_summary_lbl', None)):
                self._dlna_summary_lbl.config(text=get_text('dlna_summary').format(
                    dirs=len(self.get_configured_dlna_dirs()),
                    port=app_config.get('dlna_port', 8090),
                ))
            if alive(getattr(self, '_sb_dlna_lbl', None)):
                self._sb_dlna_lbl.config(
                    text='● ' + get_text('sb_dlna_running' if running else 'sb_dlna_stopped'),
                    fg=self.palette.DOT_RUNNING if running else self.palette.DOT_STOPPED,
                )
        except tk.TclError:
            pass

    def check_dlna_status(self):
        if self.dlna_process is not None:
            if self.dlna_process.poll() is not None:
                self.dlna_process = None
                self._refresh_dlna_ui()
        self.root.after(1000, self.check_dlna_status)

    def get_configured_dlna_dirs(self):
        raw_dirs = app_config.get('dlna_video_dirs', '') or ''
        return filter_supported_dlna_video_dirs(raw_dirs)

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
                self._refresh_dlna_ui()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start DLNA Server: {e}")
        else:
            if is_packaged_mode():
                terminate_packaged_dlna_server_instances()
            else:
                terminate_process_tree(self.dlna_process, timeout=2)
            self.dlna_process = None
            self._refresh_dlna_ui()

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
        from tool_si import logic as si_logic

        current_name = app_config.get('dlna_server_name', 'VR Video Server')
        current_port = app_config.get('dlna_port', 8090)
        current_auto_sub = app_config.get('dlna_auto_subnotes', True) if app_config.get('dlna_auto_subnotes') is not None else app_config.get('dlna_auto_subtitles', True)
        current_dirs_str = app_config.get('dlna_video_dirs', '') or ''
        current_dirs = filter_supported_dlna_video_dirs(current_dirs_str)
        current_si_enabled = bool(app_config.get('dlna_si_enabled', True))

        def _saved_si_value(key, default):
            return app_config.get(key, default) if current_si_enabled else default

        def _int_choice(value, choices, default):
            try:
                value = int(value)
            except (TypeError, ValueError):
                return default
            return value if value in choices else default

        def _float_choice(value, choices, default):
            try:
                value = round(float(value), 1)
            except (TypeError, ValueError):
                return default
            return value if value in choices else default

        dialog = tk.Toplevel(self.root)
        dialog.title(get_text('dlna_config_title'))
        dialog.geometry("700x560")
        dialog.minsize(620, 500)
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)
        dialog.rowconfigure(6, weight=1)
        saved = {'ok': False}

        # Name
        ttk.Label(dialog, text=get_text('lbl_dlna_name')).grid(row=0, column=0, sticky='w', padx=(18, 8), pady=(12, 3))
        name_var = tk.StringVar(value=current_name)
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=32)
        name_entry.grid(row=0, column=1, sticky='ew', padx=(8, 18), pady=(12, 3))

        # Port
        ttk.Label(dialog, text=get_text('lbl_dlna_port')).grid(row=1, column=0, sticky='w', padx=(18, 8), pady=3)
        port_var = tk.IntVar(value=current_port)
        port_entry = ttk.Entry(dialog, textvariable=port_var, width=32)
        port_entry.grid(row=1, column=1, sticky='ew', padx=(8, 18), pady=3)
        ttk.Label(
            dialog,
            text=get_text('lbl_dlna_port_note'),
            foreground='gray',
            wraplength=560,
        ).grid(row=2, column=0, columnspan=2, sticky='w', padx=18, pady=(0, 3))

        # Subs
        auto_sub_var = tk.BooleanVar(value=current_auto_sub)
        ttk.Checkbutton(dialog, text=get_text('lbl_dlna_auto_sub'), variable=auto_sub_var).grid(row=3, column=0, columnspan=2, sticky='w', padx=18, pady=3)

        # Simultaneous interpretation audio
        si_frame = ttk.LabelFrame(dialog, text=get_text('grp_dlna_si_audio'), padding=(8, 6))
        si_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=18, pady=(2, 4))
        si_frame.columnconfigure(1, weight=1)
        si_frame.columnconfigure(3, weight=1)

        si_enabled_var = tk.BooleanVar(value=current_si_enabled)
        si_enabled_check = ttk.Checkbutton(
            si_frame,
            text=get_text('lbl_dlna_si_enabled'),
            variable=si_enabled_var,
        )
        si_enabled_check.grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 3))

        dub_mode_row = ttk.Frame(si_frame)
        dub_mode_row.grid(row=1, column=0, columnspan=4, sticky='w', pady=(0, 3))
        si_dub_mode_var = tk.BooleanVar(value=bool(_saved_si_value('dlna_si_dub_mode', True)))
        ttk.Checkbutton(
            dub_mode_row,
            text=get_text('lbl_dlna_si_dub_mode'),
            variable=si_dub_mode_var,
        ).pack(side='left')

        def show_dub_mode_help():
            messagebox.showinfo(
                get_text('lbl_dlna_si_dub_mode_help_title'),
                get_text('lbl_dlna_si_dub_mode_help'),
                parent=dialog,
            )

        ttk.Button(dub_mode_row, text='?', width=2, command=show_dub_mode_help).pack(side='left', padx=(6, 0))

        si_options_frame = ttk.Frame(si_frame)
        si_options_frame.grid(row=2, column=0, columnspan=4, sticky='ew')
        si_options_frame.columnconfigure(1, weight=1)
        si_options_frame.columnconfigure(3, weight=1)

        channel_map = {
            get_text('opt_dlna_si_channel_both'): 'both',
            get_text('opt_dlna_si_channel_left'): 'left',
            get_text('opt_dlna_si_channel_right'): 'right',
        }
        channel_label_by_value = {value: label for label, value in channel_map.items()}
        current_channel = str(_saved_si_value('dlna_si_mix_channel', 'both')).strip().lower()
        si_channel_var = tk.StringVar(value=channel_label_by_value.get(current_channel, channel_label_by_value['both']))

        current_orig_vol = _int_choice(
            _saved_si_value('dlna_si_original_volume_percent', 100),
            si_logic.ORIGINAL_VOLUME_CHOICES,
            100,
        )
        current_si_vol = _int_choice(
            _saved_si_value('dlna_si_volume_percent', 100),
            si_logic.SI_VOLUME_CHOICES,
            100,
        )
        current_si_delay = _float_choice(
            _saved_si_value('dlna_si_delay_seconds', 1.0),
            si_logic.SI_DELAY_SECONDS_CHOICES,
            1.0,
        )
        si_origvol_var = tk.StringVar(value=f"{current_orig_vol}%")
        si_volume_var = tk.StringVar(value=f"{current_si_vol}%")
        si_delay_var = tk.StringVar(value=f"{current_si_delay:g}s")
        si_duck_var = tk.BooleanVar(value=bool(_saved_si_value('dlna_si_duck_original', True)))
        duck_preset_map = {
            get_text('opt_duck_preset_light'): 'light',
            get_text('opt_duck_preset_normal'): 'normal',
            get_text('opt_duck_preset_strong'): 'strong',
        }
        duck_preset_label_by_value = {value: label for label, value in duck_preset_map.items()}
        current_duck_preset = str(_saved_si_value('dlna_si_duck_preset', 'normal')).strip().lower()
        si_duck_preset_var = tk.StringVar(
            value=duck_preset_label_by_value.get(current_duck_preset, duck_preset_label_by_value['normal'])
        )

        ttk.Label(si_options_frame, text=get_text('lbl_dlna_si_channel')).grid(row=0, column=0, sticky='w', padx=(0, 6), pady=1)
        ttk.Combobox(
            si_options_frame,
            textvariable=si_channel_var,
            values=list(channel_map),
            width=18,
            state='readonly',
        ).grid(row=0, column=1, sticky='w', pady=1)
        ttk.Label(si_options_frame, text=get_text('lbl_dlna_si_original_volume')).grid(row=0, column=2, sticky='w', padx=(14, 6), pady=1)
        ttk.Combobox(
            si_options_frame,
            textvariable=si_origvol_var,
            values=[f"{v}%" for v in si_logic.ORIGINAL_VOLUME_CHOICES],
            width=10,
            state='readonly',
        ).grid(row=0, column=3, sticky='w', pady=1)
        ttk.Label(si_options_frame, text=get_text('lbl_dlna_si_volume')).grid(row=1, column=0, sticky='w', padx=(0, 6), pady=1)
        ttk.Combobox(
            si_options_frame,
            textvariable=si_volume_var,
            values=[f"{v}%" for v in si_logic.SI_VOLUME_CHOICES],
            width=10,
            state='readonly',
        ).grid(row=1, column=1, sticky='w', pady=1)
        ttk.Label(si_options_frame, text=get_text('lbl_dlna_si_delay')).grid(row=1, column=2, sticky='w', padx=(14, 6), pady=1)
        ttk.Combobox(
            si_options_frame,
            textvariable=si_delay_var,
            values=[f"{v:g}s" for v in si_logic.SI_DELAY_SECONDS_CHOICES],
            width=10,
            state='readonly',
        ).grid(row=1, column=3, sticky='w', pady=1)
        si_duck_check = ttk.Checkbutton(
            si_options_frame,
            text=get_text('lbl_dlna_si_duck_original'),
            variable=si_duck_var,
        )
        si_duck_check.grid(row=2, column=0, columnspan=2, sticky='w', pady=(2, 0))
        ttk.Label(si_options_frame, text=get_text('lbl_duck_preset')).grid(row=2, column=2, sticky='w', padx=(14, 6), pady=(2, 0))
        si_duck_preset_combo = ttk.Combobox(
            si_options_frame,
            textvariable=si_duck_preset_var,
            values=list(duck_preset_map),
            width=10,
            state='readonly',
        )
        si_duck_preset_combo.grid(row=2, column=3, sticky='w', pady=(2, 0))

        def refresh_duck_preset_state():
            si_duck_preset_combo.config(state='readonly' if si_duck_var.get() else 'disabled')

        def refresh_si_options():
            if si_enabled_var.get():
                dub_mode_row.grid()
                si_options_frame.grid()
            else:
                dub_mode_row.grid_remove()
                si_options_frame.grid_remove()
            refresh_duck_preset_state()

        si_enabled_check.config(command=refresh_si_options)
        si_duck_check.config(command=refresh_duck_preset_state)
        refresh_si_options()

        # Path List
        ttk.Label(dialog, text=get_text('lbl_dlna_dirs')).grid(row=5, column=0, columnspan=2, sticky='w', padx=18, pady=(6, 2))

        list_frame = ttk.Frame(dialog)
        list_frame.grid(row=6, column=0, columnspan=2, sticky='nsew', padx=18, pady=2)
        
        listbox = tk.Listbox(list_frame, height=5, font=('Arial', 10))
        listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        scrollbar.pack(side='right', fill='y')
        listbox.config(yscrollcommand=scrollbar.set)
        
        for d in current_dirs:
            listbox.insert('end', d)

        # Add/Del
        btn_dirs_frame = ttk.Frame(dialog)
        btn_dirs_frame.grid(row=7, column=0, columnspan=2, sticky='e', padx=18, pady=(4, 3))
        
        def add_directory():
            chosen = filedialog.askdirectory(parent=dialog)
            if chosen:
                chosen = os.path.normpath(chosen)
                if dlna_media_library.is_unc_path(chosen):
                    messagebox.showwarning(
                        get_text('msg_dlna_unc_rejected_title'),
                        get_text('msg_dlna_unc_rejected'),
                        parent=dialog,
                    )
                    return
                existing = listbox.get(0, 'end')
                if chosen not in existing:
                    listbox.insert('end', chosen)
                    
        def delete_directory():
            selected = listbox.curselection()
            if selected:
                listbox.delete(selected[0])

        ttk.Button(btn_dirs_frame, text=get_text('btn_add_dir'), command=add_directory, width=10).pack(side='left', padx=5)
        ttk.Button(btn_dirs_frame, text=get_text('btn_del_dir'), command=delete_directory, width=10).pack(side='left', padx=5)

        ttk.Label(
            dialog,
            text=get_text('lbl_dlna_dirs_note'),
            foreground='gray',
            wraplength=640,
            justify='left',
        ).grid(row=8, column=0, columnspan=2, sticky='ew', padx=18, pady=(0, 3))

        # Bottom Buttons
        bottom_btn_frame = ttk.Frame(dialog)
        bottom_btn_frame.grid(row=9, column=0, columnspan=2, sticky='e', padx=18, pady=(6, 12))

        def save_config():
            name = name_var.get().strip()
            try:
                port = int(port_var.get())
            except ValueError:
                port = 8090

            auto_sub = auto_sub_var.get()
            raw_dirs = [str(d).strip() for d in listbox.get(0, 'end') if str(d).strip()]
            dirs = [d for d in raw_dirs if not dlna_media_library.is_unc_path(d)]
            if len(dirs) != len(raw_dirs):
                messagebox.showwarning(
                    get_text('msg_dlna_unc_rejected_title'),
                    get_text('msg_dlna_unc_rejected'),
                    parent=dialog,
                )
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
            app_config.set('dlna_si_enabled', bool(si_enabled_var.get()))
            app_config.set('dlna_si_mix_channel', channel_map.get(si_channel_var.get(), 'both'))
            app_config.set('dlna_si_original_volume_percent', int(si_origvol_var.get().rstrip('%')))
            app_config.set('dlna_si_volume_percent', int(si_volume_var.get().rstrip('%')))
            app_config.set('dlna_si_delay_seconds', float(si_delay_var.get().rstrip('s')))
            app_config.set('dlna_si_duck_original', bool(si_duck_var.get()))
            app_config.set('dlna_si_duck_preset', duck_preset_map.get(si_duck_preset_var.get(), 'normal'))
            app_config.set('dlna_si_dub_mode', bool(si_dub_mode_var.get()))
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
        self._refresh_dlna_ui()
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


def _selftest_import() -> int:
    """Diagnose transformers lazy-import resolution in the frozen build.

    Usage: VR_Video_Toolbox.exe --selftest-import
    """
    import os
    import traceback
    try:
        import transformers
        print(f"[selftest] transformers.__file__ = {getattr(transformers, '__file__', None)!r}")
        tdir = os.path.dirname(getattr(transformers, "__file__", "") or "")
        print(f"[selftest] transformers dir isdir={os.path.isdir(tdir)} : {tdir}")
        higgs_init = os.path.join(tdir, "models", "higgs_audio_v2_tokenizer", "__init__.py")
        print(f"[selftest] higgs __init__.py on disk = {os.path.isfile(higgs_init)} : {higgs_init}")
    except Exception:
        traceback.print_exc()
        return 2
    try:
        from transformers import HiggsAudioV2TokenizerModel
        print(f"[selftest] HiggsAudioV2TokenizerModel OK -> {HiggsAudioV2TokenizerModel}")
        return 0
    except Exception:
        print("[selftest] HiggsAudioV2TokenizerModel FAILED, full chained traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if "--selftest-gpu" in sys.argv:
        sys.exit(_selftest_gpu())
    if "--selftest-import" in sys.argv:
        sys.exit(_selftest_import())
    if _startup_gpu_warmup_enabled():
        _start_gpu_warmup()
    root = tk.Tk()
    app = VRVideoToolboxLauncher(root)
    root.mainloop()

VRMosaicLauncher = VRVideoToolboxLauncher
