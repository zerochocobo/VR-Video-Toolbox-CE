"""Shared colors, ttk styling, and left-navigation UI primitives."""
from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from utils import app_config


NAV_ICON_FONT = "Segoe MDL2 Assets"


@dataclass(frozen=True)
class Palette:
    HOME_BG: str
    SIDEBAR_BG: str
    SIDEBAR_ITEM_FG: str
    SIDEBAR_HOVER_BG: str
    SIDEBAR_SEL_BG: str
    SIDEBAR_SEL_FG: str
    CARD_BG: str
    CARD_HOVER_BG: str
    CARD_BORDER: str
    CARD_TITLE_FG: str
    CARD_DESC_FG: str
    PRIMARY_BG: str
    PRIMARY_HOVER_BG: str
    PRIMARY_BORDER: str
    PRIMARY_TITLE_FG: str
    PRIMARY_DESC_FG: str
    INPUT_BG: str
    INPUT_FG: str
    BUTTON_BG: str
    BUTTON_HOVER_BG: str
    DISABLED_FG: str
    DOT_RUNNING: str = "#2e9e5b"
    DOT_STOPPED: str = "#9a9a9a"


LIGHT_PALETTE = Palette(
    HOME_BG="#f0f0f0",
    SIDEBAR_BG="#e4e4e4",
    SIDEBAR_ITEM_FG="#333333",
    SIDEBAR_HOVER_BG="#dadada",
    SIDEBAR_SEL_BG="#d3e3f8",
    SIDEBAR_SEL_FG="#1a5cb0",
    CARD_BG="#ffffff",
    CARD_HOVER_BG="#f2f6fc",
    CARD_BORDER="#c9c9c9",
    CARD_TITLE_FG="#222222",
    CARD_DESC_FG="#777777",
    PRIMARY_BG="#e8f1fc",
    PRIMARY_HOVER_BG="#dcebfb",
    PRIMARY_BORDER="#3f7fd6",
    PRIMARY_TITLE_FG="#1a5cb0",
    PRIMARY_DESC_FG="#4a7fc0",
    INPUT_BG="#ffffff",
    INPUT_FG="#202020",
    BUTTON_BG="#e8e8e8",
    BUTTON_HOVER_BG="#dcdcdc",
    DISABLED_FG="#8a8a8a",
)

DARK_PALETTE = Palette(
    HOME_BG="#1f2329",
    SIDEBAR_BG="#171a1f",
    SIDEBAR_ITEM_FG="#d9dde5",
    SIDEBAR_HOVER_BG="#292e36",
    SIDEBAR_SEL_BG="#243b57",
    SIDEBAR_SEL_FG="#8fc2ff",
    CARD_BG="#292e36",
    CARD_HOVER_BG="#323945",
    CARD_BORDER="#444b57",
    CARD_TITLE_FG="#f1f3f5",
    CARD_DESC_FG="#aeb6c2",
    PRIMARY_BG="#243b57",
    PRIMARY_HOVER_BG="#2b496d",
    PRIMARY_BORDER="#65a7f3",
    PRIMARY_TITLE_FG="#9ac8ff",
    PRIMARY_DESC_FG="#b5d5fa",
    INPUT_BG="#171a1f",
    INPUT_FG="#f1f3f5",
    BUTTON_BG="#343a44",
    BUTTON_HOVER_BG="#414955",
    DISABLED_FG="#747c87",
    DOT_RUNNING="#49bd78",
    DOT_STOPPED="#767d87",
)

PALETTES = {"light": LIGHT_PALETTE, "dark": DARK_PALETTE}


def normalize_theme(value: object) -> str:
    return "dark" if str(value or "").strip().lower() == "dark" else "light"


def get_theme() -> str:
    return normalize_theme(app_config.get("ui_theme", "light"))


def get_palette(theme: str | None = None) -> Palette:
    return PALETTES[normalize_theme(theme if theme is not None else get_theme())]


def apply_theme(root: tk.Misc, theme: str | None = None) -> Palette:
    """Apply the configured palette to ttk and future classic Tk widgets."""
    palette = get_palette(theme)
    try:
        root.configure(background=palette.HOME_BG)
    except (tk.TclError, AttributeError):
        pass

    for pattern, value in (
        ("*Background", palette.HOME_BG),
        ("*Foreground", palette.CARD_TITLE_FG),
        ("*Text.background", palette.INPUT_BG),
        ("*Text.foreground", palette.INPUT_FG),
        ("*Entry.background", palette.INPUT_BG),
        ("*Entry.foreground", palette.INPUT_FG),
        ("*Listbox.background", palette.INPUT_BG),
        ("*Listbox.foreground", palette.INPUT_FG),
        ("*Canvas.background", palette.CARD_BG),
        ("*Menu.background", palette.CARD_BG),
        ("*Menu.foreground", palette.CARD_TITLE_FG),
        ("*selectBackground", palette.SIDEBAR_SEL_BG),
        ("*selectForeground", palette.SIDEBAR_SEL_FG),
    ):
        try:
            root.option_add(pattern, value)
        except tk.TclError:
            pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    common = {"background": palette.HOME_BG, "foreground": palette.CARD_TITLE_FG}
    style.configure("TFrame", background=palette.HOME_BG)
    style.configure("TLabel", **common)
    style.configure("TLabelframe", background=palette.HOME_BG, foreground=palette.CARD_TITLE_FG)
    style.configure("TLabelframe.Label", **common)
    style.configure("TButton", background=palette.BUTTON_BG, foreground=palette.CARD_TITLE_FG)
    style.map(
        "TButton",
        background=[("active", palette.BUTTON_HOVER_BG), ("disabled", palette.HOME_BG)],
        foreground=[("disabled", palette.DISABLED_FG)],
    )
    style.configure("TCheckbutton", **common)
    style.configure("TRadiobutton", **common)
    style.map("TCheckbutton", background=[("active", palette.HOME_BG)])
    style.map("TRadiobutton", background=[("active", palette.HOME_BG)])
    style.configure("TEntry", fieldbackground=palette.INPUT_BG, foreground=palette.INPUT_FG)
    style.configure(
        "TCombobox",
        fieldbackground=palette.INPUT_BG,
        background=palette.BUTTON_BG,
        foreground=palette.INPUT_FG,
        arrowcolor=palette.CARD_TITLE_FG,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette.INPUT_BG)],
        foreground=[("readonly", palette.INPUT_FG)],
        selectbackground=[("readonly", palette.INPUT_BG)],
        selectforeground=[("readonly", palette.INPUT_FG)],
    )
    style.configure(
        "Treeview",
        background=palette.INPUT_BG,
        fieldbackground=palette.INPUT_BG,
        foreground=palette.INPUT_FG,
    )
    style.configure(
        "Treeview.Heading",
        background=palette.BUTTON_BG,
        foreground=palette.CARD_TITLE_FG,
    )
    style.map("Treeview", background=[("selected", palette.SIDEBAR_SEL_BG)])
    style.configure("Link.TLabel", background=palette.HOME_BG, foreground=palette.SIDEBAR_SEL_FG)
    style.configure("Big.TButton", font=("Arial", 11), padding=(6, 6))
    return palette


def scroll_text_to_end(widget: tk.Text) -> None:
    """Keep a log text widget pinned to its newest appended line."""
    try:
        widget.see(tk.END)
        # Text.see("end") may stop just short of the actual scrollbar end for
        # very short log panes because Tk includes the trailing empty line.
        widget.yview_moveto(1.0)
    except tk.TclError:
        pass


_ICON_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("截图", "スクショ", "screenshot", "确认", "確認", "预览", "preview"), "\uE722"),
    (("补丁", "补丁工具", "patch", "patcher", "パッチ"), "\uE90F"),
    (("嵌入", "embed", "エンコード", "再エンコード"), "\uE8F1"),
    (("位置", "範囲", "选择", "選択", "select", "locate", "区域"), "\uE81E"),
    (("处理", "処理", "process", "视频", "video"), "\uE945"),
    (("混合", "混音", "音轨", "audio remix", "mix", "ミックス"), "\uE720"),
    (("单文件", "单体", "単体", "single", "一人", "単一"), "\uE77B"),
    (("质量", "品質", "品質確認", "rank", "quality"), "\uE8CB"),
    (("马赛克", "mosaic", "モザイク"), "\uE75C"),
    (("dlna", "服务器", "server", "サーバー"), "\uEC15"),
    (("设置", "settings", "設定"), "\uE713"),
    (("工具", "tools", "utility", "ツール"), "\uE90F"),
    (("截图", "screenshot", "スクリーン"), "\uE722"),
    (("关键帧", "keyframe", "キーフレーム"), "\uE71B"),
    (("放大", "zoom", "拡大"), "\uE8A3"),
    (("裁剪", "cut", "crop", "カット"), "\uE8C6"),
    (("转码", "转编码", "transcode", "変換", "再エンコード"), "\uE8F1"),
    (("翻译", "translate", "translation", "翻訳"), "\uE8C7"),
    (("字幕", "subtitle", "srt", "ass", "キャプション"), "\uE7F0"),
    (("语音", "voice", "音声", "克隆", "clone"), "\uE720"),
    (("批量", "batch", "folder", "フォルダ"), "\uE8B7"),
    (("单人", "单个", "single", "一人", "単一"), "\uE77B"),
    (("多人", "multi", "複数"), "\uE716"),
    (("提取", "extract", "抽出"), "\uE8B5"),
    (("定位", "locate", "preview", "预览", "選択", "选区"), "\uE81E"),
    (("拆分", "split", "分割"), "\uE8C8"),
    (("合并", "merge", "combine", "混音", "回混", "結合", "ミックス"), "\uE8B3"),
    (("删除", "remove", "削除"), "\uE74D"),
    (("排序", "rank", "優先"), "\uE8CB"),
    (("鱼眼", "魚眼", "fisheye", "equirect", "投影", "正距"), "\uE909"),
    (("自动", "auto", "一键", "one-click", "処理", "process"), "\uE945"),
)
DEFAULT_NAV_ICON = "\uE713"


def icon_for_title(title: object) -> str:
    text = str(title or "").casefold()
    for keywords, glyph in _ICON_RULES:
        if any(keyword.casefold() in text for keyword in keywords):
            return glyph
    return DEFAULT_NAV_ICON


class SideNavigation(ttk.Frame):
    """Notebook-compatible page host with a left icon navigation rail."""

    def __init__(self, master=None, *, sidebar_width: int = 190, **kwargs):
        super().__init__(master, **kwargs)
        self._palette = get_palette()
        self._sidebar_width = sidebar_width
        self._pages: list[tk.Widget] = []
        self._page_text: list[str] = []
        self._nav_items: list[tuple[tk.Frame, tk.Label | None, tk.Label]] = []
        self._selected_index = -1
        self._icons_enabled = False
        try:
            self._icons_enabled = NAV_ICON_FONT in tkfont.families(self)
        except tk.TclError:
            pass

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._sidebar = tk.Frame(self, bg=self._palette.SIDEBAR_BG, width=sidebar_width)
        self._sidebar.grid(row=0, column=0, rowspan=2, sticky="nsw")
        self._sidebar.grid_propagate(False)

    def add(self, child: tk.Widget, **kwargs) -> None:
        text = str(kwargs.get("text", ""))
        icon_text = str(kwargs.get("icon") or icon_for_title(text))
        index = len(self._pages)
        self._pages.append(child)
        self._page_text.append(text)
        child.grid(row=0, column=1, sticky="nsew")

        item = tk.Frame(self._sidebar, bg=self._palette.SIDEBAR_BG, cursor="hand2")
        item.pack(fill="x", pady=(0, 1))
        icon = None
        if self._icons_enabled:
            icon = tk.Label(
                item,
                text=icon_text,
                bg=self._palette.SIDEBAR_BG,
                fg=self._palette.SIDEBAR_ITEM_FG,
                font=(NAV_ICON_FONT, 12),
                anchor="w",
            )
            icon.pack(side="left", padx=(16, 8), pady=8)
        label = tk.Label(
            item,
            text=text,
            bg=self._palette.SIDEBAR_BG,
            fg=self._palette.SIDEBAR_ITEM_FG,
            font=("Arial", 11),
            anchor="w",
            justify="left",
            wraplength=self._sidebar_width - (56 if icon else 32),
            padx=0 if icon else 16,
            pady=8,
        )
        label.pack(side="left", fill="x", expand=True)
        widgets = [widget for widget in (item, icon, label) if widget is not None]

        def enter(_event, page_index=index):
            if self._selected_index != page_index:
                self._set_item_colors(page_index, self._palette.SIDEBAR_HOVER_BG, self._palette.SIDEBAR_ITEM_FG)

        def leave(_event, page_index=index):
            if self._selected_index != page_index:
                self._set_item_colors(page_index, self._palette.SIDEBAR_BG, self._palette.SIDEBAR_ITEM_FG)

        for widget in widgets:
            widget.bind("<Button-1>", lambda _event, page_index=index: self.select(page_index))
            widget.bind("<Enter>", enter)
            widget.bind("<Leave>", leave)
        self._nav_items.append((item, icon, label))
        if self._selected_index < 0:
            self.select(0)
        else:
            # Adding/gridding a later page raises it above existing siblings in
            # Tk. Keep the visually displayed page aligned with the selected
            # navigation item while the remaining pages are being registered.
            self._pages[self._selected_index].tkraise()

    def _set_item_colors(self, index: int, background: str, foreground: str) -> None:
        item, icon, label = self._nav_items[index]
        item.configure(bg=background)
        if icon is not None:
            icon.configure(bg=background, fg=foreground)
        label.configure(bg=background, fg=foreground)

    def _resolve_index(self, tab_id) -> int:
        if isinstance(tab_id, int):
            index = tab_id
        elif isinstance(tab_id, tk.Widget):
            index = self._pages.index(tab_id)
        elif isinstance(tab_id, str) and tab_id.isdigit():
            index = int(tab_id)
        elif isinstance(tab_id, str):
            index = next(
                (page_index for page_index, page in enumerate(self._pages) if str(page) == tab_id),
                -1,
            )
        else:
            raise tk.TclError(f"unknown page {tab_id!r}")
        if index < 0 or index >= len(self._pages):
            raise tk.TclError(f"page index {index} out of range")
        return index

    def select(self, tab_id=None):
        if tab_id is None:
            if self._selected_index < 0:
                return ""
            return str(self._pages[self._selected_index])
        index = self._resolve_index(tab_id)
        self._selected_index = index
        self._pages[index].tkraise()
        for page_index, (_item, _icon, label) in enumerate(self._nav_items):
            selected = page_index == index
            bg = self._palette.SIDEBAR_SEL_BG if selected else self._palette.SIDEBAR_BG
            fg = self._palette.SIDEBAR_SEL_FG if selected else self._palette.SIDEBAR_ITEM_FG
            self._set_item_colors(page_index, bg, fg)
            label.configure(font=("Arial", 11, "bold") if selected else ("Arial", 11))
        return str(self._pages[index])

    def index(self, tab_id) -> int:
        if tab_id == "end":
            return len(self._pages)
        if isinstance(tab_id, str) and tab_id.startswith("@"):
            try:
                x_text, y_text = tab_id[1:].split(",", 1)
                x, y = int(x_text), int(y_text)
            except ValueError as exc:
                raise tk.TclError(f"bad page coordinate {tab_id!r}") from exc
            if x > self._sidebar_width:
                raise tk.TclError("coordinate is outside the navigation rail")
            for page_index, (item, _icon, _label) in enumerate(self._nav_items):
                if item.winfo_y() <= y < item.winfo_y() + item.winfo_height():
                    return page_index
            raise tk.TclError("coordinate is outside a navigation item")
        return self._resolve_index(tab_id)

    def tab(self, tab_id, option=None, **kwargs):
        index = self._resolve_index(tab_id)
        if "text" in kwargs:
            text = str(kwargs["text"])
            self._page_text[index] = text
            self._nav_items[index][2].configure(text=text)
        data = {"text": self._page_text[index]}
        return data.get(option, "") if option else data


# Explicit tab-icon vocabulary (Segoe MDL2 Assets). Keys are semantic so the
# same meaning always maps to the same glyph across every tool; values are
# built with chr() to keep this file free of invisible private-use characters.
TAB_ICONS = {
    "back": chr(0xE72B),         # left arrow
    "auto": chr(0xE945),         # lightning bolt
    "doc": chr(0xE8A5),          # document
    "eye": chr(0xE7B3),          # eye
    "folder": chr(0xE8B7),       # folder
    "folder_open": chr(0xE838),  # open folder
    "merge": chr(0xE8B3),        # combine grid
    "split": chr(0xE8C8),        # two sheets
    "extract": chr(0xE896),      # download arrow
    "locate": chr(0xE71E),       # magnifier
    "process": chr(0xE768),      # play
    "globe": chr(0xE909),        # filled globe
    "globe_wire": chr(0xE774),   # wireframe globe (also translation)
    "camera": chr(0xE722),       # camera
    "images": chr(0xEB9F),       # picture
    "wrench": chr(0xE90F),       # repair wrench
    "zoom": chr(0xE8A3),         # zoom in
    "cut": chr(0xE8C6),          # scissors
    "transcode": chr(0xE895),    # sync arrows
    "cc": chr(0xE7F0),           # closed captions
    "volume": chr(0xE767),       # speaker
    "font": chr(0xE8D2),         # AA font
    "attach": chr(0xE723),       # paperclip
    "delete": chr(0xE74D),       # trash bin
    "star": chr(0xE734),         # star
    "mic": chr(0xE720),          # microphone
    "checklist": chr(0xE762),    # multiselect list
    "person": chr(0xE77B),       # single contact
    "people": chr(0xE716),       # two people
    "preview": chr(0xE890),      # view
    "video": chr(0xE714),        # video camera
    "frame": chr(0xE91B),        # flat frame
}


class ToolShell(SideNavigation):
    """SideNavigation dressed in the launcher home page's sidebar chrome.

    The tool title sits at the top of the rail (mirroring the home page title
    block) and an optional back-to-home item is pinned at the bottom of the
    rail (the slot the home page uses for Settings), replacing the old
    full-width header row above the tabs.
    """

    def __init__(self, master=None, *, title="", back_text="", on_back=None,
                 sidebar_width: int = 190, **kwargs):
        super().__init__(master, sidebar_width=sidebar_width, **kwargs)
        self._footer = None
        if title:
            tk.Label(
                self._sidebar,
                text=title,
                bg=self._palette.SIDEBAR_BG,
                fg=self._palette.CARD_TITLE_FG,
                font=("Arial", 11, "bold"),
                wraplength=sidebar_width - 28,
                justify="left",
                anchor="w",
                padx=14,
            ).pack(fill="x", pady=(14, 10))
        if on_back is not None and back_text:
            self._add_back_item(back_text, on_back)

    def _add_back_item(self, text: str, command) -> None:
        palette = self._palette
        item = tk.Frame(self._sidebar, bg=palette.SIDEBAR_BG, cursor="hand2")
        item.pack(side="bottom", fill="x", pady=(1, 8))
        icon = None
        if self._icons_enabled:
            icon = tk.Label(
                item,
                text=TAB_ICONS["back"],
                bg=palette.SIDEBAR_BG,
                fg=palette.SIDEBAR_ITEM_FG,
                font=(NAV_ICON_FONT, 12),
                anchor="w",
            )
            icon.pack(side="left", padx=(16, 8), pady=8)
        label = tk.Label(
            item,
            text=text,
            bg=palette.SIDEBAR_BG,
            fg=palette.SIDEBAR_ITEM_FG,
            font=("Arial", 11),
            anchor="w",
            justify="left",
            wraplength=self._sidebar_width - (56 if icon else 32),
            padx=0 if icon else 16,
            pady=8,
        )
        label.pack(side="left", fill="x", expand=True)
        widgets = [widget for widget in (item, icon, label) if widget is not None]

        def set_colors(background):
            for widget in widgets:
                widget.configure(bg=background)

        for widget in widgets:
            widget.bind("<Button-1>", lambda _event: command())
            widget.bind("<Enter>", lambda _event: set_colors(palette.SIDEBAR_HOVER_BG))
            widget.bind("<Leave>", lambda _event: set_colors(palette.SIDEBAR_BG))

    def footer(self, expand: bool = False) -> ttk.Frame:
        """Shared area spanning the content column below the pages.

        Use for cross-tab widgets (logs, global settings bars). With
        expand=True the footer takes the surplus vertical space instead of the
        pages (pages keep their natural height).
        """
        if self._footer is None:
            self._footer = ttk.Frame(self)
            self._footer.grid(row=1, column=1, sticky="nsew")
            if expand:
                self.rowconfigure(0, weight=0)
                self.rowconfigure(1, weight=1)
        return self._footer
