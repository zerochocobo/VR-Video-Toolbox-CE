from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from utils import i18n
from tool_clonevoice import proofread


def get_text(key: str) -> str:
    return i18n.translate("clonevoice", key)


def _short(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


class ProofreadPanel(ttk.LabelFrame):
    def __init__(
        self,
        parent,
        *,
        app,
        get_videos,
        run_async,
        log_widget,
        show_speaker: bool,
        get_target_language,
        get_stop_event,
    ):
        super().__init__(parent, text=get_text("lbl_proofread_panel"), padding=6)
        self.app = app
        self.get_videos = get_videos
        self.run_async = run_async
        self.log_widget = log_widget
        self.show_speaker = show_speaker
        self.get_target_language = get_target_language
        self.get_stop_event = get_stop_event
        self.videos: list[str] = []
        self.iid_to_video: dict[str, str] = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        columns = ("video", "status", "ref")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=3)
        headings = {
            "video": get_text("col_pf_video"),
            "status": get_text("col_pf_trans_status"),
            "ref": get_text("col_pf_ref_srt"),
        }
        for col, width, anchor, stretch in (
            ("video", 360, "w", True),
            ("status", 160, "w", False),
            ("ref", 180, "w", False),
        ):
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=width, anchor=anchor, stretch=stretch)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<Double-1>", lambda _event: self.open_selected())

        actions = ttk.Frame(self)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        actions.columnconfigure(0, weight=1)
        self.btn_proofread = ttk.Button(actions, text=get_text("btn_proofread"), command=self.open_selected)
        self.btn_proofread.grid(row=0, column=1, sticky="e")
        self.action_buttons = [self.btn_proofread]

    def refresh(self) -> None:
        self.videos = list(self.get_videos() or [])
        self.iid_to_video.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for index, video in enumerate(self.videos):
            status = proofread.video_status(video)
            iid = str(index)
            self.iid_to_video[iid] = video
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(Path(video).name, self._format_status(status), self._format_ref(status)),
            )
        if self.videos:
            first = "0"
            self.tree.selection_set(first)
            self.tree.focus(first)

    def _format_status(self, status: dict) -> str:
        code = status.get("status")
        if code == "no_manifest":
            return get_text("pf_status_no_manifest")
        if code == "proofread":
            return get_text("pf_status_proofread").format(status.get("edited", 0), status.get("total", 0))
        if code == "translated":
            return get_text("pf_status_translated").format(status.get("translated", 0))
        return get_text("pf_status_untranslated").format(status.get("translated", 0), status.get("total", 0))

    def _format_ref(self, status: dict) -> str:
        ref = status.get("reference_srt") or ""
        return Path(ref).name if ref else get_text("pf_ref_none")

    def _selected_video(self) -> str:
        selection = self.tree.selection()
        if not selection:
            return ""
        return self.iid_to_video.get(selection[0], "")

    def _log_widget(self):
        return self.log_widget() if callable(self.log_widget) else self.log_widget

    def open_selected(self) -> None:
        # Double-click bypasses the disabled button, so gate on its state too;
        # saving the manifest while an export worker runs would race it.
        if str(self.btn_proofread.cget("state")) == "disabled":
            return
        video = self._selected_video()
        if not video:
            return
        status = proofread.video_status(video)
        if status.get("status") == "no_manifest":
            messagebox.showerror("Error", get_text("err_pf_need_transcribe"))
            return
        if status.get("status") == "untranslated":
            if not messagebox.askyesno("Confirm", get_text("confirm_pf_translate_first")):
                return
            if hasattr(self.app, "_translation_api_configured") and not self.app._translation_api_configured():
                messagebox.showerror("Error", get_text("err_no_translation_api_key"))
                return
            target_language = self.get_target_language()

            def worker(_holder, _release_holder):
                from tool_clonevoice import logic

                return logic.run_translate(
                    video,
                    target_language=target_language,
                    log=lambda m: self.app.log(self._log_widget(), m),
                    stop_event=self.get_stop_event(),
                )

            def done(_result):
                self.refresh()
                self._open_dialog(video)

            self.run_async(worker, done)
            return
        self._open_dialog(video)

    def _open_dialog(self, video: str) -> None:
        try:
            ProofreadDialog(
                self.app.root,
                video,
                show_speaker=self.show_speaker,
                on_saved=lambda result: self._on_saved(video, result),
            )
        except Exception as exc:
            messagebox.showerror("Error", get_text("err_pf_load_failed").format(exc))

    def _on_saved(self, video: str, result: dict) -> None:
        self.refresh()
        message = get_text("msg_pf_saved").format(result.get("changed_count", 0), result.get("translated_srt", ""))
        try:
            from tool_si import logic as si_logic

            si_path = si_logic.default_si_audio_path(video)
        except Exception:
            si_path = ""
        if si_path and os.path.exists(si_path):
            message += "\n\n" + get_text("msg_pf_si_exists_warn").format(si_path)
        messagebox.showinfo("Info", message)


class ProofreadDialog(tk.Toplevel):
    def __init__(self, root, video: str, *, show_speaker: bool, on_saved=None):
        # Load before creating the Toplevel so a failure cannot leave an
        # orphan empty window behind.
        data = proofread.load_rows(video)
        super().__init__(root)
        self.video = str(video)
        self.show_speaker = show_speaker
        self.on_saved = on_saved
        self.data = data
        self.rows: list[dict] = self.data["rows"]
        self.has_reference = bool(self.data.get("reference_srt"))
        self.current_iid: str | None = None
        self.loading = False

        manifest = self.data.get("manifest") or {}
        target_language = manifest.get("target_language") or ""
        self.title(get_text("dlg_proofread_title").format(Path(video).name, target_language))
        self.geometry("1100x470")
        self.minsize(760, 360)
        self.transient(root)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Escape>", lambda _event: self.cancel())

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_hint()
        self._build_tree()
        self._build_editor()
        self._build_actions()
        self._populate_tree()
        if self.tree.get_children():
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)
            self._load_row(first)

    def _build_hint(self) -> None:
        if self.has_reference:
            text = get_text("msg_ref_srt_hint").format(self.data.get("reference_srt", ""))
        else:
            text = get_text("msg_no_ref_srt_hint")
        ttk.Label(self, text=text, foreground="dim gray", wraplength=1040, justify="left").grid(
            row=0, column=0, sticky="ew", padx=8, pady=(8, 4)
        )

    def _columns(self) -> list[str]:
        columns = ["num"]
        if self.has_reference:
            columns.extend(["ref_time", "ref_text"])
        columns.append("time")
        if self.show_speaker:
            columns.append("speaker")
        columns.extend(["src", "tgt"])
        return columns

    def _build_tree(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = self._columns()
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        widths = {
            "num": 52,
            "ref_time": 160,
            "ref_text": 230,
            "time": 160,
            "speaker": 96,
            "src": 260,
            "tgt": 300,
        }
        for col in columns:
            self.tree.heading(col, text=get_text(f"col_pf_{col}"))
            self.tree.column(col, width=widths.get(col, 120), anchor="w", stretch=col in {"src", "tgt", "ref_text"})
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.tag_configure("modified", background="#fff2b8")
        self.tree.tag_configure("ref_only", foreground="#777777")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _build_editor(self) -> None:
        editor = ttk.LabelFrame(self, text=get_text("lbl_pf_editor"), padding=6)
        editor.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        editor.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(editor, text=get_text("col_pf_src")).grid(row=row, column=0, sticky="nw", padx=(0, 6), pady=2)
        self.src_text = tk.Text(editor, height=2, wrap="word")
        self.src_text.grid(row=row, column=1, sticky="ew", pady=2)
        self.src_text.config(state="disabled")
        row += 1
        if self.has_reference:
            ttk.Label(editor, text=get_text("col_pf_ref_text")).grid(row=row, column=0, sticky="nw", padx=(0, 6), pady=2)
            self.ref_text = tk.Text(editor, height=2, wrap="word")
            self.ref_text.grid(row=row, column=1, sticky="ew", pady=2)
            self.ref_text.config(state="disabled")
            row += 1
        else:
            self.ref_text = None
        ttk.Label(editor, text=get_text("col_pf_tgt")).grid(row=row, column=0, sticky="nw", padx=(0, 6), pady=2)
        self.tgt_text = tk.Text(editor, height=3, wrap="word", undo=True)
        self.tgt_text.grid(row=row, column=1, sticky="ew", pady=2)
        self.tgt_text.bind("<Control-Return>", self._commit_and_next)
        self.tgt_text.bind("<Control-Down>", self._commit_and_next)
        buttons = ttk.Frame(editor)
        buttons.grid(row=row, column=2, sticky="nsw", padx=(6, 0))
        self.btn_apply_row = ttk.Button(buttons, text=get_text("btn_apply_ref_row"), command=self.apply_ref_row)
        self.btn_apply_row.pack(fill="x", pady=(0, 4))
        self.btn_revert_row = ttk.Button(buttons, text=get_text("btn_revert_row"), command=self.revert_row)
        self.btn_revert_row.pack(fill="x")
        self.btn_play_source = ttk.Button(buttons, text=get_text("btn_pf_play_source"), command=self.play_row_source)
        self.btn_play_source.pack(fill="x", pady=(4, 0))
        if not self.has_reference:
            self.btn_apply_row.pack_forget()

    def _build_actions(self) -> None:
        actions = ttk.Frame(self)
        actions.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 8))
        actions.columnconfigure(1, weight=1)
        self.btn_apply_all = ttk.Button(actions, text=get_text("btn_apply_ref_all"), command=self.apply_ref_all)
        self.btn_apply_all.grid(row=0, column=0, sticky="w")
        if not self.has_reference:
            self.btn_apply_all.grid_remove()
        self.modified_var = tk.StringVar()
        ttk.Label(actions, textvariable=self.modified_var, foreground="dim gray").grid(row=0, column=1, sticky="e", padx=8)
        ttk.Button(actions, text=get_text("btn_pf_save"), command=self.save).grid(row=0, column=2, sticky="e", padx=(0, 6))
        ttk.Button(actions, text=get_text("btn_cancel"), command=self.cancel).grid(row=0, column=3, sticky="e")
        self._refresh_modified_count()

    def _populate_tree(self) -> None:
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        seg_index = 0
        for index, row in enumerate(self.rows):
            if row.get("kind") == "seg":
                seg_index += 1
                num = str(seg_index)
            else:
                num = ""
            values = {"num": num}
            if self.has_reference:
                values["ref_time"] = row.get("ref_time", "")
                values["ref_text"] = _short(row.get("ref_text", ""))
            values["time"] = row.get("time", "")
            if self.show_speaker:
                values["speaker"] = row.get("speaker", "")
            values["src"] = _short(row.get("src_text", ""))
            values["tgt"] = _short(row.get("tgt_text", ""))
            self.tree.insert("", "end", iid=str(index), values=[values.get(col, "") for col in self._columns()])
            self._refresh_row_tags(str(index))

    def _on_select(self, _event=None) -> None:
        if self.loading:
            return
        selection = self.tree.selection()
        if not selection:
            return
        new_iid = selection[0]
        if self.current_iid is not None and self.current_iid != new_iid:
            self._commit_editor()
        self._load_row(new_iid)

    def _row(self, iid: str | None = None) -> dict | None:
        target = iid if iid is not None else self.current_iid
        if target is None:
            return None
        try:
            return self.rows[int(target)]
        except Exception:
            return None

    def _set_text(self, widget: tk.Text | None, value: str, *, disabled: bool) -> None:
        if widget is None:
            return
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value or "")
        widget.config(state="disabled" if disabled else "normal")

    def _load_row(self, iid: str) -> None:
        row = self._row(iid)
        if row is None:
            return
        self.loading = True
        try:
            self.current_iid = iid
            readonly = row.get("kind") != "seg"
            self._set_text(self.src_text, row.get("src_text", ""), disabled=True)
            self._set_text(self.ref_text, row.get("ref_text", ""), disabled=True)
            self._set_text(self.tgt_text, row.get("tgt_text", ""), disabled=readonly)
            if hasattr(self, "btn_apply_row"):
                state = "normal" if self.has_reference and row.get("kind") == "seg" and row.get("ref_text") else "disabled"
                self.btn_apply_row.config(state=state)
            self.btn_revert_row.config(state="normal" if row.get("kind") == "seg" else "disabled")
            # ref_only rows keep the play button: hearing the span whisper
            # missed is exactly what they are shown for.
            has_span = (row.get("end") or 0.0) > (row.get("start") or 0.0)
            self.btn_play_source.config(state="normal" if has_span else "disabled")
        finally:
            self.loading = False

    def _commit_editor(self) -> None:
        row = self._row()
        if row is None or row.get("kind") != "seg":
            return
        text = self.tgt_text.get("1.0", "end-1c").strip()
        row["tgt_text"] = text
        self._refresh_tree_row(self.current_iid)
        self._refresh_modified_count()

    def _refresh_tree_row(self, iid: str | None) -> None:
        if iid is None:
            return
        row = self._row(iid)
        if row is None:
            return
        values = {
            "num": self.tree.set(iid, "num") if "num" in self._columns() else "",
            "time": row.get("time", ""),
            "src": _short(row.get("src_text", "")),
            "tgt": _short(row.get("tgt_text", "")),
        }
        if self.has_reference:
            values["ref_time"] = row.get("ref_time", "")
            values["ref_text"] = _short(row.get("ref_text", ""))
        if self.show_speaker:
            values["speaker"] = row.get("speaker", "")
        self.tree.item(iid, values=[values.get(col, "") for col in self._columns()])
        self._refresh_row_tags(iid)

    def _refresh_row_tags(self, iid: str) -> None:
        row = self._row(iid)
        if row is None:
            return
        tags = []
        if row.get("kind") == "ref_only":
            tags.append("ref_only")
        if row.get("kind") == "seg" and (row.get("tgt_text") or "").strip() != (row.get("original_tgt_text") or "").strip():
            tags.append("modified")
        self.tree.item(iid, tags=tags)

    def _modified_count(self) -> int:
        return sum(
            1
            for row in self.rows
            if row.get("kind") == "seg"
            and (row.get("tgt_text") or "").strip() != (row.get("original_tgt_text") or "").strip()
        )

    def _refresh_modified_count(self) -> None:
        self.modified_var.set(get_text("msg_pf_modified_count").format(self._modified_count()))

    def _commit_and_next(self, _event=None):
        self._commit_editor()
        children = list(self.tree.get_children())
        if self.current_iid in children:
            next_index = min(len(children) - 1, children.index(self.current_iid) + 1)
            next_iid = children[next_index]
            self.tree.selection_set(next_iid)
            self.tree.focus(next_iid)
            self.tree.see(next_iid)
        return "break"

    def apply_ref_row(self) -> None:
        row = self._row()
        if row is None or row.get("kind") != "seg" or not row.get("ref_text"):
            return
        self._set_text(self.tgt_text, row.get("ref_text", ""), disabled=False)
        self._commit_editor()

    def apply_ref_all(self) -> None:
        self._commit_editor()
        for index, row in enumerate(self.rows):
            if row.get("kind") == "seg" and row.get("ref_text"):
                row["tgt_text"] = row.get("ref_text", "").strip()
                self._refresh_tree_row(str(index))
        if self.current_iid is not None:
            self._load_row(self.current_iid)
        self._refresh_modified_count()

    def revert_row(self) -> None:
        row = self._row()
        if row is None or row.get("kind") != "seg":
            return
        row["tgt_text"] = row.get("original_tgt_text", "")
        self._set_text(self.tgt_text, row.get("tgt_text", ""), disabled=False)
        self._refresh_tree_row(self.current_iid)
        self._refresh_modified_count()

    def play_row_source(self) -> None:
        row = self._row()
        if row is None:
            return
        try:
            import winsound

            # Stop the current playback first so the shared preview file is
            # released before it gets rewritten.
            winsound.PlaySound(None, 0)
            clip = proofread.cut_segment_preview(
                self.video, float(row.get("start") or 0.0), float(row.get("end") or 0.0)
            )
            winsound.PlaySound(str(clip), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            messagebox.showerror("Error", get_text("err_pf_play_failed").format(exc), parent=self)

    def destroy(self) -> None:
        try:
            import winsound

            winsound.PlaySound(None, 0)
        except Exception:
            pass
        super().destroy()

    def save(self) -> None:
        self._commit_editor()
        try:
            result = proofread.save_rows(self.video, self.rows)
        except Exception as exc:
            messagebox.showerror("Error", get_text("err_pf_save_failed").format(exc), parent=self)
            return
        self.destroy()
        if self.on_saved:
            self.on_saved(result)

    def cancel(self) -> None:
        self._commit_editor()
        if self._modified_count() and not messagebox.askyesno("Confirm", get_text("confirm_pf_unsaved")):
            return
        self.destroy()


def build_proofread_panel(parent, **kwargs):
    panel = ProofreadPanel(parent, **kwargs)
    return panel, panel.refresh, panel.action_buttons
