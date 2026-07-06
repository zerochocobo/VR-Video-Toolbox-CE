from __future__ import annotations

from pathlib import Path
from tkinter import ttk
from typing import Callable


class CandidateBasisPanel:
    """Reusable candidate table for choosing a target-language voice basis."""

    def __init__(
        self,
        parent,
        *,
        get_text: Callable[[str], str],
        play_wav: Callable[[str, str], None],
        include_video: bool = True,
        height: int = 9,
    ) -> None:
        self.get_text = get_text
        self.play_wav = play_wav
        self.include_video = include_video
        self.candidates: list[dict] = []
        self.iid_to_index: dict[str, int] = {}

        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        columns = ["play_source", "play_translation", "play_sample", "rank"]
        if include_video:
            columns.append("video")
        columns.extend(["time", "dur", "score", "sim", "text"])
        self.columns = tuple(columns)

        self.tree = ttk.Treeview(self.frame, columns=self.columns, show="headings", height=height)
        widths = {
            "play_source": 58,
            "play_translation": 58,
            "play_sample": 58,
            "rank": 48,
            "video": 140,
            "time": 110,
            "dur": 58,
            "score": 70,
            "sim": 78,
            "text": 320,
        }
        anchors = {
            "play_source": "center",
            "play_translation": "center",
            "play_sample": "center",
            "rank": "center",
            "video": "w",
            "time": "center",
            "dur": "center",
            "score": "center",
            "sim": "center",
            "text": "w",
        }
        for col in self.columns:
            self.tree.heading(col, text=get_text(f"col_{col}"))
            self.tree.column(col, width=widths[col], anchor=anchors[col], stretch=(col == "text"))
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<ButtonRelease-1>", self._on_click)

        scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

    def grid(self, *args, **kwargs) -> None:
        self.frame.grid(*args, **kwargs)

    def pack(self, *args, **kwargs) -> None:
        self.frame.pack(*args, **kwargs)

    def set_candidates(self, candidates: list[dict]) -> None:
        self.candidates = list(candidates or [])
        self.tree.delete(*self.tree.get_children())
        self.iid_to_index = {}
        for idx, cand in enumerate(self.candidates):
            iid = str(idx)
            self.iid_to_index[iid] = idx
            start = float(cand.get("start") or 0.0)
            end = float(cand.get("end") or 0.0)
            score = cand.get("score")
            sim = cand.get("ecapa_similarity")
            values = [
                self.get_text("btn_play_inline") if cand.get("source_audio") else "",
                self.get_text("btn_play_inline") if cand.get("translated_audio") else "",
                self.get_text("btn_play_inline") if cand.get("target_sample_audio") else "",
                cand.get("global_rank") or idx + 1,
            ]
            if self.include_video:
                values.append(Path(cand.get("video") or "").name)
            values.extend(
                [
                    f"{start:.2f}-{end:.2f}",
                    f"{float(cand.get('dur') or 0.0):.1f}",
                    "" if score is None else f"{float(score):.3f}",
                    "" if sim is None else f"{float(sim):.3f}",
                    (cand.get("tgt_text") or cand.get("src_text") or "")[:160],
                ]
            )
            self.tree.insert("", "end", iid=iid, values=tuple(values))

    def selected_candidate(self) -> dict | None:
        selection = self.tree.selection()
        if not selection:
            return None
        idx = self.iid_to_index.get(selection[0])
        if idx is None or idx >= len(self.candidates):
            return None
        return self.candidates[idx]

    def _on_click(self, event) -> None:
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not row_id:
            return
        self.tree.selection_set(row_id)
        idx = self.iid_to_index.get(row_id)
        if idx is None or idx >= len(self.candidates):
            return
        cand = self.candidates[idx]
        if col == "#1":
            self.play_wav(cand.get("source_audio") or "", self.get_text("col_play_source"))
        elif col == "#2":
            self.play_wav(cand.get("translated_audio") or "", self.get_text("col_play_translation"))
        elif col == "#3":
            self.play_wav(cand.get("target_sample_audio") or "", self.get_text("col_play_sample"))
