from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from tool_clonevoice import gui_proofread


class _EnabledButton:
    def cget(self, name: str) -> str:
        assert name == "state"
        return "normal"


def test_untranslated_video_opens_proofread_without_ai_prompt(monkeypatch):
    open_dialog = Mock()
    panel = SimpleNamespace(
        btn_proofread=_EnabledButton(),
        _selected_video=lambda: "movie.mp4",
        _open_dialog=open_dialog,
    )
    monkeypatch.setattr(
        gui_proofread.proofread,
        "video_status",
        lambda _video: {"status": "untranslated", "translated": 1, "total": 2},
    )

    def fail_if_prompted(*_args, **_kwargs):
        raise AssertionError("AI translation prompt must not be shown while proofreading")

    monkeypatch.setattr(gui_proofread.messagebox, "askyesno", fail_if_prompted)

    gui_proofread.ProofreadPanel.open_selected(panel)

    open_dialog.assert_called_once_with("movie.mp4")


def test_status_uses_neutral_completed_count(monkeypatch):
    monkeypatch.setattr(gui_proofread, "get_text", lambda key: "Completed {}/{}" if key == "pf_status_completed" else key)

    text = gui_proofread.ProofreadPanel._format_status(
        SimpleNamespace(),
        {"status": "untranslated", "translated": 3, "total": 5},
    )

    assert text == "Completed 3/5"
