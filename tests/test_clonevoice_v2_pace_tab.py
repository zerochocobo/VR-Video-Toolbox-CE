"""The subtitle-debug tab is reachable, populated, and reads real rows.

a265bda registered the refined-clone tab and left it unreachable for months
because nothing checked that a tab is actually built and mapped. These do.
"""
import sys
import tkinter as tk
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import i18n
from tool_clonevoice_v2 import gui as gui_module
from tool_clonevoice_v2.gui_pace import PaceTab


def get_text(key):
    return i18n.translate("clonevoice", key)


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    yield window
    window.destroy()


@pytest.fixture
def app(root):
    app_cls = next(
        value for value in vars(gui_module).values()
        if isinstance(value, type) and hasattr(value, "_setup_ui")
    )
    return app_cls(root)


def test_the_tab_sits_below_the_mixing_tab(app):
    """Asked for: under 混音配音 rather than at the top -- it is where you go
    after a dub exists, not before."""
    assert app.notebook.index(app.tab_pace) == app.notebook.index(app.tab_mix) + 1
    assert app.notebook.tab(app.tab_pace, "text") == get_text("tab_pace")


def test_the_tab_is_actually_populated(app):
    assert app.tab_pace.winfo_children(), "tab registered but never built"
    assert isinstance(app.pace_tab, PaceTab)


def test_selecting_the_tab_maps_its_body(app, root):
    """Reachable, not merely constructed: a tab can exist and never be shown."""
    app.notebook.select(app.tab_pace)
    root.update()
    assert app.tab_pace.winfo_ismapped()
    assert app.pace_tab.tree.winfo_ismapped()


def test_it_shows_both_texts_and_both_play_buttons(app):
    """The whole point: read source against translation, hear either side."""
    tab = app.pace_tab
    assert tab.src_text is not None and tab.tgt_text is not None
    assert tab.btn_play_source["text"] == get_text("btn_pace_play_source")
    assert tab.btn_play_clone["text"] == get_text("btn_pace_play_clone")


def test_the_table_carries_the_columns_the_diagnosis_needs(app):
    columns = app.pace_tab.tree["columns"]
    for needed in ("slot", "clone", "coverage", "src", "tgt"):
        assert needed in columns


def test_rows_render_with_coverage_flagged(app, root):
    """A short line is marked so a title's problem lines can be found by eye."""
    tab = app.pace_tab
    tab._loaded({
        "rows": [
            {"id": 1, "start": 1.0, "end": 5.0, "slot": 4.0, "clone": 1.0,
             "coverage": 0.25, "exposed": 3.0, "speaker": "SPEAKER_00",
             "src_text": "ああ", "tgt_text": "啊", "src_chars": 2, "tgt_chars": 1},
            {"id": 2, "start": 6.0, "end": 8.0, "slot": 2.0, "clone": 1.9,
             "coverage": 0.95, "exposed": 0.1, "speaker": "SPEAKER_00",
             "src_text": "そう", "tgt_text": "是", "src_chars": 2, "tgt_chars": 1},
        ],
        "summary": {"lines": 2, "translated": 2, "measured": 2,
                    "median_coverage": 0.6, "under_exposed": 1, "under_half": 1,
                    "exposed_total": 3.1, "median_exposed": 1.55,
                    "median_char_ratio": 0.5},
    })
    root.update_idletasks()

    children = tab.tree.get_children()
    assert len(children) == 2
    assert "bad" in tab.tree.item(children[0], "tags")
    assert not tab.tree.item(children[1], "tags")
    assert tab.tree.set(children[0], "coverage") == "25%"
    assert tab.summary_var.get(), "a summary must be shown"


def test_selecting_a_row_shows_its_full_text(app, root):
    tab = app.pace_tab
    tab._loaded({
        "rows": [{"id": 1, "start": 0.0, "end": 4.0, "slot": 4.0, "clone": 2.0,
                  "coverage": 0.5, "exposed": 2.0, "speaker": "",
                  "src_text": "とても長い日本語", "tgt_text": "很长的中文",
                  "src_chars": 8, "tgt_chars": 5}],
        "summary": {"lines": 1, "translated": 1, "measured": 1,
                    "median_coverage": 0.5, "under_exposed": 1, "under_half": 0,
                    "exposed_total": 2.0, "median_exposed": 2.0,
                    "median_char_ratio": 0.62},
    })
    root.update_idletasks()

    assert tab.src_text.get("1.0", "end").strip() == "とても長い日本語"
    assert tab.tgt_text.get("1.0", "end").strip() == "很长的中文"


def test_a_title_with_no_dub_still_renders(app, root):
    """Comparing the texts is worth doing before anything has been exported."""
    tab = app.pace_tab
    tab._loaded({
        "rows": [{"id": 1, "start": 0.0, "end": 4.0, "slot": 4.0, "clone": None,
                  "coverage": None, "exposed": None, "speaker": "",
                  "src_text": "あ", "tgt_text": "啊", "src_chars": 1, "tgt_chars": 1}],
        "summary": {"lines": 1, "translated": 1, "measured": 0,
                    "median_coverage": None, "under_exposed": 0, "under_half": 0,
                    "exposed_total": 0.0, "median_exposed": None,
                    "median_char_ratio": 1.0},
    })
    root.update_idletasks()

    row = tab.tree.get_children()[0]
    assert tab.tree.set(row, "clone") == "-"
    assert tab.tree.set(row, "coverage") == "-"
    assert tab.summary_var.get()


def test_generated_outputs_are_not_offered_as_input(app, tmp_path):
    (tmp_path / "title.mp4").write_bytes(b"")
    (tmp_path / "title_si.mp4").write_bytes(b"")
    (tmp_path / "title_dub.mp4").write_bytes(b"")

    found = [Path(p).name for p in app.pace_tab._scan(str(tmp_path))]

    assert found == ["title.mp4"], "the tool's own exports are not source videos"


def test_the_stop_button_is_idle_until_something_plays(app):
    """The launcher treats any enabled button labelled "stop" as a running
    task, so an always-live stop button made returning to the menu warn about
    work that never started."""
    import main

    assert str(app.pace_tab.btn_stop["state"]) == "disabled"
    assert not main._enabled_stop_buttons(app), "the menu would refuse to return"


def test_returning_to_the_menu_is_not_blocked_by_an_idle_debug_tab(app):
    import main

    assert not main._app_has_running_tasks(app)


def test_the_waveform_area_exists_at_the_bottom(app):
    """The point of the tab: seeing where each side's sound starts and stops."""
    tab = app.pace_tab
    assert tab.canvas is not None
    assert int(tab.canvas.grid_info()["row"]) == 1
    assert int(tab.canvas.master.grid_info()["row"]) > int(tab.tree.master.grid_info()["row"])


def test_selecting_a_line_draws_both_tracks(app, root, tmp_path, monkeypatch):
    """Two lanes: the original on top, the dub under it, over the same window."""
    import numpy as np
    import soundfile as sf
    from tool_clonevoice_v2 import logic, proofread

    clone_dir = tmp_path / "t.clone"
    clone_dir.mkdir()
    source = clone_dir / logic.AUDIO16K_NAME
    sf.write(str(source), np.sin(np.arange(16000 * 10) / 20.0).astype("float32"), 16000)
    dub = clone_dir / "t.si.wav"
    track = np.zeros(24000 * 10, dtype="float32")
    track[24000 * 2:24000 * 3] = 0.5
    sf.write(str(dub), track, 24000)
    monkeypatch.setattr(logic, "clone_dir", lambda _v: clone_dir)
    monkeypatch.setattr(proofread, "cloned_track_path", lambda _v: dub)

    tab = app.pace_tab
    tab.current_video = str(tmp_path / "t.mp4")
    tab._loaded({
        "rows": [{"id": 1, "start": 2.0, "end": 6.0, "slot": 4.0, "clone": 1.0,
                  "coverage": 0.25, "exposed": 3.0, "speaker": "",
                  "src_text": "a", "tgt_text": "b", "src_chars": 1, "tgt_chars": 1}],
        "summary": {"lines": 1, "translated": 1, "measured": 1,
                    "median_coverage": 0.25, "under_exposed": 1, "under_half": 1,
                    "exposed_total": 3.0, "median_exposed": 3.0,
                    "median_char_ratio": 1.0},
    })
    root.update()

    assert set(tab.lanes) == {"src", "clone"}, "both sides must be drawn"
    assert tab.window is not None
    # The window opens before the line and runs past it, so the uncovered tail
    # and the line's own lead-in are both visible.
    window_start, window_end, start, end = tab.window
    assert window_start < start and window_end > end
    assert tab.canvas.find_all(), "canvas drawn"


def test_a_line_with_no_dub_still_draws_the_original(app, root, tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from tool_clonevoice_v2 import logic, proofread

    clone_dir = tmp_path / "t.clone"
    clone_dir.mkdir()
    sf.write(str(clone_dir / logic.AUDIO16K_NAME),
             np.zeros(16000 * 5, dtype="float32") + 0.1, 16000)
    monkeypatch.setattr(logic, "clone_dir", lambda _v: clone_dir)
    monkeypatch.setattr(proofread, "cloned_track_path", lambda _v: None)

    tab = app.pace_tab
    tab.current_video = str(tmp_path / "t.mp4")
    tab._load_wave({"start": 1.0, "end": 3.0, "clone": None})
    root.update()

    assert set(tab.lanes) == {"src"}
