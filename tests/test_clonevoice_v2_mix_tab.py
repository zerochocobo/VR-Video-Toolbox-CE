"""The mix tab after the bandit-bed dubbing mode was withdrawn.

Dub mode built a "background bed" by separating the original speech away with
bandit and mixing the clone over it. It did not work on this material: measured
over the whole of HNVR-174, 279 of 369 lines came back within 3 dB of the
original, median attenuation -0.4 dB, so the bed still carried the dialogue the
dub was supposed to replace. Window length (9.6s -> 49.3s changed it by under
2 dB), input gain (+18 dB reached only -3.3 dB on a failing line), span
coverage (zero leaked frames outside spans) and the stereo downmix (0.6-1.1 dB)
were each ruled out. Ducking the original is now the only mode.

Bandit is still used to *measure* each line's level, which is a much weaker
demand than removing a voice; only bed generation is gone.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


def _gui_source() -> str:
    return Path("tool_clonevoice_v2/gui.py").read_text(encoding="utf-8")


# --- the withdrawal ---

def test_the_mix_panel_offers_no_mode_choice():
    """One path only. A leftover radio would let a user pick the broken mode."""
    source = _gui_source()
    assert "single_mix_mode_var" not in source
    assert "opt_mode_dub_spans" not in source


@pytest.mark.parametrize("name", ["_run_dub_task", "_dubbing_available",
                                  "_on_single_mix_mode_change"])
def test_the_dub_entry_points_are_gone(name):
    assert name not in _gui_source(), f"{name} still present"


@pytest.mark.parametrize("lang", ["en", "ja", "zh"])
def test_no_language_still_offers_the_dub_label(lang):
    raw = Path(f"i18n/{lang}.json").read_text(encoding="utf-8-sig")
    assert "opt_mode_dub_spans" not in raw
    assert json.loads(raw), "i18n file must stay valid JSON"


# --- geometry, kept from the regression that hid the buttons ---

@pytest.fixture
def mix_tab():
    """A realised window, so grid collisions actually show up.

    Adding the mode selector pushed the options and button rows down one each,
    and the button row landed on the log's row. Both widgets still constructed
    fine and every source check passed; the log simply covered the buttons,
    because it is the only stretching row. Only real geometry catches that.
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display: {exc}")
    root.geometry("1200x900")
    from tool_clonevoice_v2 import gui as gui_module

    app_cls = next(
        value for value in vars(gui_module).values()
        if isinstance(value, type) and hasattr(value, "_setup_ui")
    )
    app = app_cls(root)
    root.update_idletasks()
    root.update()
    yield app
    root.destroy()


def test_no_two_widgets_share_a_grid_row_on_the_mix_tab(mix_tab):
    seen: dict[int, str] = {}
    for child in mix_tab.tab_mix.grid_slaves():
        row = int(child.grid_info()["row"])
        assert row not in seen, (
            f"row {row} holds both {seen[row]} and {child.winfo_class()}"
        )
        seen[row] = child.winfo_class()


@pytest.mark.parametrize("button", ["single_mix_btn_start", "single_mix_btn_stop"])
def test_the_mix_buttons_are_actually_visible(mix_tab, button):
    widget = getattr(mix_tab, button)
    assert widget.winfo_ismapped(), f"{button} is not on screen"
    assert widget.winfo_height() > 1, f"{button} has no height"
    assert widget.winfo_width() > 1, f"{button} has no width"
