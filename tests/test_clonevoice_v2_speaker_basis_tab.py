"""The per-speaker basis tab, and that it is actually reachable.

a265bda ("Refactor v2 voice clone to sentence references") removed the three
lines that registered this tab but left its ~500-line body and all of its
handlers in the file. Everything still imported and every unit test passed --
the tab was simply not in the notebook, so choosing a speaker's reference by
ear was impossible and the automatic timbre anchor was the only path. Only a
realised window catches that class of regression.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def app():
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
    instance = app_cls(root)
    root.update_idletasks()
    root.update()
    yield instance
    root.destroy()


def test_the_speaker_basis_tab_is_in_the_notebook(app):
    """The page host is a SideNavigation rail, not a ttk.Notebook, so ask it
    for the page rather than for a tab list."""
    from tool_clonevoice_v2.gui import get_text

    assert app.notebook.index(app.tab_multi_clone) >= 0
    assert app.notebook.tab(app.tab_multi_clone, "text") == get_text("tab_multi_clone")


def test_the_tab_body_was_actually_built(app):
    assert app.tab_multi_clone.winfo_children(), "tab registered but never populated"


@pytest.mark.parametrize(
    "widget",
    ["multi_clone_btn_play_basis", "multi_clone_speaker_tree",
     "multi_clone_btn_select_basis"],
)
def test_the_audition_controls_are_reachable(app, widget):
    """Reachable, not merely constructed: a widget can exist and never be shown.

    The tab is a wizard, and the speaker list lives on step 2, so this walks to
    that step the way the user does rather than asserting on the start state.
    """
    target = getattr(app, widget, None)
    assert target is not None, f"{widget} is missing"
    app.notebook.select(app.tab_multi_clone)
    app._show_multi_clone_step(1)
    app.root.update()
    assert target.winfo_ismapped(), f"{widget} exists but is not visible"


def test_no_two_widgets_share_a_grid_row(app):
    seen: dict[int, str] = {}
    for child in app.tab_multi_clone.grid_slaves():
        row = int(child.grid_info()["row"])
        assert row not in seen, (
            f"row {row} holds both {seen[row]} and {child.winfo_class()}"
        )
        seen[row] = child.winfo_class()


def test_the_speaker_count_offers_one_and_defaults_to_it(app):
    """"Auto" was the default and is what let pyannote guess -- the same guess
    that produces the fragment clusters this tab then has to ignore. A single
    speaker is both the common case and the single-voice refined-clone path,
    so it is the default and "auto" is gone."""
    assert app.multi_clone_num_var.get() == "1"
    assert list(app.multi_clone_num_combo["values"]) == [str(n) for n in range(1, 8)]
    assert None not in app.multi_clone_num_map.values()
    assert app.multi_clone_num_map[app.multi_clone_num_var.get()] == 1


def test_the_tab_is_named_for_refined_cloning(app):
    from tool_clonevoice_v2.gui import get_text

    assert app.notebook.tab(app.tab_multi_clone, "text") == get_text("tab_multi_clone")
    # ...and the name is no longer about speaker count, since one speaker is
    # now the default path through it.
    assert "多人" not in get_text("tab_multi_clone")


# --- candidate previews must not route through the OmniVoice stub -----------

STUB_ONLY_HELPERS = (
    "build_candidate_target_sample_job",
    "finish_candidate_target_sample_jobs",
    "generate_candidate_translated_previews_with_model",
)


def test_every_stub_helper_really_does_raise():
    """These are the v2 compatibility guards; if one ever becomes real, the
    check below stops being meaningful and should be revisited."""
    import pytest as _pytest

    from tool_clonevoice_v2 import omnivoice_backend as ov

    for name in ("_generate_target_reference_takes_with_model",
                 "process_target_reference_batch"):
        with _pytest.raises(RuntimeError, match="reference audio directly"):
            getattr(ov, name)()


@pytest.mark.parametrize("helper", STUB_ONLY_HELPERS)
def test_no_gui_path_reaches_the_omnivoice_stub(helper):
    """Auditioning a speaker's candidates called OmniVoice's two-pass
    "target reference take, then preview" flow, which v2 replaced with a stub
    that raises: "IndexTTS-2.5 uses reference audio directly...". IndexTTS
    clones straight from the candidate WAV, so one pass is all it needs.
    """
    from pathlib import Path

    source = Path("tool_clonevoice_v2/gui.py").read_text(encoding="utf-8")
    assert helper not in source, f"{helper} routes through the raising stub"


def test_both_tabs_preview_candidates_the_same_way():
    from pathlib import Path

    source = Path("tool_clonevoice_v2/gui.py").read_text(encoding="utf-8")
    assert source.count("sc.generate_indextts_candidate_previews(") == 3


def test_nothing_in_the_tab_still_calls_itself_multi_speaker():
    """The speaker count defaults to 1 and the tab handles a single voice as
    well as several, so labelling it "multi-speaker" tells a single-speaker
    user it is not for them."""
    import inspect
    import json
    import re
    from pathlib import Path

    from tool_clonevoice_v2 import gui as gui_module

    source = inspect.getsource(gui_module.__dict__[
        next(n for n, v in vars(gui_module).items()
             if isinstance(v, type) and hasattr(v, "_setup_multi_clone_tab"))
    ]._setup_multi_clone_tab)
    keys = set(re.findall(r'get_text\("([^"]+)"\)', source))
    assert keys, "could not read the tab's strings"

    banned = {"zh": ["多人"], "en": ["multi-speaker"], "ja": ["複数話者"]}
    for lang, words in banned.items():
        data = json.loads(
            Path(f"i18n/{lang}.json").read_text(encoding="utf-8-sig")
        )
        flat: dict[str, str] = {}

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value) if isinstance(value, dict) else flat.__setitem__(key, value)

        walk(data)
        for key in sorted(keys):
            text = str(flat.get(key, ""))
            for word in words:
                assert word.lower() not in text.lower(), f"{lang}/{key}: {text}"
