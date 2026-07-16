from pathlib import Path
from types import SimpleNamespace


def test_home_has_separate_subtitle_and_voice_pages():
    source = Path("main.py").read_text(encoding="utf-8-sig")
    assert "('voice', self._build_page_voice)" in source
    assert "('voice', 'nav_voice')" in source
    assert "'voice': '\\uE720'" in source


def test_voice_page_orders_clone_before_simultaneous_interpretation():
    source = Path("main.py").read_text(encoding="utf-8-sig")
    start = source.index("    def _build_page_voice")
    end = source.index("    def _build_page_dlna", start)
    block = source[start:end]
    assert block.index("btn_clonevoice") < block.index("btn_si_voice")


def test_subtitle_page_no_longer_contains_voice_cards():
    source = Path("main.py").read_text(encoding="utf-8-sig")
    start = source.index("    def _build_page_subtitle")
    end = source.index("    def _build_page_voice", start)
    block = source[start:end]
    assert "btn_clonevoice" not in block
    assert "btn_si_voice" not in block


def test_mosaic_page_uses_global_settings_group_and_engine_radios():
    source = Path("main.py").read_text(encoding="utf-8-sig")
    start = source.index("    def _build_page_mosaic")
    end = source.index("    def _build_page_subtitle", start)
    block = source[start:end]

    assert "grp_global_mosaic_settings" in block
    assert block.index("lbl_encode_profile") < block.index("lbl_engine")
    assert "ttk.Radiobutton(" in block
    assert "engine_combo" not in block


def test_one_click_page_no_longer_builds_encode_profile_footer():
    source = Path("one_click/main.py").read_text(encoding="utf-8-sig")

    assert "create_settings_bar(self.notebook.footer())" not in source
    assert "def create_settings_bar" not in source


def test_home_encode_profile_selection_persists_global_setting(monkeypatch):
    import main

    applied = []
    launcher = SimpleNamespace(
        _encode_profile_display_to_key={"Fast": "fast_quality"},
        _encode_profile_var=SimpleNamespace(get=lambda: "Fast"),
    )
    monkeypatch.setattr(main.encode_config, "apply_encode_profile", applied.append)

    main.VRVideoToolboxLauncher._on_encode_profile_change(launcher)

    assert applied == ["fast_quality"]


def test_native_engine_hides_custom_arguments_controls():
    import main

    custom_frame = SimpleNamespace(pack_forget=lambda: setattr(custom_frame, "hidden", True))
    custom_frame.hidden = False
    launcher = SimpleNamespace(
        _selected_engine=lambda: "native_gpu",
        _custom_args_frame=custom_frame,
    )

    main.VRVideoToolboxLauncher._update_custom_args_display(launcher)

    assert custom_frame.hidden


def test_external_engine_restores_custom_arguments_controls(monkeypatch):
    import main

    state = {"manager": "", "text": None}
    custom_frame = SimpleNamespace(
        winfo_manager=lambda: state["manager"],
        pack=lambda **_kwargs: state.__setitem__("manager", "pack"),
    )
    label = SimpleNamespace(config=lambda **kwargs: state.__setitem__("text", kwargs["text"]))
    launcher = SimpleNamespace(
        _selected_engine=lambda: "lada",
        _custom_args_frame=custom_frame,
        _lbl_custom_args=label,
    )
    monkeypatch.setattr(main.app_config, "get_custom_args", lambda _engine: "--quality high")

    main.VRVideoToolboxLauncher._update_custom_args_display(launcher)

    assert state == {"manager": "pack", "text": "[--quality high]"}
