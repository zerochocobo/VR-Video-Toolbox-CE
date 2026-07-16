from pathlib import Path


def test_all_clonevoice_source_language_controls_default_to_japanese():
    source = Path("tool_clonevoice/gui.py").read_text(encoding="utf-8-sig")
    assert 'self.src_lang_var = tk.StringVar(value=get_text("opt_lang_ja"))' in source
    assert 'self.single_clone_src_lang_var = tk.StringVar(value=get_text("opt_lang_ja"))' in source
    assert 'self.multi_clone_src_lang_var = tk.StringVar(value=get_text("opt_lang_ja"))' in source
    assert 'src_lang_var = tk.StringVar(value=get_text("opt_lang_auto"))' not in source

