from tool_subtitle import logic


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _entries():
    return {
        10: {"text": "今日は天気がいい"},
        20: {"text": "広角でいいの？"},
        30: {"text": "ああー"},
    }


def test_source_correction_accepts_sparse_changes_and_preserves_omitted_lines(monkeypatch):
    monkeypatch.setattr(logic, "_load_correct_prompt_template", lambda _adult: logic.DEFAULT_CORRECT_PROMPT)
    client = FakeClient(["<2>公園でいいの？</2>\n<3></3>"])
    entries = _entries()

    changed, deleted = logic.correct_entries(
        client, entries, "Japanese", 10000, True, 3, lambda _message: None, None,
    )

    assert len(client.prompts) == 1
    assert entries[10]["text"] == "今日は天気がいい"
    assert entries[20]["text"] == "公園でいいの？"
    assert 30 not in entries
    assert changed == 1
    assert deleted == {30}


def test_source_correction_treats_empty_sparse_response_as_no_changes(monkeypatch):
    monkeypatch.setattr(logic, "_load_correct_prompt_template", lambda _adult: logic.DEFAULT_CORRECT_PROMPT)
    client = FakeClient(["START\n\nEND"])
    entries = _entries()

    changed, deleted = logic.correct_entries(
        client, entries, "Japanese", 10000, True, 3, lambda _message: None, None,
    )

    assert len(client.prompts) == 1
    assert entries == _entries()
    assert changed == 0
    assert deleted == set()


def test_source_correction_accepts_natural_language_no_change_response(monkeypatch):
    monkeypatch.setattr(logic, "_load_correct_prompt_template", lambda _adult: logic.DEFAULT_CORRECT_PROMPT)
    client = FakeClient(["修正はありません。"])
    entries = _entries()
    messages = []

    changed, deleted = logic.correct_entries(
        client, entries, "Japanese", 10000, True, 3, messages.append, None,
    )

    assert len(client.prompts) == 1
    assert entries == _entries()
    assert changed == 0
    assert deleted == set()
    assert any("reported no changes" in message for message in messages)


def test_source_correction_still_retries_ambiguous_untagged_response(monkeypatch):
    monkeypatch.setattr(logic, "_load_correct_prompt_template", lambda _adult: logic.DEFAULT_CORRECT_PROMPT)
    client = FakeClient(["字幕を確認しました。", "START\n\nEND"])
    entries = _entries()
    messages = []

    changed, deleted = logic.correct_entries(
        client, entries, "Japanese", 10000, True, 2, messages.append, None,
    )

    assert len(client.prompts) == 2
    assert entries == _entries()
    assert changed == 0
    assert deleted == set()
    assert any("unsupported response format" in message for message in messages)


def test_translation_chunk_still_retries_missing_ids():
    client = FakeClient(["<1>甲</1>", "<1>甲</1>\n<2>乙</2>"])
    messages = []

    result = logic._llm_chunk_pass(
        client,
        {10: "a", 20: "b"},
        "{subtitles}",
        {},
        2,
        messages.append,
        None,
    )

    assert len(client.prompts) == 2
    assert result == {10: "甲", 20: "乙"}
    assert any("missing" in message for message in messages)


def test_legacy_correction_prompt_gets_sparse_output_override(monkeypatch, tmp_path):
    legacy_prompt = "Return every subtitle unchanged when it is correct.\n{subtitles}\n/no_think\n"
    (tmp_path / "asr_correct_prompt.txt").write_text(legacy_prompt, encoding="utf-8-sig")
    monkeypatch.setattr(logic, "get_config_dir", lambda: str(tmp_path))

    loaded = logic._load_correct_prompt_template(True)

    assert "Return every subtitle unchanged when it is correct." in loaded
    assert "IMPORTANT OUTPUT OVERRIDE — SPARSE_CHANGES_ONLY" in loaded
    assert "Omit unchanged" in loaded
    assert loaded.rstrip().endswith("/no_think")
