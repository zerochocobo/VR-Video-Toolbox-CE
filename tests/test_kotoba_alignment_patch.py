from __future__ import annotations

import json

from tool_subtitle import logic


def write_config(path, data):
    (path / "config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_config(path):
    return json.loads((path / "config.json").read_text(encoding="utf-8-sig"))


def test_kotoba_alignment_heads_are_repaired_when_out_of_range(tmp_path):
    write_config(tmp_path, {"alignment_heads": [[7, 0], [10, 17]], "keep": "value"})
    logs = []

    repaired = logic.repair_kotoba_alignment_heads("kotoba", str(tmp_path), logs.append)

    assert repaired is True
    assert read_config(tmp_path)["alignment_heads"] == logic.KOTOBA_ALIGNMENT_HEADS
    assert read_config(tmp_path)["keep"] == "value"
    assert (tmp_path / "config.json.kotoba_alignment_heads.bak").exists()
    assert any("Patched kotoba alignment_heads" in message for message in logs)


def test_kotoba_alignment_heads_are_left_unchanged_when_valid(tmp_path):
    valid_heads = [[0, 0], [1, 19]]
    write_config(tmp_path, {"alignment_heads": valid_heads})

    repaired = logic.repair_kotoba_alignment_heads("kotoba", str(tmp_path))

    assert repaired is False
    assert read_config(tmp_path)["alignment_heads"] == valid_heads
    assert not (tmp_path / "config.json.kotoba_alignment_heads.bak").exists()


def test_alignment_repair_is_only_applied_to_kotoba(tmp_path):
    invalid_heads = [[25, 6]]
    write_config(tmp_path, {"alignment_heads": invalid_heads})

    repaired = logic.repair_kotoba_alignment_heads("large-v3", str(tmp_path))

    assert repaired is False
    assert read_config(tmp_path)["alignment_heads"] == invalid_heads


def test_kotoba_word_timestamps_are_enabled_by_default():
    assert logic.KOTOBA_BALANCED_OPTIONS["word_timestamps"] is True
    assert logic.KOTOBA_SCENE_OPTIONS["word_timestamps"] is True
    assert logic.MODEL_SCENE_OVERRIDES["kotoba"]["word_timestamps"] is True
