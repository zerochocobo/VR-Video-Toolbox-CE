from tool_clonevoice import proofread


def test_single_kana_without_explicit_metadata_is_not_assumed_complete():
    manifest = {
        "language": "ja",
        "target_language": "Chinese",
        "segments": [
            {"id": 1, "src_text": "好き", "tgt_text": "喜欢你"},
            {"id": 2, "src_text": "き", "tgt_text": ""},
        ],
    }
    assert proofread.effective_cleared_segment_ids(manifest) == set()


def test_normal_missing_translation_is_not_hidden():
    manifest = {
        "language": "ja",
        "target_language": "Chinese",
        "segments": [{"id": 1, "src_text": "好き", "tgt_text": ""}],
    }
    assert proofread.effective_cleared_segment_ids(manifest) == set()


def test_explicit_proofread_metadata_keeps_real_missing_translation_visible():
    manifest = {
        "language": "ja",
        "target_language": "Chinese",
        "proofread": {"cleared_ids": []},
        "segments": [{"id": 1, "src_text": "好き", "tgt_text": ""}],
    }
    assert proofread.effective_cleared_segment_ids(manifest) == set()


def test_valid_multichar_japanese_missing_translation_is_not_hidden():
    manifest = {
        "language": "ja",
        "target_language": "Chinese",
        "proofread": {"cleared_ids": []},
        "segments": [{"id": 7, "src_text": "お菓子を", "tgt_text": ""}],
    }
    assert proofread.effective_cleared_segment_ids(manifest) == set()
