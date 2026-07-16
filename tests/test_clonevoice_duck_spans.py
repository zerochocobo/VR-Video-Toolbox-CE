from tool_clonevoice.omnivoice_backend import _duck_spans_for_segments


def test_duck_spans_exclude_ai_or_user_deleted_translation_lines():
    segments = [
        {"start": 1.0, "end": 2.0, "speaker": "A", "tgt_text": "保留"},
        {"start": 3.0, "end": 4.0, "speaker": "A", "tgt_text": ""},
        {"start": 5.0, "end": 6.0, "speaker": "A", "tgt_text": "   "},
    ]
    assert _duck_spans_for_segments(segments, "tgt_text") == [{"start": 1.0, "end": 2.0}]


def test_duck_spans_exclude_skipped_speakers():
    segments = [
        {"start": 1.0, "end": 2.0, "speaker": "A", "tgt_text": "A"},
        {"start": 3.0, "end": 4.0, "speaker": "B", "tgt_text": "B"},
    ]
    assert _duck_spans_for_segments(segments, "tgt_text", {"B"}) == [{"start": 1.0, "end": 2.0}]

