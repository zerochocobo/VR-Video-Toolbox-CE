from pathlib import Path

import numpy as np

from tool_subtitle.debug_analyzer import (
    build_peak_envelope,
    find_debug_sessions,
    find_media_subtitles,
    parse_srt,
    waveform_display_gain,
)


def test_parse_srt_accepts_bom_multiline_and_comma_timestamp(tmp_path: Path):
    path = tmp_path / "sample.srt"
    path.write_text(
        "\ufeff1\n00:00:01,250 --> 00:00:03,500\n第一行\n第二行\n\n"
        "2\n00:00:04.000 --> 00:00:05.000\nnext\n",
        encoding="utf-8",
    )
    entries = parse_srt(path)
    assert len(entries) == 2
    assert entries[0].start == 1.25
    assert entries[0].end == 3.5
    assert entries[0].text == "第一行\n第二行"


def test_peak_envelope_distinguishes_silence_and_speech():
    samples = np.concatenate([np.zeros(1000, dtype=np.float32), np.ones(1000, dtype=np.float32) * 0.8])
    low, high = build_peak_envelope(samples, 20)
    assert np.max(np.abs(high[:10])) == 0
    assert np.max(high[10:]) >= 0.79
    assert len(low) == len(high) == 20


def test_waveform_display_gain_is_at_least_double_and_boosts_quiet_speech():
    silence = np.zeros(20, dtype=np.float32)
    loud_low = np.full(20, -0.8, dtype=np.float32)
    loud_high = np.full(20, 0.8, dtype=np.float32)
    quiet_low = np.full(20, -0.2, dtype=np.float32)
    quiet_high = np.full(20, 0.2, dtype=np.float32)

    assert waveform_display_gain(silence, silence) == 2.0
    assert waveform_display_gain(loud_low, loud_high) == 2.0
    assert waveform_display_gain(quiet_low, quiet_high) > 4.0
    assert waveform_display_gain(quiet_low, quiet_high) <= 12.0


def test_find_debug_sessions_requires_wav_and_sorts_newest(tmp_path: Path):
    old = tmp_path / "old_debug"
    new = tmp_path / "nested" / "new_debug"
    empty = tmp_path / "empty_debug"
    old.mkdir()
    new.mkdir(parents=True)
    empty.mkdir()
    (old / "old.wav").write_bytes(b"RIFF")
    (new / "new.wav").write_bytes(b"RIFF")
    old.touch()
    new.touch()
    sessions = find_debug_sessions(tmp_path)
    assert set(sessions) == {old, new}
    assert empty not in sessions


def test_find_debug_sessions_accepts_selected_debug_directory_itself(tmp_path: Path):
    session = tmp_path / "movie_debug"
    session.mkdir()
    (session / "movie.wav").write_bytes(b"RIFF")

    assert find_debug_sessions(session) == [session]


def test_empty_base_directory_does_not_scan_current_working_directory():
    # The analyzer passes None for an empty GUI directory; this is intentionally
    # different from scanning Path.cwd(), which may contain a large repository.
    assert find_debug_sessions(None) == []


def test_find_media_subtitles_prioritizes_plain_and_jp_srt(tmp_path: Path):
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"")
    for name in ("movie.en.srt", "movie.jp.srt", "movie.srt", "movie.raw.srt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert [p.name for p in find_media_subtitles(media)] == ["movie.srt", "movie.jp.srt", "movie.en.srt"]


def test_find_debug_sessions_discovers_clone_directories(tmp_path: Path):
    clone = tmp_path / "movie.clone"
    clone.mkdir()
    (clone / "audio16k.wav").write_bytes(b"RIFF")
    (clone / "source.srt").write_text("", encoding="utf-8")
    assert clone in find_debug_sessions(tmp_path)
