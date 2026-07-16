from pathlib import Path

from tool_subtitle.logic import SubtitleGenerator


def test_retain_debug_audio_moves_wav_into_debug_directory(tmp_path: Path):
    audio = tmp_path / "movie.asr.wav"
    output = tmp_path / "movie.jp.srt"
    audio.write_bytes(b"wave-data")
    generator = object.__new__(SubtitleGenerator)
    generator.log_callback = lambda _message: None

    destination = generator.retain_debug_audio(str(audio), str(output))

    assert destination == str(tmp_path / "movie_debug" / "movie.wav")
    assert Path(destination).read_bytes() == b"wave-data"
    assert not audio.exists()

