"""Real AAC export plus synthesis integration, without loading a voice model."""
import json
import shutil
import subprocess
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest

from tool_si import dlna_audio, logic as si
from utils.si_prepared_audio import find_prepared_audio, prepared_paths


@pytest.fixture
def movie(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required")
    video = tmp_path / "movie.mp4"
    voice = video.with_suffix(".si.wav")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=96x64:r=10:d=3",
        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video),
    ], check=True, capture_output=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "sine=frequency=880:sample_rate=24000:duration=1", str(voice),
    ], check=True, capture_output=True)
    return video, voice


@pytest.mark.parametrize("dub", [False, True])
def test_real_aac_export_covers_source_and_reuses_current_mix(movie, dub):
    video, voice = movie
    if dub:
        si.write_duck_key_wav(video.with_suffix(".si.duck.wav"), [{"start": 0, "end": 1}], 1, 24000)
    output = Path(dlna_audio.prepare_audio(video))
    filt, duck = dlna_audio.mix_inputs(video, voice)
    assert find_prepared_audio(video, voice, duck, filt) == output
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output),
    ]))
    stream, = info["streams"]
    assert (stream["codec_name"], stream["profile"], stream["sample_rate"], stream["channels"]) == ("aac", "LC", "48000", 2)
    assert abs(float(info["format"]["duration"]) - 3) < 0.06
    assert output.read_bytes().index(b"moov") < output.read_bytes().index(b"mdat")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
                   check=True, capture_output=True)
    with patch.object(dlna_audio.subprocess, "Popen", side_effect=AssertionError("No re-encode")):
        assert dlna_audio.prepare_audio(video) == str(output)


def test_failed_replacement_preserves_previous_audio_and_wav(movie):
    video, voice = movie
    audio = Path(dlna_audio.prepare_audio(video))
    _, metadata = prepared_paths(video)
    previous_audio, previous_metadata = audio.read_bytes(), metadata.read_bytes()
    voice.write_bytes(b"broken replacement wav")
    with pytest.raises(RuntimeError, match="encoding failed"):
        dlna_audio.prepare_audio(video)
    assert audio.read_bytes() == previous_audio
    assert metadata.read_bytes() == previous_metadata
    assert voice.read_bytes() == b"broken replacement wav"
    assert not list(video.parent.glob(".*.tmp.*"))
    filt, duck = dlna_audio.mix_inputs(video, voice)
    assert find_prepared_audio(video, voice, duck, filt) is None


def test_cancellation_during_encoder_stops_process_and_cleans_temp(movie):
    video, _ = movie
    stop = Event()
    class RunningEncoder:
        terminated = False
        def wait(self, timeout=None):
            stop.set()
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        def poll(self):
            return 1 if self.terminated else None
    process = RunningEncoder()
    def terminate(p):
        p.terminated = True
    with patch.object(dlna_audio.subprocess, "Popen", return_value=process), \
         patch.object(si, "_terminate_process", side_effect=terminate):
        with pytest.raises(RuntimeError, match="Stopped by user"):
            dlna_audio.prepare_after_synthesis(video, stop_event=stop)
    assert process.terminated
    assert not prepared_paths(video)[1].exists()
    assert not list(video.parent.glob(".*.tmp.*"))


@pytest.mark.parametrize("options,expected", [({}, 1), ({"max_entries": 1}, 0),
                                             ({"start_seconds": 0.1, "duration_seconds": 0.8}, 0)])
def test_only_full_paired_si_synthesis_prepares_dlna(tmp_path, options, expected):
    import numpy as np
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    with (patch.object(si, "check_model_files", return_value=True),
          patch.object(si, "_release_cuda_cache"),
          patch.object(si, "_generate_audio_batch", return_value=[(np.zeros(24000), 24000)]),
          patch.object(si, "prepare_dlna_audio") as prepare):
        si.subtitle_to_audio(subtitle, None, "English", "Ryan", "models", tts_model=object(), **options)
    assert prepare.call_count == expected


def test_batch_skip_backfills_without_loading_models(tmp_path):
    from tool_clonevoice_v2 import logic
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    voice = video.with_suffix(".si.wav")
    voice.write_bytes(b"existing")
    with (patch.object(logic, "synthesis_is_current", return_value=True),
          patch.object(si, "prepare_dlna_audio") as prepare,
          patch("tool_clonevoice_v2.backend.load_model", side_effect=AssertionError("No TTS load"))):
        assert logic.run_batch([video], models_root="models") == [str(voice)]
    prepare.assert_called_once_with(video, voice, log_callback=print, stop_event=None)


def test_clone_export_prepares_only_after_voice_and_duck_key_are_saved(tmp_path):
    import numpy as np
    from tool_clonevoice_v2 import logic, backend
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    logic.save_manifest(video, {"language": "ja", "target_language": "Chinese", "segments": [
        {"id": 1, "start": 0.0, "end": 1.0, "tgt_text": "Hello"},
    ]})
    calls = []
    def synthesize(model, source, segments, output, **kwargs):
        si.write_wav_mono(output, np.zeros(backend.SAMPLE_RATE), backend.SAMPLE_RATE)
        return output
    def prepare(source, voice, **kwargs):
        assert Path(voice).is_file()
        assert Path(si.default_si_duck_key_path(voice)).is_file()
        assert logic.load_manifest(source)["synthesis"]["signature"]
        calls.append(source)
    with (patch.object(backend, "synthesize_manifest", side_effect=synthesize),
          patch.object(logic, "_release_cuda_cache"), patch.object(logic, "log_memory"),
          patch.object(si, "prepare_dlna_audio", side_effect=prepare)):
        logic.run_synthesize(video, models_root="models", model=object())
    assert calls == [video]


@pytest.mark.parametrize("module_name", ["tool_clonevoice", "tool_clonevoice_v2"])
def test_single_clone_skip_backfills_without_synthesis(tmp_path, module_name):
    import importlib
    single = importlib.import_module(module_name + ".single_clone")
    logic = importlib.import_module(module_name + ".logic")
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    voice = video.with_suffix(".si.wav")
    voice.write_bytes(b"voice")
    from contextlib import nullcontext
    current = patch.object(logic, "synthesis_is_current", return_value=True) if module_name.endswith("v2") else nullcontext()
    with (current, patch.object(si, "prepare_dlna_audio") as prepare,
          patch.object(logic, "run_synthesize", side_effect=AssertionError("No synthesis")),
          patch.object(single, "ensure_translated", side_effect=AssertionError("No translation"))):
        result = single.translate_and_synthesize([str(video)], target_language="Chinese", models_root="models")
    assert result["skipped"] == [str(voice)]
    prepare.assert_called_once()
