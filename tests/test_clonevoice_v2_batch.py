from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from tool_clonevoice_v2 import logic
from tool_clonevoice_v2.gui import ClonevoiceToolsApp


def test_batch_scan_respects_recursive_option(tmp_path: Path) -> None:
    (tmp_path / "root.mp4").write_bytes(b"video")
    (tmp_path / "root_SI.mp4").write_bytes(b"generated")
    child = tmp_path / "child"
    child.mkdir()
    (child / "nested.mkv").write_bytes(b"video")

    direct = ClonevoiceToolsApp._scan_clone_batch_videos(
        object(), str(tmp_path), recursive=False
    )
    recursive = ClonevoiceToolsApp._scan_clone_batch_videos(
        object(), str(tmp_path), recursive=True
    )

    assert [Path(path).name for path in direct] == ["root.mp4"]
    assert [Path(path).name for path in recursive] == ["nested.mkv", "root.mp4"]


def test_batch_pipeline_skips_diarization_and_reference_extraction(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "movie.si.wav"
    holder: list[object] = []

    with (
        patch("tool_si.logic.default_si_audio_path", return_value=str(output)),
        patch("tool_clonevoice_v2.logic.run_transcribe_diarize") as transcribe,
        patch("tool_clonevoice_v2.logic.run_extract_references") as extract,
        patch("tool_clonevoice_v2.logic.run_translate") as translate,
        patch("tool_clonevoice_v2.logic.run_synthesize", return_value=str(output)) as synthesize,
        patch("tool_clonevoice_v2.logic.release_model_holder") as release,
    ):
        result = logic.run_full(
            video,
            models_root=str(tmp_path / "models"),
            target_language="Chinese",
            skip_existing=False,
            model_holder=holder,
        )

    assert result == str(output)
    assert transcribe.call_args.kwargs["skip_diarization"] is True
    assert transcribe.call_args.kwargs["diarize_backend"] == "none"
    assert transcribe.call_args.kwargs["num_speakers"] is None
    assert transcribe.call_args.kwargs["skip_existing"] is False
    extract.assert_not_called()
    assert translate.call_args.kwargs["skip_existing"] is False
    synthesize.assert_called_once()
    release.assert_called_once_with(holder)


def test_batch_pipeline_loads_indextts_once_for_all_videos(tmp_path: Path) -> None:
    videos = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    for video in videos:
        video.write_bytes(b"video")
    shared_model = object()
    holder: list[object] = []

    def output_path(video: str) -> str:
        return str(Path(video).with_suffix(".si.wav"))

    def synthesize(video, **kwargs) -> str:
        assert kwargs["model"] is shared_model
        return output_path(str(video))

    with (
        patch("tool_si.logic.default_si_audio_path", side_effect=output_path),
        patch("tool_clonevoice_v2.logic.run_transcribe_diarize") as transcribe,
        patch("tool_clonevoice_v2.logic.run_translate") as translate,
        patch("tool_clonevoice_v2.backend.load_model", return_value=shared_model) as load_model,
        patch("tool_clonevoice_v2.logic.run_synthesize", side_effect=synthesize) as run_synth,
        patch("tool_clonevoice_v2.logic.release_model_holder") as release,
    ):
        outputs = logic.run_batch(
            videos,
            models_root=str(tmp_path / "models"),
            target_language="Chinese",
            skip_existing=False,
            model_holder=holder,
        )

    assert outputs == [output_path(str(video)) for video in videos]
    assert transcribe.call_count == 2
    assert all(call.kwargs["skip_diarization"] is True for call in transcribe.call_args_list)
    assert translate.call_count == 2
    assert release.call_count == 2
    load_model.assert_called_once()
    assert run_synth.call_count == 2
    assert holder == [shared_model]


def test_batch_completes_each_directory_before_starting_the_next(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    videos = [
        first_dir / "one.mp4",
        first_dir / "one_more.mp4",
        second_dir / "two.mp4",
    ]
    for video in videos:
        video.write_bytes(b"video")

    events: list[str] = []
    models = [object(), object()]
    holder: list[object] = []

    def output_path(video: str) -> str:
        return str(Path(video).with_suffix(".si.wav"))

    def transcribe(video, **kwargs):
        events.append(f"transcribe:{Path(video).parent.name}")

    def translate(video, **kwargs):
        events.append(f"translate:{Path(video).parent.name}")

    def load_model(*args, **kwargs):
        events.append("load")
        return models.pop(0)

    def synthesize(video, **kwargs) -> str:
        events.append(f"synthesize:{Path(video).parent.name}")
        return output_path(str(video))

    def release(holder_arg):
        if holder_arg is not None:
            holder_arg.clear()

    with (
        patch("tool_si.logic.default_si_audio_path", side_effect=output_path),
        patch("tool_clonevoice_v2.logic.run_transcribe_diarize", side_effect=transcribe),
        patch("tool_clonevoice_v2.logic.run_translate", side_effect=translate),
        patch("tool_clonevoice_v2.backend.load_model", side_effect=load_model),
        patch("tool_clonevoice_v2.logic.run_synthesize", side_effect=synthesize),
        patch("tool_clonevoice_v2.logic.release_model_holder", side_effect=release),
    ):
        outputs = logic.run_batch(
            videos,
            models_root=str(tmp_path / "models"),
            target_language="Chinese",
            skip_existing=True,
            model_holder=holder,
        )

    assert outputs == [output_path(str(video)) for video in videos]
    assert events == [
        "transcribe:first",
        "translate:first",
        "transcribe:first",
        "translate:first",
        "load",
        "synthesize:first",
        "synthesize:first",
        "transcribe:second",
        "translate:second",
        "load",
        "synthesize:second",
    ]


def test_resume_reuses_audio_manifest_and_subtitles_before_synthesis(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    cdir = logic.clone_dir(video)
    cdir.mkdir()
    (cdir / logic.AUDIO16K_NAME).write_bytes(b"cached audio")
    (cdir / logic.SOURCE_SRT_NAME).write_text("cached source", encoding="utf-8")
    (cdir / logic.TRANSLATED_SRT_NAME).write_text("cached translation", encoding="utf-8")
    logic.save_manifest(video, {
        "video": str(video),
        "language": "ja",
        "target_language": "Chinese",
        "segments": [{
            "id": 1,
            "start": 0.0,
            "end": 2.0,
            "src_text": "こんにちは",
            "tgt_text": "你好",
        }],
    })
    output = tmp_path / "movie.si.wav"
    messages: list[str] = []

    with (
        patch("tool_si.logic.default_si_audio_path", return_value=str(output)),
        patch("tool_clonevoice_v2.whisperx_backend.extract_audio_16k") as extract_audio,
        patch("tool_clonevoice_v2.segment_engine.transcribe") as transcribe,
        patch("tool_subtitle.logic.make_llm_client") as make_client,
        patch("tool_clonevoice_v2.logic.run_synthesize", return_value=str(output)) as synthesize,
    ):
        result = logic.run_full(
            video,
            models_root=str(tmp_path / "models"),
            target_language="Chinese",
            skip_existing=True,
            log=messages.append,
        )

    assert result == str(output)
    extract_audio.assert_not_called()
    transcribe.assert_not_called()
    make_client.assert_not_called()
    synthesize.assert_called_once()
    assert any("skipping ASR" in message for message in messages)
    assert any("skipping translation API" in message for message in messages)
    assert logic.translation_checkpoint_ready(video, "Chinese") is True
    assert logic.translation_checkpoint_ready(video, "English") is False


def test_resumed_directories_reuse_loaded_indextts_when_asr_is_cached(tmp_path: Path) -> None:
    videos = [tmp_path / "first" / "one.mp4", tmp_path / "second" / "two.mp4"]
    for video in videos:
        video.parent.mkdir()
        video.write_bytes(b"video")
        logic.save_manifest(video, {
            "video": str(video),
            "language": "ja",
            "target_language": "Chinese",
            "segments": [{
                "id": 1,
                "start": 0.0,
                "end": 2.0,
                "src_text": "こんにちは",
                "tgt_text": "你好",
            }],
        })

    shared_model = object()

    def output_path(video: str) -> str:
        return str(Path(video).with_suffix(".si.wav"))

    with (
        patch("tool_si.logic.default_si_audio_path", side_effect=output_path),
        patch("tool_clonevoice_v2.logic.run_transcribe_diarize"),
        patch("tool_clonevoice_v2.logic.run_translate"),
        patch("tool_clonevoice_v2.backend.load_model", return_value=shared_model) as load_model,
        patch(
            "tool_clonevoice_v2.logic.run_synthesize",
            side_effect=lambda video, **kwargs: output_path(str(video)),
        ) as synthesize,
    ):
        outputs = logic.run_batch(
            videos,
            models_root=str(tmp_path / "models"),
            target_language="Chinese",
            skip_existing=True,
        )

    assert outputs == [output_path(str(video)) for video in videos]
    load_model.assert_called_once()
    assert all(call.kwargs["model"] is shared_model for call in synthesize.call_args_list)


def test_existing_translated_srt_restores_manifest_without_ai_calls(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    cdir = logic.clone_dir(video)
    cdir.mkdir()
    (cdir / logic.AUDIO16K_NAME).write_bytes(b"cached audio")
    logic.save_manifest(video, {
        "video": str(video),
        "language": "ja",
        "target_language": "Chinese",
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 2.0,
                "src_text": "こんにちは",
                "tgt_text": "",
            },
            {
                "id": 2,
                "start": 3.0,
                "end": 5.0,
                "src_text": "さようなら",
                "tgt_text": "",
            },
        ],
    })
    (cdir / logic.TRANSLATED_SRT_NAME).write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n你好\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\n再见\n",
        encoding="utf-8-sig",
    )
    output = tmp_path / "movie.si.wav"
    messages: list[str] = []

    assert logic.translation_checkpoint_ready(video, "Chinese") is True
    with (
        patch("tool_si.logic.default_si_audio_path", return_value=str(output)),
        patch("tool_clonevoice_v2.whisperx_backend.extract_audio_16k") as extract_audio,
        patch("tool_clonevoice_v2.segment_engine.transcribe") as transcribe,
        patch("tool_subtitle.logic.make_llm_client") as make_client,
        patch("tool_subtitle.logic.correct_entries") as correct_entries,
        patch("tool_subtitle.logic.translate_entries") as translate_entries,
        patch("tool_clonevoice_v2.logic.run_synthesize", return_value=str(output)),
    ):
        result = logic.run_full(
            video,
            models_root=str(tmp_path / "models"),
            target_language="Chinese",
            skip_existing=True,
            source_correction=True,
            log=messages.append,
        )

    assert result == str(output)
    extract_audio.assert_not_called()
    transcribe.assert_not_called()
    make_client.assert_not_called()
    correct_entries.assert_not_called()
    translate_entries.assert_not_called()
    restored = logic.load_manifest(video)
    assert restored is not None
    assert [item["tgt_text"] for item in restored["segments"]] == ["你好", "再见"]
    assert any("skipping source proofreading and translation API" in message for message in messages)


def test_resume_reuses_audio_but_rebuilds_missing_transcription(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    cdir = logic.clone_dir(video)
    cdir.mkdir()
    (cdir / logic.AUDIO16K_NAME).write_bytes(b"cached audio")

    with (
        patch("tool_clonevoice_v2.whisperx_backend.extract_audio_16k") as extract_audio,
        patch("tool_clonevoice_v2.whisperx_backend.resolve_asr_device", return_value=("cpu", "int8")),
        patch("tool_clonevoice_v2.whisperx_backend.resolve_device", return_value=("cpu", "float32")),
        patch("tool_clonevoice_v2.whisperx_backend.torch_load_compat", return_value=nullcontext()),
        patch("tool_clonevoice_v2.segment_engine.transcribe", return_value={
            "language": "ja",
            "segments": [{
                "start": 0.0,
                "end": 2.0,
                "text": "こんにちは",
                "words": [],
            }],
        }) as transcribe,
        patch("tool_clonevoice_v2.logic._release_cuda_cache"),
    ):
        manifest = logic.run_transcribe_diarize(
            video,
            models_root=str(tmp_path / "models"),
            target_language="Chinese",
            skip_diarization=True,
            skip_existing=True,
        )

    extract_audio.assert_not_called()
    transcribe.assert_called_once()
    assert manifest["segments"][0]["src_text"] == "こんにちは"
    assert (cdir / logic.SOURCE_SRT_NAME).is_file()
    assert logic.manifest_path(video).is_file()
