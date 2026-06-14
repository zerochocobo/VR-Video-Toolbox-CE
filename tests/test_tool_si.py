from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tool_si import logic


class FakeTTSModel:
    def __init__(self, fail_batch: bool = False) -> None:
        self.fail_batch = fail_batch
        self.calls: list[dict] = []

    def generate_custom_voice(self, text, speaker, language, max_new_tokens, **kwargs):
        self.calls.append(
            {
                "text": text,
                "speaker": speaker,
                "language": language,
                "max_new_tokens": max_new_tokens,
                **kwargs,
            }
        )
        if isinstance(text, list):
            if self.fail_batch:
                raise RuntimeError("batch failed")
            return [np.ones(12, dtype=np.float32) * (idx + 1) * 0.1 for idx, _ in enumerate(text)], 100
        return [np.ones(12, dtype=np.float32) * 0.2], 100


class SimultaneousInterpretationLogicTests(unittest.TestCase):
    def test_default_chinese_speaker_is_serena(self) -> None:
        self.assertEqual(logic.default_speaker_for_language("Chinese"), "Serena")
        self.assertIn("Vivian", logic.speakers_for_language("Chinese"))

    def test_speaker_note_keys_cover_all_speakers(self) -> None:
        for speaker in logic.ALL_SPEAKERS:
            self.assertTrue(logic.speaker_note_key(speaker), speaker)
        self.assertEqual(logic.speaker_note_key("Ono_Anna"), "speaker_note_ono_anna")

    def test_parse_srt_uses_first_non_empty_subtitle_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = Path(tmp_dir) / "sample.srt"
            srt_path.write_text(
                "\ufeff1\n"
                "00:00:01,000 --> 00:00:03,500\n"
                "こんにちは\n"
                "Hello\n\n"
                "2\n"
                "00:00:04,000 --> 00:00:05,000\n"
                "{\\an8}<i>次の字幕</i>\n",
                encoding="utf-8",
            )

            entries = logic.parse_srt(srt_path)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].text, "こんにちは")
        self.assertEqual(entries[1].text, "次の字幕")
        self.assertAlmostEqual(entries[0].start, 1.0)
        self.assertAlmostEqual(entries[0].end, 3.5)

    def test_parse_srt_selects_text_line_for_requested_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = Path(tmp_dir) / "sample.srt"
            srt_path.write_text(
                "1\n"
                "00:00:01,000 --> 00:00:03,000\n"
                "最近肩膀疼\n"
                "最近肩が痛くて\n",
                encoding="utf-8",
            )

            chinese_entries = logic.parse_srt(srt_path, language="Chinese")
            japanese_entries = logic.parse_srt(srt_path, language="Japanese")

        self.assertEqual(chinese_entries[0].text, "最近肩膀疼")
        self.assertEqual(japanese_entries[0].text, "最近肩が痛くて")

    def test_collect_paired_srt_tasks_finds_srt_matching_mp4_or_mkv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.srt").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.mkv").write_bytes(b"")
            (nested / "b.srt").write_text("x", encoding="utf-8")
            (nested / "ignored.srt").write_text("x", encoding="utf-8")

            tasks = logic.collect_paired_srt_tasks(root)

        self.assertEqual([path.name for path in tasks], ["a.srt", "b.srt"])

    def test_collect_paired_srt_tasks_ignores_generated_mp4_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.srt").write_text("x", encoding="utf-8")
            (root / "a_SI.mp4").write_bytes(b"")
            (root / "a_SI.srt").write_text("x", encoding="utf-8")
            (root / "a_DUB.mp4").write_bytes(b"")
            (root / "a_DUB.srt").write_text("x", encoding="utf-8")

            tasks = logic.collect_paired_srt_tasks(root)

        self.assertEqual([path.name for path in tasks], ["a.srt"])

    def test_collect_paired_srt_tasks_can_skip_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.srt").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.mp4").write_bytes(b"")
            (nested / "b.srt").write_text("x", encoding="utf-8")

            tasks = logic.collect_paired_srt_tasks(root, recursive=False)

        self.assertEqual([path.name for path in tasks], ["a.srt"])

    def test_collect_paired_si_mix_tasks_finds_si_wav_matching_mp4_or_mkv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.si.wav").write_bytes(b"wav")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.mkv").write_bytes(b"")
            (nested / "b.si.wav").write_bytes(b"wav")
            (nested / "ignored.si.wav").write_bytes(b"wav")

            tasks = logic.collect_paired_si_mix_tasks(root)

        self.assertEqual([task.video_path.name for task in tasks], ["a.mp4", "b.mkv"])
        self.assertEqual([task.si_audio_path.name for task in tasks], ["a.si.wav", "b.si.wav"])
        self.assertEqual([task.output_path.name for task in tasks], ["a_SI.mp4", "b_SI.mp4"])

    def test_collect_paired_si_mix_tasks_ignores_generated_mp4_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.si.wav").write_bytes(b"wav")
            (root / "a_SI.mp4").write_bytes(b"")
            (root / "a_SI.si.wav").write_bytes(b"wav")
            (root / "a_DUB.mp4").write_bytes(b"")
            (root / "a_DUB.si.wav").write_bytes(b"wav")

            tasks = logic.collect_paired_si_mix_tasks(root)

        self.assertEqual([task.video_path.name for task in tasks], ["a.mp4"])

    def test_collect_paired_si_mix_tasks_can_skip_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.si.wav").write_bytes(b"wav")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.mp4").write_bytes(b"")
            (nested / "b.si.wav").write_bytes(b"wav")

            tasks = logic.collect_paired_si_mix_tasks(root, recursive=False)

        self.assertEqual([task.video_path.name for task in tasks], ["a.mp4"])

    def test_batch_mix_si_audio_tracks_uses_default_si_outputs_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.si.wav").write_bytes(b"wav")

            with patch.object(
                logic,
                "mix_si_audio_track",
                side_effect=lambda **kwargs: str(kwargs["output_path"]),
            ) as mix_audio:
                outputs = logic.batch_mix_si_audio_tracks(
                    base_dir=root,
                    mix_channel="right",
                    original_volume_percent=90,
                    si_volume_percent=60,
                    si_delay_seconds=1.2,
                    add_independent_track=True,
                    duck_original=True,
                    log_callback=lambda _message: None,
                    recursive=False,
                )

        self.assertEqual([Path(output).name for output in outputs], ["a_SI.mp4"])
        kwargs = mix_audio.call_args.kwargs
        self.assertEqual(kwargs["video_path"].name, "a.mp4")
        self.assertEqual(kwargs["si_audio_path"].name, "a.si.wav")
        self.assertEqual(kwargs["output_path"].name, "a_SI.mp4")
        self.assertEqual(kwargs["mix_channel"], "right")
        self.assertEqual(kwargs["original_volume_percent"], 90)
        self.assertEqual(kwargs["si_volume_percent"], 60)
        self.assertEqual(kwargs["si_delay_seconds"], 1.2)
        self.assertEqual(kwargs["add_independent_track"], True)
        self.assertEqual(kwargs["duck_original"], True)

    def test_default_si_audio_mix_paths_use_same_stem(self) -> None:
        video_path = Path("work") / "test.mp4"

        self.assertEqual(logic.default_si_audio_path(video_path), str(Path("work") / "test.si.wav"))
        self.assertEqual(logic.default_si_mix_output_path(video_path), str(Path("work") / "test_SI.mp4"))

    def test_default_paths_preserve_source_separator_style(self) -> None:
        self.assertEqual(logic.default_output_path("C:/work/test.srt"), "C:/work/test.si.wav")
        self.assertEqual(logic.default_si_audio_path("C:/work/test.mp4"), "C:/work/test.si.wav")
        self.assertEqual(logic.default_si_mix_output_path("C:/work/test.mp4"), "C:/work/test_SI.mp4")
        self.assertEqual(logic.default_si_audio_path(r"C:\work\test.mp4"), r"C:\work\test.si.wav")
        self.assertEqual(logic.default_si_mix_output_path(r"C:\work\test.mp4"), r"C:\work\test_SI.mp4")

    def test_build_si_audio_mix_command_replaces_first_audio_by_default(self) -> None:
        cmd = logic.build_si_audio_mix_command(
            video_path="test.mp4",
            si_audio_path="test.si.wav",
            output_path="test_SI.mp4",
            mix_channel="left",
            original_volume_percent=100,
            si_volume_percent=50,
            audio_stream_count=2,
        )
        filter_arg = cmd[cmd.index("-filter_complex") + 1]
        mapped_streams = [cmd[index + 1] for index, token in enumerate(cmd) if token == "-map"]

        self.assertIn("[ol][si]amix", filter_arg)
        self.assertIn("[left_mix_raw][or]join", filter_arg)
        self.assertIn("alimiter=limit=0.95[si_track]", filter_arg)
        self.assertIn("aformat=channel_layouts=stereo", filter_arg)
        self.assertIn("aformat=channel_layouts=mono", filter_arg)
        self.assertIn("volume=1[orig]", filter_arg)
        self.assertIn("adelay=1000", filter_arg)
        self.assertIn("volume=0.5[si]", filter_arg)
        self.assertEqual(mapped_streams[:3], ["0:v?", "[si_track]", "0:a:1?"])
        self.assertNotIn("0:a?", cmd)
        self.assertIn("-c:a:0", cmd)
        self.assertIn("-metadata:s:a:0", cmd)
        self.assertIn("title=SI", cmd)
        self.assertEqual(cmd[-1], "test_SI.mp4")

    def test_build_si_audio_mix_command_keeps_original_audio_when_probe_unavailable(self) -> None:
        cmd = logic.build_si_audio_mix_command(
            video_path="test.mp4",
            si_audio_path="test.si.wav",
            output_path="test_SI.mp4",
            mix_channel="left",
            original_volume_percent=100,
            si_volume_percent=50,
            audio_stream_count=None,
        )
        mapped_streams = [cmd[index + 1] for index, token in enumerate(cmd) if token == "-map"]

        self.assertEqual(mapped_streams[:4], ["0:v?", "[si_track]", "0:a?", "-0:a:0"])
        self.assertIn("-c:a:0", cmd)
        self.assertIn("-metadata:s:a:0", cmd)

    def test_build_si_audio_mix_command_can_add_independent_track(self) -> None:
        cmd = logic.build_si_audio_mix_command(
            video_path="test.mp4",
            si_audio_path="test.si.wav",
            output_path="test_SI.mp4",
            mix_channel="left",
            original_volume_percent=100,
            si_volume_percent=50,
            audio_stream_count=2,
            add_independent_track=True,
        )
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        self.assertIn("[ol][si]amix", filter_arg)
        self.assertIn("[left_mix_raw][or]join", filter_arg)
        self.assertIn("alimiter=limit=0.95[si_track]", filter_arg)
        self.assertIn("volume=1[orig]", filter_arg)
        self.assertIn("adelay=1000", filter_arg)
        self.assertIn("volume=0.5[si]", filter_arg)
        self.assertIn("0:a?", cmd)
        self.assertIn("-c:a:2", cmd)
        self.assertIn("-metadata:s:a:2", cmd)
        self.assertIn("title=SI", cmd)
        self.assertEqual(cmd[-1], "test_SI.mp4")

    def test_build_si_audio_mix_command_can_duck_original_audio_left_channel(self) -> None:
        cmd = logic.build_si_audio_mix_command(
            video_path="test.mp4",
            si_audio_path="test.si.wav",
            output_path="test_SI.mp4",
            mix_channel="left",
            original_volume_percent=100,
            si_volume_percent=70,
            si_delay_seconds=1.2,
            audio_stream_count=1,
            duck_original=True,
        )
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        self.assertIn("aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1[orig_base]", filter_arg)
        self.assertIn("adelay=1200,volume=0.7,apad,asplit=2[si_key][si]", filter_arg)
        self.assertIn(
            "[orig_base][si_key]sidechaincompress=threshold=0.025:ratio=5:attack=30:release=600:makeup=1[orig]",
            filter_arg,
        )
        self.assertIn("[orig]channelsplit=channel_layout=stereo[ol][or]", filter_arg)
        self.assertIn("[ol][si]amix", filter_arg)
        self.assertIn("[left_mix_raw][or]join", filter_arg)

    def test_build_si_audio_mix_command_can_duck_original_audio_right_channel(self) -> None:
        cmd = logic.build_si_audio_mix_command(
            video_path="test.mp4",
            si_audio_path="test.si.wav",
            output_path="test_SI.mp4",
            mix_channel="right",
            original_volume_percent=90,
            si_volume_percent=60,
            si_delay_seconds=0,
            audio_stream_count=1,
            duck_original=True,
        )
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        self.assertIn("volume=0.9[orig_base]", filter_arg)
        self.assertIn("adelay=0,volume=0.6,apad,asplit=2[si_key][si]", filter_arg)
        self.assertIn("[or][si]amix", filter_arg)
        self.assertIn("[ol][right_mix_raw]join", filter_arg)

    def test_probe_audio_stream_count_times_out_cleanly(self) -> None:
        messages: list[str] = []

        with patch.object(logic.shutil, "which", return_value="ffprobe"), patch.object(
            logic.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["ffprobe"], timeout=15),
        ):
            result = logic.probe_audio_stream_count("broken.mp4", messages.append)

        self.assertIsNone(result)
        self.assertTrue(any("timed out" in message for message in messages))

    def test_build_si_audio_mix_command_overlays_right_channel(self) -> None:
        cmd = logic.build_si_audio_mix_command(
            video_path="test.mp4",
            si_audio_path="test.si.wav",
            output_path="test_SI.mp4",
            mix_channel="right",
            original_volume_percent=80,
            si_volume_percent=90,
            si_delay_seconds=1.2,
            audio_stream_count=1,
        )
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        self.assertIn("[or][si]amix", filter_arg)
        self.assertIn("[ol][right_mix_raw]join", filter_arg)
        self.assertIn("alimiter=limit=0.95[si_track]", filter_arg)
        self.assertIn("volume=0.8[orig]", filter_arg)
        self.assertIn("adelay=1200", filter_arg)
        self.assertIn("volume=0.9[si]", filter_arg)
        self.assertIn("-c:a:0", cmd)
        self.assertIn("-metadata:s:a:0", cmd)

    def test_build_si_audio_mix_command_accepts_zero_si_delay(self) -> None:
        cmd = logic.build_si_audio_mix_command(
            video_path="test.mp4",
            si_audio_path="test.si.wav",
            output_path="test_SI.mp4",
            mix_channel="left",
            original_volume_percent=100,
            si_volume_percent=50,
            si_delay_seconds=0,
            audio_stream_count=1,
        )
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        self.assertIn("adelay=0", filter_arg)

    def test_mix_timeline_segment_attenuates_overlapping_audio(self) -> None:
        timeline = np.array([0.0, 0.8, 0.8, 0.0], dtype=np.float32)
        segment = np.array([0.8, 0.8], dtype=np.float32)

        logic._mix_timeline_segment(timeline, 1, segment)

        self.assertTrue(np.allclose(timeline, [0.0, 0.8, 0.8, 0.0]))

    def test_parse_srt_skips_invalid_or_extreme_timecodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = Path(tmp_dir) / "bad_times.srt"
            srt_path.write_text(
                "1\n00:61:00,000 --> 00:61:01,000\nBad minute\n\n"
                "2\n00:00:00,000 --> 00:05:01,000\nToo long\n\n"
                "3\n07:00:00,000 --> 07:00:01,000\nToo late\n\n"
                "4\n00:00:01,000 --> 00:00:02,000\nGood\n",
                encoding="utf-8",
            )

            entries = logic.parse_srt(srt_path)

        self.assertEqual([entry.text for entry in entries], ["Good"])

    def test_speed_up_with_ffmpeg_timeout_falls_back_to_fast_resample(self) -> None:
        audio = np.linspace(-0.5, 0.5, 100, dtype=np.float32)

        with patch.object(logic.shutil, "which", return_value="ffmpeg"), patch.object(
            logic.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["ffmpeg"], timeout=30),
        ):
            result = logic._speed_up_with_ffmpeg(audio, sample_rate=1000, factor=2.0, target_samples=50)

        self.assertEqual(result.shape[0], 50)

    def test_build_si_audio_mix_command_rejects_video_without_audio(self) -> None:
        with self.assertRaises(ValueError):
            logic.build_si_audio_mix_command(
                video_path="test.mp4",
                si_audio_path="test.si.wav",
                output_path="test_SI.mp4",
                mix_channel="left",
                original_volume_percent=100,
                si_volume_percent=50,
                audio_stream_count=0,
            )

    def test_build_si_audio_mix_command_rejects_out_of_range_si_delay(self) -> None:
        with self.assertRaises(ValueError):
            logic.build_si_audio_mix_command(
                video_path="test.mp4",
                si_audio_path="test.si.wav",
                output_path="test_SI.mp4",
                mix_channel="left",
                original_volume_percent=100,
                si_volume_percent=50,
                si_delay_seconds=1.6,
                audio_stream_count=1,
            )

    def test_fit_audio_to_duration_pads_short_audio_to_target_samples(self) -> None:
        audio = np.ones(100, dtype=np.float32) * 0.5

        fitted = logic.fit_audio_to_duration(audio, sample_rate=1000, target_duration=0.2)

        self.assertEqual(fitted.shape[0], 200)
        self.assertTrue(np.allclose(fitted[:100], 0.5))
        self.assertTrue(np.allclose(fitted[100:], 0.0))

    def test_resolve_tts_batch_size_clamps_env_value(self) -> None:
        with patch.dict("os.environ", {logic.TTS_BATCH_SIZE_ENV: "99"}):
            self.assertEqual(logic.resolve_tts_batch_size(), logic.MAX_TTS_BATCH_SIZE)
        with patch.dict("os.environ", {logic.TTS_BATCH_SIZE_ENV: "bad"}):
            self.assertEqual(logic.resolve_tts_batch_size(), logic.DEFAULT_TTS_BATCH_SIZE)

    def test_resolve_tts_batch_token_spread_clamps_env_value(self) -> None:
        with patch.dict("os.environ", {logic.TTS_BATCH_TOKEN_SPREAD_ENV: "99"}):
            self.assertEqual(logic.resolve_tts_batch_token_spread(), 10.0)
        with patch.dict("os.environ", {logic.TTS_BATCH_TOKEN_SPREAD_ENV: "bad"}):
            self.assertEqual(logic.resolve_tts_batch_token_spread(), logic.DEFAULT_TTS_BATCH_TOKEN_SPREAD)

    def test_iter_tts_batches_groups_short_entries(self) -> None:
        entries = [
            logic.SubtitleEntry(index=1, start=0.0, end=1.0, text="a"),
            logic.SubtitleEntry(index=2, start=1.0, end=2.0, text="b"),
            logic.SubtitleEntry(index=3, start=2.0, end=3.0, text="c"),
            logic.SubtitleEntry(index=4, start=3.0, end=4.0, text="d"),
            logic.SubtitleEntry(index=5, start=4.0, end=5.0, text="e"),
        ]

        batches = logic._iter_tts_batches(entries, batch_size=2)

        self.assertEqual([[entry.index for entry in batch] for batch in batches], [[1, 2], [3, 4], [5]])

    def test_subtitle_to_audio_uses_batch_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                "2\n00:00:00,100 --> 00:00:00,200\nB\n",
                encoding="utf-8",
            )
            model_dir = root / logic.MODEL_DIR_NAME
            model_dir.mkdir()
            fake_model = FakeTTSModel()

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "2"}
            ):
                result = logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                )

            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertEqual(len(fake_model.calls), 1)
            self.assertEqual(fake_model.calls[0]["text"], ["A", "B"])

    def test_subtitle_to_audio_can_limit_test_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                "2\n00:00:00,100 --> 00:00:00,200\nB\n\n"
                "3\n00:00:00,200 --> 00:00:00,300\nC\n",
                encoding="utf-8",
            )
            (root / logic.MODEL_DIR_NAME).mkdir()
            fake_model = FakeTTSModel()
            messages: list[str] = []

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=messages.append,
                    tts_model=fake_model,
                    max_entries=2,
                )

            self.assertEqual(len(fake_model.calls), 1)
            self.assertEqual(fake_model.calls[0]["text"], ["A", "B"])
            self.assertTrue(any("converting first 2 for test" in message for message in messages))

    def test_subtitle_to_audio_filters_by_processing_time_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                "2\n00:00:00,100 --> 00:00:00,200\nB\n\n"
                "3\n00:00:00,350 --> 00:00:00,450\nC\n",
                encoding="utf-8",
            )
            (root / logic.MODEL_DIR_NAME).mkdir()
            fake_model = FakeTTSModel()
            messages: list[str] = []

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=messages.append,
                    tts_model=fake_model,
                    start_seconds=0.0,
                    duration_seconds=0.25,
                )

            audio, sr = logic.read_wav_mono(output_path)
            self.assertEqual(sr, 100)
            self.assertEqual(audio.shape[0], 25)
            self.assertEqual(len(fake_model.calls), 1)
            self.assertEqual(fake_model.calls[0]["text"], ["A", "B"])
            self.assertTrue(any("selected 2 in 0.000-0.250s" in message for message in messages))

    def test_subtitle_to_audio_shifts_processing_window_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:01,000 --> 00:00:01,200\nA\n\n"
                "2\n00:00:01,300 --> 00:00:01,500\nB\n",
                encoding="utf-8",
            )
            (root / logic.MODEL_DIR_NAME).mkdir()
            fake_model = FakeTTSModel()

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                    start_seconds=1.0,
                    duration_seconds=0.6,
                )

            audio, sr = logic.read_wav_mono(output_path)
            self.assertEqual(sr, 100)
            self.assertEqual(audio.shape[0], 60)
            self.assertGreater(float(np.abs(audio[:5]).max()), 0.0)

    def test_subtitle_to_audio_preserves_srt_order_when_limited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            long_text = "长" * 80
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                f"2\n00:00:00,100 --> 00:00:03,100\n{long_text}\n\n"
                "3\n00:00:03,100 --> 00:00:03,200\nC\n",
                encoding="utf-8",
            )
            (root / logic.MODEL_DIR_NAME).mkdir()
            fake_model = FakeTTSModel()

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                    max_entries=3,
                )

            seen_texts: list[str] = []
            for call in fake_model.calls:
                text = call["text"]
                if isinstance(text, list):
                    seen_texts.extend(text)
                else:
                    seen_texts.append(text)
            self.assertEqual(seen_texts, ["A", long_text, "C"])

    def test_subtitle_to_audio_retries_one_by_one_when_batch_generation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                "2\n00:00:00,100 --> 00:00:00,200\nB\n",
                encoding="utf-8",
            )
            model_dir = root / logic.MODEL_DIR_NAME
            model_dir.mkdir()
            fake_model = FakeTTSModel(fail_batch=True)

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "2"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                )

            self.assertEqual(len(fake_model.calls), 3)
            self.assertEqual(fake_model.calls[0]["text"], ["A", "B"])
            self.assertEqual([call["text"] for call in fake_model.calls[1:]], ["A", "B"])

    def test_iter_tts_batches_packs_similar_length_entries_after_sorting(self) -> None:
        # Interleave short / long entries the way a real SRT does. With pre-sort the
        # short ones cluster together and the batch packer can fill `batch_size` cleanly
        # instead of breaking at every length spike.
        entries = [
            logic.SubtitleEntry(index=1, start=0.0, end=0.2, text="a"),     # very short
            logic.SubtitleEntry(index=2, start=1.0, end=11.0, text="x" * 80),  # very long
            logic.SubtitleEntry(index=3, start=12.0, end=12.2, text="b"),
            logic.SubtitleEntry(index=4, start=13.0, end=23.0, text="y" * 80),
            logic.SubtitleEntry(index=5, start=24.0, end=24.2, text="c"),
        ]

        batches = logic._iter_tts_batches(entries, batch_size=4)

        # The two long entries must not share a batch with the short ones (token_spread
        # would otherwise force every short entry to decode out to ~120 tokens).
        for batch in batches:
            budgets = [
                logic._max_new_tokens_for_duration(e.duration, e.text) for e in batch
            ]
            self.assertLessEqual(max(budgets), min(budgets) * logic.resolve_tts_batch_token_spread() + 1)

        # Shorts should pack into a single batch (sort-then-pack), not split across many.
        short_indices = {1, 3, 5}
        short_batches = [b for b in batches if {e.index for e in b} & short_indices]
        self.assertEqual(len(short_batches), 1,
                         f"shorts split across batches: {[[e.index for e in b] for b in batches]}")
        self.assertEqual({e.index for e in short_batches[0]}, short_indices)

    def test_subtitle_to_audio_places_entries_by_start_time_independent_of_batch_order(self) -> None:
        # Pre-sort means batch generation order differs from SRT order. Output WAV must
        # still place each entry at its `entry.start`. We verify the timeline contains a
        # non-zero signal at each entry's start window — regardless of which batch the
        # entry was generated in.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "interleaved.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                # Short / long / short / long / short interleaved — forces the sort path
                # to reorder for batching. Entry text length controls token budget.
                "1\n00:00:01,000 --> 00:00:01,500\nA\n\n"
                "2\n00:00:02,000 --> 00:00:08,000\n" + ("X " * 30).strip() + "\n\n"
                "3\n00:00:09,000 --> 00:00:09,500\nC\n",
                encoding="utf-8",
            )
            model_dir = root / logic.MODEL_DIR_NAME
            model_dir.mkdir()
            fake_model = FakeTTSModel()

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="English",
                    speaker="Ryan",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                )

            audio, sr = logic.read_wav_mono(output_path)
            # All three entries should be present at their start positions.
            for entry_start in (1.0, 2.0, 9.0):
                idx = int(entry_start * sr)
                window = audio[idx : idx + int(0.05 * sr)]
                self.assertGreater(
                    float(np.abs(window).max()), 0.0,
                    f"no audio placed at start={entry_start}s — timeline placement broke after sort",
                )

    def test_subtalker_greedy_kwargs_default_to_argmax(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            os_environ_pop = logic.os.environ.pop(logic.TTS_SUBTALKER_GREEDY_ENV, None)
            try:
                kw = logic._subtalker_generate_kwargs()
                self.assertEqual(kw.get("subtalker_dosample"), False)
                self.assertEqual(kw.get("subtalker_top_k"), 1)
            finally:
                if os_environ_pop is not None:
                    logic.os.environ[logic.TTS_SUBTALKER_GREEDY_ENV] = os_environ_pop

    def test_subtalker_greedy_can_be_disabled_via_env(self) -> None:
        with patch.dict("os.environ", {logic.TTS_SUBTALKER_GREEDY_ENV: "0"}):
            self.assertEqual(logic._subtalker_generate_kwargs(), {})

    def test_fit_audio_to_duration_uses_fast_path_for_small_stretch(self) -> None:
        # Audio is ~10% too long → factor 1.1, should NOT call into librosa /
        # ffmpeg, just linearly resample.
        sample_rate = 24000
        target_duration = 1.0
        source = np.linspace(-0.5, 0.5, int(target_duration * sample_rate * 1.10), dtype=np.float32)

        with patch.object(logic, "_speed_up_in_memory") as expensive, \
             patch.object(logic, "_speed_up_with_ffmpeg") as expensive_ffmpeg:
            result = logic.fit_audio_to_duration(source, sample_rate, target_duration)

        expensive.assert_not_called()
        expensive_ffmpeg.assert_not_called()
        self.assertEqual(result.shape[0], int(target_duration * sample_rate))


if __name__ == "__main__":
    unittest.main()
