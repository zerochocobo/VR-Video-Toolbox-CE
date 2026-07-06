from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_dlna import si_stream
from tool_si import logic as si_logic


class _FakeSession:
    created: list["_FakeSession"] = []

    def __init__(self, video, si_wav, config, start_time, estimated_total, start_byte=0, duck_key=None):
        self.video = Path(video)
        self.si_wav = Path(si_wav)
        self.config = config
        self.duck_key = Path(duck_key) if duck_key is not None else None
        self.start_time = start_time
        self.estimated_total = estimated_total
        self.byte_cursor = start_byte
        self.closed = False
        _FakeSession.created.append(self)

    def is_usable(self):
        return not self.closed

    def read(self, n):
        if self.closed:
            return b""
        size = max(0, int(n))
        self.byte_cursor += size
        return b"x" * size

    def discard(self, n):
        skipped = max(0, int(n))
        self.byte_cursor += skipped
        return skipped

    def close(self):
        self.closed = True


class _FakePopen:
    def __init__(self):
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if not (self.terminated or self.killed) else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class SIStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSession.created = []

    def test_si_mix_config_from_app_config_defaults(self) -> None:
        config = si_stream.SIMixConfig.from_app_config(lambda _key, default=None: default)

        self.assertTrue(config.enabled)
        self.assertEqual(config.mix_channel, "both")
        self.assertEqual(config.original_volume_percent, 100)
        self.assertEqual(config.si_volume_percent, 100)
        self.assertEqual(config.si_delay_seconds, 1.0)
        self.assertTrue(config.duck_original)
        self.assertEqual(config.duck_preset, "normal")
        self.assertTrue(config.dub_mode_enabled)

    def test_si_mix_config_invalid_values_fall_back_to_dlna_defaults(self) -> None:
        config = si_stream.SIMixConfig.from_mapping(
            {
                "dlna_si_mix_channel": "invalid",
                "dlna_si_volume_percent": 999,
                "dlna_si_duck_preset": "invalid",
            }
        )

        self.assertEqual(config.mix_channel, "both")
        self.assertEqual(config.si_volume_percent, 100)
        self.assertEqual(config.duck_preset, "normal")

    def test_si_mix_config_filter_string_matches_tool_si(self) -> None:
        config = si_stream.SIMixConfig(
            enabled=True,
            mix_channel="right",
            original_volume_percent=90,
            si_volume_percent=60,
            si_delay_seconds=0.7,
            duck_original=True,
            duck_preset="light",
        )

        self.assertEqual(
            config.filter_string(),
            si_logic.build_si_mix_filter("right", 90, 60, 0.7, duck_original=True, duck_preset="light"),
        )

    def test_parse_range_header_handles_open_ended_and_malformed(self) -> None:
        self.assertEqual(si_stream.parse_range_header("bytes=1024-"), (1024, None))
        self.assertEqual(si_stream.parse_range_header("bytes=1024-2048"), (1024, 2048))
        self.assertEqual(si_stream.parse_range_header("bytes=abc"), (0, None))
        self.assertEqual(si_stream.parse_range_header("items=1-2"), (0, None))

    def test_has_si_source_detects_sibling_wav(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            si_wav = root / "movie.si.wav"
            video.write_bytes(b"video")
            si_wav.write_bytes(b"wav")
            service = si_stream.SIStreamService(config_holder=si_stream.ConfigHolder(si_stream.SIMixConfig(True)))

            self.assertEqual(service.has_si_source(video), si_wav)

    def test_has_si_source_detects_uppercase_sibling_wav(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            si_wav = root / "movie.SI.WAV"
            video.write_bytes(b"video")
            si_wav.write_bytes(b"wav")
            service = si_stream.SIStreamService(config_holder=si_stream.ConfigHolder(si_stream.SIMixConfig(True)))

            self.assertEqual(service.has_si_source(video), si_wav)

    def test_has_duck_key_detects_sibling_duck_wav(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            duck = root / "movie.si.duck.wav"
            video.write_bytes(b"video")
            duck.write_bytes(b"duck")
            service = si_stream.SIStreamService(config_holder=si_stream.ConfigHolder(si_stream.SIMixConfig(True)))

            self.assertEqual(service.has_duck_key(video), duck)
            missing = root / "other.mp4"
            missing.write_bytes(b"video")
            self.assertIsNone(service.has_duck_key(missing))

    def test_dubbing_variant_forces_dub_levels(self) -> None:
        base = si_stream.SIMixConfig(
            enabled=True, mix_channel="left", original_volume_percent=80,
            si_volume_percent=100, si_delay_seconds=1.5, duck_original=True, duck_preset="strong",
        )
        dub = base.dubbing_variant()

        self.assertEqual(dub.original_volume_percent, 100)
        self.assertEqual(dub.si_volume_percent, 120)
        self.assertEqual(dub.si_delay_seconds, 0.0)
        self.assertTrue(dub.duck_original)
        self.assertEqual(dub.mix_channel, "both")    # fixed, ignores user channel
        self.assertEqual(dub.duck_preset, "strong")  # fixed, ignores user preset

    def test_resolve_stream_switches_to_dubbing_when_duck_key_present(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            (root / "movie.si.duck.wav").write_bytes(b"duck")
            holder = si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=True, dub_mode_enabled=True))
            service = si_stream.SIStreamService(config_holder=holder)

            config, duck_key = service.resolve_stream(video)
            self.assertEqual(config.si_volume_percent, 120)
            self.assertEqual(duck_key, root / "movie.si.duck.wav")

    def test_resolve_stream_stays_base_without_duck_key_or_when_dub_off(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            # no duck key -> base config
            holder = si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=True, dub_mode_enabled=True))
            service = si_stream.SIStreamService(config_holder=holder)
            config, duck_key = service.resolve_stream(video)
            self.assertIsNone(duck_key)
            self.assertEqual(config.si_volume_percent, 100)

            # duck key present but dub mode off -> base config
            (root / "movie.si.duck.wav").write_bytes(b"duck")
            holder.set(si_stream.SIMixConfig(enabled=True, dub_mode_enabled=False))
            config, duck_key = service.resolve_stream(video)
            self.assertIsNone(duck_key)
            self.assertEqual(config.si_volume_percent, 100)

    def test_live_stream_session_adds_duck_key_third_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            wav = root / "movie.si.wav"
            duck = root / "movie.si.duck.wav"
            for p in (video, wav, duck):
                p.write_bytes(b"x")
            fake_proc = _FakePopen()

            with patch("tool_dlna.si_stream.subprocess.Popen", return_value=fake_proc) as popen:
                session = si_stream.LiveStreamSession(
                    video, wav, si_stream.SIMixConfig(enabled=True).dubbing_variant(),
                    start_time=0.0, estimated_total=1000, duck_key=duck,
                )
                cmd = popen.call_args[0][0]
                session.close()

        self.assertEqual(cmd.count("-i"), 3)
        self.assertEqual(cmd[cmd.index("-i", cmd.index("-i", cmd.index("-i") + 1) + 1) + 1], str(duck))
        filter_arg = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("[2:a:0]", filter_arg)

    def test_iter_si_mpegts_adds_duck_key_third_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            wav = root / "movie.si.wav"
            duck = root / "movie.si.duck.wav"
            for p in (video, wav, duck):
                p.write_bytes(b"x")
            fake_proc = _FakePopen()
            fake_proc.stdout = io.BytesIO(b"abc")

            with patch("tool_dlna.si_stream.subprocess.Popen", return_value=fake_proc) as popen:
                b"".join(si_stream.iter_si_mpegts(
                    video, wav, si_stream.SIMixConfig(enabled=True).dubbing_variant(),
                    start_time=0.0, duck_key=duck, chunk_size=2,
                ))
                cmd = popen.call_args[0][0]

        self.assertEqual(cmd.count("-i"), 3)
        self.assertIn("[2:a:0]", cmd[cmd.index("-filter_complex") + 1])

    def test_estimate_output_size_uses_video_size_and_audio_bitrate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "movie.mp4"
            video.write_bytes(b"0" * 1024)
            service = si_stream.SIStreamService()
            meta = {"duration": 10.0, "size": 1_000_000, "video_size": 900_000}

            with patch("tool_dlna.si_stream.content_directory.probe_cached", return_value=meta):
                size = service.estimate_output_size(video)

        self.assertEqual(size, int((900_000 + 240_000) * 1.05))

    def test_live_stream_session_starts_ffmpeg_with_seek_and_fragmented_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            wav = root / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            fake_proc = _FakePopen()

            with patch("tool_dlna.si_stream.subprocess.Popen", return_value=fake_proc) as popen:
                session = si_stream.LiveStreamSession(
                    video,
                    wav,
                    si_stream.SIMixConfig(enabled=True),
                    start_time=12.345,
                    estimated_total=1000,
                    start_byte=100,
                )
                cmd = popen.call_args[0][0]
                session.close()

        self.assertEqual(cmd[cmd.index("-ss") + 1], "12.345")
        self.assertEqual(cmd[cmd.index("-i") + 1], str(video))
        self.assertIn("+frag_keyframe+empty_moov+default_base_moof", cmd)
        self.assertIn("pipe:1", cmd)

    def test_iter_si_mpegts_starts_ffmpeg_as_mpegts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            wav = root / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            fake_proc = _FakePopen()
            fake_proc.stdout = io.BytesIO(b"abcdef")

            with patch("tool_dlna.si_stream.subprocess.Popen", return_value=fake_proc) as popen:
                data = b"".join(
                    si_stream.iter_si_mpegts(
                        video,
                        wav,
                        si_stream.SIMixConfig(enabled=True),
                        start_time=5.0,
                        chunk_size=2,
                    )
                )
                cmd = popen.call_args[0][0]

        self.assertEqual(data, b"abcdef")
        self.assertEqual(cmd[cmd.index("-ss") + 1], "5.000")
        self.assertEqual(cmd[cmd.index("-f") + 1], "mpegts")
        self.assertIn("-muxpreload", cmd)
        self.assertIn("-muxdelay", cmd)
        self.assertNotIn("+frag_keyframe+empty_moov+default_base_moof", cmd)

    def test_open_stream_starts_session_with_seek_from_range(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "movie.mp4"
            wav = Path(raw) / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            holder = si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=True))
            service = si_stream.SIStreamService(None, holder, session_factory=_FakeSession, seek_cooldown_seconds=0)
            meta = {"duration": 100.0, "size": 1_000_000, "video_size": 800_000}

            with patch("tool_dlna.si_stream.content_directory.probe_cached", return_value=meta):
                total = service.estimate_output_size(video)
                chunks, content_length, returned_total, status = service.open_stream(
                    video,
                    range_start=total // 2,
                    range_end=(total // 2) + 99,
                )
                data = b"".join(chunks)

        self.assertEqual(status, 206)
        self.assertEqual(content_length, 100)
        self.assertEqual(returned_total, total)
        self.assertEqual(len(data), 100)
        self.assertAlmostEqual(_FakeSession.created[0].start_time, 50.0, places=3)

    def test_open_stream_reuses_session_on_sequential_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "movie.mp4"
            wav = Path(raw) / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            holder = si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=True))
            service = si_stream.SIStreamService(None, holder, session_factory=_FakeSession, seek_cooldown_seconds=0)
            meta = {"duration": 10.0, "size": 1_000_000, "video_size": 900_000}

            with patch("tool_dlna.si_stream.content_directory.probe_cached", return_value=meta):
                first, _, _, _ = service.open_stream(video, range_start=0, range_end=9)
                self.assertEqual(len(b"".join(first)), 10)
                second, _, _, _ = service.open_stream(video, range_start=10, range_end=19)
                self.assertEqual(len(b"".join(second)), 10)

        self.assertEqual(len(_FakeSession.created), 1)
        self.assertEqual(_FakeSession.created[0].byte_cursor, 20)

    def test_open_stream_uses_independent_sessions_for_different_clients(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "movie.mp4"
            wav = Path(raw) / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            holder = si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=True))
            service = si_stream.SIStreamService(None, holder, session_factory=_FakeSession, seek_cooldown_seconds=0)
            meta = {"duration": 10.0, "size": 1_000_000, "video_size": 900_000}

            with patch("tool_dlna.si_stream.content_directory.probe_cached", return_value=meta):
                first, _, _, _ = service.open_stream(video, range_start=0, range_end=9, client_id="192.168.1.10")
                self.assertEqual(len(b"".join(first)), 10)
                second, _, _, _ = service.open_stream(video, range_start=0, range_end=9, client_id="192.168.1.11")
                self.assertEqual(len(b"".join(second)), 10)

        self.assertEqual(len(_FakeSession.created), 2)

    def test_open_stream_does_not_close_session_on_early_client_disconnect(self) -> None:
        # DLNA players open a new TCP connection per Range request and close it
        # as soon as they have enough bytes. The chunks() generator must NOT kill
        # the ffmpeg session on that early disconnect, otherwise every subsequent
        # Range restarts ffmpeg with a fresh moov, corrupting the virtual MP4.
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "movie.mp4"
            wav = Path(raw) / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            holder = si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=True))
            service = si_stream.SIStreamService(None, holder, session_factory=_FakeSession, seek_cooldown_seconds=0)
            meta = {"duration": 100.0, "size": 1_000_000, "video_size": 900_000}

            with patch("tool_dlna.si_stream.content_directory.probe_cached", return_value=meta):
                chunks, _, _, _ = service.open_stream(video, range_start=0, range_end=None)
                iterator = iter(chunks)
                first = next(iterator)
                self.assertTrue(first)
                # Simulate FastAPI cancelling StreamingResponse on client disconnect.
                iterator.close()

        session = _FakeSession.created[0]
        self.assertFalse(session.closed, "session must survive an early HTTP disconnect")

    def test_reload_config_terminates_active_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "movie.mp4"
            wav = Path(raw) / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            holder = si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=True))
            service = si_stream.SIStreamService(None, holder, session_factory=_FakeSession, seek_cooldown_seconds=0)
            meta = {"duration": 10.0, "size": 1_000_000, "video_size": 900_000}

            with patch("tool_dlna.si_stream.content_directory.probe_cached", return_value=meta):
                service.open_stream(video, range_start=0, range_end=9)
                session = _FakeSession.created[0]
                service.reload_config(si_stream.SIMixConfig(enabled=False))

        self.assertTrue(session.closed)
        self.assertFalse(service.current_config().enabled)


if __name__ == "__main__":
    unittest.main()
