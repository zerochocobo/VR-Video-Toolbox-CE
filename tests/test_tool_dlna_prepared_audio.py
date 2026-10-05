"""The built-in DLNA transports reuse prepared audio without encoding again."""
import io
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_si_dlna_audio import movie
from tests.test_tool_dlna_si_stream import _FakeSession, _FakePopen
from tool_dlna import si_stream
from tool_si import dlna_audio, logic as si
from utils.si_prepared_audio import input_signature, make_manifest, prepared_paths


@pytest.fixture
def prepared(movie):
    video, voice = movie
    output = Path(dlna_audio.prepare_audio(video))
    config = si_stream.SIMixConfig(enabled=True, si_delay_seconds=0.0)
    return video, voice, output, config


@pytest.mark.parametrize("dub", [False, True])
@pytest.mark.parametrize("transport", ["mpegts", "mp4"])
@pytest.mark.parametrize("start", [0.0, 1.1])
def test_real_prepared_streams_copy_audio_and_decode_at_start_and_seek(movie, dub, transport, start):
    video, voice = movie
    if dub:
        si.write_duck_key_wav(video.with_suffix(".si.duck.wav"), [{"start": 0, "end": 1}], 1, 24000)
    audio = Path(dlna_audio.prepare_audio(video))
    config = si_stream.SIMixConfig(enabled=True, si_delay_seconds=0.0)
    if dub:
        # The server must use the fixed dub preset despite customized SI knobs.
        config = replace(config, mix_channel="left", si_volume_percent=50, si_delay_seconds=1.5)
    service = si_stream.SIStreamService(config_holder=si_stream.ConfigHolder(config))
    effective, duck = service.resolve_stream(video)
    assert si_stream._prepared_audio_for_stream(video, voice, effective, duck) == audio
    # Warm the format validation, then prove playback doesn't run another probe.
    popen = subprocess.Popen
    commands = []
    def record(cmd, **kwargs):
        commands.append(cmd)
        return popen(cmd, **kwargs)
    with (patch.object(si_stream.subprocess, "run", side_effect=AssertionError("Probe is cached")),
          patch.object(si_stream.subprocess, "Popen", side_effect=record)):
        if transport == "mpegts":
            payload = b"".join(si_stream.iter_si_mpegts(video, voice, effective, start, duck_key=duck))
        else:
            session = si_stream.LiveStreamSession(video, voice, effective, start, 1000000, duck_key=duck)
            try:
                chunks = []
                while chunk := session.read(65536):
                    chunks.append(chunk)
                payload = b"".join(chunks)
            finally:
                session.close()
    cmd, = commands
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-filter_complex" not in cmd
    assert [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"] == [str(video), str(audio)]
    assert [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-ss"] == [f"{start:.3f}"] * 2
    output = video.parent / f"stream.{transport}"
    output.write_bytes(payload)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
                   check=True, capture_output=True, timeout=20)
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output),
    ]))
    assert [s["codec_name"] for s in info["streams"]] == ["h264", "aac"]
    assert info["streams"][1]["channels"] == 2
    # Compare AAC packet payloads (strip TS ADTS headers by remuxing to MP4).
    if start == 0:
        extracted = video.parent / "extracted.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(output), "-map", "0:a:0",
                        "-c:a", "copy", "-bsf:a", "aac_adtstoasc", str(extracted)],
                       check=True, capture_output=True)
        def packets(path):
            return json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
                "-show_data_hash", "sha256", "-show_entries", "packet=data_hash", "-of", "json", str(path),
            ]))["packets"]
        actual = [p["data_hash"] for p in packets(extracted)]
        original = [p["data_hash"] for p in packets(audio)]
        # Input seeking can drop AAC priming, but every retained packet is exact.
        offset = original.index(actual[0])
        assert actual == original[offset:offset + len(actual)]


@pytest.mark.parametrize("change", ["delay", "voice", "duck", "audio", "json", "corrupt_container"])
def test_unmatched_or_stale_mix_falls_back_to_live_encoding(prepared, change):
    video, voice, audio, config = prepared
    _, meta = prepared_paths(video)
    if change == "delay":
        config = replace(config, si_delay_seconds=1.0)
    elif change == "voice":
        voice.write_bytes(b"updated voice")
    elif change == "duck":
        video.with_suffix(".si.duck.wav").write_bytes(b"new key")
        config = config.dubbing_variant()
    elif change == "audio":
        audio.write_bytes(b"truncated")
    elif change == "json":
        meta.write_text("{incomplete", encoding="utf-8")
    else:
        audio.write_bytes(b"invalid mp4 with matching metadata")
        meta.write_text(json.dumps(make_manifest(input_signature(video, voice, None), config.filter_string(), audio)),
                        encoding="utf-8")
    duck = video.with_suffix(".si.duck.wav") if change == "duck" else None
    cmd = si_stream._si_input_command(video, voice, config, 1.0, duck)
    assert "-filter_complex" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert str(voice) in cmd


@pytest.mark.parametrize("change", ["voice", "duck", "audio", "json", "created", "removed"])
def test_replaced_assets_restart_existing_session(prepared, change):
    video, voice, audio, config = prepared
    _, meta = prepared_paths(video)
    if change == "created":
        meta.unlink()
    service = si_stream.SIStreamService(config_holder=si_stream.ConfigHolder(config),
                                       session_factory=_FakeSession, seek_cooldown_seconds=0)
    with patch.object(si_stream.content_directory, "probe_cached", return_value={"duration": 3, "size": 100000}):
        first, *_ = service.open_stream(video, 0, 9)
        b"".join(first)
        old = next(iter(service._sessions.values()))
        if change == "voice":
            voice.write_bytes(b"changed")
        elif change == "duck":
            video.with_suffix(".si.duck.wav").write_bytes(b"key")
        elif change == "audio":
            audio.write_bytes(b"changed")
        elif change in {"json", "created"}:
            meta.write_text("changed", encoding="utf-8")
        else:
            audio.unlink()
        second, *_ = service.open_stream(video, 10, 19)
        b"".join(second)
        assert old.closed
        assert next(iter(service._sessions.values())) is not old
    service.shutdown()
    assert not service._session_inputs


def test_finished_remux_can_still_drain_buffered_output(tmp_path):
    process = _FakePopen()
    process.stdout = io.BytesIO(b"buffered")
    process.terminated = True
    with patch.object(si_stream.subprocess, "Popen", return_value=process):
        session = si_stream.LiveStreamSession(tmp_path / "movie.mp4", tmp_path / "movie.si.wav",
                                             si_stream.SIMixConfig(enabled=True), 0, 100)
        assert session.read(100) == b"buffered"
        assert session.read(100) == b""
        session.close()


def test_real_http_live_route_uses_prepared_mix(prepared):
    from fastapi.testclient import TestClient
    from tool_dlna.dlna_server import create_app
    from tool_dlna.media_library import MediaLibrary, build_media_roots
    video, voice, audio, config = prepared
    assert si_stream._prepared_audio_for_stream(video, voice, config, None) == audio
    app = create_app(server_name="Test", port=8090, media_library=MediaLibrary(build_media_roots([video.parent])),
                     subtitles_enabled=False, device_uuid="test", lan_ip="127.0.0.1",
                     cache_dir=video.parent / "cache", si_config_holder=si_stream.ConfigHolder(config))
    popen = subprocess.Popen
    commands = []
    def record(cmd, **kwargs):
        commands.append(cmd)
        return popen(cmd, **kwargs)
    with TestClient(app) as client, patch.object(si_stream.subprocess, "Popen", side_effect=record):
        response = client.get("/si_live/movie.mp4.ts?t=1.1")
    assert response.status_code == 200 and response.content
    assert response.headers["x-si-transport"] == "mpegts-live"
    cmd, = commands
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert str(audio) in cmd
