from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from tool_dlna import dlna_server
except ModuleNotFoundError as exc:  # pragma: no cover - only for incomplete local envs
    dlna_server = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _DummyLibrary:
    pass


@unittest.skipIf(dlna_server is None, f"dlna_server import failed: {IMPORT_ERROR}")
class DlnaServerTests(unittest.TestCase):
    def test_create_app_registers_required_dlna_routes(self) -> None:
        app = dlna_server.create_app(
            server_name="Test DLNA",
            port=8090,
            media_library=_DummyLibrary(),
            subtitles_enabled=True,
            device_uuid="00000000-0000-0000-0000-000000000000",
            lan_ip="127.0.0.1",
            cache_dir=Path("."),
        )

        routes = {(tuple(sorted(getattr(route, "methods", []) or [])), getattr(route, "path", "")) for route in app.routes}

        self.assertIn((("GET",), "/"), routes)
        self.assertIn((("GET",), "/description.xml"), routes)
        self.assertIn((("POST",), "/control/cds"), routes)
        self.assertIn((("POST",), "/control/cm"), routes)
        self.assertIn((("GET",), "/media/{name:path}"), routes)
        self.assertIn((("GET",), "/media_si/{name:path}"), routes)
        self.assertIn((("GET",), "/si_live/{name:path}"), routes)
        self.assertIn((("POST",), "/admin/reload_si_config"), routes)

    def test_si_live_route_streams_mpegts_and_accepts_ts_hint_suffix(self) -> None:
        from fastapi.testclient import TestClient

        from tool_dlna.media_library import MediaLibrary, build_media_roots
        from tool_dlna.si_stream import ConfigHolder, SIMixConfig

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            wav = root / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            app = dlna_server.create_app(
                server_name="Test DLNA",
                port=8090,
                media_library=MediaLibrary(build_media_roots([root])),
                subtitles_enabled=True,
                device_uuid="00000000-0000-0000-0000-000000000000",
                lan_ip="127.0.0.1",
                cache_dir=Path(raw) / "cache",
                si_config_holder=ConfigHolder(SIMixConfig(enabled=True)),
            )

            with patch("tool_dlna.si_stream.iter_si_mpegts", return_value=iter([b"tsdata"])) as stream:
                with TestClient(app) as client:
                    response = client.get("/si_live/movie.mp4.ts?t=12")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"tsdata")
        self.assertEqual(response.headers["x-si-transport"], "mpegts-live")
        self.assertIn("video/MP2T", response.headers["content-type"])
        self.assertIn("DLNA.ORG_OP=00", response.headers["contentfeatures.dlna.org"])
        self.assertEqual(stream.call_args.args[0], video.resolve())
        self.assertEqual(stream.call_args.args[1], wav)
        self.assertAlmostEqual(stream.call_args.args[3], 12.0)

    def test_si_live_route_uses_deovr_time_seek_headers(self) -> None:
        from fastapi.testclient import TestClient

        from tool_dlna.media_library import MediaLibrary, build_media_roots
        from tool_dlna.si_stream import ConfigHolder, SIMixConfig

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            wav = root / "movie.si.wav"
            video.write_bytes(b"video")
            wav.write_bytes(b"wav")
            app = dlna_server.create_app(
                server_name="Test DLNA",
                port=8090,
                media_library=MediaLibrary(build_media_roots([root])),
                subtitles_enabled=True,
                device_uuid="00000000-0000-0000-0000-000000000000",
                lan_ip="127.0.0.1",
                cache_dir=Path(raw) / "cache",
                si_config_holder=ConfigHolder(SIMixConfig(enabled=True)),
            )

            with patch("tool_dlna.si_stream.iter_si_mpegts", return_value=iter([b"tsdata"])):
                with TestClient(app) as client:
                    response = client.get("/si_live/movie.mp4?t=0", headers={"User-Agent": "DeoVR/14.0"})

        self.assertEqual(response.status_code, 200)
        features = response.headers["contentfeatures.dlna.org"]
        self.assertIn("DLNA.ORG_OP=10", features)
        self.assertIn("DLNA.ORG_CI=1", features)
        self.assertIn("DLNA.ORG_FLAGS=41700000000000000000000000000000", features)

    def test_cds_client_profile_detects_deovr_filter_without_user_agent(self) -> None:
        fields = {
            "BrowseFlag": "BrowseDirectChildren",
            "RequestedCount": "0",
            "Filter": "res,res@size,res@duration,dc:date,upnp:albumArtURI",
        }

        self.assertEqual(dlna_server._cds_client_profile({}, fields), "deovr")
        self.assertIsNone(dlna_server._cds_client_profile({"user-agent": "SKYBOX/2.0"}, fields))

    def test_loopback_host_detection(self) -> None:
        self.assertTrue(dlna_server.is_loopback_host("127.0.0.1"))
        self.assertTrue(dlna_server.is_loopback_host("::1"))
        self.assertFalse(dlna_server.is_loopback_host("192.168.1.10"))

    def test_classify_moov_probe_detects_open_ended_tail_probe(self) -> None:
        # Real-world capture from runtime_cache/logs/dlna_server.log:
        # range=bytes=7638679552- total=7638793254  (113 KB remaining)
        total = 7_638_793_254
        self.assertEqual(
            dlna_server.classify_moov_probe(7_638_679_552, None, total),
            "tail",
        )

    def test_classify_moov_probe_detects_closed_tail_probe(self) -> None:
        total = 10 * 1024 * 1024 * 1024
        # Closed Range for last 256 KB.
        self.assertEqual(
            dlna_server.classify_moov_probe(total - 256 * 1024, total - 1, total),
            "tail",
        )

    def test_classify_moov_probe_detects_mid_probe(self) -> None:
        total = 10 * 1024 * 1024 * 1024
        # Closed 1MB request at 60% of the file.
        self.assertEqual(
            dlna_server.classify_moov_probe(int(total * 0.6), int(total * 0.6) + 1024 * 1024 - 1, total),
            "mid",
        )

    def test_classify_moov_probe_passes_through_open_ended_user_seek(self) -> None:
        # User dragging the seek bar produces an open-ended Range. Anywhere
        # outside the last ~5MB dead-zone must NOT be flagged as a probe.
        total = 10 * 1024 * 1024 * 1024
        self.assertEqual(dlna_server.classify_moov_probe(int(total * 0.8), None, total), "")
        # Even 99% in is still real seek territory (>>5MB to the end).
        self.assertEqual(dlna_server.classify_moov_probe(int(total * 0.99), None, total), "")

    def test_classify_moov_probe_passes_through_initial_header(self) -> None:
        total = 10 * 1024 * 1024 * 1024
        # First 64KB probe from the start of the file is not a tail probe.
        self.assertEqual(dlna_server.classify_moov_probe(0, 65535, total), "")

    def test_classify_moov_probe_passes_through_sequential_open_chunk(self) -> None:
        # Real-world capture: range=bytes=9437184- (9 MB into a 7.6 GB file).
        total = 7_638_793_254
        self.assertEqual(dlna_server.classify_moov_probe(9_437_184, None, total), "")

    def test_absolute_form_request_path_is_normalized(self) -> None:
        scope = {"path": "//127.0.0.1:8090/control/cds"}

        original, normalized = dlna_server.normalize_absolute_form_path(scope)

        self.assertEqual(original, "//127.0.0.1:8090/control/cds")
        self.assertEqual(normalized, "/control/cds")
        self.assertEqual(scope["path"], "/control/cds")
        self.assertEqual(scope["raw_path"], b"/control/cds")

    def test_normal_route_path_is_left_unchanged(self) -> None:
        scope = {"path": "/control/cds"}

        original, normalized = dlna_server.normalize_absolute_form_path(scope)

        self.assertEqual(original, "/control/cds")
        self.assertEqual(normalized, "/control/cds")
        self.assertEqual(scope["path"], "/control/cds")

    def test_setup_logging_writes_rotating_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            log_dir = Path(raw)
            logger = dlna_server.setup_logging(log_dir, max_bytes=2048, backup_count=1)
            logger.info("test log entry")
            for handler in logger.handlers:
                handler.flush()

            log_path = log_dir / "dlna_server.log"
            self.assertTrue(log_path.exists())
            self.assertIn("test log entry", log_path.read_text(encoding="utf-8"))
            self.assertTrue(any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers))
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    unittest.main()
