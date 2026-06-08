from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

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
        self.assertIn((("POST",), "/admin/reload_si_config"), routes)

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
