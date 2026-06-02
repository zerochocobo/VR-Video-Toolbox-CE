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
