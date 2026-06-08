from __future__ import annotations

import html
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_dlna import content_directory as cds
from tool_dlna.media_library import MediaLibrary, build_media_roots
from tool_dlna.si_stream import SIMixConfig


def _browse_body(object_id: str = "0", flag: str = "BrowseDirectChildren") -> bytes:
    return (
        b'<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
        b"<s:Body>"
        b'<u:Browse xmlns:u="urn:schemas-upnp-org:service:ContentDirectory:1">'
        + f"<ObjectID>{object_id}</ObjectID>".encode("utf-8")
        + f"<BrowseFlag>{flag}</BrowseFlag>".encode("utf-8")
        + b"<StartingIndex>0</StartingIndex><RequestedCount>0</RequestedCount>"
        b"</u:Browse></s:Body></s:Envelope>"
    )


class ContentDirectoryTests(unittest.TestCase):
    def test_didl_namespace_has_trailing_slash_and_subtitle_namespace(self) -> None:
        didl = cds._didl_for([])

        self.assertIn('xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/"', didl)
        self.assertIn('xmlns:sec="http://www.sec.co.kr/"', didl)

    def test_didl_includes_external_subtitles(self) -> None:
        didl = cds._didl_for(
            [
                {
                    "id": "v_movie.mp4",
                    "parent_id": "0",
                    "title": "movie",
                    "url": "http://127.0.0.1:8090/media/movie.mp4",
                    "thumb": "http://127.0.0.1:8090/thumb/movie.mp4",
                    "size": 1024,
                    "duration": 60.0,
                    "resolution": "1920x1080",
                    "bitrate": 1000,
                    "mime": "video/mp4",
                    "dlna_pn": "AVC_MP4_HP_HD_AAC",
                    "subtitles": [
                        {
                            "url": "http://127.0.0.1:8090/subs/movie.zh.srt",
                            "lang": "zh",
                            "type": "srt",
                            "mime": "application/x-subrip",
                        }
                    ],
                }
            ]
        )

        self.assertIn('protocolInfo="http-get:*:application/x-subrip:*" xml:lang="zh"', didl)
        self.assertIn("<sec:CaptionInfoEx sec:type=\"srt\">http://127.0.0.1:8090/subs/movie.zh.srt</sec:CaptionInfoEx>", didl)
        self.assertIn("<sec:CaptionInfo sec:type=\"srt\">http://127.0.0.1:8090/subs/movie.zh.srt</sec:CaptionInfo>", didl)

    def test_soap_parser_rejects_entity_declarations(self) -> None:
        body = b"""<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY x "boom">]>
<s:Envelope><s:Body><ObjectID>&x;</ObjectID></s:Body></s:Envelope>"""

        self.assertEqual(cds._parse_soap_args(body), {})

    def test_soap_parser_rejects_oversized_body(self) -> None:
        body = b"<Envelope>" + (b"x" * (cds._MAX_SOAP_BODY_BYTES + 1)) + b"</Envelope>"

        self.assertEqual(cds._parse_soap_args(body), {})

    def test_browse_root_lists_multi_root_virtual_folders(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            root1 = Path(d1)
            root2 = Path(d2)
            library = MediaLibrary(build_media_roots([root1, root2]))

            payload, status = cds.handle_soap(
                '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
                _browse_body(),
                "http://127.0.0.1:8090",
                library,
                True,
            )

        self.assertEqual(status, 200)
        text = html.unescape(payload.decode("utf-8"))
        self.assertIn(f"<dc:title>{root1.name}</dc:title>", text)
        self.assertIn(f"<dc:title>{root2.name}</dc:title>", text)

    def test_browse_nested_root_uses_nested_root_ids_for_children(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "Downloads"
            nested = root / "VR" / "VR110"
            child = nested / "fcvr-040"
            child.mkdir(parents=True)
            library = MediaLibrary(build_media_roots([root, nested]))

            payload, status = cds.handle_soap(
                '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
                _browse_body("d_VR110"),
                "http://127.0.0.1:8090",
                library,
                True,
            )

        self.assertEqual(status, 200)
        text = html.unescape(payload.decode("utf-8"))
        self.assertIn('id="d_VR110/fcvr-040"', text)
        self.assertNotIn('id="d_Downloads/VR/VR110/fcvr-040"', text)

    def test_browse_video_uses_virtual_lr_180_sbs_title_without_passthrough_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            library = MediaLibrary(build_media_roots([root]))

            with patch.object(
                cds,
                "probe_cached",
                return_value={"width": 3840, "height": 1920, "duration": 60.0, "size": 5, "bitrate": 1000},
            ):
                payload, status = cds.handle_soap(
                    '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
                    _browse_body(),
                    "http://127.0.0.1:8090",
                    library,
                    True,
                )

        self.assertEqual(status, 200)
        text = html.unescape(payload.decode("utf-8"))
        self.assertIn("<dc:title>movie_LR_180_SBS</dc:title>", text)
        self.assertIn("/media/movie.mp4", text)
        self.assertNotIn("passthrough", text.casefold())
        self.assertNotIn("/passthrough", text.casefold())

    def test_browse_lists_si_entry_when_enabled_and_wav_exists(self) -> None:
        class FakeSIService:
            def current_config(self):
                return SIMixConfig(enabled=True)

            def has_si_source(self, video):
                candidate = Path(video).with_suffix(".si.wav")
                return candidate if candidate.is_file() else None

            def estimate_output_size(self, _video):
                return 123456

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"wav")
            library = MediaLibrary(build_media_roots([root]))

            with patch.object(
                cds,
                "probe_cached",
                return_value={"width": 3840, "height": 1920, "duration": 60.0, "size": 5, "bitrate": 1000},
            ):
                payload, status = cds.handle_soap(
                    '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
                    _browse_body(),
                    "http://127.0.0.1:8090",
                    library,
                    True,
                    si_service=FakeSIService(),
                )

        self.assertEqual(status, 200)
        text = html.unescape(payload.decode("utf-8"))
        self.assertIn('id="v_movie.mp4"', text)
        self.assertIn('id="vs_movie.mp4"', text)
        self.assertIn("<dc:title>[SI] movie_LR_180_SBS</dc:title>", text)
        self.assertIn("/media_si/movie.mp4", text)
        self.assertIn("DLNA.ORG_CI=1", text)

    def test_browse_omits_si_entry_when_disabled(self) -> None:
        class FakeSIService:
            def current_config(self):
                return SIMixConfig(enabled=False)

            def has_si_source(self, video):
                return Path(video).with_suffix(".si.wav")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"wav")
            library = MediaLibrary(build_media_roots([root]))

            with patch.object(
                cds,
                "probe_cached",
                return_value={"width": 3840, "height": 1920, "duration": 60.0, "size": 5, "bitrate": 1000},
            ):
                payload, status = cds.handle_soap(
                    '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
                    _browse_body(),
                    "http://127.0.0.1:8090",
                    library,
                    True,
                    si_service=FakeSIService(),
                )

        self.assertEqual(status, 200)
        text = html.unescape(payload.decode("utf-8"))
        self.assertIn('id="v_movie.mp4"', text)
        self.assertNotIn('id="vs_movie.mp4"', text)

    def test_browse_metadata_for_si_entry(self) -> None:
        class FakeSIService:
            def current_config(self):
                return SIMixConfig(enabled=True)

            def has_si_source(self, video):
                return Path(video).with_suffix(".si.wav")

            def estimate_output_size(self, _video):
                return 123456

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"wav")
            library = MediaLibrary(build_media_roots([root]))

            with patch.object(
                cds,
                "probe_cached",
                return_value={"width": 3840, "height": 1920, "duration": 60.0, "size": 5, "bitrate": 1000},
            ):
                payload, status = cds.handle_soap(
                    '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
                    _browse_body("vs_movie.mp4", "BrowseMetadata"),
                    "http://127.0.0.1:8090",
                    library,
                    True,
                    si_service=FakeSIService(),
                )

        self.assertEqual(status, 200)
        text = html.unescape(payload.decode("utf-8"))
        self.assertIn('id="vs_movie.mp4"', text)
        self.assertIn("<dc:title>[SI] movie_LR_180_SBS</dc:title>", text)
        self.assertIn("/media_si/movie.mp4", text)

    def test_browse_metadata_for_disabled_si_entry_reports_zero_matches(self) -> None:
        class FakeSIService:
            def current_config(self):
                return SIMixConfig(enabled=False)

            def has_si_source(self, video):
                return Path(video).with_suffix(".si.wav")

            def estimate_output_size(self, _video):
                return 123456

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"wav")
            library = MediaLibrary(build_media_roots([root]))

            with patch.object(
                cds,
                "probe_cached",
                return_value={"width": 3840, "height": 1920, "duration": 60.0, "size": 5, "bitrate": 1000},
            ):
                payload, status = cds.handle_soap(
                    '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"',
                    _browse_body("vs_movie.mp4", "BrowseMetadata"),
                    "http://127.0.0.1:8090",
                    library,
                    True,
                    si_service=FakeSIService(),
                )

        self.assertEqual(status, 200)
        text = html.unescape(payload.decode("utf-8"))
        self.assertIn("<NumberReturned>0</NumberReturned>", text)
        self.assertIn("<TotalMatches>0</TotalMatches>", text)
        self.assertNotIn("<item", text)


if __name__ == "__main__":
    unittest.main()
