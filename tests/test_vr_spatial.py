from __future__ import annotations

import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from utils import vr_spatial
from utils.vr_spatial import SpatialMetadataError


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _full(kind: bytes, body: bytes, version: int = 0, flags: int = 0) -> bytes:
    return _box(kind, struct.pack(">B3s", version, flags.to_bytes(3, "big")) + body)


_MDAT_MARKER = b"CHUNK-MARKER-BYTES"
_MDAT_MARKER_OFF = 16  # marker offset inside the mdat payload


def _avc1_entry(extra_children: bytes = b"") -> bytes:
    avcc = _box(b"avcC", b"\x01\x64\x00\x1f\xff\xe1\x00")
    payload = b"\x00" * 6 + struct.pack(">H", 1) + b"\x00" * 70 + avcc + extra_children
    return _box(b"avc1", payload)


def _video_trak(chunk_offset: int) -> bytes:
    stsd = _full(b"stsd", struct.pack(">I", 1) + _avc1_entry())
    stco = _full(b"stco", struct.pack(">II", 1, chunk_offset))
    stbl = _box(b"stbl", stsd + stco)
    minf = _box(b"minf", stbl)
    hdlr = _full(b"hdlr", struct.pack(">I4s", 0, b"vide") + b"\x00" * 12 + b"Video\x00")
    mdia = _box(b"mdia", _full(b"mdhd", b"\x00" * 20) + hdlr + minf)
    return _box(b"trak", _full(b"tkhd", b"\x00" * 80) + mdia)


def _audio_trak(chunk_offset: int) -> bytes:
    stco = _full(b"stco", struct.pack(">II", 1, chunk_offset))
    stbl = _box(b"stbl", stco)
    minf = _box(b"minf", stbl)
    hdlr = _full(b"hdlr", struct.pack(">I4s", 0, b"soun") + b"\x00" * 12 + b"Sound\x00")
    mdia = _box(b"mdia", _full(b"mdhd", b"\x00" * 20) + hdlr + minf)
    return _box(b"trak", _full(b"tkhd", b"\x00" * 80) + mdia)


def _write_mp4(path: Path, *, faststart: bool) -> int:
    """Build a tiny synthetic MP4; returns the absolute offset of the marker."""
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomavc1")
    mdat_payload = b"\x00" * _MDAT_MARKER_OFF + _MDAT_MARKER + b"\x00" * 8

    def moov_for(marker_abs: int) -> bytes:
        return _box(
            b"moov",
            _full(b"mvhd", b"\x00" * 96)
            + _audio_trak(marker_abs)
            + _video_trak(marker_abs),
        )

    if faststart:
        # moov size does not depend on the offsets it stores (fixed-width
        # fields), so build once with zeros to learn the layout.
        moov_size = len(moov_for(0))
        marker_abs = len(ftyp) + moov_size + 8 + _MDAT_MARKER_OFF
        data = ftyp + moov_for(marker_abs) + _box(b"mdat", mdat_payload)
    else:
        marker_abs = len(ftyp) + 8 + _MDAT_MARKER_OFF
        data = ftyp + _box(b"mdat", mdat_payload) + moov_for(marker_abs)
    path.write_bytes(data)
    return marker_abs


class VrSpatialSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_plain_file_has_no_atoms(self) -> None:
        path = self.dir / "plain.mp4"
        _write_mp4(path, faststart=False)
        self.assertEqual(vr_spatial.read_spatial_atoms(path), (None, None))

    def test_inject_and_read_back_moov_last(self) -> None:
        path = self.dir / "out.mp4"
        marker_abs = _write_mp4(path, faststart=False)
        st3d, sv3d = vr_spatial.vr180_lr_atoms()

        vr_spatial.inject_spatial_atoms(path, st3d, sv3d)

        self.assertEqual(vr_spatial.read_spatial_atoms(path), (st3d, sv3d))
        data = path.read_bytes()
        self.assertIn(b"avcC", data)  # existing sample-entry children survive
        # moov after mdat: chunk offsets must be untouched.
        self.assertEqual(data[marker_abs:marker_abs + len(_MDAT_MARKER)], _MDAT_MARKER)

    def test_inject_faststart_shifts_all_chunk_offsets(self) -> None:
        path = self.dir / "fast.mp4"
        _write_mp4(path, faststart=True)
        st3d, sv3d = vr_spatial.vr180_lr_atoms()

        vr_spatial.inject_spatial_atoms(path, st3d, sv3d)

        self.assertEqual(vr_spatial.read_spatial_atoms(path), (st3d, sv3d))
        data = path.read_bytes()
        # Both traks' stco values must now point at the marker again.
        offsets = []
        pos = data.find(b"stco")
        while pos != -1:
            offsets.append(struct.unpack_from(">I", data, pos + 12)[0])
            pos = data.find(b"stco", pos + 4)
        self.assertEqual(len(offsets), 2)
        for offset in offsets:
            self.assertEqual(data[offset:offset + len(_MDAT_MARKER)], _MDAT_MARKER)

    def test_reinjection_replaces_instead_of_duplicating(self) -> None:
        path = self.dir / "twice.mp4"
        _write_mp4(path, faststart=False)
        st3d, sv3d = vr_spatial.vr180_lr_atoms()
        vr_spatial.inject_spatial_atoms(path, st3d, sv3d)
        size_once = path.stat().st_size

        vr_spatial.inject_spatial_atoms(path, st3d, sv3d)

        self.assertEqual(path.stat().st_size, size_once)
        self.assertEqual(vr_spatial.read_spatial_atoms(path), (st3d, sv3d))

    def test_fragmented_file_is_refused(self) -> None:
        path = self.dir / "frag.mp4"
        _write_mp4(path, faststart=False)
        original = path.read_bytes() + _box(b"moof", _full(b"mfhd", b"\x00" * 4))
        path.write_bytes(original)
        st3d, sv3d = vr_spatial.vr180_lr_atoms()

        with self.assertRaises(SpatialMetadataError):
            vr_spatial.inject_spatial_atoms(path, st3d, sv3d)
        self.assertEqual(path.read_bytes(), original)


def _lr_st3d() -> bytes:
    return _full(b"st3d", struct.pack(">B", 2))


def _equi_sv3d(bounds=(1, 2, 3, 4)) -> bytes:
    equi = _full(b"equi", struct.pack(">IIII", *bounds))
    return _box(b"sv3d", _full(b"svhd", b"other\x00") + _box(b"proj", equi))


class SelectAtomsTests(unittest.TestCase):
    def test_no_source_atoms_means_no_tagging(self) -> None:
        self.assertIsNone(vr_spatial.select_atoms_for_sbs_output(None, None))

    def test_matching_pair_is_carried_together(self) -> None:
        source_st3d = _lr_st3d()
        source_sv3d = _equi_sv3d()
        self.assertEqual(
            vr_spatial.select_atoms_for_sbs_output(source_st3d, source_sv3d),
            (source_st3d, source_sv3d),
        )

    def test_st3d_without_sv3d_is_not_carried(self) -> None:
        self.assertIsNone(vr_spatial.select_atoms_for_sbs_output(_lr_st3d(), None))

    def test_sv3d_without_st3d_is_not_carried(self) -> None:
        self.assertIsNone(vr_spatial.select_atoms_for_sbs_output(None, _equi_sv3d()))

    def test_top_bottom_st3d_rejects_whole_pair(self) -> None:
        source_st3d = _full(b"st3d", struct.pack(">B", 1))
        self.assertIsNone(
            vr_spatial.select_atoms_for_sbs_output(source_st3d, _equi_sv3d())
        )

    def test_mesh_projection_rejects_whole_pair(self) -> None:
        mshp = _full(b"mshp", b"\x00" * 8)
        source_sv3d = _box(b"sv3d", _full(b"svhd", b"other\x00") + _box(b"proj", mshp))
        self.assertIsNone(vr_spatial.select_atoms_for_sbs_output(_lr_st3d(), source_sv3d))

    def test_sv3d_without_projection_rejects_whole_pair(self) -> None:
        source_sv3d = _box(b"sv3d", _full(b"svhd", b"other\x00"))
        self.assertIsNone(vr_spatial.select_atoms_for_sbs_output(_lr_st3d(), source_sv3d))


class TagSbsOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_source_atoms_are_carried_verbatim(self) -> None:
        source = self.dir / "source.mp4"
        output = self.dir / "output.mp4"
        _write_mp4(source, faststart=False)
        _write_mp4(output, faststart=False)
        st3d = _full(b"st3d", struct.pack(">B", 2))
        equi = _full(b"equi", struct.pack(">IIII", 0, 0, 7, 7))
        sv3d = _box(b"sv3d", _full(b"svhd", b"studio\x00") + _box(b"proj", equi))
        vr_spatial.inject_spatial_atoms(source, st3d, sv3d)

        logs: list[str] = []
        self.assertTrue(vr_spatial.tag_sbs_output(source, output, log_callback=logs.append))

        self.assertEqual(vr_spatial.read_spatial_atoms(output), (st3d, sv3d))
        self.assertTrue(any("carried source" in line for line in logs))

    def test_sourceless_output_stays_untagged(self) -> None:
        output = self.dir / "output.mp4"
        _write_mp4(output, faststart=False)
        before = output.read_bytes()

        self.assertFalse(vr_spatial.tag_sbs_output(None, output))

        self.assertEqual(output.read_bytes(), before)

    def test_source_without_atoms_leaves_output_untouched(self) -> None:
        source = self.dir / "plain_source.mp4"
        output = self.dir / "output.mp4"
        _write_mp4(source, faststart=False)
        _write_mp4(output, faststart=False)
        before = output.read_bytes()

        logs: list[str] = []
        self.assertFalse(vr_spatial.tag_sbs_output(source, output, log_callback=logs.append))

        self.assertEqual(output.read_bytes(), before)
        self.assertTrue(any("left untagged" in line for line in logs))

    def test_non_mp4_output_is_skipped(self) -> None:
        output = self.dir / "output.mkv"
        output.write_bytes(b"not an mp4")
        self.assertFalse(vr_spatial.tag_sbs_output(None, output))
        self.assertEqual(output.read_bytes(), b"not an mp4")


class ConfigBoolTests(unittest.TestCase):
    def setUp(self) -> None:
        from utils import app_config

        self.app_config = app_config
        app_config._load()
        self._old_cache = dict(app_config._cache)

    def tearDown(self) -> None:
        self.app_config._cache = self._old_cache

    def test_string_falsy_values_disable(self) -> None:
        for value in ("false", "0", "off", "no", "", "FALSE", " Off ", False, 0):
            self.app_config._cache["vr_spatial_metadata"] = value
            self.assertFalse(
                self.app_config.get_bool("vr_spatial_metadata", True),
                f"value {value!r} should disable",
            )

    def test_truthy_values_enable(self) -> None:
        for value in ("true", "1", "yes", "on", True, 1):
            self.app_config._cache["vr_spatial_metadata"] = value
            self.assertTrue(
                self.app_config.get_bool("vr_spatial_metadata", False),
                f"value {value!r} should enable",
            )

    def test_missing_key_uses_default(self) -> None:
        self.app_config._cache.pop("vr_spatial_metadata", None)
        self.assertTrue(self.app_config.get_bool("vr_spatial_metadata", True))
        self.assertFalse(self.app_config.get_bool("vr_spatial_metadata", False))

    def test_fisheye_delta_gate_respects_string_false(self) -> None:
        from gpu_engine.native_mosaic import fisheye_delta

        self.app_config._cache["native_fisheye_delta"] = "false"
        self.assertFalse(fisheye_delta.enabled())
        self.app_config._cache["native_fisheye_delta"] = "true"
        self.assertTrue(fisheye_delta.enabled())


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg/ffprobe not available",
)
class FfmpegIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_clip(self, name: str, faststart: bool) -> Path:
        path = self.dir / name
        cmd = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc=size=128x64:rate=10:duration=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
        ]
        if faststart:
            cmd += ["-movflags", "+faststart"]
        cmd += [str(path), "-y"]
        subprocess.run(cmd, check=True, capture_output=True)
        return path

    def _probe_side_data_types(self, path: Path) -> set[str]:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_streams", "-print_format", "json", str(path),
            ],
            check=True, capture_output=True, text=True,
        ).stdout
        streams = json.loads(out).get("streams", [])
        side_data = streams[0].get("side_data_list", []) if streams else []
        return {str(item.get("side_data_type", "")) for item in side_data}

    def _assert_decodable(self, path: Path) -> None:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.strip(), "")

    def test_carried_atoms_are_recognized_and_decodable(self) -> None:
        st3d, sv3d = vr_spatial.vr180_lr_atoms()
        for faststart in (False, True):
            with self.subTest(faststart=faststart):
                source = self._make_clip(f"src_{int(faststart)}.mp4", faststart)
                vr_spatial.inject_spatial_atoms(source, st3d, sv3d)  # a tagged VR source
                self._assert_decodable(source)
                output = self._make_clip(f"out_{int(faststart)}.mp4", faststart)
                self.assertTrue(vr_spatial.tag_sbs_output(source, output))
                types = self._probe_side_data_types(output)
                self.assertTrue(
                    any("Stereo 3D" in t for t in types),
                    f"missing Stereo 3D side data, got {types}",
                )
                self.assertTrue(
                    any("Spherical" in t for t in types),
                    f"missing Spherical Mapping side data, got {types}",
                )
                self._assert_decodable(output)

    def test_sourceless_clip_stays_untagged(self) -> None:
        clip = self._make_clip("plain.mp4", False)
        before = clip.read_bytes()
        self.assertFalse(vr_spatial.tag_sbs_output(None, clip))
        self.assertEqual(clip.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
