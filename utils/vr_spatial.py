"""Read and re-inject MP4 Spherical Video V2 atoms (st3d / sv3d).

One-click SBS outputs are VR180 left-right half-equirect videos.  VR players
such as the Quest native player rely on the Spherical Video V2 sample-entry
atoms to auto-detect the stereo layout, but every ffmpeg/NVENC mux hop in the
pipeline drops them.  After the final output is produced we copy the atoms
over from the source file, or synthesize canonical VR180 left-right atoms
when the source carries none.

Implemented from the Spherical Video V2 RFC (box layout is a public spec):
https://github.com/google/spatial-media/blob/master/docs/spherical-video-v2-rfc.md

Strategy: locate the absolute byte range of the video sample entry inside a
fully-loaded ``moov``, rebuild just that entry with the atoms appended, splice
it back and patch the size field of each ancestor box in place.  When ``moov``
precedes ``mdat`` (faststart files) every stco/co64 chunk offset is shifted by
the growth delta.  The file itself is rewritten box-by-box into a temp file
and swapped in atomically after verification.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

_MP4_SUFFIXES = {".mp4", ".mov", ".m4v"}
_VIDEO_ENTRY_KINDS = {
    b"avc1", b"avc3", b"hvc1", b"hev1", b"hvt1",
    b"dvh1", b"dvhe", b"av01", b"vp08", b"vp09", b"encv",
}
# Fixed VisualSampleEntry fields between the box header and the child boxes:
# 6 reserved + 2 data_reference_index + 70 visual fields.
_VISUAL_ENTRY_FIXED = 78
_CHUNK_TABLE_PARENTS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


class SpatialMetadataError(ValueError):
    """Raised when an MP4 cannot be parsed or safely rewritten."""


@dataclass(frozen=True)
class _TopBox:
    kind: bytes
    offset: int
    size: int


def _make_box(kind: bytes, payload: bytes) -> bytes:
    total = 8 + len(payload)
    if total >= 1 << 32:
        raise SpatialMetadataError(f"box {kind!r} too large to serialize")
    return struct.pack(">I4s", total, kind) + payload


def _full_box(kind: bytes, body: bytes, version: int = 0, flags: int = 0) -> bytes:
    return _make_box(kind, struct.pack(">B3s", version, flags.to_bytes(3, "big")) + body)


def _header_at(buf, pos: int, limit: int) -> tuple[bytes, int, int]:
    """Return (kind, header_len, box_end) for the box starting at ``pos``."""
    if pos + 8 > limit:
        raise SpatialMetadataError(f"truncated box header at offset {pos}")
    size32, kind = struct.unpack_from(">I4s", buf, pos)
    if size32 == 1:
        if pos + 16 > limit:
            raise SpatialMetadataError(f"truncated large box header at offset {pos}")
        size = struct.unpack_from(">Q", buf, pos + 8)[0]
        header_len = 16
    elif size32 == 0:
        # "Extends to end of enclosing scope" never appears inside moov in
        # practice; refuse rather than guess so we abort without touching data.
        raise SpatialMetadataError(f"box {kind!r} with size 0 at offset {pos}")
    else:
        size = size32
        header_len = 8
    if size < header_len or pos + size > limit:
        raise SpatialMetadataError(f"box {kind!r} has bad size {size} at offset {pos}")
    return kind, header_len, pos + size


def _iter_children(buf, lo: int, hi: int):
    """Yield (kind, header_len, start, end) for each box in buf[lo:hi]."""
    pos = lo
    while pos < hi:
        kind, header_len, end = _header_at(buf, pos, hi)
        yield kind, header_len, pos, end
        pos = end


def _find_child(buf, lo: int, hi: int, wanted: bytes):
    for kind, header_len, start, end in _iter_children(buf, lo, hi):
        if kind == wanted:
            return header_len, start, end
    return None


def _scan_top_level(path: Path) -> list[_TopBox]:
    boxes: list[_TopBox] = []
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        pos = 0
        while pos < file_size:
            fh.seek(pos)
            head = fh.read(16)
            if len(head) < 8:
                raise SpatialMetadataError(f"truncated top-level box at offset {pos}")
            size32, kind = struct.unpack_from(">I4s", head)
            if size32 == 1:
                if len(head) < 16:
                    raise SpatialMetadataError(f"truncated large top-level box at {pos}")
                size = struct.unpack_from(">Q", head, 8)[0]
            elif size32 == 0:
                size = file_size - pos
            else:
                size = size32
            if size < 8 or pos + size > file_size:
                raise SpatialMetadataError(f"top-level box {kind!r} has bad size {size}")
            boxes.append(_TopBox(kind=kind, offset=pos, size=size))
            pos += size
    return boxes


def _load_top_box(path: Path, boxes: list[_TopBox], wanted: bytes) -> bytes:
    for box in boxes:
        if box.kind == wanted:
            with path.open("rb") as fh:
                fh.seek(box.offset)
                data = fh.read(box.size)
            if len(data) != box.size:
                raise SpatialMetadataError(f"short read of {wanted!r}")
            return data
    raise SpatialMetadataError(f"file has no {wanted!r} box")


def _is_video_trak(moov: bytes, trak_lo: int, trak_hi: int, trak_header: int) -> bool:
    mdia = _find_child(moov, trak_lo + trak_header, trak_hi, b"mdia")
    if mdia is None:
        return False
    hdlr = _find_child(moov, mdia[1] + mdia[0], mdia[2], b"hdlr")
    if hdlr is None:
        return False
    header_len, start, end = hdlr
    # FullBox(4) + pre_defined(4) + handler_type(4)
    payload_lo = start + header_len
    if payload_lo + 12 > end:
        return False
    return moov[payload_lo + 8:payload_lo + 12] == b"vide"


def _locate_video_sample_entry(moov: bytes):
    """Find the first supported video sample entry inside moov.

    Returns (ancestors, entry_header_len, entry_lo, entry_hi) where
    ``ancestors`` is a list of (offset, header_len) size fields to patch when
    the entry's size changes, ordered outermost (moov itself) first.
    """
    limit = len(moov)
    kind, moov_header, _ = _header_at(moov, 0, limit)
    if kind != b"moov":
        raise SpatialMetadataError("buffer does not start with a moov box")
    for trak_kind, trak_header, trak_lo, trak_hi in _iter_children(moov, moov_header, limit):
        if trak_kind != b"trak" or not _is_video_trak(moov, trak_lo, trak_hi, trak_header):
            continue
        chain = [(0, moov_header), (trak_lo, trak_header)]
        lo, hi, header_len = trak_lo, trak_hi, trak_header
        for step in (b"mdia", b"minf", b"stbl", b"stsd"):
            found = _find_child(moov, lo + header_len, hi, step)
            if found is None:
                raise SpatialMetadataError(f"video trak has no {step!r} box")
            header_len, lo, hi = found
            chain.append((lo, header_len))
        # stsd payload: FullBox(4) + entry_count(4), then the entries.
        entries_lo = lo + header_len + 8
        for entry_kind, entry_header, entry_lo, entry_hi in _iter_children(moov, entries_lo, hi):
            if entry_kind in _VIDEO_ENTRY_KINDS:
                return chain, entry_header, entry_lo, entry_hi
        raise SpatialMetadataError("stsd has no supported video sample entry")
    raise SpatialMetadataError("moov has no video trak")


def _entry_spatial_atoms(moov: bytes, entry_header: int, entry_lo: int, entry_hi: int):
    st3d = sv3d = None
    children_lo = entry_lo + entry_header + _VISUAL_ENTRY_FIXED
    if children_lo > entry_hi:
        raise SpatialMetadataError("video sample entry shorter than VisualSampleEntry")
    for kind, _hl, start, end in _iter_children(moov, children_lo, entry_hi):
        if kind == b"st3d":
            st3d = bytes(moov[start:end])
        elif kind == b"sv3d":
            sv3d = bytes(moov[start:end])
    return st3d, sv3d


def read_spatial_atoms(path) -> tuple[bytes | None, bytes | None]:
    """Return the raw (st3d, sv3d) atoms of the first video track, if any."""
    path = Path(path)
    boxes = _scan_top_level(path)
    moov = _load_top_box(path, boxes, b"moov")
    _chain, entry_header, entry_lo, entry_hi = _locate_video_sample_entry(moov)
    return _entry_spatial_atoms(moov, entry_header, entry_lo, entry_hi)


def vr180_lr_atoms() -> tuple[bytes, bytes]:
    """Canonical atoms for left-right stereo, 180x180 degree equirect."""
    st3d = _full_box(b"st3d", struct.pack(">B", 2))  # 2 = left-right
    svhd = _full_box(b"svhd", b"VR Video Toolbox\x00")
    prhd = _full_box(b"prhd", struct.pack(">iii", 0, 0, 0))  # yaw/pitch/roll
    # equi bounds are 0.32 fixed-point crop amounts per edge; cropping a
    # quarter from the left and right edges leaves the central 180 degrees.
    equi = _full_box(b"equi", struct.pack(">IIII", 0, 0, 0x40000000, 0x40000000))
    sv3d = _make_box(b"sv3d", svhd + _make_box(b"proj", prhd + equi))
    return st3d, sv3d


def _st3d_is_left_right(st3d: bytes | None) -> bool:
    if not st3d:
        return False
    try:
        _kind, header_len, end = _header_at(st3d, 0, len(st3d))
    except SpatialMetadataError:
        return False
    # FullBox(4) + stereo_mode(1)
    return end - header_len >= 5 and st3d[header_len + 4] == 2


def _sv3d_is_equirect(sv3d: bytes | None) -> bool:
    if not sv3d:
        return False
    try:
        _kind, header_len, end = _header_at(sv3d, 0, len(sv3d))
        proj = _find_child(sv3d, header_len, end, b"proj")
        if proj is None:
            return False
        proj_header, proj_lo, proj_hi = proj
        return _find_child(sv3d, proj_lo + proj_header, proj_hi, b"equi") is not None
    except SpatialMetadataError:
        return False


def select_atoms_for_sbs_output(
    source_st3d: bytes | None,
    source_sv3d: bytes | None,
) -> tuple[bytes, bytes] | None:
    """Pick atoms to carry onto an SBS output, or None to leave it untagged.

    The pair is carried only as a whole: the source st3d must describe
    left-right stereo AND the source sv3d must describe an equirectangular
    projection, matching what the SBS pipeline actually outputs.  Anything
    else — a source without spatial metadata, top-bottom stereo, or
    mesh/cubemap projections — yields None: the output mirrors the source
    and gets no atoms, because the pipeline cannot verify the content
    actually is VR180 left-right, and stamping it would change playback
    behavior relative to the source.
    """
    if _st3d_is_left_right(source_st3d) and _sv3d_is_equirect(source_sv3d):
        return source_st3d, source_sv3d
    return None


def _rebuild_sample_entry(moov: bytes, entry_header: int, entry_lo: int, entry_hi: int,
                          st3d: bytes, sv3d: bytes) -> bytes:
    kind = moov[entry_lo + 4:entry_lo + 8]
    fixed_lo = entry_lo + entry_header
    children_lo = fixed_lo + _VISUAL_ENTRY_FIXED
    if children_lo > entry_hi:
        raise SpatialMetadataError("video sample entry shorter than VisualSampleEntry")
    kept = [
        bytes(moov[start:end])
        for child_kind, _hl, start, end in _iter_children(moov, children_lo, entry_hi)
        if child_kind not in {b"st3d", b"sv3d"}
    ]
    payload = bytes(moov[fixed_lo:children_lo]) + b"".join(kept) + st3d + sv3d
    return _make_box(kind, payload)


def _patch_box_size(moov: bytearray, offset: int, header_len: int, delta: int) -> None:
    if header_len == 16:
        old = struct.unpack_from(">Q", moov, offset + 8)[0]
        struct.pack_into(">Q", moov, offset + 8, old + delta)
        return
    old = struct.unpack_from(">I", moov, offset)[0]
    new = old + delta
    if not 8 <= new < 1 << 32:
        raise SpatialMetadataError("ancestor box size overflow while patching")
    struct.pack_into(">I", moov, offset, new)


def _shift_chunk_offsets(moov: bytearray, lo: int, hi: int, delta: int) -> None:
    for kind, header_len, start, end in _iter_children(moov, lo, hi):
        if kind in {b"stco", b"co64"}:
            table_lo = start + header_len + 4  # skip FullBox
            if table_lo + 4 > end:
                raise SpatialMetadataError(f"truncated {kind!r} box")
            count = struct.unpack_from(">I", moov, table_lo)[0]
            width, fmt = (4, ">I") if kind == b"stco" else (8, ">Q")
            if table_lo + 4 + count * width > end:
                raise SpatialMetadataError(f"{kind!r} entry table overruns its box")
            for index in range(count):
                pos = table_lo + 4 + index * width
                value = struct.unpack_from(fmt, moov, pos)[0] + delta
                if not 0 <= value < 1 << (width * 8):
                    raise SpatialMetadataError("chunk offset overflow while shifting")
                struct.pack_into(fmt, moov, pos, value)
        elif kind in _CHUNK_TABLE_PARENTS:
            _shift_chunk_offsets(moov, start + header_len, end, delta)


def inject_spatial_atoms(path, st3d: bytes, sv3d: bytes) -> None:
    """Stamp st3d/sv3d onto the first video track of an MP4, atomically."""
    path = Path(path)
    boxes = _scan_top_level(path)
    if any(box.kind == b"moof" for box in boxes):
        raise SpatialMetadataError("fragmented MP4 files are not supported")
    moov_top = next((box for box in boxes if box.kind == b"moov"), None)
    if moov_top is None:
        raise SpatialMetadataError("file has no moov box")
    moov = _load_top_box(path, boxes, b"moov")

    chain, entry_header, entry_lo, entry_hi = _locate_video_sample_entry(moov)
    new_entry = _rebuild_sample_entry(moov, entry_header, entry_lo, entry_hi, st3d, sv3d)
    delta = len(new_entry) - (entry_hi - entry_lo)
    new_moov = bytearray(moov[:entry_lo] + new_entry + moov[entry_hi:])
    for offset, header_len in chain:
        _patch_box_size(new_moov, offset, header_len, delta)

    mdat_top = next((box for box in boxes if box.kind == b"mdat"), None)
    if delta and mdat_top is not None and moov_top.offset < mdat_top.offset:
        _kind, moov_header, _ = _header_at(new_moov, 0, len(new_moov))
        _shift_chunk_offsets(new_moov, moov_header, len(new_moov), delta)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.vrmeta-", suffix=path.suffix
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, path.open("rb") as src:
            for box in boxes:
                if box.kind == b"moov":
                    out.write(new_moov)
                    continue
                src.seek(box.offset)
                remaining = box.size
                while remaining:
                    chunk = src.read(min(remaining, 32 * 1024 * 1024))
                    if not chunk:
                        raise SpatialMetadataError("unexpected end of file while copying")
                    out.write(chunk)
                    remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        got_st3d, got_sv3d = read_spatial_atoms(tmp_path)
        if got_st3d != st3d or got_sv3d != sv3d:
            raise SpatialMetadataError("verification re-read did not match injected atoms")
        try:
            shutil.copymode(path, tmp_path)
        except OSError:
            pass
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def tag_sbs_output(source_path, output_path, log_callback=None) -> bool:
    """Carry VR spatial metadata from ``source_path`` onto ``output_path``.

    Best used on finished SBS left-right outputs.  The output mirrors the
    source: when the source has no (or an incompatible) st3d/sv3d pair the
    output is left untagged.  Returns True when the output was stamped;
    raises SpatialMetadataError when the output cannot be rewritten (the
    output file is left untouched in that case).
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() not in _MP4_SUFFIXES:
        if log_callback:
            log_callback(f"[vr-meta] skipping {output_path.suffix or 'extensionless'} output")
        return False

    source_st3d = source_sv3d = None
    source_path = Path(source_path) if source_path else None
    if source_path and source_path.suffix.lower() in _MP4_SUFFIXES and source_path.is_file():
        try:
            source_st3d, source_sv3d = read_spatial_atoms(source_path)
        except SpatialMetadataError as exc:
            if log_callback:
                log_callback(f"[vr-meta] could not read source atoms: {exc}")

    atoms = select_atoms_for_sbs_output(source_st3d, source_sv3d)
    if atoms is None:
        if log_callback:
            log_callback(
                "[vr-meta] source has no left-right equirect st3d/sv3d; output left untagged"
            )
        return False
    inject_spatial_atoms(output_path, *atoms)
    if log_callback:
        log_callback(f"[vr-meta] carried source st3d/sv3d onto {output_path.name}")
    return True
