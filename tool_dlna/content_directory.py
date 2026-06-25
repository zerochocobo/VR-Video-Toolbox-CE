"""UPnP ContentDirectory SOAP handler for browsing the local media library.

Handles XML metadata packaging for directories, videos, and associated subtitles.
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from tool_dlna import subtitles, vr_naming
from tool_dlna.firewall import hidden_subprocess_kwargs
from tool_dlna.media_library import safe_resolve_path

ROOT_ID = "0"
FOLDER_PREFIX = "d_"
VIDEO_PREFIX = "v_"
VIDEO_SI_PREFIX = "vs_"
SI_CHAPTER_PREFIX = "vsc_"
SI_TIME_INDEX_PREFIX = "vst_"
SI_TIME_GROUP_PREFIX = "vsg_"
SI_TIME_MINUTE_PREFIX = "vsm_"
SI_TIME_POINT_PREFIX = "vsp_"
DLNA_FLAGS_BASE = "01700000000000000000000000000000"
DLNA_FLAGS_TIME_SEEK = "41700000000000000000000000000000"
DLNA_OP_TIME_SEEK = "10"
SI_LIVE_CHAPTER_MAX_ITEMS = 10
SI_LIVE_CHAPTER_MIN_INTERVAL_SEC = 600
SI_TIME_INDEX_GROUP_SEC = 600
SI_TIME_INDEX_MINUTE_SEC = 60
SI_TIME_INDEX_POINT_SEC = 5
CDS_CLIENT_DEOVR = "deovr"
_SELECT_TIME_INDEX_LABELS = {
    "en_US": "Select Time Index",
    "zh_CN": "选择时间索引",
    "ja_JP": "時間インデックス選択",
}

_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}
_probe_cache: dict[str, dict] = {}
_SOAP_RE = re.compile(r"<([\w:]+)>([\s\S]*?)</\1>")
_MAX_SOAP_BODY_BYTES = 1024 * 1024
_UNSAFE_XML_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
log = logging.getLogger("vrtoolbox.dlna")


def _fmt_duration(sec: float) -> str:
    if sec <= 0:
        return "0:00:00.000"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:06.3f}"


def _fmt_title_time(sec: int) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h:02d}:{m:02d}"


def _fmt_index_time(sec: int, force_hours: bool = False) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if force_hours or h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _normalise_ui_language(language: str | None) -> str:
    value = str(language or "").strip().lower().replace("-", "_")
    if value.startswith("zh"):
        return "zh_CN"
    if value.startswith("ja"):
        return "ja_JP"
    return "en_US"


def _select_time_index_label(language: str | None = None) -> str:
    return _SELECT_TIME_INDEX_LABELS[_normalise_ui_language(language)]


def _is_deovr_cds_client(client_profile: str | None) -> bool:
    return str(client_profile or "").strip().lower() == CDS_CLIENT_DEOVR


def _si_live_route_hint_suffix(client_profile: str | None = None) -> str:
    return "" if _is_deovr_cds_client(client_profile) else ".ts"


def _duration_seconds(duration: float) -> int:
    return max(0, int(math.ceil(float(duration or 0.0))))


def _si_chapter_offsets(duration: float) -> list[int]:
    max_items = max(1, int(SI_LIVE_CHAPTER_MAX_ITEMS))
    min_interval = max(1, int(SI_LIVE_CHAPTER_MIN_INTERVAL_SEC))
    if duration <= min_interval or max_items == 1:
        return [0]
    duration_sec = _duration_seconds(duration)
    raw_interval = int(math.ceil(duration_sec / max_items))
    interval_sec = max(min_interval, int(math.ceil(raw_interval / 60.0)) * 60)
    offsets: list[int] = []
    offset = 0
    while len(offsets) < max_items and offset < duration_sec:
        if duration_sec - offset <= 60 and offset != 0:
            break
        offsets.append(offset)
        offset += interval_sec
    return offsets or [0]


def _si_time_group_ranges(duration: float) -> list[tuple[int, int]]:
    duration_sec = _duration_seconds(duration)
    if duration_sec <= 0:
        return [(0, 0)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < duration_sec:
        end = min(start + SI_TIME_INDEX_GROUP_SEC, duration_sec)
        ranges.append((start, end))
        start += SI_TIME_INDEX_GROUP_SEC
    return ranges or [(0, 0)]


def _si_time_minute_offsets(start: int, end: int) -> list[int]:
    start = max(0, int(start))
    end = max(start, int(end))
    offsets = list(range(start, end, SI_TIME_INDEX_MINUTE_SEC))
    return offsets or [start]


def _si_time_point_offsets(minute_start: int, duration: float) -> list[int]:
    duration_sec = _duration_seconds(duration)
    minute_start = max(0, int(minute_start))
    if duration_sec <= 0:
        return [minute_start]
    if minute_start >= duration_sec:
        return []
    minute_end = min(minute_start + SI_TIME_INDEX_MINUTE_SEC, duration_sec)
    offsets = list(range(minute_start, minute_end, SI_TIME_INDEX_POINT_SEC))
    return offsets or [minute_start]


def _si_time_force_hours(duration: float) -> bool:
    return _duration_seconds(duration) >= 3600


def _si_time_index_child_count(duration: float, level: str, start: int = 0, end: int = 0) -> int:
    if level == "index":
        groups = _si_time_group_ranges(duration)
        if len(groups) == 1:
            return len(_si_time_minute_offsets(*groups[0]))
        return len(groups)
    if level == "group":
        return len(_si_time_minute_offsets(start, end))
    if level == "minute":
        return len(_si_time_point_offsets(start, duration))
    return 0


def _si_directory_child_count(duration: float) -> int:
    return len(_si_chapter_offsets(duration)) + 1


def _probe_video(path: Path) -> dict:
    """Execute ffprobe to extract size, resolution, duration, and bitrate."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,bit_rate,size:format=duration,size,bit_rate",
        "-of", "json",
        str(path)
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=5, **hidden_subprocess_kwargs())
        if r.returncode == 0 and r.stdout:
            stdout_str = r.stdout.decode('utf-8', errors='ignore')
            data = json.loads(stdout_str)
            streams = data.get("streams", [{}])
            fmt = data.get("format", {})

            width = int(streams[0].get("width") or 0)
            height = int(streams[0].get("height") or 0)
            duration = float(streams[0].get("duration") or fmt.get("duration") or 0.0)
            size = int(fmt.get("size") or path.stat().st_size)
            bitrate = int(streams[0].get("bit_rate") or fmt.get("bit_rate") or 0)
            if bitrate <= 0 and duration > 0:
                bitrate = int(size * 8 / duration)
            video_size = int(streams[0].get("size") or 0)
            if video_size <= 0 and duration > 0 and bitrate > 0:
                video_size = int(bitrate * duration / 8)
            if video_size <= 0:
                video_size = size
            if size > 0:
                video_size = min(video_size, size)

            return {
                "width": width,
                "height": height,
                "duration": duration,
                "size": size,
                "video_size": video_size,
                "bitrate": bitrate
            }
    except Exception:
        pass
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {"width": 0, "height": 0, "duration": 0.0, "size": size, "video_size": size, "bitrate": 0}


def probe_cached(path: Path) -> dict:
    """Retrieve metadata from cache or probe the video file."""
    key = str(safe_resolve_path(path))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0

    cached = _probe_cache.get(key)
    if cached is not None and cached["mtime"] == mtime:
        return cached["data"]

    data = _probe_video(path)
    _probe_cache[key] = {"mtime": mtime, "data": data}
    return data


def _get_dlna_pn(path: Path) -> str:
    """Return matching DLNA profile name based on file extension."""
    ext = path.suffix.lower()
    if ext == ".mkv":
        return "MATROSKA"
    return "AVC_MP4_HP_HD_AAC"


def _get_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".mkv":
        return "video/x-matroska"
    return "video/mp4"


def _si_live_protocol_info(client_profile: str | None = None) -> str:
    if _is_deovr_cds_client(client_profile):
        return (
            "http-get:*:video/MP2T:DLNA.ORG_PN=HEVC_TS_NA_ISO;"
            f"DLNA.ORG_OP={DLNA_OP_TIME_SEEK};"
            f"DLNA.ORG_CI=1;DLNA.ORG_FLAGS={DLNA_FLAGS_TIME_SEEK}"
        )
    return (
        "http-get:*:video/MP2T:"
        "DLNA.ORG_PN=HEVC_TS_NA_ISO;DLNA.ORG_OP=00;DLNA.ORG_CI=1"
    )


def _si_id(prefix: str, rel_key: str, extra: object | None = None) -> str:
    suffix = "" if extra is None else f"@{extra}"
    return f"{prefix}{rel_key}{suffix}"


def _split_si_id(object_id: str, prefix: str, *, expect_extra: bool) -> tuple[str, str] | None:
    if not object_id.startswith(prefix):
        return None
    rest = object_id[len(prefix):]
    if not expect_extra:
        return rest.replace("\\", "/").strip("/"), ""
    rel, sep, extra = rest.rpartition("@")
    if not sep:
        return None
    return rel.replace("\\", "/").strip("/"), extra.strip()


def _si_title(title: str) -> str:
    return f"[SI] {title}"


def _si_time_index_title_for_language(title: str, language: str | None = None) -> str:
    return f"[{_select_time_index_label(language)}]_{_si_title(title)}"


def _si_play_leaf(
    *,
    base_url: str,
    rel_key: str,
    item_id: str,
    parent_id: str,
    title: str,
    offset: int,
    meta: dict,
    client_profile: str | None = None,
) -> dict:
    quoted_key = quote(rel_key)
    duration = float(meta.get("duration") or 0.0)
    remaining = max(0.0, duration - float(offset)) if duration > 0 else 0.0
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    omit_filelike_attrs = not _is_deovr_cds_client(client_profile)
    return {
        "id": item_id,
        "parent_id": parent_id,
        "title": title,
        "url": f"{base_url}/si_live/{quoted_key}{_si_live_route_hint_suffix(client_profile)}?t={int(offset)}",
        "thumb": f"{base_url}/thumb/{quoted_key}",
        "size": 0,
        "duration": remaining,
        "resolution": f"{width}x{height}" if width > 0 and height > 0 else "",
        "bitrate": int(meta.get("bitrate") or 0),
        "mime": "video/MP2T",
        "dlna_pn": "HEVC_TS_NA_ISO",
        "protocol_info": _si_live_protocol_info(client_profile),
        "omit_duration": omit_filelike_attrs,
        "omit_bitrate": omit_filelike_attrs,
        "subtitles": [],
    }


def _si_time_index_root_item(
    rel_key: str,
    parent_id: str,
    title: str,
    duration: float,
    language: str | None = None,
) -> dict:
    return {
        "container": True,
        "id": _si_id(SI_TIME_INDEX_PREFIX, rel_key),
        "parent_id": parent_id,
        "title": _si_time_index_title_for_language(title, language),
        "child_count": _si_time_index_child_count(duration, "index"),
    }


def _si_container_item(rel_key: str, parent_id: str, title: str, duration: float) -> dict:
    return {
        "container": True,
        "id": _si_id(VIDEO_SI_PREFIX, rel_key),
        "parent_id": parent_id,
        "title": _si_title(title),
        "child_count": _si_directory_child_count(duration),
    }


def _si_chapter_items(
    path: Path,
    rel_key: str,
    parent_id: str,
    base_url: str,
    client_profile: str | None = None,
    language: str | None = None,
) -> list[dict]:
    meta = probe_cached(path)
    duration = float(meta.get("duration") or 0.0)
    title = vr_naming.source_display_stem(path.stem, int(meta.get("width") or 0), int(meta.get("height") or 0))
    items = [_si_time_index_root_item(rel_key, parent_id, title, duration, language)]
    for offset in _si_chapter_offsets(duration):
        items.append(
            _si_play_leaf(
                base_url=base_url,
                rel_key=rel_key,
                item_id=_si_id(SI_CHAPTER_PREFIX, rel_key, int(offset)),
                parent_id=parent_id,
                title=f"{_fmt_title_time(offset)}_{_si_title(title)}",
                offset=int(offset),
                meta=meta,
                client_profile=client_profile,
            )
        )
    return items


def _si_time_minute_items(
    *,
    rel_key: str,
    parent_id: str,
    title: str,
    duration: float,
    start: int,
    end: int,
) -> list[dict]:
    force_hours = _si_time_force_hours(duration)
    return [
        {
            "container": True,
            "id": _si_id(SI_TIME_MINUTE_PREFIX, rel_key, int(minute)),
            "parent_id": parent_id,
            "title": f"{_fmt_index_time(minute, force_hours)}_{_si_title(title)}",
            "child_count": _si_time_index_child_count(duration, "minute", int(minute)),
        }
        for minute in _si_time_minute_offsets(start, end)
    ]


def _si_time_index_items(
    path: Path,
    rel_key: str,
    level: str,
    base_url: str,
    *,
    start: int = 0,
    end: int = 0,
    client_profile: str | None = None,
) -> list[dict]:
    meta = probe_cached(path)
    duration = float(meta.get("duration") or 0.0)
    title = vr_naming.source_display_stem(path.stem, int(meta.get("width") or 0), int(meta.get("height") or 0))
    force_hours = _si_time_force_hours(duration)

    if level == "index":
        parent_id = _si_id(SI_TIME_INDEX_PREFIX, rel_key)
        groups = _si_time_group_ranges(duration)
        if len(groups) == 1:
            group_start, group_end = groups[0]
            return _si_time_minute_items(
                rel_key=rel_key,
                parent_id=parent_id,
                title=title,
                duration=duration,
                start=group_start,
                end=group_end,
            )
        return [
            {
                "container": True,
                "id": _si_id(SI_TIME_GROUP_PREFIX, rel_key, f"{group_start}-{group_end}"),
                "parent_id": parent_id,
                "title": (
                    f"{_fmt_index_time(group_start, force_hours)}-{_fmt_index_time(group_end, force_hours)}"
                    f"_{_si_title(title)}"
                ),
                "child_count": _si_time_index_child_count(duration, "group", group_start, group_end),
            }
            for group_start, group_end in groups
        ]

    if level == "group":
        parent_id = _si_id(SI_TIME_GROUP_PREFIX, rel_key, f"{int(start)}-{int(end)}")
        return _si_time_minute_items(
            rel_key=rel_key,
            parent_id=parent_id,
            title=title,
            duration=duration,
            start=start,
            end=end,
        )

    if level != "minute":
        return []

    parent_id = _si_id(SI_TIME_MINUTE_PREFIX, rel_key, int(start))
    return [
        _si_play_leaf(
            base_url=base_url,
            rel_key=rel_key,
            item_id=_si_id(SI_TIME_POINT_PREFIX, rel_key, int(offset)),
            parent_id=parent_id,
            title=f"{_fmt_index_time(offset, force_hours)}_{_si_title(title)}",
            offset=int(offset),
            meta=meta,
            client_profile=client_profile,
        )
        for offset in _si_time_point_offsets(start, duration)
    ]


def _si_time_index_metadata_item(
    path: Path,
    rel_key: str,
    level: str,
    *,
    start: int = 0,
    end: int = 0,
    language: str | None = None,
) -> dict | None:
    meta = probe_cached(path)
    duration = float(meta.get("duration") or 0.0)
    title = vr_naming.source_display_stem(path.stem, int(meta.get("width") or 0), int(meta.get("height") or 0))
    force_hours = _si_time_force_hours(duration)
    if level == "index":
        return _si_time_index_root_item(rel_key, _si_id(VIDEO_SI_PREFIX, rel_key), title, duration, language)
    if level == "group":
        return {
            "container": True,
            "id": _si_id(SI_TIME_GROUP_PREFIX, rel_key, f"{int(start)}-{int(end)}"),
            "parent_id": _si_id(SI_TIME_INDEX_PREFIX, rel_key),
            "title": f"{_fmt_index_time(start, force_hours)}-{_fmt_index_time(end, force_hours)}_{_si_title(title)}",
            "child_count": _si_time_index_child_count(duration, "group", start, end),
        }
    if level == "minute":
        parent_id = _si_id(SI_TIME_INDEX_PREFIX, rel_key)
        groups = _si_time_group_ranges(duration)
        if len(groups) > 1:
            group_start = (max(0, int(start)) // SI_TIME_INDEX_GROUP_SEC) * SI_TIME_INDEX_GROUP_SEC
            group_end = min(group_start + SI_TIME_INDEX_GROUP_SEC, _duration_seconds(duration))
            parent_id = _si_id(SI_TIME_GROUP_PREFIX, rel_key, f"{group_start}-{group_end}")
        return {
            "container": True,
            "id": _si_id(SI_TIME_MINUTE_PREFIX, rel_key, int(start)),
            "parent_id": parent_id,
            "title": f"{_fmt_index_time(start, force_hours)}_{_si_title(title)}",
            "child_count": _si_time_index_child_count(duration, "minute", start),
        }
    return None


def _si_point_metadata_item(
    path: Path,
    rel_key: str,
    prefix: str,
    offset: int,
    base_url: str,
    client_profile: str | None = None,
) -> dict | None:
    if prefix == SI_CHAPTER_PREFIX:
        candidates = _si_chapter_items(
            path,
            rel_key,
            _si_id(VIDEO_SI_PREFIX, rel_key),
            base_url,
            client_profile=client_profile,
        )
    else:
        minute_start = (max(0, int(offset)) // SI_TIME_INDEX_MINUTE_SEC) * SI_TIME_INDEX_MINUTE_SEC
        candidates = _si_time_index_items(
            path,
            rel_key,
            "minute",
            base_url,
            start=minute_start,
            client_profile=client_profile,
        )
    item_id = _si_id(prefix, rel_key, int(offset))
    for item in candidates:
        if item.get("id") == item_id:
            return item
    return None


def _didl_for(items: list[dict]) -> str:
    """Generate DIDL-Lite XML string for a list of items."""
    out = [
        '<DIDL-Lite '
        'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/" '
        'xmlns:dlna="urn:schemas-dlna-org:metadata-1-0/" '
        'xmlns:sec="http://www.sec.co.kr/">'
    ]
    for it in items:
        title = html.escape(it["title"])
        parent_id = html.escape(it.get("parent_id", ROOT_ID))
        if it.get("container"):
            out.append(
                f'<container id="{html.escape(it["id"])}" parentID="{parent_id}" '
                f'childCount="{int(it.get("child_count", 0))}" restricted="1">'
                f"<dc:title>{title}</dc:title>"
                f"<upnp:class>object.container.storageFolder</upnp:class>"
                f"</container>"
            )
            continue

        url = html.escape(it["url"])
        thumb = html.escape(it["thumb"])
        size = it["size"]
        duration = _fmt_duration(it["duration"])
        resolution = it["resolution"]
        bitrate = it["bitrate"]
        mime = it["mime"]

        dlna_ci = int(it.get("dlna_ci", 0))
        proto = it.get("protocol_info") or (
            f"http-get:*:{mime}:DLNA.ORG_PN={it['dlna_pn']};"
            f"DLNA.ORG_OP=01;DLNA.ORG_CI={dlna_ci};DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"
        )

        attrs: list[str] = []
        if size > 0:
            attrs.append(f'size="{size}"')
        if it["duration"] > 0 and not it.get("omit_duration"):
            attrs.append(f'duration="{duration}"')
        if bitrate > 0 and not it.get("omit_bitrate"):
            attrs.append(f'bitrate="{bitrate}"')
        if resolution:
            attrs.append(f'resolution="{resolution}"')
        attrs.append(f'protocolInfo="{proto}"')
        res_attrs = " ".join(attrs)

        subtitle_xml = []
        for sub in it.get("subtitles", []):
            sub_url = html.escape(sub["url"])
            sub_mime = html.escape(sub["mime"])
            sub_type = html.escape(sub["type"])
            lang = str(sub.get("lang") or "")
            lang_attr = f' xml:lang="{html.escape(lang)}"' if lang else ""
            subtitle_xml.append(f'<res protocolInfo="http-get:*:{sub_mime}:*"{lang_attr}>{sub_url}</res>')
            subtitle_xml.append(f'<sec:CaptionInfoEx sec:type="{sub_type}">{sub_url}</sec:CaptionInfoEx>')
            subtitle_xml.append(f'<sec:CaptionInfo sec:type="{sub_type}">{sub_url}</sec:CaptionInfo>')

        out.append(
            f'<item id="{html.escape(it["id"])}" parentID="{parent_id}" restricted="1">'
            f"<dc:title>{title}</dc:title>"
            f"<upnp:class>object.item.videoItem</upnp:class>"
            f'<upnp:albumArtURI dlna:profileID="JPEG_TN">{thumb}</upnp:albumArtURI>'
            f"<res {res_attrs}>{url}</res>"
            f"{''.join(subtitle_xml)}"
            f"</item>"
        )
    out.append("</DIDL-Lite>")
    return "".join(out)


def _child_count(path: Path) -> int:
    """Return count of directories and video files in the path."""
    try:
        count = 0
        for child in path.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_dir() or child.suffix.lower() in _VIDEO_EXTS:
                count += 1
        return count
    except Exception as e:
        log.warning("Cannot count DLNA children for %s: %s", path, e)
        return 0


def _get_items_for_dir(
    directory: Path,
    parent_id: str,
    base_url: str,
    media_library,
    subtitles_enabled: bool,
    si_service=None,
) -> list[dict]:
    items: list[dict] = []
    try:
        children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    except Exception as e:
        log.warning("Cannot browse DLNA directory %s: %s", directory, e)
        return items

    skipped = 0
    for child in children:
        try:
            if child.name.startswith("."):
                skipped += 1
                continue

            rel_key = media_library.path_to_key(child)
            quoted_key = quote(rel_key)

            if child.is_dir():
                items.append(
                    {
                        "container": True,
                        "id": f"{FOLDER_PREFIX}{rel_key}",
                        "parent_id": parent_id,
                        "title": child.name,
                        "child_count": _child_count(child),
                    }
                )
            elif child.suffix.lower() in _VIDEO_EXTS:
                meta = probe_cached(child)
                title = vr_naming.source_display_stem(child.stem, meta["width"], meta["height"])

                # Build subtitle tracks
                sub_list = []
                tracks = subtitles.find_external_subtitles(child, subtitles_enabled, media_library)
                for track in tracks:
                    try:
                        sub_rel = media_library.path_to_key(track.path)
                        sub_list.append(
                            {
                                "url": f"{base_url}/subs/{quote(sub_rel)}",
                                "lang": track.lang,
                                "type": track.kind,
                                "mime": track.mime,
                            }
                        )
                    except Exception:
                        pass

                items.append(
                    {
                        "id": f"{VIDEO_PREFIX}{rel_key}",
                        "parent_id": parent_id,
                        "title": title,
                        "url": f"{base_url}/media/{quoted_key}",
                        "thumb": f"{base_url}/thumb/{quoted_key}",
                        "size": meta["size"],
                        "duration": meta["duration"],
                        "resolution": f"{meta['width']}x{meta['height']}" if meta["width"] > 0 else "",
                        "bitrate": meta["bitrate"],
                        "mime": _get_mime(child),
                        "dlna_pn": _get_dlna_pn(child),
                        "subtitles": sub_list,
                    }
                )
                try:
                    si_config = si_service.current_config() if si_service is not None else None
                    si_source = si_service.has_si_source(child) if si_config is not None and si_config.enabled else None
                except Exception as e:
                    log.warning("Cannot inspect SI source for %s: %s", child, e)
                    si_config = None
                    si_source = None
                if si_config is not None and si_config.enabled and si_source is not None:
                    items.append(
                        _si_container_item(
                            rel_key,
                            parent_id,
                            title,
                            float(meta.get("duration") or 0.0),
                        )
                    )
            else:
                skipped += 1
        except Exception as e:
            log.warning("Cannot add DLNA child %s from %s: %s", child, directory, e)
    log.info(
        "CDS directory scan path=%s scanned=%d returned=%d skipped=%d",
        directory,
        len(children),
        len(items),
        skipped,
    )
    return items


def _log_browse_item_preview(object_id: str, items: list[dict]) -> None:
    if not items:
        log.info("CDS Browse items object_id=%s: (empty)", object_id)
        return
    preview_parts = []
    for item in items[:20]:
        kind = "dir" if item.get("container") else "video"
        preview_parts.append(f"{kind}:{item.get('id', '')}({item.get('title', '')})")
    if len(items) > 20:
        preview_parts.append(f"... +{len(items) - 20} more")
    log.info("CDS Browse items object_id=%s: %s", object_id, " | ".join(preview_parts))


def _wrap_soap(action: str, body_xml: str) -> bytes:
    env = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        f'<u:{action}Response xmlns:u="urn:schemas-upnp-org:service:ContentDirectory:1">'
        f"{body_xml}"
        f"</u:{action}Response>"
        "</s:Body></s:Envelope>"
    )
    return env.encode("utf-8")


def _metadata_didl_for_dir(directory: Path, object_id: str, parent_id: str, media_library) -> str:
    title = "VR Video Server"
    if object_id != ROOT_ID:
        title = directory.name
    return (
        f'<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
        f'<container id="{html.escape(object_id)}" parentID="{html.escape(parent_id)}" '
        f'childCount="{_child_count(directory)}" restricted="1">'
        f"<dc:title>{html.escape(title)}</dc:title>"
        f"<upnp:class>object.container.storageFolder</upnp:class>"
        f"</container></DIDL-Lite>"
    )


def _parse_soap_args(body: bytes) -> dict:
    import xml.etree.ElementTree as ET
    if len(body) > _MAX_SOAP_BODY_BYTES:
        return {}
    text = body.decode("utf-8", errors="ignore")
    if _UNSAFE_XML_RE.search(text):
        return {}
    args: dict = {}
    try:
        root = ET.fromstring(text)
        for elem in root.iter():
            tag = elem.tag.rsplit("}", 1)[-1].split(":")[-1]
            value = (elem.text or "").strip()
            if value:
                args[tag] = value
        return args
    except ET.ParseError:
        pass
    for match in _SOAP_RE.finditer(text):
        tag = match.group(1).split(":")[-1]
        value = match.group(2).strip()
        if value:
            args[tag] = value
    return args


def handle_soap(
    soap_action: str,
    body: bytes,
    base_url: str,
    media_library,
    subtitles_enabled: bool,
    si_service=None,
    client_profile: str | None = None,
    language: str | None = None,
) -> tuple[bytes, int]:
    """Parse SOAP action and compile matching Browse result XML."""
    action = soap_action.strip('"').split("#")[-1]
    args = _parse_soap_args(body)

    if action == "Browse":
        object_id = args.get("ObjectID", ROOT_ID)
        flag = args.get("BrowseFlag", "BrowseDirectChildren")
        start = int(args.get("StartingIndex", "0") or 0)
        count = int(args.get("RequestedCount", "0") or 0)
        log.info("CDS Browse object_id=%s flag=%s start=%s count=%s", object_id, flag, start, count)

        # 1. ObjectID mapping to directory
        directory = None
        parent_id = "0"

        if object_id == ROOT_ID:
            if not media_library.multi_root:
                directory = media_library.first_root.path
                parent_id = "-1"
        elif object_id.startswith(FOLDER_PREFIX):
            rel = object_id[len(FOLDER_PREFIX):]
            directory = media_library.key_to_path(rel)
            if directory is not None:
                try:
                    parent_rel = media_library.path_to_key(directory.parent)
                    if not parent_rel or parent_rel == ".":
                        parent_id = ROOT_ID
                    else:
                        parent_id = f"{FOLDER_PREFIX}{parent_rel}"
                except ValueError:
                    parent_id = ROOT_ID

        # 2. ObjectID mapping to single video item
        video_file = None
        video_rel = ""
        is_si_video = False
        si_object_kind = ""
        si_time_level = ""
        si_start = 0
        si_end = 0
        si_point_prefix = ""
        if object_id.startswith(VIDEO_PREFIX):
            rel = object_id[len(VIDEO_PREFIX):]
            video_rel = rel
            video_file = media_library.key_to_path(rel)
            if video_file is not None:
                try:
                    parent_rel = media_library.path_to_key(video_file.parent)
                    if not parent_rel or parent_rel == ".":
                        parent_id = ROOT_ID
                    else:
                        parent_id = f"{FOLDER_PREFIX}{parent_rel}"
                except ValueError:
                    parent_id = ROOT_ID
        elif object_id.startswith(VIDEO_SI_PREFIX):
            rel = object_id[len(VIDEO_SI_PREFIX):]
            video_rel = rel
            is_si_video = True
            si_object_kind = "dir"
            video_file = media_library.key_to_path(rel)
            if video_file is not None:
                try:
                    parent_rel = media_library.path_to_key(video_file.parent)
                    if not parent_rel or parent_rel == ".":
                        parent_id = ROOT_ID
                    else:
                        parent_id = f"{FOLDER_PREFIX}{parent_rel}"
                except ValueError:
                    parent_id = ROOT_ID
        else:
            parsed = _split_si_id(object_id, SI_TIME_INDEX_PREFIX, expect_extra=False)
            if parsed is not None:
                video_rel = parsed[0]
                video_file = media_library.key_to_path(video_rel)
                is_si_video = True
                si_object_kind = "time"
                si_time_level = "index"
            else:
                for prefix, level in (
                    (SI_TIME_GROUP_PREFIX, "group"),
                    (SI_TIME_MINUTE_PREFIX, "minute"),
                ):
                    parsed = _split_si_id(object_id, prefix, expect_extra=True)
                    if parsed is None:
                        continue
                    video_rel, extra = parsed
                    video_file = media_library.key_to_path(video_rel)
                    is_si_video = True
                    si_object_kind = "time"
                    si_time_level = level
                    try:
                        if level == "group":
                            left, right = extra.split("-", 1)
                            si_start = max(0, int(left))
                            si_end = max(si_start, int(right))
                        else:
                            si_start = max(0, int(extra))
                    except ValueError:
                        video_file = None
                    break
                if video_file is None:
                    for prefix in (SI_CHAPTER_PREFIX, SI_TIME_POINT_PREFIX):
                        parsed = _split_si_id(object_id, prefix, expect_extra=True)
                        if parsed is None:
                            continue
                        video_rel, extra = parsed
                        video_file = media_library.key_to_path(video_rel)
                        is_si_video = True
                        si_object_kind = "point"
                        si_point_prefix = prefix
                        try:
                            si_start = max(0, int(extra))
                        except ValueError:
                            video_file = None
                        break

        def si_available(path: Path | None) -> bool:
            if path is None or si_service is None:
                return False
            try:
                si_config = si_service.current_config()
                return bool(si_config is not None and si_config.enabled and si_service.has_si_source(path) is not None)
            except Exception:
                return False

        # 3. Handle BrowseMetadata
        if flag == "BrowseMetadata":
            metadata_count = 1
            if video_file is not None and video_file.is_file():
                # Browse single video details
                meta = probe_cached(video_file)
                title = vr_naming.source_display_stem(video_file.stem, meta["width"], meta["height"])
                if is_si_video:
                    item = None
                    if si_available(video_file):
                        if si_object_kind == "dir":
                            item = _si_container_item(video_rel, parent_id, title, float(meta.get("duration") or 0.0))
                        elif si_object_kind == "time":
                            item = _si_time_index_metadata_item(
                                video_file,
                                video_rel,
                                si_time_level,
                                start=si_start,
                                end=si_end,
                                language=language,
                            )
                        elif si_object_kind == "point":
                            item = _si_point_metadata_item(
                                video_file,
                                video_rel,
                                si_point_prefix,
                                si_start,
                                base_url,
                                client_profile=client_profile,
                            )
                else:
                    sub_list = []
                    tracks = subtitles.find_external_subtitles(video_file, subtitles_enabled, media_library)
                    for track in tracks:
                        try:
                            sub_rel = media_library.path_to_key(track.path)
                            sub_list.append(
                                {
                                    "url": f"{base_url}/subs/{quote(sub_rel)}",
                                    "lang": track.lang,
                                    "type": track.kind,
                                    "mime": track.mime,
                                }
                            )
                        except Exception:
                            pass
                    item = {
                        "id": object_id,
                        "parent_id": parent_id,
                        "title": title,
                        "url": f"{base_url}/media/{quote(video_rel)}",
                        "thumb": f"{base_url}/thumb/{quote(video_rel)}",
                        "size": meta["size"],
                        "duration": meta["duration"],
                        "resolution": f"{meta['width']}x{meta['height']}" if meta["width"] > 0 else "",
                        "bitrate": meta["bitrate"],
                        "mime": _get_mime(video_file),
                        "dlna_pn": _get_dlna_pn(video_file),
                        "dlna_ci": 0,
                        "subtitles": sub_list,
                    }
                didl = _didl_for([item] if item is not None else [])
                metadata_count = 1 if item is not None else 0
            else:
                # Browse directory details
                target_dir = directory or media_library.first_root.path
                didl = _metadata_didl_for_dir(target_dir, object_id, parent_id, media_library)

            return _wrap_soap(
                "Browse",
                f"<Result>{html.escape(didl)}</Result>"
                f"<NumberReturned>{metadata_count}</NumberReturned>"
                f"<TotalMatches>{metadata_count}</TotalMatches>"
                f"<UpdateID>1</UpdateID>",
            ), 200

        # 4. Handle BrowseDirectChildren
        all_items = []
        if object_id == ROOT_ID and media_library.multi_root:
            # With multiple physical roots, expose a virtual folder for each root.
            # With multiple physical roots, the root node returns a directory list.
            for root in media_library.roots:
                all_items.append(
                    {
                        "container": True,
                        "id": f"{FOLDER_PREFIX}{root.label}",
                        "parent_id": ROOT_ID,
                        "title": root.label,
                        "child_count": _child_count(root.path),
                    }
                )
        elif directory is not None and directory.is_dir():
            all_items = _get_items_for_dir(
                directory,
                object_id,
                base_url,
                media_library,
                subtitles_enabled,
                si_service=si_service,
            )
        elif is_si_video and video_file is not None and video_file.is_file() and si_available(video_file):
            if si_object_kind == "dir":
                all_items = _si_chapter_items(
                    video_file,
                    video_rel,
                    object_id,
                    base_url,
                    client_profile=client_profile,
                    language=language,
                )
            elif si_object_kind == "time":
                all_items = _si_time_index_items(
                    video_file,
                    video_rel,
                    si_time_level,
                    base_url,
                    start=si_start,
                    end=si_end,
                    client_profile=client_profile,
                )
        elif directory is None:
            log.warning("CDS Browse resolved no directory for object_id=%s", object_id)
        else:
            log.warning("CDS Browse target is not a directory for object_id=%s path=%s", object_id, directory)

        end = start + count if count > 0 else len(all_items)
        page = all_items[start:end]
        log.info("CDS Browse result object_id=%s returned=%d total=%d", object_id, len(page), len(all_items))
        _log_browse_item_preview(object_id, page)
        didl = _didl_for(page)
        body_xml = (
            f"<Result>{html.escape(didl)}</Result>"
            f"<NumberReturned>{len(page)}</NumberReturned>"
            f"<TotalMatches>{len(all_items)}</TotalMatches>"
            f"<UpdateID>1</UpdateID>"
        )
        return _wrap_soap("Browse", body_xml), 200

    if action == "GetSearchCapabilities":
        return _wrap_soap("GetSearchCapabilities", "<SearchCaps></SearchCaps>"), 200
    if action == "GetSortCapabilities":
        return _wrap_soap("GetSortCapabilities", "<SortCaps></SortCaps>"), 200
    if action == "GetSystemUpdateID":
        return _wrap_soap("GetSystemUpdateID", "<Id>1</Id>"), 200

    fault = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
        "<s:Body><s:Fault><faultcode>s:Client</faultcode>"
        "<faultstring>UPnPError</faultstring><detail>"
        '<UPnPError xmlns="urn:schemas-upnp-org:control-1-0">'
        "<errorCode>401</errorCode><errorDescription>Invalid Action</errorDescription>"
        "</UPnPError></detail></s:Fault></s:Body></s:Envelope>"
    )
    return fault.encode("utf-8"), 401
