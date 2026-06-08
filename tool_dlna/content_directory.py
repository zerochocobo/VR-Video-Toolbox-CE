"""UPnP ContentDirectory SOAP handler for browsing the local media library.

Handles XML metadata packaging for directories, videos, and associated subtitles.
"""
from __future__ import annotations

import html
import json
import logging
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from tool_dlna import subtitles, vr_naming
from tool_dlna.firewall import hidden_subprocess_kwargs

ROOT_ID = "0"
FOLDER_PREFIX = "d_"
VIDEO_PREFIX = "v_"
VIDEO_SI_PREFIX = "vs_"
DLNA_FLAGS_BASE = "01700000000000000000000000000000"

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
    key = str(path.resolve())
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
        proto = f"http-get:*:{mime}:DLNA.ORG_PN={it['dlna_pn']};DLNA.ORG_OP=01;DLNA.ORG_CI={dlna_ci};DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"

        attrs: list[str] = []
        if size > 0:
            attrs.append(f'size="{size}"')
        if it["duration"] > 0:
            attrs.append(f'duration="{duration}"')
        if bitrate > 0:
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
                        {
                            "id": f"{VIDEO_SI_PREFIX}{rel_key}",
                            "parent_id": parent_id,
                            "title": f"[SI] {title}",
                            "url": f"{base_url}/media_si/{quoted_key}",
                            "thumb": f"{base_url}/thumb/{quoted_key}",
                            "size": si_service.estimate_output_size(child),
                            "duration": meta["duration"],
                            "resolution": f"{meta['width']}x{meta['height']}" if meta["width"] > 0 else "",
                            "bitrate": meta["bitrate"],
                            "mime": "video/mp4",
                            "dlna_pn": "AVC_MP4_HP_HD_AAC",
                            "dlna_ci": 1,
                            "subtitles": sub_list,
                        }
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

        # 3. Handle BrowseMetadata
        if flag == "BrowseMetadata":
            metadata_count = 1
            if video_file is not None and video_file.is_file():
                # Browse single video details
                meta = probe_cached(video_file)
                title = vr_naming.source_display_stem(video_file.stem, meta["width"], meta["height"])
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
                    "title": f"[SI] {title}" if is_si_video else title,
                    "url": f"{base_url}/media_si/{quote(video_rel)}" if is_si_video else f"{base_url}/media/{quote(video_rel)}",
                    "thumb": f"{base_url}/thumb/{quote(video_rel)}",
                    "size": si_service.estimate_output_size(video_file) if is_si_video and si_service is not None else meta["size"],
                    "duration": meta["duration"],
                    "resolution": f"{meta['width']}x{meta['height']}" if meta["width"] > 0 else "",
                    "bitrate": meta["bitrate"],
                    "mime": "video/mp4" if is_si_video else _get_mime(video_file),
                    "dlna_pn": "AVC_MP4_HP_HD_AAC" if is_si_video else _get_dlna_pn(video_file),
                    "dlna_ci": 1 if is_si_video else 0,
                    "subtitles": sub_list,
                }
                if is_si_video:
                    try:
                        si_config = si_service.current_config() if si_service is not None else None
                        if si_config is None or not si_config.enabled or si_service.has_si_source(video_file) is None:
                            item = None
                    except Exception:
                        item = None
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
