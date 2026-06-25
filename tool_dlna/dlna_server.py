"""DLNA Core Server running FastAPI + SSDP discovery service.

Defines HTTP routes for devices, SOAP commands, Range-supported streams, and ffmpeg thumbs.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import random
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from tool_dlna import connection_manager, content_directory, descriptions, si_stream, subtitles
from tool_dlna.firewall import hidden_subprocess_kwargs
from tool_dlna.media_library import safe_resolve_path

MCAST_GRP = "239.255.255.250"
MCAST_PORT = 1900
XML_MEDIA_TYPE = "text/xml; charset=utf-8"
DLNA_FLAGS_BASE = "01700000000000000000000000000000"
DLNA_FLAGS_TIME_SEEK = "41700000000000000000000000000000"
LOGGER_NAME = "vrtoolbox.dlna"
LOG_MAX_BYTES = 1024 * 1024
LOG_BACKUP_COUNT = 3
SI_LIVE_ROUTE_HINT_SUFFIXES = (".ts", ".m2ts", ".mpegts")
_SOAP_FIELD_RE = re.compile(
    rb"<(?:\w+:)?(ObjectID|BrowseFlag|Filter|RequestedCount|StartingIndex)>(.*?)</(?:\w+:)?\1>",
    re.IGNORECASE | re.DOTALL,
)
_DEOVR_CDS_FILTER = {"res", "res@size", "res@duration", "dc:date", "upnp:albumarturi"}
_SUPPORTED_CDS_UI_LANGUAGES = ("zh_CN", "ja_JP", "en_US")


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_dir: Path, max_bytes: int = LOG_MAX_BYTES, backup_count: int = LOG_BACKUP_COUNT) -> logging.Logger:
    """Configure terminal and rotating-file logging for the DLNA server."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dlna_server.log"
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max(1024, int(max_bytes)),
        backupCount=max(0, int(backup_count)),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.info("Logging initialized: %s", log_path)
    return logger


def normalize_absolute_form_path(scope: dict) -> tuple[str, str]:
    """Normalize proxy-style absolute-form request targets into route paths."""
    original = str(scope.get("path") or "")
    if not original.startswith("//"):
        return original, original
    without_slashes = original[2:]
    _authority, separator, rest = without_slashes.partition("/")
    normalized = f"/{rest}" if separator else "/"
    scope["path"] = normalized
    try:
        scope["raw_path"] = normalized.encode("utf-8")
    except Exception:
        pass
    return original, normalized


def is_loopback_host(host: str | None) -> bool:
    value = (host or "").strip().strip("[]")
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


TAIL_PROBE_DEAD_ZONE_BYTES = 5 * 1024 * 1024
MID_PROBE_MAX_SIZE_BYTES = 2 * 1024 * 1024


def classify_moov_probe(range_start: int, range_end: int | None, total: int) -> str:
    """Identify DLNA player moov-atom probes on virtual SI streams.

    Many MP4 players probe a fragmented MP4 stream for a trailing moov atom by
    requesting a slice very close to the end of the file (typically the last
    ~100KB-1MB). Some clients also probe inside the file body. Both probes are
    open-ended (``bytes=N-``) on most clients, so a closed Range is NOT a
    reliable signal.

    Tail probe (open OR closed Range): ``total - range_start`` is in the last
    few MB of the virtual file. Even a user who deliberately seeks the very end
    of the video cannot actually play anything from <5MB of bytes, so refusing
    these is safe.

    Mid probe: small closed Range in the latter half of the file. A real user
    seek arrives as an open-ended Range, so requiring ``range_end`` here keeps
    us from blocking it.

    Returns "tail" or "mid" if the request looks like a probe, otherwise "".
    The handler should refuse probe requests with HTTP 416 so the player falls
    back to sequential playback from byte 0.
    """
    if total <= 0 or range_start < 0:
        return ""
    if (total - range_start) <= TAIL_PROBE_DEAD_ZONE_BYTES:
        return "tail"
    if range_end is not None:
        probe_size = range_end - range_start + 1
        if 0 < probe_size <= MID_PROBE_MAX_SIZE_BYTES and range_start >= int(total * 0.5):
            return "mid"
    return ""


def _soap_history_fields(body: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _SOAP_FIELD_RE.finditer(body[:64 * 1024]):
        key = match.group(1).decode("ascii", "ignore")
        value = match.group(2).decode("utf-8", "ignore").strip()
        if value:
            fields[key] = value
    return fields


def _normalise_filter_set(value: str) -> set[str]:
    return {part.strip().lower() for part in str(value or "").split(",") if part.strip()}


def _header_value(headers: Mapping[str, str], name: str) -> str:
    return str(headers.get(name) or headers.get(name.lower()) or headers.get(name.title()) or "")


def _normalise_cds_ui_language(value: str) -> str | None:
    language = str(value or "").strip().lower().replace("-", "_")
    if not language:
        return None
    if language.startswith("zh"):
        return "zh_CN"
    if language.startswith("ja"):
        return "ja_JP"
    if language.startswith("en"):
        return "en_US"
    return None


def _cds_ui_language(headers: Mapping[str, str]) -> str | None:
    accept_language = _header_value(headers, "accept-language")
    candidates: list[tuple[float, int, str]] = []
    for index, raw_part in enumerate(accept_language.split(",")):
        part = raw_part.strip()
        if not part:
            continue
        language, *params = [piece.strip() for piece in part.split(";")]
        q = 1.0
        for param in params:
            if param.lower().startswith("q="):
                try:
                    q = float(param[2:])
                except ValueError:
                    q = 0.0
        normalised = _normalise_cds_ui_language(language)
        if normalised in _SUPPORTED_CDS_UI_LANGUAGES and q > 0:
            candidates.append((q, index, normalised))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _cds_client_profile(headers: Mapping[str, str], fields: dict[str, str]) -> str | None:
    ua = str(headers.get("user-agent", "") or "").strip().lower()
    if "deovr" in ua or "[deo" in ua:
        return "deovr"
    if ua:
        return None
    # DeoVR's CDS Browse request observed on Quest sends no User-Agent. Keep
    # this fallback deliberately narrow so other DLNA clients keep the default
    # Skybox-compatible live metadata.
    if fields.get("BrowseFlag", "").strip().lower() != "browsedirectchildren":
        return None
    if fields.get("RequestedCount", "").strip() != "0":
        return None
    if _normalise_filter_set(fields.get("Filter", "")) != _DEOVR_CDS_FILTER:
        return None
    return "deovr"


def _is_deovr_user_agent(user_agent: str | None) -> bool:
    ua = str(user_agent or "").strip().lower()
    return "deovr" in ua or "[deo" in ua


def si_live_content_features(user_agent: str | None = None) -> str:
    if _is_deovr_user_agent(user_agent):
        return (
            "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
            "DLNA.ORG_OP=10;DLNA.ORG_CI=1;"
            f"DLNA.ORG_FLAGS={DLNA_FLAGS_TIME_SEEK}"
        )
    return (
        "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
        "DLNA.ORG_OP=00;DLNA.ORG_CI=1"
    )


class SSDPServer:
    """Background SSDP responder and notifier for DLNA/UPnP discovery."""

    def __init__(self, lan_ip: str, port: int, server_name: str, device_uuid: str):
        self.lan_ip = lan_ip
        self.port = port
        self.server_name = server_name
        self.device_uuid = device_uuid
        self.device_usn = f"uuid:{device_uuid}"
        self.stop_event = threading.Event()
        self.sock = None
        self.thread = None

        self.targets = [
            "upnp:rootdevice",
            self.device_usn,
            "urn:schemas-upnp-org:device:MediaServer:1",
            "urn:schemas-upnp-org:service:ContentDirectory:1",
            "urn:schemas-upnp-org:service:ConnectionManager:1",
        ]

    def _location(self) -> str:
        return f"http://{self.lan_ip}:{self.port}/description.xml"

    def _server_header(self) -> str:
        return f"Windows/10 UPnP/1.0 {self.server_name}/1.0"

    def _usn_for(self, nt: str) -> str:
        if nt == self.device_usn:
            return self.device_usn
        return f"{self.device_usn}::{nt}"

    def _build_response(self, st: str) -> bytes:
        msg = (
            "HTTP/1.1 200 OK\r\n"
            "CACHE-CONTROL: max-age=1800\r\n"
            "EXT:\r\n"
            f"LOCATION: {self._location()}\r\n"
            f"SERVER: {self._server_header()}\r\n"
            f"ST: {st}\r\n"
            f"USN: {self._usn_for(st)}\r\n"
            "DATE: " + time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()) + "\r\n"
            "\r\n"
        )
        return msg.encode("utf-8")

    def _build_notify(self, nt: str, alive: bool = True) -> bytes:
        if alive:
            msg = (
                "NOTIFY * HTTP/1.1\r\n"
                f"HOST: {MCAST_GRP}:{MCAST_PORT}\r\n"
                "CACHE-CONTROL: max-age=1800\r\n"
                f"LOCATION: {self._location()}\r\n"
                f"SERVER: {self._server_header()}\r\n"
                "NTS: ssdp:alive\r\n"
                f"NT: {nt}\r\n"
                f"USN: {self._usn_for(nt)}\r\n"
                "\r\n"
            )
        else:
            msg = (
                "NOTIFY * HTTP/1.1\r\n"
                f"HOST: {MCAST_GRP}:{MCAST_PORT}\r\n"
                "NTS: ssdp:byebye\r\n"
                f"NT: {nt}\r\n"
                f"USN: {self._usn_for(nt)}\r\n"
                "\r\n"
            )
        return msg.encode("utf-8")

    def _recv_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        s.bind(("", MCAST_PORT))
        mreq = struct.pack("=4s4s", socket.inet_aton(MCAST_GRP), socket.inet_aton(self.lan_ip))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(self.lan_ip))
        self.sock = s
        print(f"[SSDP] Listening on {MCAST_GRP}:{MCAST_PORT} (iface {self.lan_ip})")

        while not self.stop_event.is_set():
            try:
                data, addr = s.recvfrom(2048)
            except OSError:
                break
            try:
                self._handle(data, addr)
            except Exception as e:
                print(f"[SSDP] Handle error: {e}")

    def _handle(self, data: bytes, addr):
        text = data.decode("utf-8", errors="ignore")
        if not text.startswith("M-SEARCH"):
            return
        st = ""
        mx = 1.0
        for line in text.split("\r\n"):
            lower = line.lower()
            if lower.startswith("st:"):
                st = line.split(":", 1)[1].strip()
            elif lower.startswith("mx:"):
                try:
                    mx = float(line.split(":", 1)[1].strip())
                except ValueError:
                    mx = 1.0
        if not st:
            return
        replies = []
        if st == "ssdp:all":
            replies = self.targets
        elif st in self.targets or st.startswith("uuid:"):
            replies = [st if st in self.targets else self.device_usn]
        if not replies:
            return
        delay = random.uniform(0.0, max(0.0, min(mx, 2.0)))
        if delay > 0:
            if self.stop_event.wait(delay):
                return
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for s in replies:
            try:
                out.sendto(self._build_response(s), addr)
            except OSError:
                pass
        out.close()

    def _notify_loop(self):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
        sender.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(self.lan_ip))
        # Burst at startup
        for _ in range(3):
            self._broadcast(sender, alive=True)
            time.sleep(0.3)
        while not self.stop_event.wait(60):
            self._broadcast(sender, alive=True)
        # Bye bye
        self._broadcast(sender, alive=False)
        sender.close()

    def _broadcast(self, sender: socket.socket, alive: bool):
        for nt in self.targets:
            try:
                sender.sendto(self._build_notify(nt, alive), (MCAST_GRP, MCAST_PORT))
            except OSError:
                pass

    def start(self):
        """Start the background SSDP thread."""
        self.thread = threading.Thread(target=self.run, name="ssdp", daemon=True)
        self.thread.start()

    def run(self):
        threading.Thread(target=self._notify_loop, name="ssdp-notify", daemon=True).start()
        self._recv_loop()

    def stop(self):
        self.stop_event.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass


def extract_thumbnail(video_path: Path, thumb_path: Path):
    """Invoke ffmpeg silently to extract a video frame as thumbnail."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", "10.000",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=480:-2",
        str(thumb_path)
    ]
    try:
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, capture_output=True, timeout=5, **hidden_subprocess_kwargs())
    except Exception as e:
        get_logger().warning("[Thumb] Extract error: %s", e)


def create_app(
    server_name: str,
    port: int,
    media_library,
    subtitles_enabled: bool,
    device_uuid: str,
    lan_ip: str,
    cache_dir: Path,
    si_config_holder: si_stream.ConfigHolder | None = None,
) -> FastAPI:
    """Create and configure FastAPI app with DLNA endpoints."""
    base_url = f"http://{lan_ip}:{port}"
    thumb_dir = cache_dir / "thumbs"
    logger = get_logger()
    si_holder = si_config_holder or si_stream.ConfigHolder(si_stream.SIMixConfig(enabled=False))
    si_service = si_stream.SIStreamService(media_library, si_holder)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            si_service.shutdown()

    app = FastAPI(title="VRVideoToolbox-DLNA", lifespan=lifespan)

    @app.middleware("http")
    async def log_http_requests(request: Request, call_next):
        original_path, normalized_path = normalize_absolute_form_path(request.scope)
        if original_path != normalized_path:
            logger.info("Normalized absolute-form path: %s -> %s", original_path, normalized_path)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "HTTP %s %s failed after %.1fms",
                request.method,
                request.scope.get("path", ""),
                elapsed_ms,
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "HTTP %s %s -> %s %.1fms",
            request.method,
            request.scope.get("path", ""),
            response.status_code,
            elapsed_ms,
        )
        return response

    def _safe_video_path(name: str) -> Path:
        decoded = unquote(name)
        p = media_library.key_to_path(decoded)
        if p is None:
            raise HTTPException(403, "Forbidden")
        p = safe_resolve_path(p)
        if not media_library.contains(p):
            raise HTTPException(403, "Forbidden")
        if not p.is_file():
            raise HTTPException(404, "Not Found")
        return p

    @app.get("/")
    async def index():
        return Response(
            content=(
                f"{server_name} is running.\n"
                "DLNA endpoints:\n"
                "- /description.xml\n"
                "- /control/cds\n"
                "- /control/cm\n"
            ),
            media_type="text/plain; charset=utf-8",
        )

    # ---- UPnP XML metadata ----
    @app.get("/description.xml")
    async def get_description():
        return Response(content=descriptions.device_description(server_name, device_uuid), media_type=XML_MEDIA_TYPE)

    @app.get("/cds.xml")
    async def get_cds_scpd():
        return Response(content=descriptions.cds_scpd(), media_type=XML_MEDIA_TYPE)

    @app.get("/cm.xml")
    async def get_cm_scpd():
        return Response(content=descriptions.cm_scpd(), media_type=XML_MEDIA_TYPE)

    # ---- UPnP SOAP actions ----
    @app.post("/control/cds")
    async def control_cds(request: Request):
        soap_action = request.headers.get("SOAPAction", "")
        body = await request.body()
        fields = _soap_history_fields(body)
        client_profile = _cds_client_profile(request.headers, fields)
        ui_language = _cds_ui_language(request.headers)
        logger.info(
            "SOAP ContentDirectory action=%s bytes=%d profile=%s language=%s",
            soap_action,
            len(body),
            client_profile or "",
            ui_language or "",
        )
        payload, status = content_directory.handle_soap(
            soap_action,
            body,
            base_url,
            media_library,
            subtitles_enabled,
            si_service=si_service,
            client_profile=client_profile,
            language=ui_language,
        )
        return Response(content=payload, status_code=status, media_type=XML_MEDIA_TYPE)

    @app.post("/control/cm")
    async def control_cm(request: Request):
        soap_action = request.headers.get("SOAPAction", "")
        body = await request.body()
        logger.info("SOAP ConnectionManager action=%s bytes=%d", soap_action, len(body))
        payload, status = connection_manager.handle_soap(soap_action, body)
        return Response(content=payload, status_code=status, media_type=XML_MEDIA_TYPE)

    # ---- Media playback and subtitles ----
    @app.get("/media/{name:path}")
    async def media_get(name: str):
        path = _safe_video_path(name)
        headers = {
            "Accept-Ranges": "bytes",
            "transferMode.dlna.org": "Streaming",
            "contentFeatures.dlna.org": f"DLNA.ORG_PN=AVC_MP4_HP_HD_AAC;DLNA.ORG_OP=01;DLNA.ORG_CI=0;DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"
        }
        # Injects DLNA external subtitles headers for Samsung/LG and other specific clients
        tracks = subtitles.find_external_subtitles(path, subtitles_enabled, media_library)
        if tracks:
            try:
                sub_rel = media_library.path_to_key(tracks[0].path)
                headers["CaptionInfo.sec"] = f"{base_url}/subs/{quote(sub_rel)}"
                headers["getCaptionInfo.sec"] = "1"
            except Exception:
                pass
        return FileResponse(path, headers=headers, media_type="video/mp4")

    @app.get("/media_si/{name:path}")
    async def media_si_get(request: Request, name: str):
        path = _safe_video_path(name)
        config = si_service.current_config()
        if not config.enabled:
            raise HTTPException(404, "SI entries disabled")
        if si_service.has_si_source(path) is None:
            raise HTTPException(404, "No SI source")

        range_header = request.headers.get("range")
        range_start, range_end = si_stream.parse_range_header(range_header)
        total = si_service.estimate_output_size(path)
        logger.info(
            "SI request path=%s range=%s -> start=%d end=%s total=%d",
            name,
            range_header or "(none)",
            range_start,
            range_end if range_end is not None else "open",
            total,
        )
        if total > 0 and range_start >= total:
            return Response(
                status_code=416,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes */{total}",
                },
            )

        probe_kind = classify_moov_probe(range_start, range_end, total)
        if probe_kind:
            logger.info(
                "Refusing SI moov-probe request bytes=%d-%s total=%d (%s)",
                range_start,
                range_end,
                total,
                probe_kind,
            )
            return Response(
                status_code=416,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes */{total}",
                },
            )

        safe_start = min(max(0, range_start), max(0, total - 1))
        safe_end = min((range_end if range_end is not None else total - 1), total - 1)
        client_id = request.client.host if request.client else None
        chunks, content_length, total, status = si_service.open_stream(
            path,
            safe_start,
            safe_end if range_end is not None else None,
            client_id=client_id,
        )
        content_end = safe_start + content_length - 1 if content_length > 0 else safe_start
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "transferMode.dlna.org": "Streaming",
            "contentFeatures.dlna.org": (
                f"DLNA.ORG_PN=AVC_MP4_HP_HD_AAC;DLNA.ORG_OP=01;DLNA.ORG_CI=1;"
                f"DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"
            ),
        }
        if status == 206:
            headers["Content-Range"] = f"bytes {safe_start}-{content_end}/{total}"
        return StreamingResponse(chunks, status_code=status, headers=headers, media_type="video/mp4")

    @app.get("/si_live/{name:path}")
    async def si_live_get(request: Request, name: str, t: float = 0.0):
        for suffix in SI_LIVE_ROUTE_HINT_SUFFIXES:
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
                break
        path = _safe_video_path(name)
        config = si_service.current_config()
        si_wav = si_service.has_si_source(path)
        if not config.enabled or si_wav is None:
            raise HTTPException(404, "SI stream not available")
        start_time = max(0.0, float(t or 0.0))
        logger.info(
            "SI live request path=%s t=%.3f client=%s ua=%r",
            name,
            start_time,
            request.client.host if request.client else "",
            request.headers.get("user-agent", ""),
        )
        headers = {
            "Accept-Ranges": "none",
            "X-SI-Enabled": "1",
            "X-SI-Transport": "mpegts-live",
            "transferMode.dlna.org": "Streaming",
            "contentFeatures.dlna.org": si_live_content_features(request.headers.get("user-agent", "")),
        }
        return StreamingResponse(
            si_stream.iter_si_mpegts(path, si_wav, config, start_time),
            status_code=200,
            headers=headers,
            media_type="video/MP2T",
        )

    @app.post("/admin/reload_si_config")
    async def reload_si_config(request: Request):
        client_host = request.client.host if request.client else ""
        if not is_loopback_host(client_host):
            raise HTTPException(403, "Forbidden")
        try:
            data = await request.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            raise HTTPException(400, "Invalid SI config payload")
        new_config = si_stream.SIMixConfig.from_mapping(data)
        si_service.reload_config(new_config)
        return {"ok": True, "config": new_config.as_dict()}

    @app.get("/subs/{name:path}")
    async def subtitle_get(name: str):
        decoded = unquote(name)
        p = media_library.key_to_path(decoded)
        if p is None:
            raise HTTPException(403, "Forbidden")
        p = safe_resolve_path(p)
        if not media_library.contains(p) or not p.is_file() or not subtitles.is_subtitle_path(p):
            raise HTTPException(404, "Not Found")
        headers = {
            "Content-Disposition": "inline",
            "Access-Control-Allow-Origin": "*",
        }
        return FileResponse(p, headers=headers, media_type=subtitles.subtitle_mime(p))

    # ---- Dynamic cover thumbnails ----
    @app.get("/thumb/{name:path}")
    async def thumb_get(name: str):
        path = _safe_video_path(name)
        # Unique fingerprint based on mtime and size
        try:
            st = path.stat()
            fp = hashlib.md5(f"{st.st_mtime}-{st.st_size}".encode()).hexdigest()[:12]
        except Exception:
            fp = "unknown"
        thumb_path = thumb_dir / f"{path.stem}_{fp}.jpg"

        if not thumb_path.exists():
            # Extract on fallback thread to keep async event loop completely free
            threading.Thread(target=extract_thumbnail, args=(path, thumb_path), daemon=True).start()
            # Wait briefly for extraction to finish
            for _ in range(15):
                if thumb_path.exists():
                    break
                time.sleep(0.1)

        if not thumb_path.exists():
            raise HTTPException(404, "Thumbnail not available")

        return FileResponse(thumb_path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    routes = []
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = ",".join(sorted(getattr(route, "methods", []) or []))
        if path:
            routes.append(f"{methods or '-'} {path}")
    logger.info("Registered HTTP routes: %s", " | ".join(routes))
    return app
