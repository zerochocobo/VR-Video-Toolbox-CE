from __future__ import annotations

import html
import json
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import ProxyHandler, Request, build_opener
from xml.etree import ElementTree as ET


APP_HOST = "127.0.0.1"
APP_PORT = 8765
SOAP_ACTION = '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"'
DC_NS = "{http://purl.org/dc/elements/1.1/}"
DIDL_RES_NS = "{urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/}res"
VIDEO_STREAM_TYPES = {0x01, 0x02, 0x10, 0x1B, 0x24, 0x27, 0xD2}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PT DLNA Client Simulator</title>
<style>
:root { color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; }
body { margin: 0; background: #f5f7fb; color: #1b2430; }
header { height: 52px; display: flex; align-items: center; gap: 10px; padding: 0 14px; background: #ffffff; border-bottom: 1px solid #cfd6e3; }
header input { width: 360px; max-width: 38vw; padding: 7px 8px; border: 1px solid #b8c1d1; border-radius: 4px; }
button, select, input, textarea { font: inherit; }
button { padding: 7px 12px; border: 1px solid #9ba8bd; border-radius: 4px; background: #ffffff; cursor: pointer; }
button.primary { background: #1f6feb; border-color: #1f6feb; color: #ffffff; }
button.danger { background: #b42318; border-color: #b42318; color: #ffffff; }
button:disabled { opacity: .55; cursor: default; }
main { display: grid; grid-template-columns: 38% 62%; height: calc(100vh - 53px); }
#treePane { border-right: 1px solid #cfd6e3; background: #ffffff; overflow: auto; }
#detailPane { padding: 14px; overflow: auto; }
.node { display: flex; align-items: center; gap: 6px; min-height: 28px; padding: 2px 8px; white-space: nowrap; }
.node:hover { background: #eef4ff; }
.node.selected { background: #dbeafe; }
.twisty { width: 20px; text-align: center; border: 0; background: transparent; padding: 0; color: #3d4b61; }
.title { overflow: hidden; text-overflow: ellipsis; }
.kind { margin-left: auto; font-size: 12px; color: #5d6b82; }
.children { margin-left: 18px; }
.panel { background: #ffffff; border: 1px solid #cfd6e3; border-radius: 6px; padding: 12px; margin-bottom: 12px; }
.panel h2 { font-size: 15px; margin: 0 0 10px; }
.field { margin: 7px 0; color: #39475b; overflow-wrap: anywhere; }
.grid { display: grid; grid-template-columns: 150px 1fr 150px 1fr; gap: 8px 12px; align-items: center; }
.metric { padding: 8px; border: 1px solid #d9dfeb; border-radius: 4px; background: #f8fafd; }
textarea { width: 100%; height: 76px; box-sizing: border-box; border: 1px solid #b8c1d1; border-radius: 4px; padding: 8px; }
#log { height: 180px; overflow: auto; white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 10px; border-radius: 4px; font-family: Consolas, monospace; font-size: 12px; }
.row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.row input { width: 90px; padding: 6px 8px; border: 1px solid #b8c1d1; border-radius: 4px; }
.spacer { flex: 1; }
</style>
</head>
<body>
<header>
  <strong>PT DLNA Client Simulator</strong>
  <label>Server <input id="baseUrl" value="http://127.0.0.1:8090"></label>
  <button onclick="refreshRoot()">Refresh</button>
  <button onclick="discover()">Discover</button>
  <span id="status">Ready</span>
  <span class="spacer"></span>
  <button class="danger" onclick="shutdownApp()">Close App</button>
</header>
<main>
  <section id="treePane"><div id="tree"></div></section>
  <section id="detailPane">
    <div class="panel">
      <h2>Selected Item</h2>
      <div class="field"><strong id="selectedTitle">-</strong></div>
      <div class="field" id="selectedId">id: -</div>
      <div class="field" id="selectedUrl">url: -</div>
    </div>
    <div class="panel">
      <h2>Simulated Playback</h2>
      <div class="row">
        <label>Profile
          <select id="profile">
            <option>SKYBOX/libmpv</option>
            <option>4XVR/Quest</option>
            <option>Lavf Range Probe</option>
            <option>Default</option>
          </select>
        </label>
        <label>Duration <input id="duration" value="30"></label>
        <label>Chunk <input id="chunkSize" value="262144"></label>
        <button class="primary" onclick="startPull()">Start Pull</button>
        <button onclick="stopPull()">Stop</button>
      </div>
      <p class="field">Extra headers, one per line:</p>
      <textarea id="extraHeaders"></textarea>
    </div>
    <div class="panel">
      <h2>Measured Result</h2>
      <div class="grid">
        <div>Elapsed</div><div class="metric" id="mElapsed">-</div>
        <div>First Byte</div><div class="metric" id="mFirst">-</div>
        <div>Bytes</div><div class="metric" id="mBytes">-</div>
        <div>Mbps avg/inst</div><div class="metric" id="mMbps">-</div>
        <div>FPS avg/inst</div><div class="metric" id="mFps">-</div>
        <div>Video frames</div><div class="metric" id="mFrames">-</div>
        <div>HTTP</div><div class="metric" id="mHttp">-</div>
        <div>Content-Type</div><div class="metric" id="mType">-</div>
      </div>
    </div>
    <div class="panel">
      <h2>Log</h2>
      <div id="log"></div>
    </div>
  </section>
</main>
<script>
let selected = null;
let pollTimer = null;

function status(text) { document.getElementById("status").textContent = text; }
function log(text) {
  const el = document.getElementById("log");
  el.textContent += new Date().toLocaleTimeString() + " " + text + "\n";
  el.scrollTop = el.scrollHeight;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function baseParam() {
  return "base=" + encodeURIComponent(document.getElementById("baseUrl").value.trim());
}
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}
async function refreshRoot() {
  selected = null;
  document.getElementById("tree").innerHTML = "";
  status("Browsing root...");
  try {
    const data = await api("/api/browse?object_id=0&" + baseParam());
    const root = document.getElementById("tree");
    renderNodes(root, data.nodes, 0);
    status("Loaded root");
  } catch (e) { status("Error"); log(e.message); }
}
async function discover() {
  status("Discovering...");
  try {
    const data = await api("/api/discover");
    if (data.servers.length) {
      document.getElementById("baseUrl").value = data.servers[0];
      log("Discovered " + data.servers[0]);
    } else {
      log("No DLNA media server discovered");
    }
    status("Ready");
  } catch (e) { status("Error"); log(e.message); }
}
function renderNodes(parent, nodes, depth) {
  for (const node of nodes) {
    const row = document.createElement("div");
    row.className = "node";
    row.dataset.node = JSON.stringify(node);
    row.style.paddingLeft = (8 + depth * 14) + "px";
    const kind = node.is_container ? "Folder" : (node.url.includes("/passthrough_live/") ? "Live" : "File");
    row.innerHTML = `<button class="twisty">${node.is_container ? ">" : ""}</button><span class="title">${esc(node.title)}</span><span class="kind">${kind}</span>`;
    parent.appendChild(row);
    const children = document.createElement("div");
    children.className = "children";
    children.style.display = "none";
    parent.appendChild(children);
    row.onclick = () => selectNode(row, node);
    row.querySelector(".twisty").onclick = async (ev) => {
      ev.stopPropagation();
      if (!node.is_container) return;
      if (children.dataset.loaded !== "1") {
        status("Browsing " + node.title + "...");
        try {
          const data = await api("/api/browse?object_id=" + encodeURIComponent(node.object_id) + "&" + baseParam());
          renderNodes(children, data.nodes, depth + 1);
          children.dataset.loaded = "1";
        } catch (e) { log(e.message); }
      }
      const open = children.style.display !== "none";
      children.style.display = open ? "none" : "block";
      row.querySelector(".twisty").textContent = open ? ">" : "v";
      status("Ready");
    };
  }
}
function selectNode(row, node) {
  document.querySelectorAll(".node.selected").forEach(el => el.classList.remove("selected"));
  row.classList.add("selected");
  selected = node;
  document.getElementById("selectedTitle").textContent = node.title;
  document.getElementById("selectedId").textContent = "id: " + node.object_id;
  document.getElementById("selectedUrl").textContent = node.url || "(container)";
}
async function startPull() {
  if (!selected || !selected.url) { log("Select a playable item first"); return; }
  const payload = {
    url: selected.url,
    profile: document.getElementById("profile").value,
    duration: Number(document.getElementById("duration").value || 30),
    chunk_size: Number(document.getElementById("chunkSize").value || 262144),
    extra_headers: document.getElementById("extraHeaders").value
  };
  try {
    const data = await api("/api/pull/start", {method:"POST", body: JSON.stringify(payload)});
    log("Started pull " + data.id + ": " + selected.title);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStats, 500);
  } catch (e) { log(e.message); }
}
async function stopPull() {
  try { await api("/api/pull/stop", {method:"POST"}); log("Stop requested"); } catch (e) { log(e.message); }
}
async function shutdownApp() {
  if (!confirm("Close the DLNA Client Simulator?")) return;
  try {
    await api("/api/shutdown", {method:"POST"});
    document.body.innerHTML = "<main style='display:block;height:auto;padding:24px'><h2>PT DLNA Client Simulator closed</h2><p>You can close this browser tab.</p></main>";
  } catch (e) {
    log(e.message);
  }
}
async function pollStats() {
  try {
    const data = await api("/api/pull/stats");
    updateMetrics(data.stats);
    if (data.stats.done) {
      clearInterval(pollTimer);
      pollTimer = null;
      log(data.stats.error ? "Done with error: " + data.stats.error : "Pull done");
    }
  } catch (e) { log(e.message); }
}
function updateMetrics(s) {
  document.getElementById("mElapsed").textContent = s.elapsed.toFixed(2) + "s";
  document.getElementById("mFirst").textContent = s.first_byte == null ? "-" : s.first_byte.toFixed(3) + "s";
  document.getElementById("mBytes").textContent = s.bytes_read.toLocaleString();
  document.getElementById("mMbps").textContent = s.avg_mbps.toFixed(2) + " / " + s.inst_mbps.toFixed(2);
  document.getElementById("mFps").textContent = s.frames > 0 ? s.avg_fps.toFixed(2) + " / " + s.inst_fps.toFixed(2) : "n/a";
  document.getElementById("mFrames").textContent = s.frames;
  document.getElementById("mHttp").textContent = s.status || "-";
  document.getElementById("mType").textContent = s.content_type || "-";
}
refreshRoot();
</script>
</body>
</html>
"""


@dataclass
class BrowseNode:
    object_id: str
    parent_id: str
    title: str
    is_container: bool
    child_count: int = 0
    url: str = ""
    protocol_info: str = ""


@dataclass
class PullStats:
    elapsed: float = 0.0
    first_byte: float | None = None
    bytes_read: int = 0
    avg_mbps: float = 0.0
    inst_mbps: float = 0.0
    frames: int = 0
    avg_fps: float = 0.0
    inst_fps: float = 0.0
    status: int = 0
    content_type: str = ""
    error: str = ""
    done: bool = False


def soap_body(object_id: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        '<u:Browse xmlns:u="urn:schemas-upnp-org:service:ContentDirectory:1">'
        f"<ObjectID>{html.escape(object_id)}</ObjectID>"
        "<BrowseFlag>BrowseDirectChildren</BrowseFlag>"
        "<Filter>*</Filter>"
        "<StartingIndex>0</StartingIndex>"
        "<RequestedCount>0</RequestedCount>"
        "<SortCriteria></SortCriteria>"
        "</u:Browse>"
        "</s:Body></s:Envelope>"
    ).encode("utf-8")


def text_of(elem: ET.Element, local_name: str) -> str:
    for child in elem.iter():
        if child.tag.rsplit("}", 1)[-1] == local_name:
            return child.text or ""
    return ""


def profile_headers(profile: str) -> dict[str, str]:
    if profile == "SKYBOX/libmpv":
        return {
            "User-Agent": "SKYBOX/2.0.2",
            "Accept": "*/*",
            "Range": "bytes=0-",
            "transferMode.dlna.org": "Streaming",
            "getcontentFeatures.dlna.org": "1",
        }
    if profile == "4XVR/Quest":
        return {
            "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 12; Quest 3)",
            "Accept": "*/*",
            "transferMode.dlna.org": "Streaming",
            "getcontentFeatures.dlna.org": "1",
        }
    if profile == "Lavf Range Probe":
        return {
            "User-Agent": "Lavf/58.45.100",
            "Accept": "*/*",
            "Range": "bytes=564-",
        }
    return {
        "User-Agent": "PT-DLNA-Client-Simulator/1.0",
        "Accept": "*/*",
        "transferMode.dlna.org": "Streaming",
    }


def parse_extra_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid header: {line}")
        key, value = line.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


class ContentDirectoryClient:
    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def browse(self, object_id: str) -> list[BrowseNode]:
        url = f"{self.base_url}/control/cds"
        req = Request(
            url,
            data=soap_body(object_id),
            headers={
                "Content-Type": 'text/xml; charset="utf-8"',
                "SOAPAction": SOAP_ACTION,
                "User-Agent": "PT-DLNA-Client-Simulator/1.0",
            },
            method="POST",
        )
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except HTTPError as exc:
            body = exc.read(500).decode("utf-8", "ignore")
            hint = ""
            if exc.code == 404:
                hint = (
                    f" {url} returned 404. Confirm the VR Video DLNA Server is running on "
                    f"{self.base_url}, not just the client simulator, and restart it after code changes."
                )
            raise RuntimeError(f"DLNA Browse failed: HTTP {exc.code} {exc.reason}.{hint} {body}".strip()) from exc
        root = ET.fromstring(payload)
        didl_text = text_of(root, "Result")
        if not didl_text.strip():
            return []
        didl = ET.fromstring(didl_text)
        nodes: list[BrowseNode] = []
        for elem in list(didl):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag not in {"container", "item"}:
                continue
            title = elem.findtext(f"{DC_NS}title") or ""
            res = elem.find(DIDL_RES_NS)
            nodes.append(
                BrowseNode(
                    object_id=elem.attrib.get("id", ""),
                    parent_id=elem.attrib.get("parentID", ""),
                    title=title,
                    is_container=(tag == "container"),
                    child_count=int(elem.attrib.get("childCount", "0") or 0),
                    url=(res.text or "") if res is not None else "",
                    protocol_info=res.attrib.get("protocolInfo", "") if res is not None else "",
                )
            )
        return nodes


class MpegTsFpsCounter:
    def __init__(self):
        self._buf = bytearray()
        self._pmt_pid: int | None = None
        self._video_pids: set[int] = set()
        self.frames = 0

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)
        while len(self._buf) >= 188:
            if self._buf[0] != 0x47:
                pos = self._buf.find(b"\x47")
                if pos < 0:
                    self._buf.clear()
                    return
                del self._buf[:pos]
                if len(self._buf) < 188:
                    return
            packet = bytes(self._buf[:188])
            del self._buf[:188]
            self._parse_packet(packet)

    def _parse_packet(self, packet: bytes) -> None:
        payload_start = bool(packet[1] & 0x40)
        pid = ((packet[1] & 0x1F) << 8) | packet[2]
        adaptation_control = (packet[3] >> 4) & 0x03
        offset = 4
        if adaptation_control in {0, 2}:
            return
        if adaptation_control == 3:
            if offset >= len(packet):
                return
            offset += 1 + packet[offset]
        if offset >= len(packet):
            return
        payload = packet[offset:]
        if pid == 0:
            self._parse_pat(payload, payload_start)
        elif self._pmt_pid is not None and pid == self._pmt_pid:
            self._parse_pmt(payload, payload_start)
        elif pid in self._video_pids and payload_start and payload.startswith(b"\x00\x00\x01"):
            stream_id = payload[3] if len(payload) > 3 else 0
            if 0xE0 <= stream_id <= 0xEF:
                self.frames += 1

    def _section_payload(self, payload: bytes, payload_start: bool) -> bytes:
        if payload_start:
            if not payload:
                return b""
            start = 1 + payload[0]
            return payload[start:] if start < len(payload) else b""
        return payload

    def _parse_pat(self, payload: bytes, payload_start: bool) -> None:
        section = self._section_payload(payload, payload_start)
        if len(section) < 12 or section[0] != 0x00:
            return
        section_len = ((section[1] & 0x0F) << 8) | section[2]
        end = min(3 + section_len - 4, len(section))
        pos = 8
        while pos + 4 <= end:
            program = (section[pos] << 8) | section[pos + 1]
            pid = ((section[pos + 2] & 0x1F) << 8) | section[pos + 3]
            if program != 0:
                self._pmt_pid = pid
                return
            pos += 4

    def _parse_pmt(self, payload: bytes, payload_start: bool) -> None:
        section = self._section_payload(payload, payload_start)
        if len(section) < 17 or section[0] != 0x02:
            return
        section_len = ((section[1] & 0x0F) << 8) | section[2]
        program_info_len = ((section[10] & 0x0F) << 8) | section[11]
        pos = 12 + program_info_len
        end = min(3 + section_len - 4, len(section))
        video_pids: set[int] = set()
        while pos + 5 <= end:
            stream_type = section[pos]
            elementary_pid = ((section[pos + 1] & 0x1F) << 8) | section[pos + 2]
            es_info_len = ((section[pos + 3] & 0x0F) << 8) | section[pos + 4]
            if stream_type in VIDEO_STREAM_TYPES:
                video_pids.add(elementary_pid)
            pos += 5 + es_info_len
        if video_pids:
            self._video_pids = video_pids


class PullJob(threading.Thread):
    def __init__(self, url: str, headers: dict[str, str], duration: float, chunk_size: int):
        super().__init__(daemon=True)
        self.url = url
        self.headers = headers
        self.duration = max(1.0, float(duration))
        self.chunk_size = max(188, int(chunk_size))
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.stats = PullStats()
        self.counter = MpegTsFpsCounter()
        self.opener = build_opener(ProxyHandler({}))

    def snapshot(self) -> PullStats:
        with self.lock:
            return PullStats(**asdict(self.stats))

    def _set_stats(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self.stats, key, value)

    def run(self) -> None:
        started = time.perf_counter()
        first_byte_at: float | None = None
        last_report_at = started
        last_bytes = 0
        last_frames = 0
        total_bytes = 0
        status = 0
        content_type = ""
        error = ""
        try:
            req = Request(self.url, headers=self.headers, method="GET")
            with self.opener.open(req, timeout=30.0) as resp:
                status = int(resp.status)
                content_type = resp.headers.get("Content-Type", "")
                deadline: float | None = None
                while not self.stop_event.is_set():
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    try:
                        chunk = resp.read(self.chunk_size)
                    except socket.timeout:
                        error = "socket timeout while reading"
                        break
                    if not chunk:
                        break
                    now = time.perf_counter()
                    if first_byte_at is None:
                        first_byte_at = now
                        deadline = now + self.duration
                    total_bytes += len(chunk)
                    self.counter.feed(chunk)
                    if now - last_report_at >= 0.5:
                        elapsed = now - started
                        window = now - last_report_at
                        self._set_stats(
                            elapsed=elapsed,
                            first_byte=(first_byte_at - started) if first_byte_at else None,
                            bytes_read=total_bytes,
                            avg_mbps=total_bytes * 8.0 / elapsed / 1_000_000.0 if elapsed > 0 else 0.0,
                            inst_mbps=(total_bytes - last_bytes) * 8.0 / window / 1_000_000.0,
                            frames=self.counter.frames,
                            avg_fps=self.counter.frames / (now - first_byte_at) if first_byte_at and now > first_byte_at else 0.0,
                            inst_fps=(self.counter.frames - last_frames) / window,
                            status=status,
                            content_type=content_type,
                        )
                        last_report_at = now
                        last_bytes = total_bytes
                        last_frames = self.counter.frames
        except HTTPError as exc:
            status = int(exc.code)
            content_type = exc.headers.get("Content-Type", "")
            error = exc.read(400).decode("utf-8", "ignore")
        except URLError as exc:
            error = str(exc.reason)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        self._set_stats(
            elapsed=elapsed,
            first_byte=(first_byte_at - started) if first_byte_at else None,
            bytes_read=total_bytes,
            avg_mbps=total_bytes * 8.0 / elapsed / 1_000_000.0 if elapsed > 0 else 0.0,
            inst_mbps=0.0,
            frames=self.counter.frames,
            avg_fps=self.counter.frames / (time.perf_counter() - first_byte_at) if first_byte_at else 0.0,
            inst_fps=0.0,
            status=status,
            content_type=content_type,
            error=error,
            done=True,
        )


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.job: PullJob | None = None
        self.job_id = 0

    def start_job(self, job: PullJob) -> int:
        with self.lock:
            if self.job and self.job.is_alive():
                self.job.stop_event.set()
            self.job_id += 1
            self.job = job
            job.start()
            return self.job_id

    def stop_job(self) -> None:
        with self.lock:
            if self.job:
                self.job.stop_event.set()

    def stats(self) -> PullStats:
        with self.lock:
            job = self.job
        return job.snapshot() if job else PullStats(done=True)


STATE = State()
APP_SERVER: ThreadingHTTPServer | None = None
APP_CONTROL_CLOSE: Callable[[], None] | None = None


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def discover_servers() -> list[str]:
    request = (
        "M-SEARCH * HTTP/1.1\r\n"
        "HOST:239.255.255.250:1900\r\n"
        'MAN:"ssdp:discover"\r\n'
        "MX:1\r\n"
        "ST:urn:schemas-upnp-org:device:MediaServer:1\r\n"
        "\r\n"
    ).encode("ascii")
    found: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.settimeout(2.0)
        sock.sendto(request, ("239.255.255.250", 1900))
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(4096)
            except socket.timeout:
                break
            text = data.decode("utf-8", "ignore")
            for line in text.splitlines():
                if line.lower().startswith("location:"):
                    location = line.split(":", 1)[1].strip()
                    parsed = urlparse(location)
                    if parsed.scheme and parsed.netloc:
                        base = f"{parsed.scheme}://{parsed.netloc}"
                        if base not in found:
                            found.append(base)
    return found


class Handler(BaseHTTPRequestHandler):
    server_version = "PTDLNAClientSimulator/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/browse":
            qs = parse_qs(parsed.query)
            base = qs.get("base", ["http://127.0.0.1:8090"])[0]
            object_id = unquote(qs.get("object_id", ["0"])[0])
            try:
                nodes = ContentDirectoryClient(base).browse(object_id)
                json_response(self, {"nodes": [asdict(node) for node in nodes]})
            except Exception as exc:
                detail = traceback.format_exc()
                print(detail, file=sys.stderr)
                json_response(self, {"error": str(exc) or type(exc).__name__, "detail": detail}, 500)
            return
        if parsed.path == "/api/discover":
            try:
                json_response(self, {"servers": discover_servers()})
            except Exception as exc:
                json_response(self, {"error": str(exc)}, 500)
            return
        if parsed.path == "/api/pull/stats":
            json_response(self, {"stats": asdict(STATE.stats())})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        if parsed.path == "/api/pull/start":
            try:
                headers = profile_headers(str(payload.get("profile") or "Default"))
                headers.update(parse_extra_headers(str(payload.get("extra_headers") or "")))
                job = PullJob(
                    str(payload["url"]),
                    headers,
                    float(payload.get("duration") or 30),
                    int(payload.get("chunk_size") or 262144),
                )
                job_id = STATE.start_job(job)
                json_response(self, {"id": job_id})
            except Exception as exc:
                json_response(self, {"error": str(exc)}, 400)
            return
        if parsed.path == "/api/pull/stop":
            STATE.stop_job()
            json_response(self, {"ok": True})
            return
        if parsed.path == "/api/shutdown":
            json_response(self, {"ok": True})
            STATE.stop_job()
            close_control = APP_CONTROL_CLOSE
            if close_control is not None:
                close_control()
            server = APP_SERVER
            if server is not None:
                threading.Thread(target=server.shutdown, name="dlna-client-shutdown", daemon=True).start()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[client] {self.address_string()} {fmt % args}")


def find_free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((APP_HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free local port found")


def start_control_window(url: str, server: ThreadingHTTPServer) -> threading.Thread | None:
    if not getattr(sys, "frozen", False) or not sys.platform.startswith("win"):
        return None

    def run() -> None:
        global APP_CONTROL_CLOSE
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.DefWindowProcW.restype = LRESULT
        user32.LoadCursorW.restype = wintypes.HCURSOR
        user32.PostMessageW.restype = wintypes.BOOL
        user32.RegisterClassW.restype = wintypes.ATOM
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        WM_DESTROY = 0x0002
        WM_CLOSE = 0x0010
        WM_COMMAND = 0x0111
        WS_OVERLAPPED = 0x00000000
        WS_CAPTION = 0x00C00000
        WS_SYSMENU = 0x00080000
        WS_MINIMIZEBOX = 0x00020000
        WS_VISIBLE = 0x10000000
        WS_CHILD = 0x40000000
        BS_PUSHBUTTON = 0x00000000
        SS_LEFT = 0x00000000
        CW_USEDEFAULT = -2147483648
        IDC_ARROW = 32512
        ID_OPEN = 1001
        ID_CLOSE = 1002
        hwnd_holder: dict[str, int] = {}

        def close_app() -> None:
            STATE.stop_job()
            threading.Thread(target=server.shutdown, name="dlna-client-window-shutdown", daemon=True).start()

        def close_from_server() -> None:
            hwnd = hwnd_holder.get("hwnd")
            if hwnd:
                user32.PostMessageW(wintypes.HWND(hwnd), WM_CLOSE, 0, 0)

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_COMMAND:
                command_id = int(wparam) & 0xFFFF
                if command_id == ID_OPEN:
                    webbrowser.open(url)
                    return 0
                if command_id == ID_CLOSE:
                    close_app()
                    user32.DestroyWindow(hwnd)
                    return 0
            if msg == WM_CLOSE:
                close_app()
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        wnd_proc_ref = WNDPROC(wnd_proc)

        class WNDCLASS(ctypes.Structure):
            _fields_ = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HCURSOR),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        hinstance = kernel32.GetModuleHandleW(None)
        class_name = "PTDLNAClientSimulatorWindow"
        wc = WNDCLASS()
        wc.lpfnWndProc = wnd_proc_ref
        wc.hInstance = hinstance
        wc.hCursor = user32.LoadCursorW(None, IDC_ARROW)
        wc.hbrBackground = wintypes.HBRUSH(6)
        wc.lpszClassName = class_name
        user32.RegisterClassW(ctypes.byref(wc))
        hwnd = user32.CreateWindowExW(
            0,
            class_name,
            "PT DLNA Client Simulator",
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            430,
            150,
            None,
            None,
            hinstance,
            None,
        )
        if not hwnd:
            return
        hwnd_holder["hwnd"] = int(hwnd)
        user32.CreateWindowExW(0, "STATIC", "PT DLNA Client Simulator is running.", WS_CHILD | WS_VISIBLE | SS_LEFT, 16, 14, 380, 20, hwnd, None, hinstance, None)
        user32.CreateWindowExW(0, "STATIC", url, WS_CHILD | WS_VISIBLE | SS_LEFT, 16, 40, 380, 20, hwnd, None, hinstance, None)
        user32.CreateWindowExW(0, "BUTTON", "Open Web UI", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 16, 76, 180, 28, hwnd, ID_OPEN, hinstance, None)
        user32.CreateWindowExW(0, "BUTTON", "Close App", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 214, 76, 180, 28, hwnd, ID_CLOSE, hinstance, None)
        APP_CONTROL_CLOSE = close_from_server

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    thread = threading.Thread(target=run, name="dlna-client-control-window", daemon=True)
    thread.start()
    return thread


def main() -> None:
    global APP_SERVER
    port = find_free_port(APP_PORT)
    server = ThreadingHTTPServer((APP_HOST, port), Handler)
    APP_SERVER = server
    url = f"http://{APP_HOST}:{port}/"
    print(f"PT DLNA Client Simulator listening on {url}")
    control_thread = start_control_window(url, server)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE.stop_job()
        server.server_close()
        close_control = APP_CONTROL_CLOSE
        if close_control is not None:
            close_control()
        if control_thread and control_thread.is_alive():
            control_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
