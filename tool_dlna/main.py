"""Independent entry point for the VR DLNA Server executable.

Loads toolbox config, binds network interface, setups windows firewall and spins up SSDP + FastAPI.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sys
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

# Setup Cwd if run standalone
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
_parent_dir = os.path.dirname(_app_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from tool_dlna import dlna_server, firewall, media_library


def get_runtime_base_dir() -> Path:
    """Return the directory beside the real executable, not PyInstaller temp paths."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _detect_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    # 1. Determine directories
    _exe_dir = get_runtime_base_dir()
    os.chdir(_exe_dir)

    _CONFIG_PATH = _exe_dir / "vr_toolbox_config.json"

    # 2. Defaults Configuration
    config = {
        "dlna_server_name": "VR Video Server",
        "dlna_port": 8090,
        "dlna_video_dirs": "",
        "dlna_auto_subtitles": True,
    }

    # 3. Read persists config
    if _CONFIG_PATH.exists():
        try:
            with _CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
                config.update(json.load(f))
        except Exception:
            pass

    # 4. Enforce fallback video dirs
    if not config.get("dlna_video_dirs"):
        default_dir = _exe_dir / "videos"
        default_dir.mkdir(parents=True, exist_ok=True)
        config["dlna_video_dirs"] = str(default_dir)

    lan_ip = _detect_lan_ip()
    port = int(config.get("dlna_port", 8090))
    device_uuid = str(uuid5(NAMESPACE_DNS, f"vrtoolbox-dlna-{lan_ip}-{port}"))
    cache_dir = _exe_dir / "runtime_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = dlna_server.setup_logging(cache_dir / "logs")

    print("=" * 60)
    print(f"VR Video DLNA Server starting...")
    print(f"Server Name : {config['dlna_server_name']}")
    print(f"LAN IP      : {lan_ip}")
    print(f"Port        : {port}")
    print(f"Video Dirs  : {config['dlna_video_dirs']}")
    print(f"UUID        : {device_uuid}")
    print("=" * 60)
    logger.info("VR Video DLNA Server starting")
    logger.info("Server Name : %s", config["dlna_server_name"])
    logger.info("LAN IP      : %s", lan_ip)
    logger.info("Port        : %s", port)
    logger.info("Video Dirs  : %s", config["dlna_video_dirs"])
    logger.info("UUID        : %s", device_uuid)
    logger.info("Python exe  : %s", sys.executable)
    logger.info("DLNA module : %s", getattr(dlna_server, "__file__", "unknown"))

    # 5. Firewall permissions
    firewall_ok = firewall.ensure_rules(port)
    logger.info("Firewall rules ok: %s", firewall_ok)

    # 6. Parse media roots
    video_paths = media_library.parse_video_dirs(config["dlna_video_dirs"], Path(_exe_dir) / "videos")
    roots = media_library.build_media_roots(video_paths)
    lib = media_library.MediaLibrary(roots)
    logger.info("Media roots  : %s", " | ".join(f"{root.label}={root.path}" for root in roots))

    # 7. Boot SSDP responder thread
    ssdp = dlna_server.SSDPServer(lan_ip, port, config["dlna_server_name"], device_uuid)
    ssdp.start()

    # 8. Start uvicorn Web Service
    app = dlna_server.create_app(
        server_name=config["dlna_server_name"],
        port=port,
        media_library=lib,
        subtitles_enabled=bool(config["dlna_auto_subtitles"]),
        device_uuid=device_uuid,
        lan_ip=lan_ip,
        cache_dir=cache_dir,
    )

    import uvicorn

    try:
        # Register terminate signals handler for clean exit
        def _exit_handler(sig, frame):
            logger.info("[Server] Terminate signal received, exiting...")
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _exit_handler)
        signal.signal(signal.SIGTERM, _exit_handler)

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=False,
            timeout_graceful_shutdown=3,
        )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("[Server] Shutting down SSDP and cleaning resources...")
        ssdp.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
