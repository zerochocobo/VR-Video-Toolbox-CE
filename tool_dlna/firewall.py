"""Windows Firewall setup helper for DLNA HTTP port and SSDP UDP/1900 port.

Requests UAC elevation if not already running as admin.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class SimpleLogger:
    def info(self, msg, *args):
        print(f"[Firewall] INFO: {msg % args if args else msg}")

    def warning(self, msg, *args):
        print(f"[Firewall] WARNING: {msg % args if args else msg}")

    def error(self, msg, *args):
        print(f"[Firewall] ERROR: {msg % args if args else msg}")


log = SimpleLogger()

OLD_RULE_HTTP = "PTServer HTTP"
OLD_RULE_SSDP = "PTServer SSDP"
RULE_HTTP = "VRVideoToolbox DLNA HTTP"
RULE_SSDP = "VRVideoToolbox DLNA SSDP"


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def hidden_subprocess_kwargs() -> dict:
    if not _is_windows():
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE
    return {"startupinfo": startupinfo}


def _rule_exists(name: str) -> bool:
    try:
        r = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
            capture_output=True,
            **hidden_subprocess_kwargs(),
        )
    except FileNotFoundError:
        return False
    stdout = ""
    if r.stdout:
        try:
            stdout = r.stdout.decode("gbk", errors="ignore")
        except Exception:
            stdout = r.stdout.decode("utf-8", errors="ignore")
    return r.returncode == 0 and "No rules match" not in stdout and "没有与指定标准相匹配的规则" not in stdout


def _netsh_add(name: str, proto: str, port: int) -> bool:
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name={name}",
        "dir=in", "action=allow",
        f"protocol={proto}",
        f"localport={port}",
        "profile=private",
        "edge=no",
        "enable=yes",
    ]
    r = subprocess.run(cmd, capture_output=True, **hidden_subprocess_kwargs())
    if r.returncode != 0:
        stderr = ""
        raw = r.stderr or r.stdout
        if raw:
            try:
                stderr = raw.decode("gbk", errors="ignore").strip()
            except Exception:
                stderr = raw.decode("utf-8", errors="ignore").strip()
        log.warning("netsh add failed (%s): %s", name, stderr)
        return False
    return True


def _netsh_delete(name: str) -> None:
    subprocess.run(
        ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={name}"],
        capture_output=True,
        **hidden_subprocess_kwargs(),
    )


def _add_rules_direct(http_port: int) -> bool:
    _netsh_delete(OLD_RULE_HTTP)
    _netsh_delete(OLD_RULE_SSDP)
    _netsh_delete(RULE_HTTP)
    _netsh_delete(RULE_SSDP)
    ok1 = _netsh_add(RULE_HTTP, "TCP", http_port)
    ok2 = _netsh_add(RULE_SSDP, "UDP", 1900)
    return ok1 and ok2


def _build_bat(http_port: int) -> Path:
    p = Path(tempfile.gettempdir()) / "vrtoolbox_firewall_setup.bat"
    lines = [
        "@echo off",
        "setlocal",
        f'netsh advfirewall firewall delete rule name="{OLD_RULE_HTTP}" >nul 2>nul',
        f'netsh advfirewall firewall delete rule name="{OLD_RULE_SSDP}" >nul 2>nul',
        f'netsh advfirewall firewall delete rule name="{RULE_HTTP}" >nul 2>nul',
        f'netsh advfirewall firewall delete rule name="{RULE_SSDP}" >nul 2>nul',
        f'netsh advfirewall firewall add rule name="{RULE_HTTP}" dir=in '
        f'action=allow protocol=TCP localport={http_port} profile=private edge=no enable=yes',
        f'netsh advfirewall firewall add rule name="{RULE_SSDP}" dir=in '
        f'action=allow protocol=UDP localport=1900 profile=private edge=no enable=yes',
        f'(goto) 2>nul & del "{p}"',
    ]
    p.write_text("\r\n".join(lines), encoding="utf-8")
    return p


def _elevate_run(bat: Path) -> bool:
    SW_HIDE = 0
    rc = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", "cmd.exe", f'/c "{bat}"', None, SW_HIDE
    )
    return int(rc) > 32


def ensure_rules(http_port: int) -> bool:
    """Ensure firewall rules exist; return False only when setup is rejected."""
    if not _is_windows():
        return True

    try:
        if _rule_exists(RULE_HTTP) and _rule_exists(RULE_SSDP):
            log.info("firewall rules ok")
            return True
    except Exception as e:
        log.warning("rule check error: %s", e)

    if _is_admin():
        log.info("admin detected, adding firewall rules directly")
        return _add_rules_direct(http_port)

    log.info("requesting UAC to add firewall rules (one-time)")
    bat = _build_bat(http_port)
    if not _elevate_run(bat):
        log.warning("user denied UAC; HTTP/SSDP may be blocked")
        try:
            bat.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    for _ in range(20):
        time.sleep(0.3)
        if _rule_exists(RULE_HTTP) and _rule_exists(RULE_SSDP):
            log.info("firewall rules added")
            return True
    log.warning("firewall rule verification timed out (rules may still be applied)")
    return False
