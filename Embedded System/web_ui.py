#!/usr/bin/env python3
"""
Web settings UI for the Bat Activity Monitor.

Runs inside main.py (so it can show live status and apply changes immediately)
and listens on 127.0.0.1:8080 by default - i.e. only reachable from the Pi
itself. From your computer, open an SSH tunnel and browse to it:

    ssh -L 8080:localhost:8080 pi@<pi-address>
    then open http://localhost:8080

It can also run on its own (to edit settings while the monitor service is
stopped):  python3 web_ui.py [--host 0.0.0.0] [--port 8080]

Only the Python standard library is used.
"""

import ipaddress
import json
import logging
import os
import re
import socket
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from config_manager import ConfigManager

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
HTML_PATH = SCRIPT_DIR / "web_ui.html"
LOG_PATH = SCRIPT_DIR / "bat_monitor.log"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Editable settings: section -> key -> validator
# ---------------------------------------------------------------------------

def _num(kind: type, lo: float, hi: float) -> Callable[[Any], Any]:
    def check(v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("must be a number")
        try:
            n = kind(v)
        except (TypeError, ValueError):
            raise ValueError("must be a number")
        if kind is int and float(v) != n:
            raise ValueError("must be a whole number")
        if not lo <= n <= hi:
            raise ValueError(f"must be between {lo:g} and {hi:g}")
        return n
    return check


def _bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise ValueError("must be true or false")
    return v


def _text(max_len: int = 200, allow_empty: bool = True) -> Callable[[Any], str]:
    def check(v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("must be text")
        if len(v) > max_len:
            raise ValueError(f"must be at most {max_len} characters")
        if not allow_empty and not v.strip():
            raise ValueError("is required")
        return v
    return check


def _time(v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError("must be HH:MM")
    return ConfigManager.normalize_time(v)


def _email(v: Any) -> str:
    v = _text(254)(v).strip()
    if v and not EMAIL_RE.match(v):
        raise ValueError("is not a valid email address")
    return v


def _email_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        raise ValueError("must be a list")
    out: List[str] = []
    for item in v:
        addr = _email(item)
        if not addr:
            continue
        if addr.lower() not in (a.lower() for a in out):
            out.append(addr)
    return out


def _timezone(v: Any) -> str:
    from rtc_scheduler import valid_timezone
    if not isinstance(v, str) or not valid_timezone(v.strip()):
        raise ValueError("is not a known time zone (e.g. America/Los_Angeles)")
    return v.strip()


def _choice(*options: Any) -> Callable[[Any], Any]:
    def check(v: Any) -> Any:
        if v not in options:
            raise ValueError(f"must be one of {', '.join(str(o) for o in options)}")
        return v
    return check


FIELDS: Dict[str, Dict[str, Callable[[Any], Any]]] = {
    "schedule": {
        "start_time": _time,
        "stop_time": _time,
        "enabled": _bool,
    },
    "email": {
        "smtp_host": _text(253, allow_empty=False),
        "smtp_port": _num(int, 1, 65535),
        "sender_email": _email,
        "sender_password": _text(200),
        "recipients": _email_list,
        "trendline_slope": _num(float, -1e6, 1e6),
        "trendline_intercept": _num(float, -1e6, 1e6),
    },
    "audio": {
        "energy_threshold": _num(float, 0.0, 1.0),
        "min_vocalization_duration": _num(float, 0.02, 30.0),
        "max_vocalization_duration": _num(float, 0.1, 60.0),
        "silence_duration": _num(float, 0.02, 5.0),
        "sample_rate": _choice(16000, 22050, 32000, 44100, 48000),
        "channels": _choice(1, 2),
        "chunk_duration": _num(float, 0.2, 10.0),
    },
    "model": {
        "confidence_threshold": _num(float, 0.0, 1.0),
        "sample_rate": _choice(16000, 22050, 32000, 44100, 48000),
    },
    "wifi": {
        "ssid": _text(32),
        "password": _text(64),
        "setup_hotspot": _bool,
        "hotspot_password": _text(63),
    },
    "system": {
        "timezone": _timezone,
        "log_level": _choice("DEBUG", "INFO", "WARNING", "ERROR"),
        "email_retry_attempts": _num(int, 1, 20),
        "email_retry_delay": _num(int, 5, 3600),
    },
    "web": {
        "host": _text(64, allow_empty=False),
        "port": _num(int, 1024, 65535),
    },
    "display": {
        "rotate_180": _bool,
        "encoder_reverse": _bool,
        "timeout": _num(int, 0, 86400),
    },
}

# Write-only fields: never sent back to the browser
SECRETS = {("email", "sender_password"), ("wifi", "password"), ("wifi", "hotspot_password")}

# Changing these only takes effect after the monitor restarts / next session
NEXT_SESSION = {("audio", "sample_rate"), ("audio", "channels"), ("audio", "chunk_duration")}
NEEDS_RESTART = {("web", "host"), ("web", "port"), ("model", "sample_rate")}


def public_config(config: ConfigManager) -> Dict[str, Any]:
    """Editable settings with secrets replaced by a 'is set' flag."""
    snap = config.snapshot()
    out: Dict[str, Any] = {}
    for section, fields in FIELDS.items():
        out[section] = {}
        for key in fields:
            value = snap.get(section, {}).get(key)
            if (section, key) in SECRETS:
                out[section][key + "_set"] = bool(value)
            else:
                out[section][key] = value
    return out


def validate_updates(config: ConfigManager, updates: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Check a {section: {key: value}} update. Returns (clean_updates, errors)."""
    clean: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    if not isinstance(updates, dict):
        return clean, {"_": "Expected a JSON object"}

    for section, values in updates.items():
        if section not in FIELDS or not isinstance(values, dict):
            errors[str(section)] = "Unknown settings section"
            continue
        for key, value in values.items():
            validator = FIELDS[section].get(key)
            if validator is None:
                errors[f"{section}.{key}"] = "Unknown setting"
                continue
            try:
                clean.setdefault(section, {})[key] = validator(value)
            except ValueError as e:
                errors[f"{section}.{key}"] = str(e)

    # Cross-field checks against the resulting values
    audio = {**config.get_audio_config(), **clean.get("audio", {})}
    if audio["min_vocalization_duration"] >= audio["max_vocalization_duration"]:
        errors["audio.max_vocalization_duration"] = "must be longer than the minimum duration"
    hp = clean.get("wifi", {}).get("hotspot_password")
    if hp is not None and not 8 <= len(hp) <= 63:
        errors["wifi.hotspot_password"] = "must be 8-63 characters"
    if "web" in clean and "host" in clean["web"]:
        host = clean["web"]["host"].strip()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if host != "localhost":
                errors["web.host"] = "must be an IP address such as 127.0.0.1 or 0.0.0.0"
    return clean, errors


def tail_log(path: Path, lines: int) -> List[str]:
    """Last `lines` lines of a text file without reading all of it."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block, data = 8192, b""
        while size > 0 and data.count(b"\n") <= lines:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    return data.decode("utf-8", errors="replace").splitlines()[-lines:]


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class WebUI:
    """Serves the settings page and its JSON API."""

    def __init__(self, system: Any = None, config: Optional[ConfigManager] = None,
                 host: str = "127.0.0.1", port: int = 8080):
        """
        Args:
            system: Running BatMonitorSystem (None = settings-only mode)
            config: ConfigManager (defaults to the system's)
        """
        self.system = system
        self.config: ConfigManager = config or (system.config if system else ConfigManager())
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._wifi_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start(self, blocking: bool = False):
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        where = f"http://{self.host}:{self.port}"
        if self.host in ("127.0.0.1", "localhost", "::1"):
            logger.info(f"Web settings UI on {where} - from your computer run: "
                        f"ssh -L {self.port}:localhost:{self.port} <user>@<pi-address>  "
                        f"then open http://localhost:{self.port}")
        else:
            logger.warning(f"Web settings UI on {where} is reachable by anyone on the network")
        if blocking:
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="web-ui")
            self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # -- request security -------------------------------------------------

    def _allowed_host(self, host_header: Optional[str]) -> bool:
        """Block DNS-rebinding: only accept IP literals and this machine's own names."""
        if not host_header:
            return False
        host = urlparse("//" + host_header).hostname or ""
        if host in ("localhost",):
            return True
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            pass
        name = socket.gethostname().lower()
        return host.lower() in (name, name + ".local", name + ".lan", name + ".home")

    # -- API ----------------------------------------------------------------

    def api_status(self) -> Dict[str, Any]:
        if self.system is None:
            return {"service_running": False, "hostname": socket.gethostname(),
                    "message": "Settings-only mode: the monitor isn't running in this process."}
        status = self.system.get_status()
        status["service_running"] = True
        return status

    def api_save_config(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        clean, errors = validate_updates(self.config, body)
        if errors:
            return 400, {"ok": False, "errors": errors}
        notes: List[str] = []
        for section, values in clean.items():
            self.config.update_section(section, values, save=False)
            for key in values:
                if (section, key) in NEXT_SESSION:
                    notes.append("Audio capture changes apply the next time monitoring starts.")
                if (section, key) in NEEDS_RESTART:
                    notes.append("This change applies after restarting the service (sudo systemctl restart bat-monitor).")
        if clean:
            self.config.save_config()
            if self.system:
                self.system.apply_config_changes()
            logger.info("Settings updated: " + ", ".join(
                f"{s}.{k}" for s, v in clean.items() for k in v))
        return 200, {"ok": True, "config": public_config(self.config), "notes": sorted(set(notes))}

    def api_monitor(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        if self.system is None:
            return 409, {"ok": False, "error": "The monitor service isn't running"}
        action = body.get("action") if isinstance(body, dict) else None
        if action == "start":
            error = self.system.start_monitoring(manual=True)
        elif action == "stop":
            error = self.system.stop_monitoring(manual=True)
        else:
            return 400, {"ok": False, "error": "action must be 'start' or 'stop'"}
        return (200, {"ok": True}) if not error else (409, {"ok": False, "error": error})

    def api_test_email(self) -> Tuple[int, Dict[str, Any]]:
        if self.system and self.system.email_sender:
            sender = self.system.email_sender
        else:
            from email_sender import EmailSender
            sender = EmailSender(self.config.get_email_config())
        error = sender.send_test_email()
        return (200, {"ok": True}) if not error else (200, {"ok": False, "error": error})

    def api_clock(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        try:
            dt = datetime.fromisoformat(str(body.get("datetime"))).replace(tzinfo=None, microsecond=0)
        except (AttributeError, ValueError):
            return 400, {"ok": False, "error": "datetime must be ISO format, e.g. 2026-10-23T21:15:00"}
        if self.system is None:
            from rtc_scheduler import set_system_time
            try:
                set_system_time(dt)
                return 200, {"ok": True, "message": "System clock set"}
            except Exception as e:
                return 200, {"ok": False, "error": f"Could not set system clock: {e}"}
        message = self.system.set_clock(dt)
        ok = "RTC set" in message or "system clock set" in message
        return 200, {"ok": ok, "message": message}

    def api_wifi_status(self) -> Dict[str, Any]:
        import wifi_manager
        return {"current": wifi_manager.current_status(max_age=0),
                "networkmanager": wifi_manager.has_networkmanager(),
                "hotspot_address": wifi_manager.HOTSPOT_ADDRESS}

    def api_wifi_scan(self) -> Dict[str, Any]:
        import wifi_manager
        with self._wifi_lock:
            return {"networks": wifi_manager.scan()}

    def api_wifi_connect(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        import wifi_manager
        if not isinstance(body, dict):
            return 400, {"ok": False, "error": "Expected a JSON object"}
        wifi_cfg = self.config.get_wifi_config()
        ssid = str(body.get("ssid", wifi_cfg.get("ssid", ""))).strip()
        password = body.get("password")
        if password is None:  # keep the saved password
            password = wifi_cfg.get("password", "") if ssid == wifi_cfg.get("ssid") else ""
        clean, errors = validate_updates(self.config, {"wifi": {"ssid": ssid, "password": str(password)}})
        if errors:
            return 400, {"ok": False, "errors": errors}
        with self._wifi_lock:
            error = wifi_manager.connect(ssid, str(password))
        if error:
            return 200, {"ok": False, "error": error}
        self.config.update_section("wifi", clean["wifi"])
        return 200, {"ok": True}

    # -- handler ------------------------------------------------------------

    def _make_handler(self):
        ui = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "BatMonitor"
            sys_version = ""

            def log_message(self, fmt, *args):
                logger.debug("web: " + fmt % args)

            def _send(self, code: int, body: bytes, content_type: str):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, payload: Any):
                self._send(code, json.dumps(payload, default=str).encode(), "application/json")

            def _guard(self, is_post: bool) -> bool:
                if not ui._allowed_host(self.headers.get("Host")):
                    self._json(403, {"error": "Host not allowed"})
                    return False
                if is_post:
                    origin = self.headers.get("Origin")
                    if origin and not ui._allowed_host(urlparse(origin).netloc):
                        self._json(403, {"error": "Cross-origin request blocked"})
                        return False
                    # JSON content type forces a CORS preflight for cross-site requests
                    if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                        self._json(415, {"error": "Content-Type must be application/json"})
                        return False
                return True

            def _body(self) -> Any:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 64_000:
                    raise ValueError("Request too large")
                raw = self.rfile.read(length) if length else b"{}"
                return json.loads(raw or b"{}")

            def do_GET(self):
                if not self._guard(is_post=False):
                    return
                url = urlparse(self.path)
                try:
                    if url.path in ("/", "/index.html"):
                        self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                    elif url.path == "/api/status":
                        self._json(200, ui.api_status())
                    elif url.path == "/api/config":
                        self._json(200, public_config(ui.config))
                    elif url.path == "/api/logs":
                        n = int(parse_qs(url.query).get("lines", ["300"])[0])
                        self._json(200, {"lines": tail_log(LOG_PATH, max(1, min(n, 2000)))})
                    elif url.path == "/api/wifi":
                        self._json(200, ui.api_wifi_status())
                    elif url.path == "/api/wifi/scan":
                        self._json(200, ui.api_wifi_scan())
                    else:
                        self._json(404, {"error": "Not found"})
                except Exception as e:
                    logger.exception("Web UI request failed")
                    self._json(500, {"error": str(e)})

            def do_POST(self):
                if not self._guard(is_post=True):
                    return
                try:
                    body = self._body()
                except (ValueError, json.JSONDecodeError) as e:
                    self._json(400, {"error": f"Bad request: {e}"})
                    return
                path = urlparse(self.path).path
                try:
                    if path == "/api/config":
                        self._json(*ui.api_save_config(body))
                    elif path == "/api/monitor":
                        self._json(*ui.api_monitor(body))
                    elif path == "/api/email/test":
                        self._json(*ui.api_test_email())
                    elif path == "/api/clock":
                        self._json(*ui.api_clock(body))
                    elif path == "/api/wifi/connect":
                        self._json(*ui.api_wifi_connect(body))
                    else:
                        self._json(404, {"error": "Not found"})
                except Exception as e:
                    logger.exception("Web UI request failed")
                    self._json(500, {"error": str(e)})

        return Handler


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bat Monitor settings UI (settings-only mode)")
    parser.add_argument("--host", default=None, help="Address to listen on (default from config: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port (default from config: 8080)")
    parser.add_argument("--config", default=None, help="Path to config.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        force=True)
    config = ConfigManager(args.config)
    web_cfg = config.get_web_config()
    ui = WebUI(None, config, host=args.host or web_cfg.get("host", "127.0.0.1"),
               port=args.port or int(web_cfg.get("port", 8080)))
    print(f"Settings UI: http://{ui.host}:{ui.port}  (Ctrl-C to stop)")
    try:
        ui.start(blocking=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
