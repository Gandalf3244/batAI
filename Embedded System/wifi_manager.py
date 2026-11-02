"""
WiFi helpers shared by the TFT menu, the control port (web_ui.py) and batmon.

Raspberry Pi OS Bookworm and later manage WiFi with NetworkManager (nmcli);
older releases use wpa_supplicant. Both are supported for connecting.

Setup hotspot: if the Pi hasn't joined a network ~90 s after boot, it starts
its own WiFi network ("BatPi-Setup") so you can always reach it
(ssh <user>@10.42.0.1) and give it a working network. It keeps retrying the
saved networks in the background (e.g. the router was slower to boot than the Pi).
"""

import logging
import re
import shutil
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

IFACE = "wlan0"
HOTSPOT_NAME = "BatPi-Setup"
HOTSPOT_PASSWORD = "batsetup123"
HOTSPOT_ADDRESS = "10.42.0.1"      # NetworkManager's address for a shared hotspot

_scan_cache: Dict = {"networks": [], "time": 0.0}
_status_cache: Dict = {"value": None, "time": 0.0}
_nm_available: Optional[bool] = None
_lock = threading.Lock()           # one WiFi change at a time (knob, batmon, watchdog)


# ---------------------------------------------------------------------------
# Running nmcli
# ---------------------------------------------------------------------------

def _run(cmd: List[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _nmcli(args: List[str], timeout: float = 30) -> subprocess.CompletedProcess:
    """Run nmcli, retrying through passwordless sudo if the service user isn't allowed."""
    proc = _run(["nmcli"] + args, timeout)
    out = (proc.stderr + proc.stdout).lower()
    if proc.returncode != 0 and ("not authorized" in out or "insufficient privileges" in out):
        proc = _run(["sudo", "-n", "nmcli"] + args, timeout)
    return proc


def has_networkmanager() -> bool:
    global _nm_available
    if _nm_available is None:
        try:
            _nm_available = bool(shutil.which("nmcli")) and \
                _run(["nmcli", "-t", "-f", "RUNNING", "general"], 5).stdout.strip() == "running"
        except Exception:
            _nm_available = False
    return _nm_available


def _split_terse(line: str) -> List[str]:
    """nmcli -t output: fields separated by ':', literal ':' escaped as '\\:'."""
    return [p.replace("\x00", ":") for p in line.replace("\\:", "\x00").split(":")]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _wlan_connection() -> Optional[str]:
    """Name of the NetworkManager connection active on wlan0 (None if none)."""
    proc = _nmcli(["-t", "-f", "DEVICE,STATE,CONNECTION", "device"], 10)
    for line in proc.stdout.splitlines():
        parts = _split_terse(line)
        if len(parts) >= 3 and parts[0] == IFACE and parts[1].startswith("connected"):
            return parts[2] or None
    return None


def hotspot_active(name: str = HOTSPOT_NAME) -> bool:
    try:
        return has_networkmanager() and _wlan_connection() == name
    except Exception:
        return False


def wifi_connected() -> bool:
    """wlan0 joined a normal network (not our own setup hotspot)."""
    try:
        if has_networkmanager():
            conn = _wlan_connection()
            return bool(conn) and conn != HOTSPOT_NAME
        return bool(shutil.which("iwgetid") and _run(["iwgetid", "-r"], 5).stdout.strip())
    except Exception:
        return False


def _ip_of(iface: str) -> Optional[str]:
    try:
        out = _run(["ip", "-4", "-o", "addr", "show", "dev", iface], 5).stdout
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def current_status(max_age: float = 5.0) -> Dict:
    """
    {'ssid', 'ip', 'connected', 'hotspot', 'hotspot_name', 'other_ips'} - cached for
    a few seconds, because menus ask for it on every redraw.
    """
    now = time.monotonic()
    if _status_cache["value"] is not None and now - _status_cache["time"] < max_age:
        return dict(_status_cache["value"])
    status: Dict = {"ssid": None, "ip": None, "connected": False, "hotspot": False,
                    "hotspot_name": HOTSPOT_NAME, "other_ips": []}
    try:
        conn = _wlan_connection() if has_networkmanager() else None
        if conn == HOTSPOT_NAME:
            status["hotspot"] = True
        elif shutil.which("iwgetid"):
            status["ssid"] = _run(["iwgetid", "-r"], 5).stdout.strip() or conn
        else:
            status["ssid"] = conn
        status["connected"] = bool(status["ssid"])
        status["ip"] = _ip_of(IFACE)
        if shutil.which("hostname"):
            status["other_ips"] = [ip for ip in _run(["hostname", "-I"], 5).stdout.split()
                                   if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip) and ip != status["ip"]]
    except Exception as e:
        logger.debug(f"WiFi status unavailable: {e}")
    _status_cache.update(value=status, time=now)
    return dict(status)


def _forget_status():
    _status_cache.update(value=None, time=0.0)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan() -> List[Dict]:
    """
    Nearby networks as [{'ssid', 'signal', 'secure', 'band'}], strongest first.
    While the setup hotspot is running the radio can't scan, so the last
    results (taken before the hotspot started) are returned.
    """
    if not has_networkmanager():
        return []
    if hotspot_active():
        return list(_scan_cache["networks"])
    proc = _nmcli(["-t", "-f", "SSID,SIGNAL,SECURITY,FREQ", "dev", "wifi", "list", "--rescan", "yes"], 30)
    networks: Dict[str, Dict] = {}
    for line in proc.stdout.splitlines():
        parts = _split_terse(line)
        if len(parts) < 3:
            continue
        ssid, signal, security = parts[0], parts[1], parts[2]
        if not ssid or ssid == HOTSPOT_NAME:
            continue
        try:
            strength = int(signal)
        except ValueError:
            strength = 0
        freq = parts[3] if len(parts) > 3 else ""
        if ssid not in networks or networks[ssid]["signal"] < strength:
            networks[ssid] = {"ssid": ssid, "signal": strength,
                              "secure": bool(security and security != "--"),
                              "band": "5 GHz" if freq.startswith("5") else "2.4 GHz"}
    result = sorted(networks.values(), key=lambda n: n["signal"], reverse=True)
    if result:
        _scan_cache.update(networks=result, time=time.monotonic())
    return result


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------

def friendly_error(text: str) -> str:
    """Turn nmcli's messages into something you can act on."""
    t = (text or "").lower()
    if "no network with ssid" in t or "not found" in t and "ssid" in t:
        return ("Network not visible. Check the name, that it's in range, and that it's "
                "2.4 GHz - the Pi Zero can't use 5 GHz networks.")
    if "secrets were required" in t or "psk" in t and "invalid" in t or "wrong password" in t \
            or "4-way handshake" in t or "(7)" in t:
        return "Wrong WiFi password."
    if "key-mgmt" in t:
        return "Old saved settings for this network got in the way. Try again."
    if "not authorized" in t or "insufficient privileges" in t or "a password is required" in t:
        return "Not allowed to change WiFi: the monitor's user needs passwordless sudo."
    if "timeout" in t or "timed out" in t:
        return ("Timed out joining the network: weak signal, or a network that needs a "
                "sign-in page (hotels, schools), which the Pi can't use.")
    return (text or "").strip() or "Couldn't connect"


def _valid_password(password: str) -> bool:
    return 8 <= len(password) <= 63 or (len(password) == 64 and bool(re.fullmatch(r"[0-9a-fA-F]{64}", password)))


def connect(ssid: str, password: str = "") -> str:
    """
    Connect wlan0 to a network and remember it (replacing any older saved
    settings for the same name).

    Returns:
        Empty string on success, a plain-language message otherwise
    """
    if not ssid or not ssid.strip():
        return "No WiFi network name set"
    if password and not _valid_password(password):
        return "WiFi passwords are 8-63 characters"

    with _lock:
        _forget_status()
        try:
            if has_networkmanager():
                return _connect_nm(ssid, password)
            return _connect_wpa(ssid, password)
        except FileNotFoundError as e:
            return f"WiFi tools not available on this system ({e.filename})"
        except subprocess.TimeoutExpired:
            return friendly_error("timeout")
        except Exception as e:
            return f"WiFi error: {e}"
        finally:
            _forget_status()


def _connect_nm(ssid: str, password: str) -> str:
    was_hotspot = hotspot_active()
    if was_hotspot:
        _nmcli(["connection", "down", "id", HOTSPOT_NAME], 15)     # one radio: leave setup mode first
    _nmcli(["dev", "wifi", "rescan"], 15)                            # fresh network list (may fail: fine)
    _nmcli(["connection", "delete", "id", ssid], 10)                 # stale profile (may not exist: fine)
    args = ["connection", "add", "type", "wifi", "ifname", IFACE, "con-name", ssid, "ssid", ssid,
            "connection.autoconnect", "yes"]
    if password:
        args += ["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password]
    proc = _nmcli(args, 20)
    if proc.returncode == 0:
        proc = _nmcli(["connection", "up", "id", ssid], 60)
    if proc.returncode != 0:
        error = friendly_error(proc.stderr or proc.stdout)
        if not error.startswith("Network not visible"):
            # Wrong settings: don't leave a broken profile that NetworkManager keeps retrying.
            # (A network that's just out of range stays saved and joins when it's back.)
            _nmcli(["connection", "delete", "id", ssid], 10)
        if was_hotspot:
            start_hotspot()            # never leave the Pi unreachable
        logger.warning(f"WiFi connect to {ssid!r} failed: {error}")
        return error

    logger.info(f"Connected to WiFi network: {ssid}")
    check = _nmcli(["networking", "connectivity", "check"], 20).stdout.strip()
    if check == "portal":
        return (f"Joined {ssid}, but it needs a sign-in page in a browser (hotel/school WiFi). "
                "The Pi can't do that, so reports can't be emailed from it.")
    if check in ("none", "limited"):
        return f"Joined {ssid}, but it has no internet connection right now."
    return ""


def _wpa_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _connect_wpa(ssid: str, password: str) -> str:
    """Older Raspberry Pi OS: replace this network's block in wpa_supplicant.conf."""
    conf_path = "/etc/wpa_supplicant/wpa_supplicant.conf"
    current = _run(["sudo", "-n", "cat", conf_path], 10)
    text = current.stdout if current.returncode == 0 else ""
    # drop earlier blocks for the same network so they don't pile up
    blocks = re.split(r"(?=\nnetwork=\{)", text)
    kept = [b for b in blocks if f"ssid={_wpa_quote(ssid)}" not in b]
    psk_line = f"    psk={_wpa_quote(password)}" if password else "    key_mgmt=NONE"
    new = "".join(kept).rstrip("\n") + \
        f"\n\nnetwork={{\n    ssid={_wpa_quote(ssid)}\n{psk_line}\n    priority=10\n}}\n"
    proc = subprocess.run(["sudo", "-n", "tee", conf_path], input=new,
                          capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        return f"Could not write {conf_path}: {proc.stderr.strip()}"
    _run(["sudo", "-n", "wpa_cli", "-i", IFACE, "reconfigure"], 15)
    logger.info(f"WiFi network saved in wpa_supplicant: {ssid}")
    return ""


# ---------------------------------------------------------------------------
# Setup hotspot
# ---------------------------------------------------------------------------

def start_hotspot(name: str = HOTSPOT_NAME, password: str = HOTSPOT_PASSWORD) -> str:
    """Start the setup hotspot. Returns "" or an error."""
    if not has_networkmanager():
        return "Setup hotspot needs NetworkManager"
    scan()                         # remember nearby networks: the radio can't scan in hotspot mode
    proc = _nmcli(["dev", "wifi", "hotspot", "ifname", IFACE, "con-name", name, "ssid", name,
                   "band", "bg", "password", password], 30)
    _forget_status()
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip() or "Couldn't start the setup hotspot"
    # Never start it by itself at boot - only when no network could be joined
    _nmcli(["connection", "modify", name, "connection.autoconnect", "no"], 10)
    logger.warning(f"No WiFi: started setup hotspot {name!r} ({HOTSPOT_ADDRESS})")
    return ""


def stop_hotspot(name: str = HOTSPOT_NAME):
    _nmcli(["connection", "down", "id", name], 15)
    _forget_status()


def hotspot_clients() -> int:
    """Devices currently connected to the setup hotspot (best effort)."""
    try:
        out = _run(["ip", "neigh", "show", "dev", IFACE], 5).stdout
        return sum(1 for line in out.splitlines()
                   if line.startswith("10.42.") and any(s in line for s in ("REACHABLE", "STALE", "DELAY")))
    except Exception:
        return 0


class SetupHotspotWatchdog(threading.Thread):
    """
    After boot: if WiFi hasn't connected within `boot_wait` seconds, start the
    setup hotspot. While it runs, every `retry_every` seconds (when nobody is
    connected to it) step aside briefly so the saved networks can be tried again.
    """

    def __init__(self, enabled: Callable[[], bool] = lambda: True,
                 on_change: Optional[Callable[[str], None]] = None,
                 boot_wait: float = 90, retry_every: float = 300, retry_wait: float = 45,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 password: Callable[[], str] = lambda: HOTSPOT_PASSWORD):
        super().__init__(daemon=True, name="wifi-watchdog")
        self.enabled, self.on_change = enabled, on_change or (lambda msg: None)
        self.boot_wait, self.retry_every, self.retry_wait = boot_wait, retry_every, retry_wait
        self.clock, self.sleep, self.password = clock, sleep, password
        self.stopped = False

    def _wait_for_wifi(self, seconds: float) -> bool:
        end = self.clock() + seconds
        while self.clock() < end and not self.stopped:
            if wifi_connected():
                return True
            self.sleep(5)
        return wifi_connected()

    def run(self):
        try:
            if not has_networkmanager() or not self.enabled():
                return
            if self._wait_for_wifi(self.boot_wait) or self.stopped:
                return
            with _lock:
                error = start_hotspot(password=self.password())
            if error:
                logger.error(f"Setup hotspot failed: {error}")
                return
            self.on_change(f"No WiFi - setup mode. Join '{HOTSPOT_NAME}' and ssh to {HOTSPOT_ADDRESS}")
            while not self.stopped:
                self.sleep(self.retry_every)
                if not hotspot_active():
                    return                 # someone connected it to a network meanwhile
                if hotspot_clients():
                    continue               # don't kick off whoever is setting it up
                with _lock:
                    stop_hotspot()
                if self._wait_for_wifi(self.retry_wait):
                    self.on_change("WiFi connected")
                    return
                with _lock:
                    if not wifi_connected():
                        start_hotspot(password=self.password())
        except Exception:
            logger.exception("WiFi watchdog error")
