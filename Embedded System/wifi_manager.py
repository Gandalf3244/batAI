"""
WiFi helpers shared by the TFT menu and the web settings UI.

Raspberry Pi OS Bookworm and later manage WiFi with NetworkManager (nmcli);
older releases use wpa_supplicant. Both are supported.
"""

import shutil
import subprocess
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _run(cmd: List[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _nmcli(args: List[str], timeout: float = 30) -> subprocess.CompletedProcess:
    """Run nmcli, retrying through passwordless sudo if the service user isn't allowed."""
    proc = _run(["nmcli"] + args, timeout)
    if proc.returncode != 0 and "not authorized" in (proc.stderr + proc.stdout).lower():
        proc = _run(["sudo", "-n", "nmcli"] + args, timeout)
    return proc


def has_networkmanager() -> bool:
    if not shutil.which("nmcli"):
        return False
    try:
        return _run(["nmcli", "-t", "-f", "RUNNING", "general"], 5).stdout.strip() == "running"
    except Exception:
        return False


def current_status() -> Dict[str, Optional[str]]:
    """Return {'ssid': ..., 'ip': ...} for wlan0 (values None when unknown)."""
    status: Dict[str, Optional[str]] = {"ssid": None, "ip": None}
    try:
        if shutil.which("iwgetid"):
            ssid = _run(["iwgetid", "-r"], 5).stdout.strip()
            status["ssid"] = ssid or None
        elif has_networkmanager():
            for line in _run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"], 10).stdout.splitlines():
                if line.startswith("yes:"):
                    status["ssid"] = line[4:] or None
        if shutil.which("hostname"):
            ips = _run(["hostname", "-I"], 5).stdout.split()
            status["ip"] = ips[0] if ips else None
    except Exception as e:
        logger.debug(f"WiFi status unavailable: {e}")
    return status


def scan() -> List[Dict]:
    """List nearby networks as [{'ssid', 'signal', 'secure'}], strongest first."""
    if not has_networkmanager():
        return []
    proc = _nmcli(["-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list", "--rescan", "yes"], 30)
    networks: Dict[str, Dict] = {}
    for line in proc.stdout.splitlines():
        # nmcli -t escapes ':' inside fields as '\:'
        parts = line.replace("\\:", "\x00").split(":")
        if len(parts) < 3:
            continue
        ssid, signal, security = (p.replace("\x00", ":") for p in parts[:3])
        if not ssid:
            continue
        try:
            strength = int(signal)
        except ValueError:
            strength = 0
        if ssid not in networks or networks[ssid]["signal"] < strength:
            networks[ssid] = {"ssid": ssid, "signal": strength, "secure": bool(security and security != "--")}
    return sorted(networks.values(), key=lambda n: n["signal"], reverse=True)


def _wpa_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def connect(ssid: str, password: str = "") -> str:
    """
    Connect wlan0 to a network and remember it.

    Returns:
        Empty string on success, error message on failure
    """
    ssid = ssid.strip()
    if not ssid:
        return "No WiFi network name (SSID) set"
    if password and not 8 <= len(password) <= 63:
        return "WiFi password must be 8-63 characters"

    try:
        if has_networkmanager():
            args = ["dev", "wifi", "connect", ssid]
            if password:
                args += ["password", password]
            proc = _nmcli(args, 60)
            if proc.returncode != 0:
                return (proc.stderr or proc.stdout).strip() or "nmcli failed"
            logger.info(f"Connected to WiFi network: {ssid}")
            return ""

        # Legacy wpa_supplicant: add a network block instead of replacing the whole file
        conf_path = "/etc/wpa_supplicant/wpa_supplicant.conf"
        psk_line = f"    psk={_wpa_quote(password)}" if password else "    key_mgmt=NONE"
        block = f"\nnetwork={{\n    ssid={_wpa_quote(ssid)}\n{psk_line}\n    priority=10\n}}\n"
        proc = subprocess.run(["sudo", "-n", "tee", "-a", conf_path], input=block,
                              capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return f"Could not write {conf_path}: {proc.stderr.strip()}"
        _run(["sudo", "-n", "wpa_cli", "-i", "wlan0", "reconfigure"], 15)
        logger.info(f"WiFi network added to wpa_supplicant: {ssid}")
        return ""
    except FileNotFoundError as e:
        return f"WiFi tools not available on this system ({e.filename})"
    except subprocess.TimeoutExpired:
        return "Timed out connecting to WiFi"
    except Exception as e:
        return f"WiFi error: {e}"
