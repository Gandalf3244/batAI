#!/usr/bin/env python3
"""
batmon - terminal control panel for the Bat Monitor.

Run it on the Pi (it opens automatically when you SSH in):

    batmon                 open the control panel
    batmon --autostart off stop it opening on SSH login (on to re-enable)
    batmon --config PATH   use a config.json other than the one next to this file

It talks to the running monitor service through its local-only control port
(127.0.0.1, config "web.port"), so changes apply immediately. If the service
isn't running it edits config.json directly, and the service uses the
settings when it next starts.

Keys: 1-8 switch pages, s start/stop monitoring, ctrl+s save the page,
r refresh, q quit (back to the normal shell).
"""

import json
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
AUTOSTART_OFF = Path.home() / ".batmon-no-autostart"

from rtc_scheduler import local_now  # noqa: E402  (the configured time zone)

from rich.text import Text  # noqa: E402  (installed with textual)
from textual import on, work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll  # noqa: E402
from textual.screen import ModalScreen  # noqa: E402
from textual.widget import Widget  # noqa: E402
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, ListItem,  # noqa: E402
                             ListView, RichLog, Select, Static, Switch, TabbedContent, TabPane)

CLASS_INFO = {
    "Rods_Fighting": ("Rods · fighting", "#ff8a65"),
    "Straws_Fighting": ("Straws · fighting", "#7c9cff"),
    "Straws_Talking": ("Straws · talking", "#3fd18f"),
    "Straws_Want_Food": ("Straws · want food", "#f5b544"),
}
LEVEL_STYLE = {"ERROR": "bold red", "CRITICAL": "bold red", "WARNING": "yellow", "DEBUG": "dim"}

# Typed settings fields: widget id "section__key" -> parser
NUMBER_FIELDS = {
    "email__smtp_port": int, "email__trendline_slope": float, "email__trendline_intercept": float,
    "audio__energy_threshold": float, "audio__min_vocalization_duration": float,
    "audio__max_vocalization_duration": float, "audio__silence_duration": float,
    "audio__chunk_duration": float, "model__confidence_threshold": float,
    "system__email_retry_attempts": int, "system__email_retry_delay": int,
    "display__timeout": int,
}
SECRET_FIELDS = {"email__sender_password"}
LABELS = {
    "schedule.start_time": "Start time", "schedule.stop_time": "Stop time",
    "email.smtp_host": "SMTP server", "email.smtp_port": "Port", "email.sender_email": "Sender address",
    "email.sender_password": "App password", "email.recipients": "Recipients",
    "email.trendline_slope": "Slope", "email.trendline_intercept": "Intercept",
    "audio.energy_threshold": "Energy threshold", "audio.min_vocalization_duration": "Shortest call",
    "audio.max_vocalization_duration": "Longest call", "audio.silence_duration": "Gap that ends a call",
    "model.confidence_threshold": "Minimum confidence", "audio.chunk_duration": "Processing block",
    "system.email_retry_attempts": "Email attempts", "system.email_retry_delay": "Wait between attempts",
    "display.timeout": "Screen off after",
}


# ---------------------------------------------------------------------------
# Talking to the monitor
# ---------------------------------------------------------------------------

class Backend:
    """Same requests as the monitor's control API. Falls back to editing
    config.json directly when the service isn't running."""

    def __init__(self, config_path: Optional[str] = None):
        from config_manager import ConfigManager
        from rtc_scheduler import apply_timezone
        self.config_path = config_path
        cfg = ConfigManager(config_path)
        self.port = int(cfg.get("web.port", 8080))
        apply_timezone(str(cfg.get("system.timezone") or "America/Los_Angeles"))   # header clock etc.
        self.base = f"http://127.0.0.1:{self.port}"
        self.live = False

    def request(self, path: str, body: Any = None, timeout: float = 70) -> Dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method="GET" if data is None else "POST",
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                self.live = True
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            self.live = True
            try:
                return json.loads(e.read())
            except ValueError:
                return {"ok": False, "error": f"HTTP {e.code}"}
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, OSError):
            self.live = False
            return self._local(path, body)

    def _local(self, path: str, body: Any) -> Dict[str, Any]:
        import web_ui
        from config_manager import ConfigManager
        ui = web_ui.WebUI(None, ConfigManager(self.config_path))
        route, _, query = path.partition("?")
        if body is None:
            if route == "/api/status":
                return ui.api_status()
            if route == "/api/config":
                return web_ui.public_config(ui.config)
            if route == "/api/logs":
                n = int(dict(p.split("=") for p in query.split("&") if "=" in p).get("lines", 300))
                return {"lines": web_ui.tail_log(web_ui.LOG_PATH, n)}
            if route == "/api/wifi":
                return ui.api_wifi_status()
            if route == "/api/wifi/scan":
                return ui.api_wifi_scan()
        else:
            if route == "/api/email/test":
                return ui.api_test_email()[1]
            handler = {"/api/config": ui.api_save_config, "/api/monitor": ui.api_monitor,
                       "/api/clock": ui.api_clock, "/api/wifi/connect": ui.api_wifi_connect}.get(route)
            if handler:
                return handler(body)[1]
        return {"ok": False, "error": f"Unknown request {path}"}


def service(action: str) -> Tuple[bool, str]:
    """systemctl for the monitor service (start/stop/restart need passwordless sudo)."""
    cmd = ["systemctl", "is-active", "bat-monitor"] if action == "status" else \
        ["sudo", "-n", "systemctl", action, "bat-monitor"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except FileNotFoundError:
        return False, "systemctl not available on this computer"
    except subprocess.TimeoutExpired:
        return False, "timed out"
    out = (p.stdout + p.stderr).strip()
    return p.returncode == 0, out or ("ok" if p.returncode == 0 else f"exit {p.returncode}")


def ntp_synced() -> Optional[bool]:
    try:
        p = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                           capture_output=True, text=True, timeout=5)
        return p.stdout.strip() == "yes" if p.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "—"
    s = max(0, int(s))
    h, m = s // 3600, s % 3600 // 60
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s % 60:02d}s"
    return f"{s}s"


def to_db(v: float) -> float:
    import math
    return 20 * math.log10(v) if v and v > 0 else -120.0


def class_label(c: str) -> Tuple[str, str]:
    return CLASS_INFO.get(c, (c.replace("_", " "), "#8d99b8"))


# ---------------------------------------------------------------------------
# Custom display widgets (they size themselves to the terminal)
# ---------------------------------------------------------------------------

class Meter(Widget):
    """Microphone level: background, loudest sound and threshold on a -100..0 dB scale."""
    DEFAULT_CSS = "Meter { height: 3; }"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.audio: Dict[str, Any] = {}

    def update_audio(self, audio: Dict[str, Any]):
        self.audio = audio or {}
        self.refresh()

    def render(self) -> Text:
        w = max(20, self.size.width)
        a = self.audio
        pos = lambda v: max(0, min(w - 1, int((to_db(v) + 100) / 100 * w)))
        fresh = a.get("fresh")
        floor = pos(a.get("floor", 0)) if fresh else -1
        loud = pos(a.get("loudest", 0)) if fresh else -1
        thr = pos(a.get("threshold", 0.001))
        t = Text()
        for i in range(w):
            if i == thr:
                t.append("┃", style="bold #ff6b6b")
            elif i <= floor:
                t.append("█", style="#5b78d6")
            elif i <= loud:
                t.append("█", style="#3fd18f" if i < thr else "#f5b544")
            else:
                t.append("░", style="#2a3552")
        t.append("\n")
        scale = [" "] * w
        for db in (-100, -80, -60, -40, -20, 0):
            label = f"{db}" if db else "0 dB"
            i = min(w - len(label), int((db + 100) / 100 * w))
            for j, ch in enumerate(label):
                scale[i + j] = ch
        t.append("".join(scale), style="dim")
        t.append("\n")
        if fresh:
            t.append("■ ", style="#5b78d6").append(f"background {to_db(a['floor']):.0f} dB   ")
            t.append("■ ", style="#3fd18f").append(f"loudest {to_db(a['loudest']):.0f} dB   ")
        else:
            t.append("no live audio (monitoring is off)   ", style="dim")
        t.append("┃ ", style="bold #ff6b6b").append(f"threshold {to_db(a.get('threshold', 0.001)):.0f} dB")
        return t


class Bars(Widget):
    """Calls per type as horizontal bars."""
    DEFAULT_CSS = "Bars { height: 4; }"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.counts: Dict[str, int] = {}

    def update_counts(self, counts: Dict[str, int]):
        self.counts = counts or {}
        self.refresh()

    def render(self) -> Text:
        names = list(CLASS_INFO) + [c for c in self.counts if c not in CLASS_INFO]
        top = max([1] + list(self.counts.values()))
        label_w = 19
        bar_w = max(5, self.size.width - label_w - 7)
        t = Text()
        for i, c in enumerate(names):
            label, color = class_label(c)
            n = self.counts.get(c, 0)
            fill = round(n / top * bar_w)
            t.append(f"{label:<{label_w}}", style="dim" if not n else "")
            t.append("█" * fill, style=color).append("░" * (bar_w - fill), style="#2a3552")
            t.append(f" {n:>5}")
            if i < len(names) - 1:
                t.append("\n")
        return t


class Timeline(Widget):
    """24-hour bar showing the recording window and the current time."""
    DEFAULT_CSS = "Timeline { height: 3; }"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.start, self.stop, self.enabled = "18:00", "06:00", True

    def set_window(self, start: str, stop: str, enabled: bool):
        self.start, self.stop, self.enabled = start, stop, enabled
        self.refresh()

    def render(self) -> Text:
        w = max(24, self.size.width)

        def mins(s):
            h, m = (int(x) for x in s.split(":"))
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError(s)
            return h * 60 + m
        try:
            a, b = mins(self.start), mins(self.stop)
        except ValueError:
            return Text("\nEnter valid times as HH:MM (00:00 to 23:59)", style="red")
        now = local_now()
        now_col = int((now.hour * 60 + now.minute) / 1440 * w)
        t = Text()
        ticks = [" "] * w
        for h in range(0, 24, 3):
            i = int(h * 60 / 1440 * w)
            for j, ch in enumerate(f"{h:02d}"):
                if i + j < w:
                    ticks[i + j] = ch
        t.append("".join(ticks), style="dim").append("\n")
        for i in range(w):
            m = (i + 0.5) / w * 1440
            inside = (a == b) or (a < b and a <= m < b) or (a > b and (m >= a or m < b))
            t.append("█" if inside else "░", style=("#f5b544" if self.enabled else "#8d6a2a") if inside else "#2a3552")
        t.append("\n" + " " * now_col + "▲ now", style="bold")
        return t


class Confirm(ModalScreen[bool]):
    DEFAULT_CSS = """
    Confirm { align: center middle; background: $background 60%; }
    Confirm > Vertical { width: 64; height: auto; padding: 1 2; border: thick $warning; background: $surface; }
    Confirm Label { width: 100%; margin-bottom: 1; }
    Confirm .title { text-style: bold; }
    Confirm Horizontal { height: auto; align-horizontal: right; }
    Confirm Button { margin-left: 2; }
    """

    def __init__(self, title: str, message: str, yes: str = "Confirm"):
        super().__init__()
        self.title_text, self.message, self.yes = title, message, yes

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.title_text, classes="title")
            yield Label(self.message)
            with Horizontal():
                yield Button("Cancel", id="no")
                yield Button(self.yes, id="yes", variant="warning")

    def on_mount(self):
        self.query_one("#no", Button).focus()

    @on(Button.Pressed)
    def done(self, event: Button.Pressed):
        self.dismiss(event.button.id == "yes")

    def key_escape(self):
        self.dismiss(False)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

def field(label: str, widget: Widget, hint: str = "") -> Vertical:
    parts: List[Widget] = [Label(label, classes="flabel"), widget]
    if hint:
        parts.append(Label(hint, classes="hint"))
    return Vertical(*parts, classes="field")


def num(id_: str, integer: bool = False) -> Input:
    return Input(id=id_, type="integer" if integer else "number", select_on_focus=False)


class BatMon(App):
    TITLE = "Bat Monitor"
    CSS = """
    #statusbar { height: 3; padding: 0 1; background: $panel; }
    #statusbar #status { width: 1fr; content-align: left middle; height: 3; }
    #statusbar Button { margin: 0 0 0 1; min-width: 20; }
    .row { height: auto; }
    .panel { border: round $primary 60%; border-title-color: $text-muted; border-title-style: bold;
             padding: 0 1; height: auto; width: 1fr; margin: 0 1 1 0; }
    .big { text-style: bold; color: $accent; }
    .dim, .hint { color: $text-muted; }
    Label.dim, Label.flabel, Label.hint { width: 100%; }
    .switch-row { height: auto; margin: 1 0; }
    .switch-row Label { height: 3; content-align: left middle; padding-left: 1; }
    #recent { height: auto; max-height: 12; }
    .form { padding: 1 2; height: auto; }
    .fields { grid-size: 2; grid-gutter: 0 2; grid-rows: auto; height: auto; }
    .field { height: auto; margin-bottom: 1; }
    .flabel { text-style: bold; }
    .field Input, .field Select { width: 100%; }
    .field Input.error { border: tall $error; }
    .actions { height: auto; margin-top: 1; }
    .actions Button { margin-right: 2; }
    .unsaved { color: $warning; content-align: left middle; height: 3; }
    #recipients { height: auto; max-height: 8; border: round $primary 40%; }
    #recip-row { height: auto; }
    #recip-row Input { width: 1fr; }
    #nets { height: auto; max-height: 14; }
    #logbox { height: 1fr; border: round $primary 40%; }
    #log-tools { height: auto; }
    #log-tools Select { width: 24; }
    #log-tools Input { width: 1fr; }
    .preview { color: $text-muted; margin-top: 1; }
    #thr-row { height: auto; }
    #thr-row Input { width: 18; }
    #thr-row Label { content-align: left middle; height: 3; padding: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("escape", "unfocus", "Leave field", show=False),
        Binding("s", "toggle_monitoring", "Start/stop"),
        Binding("ctrl+s", "save", "Save page"),
        Binding("r", "refresh", "Refresh"),
        *[Binding(str(i + 1), f"page('{p}')", show=False) for i, p in enumerate(
            ["dashboard", "schedule", "email", "detection", "wifi", "clock", "system", "logs"])],
    ]

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.backend = Backend(config_path)
        self.status: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.baseline: Dict[str, str] = {}
        self.recipients: List[str] = []

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(icon="🦇")          # the clock is in the subtitle, in the configured zone
        with Horizontal(id="statusbar"):
            yield Static("Connecting…", id="status")
            yield Button("…", id="monitor-btn", disabled=True)
        with TabbedContent(initial="dashboard"):
            with TabPane("1 Dashboard", id="dashboard"):
                with VerticalScroll():
                    with Horizontal(classes="row"):
                        with Vertical(classes="panel", id="p-tonight"):
                            yield Static("—", id="tonight-total", classes="big")
                            yield Bars(id="bars")
                        with Vertical(classes="panel", id="p-session"):
                            yield Static("", id="session")
                    with Vertical(classes="panel", id="p-mic"):
                        yield Meter(id="meter")
                    with Horizontal(classes="row"):
                        with Vertical(classes="panel", id="p-food"):
                            yield Static("", id="food")
                        with Vertical(classes="panel", id="p-health"):
                            yield Static("", id="health")
                    with Vertical(classes="panel", id="p-recent"):
                        yield DataTable(id="recent", cursor_type="none", zebra_stripes=True)

            with TabPane("2 Schedule", id="schedule"):
                with VerticalScroll(classes="form"):
                    yield Label("Monitoring starts and stops automatically every day. The window can run past "
                                "midnight (e.g. 18:00 → 06:00). The report is emailed when it ends.", classes="dim")
                    with Grid(classes="fields"):
                        yield field("Start (HH:MM)", Input(id="schedule__start_time", restrict=r"[0-9:]*", max_length=5))
                        yield field("Stop (HH:MM)", Input(id="schedule__stop_time", restrict=r"[0-9:]*", max_length=5))
                    yield Timeline(id="timeline")
                    yield Static("", id="timeline-sum", classes="preview")
                    yield Horizontal(Switch(id="schedule__enabled"),
                                     Label("Use the schedule  (off = monitoring only runs when started by hand)"),
                                     classes="switch-row")
                    yield self.actions("schedule")

            with TabPane("3 Email", id="email"):
                with VerticalScroll(classes="form"):
                    yield Label("Gmail needs an app password (Google account → Security → App passwords).",
                                classes="dim")
                    with Grid(classes="fields"):
                        yield field("SMTP server", Input(id="email__smtp_host"), "smtp.gmail.com / smtp.office365.com")
                        yield field("Port", num("email__smtp_port", True), "587 = STARTTLS, 465 = SSL")
                        yield field("Sender address", Input(id="email__sender_email"))
                        yield field("App password", Input(id="email__sender_password", password=True),
                                    "Leave blank to keep the saved one")
                    yield Label("Recipients", classes="flabel")
                    yield ListView(id="recipients")
                    with Horizontal(id="recip-row"):
                        yield Input(placeholder="add an address and press Enter", id="recip-new")
                        yield Button("Remove selected", id="recip-remove")
                    yield Label("\nFood prediction trendline:  consumption = slope × want-food calls/hr + intercept",
                                classes="flabel")
                    with Grid(classes="fields"):
                        yield field("Slope", num("email__trendline_slope"))
                        yield field("Intercept", num("email__trendline_intercept"))
                    yield Static("", id="trend-preview", classes="preview")
                    with Horizontal(classes="actions"):
                        yield Button("Save", id="save-email", variant="primary")
                        yield Button("Send test email", id="test-email")
                        yield Label("", id="unsaved-email", classes="unsaved")

            with TabPane("4 Detection", id="detection"):
                with VerticalScroll(classes="form"):
                    yield Label("The AI labels every sound it hears, even silence, so only sounds louder than the "
                                "threshold are classified. Set it just above the background level during a quiet "
                                "period, below the level of real calls. Applies within a second.", classes="dim")
                    yield Meter(id="meter2")
                    yield Label("Energy threshold", classes="flabel")
                    with Horizontal(id="thr-row"):
                        yield Button("−3 dB", id="thr-down")
                        yield num("audio__energy_threshold")
                        yield Button("+3 dB", id="thr-up")
                        yield Label("", id="thr-db")
                    with Grid(classes="fields"):
                        yield field("Minimum confidence (0–1)", num("model__confidence_threshold"),
                                    "Less certain classifications are ignored")
                        yield field("Shortest call (s)", num("audio__min_vocalization_duration"))
                        yield field("Longest call (s)", num("audio__max_vocalization_duration"), "Longer sounds are split")
                        yield field("Gap that ends a call (s)", num("audio__silence_duration"))
                    yield Label("\nAudio capture (applies next time monitoring starts)", classes="flabel")
                    with Grid(classes="fields"):
                        rates = [("16 kHz", 16000), ("22.05 kHz", 22050), ("32 kHz", 32000), ("44.1 kHz", 44100),
                                 ("48 kHz", 48000)]
                        yield field("Microphone sample rate", Select(rates, id="audio__sample_rate", allow_blank=False))
                        yield field("Channels", Select([("1 (mono)", 1), ("2 (I²S, INMP441)", 2)],
                                                       id="audio__channels", allow_blank=False))
                        yield field("Processing block (s)", num("audio__chunk_duration"))
                        yield field("Model sample rate", Select(rates, id="model__sample_rate", allow_blank=False),
                                    "Rate the AI was trained on; needs a service restart")
                    yield self.actions("detection")

            with TabPane("5 WiFi", id="wifi"):
                with VerticalScroll(classes="form"):
                    yield Static("", id="wifi-now")
                    with Horizontal(classes="actions"):
                        yield Button("Scan for networks", id="wifi-scan")
                    yield DataTable(id="nets", cursor_type="row")
                    yield Label("\n⚠ If you're connected over WiFi, switching networks drops this SSH session. "
                                "Reconnect using the Pi's address on the new network.", classes="dim")
                    with Grid(classes="fields"):
                        yield field("Network name (SSID)", Input(id="wifi-ssid"), "Select a scanned network to fill this")
                        yield field("Password", Input(id="wifi-pw", password=True), "Blank = saved password")
                    with Horizontal(classes="actions"):
                        yield Button("Connect", id="wifi-connect", variant="primary")

            with TabPane("6 Clock", id="clock"):
                with VerticalScroll(classes="form"):
                    yield Static("", id="clock-now")
                    yield Label("\nThe schedule runs on the DS3231 clock, which keeps time without power or "
                                "internet. When the Pi has internet its own clock is set automatically and can "
                                "be copied to the DS3231.", classes="dim")
                    with Horizontal(classes="actions"):
                        yield Button("Copy Pi system time → RTC", id="clock-sync", variant="primary")
                    yield Label("\nTime zone (used for the schedule and every time shown)", classes="flabel")
                    with Horizontal(classes="actions"):
                        from rtc_scheduler import COMMON_TIMEZONES
                        yield Select([(f"{label}  ({zone})", zone) for label, zone in COMMON_TIMEZONES],
                                     id="clock-tz", allow_blank=False, value=COMMON_TIMEZONES[0][1])
                        yield Button("Set time zone", id="clock-tz-set", variant="primary")
                    yield field("Or set by hand (YYYY-MM-DD HH:MM)", Input(id="clock-manual",
                                                                            restrict=r"[0-9: -]*", max_length=19))
                    with Horizontal(classes="actions"):
                        yield Button("Set clock", id="clock-set")

            with TabPane("7 System", id="system"):
                with VerticalScroll(classes="form"):
                    yield Static("", id="sys-info")
                    with Grid(classes="fields"):
                        yield field("Log detail", Select([(x, x) for x in ("DEBUG", "INFO", "WARNING", "ERROR")],
                                                         id="system__log_level", allow_blank=False),
                                    "DEBUG logs every classification")
                        yield field("Email attempts", num("system__email_retry_attempts", True))
                        yield field("Wait between attempts (s)", num("system__email_retry_delay", True))
                    yield Label("\nDevice screen and knob", classes="flabel")
                    yield Horizontal(Switch(id="display__rotate_180"), Label("Turn the screen 180°"),
                                     classes="switch-row")
                    yield Horizontal(Switch(id="display__encoder_reverse"), Label("Reverse the knob direction"),
                                     classes="switch-row")
                    with Grid(classes="fields"):
                        yield field("Screen off after (s, 0 = never)", num("display__timeout", True),
                                    "Any knob input wakes it")
                    yield self.actions("system")
                    yield Label("\nMonitor service", classes="flabel")
                    with Horizontal(classes="actions"):
                        yield Button("Restart service", id="svc-restart")
                        yield Button("Start service", id="svc-start")
                        yield Button("Stop service", id="svc-stop", variant="error")

            with TabPane("8 Logs", id="logs"):
                with Vertical(classes="form"):
                    with Horizontal(id="log-tools"):
                        yield Select([("All", "ALL"), ("Info and up", "INFO"), ("Warnings", "WARNING"),
                                      ("Errors", "ERROR")], id="log-level", allow_blank=False, value="ALL")
                        yield Input(placeholder="filter…", id="log-filter")
                    yield RichLog(id="logbox", wrap=True, markup=False, auto_scroll=True)
        yield Footer()

    def actions(self, page: str) -> Horizontal:
        return Horizontal(Button("Save", id=f"save-{page}", variant="primary"),
                          Label("", id=f"unsaved-{page}", classes="unsaved"), classes="actions")

    # -- startup / refresh -----------------------------------------------------

    def on_mount(self):
        for theme in ("tokyo-night", "textual-dark"):
            if theme in self.available_themes:
                self.theme = theme
                break
        self.tick_clock()
        self.set_interval(1, self.tick_clock)
        for pid, title in [("p-tonight", "Tonight"), ("p-session", "Session"), ("p-mic", "Microphone"),
                           ("p-food", "Food prediction"), ("p-health", "Health"), ("p-recent", "Recent detections")]:
            self.query_one(f"#{pid}").border_title = title
        t = self.query_one("#recent", DataTable)
        t.add_columns("Time", "Call type", "Confidence")
        n = self.query_one("#nets", DataTable)
        n.add_columns("Signal", "Network", "Security")
        self.load_config()
        self.refresh_status()
        self.set_interval(2, self.refresh_status)
        self.set_interval(3, self.refresh_logs)

    def tick_clock(self):
        """Header: host and the time in the monitor's configured time zone."""
        from rtc_scheduler import timezone_abbr
        self.sub_title = f"{socket.gethostname()}  ·  {local_now():%H:%M} {timezone_abbr()}"

    def action_unfocus(self):
        self.set_focus(None)

    def action_refresh(self):
        self.refresh_status()
        if not self.any_dirty():
            self.load_config()

    def action_page(self, page: str):
        self.query_one(TabbedContent).active = page

    @on(TabbedContent.TabActivated)
    def tab_changed(self, event: TabbedContent.TabActivated):
        page = event.pane.id
        if page == "logs":
            self.refresh_logs(force=True)
        elif page == "wifi":
            self.load_wifi()
        elif page and page in self.baseline and not self.is_dirty(page):
            self.load_config()   # pick up changes made on the device's own screen

    @work(thread=True, exclusive=True, group="status")
    def refresh_status(self):
        s = self.backend.request("/api/status", timeout=5)
        now = datetime.now().timestamp()
        if now - getattr(self, "_ntp_checked", 0) > 30:
            self._ntp_checked, self._ntp = now, ntp_synced()
        s["_ntp"] = self._ntp
        self.call_from_thread(self.render_status, s)

    # -- dashboard -------------------------------------------------------------

    def render_status(self, s: Dict[str, Any]):
        self.status = s
        btn = self.query_one("#monitor-btn", Button)
        st = self.query_one("#status", Static)
        sched = s.get("schedule") or {}
        if not s.get("service_running"):
            st.update(Text.assemble(("⚠ Monitor service is not running", "bold yellow"),
                                    ("  — settings are saved to config.json and used when it starts", "dim")))
            btn.label, btn.variant, btn.disabled = "Start service", "warning", False
        elif s.get("monitoring"):
            sess = s.get("session") or {}
            mode = " (manual)" if s.get("mode") == "manual" else ""
            tail = f" · stops in {fmt_dur(sched.get('time_until_stop'))}" if sched.get("time_until_stop") else ""
            st.update(Text.assemble(("● MONITORING" + mode, "bold green"),
                                    (f"  since {sess.get('start', '')[11:16]} · {fmt_dur(sess.get('duration_s'))}"
                                     f" · {sess.get('total', 0)} calls{tail}", "")))
            btn.label, btn.variant, btn.disabled = "Stop monitoring", "error", False
        else:
            if sched.get("enabled") is False:
                tail = "schedule off — start by hand"
            elif s.get("skip_window"):
                tail = "stopped by hand — resumes at the next window"
            else:
                tail = f"starts in {fmt_dur(sched.get('time_until_start'))} ({sched.get('start_time', '?')})"
            st.update(Text.assemble(("○ IDLE", "bold"), (f"  {tail}", "dim")))
            btn.label, btn.variant, btn.disabled = "Start monitoring", "success", False
        if s.get("last_error"):
            self.notify_once(s["last_error"])

        if not s.get("service_running"):
            return
        sess, last = s.get("session"), s.get("last_session")
        src = sess or last
        self.query_one("#tonight-total", Static).update(Text.assemble(
            (str(src["total"]) if src else "—", "bold"),
            ("  calls tonight" if sess else
             (f"  calls last session ({last['start'][11:16]}–{(last.get('end') or '')[11:16]})" if last
              else "  no session yet"), "dim")))
        self.query_one("#bars", Bars).update_counts(src["by_class"] if src else {})
        lines = [("Status", "Recording" + (" (manual)" if s.get("mode") == "manual" else "") if s.get("monitoring")
                  else "Idle"),
                 ("Started", sess["start"][11:16] if sess else "—"),
                 ("Running for", fmt_dur(sess["duration_s"]) if sess else "—"),
                 ("Schedule", f"{sched.get('start_time', '?')} → {sched.get('stop_time', '?')}"
                  + ("" if sched.get("enabled", True) else " (off)")),
                 ("Sounds heard", str(sess["segments"]) if sess else "—"),
                 ("Below confidence", str(sess["low_confidence"]) if sess else "—")]
        self.query_one("#session", Static).update(self.kv(lines))

        m = float((self.config.get("email") or {}).get("trendline_slope") or 0)
        b = float((self.config.get("email") or {}).get("trendline_intercept") or 0)
        if sess:
            rate = sess.get("want_food_rate", 0)
            self.query_one("#food", Static).update(Text.assemble(
                (f"{m * rate + b:.1f}", "bold"), ("  predicted consumption tomorrow\n", "dim"),
                (f"{m} × {rate:.2f} want-food calls/hr + {b}", "dim")))
        else:
            self.query_one("#food", Static).update(Text(f"—  (starts with monitoring)\n{m} × calls/hr + {b}", "dim"))

        c, h, e = s.get("clock") or {}, s.get("system") or {}, s.get("email") or {}
        hl = [("Clock", "DS3231 RTC" if c.get("rtc_present") and not c.get("rtc_error")
               else ("RTC error" if c.get("rtc_error") else "system clock (no RTC)")),
              ("CPU temp", f"{h['cpu_temp_c']} °C" if "cpu_temp_c" in h else "—"),
              ("Memory · load", f"{h.get('mem_used_pct', '—')}% · {h.get('load', '—')}"),
              ("Disk free", f"{h['disk_free_gb']} GB" if "disk_free_gb" in h else "—"),
              ("Email", "not set up" if not e.get("configured") else
               f"{e.get('recipients', 0)} recipient(s)" + (f", {e['queued']} queued" if e.get("queued") else "")),
              ("Network", self.network_text(s.get("network") or {})),
              ("Uptime", fmt_dur(s.get("uptime_s")))]
        self.query_one("#health", Static).update(self.kv(hl))

        audio = s.get("audio") or {}
        self.query_one("#meter", Meter).update_audio(audio)
        self.query_one("#meter2", Meter).update_audio(audio)

        table = self.query_one("#recent", DataTable)
        table.clear()
        for r in (sess or {}).get("recent", []):
            label, color = class_label(r["class"])
            table.add_row(r["time"], Text("● ", style=color) + Text(label), f"{r['confidence'] * 100:.0f}%")
        if not table.row_count:
            table.add_row("", Text("listening — nothing yet" if s.get("monitoring") else "not monitoring",
                                   style="dim"), "")

        self.render_clock(c, s.get("_ntp"))
        tz_select = self.query_one("#clock-tz", Select)
        if c.get("timezone") and not tz_select.has_focus and not tz_select.expanded:
            try:
                tz_select.value = c["timezone"]
            except Exception:
                pass                              # a zone that isn't in the short list
        mdl = s.get("model") or {}
        self.query_one("#sys-info", Static).update(self.kv([
            ("Host", s.get("hostname", "")),
            ("Model", f"{mdl.get('file', '?')}" + ("  (count correction on)" if mdl.get("count_correction") else "")),
            ("Device screen", "connected" if s.get("tft_display") else "not found (headless)"),
            ("Monitor uptime", fmt_dur(s.get("uptime_s")))]))

    @staticmethod
    def network_text(n: Dict[str, Any]) -> str:
        if n.get("hotspot"):
            return f"setup mode ({n.get('hotspot_name')})"
        if n.get("connected"):
            return f"{n.get('ssid')} · {n.get('ip') or '…'}"
        return "WiFi not connected"

    def notify_once(self, msg: str):
        if getattr(self, "_last_error_shown", None) != msg:
            self._last_error_shown = msg
            self.notify(msg, title="Monitor reported a problem", severity="error", timeout=15)

    @staticmethod
    def kv(rows) -> Text:
        t = Text()
        for i, (k, v) in enumerate(rows):
            t.append(f"{k:<18}", style="dim").append(str(v))
            if i < len(rows) - 1:
                t.append("\n")
        return t

    # -- start / stop ------------------------------------------------------------

    def action_toggle_monitoring(self):
        self.on_monitor_btn()

    @on(Button.Pressed, "#monitor-btn")
    def on_monitor_btn(self):
        s = self.status
        if not s.get("service_running"):
            self.run_service("start")
        elif s.get("monitoring"):
            msg = "This ends the current session and emails the report now."
            if (s.get("schedule") or {}).get("is_within_schedule"):
                msg += " It won't restart until the next scheduled window."
            self.push_screen(Confirm("Stop monitoring?", msg, "Stop monitoring"),
                             lambda yes: yes and self.monitor("stop"))
        else:
            self.monitor("start")

    @work(thread=True, exclusive=True, group="monitor")
    def monitor(self, action: str):
        self.call_from_thread(self.notify, "Stopping (finishing the report)…" if action == "stop" else "Starting…")
        r = self.backend.request("/api/monitor", {"action": action})
        self.call_from_thread(self.notify, ("Monitoring started" if action == "start" else "Monitoring stopped")
                              if r.get("ok") else r.get("error", "Failed"), severity="information" if r.get("ok") else "error")
        self.call_from_thread(self.refresh_status)

    @work(thread=True, group="service")
    def run_service(self, action: str):
        self.call_from_thread(self.notify, f"{action.capitalize()}ing the monitor service…")
        ok, out = service(action)
        self.call_from_thread(self.notify, f"Service {action}: {'done' if ok else out}",
                              severity="information" if ok else "error", timeout=8)
        self.call_from_thread(self.refresh_status)

    @on(Button.Pressed, "#svc-restart")
    def svc_restart(self):
        self.push_screen(Confirm("Restart the monitor service?",
                                 "Monitoring stops briefly; tonight's session report is emailed first.", "Restart"),
                         lambda yes: yes and self.run_service("restart"))

    @on(Button.Pressed, "#svc-start")
    def svc_start(self):
        self.run_service("start")

    @on(Button.Pressed, "#svc-stop")
    def svc_stop(self):
        self.push_screen(Confirm("Stop the monitor service?",
                                 "Nothing is recorded until it's started again (or the Pi reboots).", "Stop service"),
                         lambda yes: yes and self.run_service("stop"))

    # -- settings forms ----------------------------------------------------------

    def pages(self):
        return ["schedule", "email", "detection", "system"]

    def form_widgets(self, page: str):
        return [w for w in self.query_one(f"#{page}").query("Input, Switch, Select") if w.id and "__" in w.id]

    def read_form(self, page: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        updates: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        for w in self.form_widgets(page):
            section, key = w.id.split("__", 1)
            if isinstance(w, Switch):
                value: Any = w.value
            elif isinstance(w, Select):
                value = w.value
            else:
                raw = w.value.strip()
                if w.id in SECRET_FIELDS and not raw:
                    continue
                if w.id in NUMBER_FIELDS:
                    try:
                        value = NUMBER_FIELDS[w.id](raw)
                    except ValueError:
                        errors.append(f"{LABELS.get(f'{section}.{key}', key)}: enter a number")
                        w.add_class("error")
                        continue
                else:
                    value = raw
            updates.setdefault(section, {})[key] = value
        if page == "email":
            updates.setdefault("email", {})["recipients"] = list(self.recipients)
        return updates, errors

    def snapshot(self, page: str) -> str:
        try:
            return json.dumps(self.read_form(page)[0], sort_keys=True, default=str)
        except Exception:
            return ""

    def is_dirty(self, page: str) -> bool:
        return page in self.baseline and self.snapshot(page) != self.baseline[page]

    def any_dirty(self) -> bool:
        return any(self.is_dirty(p) for p in self.pages())

    def update_dirty(self, page: str):
        try:
            self.query_one(f"#unsaved-{page}", Label).update("● unsaved changes" if self.is_dirty(page) else "")
        except Exception:
            pass

    @work(thread=True, exclusive=True, group="config")
    def load_config(self):
        cfg = self.backend.request("/api/config", timeout=10)
        self.call_from_thread(self.fill_forms, cfg)

    def fill_forms(self, cfg: Dict[str, Any]):
        if "email" not in cfg:
            self.notify(f"Couldn't load settings: {cfg.get('error', cfg)}", severity="error")
            return
        self.config = cfg
        for page in self.pages():
            for w in self.form_widgets(page):
                section, key = w.id.split("__", 1)
                w.remove_class("error")
                if w.id in SECRET_FIELDS:
                    w.value = ""
                    w.placeholder = "•••••••• saved" if cfg[section].get(key + "_set") else "not set"
                    continue
                value = cfg.get(section, {}).get(key)
                if isinstance(w, Switch):
                    w.value = bool(value)
                elif isinstance(w, Select):
                    try:
                        w.value = value
                    except Exception:
                        pass
                else:
                    w.value = "" if value is None else str(value)
        self.recipients = list(cfg["email"].get("recipients") or [])
        self.render_recipients()
        self.query_one("#wifi-ssid", Input).value = cfg.get("wifi", {}).get("ssid", "") or ""
        self.call_after_refresh(self.reset_baselines)

    def reset_baselines(self):
        for page in self.pages():
            self.baseline[page] = self.snapshot(page)
            self.update_dirty(page)
        self.update_previews()

    @on(Input.Changed)
    @on(Switch.Changed)
    @on(Select.Changed)
    def field_changed(self, event):
        w = event.control if hasattr(event, "control") else event.input
        if w is None or not w.id:
            return
        if isinstance(w, Input):
            w.remove_class("error")
        for page in self.pages():
            if w.id.split("__")[0] in ("schedule", "email", "audio", "model", "system", "display"):
                self.update_dirty(page)
        self.update_previews()
        if w.id == "log-filter" or w.id == "log-level":
            self.render_logs()

    def update_previews(self):
        q = lambda i: self.query_one(f"#{i}")
        start, stop = q("schedule__start_time").value, q("schedule__stop_time").value
        enabled = q("schedule__enabled").value
        q("timeline").set_window(start, stop, enabled)
        try:
            a = [int(x) for x in start.split(":")]
            b = [int(x) for x in stop.split(":")]
            if len(a) != 2 or len(b) != 2 or a[0] > 23 or b[0] > 23 or a[1] > 59 or b[1] > 59:
                raise ValueError
            length = ((b[0] * 60 + b[1]) - (a[0] * 60 + a[1])) % 1440 or 1440
            q("timeline-sum").update(f"Records {length // 60}h {length % 60:02d}m every day"
                                     + (" overnight" if a > b else "") + ("" if enabled else " — schedule is OFF"))
        except (ValueError, IndexError):
            q("timeline-sum").update("Enter times as HH:MM")
        try:
            m, b = float(q("email__trendline_slope").value), float(q("email__trendline_intercept").value)
            q("trend-preview").update("   ".join(f"{x} calls/hr → {m * x + b:.1f}" for x in (0, 5, 10, 20)))
        except ValueError:
            q("trend-preview").update("")
        try:
            v = float(q("audio__energy_threshold").value)
            q("thr-db").update(f"≈ {to_db(v):.1f} dBFS")
        except ValueError:
            q("thr-db").update("")

    def action_save(self):
        page = self.query_one(TabbedContent).active
        if page in self.pages():
            self.save(page)

    @on(Button.Pressed, "#save-schedule, #save-email, #save-detection, #save-system")
    def save_pressed(self, event: Button.Pressed):
        self.save(event.button.id.removeprefix("save-"))

    def save(self, page: str):
        updates, errors = self.read_form(page)
        if errors:
            self.notify("\n".join(errors), title="Fix these first", severity="error")
            return
        self.do_save(page, updates)

    @work(thread=True, group="save")
    def do_save(self, page: str, updates):
        r = self.backend.request("/api/config", updates)
        self.call_from_thread(self.after_save, page, r)

    def after_save(self, page: str, r: Dict[str, Any]):
        if r.get("errors"):
            for path, msg in r["errors"].items():
                try:
                    self.query_one(f"#{path.replace('.', '__')}").add_class("error")
                except Exception:
                    pass
            self.notify("\n".join(f"{LABELS.get(p, p)}: {m}" for p, m in r["errors"].items()),
                        title="Not saved", severity="error", timeout=10)
            return
        if not r.get("ok"):
            self.notify(r.get("error", "Save failed"), severity="error")
            return
        self.fill_forms(r["config"])
        where = "" if self.backend.live else " (to config.json — used when the service starts)"
        self.notify("Saved" + where)
        for note in r.get("notes", []):
            self.notify(note, timeout=8)

    # recipients
    def render_recipients(self):
        lv = self.query_one("#recipients", ListView)
        lv.clear()
        for addr in self.recipients:
            lv.append(ListItem(Label(addr)))
        if not self.recipients:
            lv.append(ListItem(Label("(no recipients yet)", classes="dim")))
        self.update_dirty("email")

    @on(Input.Submitted, "#recip-new")
    def add_recipient(self, event: Input.Submitted):
        addr = event.value.strip().rstrip(",")
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", addr):
            self.notify(f'"{addr}" doesn\'t look like an email address', severity="error")
            return
        if addr.lower() not in (a.lower() for a in self.recipients):
            self.recipients.append(addr)
        event.input.value = ""
        self.render_recipients()

    @on(Button.Pressed, "#recip-remove")
    def remove_recipient(self):
        lv = self.query_one("#recipients", ListView)
        if lv.index is not None and 0 <= lv.index < len(self.recipients):
            self.recipients.pop(lv.index)
            self.render_recipients()
        else:
            self.notify("Select an address in the list first (arrow keys or click)")

    @on(Button.Pressed, "#test-email")
    def test_email(self):
        if self.is_dirty("email"):
            self.notify("Save the email settings first", severity="warning")
            return
        self.send_test()

    @work(thread=True, group="email")
    def send_test(self):
        self.call_from_thread(self.notify, "Sending test email…")
        r = self.backend.request("/api/email/test", {})
        self.call_from_thread(self.notify, "Test email sent" if r.get("ok") else r.get("error", "Failed"),
                              severity="information" if r.get("ok") else "error", timeout=10)

    # detection threshold buttons
    @on(Button.Pressed, "#thr-up, #thr-down")
    def nudge_threshold(self, event: Button.Pressed):
        inp = self.query_one("#audio__energy_threshold", Input)
        try:
            v = float(inp.value)
        except ValueError:
            v = 0.001
        factor = 10 ** ((3 if event.button.id == "thr-up" else -3) / 20)
        inp.value = f"{min(1.0, max(1e-6, v * factor)):.3g}"

    # -- wifi ----------------------------------------------------------------------

    @work(thread=True, group="wifi")
    def load_wifi(self):
        r = self.backend.request("/api/wifi", timeout=15)
        cur = r.get("current") or {}
        if cur.get("hotspot"):
            rows = [("Setup mode", f"the Pi is running its own network '{cur.get('hotspot_name')}'"),
                    ("Reach it at", r.get("hotspot_address", "10.42.0.1")),
                    ("", "Pick your network below to leave setup mode.")]
        else:
            rows = [("Connected to", cur.get("ssid") or "not connected"), ("IP address", cur.get("ip") or "—")]
        rows.append(("Managed by", "NetworkManager" if r.get("networkmanager") else "wpa_supplicant"))
        text = self.kv(rows)
        self.call_from_thread(lambda: self.query_one("#wifi-now", Static).update(text))

    @on(Button.Pressed, "#wifi-scan")
    def wifi_scan(self):
        self.notify("Scanning…")
        self.do_scan()

    @work(thread=True, group="wifi-scan", exclusive=True)
    def do_scan(self):
        r = self.backend.request("/api/wifi/scan", timeout=40)
        self.call_from_thread(self.show_networks, r.get("networks", []))

    def show_networks(self, nets):
        t = self.query_one("#nets", DataTable)
        t.clear()
        for n in nets:
            bars = "▂▄▆█"[:max(1, round(n["signal"] / 25))]
            t.add_row(Text(f"{bars:<4} {n['signal']}%", style="green"), n["ssid"], "secured" if n["secure"] else "open",
                      key=n["ssid"])
        if not nets:
            self.notify("No networks found (scanning needs NetworkManager)", severity="warning")

    @on(DataTable.RowSelected, "#nets")
    def pick_network(self, event: DataTable.RowSelected):
        self.query_one("#wifi-ssid", Input).value = str(event.row_key.value)
        self.query_one("#wifi-pw", Input).value = ""
        self.query_one("#wifi-pw", Input).focus()

    @on(Button.Pressed, "#wifi-connect")
    def wifi_connect(self):
        ssid = self.query_one("#wifi-ssid", Input).value.strip()
        if not ssid:
            self.notify("Enter a network name", severity="error")
            return
        self.push_screen(Confirm(f"Connect to “{ssid}”?",
                                 "If you're connected over WiFi, this SSH session will drop. Reconnect on the "
                                 "new network afterwards.", "Connect"),
                         lambda yes: yes and self.do_connect(ssid, self.query_one("#wifi-pw", Input).value))

    @work(thread=True, group="wifi-connect")
    def do_connect(self, ssid: str, pw: str):
        body = {"ssid": ssid}
        if pw:
            body["password"] = pw
        r = self.backend.request("/api/wifi/connect", body, timeout=90)
        msg = f"Connected to {ssid}" if r.get("ok") else (r.get("error") or json.dumps(r.get("errors")))
        self.call_from_thread(self.notify, msg, severity="information" if r.get("ok") else "error", timeout=10)
        self.call_from_thread(self.load_wifi)

    # -- clock ---------------------------------------------------------------------

    def render_clock(self, c: Dict[str, Any], synced: Optional[bool]):
        rows = [("Time zone", c.get("timezone_label") or "—"),
                ("Pi system clock", (c.get("system_time") or "—").replace("T", "  "))]
        if c.get("rtc_time"):
            rows.append(("DS3231 RTC", c["rtc_time"].replace("T", "  ")))
            if c.get("system_time"):
                diff = (datetime.fromisoformat(c["rtc_time"]) - datetime.fromisoformat(c["system_time"])).total_seconds()
                rows.append(("Difference", "in sync" if abs(diff) < 2 else
                             f"RTC {'ahead' if diff > 0 else 'behind'} by {fmt_dur(abs(diff))}"))
            if c.get("rtc_temp_c") is not None:
                rows.append(("RTC temperature", f"{c['rtc_temp_c']:.1f} °C"))
        else:
            rows.append(("DS3231 RTC", c.get("rtc_error") or "not detected — schedule uses the system clock"))
        if synced is not None:
            rows.append(("Internet time", "synced ✓" if synced else "not synced (no internet?)"))
        self.query_one("#clock-now", Static).update(self.kv(rows))

    @on(Button.Pressed, "#clock-tz-set")
    def set_timezone(self):
        zone = self.query_one("#clock-tz", Select).value
        self.save_timezone(str(zone))

    @work(thread=True, group="clock")
    def save_timezone(self, zone: str):
        r = self.backend.request("/api/config", {"system": {"timezone": zone}})
        if r.get("ok"):
            from rtc_scheduler import apply_timezone, timezone_label
            apply_timezone(zone)                  # this panel's own clock too
            self.call_from_thread(self.notify, f"Time zone: {timezone_label(zone)}")
        else:
            self.call_from_thread(self.notify, str(r.get("errors") or r.get("error")), severity="error")
        self.call_from_thread(self.refresh_status)

    @on(Button.Pressed, "#clock-sync")
    def clock_sync(self):
        if ntp_synced() is False:
            self.push_screen(Confirm("System clock isn't internet-synced",
                                     f"The Pi's clock says {local_now():%Y-%m-%d %H:%M}. Copy it to the RTC anyway?",
                                     "Copy anyway"),
                             lambda yes: yes and self.set_clock(local_now()))
        else:
            self.set_clock(local_now())

    @on(Button.Pressed, "#clock-set")
    def clock_manual(self):
        raw = self.query_one("#clock-manual", Input).value.strip()
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S" if raw.count(":") == 2 else "%Y-%m-%d %H:%M")
        except ValueError:
            self.notify("Use the format 2026-10-24 21:15", severity="error")
            return
        self.set_clock(dt)

    @work(thread=True, group="clock")
    def set_clock(self, dt: datetime):
        r = self.backend.request("/api/clock", {"datetime": dt.replace(microsecond=0).isoformat()})
        self.call_from_thread(self.notify, r.get("message") or r.get("error") or "Done",
                              severity="information" if r.get("ok") else "error", timeout=8)
        self.call_from_thread(self.refresh_status)

    # -- logs ----------------------------------------------------------------------

    def refresh_logs(self, force: bool = False):
        if force or self.query_one(TabbedContent).active == "logs":
            self.fetch_logs()

    @work(thread=True, exclusive=True, group="logs")
    def fetch_logs(self):
        r = self.backend.request("/api/logs?lines=400", timeout=10)
        self._log_lines = r.get("lines", [])
        self.call_from_thread(self.render_logs)

    def render_logs(self):
        lines = getattr(self, "_log_lines", [])
        rank = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        level = self.query_one("#log-level", Select).value
        min_rank = -1 if level == "ALL" else rank.get(level, 0)
        q = self.query_one("#log-filter", Input).value.lower()
        box = self.query_one("#logbox", RichLog)
        box.clear()
        shown = 0
        for line in lines:
            m = re.search(r" - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - ", line)
            lvl = m.group(1) if m else "INFO"
            if rank[lvl] < min_rank or (q and q not in line.lower()):
                continue
            t = Text(line, style=LEVEL_STYLE.get(lvl, ""))
            if q:
                t.highlight_words([q], style="reverse", case_sensitive=False)
            box.write(t)
            shown += 1
        if not shown:
            box.write(Text("No log lines" + (" match." if q or min_rank > 0 else " yet."), style="dim"))


def set_autostart(on_: bool):
    if on_:
        AUTOSTART_OFF.unlink(missing_ok=True)
        print("batmon will open automatically when you SSH in.")
    else:
        AUTOSTART_OFF.touch()
        print("batmon won't open on SSH login any more. Run `batmon` to open it, "
              "or `batmon --autostart on` to turn it back on.")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--autostart"] and len(args) == 2 and args[1] in ("on", "off"):
        set_autostart(args[1] == "on")
        return
    config_path = None
    if args[:1] == ["--config"] and len(args) == 2:
        config_path, args = args[1], []
    if args:
        print(__doc__)
        return
    BatMon(config_path).run()


if __name__ == "__main__":
    main()
