"""
UI Controller for TFT LCD display (ST7789) with integrated rotary encoder (EC11).
Provides menu interface for bat monitoring system configuration.

Hardware: 2.0" TFT LCD Display with Rotary Encoder EC11, IIC/SPI Interface

Improvements over v1:
- Encoder debounce: quadrature state machine + minimum step interval
- On-screen keyboard for email / WiFi text entry
- Graph settings screen (trendline slope and y-intercept, 0.01 resolution)
- Sub-menus for Email Config and WiFi Config
- Cleaner layout with larger hit targets and persistent header clock
"""

from typing import Any, Callable, Optional, List, Tuple

try:
    import digitalio as _digitalio  # type: ignore[import-not-found]
    import board as _board          # type: ignore[import-not-found]
    from adafruit_rgb_display import st7789 as _st7789  # type: ignore[import-not-found]
except ImportError:
    _digitalio = None
    _board = None
    _st7789 = None

digitalio: Any = _digitalio
board: Any     = _board
st7789: Any    = _st7789

from PIL import Image, ImageDraw, ImageFont

try:
    import RPi.GPIO as _GPIO  # type: ignore[import-not-found]
except ImportError:
    _GPIO = None

GPIO: Any = _GPIO

import time
import threading
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Menu states
# ---------------------------------------------------------------------------

class MenuState(Enum):
    MAIN_MENU      = "main"
    EDIT_TIME      = "edit_time"
    KEYBOARD       = "keyboard"
    GRAPH_SETTINGS = "graph_settings"
    SYSTEM_STATUS  = "status"
    TEST_EMAIL     = "test_email"


# ---------------------------------------------------------------------------
# On-screen keyboard rows
# ---------------------------------------------------------------------------

KB_ROWS: List[List[str]] = [
    list("1234567890"),
    list("qwertyuiop"),
    list("asdfghjkl@"),
    list("zxcvbnm._-"),
    ["CAPS", "SPACE", "DEL", "OK"],
]

# Special-character layout for WiFi password (symbols row replaces digits)
KB_ROWS_SPECIAL: List[List[str]] = [
    list("!@#$%^&*()"),
    list("qwertyuiop"),
    list("asdfghjkl@"),
    list("zxcvbnm._-"),
    ["SYM", "SPACE", "DEL", "OK"],
]


# ---------------------------------------------------------------------------
# UI Controller
# ---------------------------------------------------------------------------

class UIController:
    """
    UI Controller for TFT LCD (ST7789) and rotary encoder (EC11).

    TFT wiring:  VCC→3.3V  GND→GND  SCK→GPIO11  SDA→GPIO10
                 RES→GPIO25  DC→GPIO24  CS→GPIO8  BLK→GPIO23
    Encoder:     CLK→GPIO17  DT→GPIO27  SW→GPIO22
    """

    # GPIO
    PIN_CLK       = 17
    PIN_DT        = 27
    PIN_SW        = 22
    PIN_BACKLIGHT = 23

    # Display (ST7789 2.0" 240x320, landscape via ROTATION=90 → 320x240)
    DISPLAY_WIDTH  = 240
    DISPLAY_HEIGHT = 320
    ROTATION       = 90

    # Font sizes
    FONT_SMALL  = 12
    FONT_MEDIUM = 16
    FONT_LARGE  = 22
    FONT_XLARGE = 30

    # Colours
    COLOR_BLACK     = (0,   0,   0)
    COLOR_WHITE     = (255, 255, 255)
    COLOR_BLUE      = (30,  100, 220)
    COLOR_BLUE_DIM  = (15,  50,  110)
    COLOR_GREEN     = (50,  200, 80)
    COLOR_RED       = (220, 50,  50)
    COLOR_YELLOW    = (255, 210, 0)
    COLOR_CYAN      = (0,   200, 220)
    COLOR_GRAY      = (140, 140, 140)
    COLOR_DARK_GRAY = (50,  50,  50)
    COLOR_HEADER    = (20,  20,  60)
    COLOR_PURPLE    = (60,  0,   100)

    # Quadrature pulse count required per action (4 = one detent ≈ 2 physical clicks per step)

    # Full Gray-code quadrature table: index = (last_ab<<2)|curr_ab → delta
    _QEM = [
         0, -1,  1,  0,
         1,  0,  0, -1,
        -1,  0,  0,  1,
         0,  1, -1,  0,
    ]

    def __init__(self, config_manager):
        self.config = config_manager

        self.display = self._init_display()
        self._init_backlight()
        self._init_encoder()
        self._load_fonts()

        # ── state ────────────────────────────────────────────────────────
        self.current_state   = MenuState.MAIN_MENU
        self.menu_items: List[str] = []
        self.selected_index  = 0
        self._in_submenu     = ""   # "email" | "wifi" | ""

        # Time editor
        self.editing_value: List[str] = []
        self.edit_position   = 0
        self.edit_type       = ""

        # Keyboard
        self.kb_text         = ""
        self.kb_field_label  = ""
        self.kb_field_key    = ""
        self.kb_row          = 1
        self.kb_col          = 0
        self.kb_caps         = False
        self.kb_special_mode = False   # True = symbols layout (WiFi password)
        self._kb_callback: Optional[Callable[[str], None]] = None

        # Recipient editing
        self._recip_edit_index: int = -1   # index of recipient being edited (-1 = none)

        # Graph settings
        self.graph_selected  = 0
        self._graph_items    = [
            "Trendline Slope",
            "Trendline Y-Intercept",
            "Back",
        ]

        # Email test result (None = pending, "" = success, else error string)
        self._test_email_result: Optional[str] = None

        # Public callbacks
        self.on_start_monitoring: Optional[Callable] = None
        self.on_test_email:       Optional[Callable] = None

        self.display_lock     = threading.Lock()
        self.last_interaction = time.time()

        logger.info("TFT UI Controller initialized")
        self.show_main_menu()

    # -----------------------------------------------------------------------
    # Hardware init
    # -----------------------------------------------------------------------

    def _init_display(self):
        if None in (digitalio, board, st7789):
            raise RuntimeError("adafruit-circuitpython-rgb-display not installed")
        cs  = digitalio.DigitalInOut(board.CE0)
        dc  = digitalio.DigitalInOut(board.D24)
        rst = digitalio.DigitalInOut(board.D25)
        disp = st7789.ST7789(
            board.SPI(), cs=cs, dc=dc, rst=rst,
            width=self.DISPLAY_WIDTH, height=self.DISPLAY_HEIGHT,
            rotation=self.ROTATION, baudrate=64_000_000,
        )
        logger.info(f"TFT display initialized ({self.DISPLAY_WIDTH}x{self.DISPLAY_HEIGHT})")
        return disp

    def _init_backlight(self):
        if GPIO is None:
            raise RuntimeError("RPi.GPIO not installed")
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.PIN_BACKLIGHT, GPIO.OUT)
        GPIO.output(self.PIN_BACKLIGHT, GPIO.HIGH)
        logger.info("Backlight initialized")

    def _init_encoder(self):
        if GPIO is None:
            raise RuntimeError("RPi.GPIO not installed")
        GPIO.setmode(GPIO.BCM)
        for pin in (self.PIN_CLK, self.PIN_DT, self.PIN_SW):
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self._enc_ab        = (GPIO.input(self.PIN_CLK) << 1) | GPIO.input(self.PIN_DT)
        self._enc_count     = 0
        self._enc_last_time = 0.0
        self._sw_last       = GPIO.input(self.PIN_SW)
        self._sw_last_time  = 0.0
        self._polling       = True

        self._poll_thread = threading.Thread(target=self._poll_encoder, daemon=True)
        self._poll_thread.start()
        logger.info("Rotary encoder (EC11) initialized (quadrature polling)")

    def _load_fonts(self):
        try:
            ttf  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ttfb = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            self.font_small  = ImageFont.truetype(ttf,  self.FONT_SMALL)
            self.font_medium = ImageFont.truetype(ttf,  self.FONT_MEDIUM)
            self.font_large  = ImageFont.truetype(ttfb, self.FONT_LARGE)
            self.font_xlarge = ImageFont.truetype(ttfb, self.FONT_XLARGE)
        except Exception:
            logger.warning("TrueType fonts not found, using default")
            f = ImageFont.load_default()
            self.font_small = self.font_medium = self.font_large = self.font_xlarge = f

    # -----------------------------------------------------------------------
    # Encoder polling – quadrature + debounce
    # -----------------------------------------------------------------------

    def _poll_encoder(self):
        while self._polling:
            try:
                clk = GPIO.input(self.PIN_CLK)
                dt  = GPIO.input(self.PIN_DT)
                sw  = GPIO.input(self.PIN_SW)

                ab    = (clk << 1) | dt
                delta = self._QEM[(self._enc_ab << 2) | ab]
                self._enc_ab = ab

                if delta:
                    self._enc_count += delta
                    if abs(self._enc_count) >= 4:
                        direction = 1 if self._enc_count > 0 else -1
                        self._enc_count     = 0
                        self._enc_last_time = time.time()
                        self.last_interaction = self._enc_last_time
                        if direction > 0:
                            self._on_rotate_cw()
                        else:
                            self._on_rotate_ccw()

                if sw == 0 and self._sw_last == 1:
                    now = time.time()
                    if now - self._sw_last_time > 0.25:
                        self._sw_last_time    = now
                        self.last_interaction = now
                        self._on_button()
                self._sw_last = sw

            except Exception as e:
                logger.warning(f"Encoder poll error: {e}")

            time.sleep(0.002)   # 500 Hz poll

    # -----------------------------------------------------------------------
    # Input dispatch
    # -----------------------------------------------------------------------

    def _on_rotate_cw(self):
        s = self.current_state
        if s == MenuState.MAIN_MENU:
            self.selected_index = (self.selected_index + 1) % max(1, len(self.menu_items))
            self._refresh_display()
        elif s == MenuState.EDIT_TIME:
            self._time_adjust(+1)
        elif s == MenuState.KEYBOARD:
            self._kb_navigate(+1)
        elif s == MenuState.GRAPH_SETTINGS:
            if self.graph_selected < len(self._graph_items) - 1:
                self._graph_adjust(+0.01)
            else:
                self.graph_selected = (self.graph_selected + 1) % len(self._graph_items)
                self._refresh_display()

    def _on_rotate_ccw(self):
        s = self.current_state
        if s == MenuState.MAIN_MENU:
            self.selected_index = (self.selected_index - 1) % max(1, len(self.menu_items))
            self._refresh_display()
        elif s == MenuState.EDIT_TIME:
            self._time_adjust(-1)
        elif s == MenuState.KEYBOARD:
            self._kb_navigate(-1)
        elif s == MenuState.GRAPH_SETTINGS:
            if self.graph_selected < len(self._graph_items) - 1:
                self._graph_adjust(-0.01)
            else:
                self.graph_selected = (self.graph_selected - 1) % len(self._graph_items)
                self._refresh_display()

    def _on_button(self):
        s = self.current_state
        if s == MenuState.MAIN_MENU:
            self._menu_select()
        elif s == MenuState.EDIT_TIME:
            self._time_confirm()
        elif s == MenuState.KEYBOARD:
            self._kb_press()
        elif s == MenuState.GRAPH_SETTINGS:
            self._graph_select()
        elif s in (MenuState.SYSTEM_STATUS, MenuState.TEST_EMAIL):
            self._test_email_result = None
            self.show_main_menu()

    # -----------------------------------------------------------------------
    # Main menu
    # -----------------------------------------------------------------------

    def show_main_menu(self):
        self.current_state     = MenuState.MAIN_MENU
        self._in_submenu       = ""
        self._recip_edit_index = -1
        self.menu_items     = [
            "Set Current Time",
            "Set Start Time",
            "Set Stop Time",
            "Email Config",
            "WiFi Config",
            "Graph Settings",
            "System Status",
            "Test Email",
            "Start Monitor",
        ]
        self.selected_index = 0
        self._refresh_display()

    def _menu_select(self):
        if self._in_submenu == "email":
            if self._recip_edit_index >= 0:
                self._recip_action_select()
            else:
                self._email_submenu_select()
            return
        if self._in_submenu == "wifi":
            self._wifi_submenu_select()
            return

        item = self.menu_items[self.selected_index]
        actions = {
            "Set Current Time": lambda: self._begin_edit_time("current_time"),
            "Set Start Time":   lambda: self._begin_edit_time("start_time"),
            "Set Stop Time":    lambda: self._begin_edit_time("stop_time"),
            "Email Config":     self._open_email_submenu,
            "WiFi Config":      self._open_wifi_submenu,
            "Graph Settings":   self._show_graph_settings,
            "System Status":    self._show_status,
            "Test Email":       self._start_email_test,
            "Start Monitor":    self._do_start_monitor,
        }
        action = actions.get(item)
        if action:
            action()

    def _do_start_monitor(self):
        if self.on_start_monitoring:
            self.on_start_monitoring()

    # -----------------------------------------------------------------------
    # Time editor
    # -----------------------------------------------------------------------

    def _begin_edit_time(self, time_type: str):
        self.edit_type = time_type
        if time_type == "current_time":
            try:
                from rtc_scheduler import DS3231
                rtc = DS3231()
                raw = rtc.get_datetime().strftime("%H%M")
                rtc.close()
            except Exception:
                raw = "1200"
        else:
            raw = self.config.get_schedule().get(time_type, "00:00").replace(":", "")
            if len(raw) != 4:
                raw = "0000"
        self.editing_value = list(raw)
        self.edit_position = 0
        self.current_state = MenuState.EDIT_TIME
        self._refresh_display()

    def _time_adjust(self, direction: int):
        pos = self.edit_position
        digit = int(self.editing_value[pos])
        if pos == 0:
            max_val = 2
        elif pos == 1:
            max_val = 9 if int(self.editing_value[0]) < 2 else 3
        elif pos == 2:
            max_val = 5
        else:
            max_val = 9
        self.editing_value[pos] = str((digit + direction) % (max_val + 1))
        self._refresh_display()

    def _time_confirm(self):
        if self.edit_position < 3:
            self.edit_position += 1
            self._refresh_display()
            return
        hhmm     = "".join(self.editing_value)
        time_str = f"{hhmm[:2]}:{hhmm[2:]}"
        if self.edit_type == "start_time":
            self.config.set_schedule(start_time=time_str)
            self.show_message(f"Start: {time_str}", duration=1.5)
        elif self.edit_type == "stop_time":
            self.config.set_schedule(stop_time=time_str)
            self.show_message(f"Stop: {time_str}", duration=1.5)
        elif self.edit_type == "current_time":
            try:
                from rtc_scheduler import DS3231
                rtc = DS3231()
                cur = rtc.get_datetime()
                rtc.set_datetime(cur.replace(hour=int(hhmm[:2]),
                                             minute=int(hhmm[2:]), second=0))
                rtc.close()
                self.show_message(f"Time set: {time_str}", duration=1.5)
            except Exception as e:
                self.show_message(f"RTC error: {str(e)[:30]}", duration=2.0)
        time.sleep(1.5)
        self.show_main_menu()

    # -----------------------------------------------------------------------
    # On-screen keyboard
    # -----------------------------------------------------------------------

    def _open_keyboard(self, label: str, field_key: str,
                       initial: str = "",
                       callback: Optional[Callable[[str], None]] = None,
                       special_chars: bool = False):
        self.kb_text         = initial
        self.kb_field_label  = label
        self.kb_field_key    = field_key
        self.kb_row          = 1
        self.kb_col          = 0
        self.kb_caps         = False
        self.kb_special_mode = False   # always start on normal layout
        self._kb_special_chars = special_chars  # whether SYM toggle is available
        self._kb_callback    = callback
        self.current_state   = MenuState.KEYBOARD
        self._refresh_display()

    def _kb_navigate(self, direction: int):
        """Move cursor left/right through the keyboard, wrapping rows."""
        active_rows = KB_ROWS_SPECIAL if self.kb_special_mode else KB_ROWS
        row = self.kb_row
        col = self.kb_col + direction
        if col < 0:
            row = (row - 1) % len(active_rows)
            col = len(active_rows[row]) - 1
        elif col >= len(active_rows[row]):
            row = (row + 1) % len(active_rows)
            col = 0
        self.kb_row = row
        self.kb_col = col
        self._refresh_display()

    def _kb_press(self):
        active_rows = KB_ROWS_SPECIAL if self.kb_special_mode else KB_ROWS
        key = active_rows[self.kb_row][self.kb_col]
        if key == "DEL":
            self.kb_text = self.kb_text[:-1]
        elif key == "SPACE":
            self.kb_text += " "
        elif key == "CAPS":
            if getattr(self, '_kb_special_chars', False):
                # CAPS becomes the SYM-layout toggle when special chars are available
                self.kb_special_mode = True
                self.kb_row = len(KB_ROWS_SPECIAL) - 1
                self.kb_col = 0  # land on SYM key
            else:
                self.kb_caps = not self.kb_caps
        elif key == "SYM":
            # Toggle back to normal layout; cursor stays on fn row
            self.kb_special_mode = False
            self.kb_row = len(KB_ROWS) - 1
            self.kb_col = 0  # land on CAPS
        elif key == "OK":
            self._kb_save()
            return
        else:
            self.kb_text += key.upper() if self.kb_caps else key
        self._refresh_display()

    def _kb_save(self):
        text = self.kb_text.strip()
        if self.kb_field_key:
            self.config.set(self.kb_field_key, text)
        if self._kb_callback:
            self._kb_callback(text)
        self.show_message("Saved!", duration=1.0)
        time.sleep(1.0)
        # Return to whichever submenu launched the keyboard
        if self._in_submenu == "email":
            self._open_email_submenu()
        elif self._in_submenu == "wifi":
            self._open_wifi_submenu()
        else:
            self.show_main_menu()

    # -----------------------------------------------------------------------
    # Email sub-menu
    # -----------------------------------------------------------------------

    def _open_email_submenu(self):
        self.current_state = MenuState.MAIN_MENU
        self._in_submenu   = "email"
        self._recip_edit_index = -1
        email_cfg  = self.config.get_email_config()
        sender     = (email_cfg.get("sender_email") or "(not set)")[:24]
        has_pw     = bool(email_cfg.get("sender_password"))
        recips     = email_cfg.get("recipients", [])
        self.menu_items = [
            f"Sender:  {sender}",
            f"Password: {'(set)' if has_pw else '(not set)'}",
            "── Recipients ──",
        ]
        for addr in recips:
            self.menu_items.append(f"  {addr[:28]}")
        self.menu_items += ["Add recipient", "Back"]
        self.selected_index = 0
        self._refresh_display()

    def _email_submenu_select(self):
        item      = self.menu_items[self.selected_index]
        email_cfg = self.config.get_email_config()
        recips    = list(email_cfg.get("recipients", []))

        if item.startswith("Sender"):
            self._open_keyboard("Sender email", "email.sender_email",
                                initial=email_cfg.get("sender_email", ""))

        elif item.startswith("Password"):
            self._open_keyboard("App password", "email.sender_password",
                                initial=email_cfg.get("sender_password", ""))

        elif item == "── Recipients ──":
            pass  # section header, not selectable

        elif item == "Add recipient":
            def _add(addr: str):
                r = list(self.config.get("email.recipients") or [])
                if addr and addr not in r:
                    r.append(addr)
                    self.config.set("email.recipients", r)
            self._open_keyboard("New recipient email", "", callback=_add)

        elif item == "Back":
            self.show_main_menu()

        else:
            # A recipient address row — open edit/remove sub-submenu
            stripped = item.strip()
            # Find the actual full address (display may be truncated)
            full_addr = next((r for r in recips if r.startswith(stripped) or stripped.startswith(r[:28])), stripped)
            recip_idx = recips.index(full_addr) if full_addr in recips else -1
            self._recip_edit_index = recip_idx
            # Show inline edit/remove choice as a mini menu
            self._open_recip_action_menu(full_addr, recip_idx)

    def _open_recip_action_menu(self, addr: str, idx: int):
        """Show Edit / Remove / Back options for a specific recipient."""
        self.current_state = MenuState.MAIN_MENU
        self._in_submenu   = "email"
        self._recip_edit_index = idx
        self.menu_items = [
            f"Editing: {addr[:26]}",
            "Edit address",
            "Remove",
            "Back to recipients",
        ]
        self.selected_index = 1
        self._refresh_display()

    def _recip_action_select(self):
        """Handle selection inside the recipient action menu."""
        item = self.menu_items[self.selected_index]
        idx  = self._recip_edit_index
        recips = list(self.config.get("email.recipients") or [])

        if item == "Edit address" and 0 <= idx < len(recips):
            old_addr = recips[idx]
            def _update(new_addr: str):
                r = list(self.config.get("email.recipients") or [])
                if new_addr and 0 <= idx < len(r):
                    r[idx] = new_addr
                    self.config.set("email.recipients", r)
            self._open_keyboard("Edit recipient", "", initial=old_addr, callback=_update)

        elif item == "Remove" and 0 <= idx < len(recips):
            recips.pop(idx)
            self.config.set("email.recipients", recips)
            self._recip_edit_index = -1
            self.show_message("Removed!", duration=1.0)
            time.sleep(1.0)
            self._open_email_submenu()

        elif item == "Back to recipients":
            self._recip_edit_index = -1
            self._open_email_submenu()

    # -----------------------------------------------------------------------
    # WiFi sub-menu
    # -----------------------------------------------------------------------

    def _open_wifi_submenu(self):
        self.current_state = MenuState.MAIN_MENU
        self._in_submenu   = "wifi"
        wifi_cfg  = self.config.get_wifi_config()
        ssid      = (wifi_cfg.get("ssid") or "(not set)")[:24]
        has_pw    = bool(wifi_cfg.get("password"))
        self.menu_items = [
            f"SSID:     {ssid}",
            f"Password: {'(set)' if has_pw else '(not set)'}",
            "Back",
        ]
        self.selected_index = 0
        self._refresh_display()

    def _wifi_submenu_select(self):
        item     = self.menu_items[self.selected_index]
        wifi_cfg = self.config.get_wifi_config()
        if item.startswith("SSID"):
            def _after_ssid(ssid: str):
                self._apply_wifi()
            self._open_keyboard("WiFi network name", "wifi.ssid",
                                initial=wifi_cfg.get("ssid", ""),
                                callback=_after_ssid)
        elif item.startswith("Password"):
            def _after_pw(pw: str):
                self._apply_wifi()
            self._open_keyboard("WiFi password", "wifi.password",
                                initial=wifi_cfg.get("password", ""),
                                special_chars=True,
                                callback=_after_pw)
        elif item == "Back":
            self.show_main_menu()

    def _apply_wifi(self):
        """Write WiFi credentials to the system network config and reconnect."""
        try:
            wifi_cfg = self.config.get_wifi_config()
            ssid     = wifi_cfg.get("ssid", "").strip()
            password = wifi_cfg.get("password", "").strip()

            if not ssid:
                return

            # Write to network-config (used by newer Pi OS)
            network_config_path = "/boot/firmware/network-config"
            config_content = f"""version: 2
ethernets:
  eth0:
    dhcp4: true
    optional: true
wifis:
  wlan0:
    dhcp4: true
    optional: true
    access-points:
      "{ssid}":
        password: "{password}"
"""
            import subprocess
            # Write config file (requires root - use sudo tee)
            proc = subprocess.run(
                ["sudo", "tee", network_config_path],
                input=config_content.encode(),
                capture_output=True
            )
            if proc.returncode != 0:
                raise RuntimeError(f"tee failed: {proc.stderr.decode()}")

            # Also write wpa_supplicant.conf for compatibility
            wpa_content = f"""country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={{
    ssid="{ssid}"
    psk="{password}"
}}
"""
            subprocess.run(
                ["sudo", "tee", "/etc/wpa_supplicant/wpa_supplicant.conf"],
                input=wpa_content.encode(),
                capture_output=True
            )

            # Reconnect
            subprocess.run(["sudo", "wpa_cli", "-i", "wlan0", "reconfigure"],
                           capture_output=True)

            logger.info(f"WiFi config applied for SSID: {ssid}")
            self.show_message("WiFi updated!\nReconnecting...", duration=2.0)

        except Exception as e:
            logger.error(f"Failed to apply WiFi config: {e}")
            self.show_message(f"WiFi error: {str(e)[:30]}", duration=3.0)

    # -----------------------------------------------------------------------
    # Graph settings
    # -----------------------------------------------------------------------

    def _show_graph_settings(self):
        self.current_state  = MenuState.GRAPH_SETTINGS
        self.graph_selected = 0
        self._refresh_display()

    def _graph_adjust(self, delta: float):
        """Increment/decrement the currently selected trendline parameter by delta."""
        item = self._graph_items[self.graph_selected]
        if item == "Trendline Slope":
            key = "display.trendline_slope"
            default = 6.4899
        elif item == "Trendline Y-Intercept":
            key = "display.trendline_intercept"
            default = 40.2899
        else:
            return
        current = round(float(self.config.get(key, default)), 2)
        self.config.set(key, round(current + delta, 2))
        self._refresh_display()

    def _graph_select(self):
        """Button press in graph settings: advance to next item or exit on Back."""
        item = self._graph_items[self.graph_selected]
        if item == "Back":
            self.show_main_menu()
        else:
            self.graph_selected = (self.graph_selected + 1) % len(self._graph_items)
            self._refresh_display()

    # -----------------------------------------------------------------------
    # Status / email test
    # -----------------------------------------------------------------------

    def _show_status(self):
        self.current_state = MenuState.SYSTEM_STATUS
        self._refresh_display()

    def _start_email_test(self):
        self._test_email_result = None
        self.current_state      = MenuState.TEST_EMAIL
        self._refresh_display()
        if self.on_test_email:
            threading.Thread(target=self._run_email_test, daemon=True).start()

    def _run_email_test(self):
        time.sleep(0.3)
        result = self.on_test_email() if self.on_test_email else "Not configured"
        self._test_email_result = result or ""
        self._refresh_display()

    # -----------------------------------------------------------------------
    # Display rendering helpers
    # -----------------------------------------------------------------------

    @property
    def _wh(self) -> Tuple[int, int]:
        return (self.DISPLAY_HEIGHT, self.DISPLAY_WIDTH) if self.ROTATION in (90, 270) \
               else (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)

    def _canvas(self):
        w, h = self._wh
        img  = Image.new("RGB", (w, h), self.COLOR_BLACK)
        draw = ImageDraw.Draw(img)
        return img, draw, w, h

    def _header(self, draw, w: int, title: str, color=None):
        color = color or self.COLOR_BLUE
        draw.rectangle((0, 0, w, 33), fill=color)
        draw.text((8, 5), title, font=self.font_large, fill=self.COLOR_WHITE)
        try:
            from rtc_scheduler import DS3231
            rtc = DS3231()
            ts  = rtc.get_datetime().strftime("%H:%M")
            rtc.close()
            bbox = draw.textbbox((0, 0), ts, font=self.font_small)
            draw.text((w - (bbox[2]-bbox[0]) - 6, 10), ts,
                      font=self.font_small, fill=self.COLOR_YELLOW)
        except Exception:
            pass

    def _refresh_display(self):
        with self.display_lock:
            s = self.current_state
            if s == MenuState.MAIN_MENU:
                self._draw_main_menu()
            elif s == MenuState.EDIT_TIME:
                self._draw_time_editor()
            elif s == MenuState.KEYBOARD:
                self._draw_keyboard()
            elif s == MenuState.GRAPH_SETTINGS:
                self._draw_graph_settings()
            elif s == MenuState.SYSTEM_STATUS:
                self._draw_status()
            elif s == MenuState.TEST_EMAIL:
                self._draw_test_email()

    # ── Main menu / sub-menus ────────────────────────────────────────────────

    def _draw_main_menu(self):
        img, draw, w, h = self._canvas()
        if self._in_submenu == "email":
            self._header(draw, w, "Email Config", color=(80, 40, 0))
        elif self._in_submenu == "wifi":
            self._header(draw, w, "WiFi Config", color=(0, 60, 80))
        else:
            self._header(draw, w, "BAT MONITOR")

        ITEM_H  = 32
        TOP     = 36
        visible = (h - TOP) // ITEM_H
        start   = max(0, min(self.selected_index - visible // 2,
                             len(self.menu_items) - visible))
        shown   = self.menu_items[start:start + visible]

        for i, label in enumerate(shown):
            idx  = start + i
            y    = TOP + i * ITEM_H
            sel  = idx == self.selected_index
            fill = self.COLOR_BLUE if sel else None
            if sel:
                draw.rectangle((0, y, w - 6, y + ITEM_H - 1), fill=fill)
                draw.text((2, y + 7), "▶", font=self.font_medium, fill=self.COLOR_YELLOW)
            draw.text((18, y + 7), label, font=self.font_medium,
                      fill=self.COLOR_WHITE if sel else self.COLOR_GRAY)

        # Scroll bar
        if len(self.menu_items) > visible:
            track_h = h - TOP
            bar_h   = max(10, track_h * visible // len(self.menu_items))
            bar_top = TOP + (track_h - bar_h) * start // max(1, len(self.menu_items) - visible)
            draw.rectangle((w - 5, TOP, w - 1, h), fill=self.COLOR_DARK_GRAY)
            draw.rectangle((w - 5, bar_top, w - 1, bar_top + bar_h), fill=self.COLOR_BLUE)

        self.display.image(img)

    # ── Time editor ──────────────────────────────────────────────────────────

    def _draw_time_editor(self):
        img, draw, w, h = self._canvas()
        titles = {
            "start_time":   "Set Start Time",
            "stop_time":    "Set Stop Time",
            "current_time": "Set Current Time",
        }
        self._header(draw, w, titles.get(self.edit_type, "Set Time"), color=self.COLOR_BLUE_DIM)

        v    = self.editing_value
        tstr = f"{v[0]}{v[1]}:{v[2]}{v[3]}"
        x, y = 24, 80

        for ci, ch in enumerate(tstr):
            if ch == ":":
                draw.text((x, y), ":", font=self.font_xlarge, fill=self.COLOR_WHITE)
                bbox = draw.textbbox((0, 0), ":", font=self.font_xlarge)
                x   += (bbox[2] - bbox[0]) + 6
                continue
            digit_idx = ci if ci < 2 else ci - 1
            selected  = digit_idx == self.edit_position
            bbox  = draw.textbbox((0, 0), ch, font=self.font_xlarge)
            cw    = bbox[2] - bbox[0]
            ch_h  = bbox[3] - bbox[1]
            if selected:
                draw.rectangle((x - 4, y - 4, x + cw + 4, y + ch_h + 6),
                               fill=self.COLOR_YELLOW)
                draw.text((x, y), ch, font=self.font_xlarge, fill=self.COLOR_BLACK)
            else:
                draw.text((x, y), ch, font=self.font_xlarge, fill=self.COLOR_WHITE)
            x += cw + 14

        draw.text((6, h - 50), "Rotate to change digit",   font=self.font_small, fill=self.COLOR_GRAY)
        draw.text((6, h - 32), "Click to advance / save",  font=self.font_small, fill=self.COLOR_GRAY)
        self.display.image(img)

    # ── On-screen keyboard ───────────────────────────────────────────────────

    def _draw_keyboard(self):
        img, draw, w, h = self._canvas()

        # Header: label + typed text
        draw.rectangle((0, 0, w, 40), fill=self.COLOR_HEADER)
        draw.text((6, 2), self.kb_field_label, font=self.font_small, fill=self.COLOR_GRAY)
        shown = self.kb_text[-26:] if len(self.kb_text) > 26 else self.kb_text
        draw.text((6, 17), shown + "▌", font=self.font_medium, fill=self.COLOR_WHITE)

        # Key grid
        KEY_W  = 27
        KEY_H  = 23
        GAP    = 2
        TOP    = 43

        active_rows = KB_ROWS_SPECIAL if self.kb_special_mode else KB_ROWS

        for ri, row in enumerate(active_rows):
            fn_keys = {"CAPS", "SPACE", "DEL", "OK", "SYM"}
            is_fn_row = any(k in row for k in fn_keys)
            if is_fn_row:
                fn_labels = row
                fn_count  = len(fn_labels)
                fn_kw     = (w - (fn_count - 1) * GAP) // fn_count
                for ci, key in enumerate(fn_labels):
                    kx  = ci * (fn_kw + GAP)
                    ky  = TOP + ri * (KEY_H + GAP)
                    sel = ri == self.kb_row and ci == self.kb_col
                    # Highlight SYM differently when special mode is active
                    is_sym_active = key == "SYM" and self.kb_special_mode
                    bg = self.COLOR_YELLOW if is_sym_active else (self.COLOR_BLUE if sel else self.COLOR_DARK_GRAY)
                    draw.rectangle((kx, ky, kx + fn_kw, ky + KEY_H - 1),
                                   fill=bg,
                                   outline=self.COLOR_YELLOW if sel else self.COLOR_GRAY)
                    bbox = draw.textbbox((0, 0), key, font=self.font_small)
                    tx   = kx + (fn_kw - (bbox[2]-bbox[0])) // 2
                    ty   = ky + (KEY_H - (bbox[3]-bbox[1])) // 2
                    draw.text((tx, ty), key, font=self.font_small,
                              fill=self.COLOR_BLACK if (sel or is_sym_active) else self.COLOR_WHITE)
            else:
                row_w = len(row) * (KEY_W + GAP) - GAP
                ox    = (w - row_w) // 2
                for ci, key in enumerate(row):
                    kx  = ox + ci * (KEY_W + GAP)
                    ky  = TOP + ri * (KEY_H + GAP)
                    sel = ri == self.kb_row and ci == self.kb_col
                    label = key.upper() if self.kb_caps else key
                    draw.rectangle((kx, ky, kx + KEY_W, ky + KEY_H - 1),
                                   fill=self.COLOR_BLUE if sel else self.COLOR_DARK_GRAY,
                                   outline=self.COLOR_YELLOW if sel else self.COLOR_GRAY)
                    bbox = draw.textbbox((0, 0), label, font=self.font_small)
                    tx   = kx + (KEY_W - (bbox[2]-bbox[0])) // 2
                    ty   = ky + (KEY_H - (bbox[3]-bbox[1])) // 2
                    draw.text((tx, ty), label, font=self.font_small,
                              fill=self.COLOR_BLACK if sel else self.COLOR_WHITE)

        # Bottom hint — show SYM tip if special chars available and not already in sym mode
        if getattr(self, '_kb_special_chars', False) and not self.kb_special_mode:
            hint = "Rotate=move  Click=type  CAPS→SYM for symbols"
        else:
            hint = "Rotate=move  Click=type  OK=save"
        draw.text((4, h - 13), hint, font=self.font_small, fill=self.COLOR_GRAY)
        self.display.image(img)

    # ── Graph settings ───────────────────────────────────────────────────────

    def _draw_graph_settings(self):
        img, draw, w, h = self._canvas()
        self._header(draw, w, "Graph Settings", color=self.COLOR_PURPLE)

        slope     = float(self.config.get("display.trendline_slope",       6.4899))
        intercept = float(self.config.get("display.trendline_intercept",   40.2899))
        values    = [f"{slope:+.2f}", f"{intercept:+.2f}", ""]

        ITEM_H = 46
        TOP    = 38
        for i, (label, val) in enumerate(zip(self._graph_items, values)):
            y   = TOP + i * ITEM_H
            sel = i == self.graph_selected
            if sel:
                draw.rectangle((0, y, w - 6, y + ITEM_H - 1), fill=self.COLOR_BLUE)
                draw.text((2, y + 14), "▶", font=self.font_medium, fill=self.COLOR_YELLOW)
            draw.text((18, y + 6), label, font=self.font_medium,
                      fill=self.COLOR_WHITE if sel else self.COLOR_GRAY)
            if val:
                bbox = draw.textbbox((0, 0), val, font=self.font_large)
                draw.text((18, y + 22), val, font=self.font_large,
                          fill=self.COLOR_YELLOW if sel else self.COLOR_CYAN)

        hint = "Rotate=adjust  Click=next item" if self.graph_selected < len(self._graph_items) - 1 \
               else "Rotate=navigate  Click=back"
        draw.text((4, h - 14), hint, font=self.font_small, fill=self.COLOR_GRAY)
        self.display.image(img)

    # ── System status ────────────────────────────────────────────────────────

    def _draw_status(self):
        img, draw, w, h = self._canvas()
        self._header(draw, w, "System Status", color=(0, 80, 60))

        try:
            from rtc_scheduler import DS3231
            rtc  = DS3231()
            now  = rtc.get_datetime()
            temp = rtc.get_temperature()
            rtc.close()
            date_s = now.strftime("%d %b %Y")
            time_s = now.strftime("%H:%M:%S")
        except Exception:
            date_s = time_s = "N/A"
            temp = 0.0

        sched     = self.config.get_schedule()
        email     = self.config.get_email_config()
        recips    = len(email.get("recipients", []))
        slope     = float(self.config.get("display.trendline_slope",     6.4899))
        intercept = float(self.config.get("display.trendline_intercept", 40.2899))

        rows = [
            (f"{date_s}  {time_s}",                          self.COLOR_WHITE),
            (f"RTC temp: {temp:.1f}\u00b0C",                 self.COLOR_CYAN),
            ("",                                              None),
            (f"Start: {sched.get('start_time','?')}   "
             f"Stop: {sched.get('stop_time','?')}",          self.COLOR_GREEN),
            (f"Schedule: {'ON' if sched.get('enabled') else 'OFF'}",
                                                              self.COLOR_YELLOW),
            ("",                                              None),
            (f"Email recipients: {recips}",                  self.COLOR_WHITE),
            (f"Trendline  m={slope:+.2f}  b={intercept:+.2f}", self.COLOR_WHITE),
            ("",                                              None),
            ("Click to return",                              self.COLOR_GRAY),
        ]
        y = 38
        for text, color in rows:
            if not text:
                y += 6
                continue
            draw.text((6, y), text, font=self.font_small, fill=color)
            bbox = draw.textbbox((0, 0), text, font=self.font_small)
            y   += (bbox[3] - bbox[1]) + 6
        self.display.image(img)

    # ── Email test ───────────────────────────────────────────────────────────

    def _draw_test_email(self):
        img, draw, w, h = self._canvas()
        result = self._test_email_result

        if result is None:
            self._header(draw, w, "Test Email", color=(80, 60, 0))
            draw.text((w // 2 - 45, h // 2 - 12), "Sending…",
                      font=self.font_large, fill=self.COLOR_YELLOW)
        elif result == "":
            self._header(draw, w, "Email OK", color=(0, 80, 0))
            draw.text((20, h // 2 - 20), "Sent successfully!",
                      font=self.font_large, fill=self.COLOR_GREEN)
            draw.text((10, h - 20), "Click to return",
                      font=self.font_small, fill=self.COLOR_GRAY)
        else:
            self._header(draw, w, "Email Failed", color=(80, 0, 0))
            words = result.split()
            line, y = "", 40
            for word in words:
                test = (line + " " + word).strip()
                bbox = draw.textbbox((0, 0), test, font=self.font_small)
                if bbox[2] - bbox[0] > w - 12:
                    if line:
                        draw.text((6, y), line, font=self.font_small, fill=self.COLOR_RED)
                        y += 18
                    line = word
                else:
                    line = test
            if line:
                draw.text((6, y), line, font=self.font_small, fill=self.COLOR_RED)
            draw.text((10, h - 20), "Click to return",
                      font=self.font_small, fill=self.COLOR_GRAY)
        self.display.image(img)

    # -----------------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------------

    def show_message(self, message: str, duration: float = 2.0):
        with self.display_lock:
            img, draw, w, h = self._canvas()
            words, lines, line = message.split(), [], ""
            for word in words:
                test = (line + " " + word).strip()
                bbox = draw.textbbox((0, 0), test, font=self.font_medium)
                if bbox[2] - bbox[0] <= w - 20:
                    line = test
                else:
                    if line:
                        lines.append(line)
                    line = word
            if line:
                lines.append(line)
            bbox_h = draw.textbbox((0, 0), "Ag", font=self.font_medium)
            lh     = max(1, bbox_h[3] - bbox_h[1]) + 8
            y0     = (h - len(lines) * lh) // 2
            for i, ln in enumerate(lines):
                bbox = draw.textbbox((0, 0), ln, font=self.font_medium)
                x    = (w - (bbox[2]-bbox[0])) // 2
                draw.text((x, y0 + i * lh), ln, font=self.font_medium, fill=self.COLOR_WHITE)
            self.display.image(img)
        if duration > 0:
            threading.Timer(duration, self._refresh_display).start()

    def cleanup(self):
        self._polling = False
        if hasattr(self, "_poll_thread") and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        if GPIO is None:
            return
        try:
            GPIO.output(self.PIN_BACKLIGHT, GPIO.LOW)
        except Exception:
            pass
        GPIO.cleanup()
        logger.info("TFT UI Controller cleaned up")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def test_ui():
    print("Testing TFT UI Controller…")
    ui = None
    try:
        from config_manager import ConfigManager
        ui = UIController(ConfigManager())
        print("Running – rotate/click encoder, Ctrl-C to exit")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if ui:
            try:
                ui.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    test_ui()
