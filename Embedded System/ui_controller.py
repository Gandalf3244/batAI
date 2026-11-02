"""
UI Controller for TFT LCD display (ST7789) with integrated rotary encoder (EC11).
Provides the on-device menu for the bat monitoring system.

Hardware: 2.0" TFT LCD Display with Rotary Encoder EC11, IIC/SPI Interface

One consistent way to use the knob everywhere:
    turn   = move / change a value (spin faster to change numbers faster)
    click  = select / confirm / next
    hold   = back / cancel (about 0.7 s)

Input handling:
- The encoder type is detected automatically: knobs that give 2 or 4 signal
  changes per click both move exactly one step per click.
- Quick turns are combined into one screen update, so the display keeps up
  and doesn't overshoot.
- Input is read on its own thread and never paused by slow actions (saving,
  WiFi, email); those run in the background.
- The screen turns off after a period without input (keeps light away from
  the bats); any input wakes it without doing anything else.
"""

import calendar
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, List, Optional, Tuple

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    ROTATE_180 = Image.Transpose.ROTATE_180
except AttributeError:          # Pillow < 9.1
    ROTATE_180 = Image.ROTATE_180  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# On-screen keyboard rows
# ---------------------------------------------------------------------------

KB_ROWS: List[List[str]] = [
    list("1234567890"),
    list("qwertyuiop"),
    list("asdfghjkl@"),
    list("zxcvbnm._-"),
    ["CAPS", "SYM", ".com", "SPACE", "DEL", "OK"],
]

# Symbols layout (symbols row replaces digits), toggled with SYM
KB_ROWS_SPECIAL: List[List[str]] = [
    list("!@#$%^&*()"),
    list("qwertyuiop"),
    list("asdfghjkl+"),
    list("zxcvbnm,?="),
    ["CAPS", "SYM", ".com", "SPACE", "DEL", "OK"],
]
FN_KEYS = {"CAPS", "SYM", ".com", "SPACE", "DEL", "OK"}

# Screen-off choices offered in the menu (seconds, 0 = never)
SCREEN_TIMEOUTS = [30, 60, 300, 900, 0]

HINT_MENU   = "turn: move   click: select   hold: back"
HINT_EDIT   = "turn: change   click: next   hold: cancel"
HINT_INFO   = "click or hold: back"


# ---------------------------------------------------------------------------
# Rotary encoder decoding (pure logic, fed with pin samples)
# ---------------------------------------------------------------------------

class EncoderDecoder:
    """
    Turns raw CLK/DT/SW samples into ("turn", ±1), ("click",) and ("long",) events.

    Quadrature: transitions are accumulated with a Gray-code table, so contact
    bounce cancels itself out. A step is reported when the knob reaches a rest
    (detent) position having moved at least 2 transitions in one direction.
    Rest positions are learned: a state that stays stable for a while at least
    twice is a detent. Full-step EC11s only rest at one state (4 transitions per
    click); half-step ones rest at two (2 per click). Either way: one click = one step.
    """

    # index = (previous_ab << 2) | current_ab -> direction
    _QEM = [0, -1, 1, 0,
            1, 0, 0, -1,
            -1, 0, 0, 1,
            0, 1, -1, 0]
    REST_STABLE_S = 0.08      # stable this long = knob is resting
    REST_LEARN_COUNT = 3      # ...this many separate times = it's a detent
    BUTTON_DEBOUNCE_S = 0.015
    LONG_PRESS_S = 0.7

    def __init__(self, clk: int, dt: int, sw: int, t: float, emit: Callable[[tuple], None],
                 half_step: bool = False):
        """half_step: the knob type is already known to rest at two positions (saved earlier)."""
        self.emit = emit
        self.ab = (clk << 1) | dt
        self.ab_since = t
        self.acc = 0
        self.rest = {self.ab}             # the knob is at a detent when the system starts
        if half_step:
            self.rest.add(self.ab ^ 0b11)
        self.stable_seen = {ab: self.REST_LEARN_COUNT for ab in self.rest}
        self._counted_this_rest = True
        self._moves = 0                                   # transitions since the knob last rested
        self._rests: Deque[Tuple[int, int]] = deque(maxlen=6)   # (state, moves) at each rest
        # button (active low)
        self.pressed = sw == 0
        self.btn_change_t: Optional[float] = None
        self.press_t = t
        self.long_sent = self.pressed     # held at startup: ignore until released

    def sample(self, clk: int, dt: int, sw: int, t: float):
        ab = (clk << 1) | dt
        if ab != self.ab:
            delta = self._QEM[(self.ab << 2) | ab]
            self.acc += delta
            self._moves += delta
            self.ab, self.ab_since = ab, t
            self._counted_this_rest = False
            if ab in self.rest:
                self._settle()
        elif t - self.ab_since >= self.REST_STABLE_S and not self._counted_this_rest:
            self._counted_this_rest = True
            if ab in self.rest:
                self._settle(final=True)
                self._check_full_step(ab)
            elif abs(self.acc) >= 2:
                # Resting two transitions away from a known click position: this
                # knob has a click there too (half-step type)
                self.stable_seen[ab] = self.stable_seen.get(ab, 0) + 1
                if self.stable_seen[ab] >= self.REST_LEARN_COUNT:
                    self.rest.add(ab)
                    logger.info("Encoder: half-step type detected (2 signal changes per click)")
                    self.emit(("learned", "half"))
                self._settle(final=True)
                self._moves = 0
            # else: paused part-way through a click - keep counting
        self._button(sw == 0, t)

    def _check_full_step(self, ab: int):
        """In half-step mode, a knob that keeps coming to rest at the same position,
        exactly 4 changes (one full cycle) apart, is really a full-step knob."""
        if abs(self._moves) >= 1:
            self._rests.append((ab, abs(self._moves)))
        self._moves = 0
        if len(self.rest) > 1 and len(self._rests) == self._rests.maxlen and \
                all(s == ab and m == 4 for s, m in self._rests):
            self.rest = {ab}
            self.stable_seen = {ab: self.REST_LEARN_COUNT}
            self._rests.clear()
            logger.info("Encoder: full-step type detected (4 signal changes per click)")
            self.emit(("learned", "full"))

    def _settle(self, final: bool = False):
        if abs(self.acc) >= 2:
            self.emit(("turn", 1 if self.acc > 0 else -1))
            self.acc = 0
        elif final:
            self.acc = 0                  # leftover bounce at rest

    def _button(self, pressed: bool, t: float):
        if pressed != self.pressed:
            if self.btn_change_t is None:
                self.btn_change_t = t
            elif t - self.btn_change_t >= self.BUTTON_DEBOUNCE_S:
                self.pressed, self.btn_change_t = pressed, None
                if pressed:
                    self.press_t, self.long_sent = t, False
                elif not self.long_sent:
                    self.emit(("click",))
        else:
            self.btn_change_t = None
        if self.pressed and not self.long_sent and t - self.press_t >= self.LONG_PRESS_S:
            self.long_sent = True
            self.emit(("long",))


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

class Screen:
    title = ""
    color = (30, 100, 220)
    hint = HINT_MENU
    live = False            # redraw every couple of seconds (live data)

    def on_turn(self, ui: "UIController", steps: int, speed: int): ...
    def on_click(self, ui: "UIController"): ...

    def on_long(self, ui: "UIController"):
        ui.pop()

    def draw(self, ui: "UIController", d: ImageDraw.ImageDraw, w: int, top: int, bottom: int): ...


MenuItem = Tuple[str, Optional[Callable[[], None]]]


class MenuScreen(Screen):
    ROW_H = 30

    def __init__(self, title: str, items: Callable[[], List[MenuItem]], color=None, back: bool = True):
        self.title, self.items_fn = title, items
        self.color = color or Screen.color
        self.back = back
        self.index = 0
        if not back:
            self.hint = "turn: move   click: select"

    def items(self) -> List[MenuItem]:
        items = self.items_fn()
        return items + [("‹ Back", None)] if self.back else items

    def _selectable(self, items: List[MenuItem], i: int) -> bool:
        return items[i][1] is not None or (self.back and i == len(items) - 1)

    def on_turn(self, ui, steps, speed):
        items = self.items()
        n = len(items)
        step = 1 if steps > 0 else -1
        for _ in range(abs(steps)):
            for _ in range(n):                        # skip information lines
                self.index = (self.index + step) % max(1, n)
                if self._selectable(items, self.index):
                    break

    def first_selectable(self):
        items = self.items()
        self.index = next((i for i in range(len(items)) if self._selectable(items, i)), 0)
        return self

    def on_click(self, ui):
        items = self.items()
        self.index = min(self.index, len(items) - 1)
        label, action = items[self.index]
        if action:
            action()
        elif self.back and self.index == len(items) - 1:
            ui.pop()

    def on_long(self, ui):
        if self.back:
            ui.pop()

    def draw(self, ui, d, w, top, bottom):
        items = self.items()
        self.index = min(self.index, len(items) - 1)
        visible = max(1, (bottom - top) // self.ROW_H)
        start = max(0, min(self.index - visible // 2, len(items) - visible))
        for row, (label, action) in enumerate(items[start:start + visible]):
            i = start + row
            y = top + row * self.ROW_H
            sel = i == self.index
            dim = action is None and not (self.back and i == len(items) - 1)
            if sel:
                d.rounded_rectangle((3, y + 1, w - 9, y + self.ROW_H - 2), radius=6, fill=ui.COLOR_BLUE)
            d.text((14, y + 6), label, font=ui.font_medium,
                   fill=ui.COLOR_WHITE if sel else (ui.COLOR_DARK_TEXT if dim else ui.COLOR_GRAY))
        if len(items) > visible:
            track = bottom - top
            bar_h = max(12, track * visible // len(items))
            bar_y = top + (track - bar_h) * start // max(1, len(items) - visible)
            d.rectangle((w - 5, top, w - 2, bottom), fill=ui.COLOR_DARK_GRAY)
            d.rectangle((w - 5, bar_y, w - 2, bar_y + bar_h), fill=ui.COLOR_BLUE)


@dataclass
class Field:
    label: str
    value: float
    lo: float
    hi: float
    step: float = 1
    fmt: str = "{:02d}"
    wrap: bool = True
    accel: Tuple[int, int, int] = (1, 1, 1)    # step multiplier at slow / medium / fast turning

    def text(self) -> str:
        v = int(round(self.value)) if self.fmt.endswith("d}") else self.value
        return self.fmt.format(v)


class NumberEditor(Screen):
    """Edit one or more numbers: turn changes the highlighted one, click moves on, the last click saves."""
    hint = HINT_EDIT

    def __init__(self, title: str, fields: List[Field], on_save: Callable[[List[float]], Optional[str]],
                 separators: Optional[List[str]] = None, color=(15, 50, 110), done_message: str = "Saved"):
        self.title, self.fields, self.on_save = title, fields, on_save
        self.separators = separators or [""] * (len(fields) - 1)
        self.color, self.done_message = color, done_message
        self.active = 0

    def on_turn(self, ui, steps, speed):
        f = self.fields[self.active]
        v = f.value + steps * f.step * f.accel[speed]
        if f.wrap:
            span = f.hi - f.lo + f.step
            v = f.lo + ((v - f.lo) % span)
        f.value = round(min(f.hi, max(f.lo, v)), 6)

    def on_click(self, ui):
        if self.active < len(self.fields) - 1:
            self.active += 1
            return
        error = self.on_save([f.value for f in self.fields])
        ui.pop()
        ui.toast(error or self.done_message, error=bool(error))

    def on_long(self, ui):
        ui.pop()
        ui.toast("Cancelled - nothing changed")

    def draw(self, ui, d, w, top, bottom):
        f = self.fields[self.active]
        step_note = f"  ({len(self.fields) - self.active} to go)" if len(self.fields) > 1 else ""
        d.text((10, top + 4), f"{f.label}{step_note}", font=ui.font_medium, fill=ui.COLOR_CYAN)
        font = ui.font_xlarge if len(self.fields) <= 3 else ui.font_large
        parts = []
        for i, fld in enumerate(self.fields):
            parts.append((fld.text(), i))
            if i < len(self.separators):
                parts.append((self.separators[i], None))
        widths = [d.textbbox((0, 0), t or " ", font=font)[2] + (14 if i is not None else 4) for t, i in parts]
        x = max(6, (w - sum(widths)) // 2)
        y = top + (bottom - top) // 2 - 16
        for (t, i), wd in zip(parts, widths):
            if i is None:
                d.text((x, y), t, font=font, fill=ui.COLOR_GRAY)
            else:
                active = i == self.active
                box = (x, y - 6, x + wd - 4, y + font.size + 10)
                if active:
                    d.rounded_rectangle(box, radius=6, fill=ui.COLOR_YELLOW)
                d.text((x + 5, y), t, font=font, fill=ui.COLOR_BLACK if active else ui.COLOR_WHITE)
            x += wd
        if any(fl.accel[2] > 1 for fl in self.fields):
            d.text((10, bottom - 20), "spin faster for bigger steps", font=ui.font_small, fill=ui.COLOR_GRAY)


class KeyboardScreen(Screen):
    """On-screen keyboard. Turn moves the cursor (faster when spun), click types, OK saves."""
    hint = "turn: move   click: type   hold: cancel"

    def __init__(self, label: str, initial: str, on_save: Callable[[str], Optional[str]],
                 warn_spaces: bool = False, saved_message: Optional[str] = "Saved"):
        """warn_spaces: ask before saving text that starts/ends with a space (passwords).
        saved_message: pop-up after saving (None when the next screen reports the outcome)."""
        self.title, self.label, self.text, self.on_save = "Enter text", label, initial, on_save
        self.warn_spaces, self.saved_message = warn_spaces, saved_message
        self.row, self.col = 1, 0
        self.caps = 0          # 0 = off, 1 = next letter only (like a phone), 2 = caps lock
        self.symbols = False
        self.color = (20, 20, 60)

    def rows(self):
        return KB_ROWS_SPECIAL if self.symbols else KB_ROWS

    def on_turn(self, ui, steps, speed):
        rows = self.rows()
        flat = [(r, c) for r, row in enumerate(rows) for c in range(len(row))]
        pos = flat.index((self.row, min(self.col, len(rows[self.row]) - 1)))
        # Only a genuinely fast spin skips keys: aiming at a key matters more than speed
        pos = (pos + steps * (1, 1, 3)[speed]) % len(flat)
        self.row, self.col = flat[pos]

    def on_click(self, ui):
        key = self.rows()[self.row][self.col]
        if key == "DEL":
            self.text = self.text[:-1]
        elif key == "SPACE":
            self.text += " "
        elif key == "CAPS":
            self.caps = (self.caps + 1) % 3
        elif key == "SYM":
            self.symbols = not self.symbols
        elif key == "OK":
            if self.warn_spaces and self.text != self.text.strip():
                ui.push(ConfirmScreen("Space at the edge", f'"{self.text}" starts or ends with a space. '
                                      "Save it like that?", "Save", lambda: self._save(ui)))
                return
            self._save(ui)
        else:
            if self.caps and key.isalpha():
                key = key.upper()
                if self.caps == 1:
                    self.caps = 0          # one capital, then back to lowercase
            self.text += key

    def _save(self, ui):
        error = self.on_save(self.text)
        if error:
            ui.toast(error, error=True)
            return
        ui.pop()
        if self.saved_message:
            ui.toast(self.saved_message)

    def on_long(self, ui):
        ui.pop()
        ui.toast("Cancelled - nothing changed")

    def _draw_text(self, ui, d, w, top):
        """The whole text, wrapped onto a second line when long (only the start is cut, with …)."""
        box_w = w - 18
        text = self.text + "▌"
        if d.textbbox((0, 0), text, font=ui.font_medium)[2] <= box_w:
            d.text((9, top + 6), text, font=ui.font_medium, fill=ui.COLOR_WHITE)
            return
        lines, line = [], ""
        for ch in text:
            if d.textbbox((0, 0), line + ch, font=ui.font_small)[2] > box_w:
                lines.append(line)
                line = ""
            line += ch
        lines.append(line)
        if len(lines) > 2:
            lines = lines[-2:]
            lines[0] = "…" + lines[0][1:]
        for i, ln in enumerate(lines):
            d.text((9, top + 2 + i * 15), ln, font=ui.font_small, fill=ui.COLOR_WHITE)

    def draw(self, ui, d, w, top, bottom):
        d.text((6, top - 2), self.label, font=ui.font_small, fill=ui.COLOR_GRAY)
        d.rounded_rectangle((4, top + 13, w - 4, top + 45), radius=4, fill=ui.COLOR_DARK_GRAY)
        self._draw_text(ui, d, w, top + 13)
        key_h, gap, y0 = 24, 2, top + 49
        for r, row in enumerate(self.rows()):
            fn = row[0] in FN_KEYS
            kw = (w - 8 - gap * (len(row) - 1)) // len(row) if fn else 28
            x0 = 4 if fn else (w - (len(row) * (kw + gap) - gap)) // 2
            for c, key in enumerate(row):
                x, y = x0 + c * (kw + gap), y0 + r * (key_h + gap)
                sel = r == self.row and c == self.col
                on = (key == "CAPS" and self.caps) or (key == "SYM" and self.symbols)
                label = key.upper() if self.caps and len(key) == 1 and key.isalpha() else key
                if key == "CAPS":
                    label = ("caps", "Caps", "CAPS")[self.caps]
                fill = ui.COLOR_YELLOW if sel else (
                    (ui.COLOR_BLUE if self.caps == 2 and key == "CAPS" else ui.COLOR_BLUE_DIM) if on
                    else ui.COLOR_DARK_GRAY)
                d.rounded_rectangle((x, y, x + kw, y + key_h - 1), radius=4, fill=fill)
                bb = d.textbbox((0, 0), label, font=ui.font_small)
                d.text((x + (kw - (bb[2] - bb[0])) // 2, y + (key_h - (bb[3] - bb[1])) // 2 - 2), label,
                       font=ui.font_small, fill=ui.COLOR_BLACK if sel else ui.COLOR_WHITE)


class ConfirmScreen(Screen):
    hint = "turn: choose   click: confirm   hold: cancel"

    def __init__(self, question: str, detail: str, yes: str, on_yes: Callable[[], None]):
        self.title, self.detail, self.yes, self.on_yes = question, detail, yes, on_yes
        self.choice = 0   # 0 = Cancel (safe default)
        self.color = (100, 60, 0)

    def on_turn(self, ui, steps, speed):
        if steps % 2:
            self.choice = 1 - self.choice

    def on_click(self, ui):
        ui.pop()
        if self.choice == 1:
            self.on_yes()

    def draw(self, ui, d, w, top, bottom):
        ui.wrap_text(d, self.detail, (10, top + 6), w - 20, ui.font_medium, ui.COLOR_WHITE)
        bw, y = (w - 30) // 2, bottom - 52
        for i, label in enumerate(("Cancel", self.yes)):
            x = 10 + i * (bw + 10)
            sel = i == self.choice
            d.rounded_rectangle((x, y, x + bw, y + 40), radius=8,
                                fill=(ui.COLOR_RED if i else ui.COLOR_BLUE) if sel else ui.COLOR_DARK_GRAY)
            bb = d.textbbox((0, 0), label, font=ui.font_medium)
            d.text((x + (bw - bb[2]) // 2, y + 10), label, font=ui.font_medium, fill=ui.COLOR_WHITE)


class InfoScreen(Screen):
    hint = HINT_INFO
    live = True

    def __init__(self, title: str, rows: Callable[[], List[Tuple[str, tuple]]], color=(0, 80, 60)):
        self.title, self.rows_fn, self.color = title, rows, color

    def on_click(self, ui):
        ui.pop()

    def draw(self, ui, d, w, top, bottom):
        y = top + 4
        for text, color in self.rows_fn():
            if not text:
                y += 6
                continue
            d.text((8, y), text, font=ui.font_medium, fill=color)
            y += 21


class TaskScreen(Screen):
    """Runs a slow action in the background and shows its result."""
    hint = "please wait…"

    def __init__(self, ui: "UIController", title: str, busy: str, work: Callable[[], Optional[str]],
                 ok: str, color=(80, 60, 0)):
        self.title, self.busy, self.ok, self.color = title, busy, ok, color
        self.result: Optional[str] = None
        self.done = False

        def run():
            try:
                self.result = work() or ""
            except Exception as e:
                self.result = str(e)
            self.done = True
            self.hint = HINT_INFO
            ui.request_redraw()
        threading.Thread(target=run, daemon=True).start()

    def on_click(self, ui):
        if self.done:
            ui.pop()

    def on_long(self, ui):
        if self.done:
            ui.pop()

    def draw(self, ui, d, w, top, bottom):
        mid = top + (bottom - top) // 2 - 30
        if not self.done:
            dots = "." * (int(time.monotonic() * 2) % 4)
            d.text((14, mid), self.busy + dots, font=ui.font_large, fill=ui.COLOR_YELLOW)
        elif not self.result:
            ui.wrap_text(d, "✓ " + self.ok, (14, mid - 10), w - 28, ui.font_large, ui.COLOR_GREEN, max_lines=3)
        else:
            d.text((10, top + 6), "Failed:", font=ui.font_large, fill=ui.COLOR_RED)
            ui.wrap_text(d, self.result, (10, top + 38), w - 20, ui.font_small, ui.COLOR_WHITE, max_lines=8)

    @property
    def live(self):  # type: ignore[override]
        return not self.done


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

    # Display (ST7789 2.0" 240x320, landscape via ROTATION=90 → 320x240).
    # The picture is additionally turned 180° in software when the config
    # setting display.rotate_180 is on (the default: the module is mounted
    # that way round).
    DISPLAY_WIDTH  = 240
    DISPLAY_HEIGHT = 320
    ROTATION       = 90
    W, H           = 320, 240
    HEADER_H, FOOTER_H = 32, 18

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
    COLOR_DARK_TEXT = (90,  90,  90)
    COLOR_PURPLE    = (60,  0,   100)

    POLL_S = 0.001            # encoder sampling period while awake
    POLL_ASLEEP_S = 0.005     # slower while the screen is off (still catches the wake-up)
    LIVE_REFRESH_S = 2.0
    IDLE_REFRESH_S = 15.0     # header clock

    def __init__(self, config_manager, rtc=None):
        self.config = config_manager
        self.rtc    = rtc   # shared DS3231 (None = use system clock)

        # Public callbacks (set by main.py)
        self.on_start_monitoring: Optional[Callable] = None
        self.on_stop_monitoring:  Optional[Callable] = None
        self.is_monitoring:       Optional[Callable[[], bool]] = None
        self.on_test_email:       Optional[Callable] = None
        self.on_settings_changed: Optional[Callable] = None

        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._stack: List[Screen] = []
        self._toast: Optional[Tuple[str, float, bool]] = None
        self._turn_times: Deque[float] = deque(maxlen=32)
        self._awake = True
        self._last_input = time.monotonic()
        self._last_draw = 0.0
        self._running = True
        self.display_lock = threading.Lock()

        self.display = self._init_display()
        self._init_backlight()
        self._load_fonts()
        self._push_main_menu()
        self._init_encoder()

        self._ui_thread = threading.Thread(target=self._ui_loop, daemon=True, name="tft-ui")
        self._ui_thread.start()
        logger.info("TFT UI Controller initialized")

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
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.PIN_BACKLIGHT, GPIO.OUT)
        GPIO.output(self.PIN_BACKLIGHT, GPIO.HIGH)

    def _init_encoder(self):
        GPIO.setmode(GPIO.BCM)
        for pin in (self.PIN_CLK, self.PIN_DT, self.PIN_SW):
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self._decoder = EncoderDecoder(GPIO.input(self.PIN_CLK), GPIO.input(self.PIN_DT),
                                       GPIO.input(self.PIN_SW), time.monotonic(), self._events.put,
                                       half_step=bool(self.config.get("display.encoder_half_step", False)))
        self._poll_thread = threading.Thread(target=self._poll_encoder, daemon=True, name="encoder")
        self._poll_thread.start()
        logger.info("Rotary encoder (EC11) initialized")

    def _poll_encoder(self):
        errors = 0
        while self._running:
            try:
                self._decoder.sample(GPIO.input(self.PIN_CLK), GPIO.input(self.PIN_DT),
                                     GPIO.input(self.PIN_SW), time.monotonic())
                errors = 0
            except Exception as e:
                errors += 1
                if errors in (1, 100):
                    logger.warning(f"Encoder read error: {e}")
            time.sleep(self.POLL_S if self._awake else self.POLL_ASLEEP_S)

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
            try:
                f = {s: ImageFont.load_default(size=s) for s in
                     (self.FONT_SMALL, self.FONT_MEDIUM, self.FONT_LARGE, self.FONT_XLARGE)}
            except TypeError:   # Pillow < 10.1
                f = {s: ImageFont.load_default() for s in
                     (self.FONT_SMALL, self.FONT_MEDIUM, self.FONT_LARGE, self.FONT_XLARGE)}
            self.font_small, self.font_medium = f[self.FONT_SMALL], f[self.FONT_MEDIUM]
            self.font_large, self.font_xlarge = f[self.FONT_LARGE], f[self.FONT_XLARGE]
        for font, size in ((self.font_small, self.FONT_SMALL), (self.font_medium, self.FONT_MEDIUM),
                           (self.font_large, self.FONT_LARGE), (self.font_xlarge, self.FONT_XLARGE)):
            if not hasattr(font, "size"):
                font.size = size  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # Event loop (all screen state is only touched on this thread)
    # -----------------------------------------------------------------------

    def _ui_loop(self):
        while self._running:
            try:
                batch = [self._events.get(timeout=0.25)]
            except queue.Empty:
                batch = []
            while True:
                try:
                    batch.append(self._events.get_nowait())
                except queue.Empty:
                    break
            try:
                self._process(batch)
            except Exception:
                logger.exception("UI error")

    def _process(self, batch: List[tuple]):
        now = time.monotonic()
        redraw = False
        inputs = [e for e in batch if e[0] in ("turn", "click", "long")]
        if inputs:
            self._last_input = now
            if not self._awake:               # first touch only wakes the screen
                self._set_awake(True)
                batch = [e for e in batch if e[0] not in ("turn", "click", "long")]
                inputs = []
                redraw = True
        reverse = bool(self.config.get("display.encoder_reverse", False))
        i = 0
        while i < len(batch):
            ev = batch[i]
            if ev[0] == "turn":
                steps = 0
                while i < len(batch) and batch[i][0] == "turn":     # combine quick turns
                    steps += batch[i][1]
                    self._turn_times.append(now)
                    i += 1
                if steps:
                    self._toast = None
                    self.screen.on_turn(self, -steps if reverse else steps, self._speed(now))
                redraw = True
                continue
            if ev[0] == "click":
                self._toast = None
                self.screen.on_click(self)
                redraw = True
            elif ev[0] == "long":
                self._toast = None
                self.screen.on_long(self)
                redraw = True
            elif ev[0] == "toast":
                self._toast = (ev[1], now + ev[2], ev[3])
                redraw = True
            elif ev[0] == "redraw":
                redraw = True
            elif ev[0] == "main_menu":
                del self._stack[1:]
                redraw = True
            elif ev[0] == "call":                  # work finished in the background
                ev[1]()
                redraw = True
            elif ev[0] == "learned":
                # remember the knob type so the first spin after a reboot is right too
                self.config.set("display.encoder_half_step", ev[1] == "half")
            i += 1

        if self._toast and now > self._toast[1]:
            self._toast = None
            redraw = True
        timeout = self._screen_timeout()
        if self._awake and timeout and now - self._last_input > timeout and not self._busy():
            self._set_awake(False)
            return
        if self._awake and (redraw or now - self._last_draw > (
                self.LIVE_REFRESH_S if self.screen.live else self.IDLE_REFRESH_S)):
            self._render()

    def _speed(self, now: float) -> int:
        """0 = slow, 1 = medium, 2 = fast turning (detents in the last 0.4 s)."""
        recent = sum(1 for t in self._turn_times if now - t < 0.4)
        return 2 if recent >= 8 else (1 if recent >= 4 else 0)

    def _busy(self) -> bool:
        return any(isinstance(s, TaskScreen) and not s.done for s in self._stack)

    def _screen_timeout(self) -> float:
        try:
            return float(self.config.get("display.timeout", 60) or 0)
        except (TypeError, ValueError):
            return 60.0

    def _set_awake(self, awake: bool):
        self._awake = awake
        try:
            GPIO.output(self.PIN_BACKLIGHT, GPIO.HIGH if awake else GPIO.LOW)
        except Exception:
            pass
        if not awake:
            # go back to the main menu so the next person starts somewhere familiar
            del self._stack[1:]
            self._toast = None

    # -- screen stack ---------------------------------------------------------

    @property
    def screen(self) -> Screen:
        return self._stack[-1]

    def push(self, screen: Screen):
        self._stack.append(screen)

    def pop(self):
        if len(self._stack) > 1:
            self._stack.pop()

    def request_redraw(self):
        self._events.put(("redraw",))

    def toast(self, message: str, duration: float = 1.6, error: bool = False):
        self._toast = (message, time.monotonic() + (duration if duration > 0 else 30), error)

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def _render(self):
        img = Image.new("RGB", (self.W, self.H), self.COLOR_BLACK)
        d = ImageDraw.Draw(img)
        s = self.screen
        top, bottom = self.HEADER_H + 2, self.H - self.FOOTER_H - 2
        try:
            s.draw(self, d, self.W, top, bottom)
        except Exception:
            logger.exception("Screen draw failed")
        self._header(d, s.title, s.color)
        d.rectangle((0, self.H - self.FOOTER_H, self.W, self.H), fill=(18, 18, 28))
        d.text((6, self.H - self.FOOTER_H + 2), s.hint, font=self.font_small, fill=self.COLOR_GRAY)
        if self._toast:
            self._draw_toast(d, self._toast[0], self._toast[2])
        if self.config.get("display.rotate_180", True):
            img = img.transpose(ROTATE_180)
        with self.display_lock:
            self.display.image(img)
        self._last_draw = time.monotonic()

    def _header(self, d, title: str, color):
        d.rectangle((0, 0, self.W, self.HEADER_H), fill=color)
        d.text((8, 4), title, font=self.font_large, fill=self.COLOR_WHITE)
        ts = self._now().strftime("%H:%M")
        bb = d.textbbox((0, 0), ts, font=self.font_small)
        x = self.W - (bb[2] - bb[0]) - 6
        d.text((x, 10), ts, font=self.font_small, fill=self.COLOR_YELLOW)
        if self._monitoring():
            d.ellipse((x - 16, 11, x - 6, 21), fill=self.COLOR_GREEN)   # recording indicator

    def _draw_toast(self, d, message: str, error: bool):
        lines = self.wrap_lines(d, message, self.W - 50, self.font_medium)[:4]
        lh = 22
        h = len(lines) * lh + 20
        y0 = (self.H - h) // 2
        d.rounded_rectangle((16, y0, self.W - 16, y0 + h), radius=10,
                            fill=(90, 20, 20) if error else (25, 45, 90), outline=self.COLOR_WHITE, width=2)
        for i, line in enumerate(lines):
            bb = d.textbbox((0, 0), line, font=self.font_medium)
            d.text(((self.W - (bb[2] - bb[0])) // 2, y0 + 10 + i * lh), line, font=self.font_medium,
                   fill=self.COLOR_WHITE)

    @staticmethod
    def wrap_lines(d, text: str, width: int, font) -> List[str]:
        lines: List[str] = []
        for para in str(text).split("\n"):
            line = ""
            for word in para.split():
                test = (line + " " + word).strip()
                if d.textbbox((0, 0), test, font=font)[2] <= width or not line:
                    line = test
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines

    def wrap_text(self, d, text, xy, width, font, fill, max_lines: int = 6):
        x, y = xy
        for line in self.wrap_lines(d, text, width, font)[:max_lines]:
            d.text((x, y), line, font=font, fill=fill)
            y += font.size + 5

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _now(self) -> datetime:
        """Current time from the shared RTC, or the system clock."""
        if self.rtc is not None:
            try:
                return self.rtc.get_datetime()
            except Exception:
                pass
        from rtc_scheduler import local_now
        return local_now()

    def _monitoring(self) -> bool:
        try:
            return bool(self.is_monitoring and self.is_monitoring())
        except Exception:
            return False

    def _settings_changed(self):
        if self.on_settings_changed:
            try:
                self.on_settings_changed()
            except Exception:
                logger.exception("Applying settings failed")

    def _background(self, fn: Callable[[], Any], busy: str, done: str):
        """Run a slow callback without freezing the knob."""
        self.toast(busy, duration=30)

        def run():
            try:
                error = fn()
                if isinstance(error, str) and error:
                    self._events.put(("toast", error, 4.0, True))
                else:
                    self._events.put(("toast", done, 1.6, False))
            except Exception as e:
                self._events.put(("toast", f"Error: {e}", 4.0, True))
        threading.Thread(target=run, daemon=True).start()

    # -----------------------------------------------------------------------
    # Menus
    # -----------------------------------------------------------------------

    def _push_main_menu(self):
        self._stack = [MenuScreen("BAT MONITOR", self._main_items, back=False)]

    def _main_items(self) -> List[MenuItem]:
        sched = self.config.get_schedule()
        return [
            ("■ Stop monitoring" if self._monitoring() else "▶ Start monitoring", self._toggle_monitoring),
            ("Status", self._open_status),
            (f"Schedule  {sched.get('start_time', '?')}–{sched.get('stop_time', '?')}"
             + ("" if sched.get("enabled", True) else "  OFF"), self._open_schedule),
            ("Set clock", self._open_clock),
            ("Email", self._open_email),
            ("WiFi", self._open_wifi),
            ("Food trendline", self._open_trendline),
            ("Screen & knob", self._open_screen_settings),
        ]

    # monitoring -------------------------------------------------------------

    def _toggle_monitoring(self):
        if self._monitoring():
            self.push(ConfirmScreen("Stop monitoring?",
                                    "Ends tonight's session and emails the report now.",
                                    "Stop", lambda: self._background(
                                        lambda: self.on_stop_monitoring and self.on_stop_monitoring(),
                                        "Stopping… saving report", "Monitoring stopped")))
        elif self.on_start_monitoring:
            self._background(self.on_start_monitoring, "Starting…", "Monitoring started")

    # status -------------------------------------------------------------------

    def _open_status(self):
        def rows():
            now = self._now()
            temp = None
            if self.rtc is not None:
                try:
                    temp = self.rtc.get_temperature()
                except Exception:
                    pass
            sched = self.config.get_schedule()
            email = self.config.get_email_config()
            slope = float(self.config.get("email.trendline_slope", 6.4899))
            icpt = float(self.config.get("email.trendline_intercept", 40.2899))
            return [
                (now.strftime("%a %d %b %Y  %H:%M:%S ") + self._tz_abbr(), self.COLOR_WHITE),
                ("Monitoring: ACTIVE" if self._monitoring() else "Monitoring: idle",
                 self.COLOR_GREEN if self._monitoring() else self.COLOR_GRAY),
                ("", None),
                (f"Schedule {sched.get('start_time')}–{sched.get('stop_time')}  "
                 f"{'ON' if sched.get('enabled') else 'OFF'}", self.COLOR_YELLOW),
                (f"Email recipients: {len(email.get('recipients', []))}", self.COLOR_WHITE),
                (f"Trendline  {slope:.2f} × calls/hr + {icpt:.2f}", self.COLOR_WHITE),
                (f"RTC temp: {temp:.1f}°C" if temp is not None else "RTC: not found (system clock)",
                 self.COLOR_CYAN),
            ] + self._network_rows()
        self.push(InfoScreen("Status", rows))

    @staticmethod
    def _tz_abbr() -> str:
        from rtc_scheduler import timezone_abbr
        return timezone_abbr()

    def _network_rows(self) -> List[Tuple[str, tuple]]:
        import wifi_manager
        st = wifi_manager.current_status()
        if st["hotspot"]:
            return [(f"Setup mode: {st['hotspot_name']}", self.COLOR_YELLOW),
                    (f"ssh to {wifi_manager.HOTSPOT_ADDRESS}", self.COLOR_YELLOW)]
        if st["connected"]:
            return [(f"WiFi: {st['ssid'][:24]}", self.COLOR_GREEN),
                    (f"IP: {st['ip'] or 'waiting for address'}", self.COLOR_GREEN)]
        other = f"  (IP {st['other_ips'][0]})" if st["other_ips"] else ""
        return [("WiFi: not connected" + other, self.COLOR_RED)]

    # schedule -----------------------------------------------------------------

    def _open_schedule(self):
        def items():
            s = self.config.get_schedule()
            return [(f"Start time      {s.get('start_time')}", lambda: self._edit_schedule_time("start_time")),
                    (f"Stop time       {s.get('stop_time')}", lambda: self._edit_schedule_time("stop_time")),
                    (f"Use schedule    {'ON' if s.get('enabled') else 'OFF'}", self._toggle_schedule)]
        self.push(MenuScreen("Schedule", items, color=(15, 50, 110)))

    def _edit_schedule_time(self, key: str):
        h, m = (int(x) for x in self.config.get_schedule().get(key, "00:00").split(":"))

        def save(v):
            self.config.set_schedule(**{key: f"{int(v[0]):02d}:{int(v[1]):02d}"})
            self._settings_changed()
        self.push(NumberEditor("Start time" if key == "start_time" else "Stop time",
                               [Field("Hour", h, 0, 23), Field("Minute", m, 0, 59, accel=(1, 5, 15))],
                               save, separators=[":"]))

    def _toggle_schedule(self):
        on = not self.config.get_schedule().get("enabled", True)
        self.config.set_schedule(enabled=on)
        self._settings_changed()
        self.toast("Schedule ON" if on else "Schedule OFF - monitoring only when started by hand")

    # clock --------------------------------------------------------------------

    def _open_clock(self):
        def items():
            from rtc_scheduler import timezone_label
            return [(f"Date & time   {self._now():%d %b %H:%M}", self._edit_clock),
                    (f"Time zone   {timezone_label()}"[:34], self._pick_timezone)]
        menu = MenuScreen("Clock", items, color=(15, 50, 110))
        menu.live = True
        self.push(menu)

    def _pick_timezone(self):
        from rtc_scheduler import COMMON_TIMEZONES, timezone_label
        current = self.config.get("system.timezone")

        def choose(zone):
            self.config.set("system.timezone", zone)
            self._settings_changed()              # the monitor applies it right away
            self.pop()
            self.toast(f"Time zone: {timezone_label(zone)}")

        def items():
            return [(("✓ " if zone == current else "   ") + timezone_label(zone), lambda z=zone: choose(z))
                    for _label, zone in COMMON_TIMEZONES]
        menu = MenuScreen("Time zone", items, color=(15, 50, 110))
        menu.index = next((i for i, (_l, z) in enumerate(COMMON_TIMEZONES) if z == current), 0)
        self.push(menu)

    def _edit_clock(self):
        now = self._now()

        def save(v):
            year, month, day, hour, minute = (int(x) for x in v)
            day = min(day, calendar.monthrange(year, month)[1])
            new = datetime(year, month, day, hour, minute)
            if self.rtc is not None:
                self.rtc.set_datetime(new)
            try:
                from rtc_scheduler import set_system_time
                set_system_time(new)
            except Exception:
                if self.rtc is None:
                    return "No clock chip and the system clock can't be set"
            return None
        self.push(NumberEditor("Set clock", [
            Field("Year", now.year, 2024, 2099, fmt="{:04d}", wrap=False),
            Field("Month", now.month, 1, 12),
            Field("Day", now.day, 1, 31, accel=(1, 3, 7)),
            Field("Hour", now.hour, 0, 23),
            Field("Minute", now.minute, 0, 59, accel=(1, 5, 15)),
        ], save, separators=["-", "-", " ", ":"], done_message="Clock set"))

    # email --------------------------------------------------------------------

    def _open_email(self):
        def items():
            e = self.config.get_email_config()
            rows: List[MenuItem] = [
                (f"Sender: {(e.get('sender_email') or '(not set)')[:22]}",
                 lambda: self._edit_text("Sender email", "email.sender_email")),
                (f"Password: {'(set)' if e.get('sender_password') else '(not set)'}",
                 lambda: self._edit_text("App password", "email.sender_password", initial="")),
            ]
            for i, addr in enumerate(e.get("recipients", [])):
                rows.append((f"To: {addr[:26]}", lambda i=i: self._open_recipient(i)))
            rows += [("+ Add recipient", self._add_recipient),
                     ("Send test email", self._test_email)]
            return rows
        self.push(MenuScreen("Email", items, color=(80, 40, 0)))

    def _edit_text(self, label: str, key: str, initial: Optional[str] = None, warn_spaces: bool = False):
        def save(text):
            value = text if "password" in key else text.strip()
            if key.endswith("email") and value and "@" not in value:
                return "That doesn't look like an email address"
            self.config.set(key, value)
            self._settings_changed()
            return None
        start = self.config.get(key, "") if initial is None else initial
        self.push(KeyboardScreen(label, start or "", save, warn_spaces=warn_spaces or "password" in key))

    def _add_recipient(self):
        def save(text):
            addr = text.strip()
            if "@" not in addr or "." not in addr.split("@")[-1]:
                return "That doesn't look like an email address"
            r = list(self.config.get("email.recipients") or [])
            if addr.lower() not in (a.lower() for a in r):
                r.append(addr)
                self.config.set("email.recipients", r)
                self._settings_changed()
            return None
        self.push(KeyboardScreen("New recipient email", "", save))

    def _open_recipient(self, idx: int):
        def edit():
            def save(text):
                addr = text.strip()
                if "@" not in addr:
                    return "That doesn't look like an email address"
                r = list(self.config.get("email.recipients") or [])
                if 0 <= idx < len(r):
                    r[idx] = addr
                    self.config.set("email.recipients", r)
                    self._settings_changed()
                return None
            self.pop()
            r = self.config.get("email.recipients") or []
            self.push(KeyboardScreen("Edit recipient", r[idx] if idx < len(r) else "", save))

        def remove():
            def do():
                r = list(self.config.get("email.recipients") or [])
                if 0 <= idx < len(r):
                    r.pop(idx)
                    self.config.set("email.recipients", r)
                    self._settings_changed()
                self.toast("Removed")
            self.pop()
            r = self.config.get("email.recipients") or []
            self.push(ConfirmScreen("Remove recipient?", r[idx] if idx < len(r) else "", "Remove", do))

        r = self.config.get("email.recipients") or []
        self.push(MenuScreen((r[idx] if idx < len(r) else "Recipient")[:20],
                             lambda: [("Edit address", edit), ("Remove", remove)], color=(80, 40, 0)))

    def _test_email(self):
        def work():
            return self.on_test_email() if self.on_test_email else "Email isn't available"
        self.push(TaskScreen(self, "Test email", "Sending", work, "Sent! Check your inbox"))

    # wifi ---------------------------------------------------------------------

    def _open_wifi(self):
        def items():
            w = self.config.get_wifi_config()
            info: List[MenuItem] = [(text, None) for text, _ in self._network_rows()]
            return info + [
                ("Choose network…", self._scan_networks),
                (f"Network: {(w.get('ssid') or '(not set)')[:18]}",
                 lambda: self._edit_text("WiFi network name", "wifi.ssid")),
                (f"Password: {'(set)' if w.get('password') else '(not set)'}",
                 lambda: self._edit_text("WiFi password", "wifi.password", initial="")),
                ("Connect now", self._connect_wifi),
            ]
        menu = MenuScreen("WiFi", items, color=(0, 60, 80)).first_selectable()
        menu.live = True                      # keep the connection/IP lines current
        self.push(menu)

    def _scan_networks(self):
        def work():
            import wifi_manager
            nets = wifi_manager.scan()
            if not nets:
                return "No networks found. Move closer, or type the name with Network:"
            self._events.put(("call", lambda: self._show_networks(nets)))
            return None
        self.push(TaskScreen(self, "WiFi", "Scanning", work, "Networks found", color=(0, 60, 80)))

    def _show_networks(self, nets: List[dict]):
        if isinstance(self.screen, TaskScreen):
            self.pop()

        def bars(signal: int) -> str:
            n = max(1, min(4, round(signal / 25)))
            return "▂▄▆█"[:n].ljust(4)

        def items():
            return [(f"{bars(n['signal'])} {n['ssid'][:19]}{'' if n['secure'] else '  (open)'}",
                     lambda n=n: self._pick_network(n)) for n in nets]
        self.push(MenuScreen("Choose network", items, color=(0, 60, 80)))

    def _pick_network(self, net: dict):
        self.pop()                            # the list
        ssid = net["ssid"]
        self.config.set("wifi.ssid", ssid)
        if not net["secure"]:
            self.config.set("wifi.password", "")
            self._settings_changed()
            self._connect_wifi()
            return

        def save(text):
            if not 8 <= len(text) <= 64:
                return "WiFi passwords are 8-63 characters"
            self.config.set("wifi.password", text)
            self._settings_changed()
            self._events.put(("call", self._connect_wifi))     # after the keyboard closes
            return None
        self.push(KeyboardScreen(f"Password for {ssid[:24]}", "", save, warn_spaces=True,
                                 saved_message=None))     # the connect screen says how it went

    def _connect_wifi(self):
        w = self.config.get_wifi_config()
        if not (w.get("ssid") or "").strip():
            self.toast("Choose a network first", error=True)
            return

        def work():
            import wifi_manager
            return wifi_manager.connect(w["ssid"], w.get("password", ""))
        self.push(TaskScreen(self, "WiFi", "Connecting", work, f"Connected to {w['ssid'][:16]}",
                             color=(0, 60, 80)))

    # trendline ----------------------------------------------------------------

    def _open_trendline(self):
        def items():
            m = float(self.config.get("email.trendline_slope", 6.4899))
            b = float(self.config.get("email.trendline_intercept", 40.2899))
            return [(f"Slope          {m:.2f}", lambda: self._edit_float("Slope", "email.trendline_slope", m)),
                    (f"Intercept      {b:.2f}", lambda: self._edit_float("Intercept", "email.trendline_intercept", b))]
        self.push(MenuScreen("Food trendline", items, color=self.COLOR_PURPLE))

    def _edit_float(self, label: str, key: str, value: float):
        def save(v):
            self.config.set(key, round(v[0], 2))
            self._settings_changed()
        self.push(NumberEditor(label, [Field(label, round(value, 2), -9999, 9999, step=0.01, fmt="{:.2f}",
                                             wrap=False, accel=(1, 10, 100))], save, color=self.COLOR_PURPLE))

    # screen & knob ------------------------------------------------------------

    def _open_screen_settings(self):
        def items():
            t = self._screen_timeout()
            t_label = "never" if not t else (f"{int(t)} s" if t < 60 else f"{int(t // 60)} min")
            return [
                (f"Turn screen 180°   {'ON' if self.config.get('display.rotate_180', True) else 'OFF'}",
                 lambda: self._toggle("display.rotate_180", True)),
                (f"Reverse knob       {'ON' if self.config.get('display.encoder_reverse', False) else 'OFF'}",
                 lambda: self._toggle("display.encoder_reverse", False)),
                (f"Screen off after   {t_label}", self._cycle_timeout),
            ]
        self.push(MenuScreen("Screen & knob", items, color=(40, 40, 40)))

    def _toggle(self, key: str, default: bool):
        self.config.set(key, not self.config.get(key, default))
        self._settings_changed()

    def _cycle_timeout(self):
        t = int(self._screen_timeout())
        nxt = SCREEN_TIMEOUTS[(SCREEN_TIMEOUTS.index(t) + 1) % len(SCREEN_TIMEOUTS)] if t in SCREEN_TIMEOUTS \
            else SCREEN_TIMEOUTS[0]
        self.config.set("display.timeout", nxt)
        self._settings_changed()

    # -----------------------------------------------------------------------
    # Public helpers (safe to call from any thread)
    # -----------------------------------------------------------------------

    def show_message(self, message: str, duration: float = 2.0):
        """Pop-up message over the current screen (does not wake a sleeping screen)."""
        self._events.put(("toast", message, duration, message.lower().startswith(("error", "init error",
                                                                                   "start error", "report error"))))

    def show_main_menu(self):
        self._events.put(("main_menu",))

    def cleanup(self):
        self._running = False
        for t in (getattr(self, "_poll_thread", None), getattr(self, "_ui_thread", None)):
            if t and t.is_alive():
                t.join(timeout=2.0)
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
    print("Testing TFT UI Controller…  turn = move, click = select, hold = back. Ctrl-C to exit")
    ui = None
    try:
        from config_manager import ConfigManager
        ui = UIController(ConfigManager())
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
