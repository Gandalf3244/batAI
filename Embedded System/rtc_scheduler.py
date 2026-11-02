"""
DS3231 Real-Time Clock interface for scheduling.
Provides accurate timekeeping and schedule management for autonomous operation.

Time handling (like Linux itself does it):
- The DS3231 chip keeps UTC, so daylight-saving changes never make it wrong.
- Everything shown or scheduled uses one configurable time zone
  (config "system.timezone"); the conversion happens here.
"""

import os
import time
from datetime import timezone as _utc_tz
from typing import Any, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:                       # Python < 3.9
    ZoneInfo = None                       # type: ignore[assignment,misc]
    ZoneInfoNotFoundError = Exception     # type: ignore[assignment,misc]

try:
    import smbus2 as _smbus2  # type: ignore[import-not-found]
except ImportError:
    _smbus2 = None

smbus2: Any = _smbus2

from datetime import datetime, time as dt_time, timedelta
import logging
import subprocess
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time zone
# ---------------------------------------------------------------------------

DEFAULT_TIMEZONE = "America/Los_Angeles"     # the Oregon Zoo
# Offered in the device's menu and the control panel (any IANA name works in config.json)
COMMON_TIMEZONES: List[Tuple[str, str]] = [
    ("Pacific", "America/Los_Angeles"),
    ("Mountain", "America/Denver"),
    ("Arizona", "America/Phoenix"),
    ("Central", "America/Chicago"),
    ("Eastern", "America/New_York"),
    ("Alaska", "America/Anchorage"),
    ("Hawaii", "Pacific/Honolulu"),
    ("UTC", "UTC"),
]

_tz: Any = None          # ZoneInfo for the configured zone (None = the computer's own zone)
_tz_name: Optional[str] = None


def valid_timezone(name: str) -> bool:
    if not name or ZoneInfo is None:
        return False
    try:
        ZoneInfo(name)
        return True
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return False


def apply_timezone(name: str, system: bool = False) -> str:
    """
    Use time zone `name` for every local time in this program. With system=True
    also set the Pi's own time zone (shell, logs of other programs) - that needs
    passwordless sudo and is skipped quietly if it isn't available.

    Returns "" or a note about what couldn't be done.
    """
    global _tz, _tz_name
    if not valid_timezone(name):
        return f"Unknown time zone {name!r}"
    _tz, _tz_name = ZoneInfo(name), name
    if hasattr(time, "tzset"):
        os.environ["TZ"] = name          # datetime.now(), log timestamps etc. follow this
        time.tzset()
    if not system:
        return ""
    try:
        current = subprocess.run(["timedatectl", "show", "-p", "Timezone", "--value"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        if current and current != name:
            proc = subprocess.run(["sudo", "-n", "timedatectl", "set-timezone", name],
                                  capture_output=True, text=True, timeout=15)
            if proc.returncode != 0:
                return f"Couldn't set the Pi's time zone: {proc.stderr.strip()}"
            logger.info(f"Pi time zone set to {name}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def timezone_name() -> str:
    return _tz_name or time.tzname[0]


def timezone_label(name: Optional[str] = None) -> str:
    """e.g. 'Pacific (PST, UTC-8)'"""
    name = name or _tz_name
    if not name or not valid_timezone(name):
        return timezone_name()
    now = datetime.now(ZoneInfo(name))
    off = now.utcoffset() or timedelta(0)
    hours = off.total_seconds() / 3600
    off_txt = "UTC" if hours == 0 else f"UTC{hours:+g}"
    friendly = next((label for label, zone in COMMON_TIMEZONES if zone == name), name)
    abbr = now.tzname()
    return f"{friendly} ({abbr}, {off_txt})" if abbr and abbr != "UTC" else f"{friendly} ({off_txt})"


def timezone_abbr() -> str:
    """e.g. 'PST' / 'PDT' right now."""
    return datetime.now(_tz).tzname() if _tz else time.strftime("%Z")


def local_from_timestamp(ts: float) -> datetime:
    """Unix timestamp -> local time in the configured zone (no tzinfo)."""
    return datetime.fromtimestamp(ts, _tz).replace(tzinfo=None) if _tz else datetime.fromtimestamp(ts)


def local_now() -> datetime:
    """Current local time (configured zone), without tzinfo."""
    return datetime.now(_tz).replace(tzinfo=None) if _tz else datetime.now()


def to_utc(local: datetime) -> datetime:
    """Local time (configured zone, no tzinfo) -> UTC (no tzinfo)."""
    aware = local.replace(tzinfo=_tz) if _tz else local.astimezone()
    return aware.astimezone(_utc_tz.utc).replace(tzinfo=None)


def from_utc(utc: datetime) -> datetime:
    """UTC (no tzinfo) -> local time in the configured zone (no tzinfo)."""
    return utc.replace(tzinfo=_utc_tz.utc).astimezone(_tz).replace(tzinfo=None)


def ntp_synced() -> Optional[bool]:
    """True if the system clock is set from the internet right now (None = can't tell)."""
    try:
        p = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                           capture_output=True, text=True, timeout=5)
        return p.stdout.strip() == "yes" if p.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


class DS3231:
    """
    Interface for DS3231 Real-Time Clock module via I2C.
    
    Hardware Connections:
    - SDA: GPIO 2 (I2C SDA)
    - SCL: GPIO 3 (I2C SCL)
    - VCC: 3.3V or 5V
    - GND: Ground
    """
    
    # DS3231 I2C address
    ADDRESS = 0x68
    
    # Register addresses
    REG_SECONDS = 0x00
    REG_MINUTES = 0x01
    REG_HOURS = 0x02
    REG_DAY = 0x03
    REG_DATE = 0x04
    REG_MONTH = 0x05
    REG_YEAR = 0x06
    REG_ALARM1_SECONDS = 0x07
    REG_CONTROL = 0x0E
    REG_STATUS = 0x0F
    REG_TEMP_MSB = 0x11
    
    def __init__(self, bus_number: int = 1):
        """
        Initialize DS3231 RTC.
        
        Args:
            bus_number: I2C bus number (1 for Raspberry Pi)
        """
        self.bus_number = bus_number
        self.bus: Optional[Any] = None
        # The main loop, TFT UI and web UI all read the clock from different threads
        self._lock = threading.Lock()
        if smbus2 is None:
            raise RuntimeError("smbus2 is not installed")
        
        try:
            self.bus = smbus2.SMBus(bus_number)
            logger.info(f"DS3231 RTC initialized on I2C bus {bus_number}")
        except Exception as e:
            logger.error(f"Failed to initialize DS3231: {e}")
            raise
    
    def _bcd_to_dec(self, bcd: int) -> int:
        """Convert BCD to decimal."""
        return ((bcd >> 4) * 10) + (bcd & 0x0F)
    
    def _dec_to_bcd(self, dec: int) -> int:
        """Convert decimal to BCD."""
        return ((dec // 10) << 4) + (dec % 10)
    
    def get_datetime(self) -> datetime:
        """Current local time (configured time zone) from the clock chip."""
        return from_utc(self.get_utc())

    def set_datetime(self, dt: datetime):
        """Set the clock chip from a local time (configured time zone)."""
        self.set_utc(to_utc(dt))
        logger.info(f"RTC set to {dt:%Y-%m-%d %H:%M:%S} local")

    def get_utc(self) -> datetime:
        """
        Read the chip's registers (UTC).

        Returns:
            datetime object with current RTC time
        """
        if self.bus is None:
            raise RuntimeError("DS3231 bus is not initialized")

        try:
            # Read all time/date registers in one transaction: the DS3231 latches
            # them for a block read, so the values can't roll over mid-read
            # (e.g. 05:59:59 being read back as 06:59:59).
            with self._lock:
                regs = self.bus.read_i2c_block_data(self.ADDRESS, self.REG_SECONDS, 7)

            seconds = self._bcd_to_dec(regs[0] & 0x7F)
            minutes = self._bcd_to_dec(regs[1] & 0x7F)
            hour_reg = regs[2]
            if hour_reg & 0x40:
                # 12-hour mode (bit 5 = PM)
                hours = self._bcd_to_dec(hour_reg & 0x1F) % 12 + (12 if hour_reg & 0x20 else 0)
            else:
                hours = self._bcd_to_dec(hour_reg & 0x3F)

            date = self._bcd_to_dec(regs[4] & 0x3F)
            month = self._bcd_to_dec(regs[5] & 0x1F)
            year = self._bcd_to_dec(regs[6]) + 2000

            return datetime(year, month, date, hours, minutes, seconds)
        
        except Exception as e:
            logger.error(f"Failed to read datetime from RTC: {e}")
            raise
    
    def set_utc(self, dt: datetime):
        """
        Write the chip's registers.

        Args:
            dt: UTC time to store
        """
        if self.bus is None:
            raise RuntimeError("DS3231 bus is not initialized")

        try:
            # Write all registers at once (24-hour mode) so the clock never holds a half-set time
            regs = [
                self._dec_to_bcd(dt.second),
                self._dec_to_bcd(dt.minute),
                self._dec_to_bcd(dt.hour),
                self._dec_to_bcd(dt.weekday() + 1),
                self._dec_to_bcd(dt.day),
                self._dec_to_bcd(dt.month),
                self._dec_to_bcd(dt.year - 2000),
            ]
            with self._lock:
                self.bus.write_i2c_block_data(self.ADDRESS, self.REG_SECONDS, regs)
                # Clear the oscillator-stop flag so the time is marked valid
                status = self.bus.read_byte_data(self.ADDRESS, self.REG_STATUS)
                self.bus.write_byte_data(self.ADDRESS, self.REG_STATUS, status & 0x7F)
            
            logger.debug(f"RTC registers set to {dt} UTC")

        except Exception as e:
            logger.error(f"Failed to set RTC datetime: {e}")
            raise
    
    def get_temperature(self) -> float:
        """
        Read temperature from RTC (DS3231 has built-in temperature sensor).
        
        Returns:
            Temperature in Celsius
        """
        if self.bus is None:
            logger.error("DS3231 bus is not initialized")
            return 0.0

        try:
            with self._lock:
                msb, lsb = self.bus.read_i2c_block_data(self.ADDRESS, self.REG_TEMP_MSB, 2)
            
            # Temperature is in MSB and upper 2 bits of LSB
            temp = msb + ((lsb >> 6) * 0.25)
            
            # Handle negative temperatures
            if msb & 0x80:
                temp = temp - 256
            
            return temp
        
        except Exception as e:
            logger.error(f"Failed to read temperature: {e}")
            return 0.0
    
    def sync_system_time(self):
        """Sync system time with RTC (requires root privileges)."""
        try:
            set_system_time(self.get_datetime())
        except Exception as e:
            logger.error(f"Failed to sync system time: {e}")
    
    def close(self):
        """Close I2C bus."""
        if self.bus:
            self.bus.close()
            self.bus = None


def set_system_time(dt: datetime):
    """Set the Linux system clock from a local time (needs root or passwordless sudo)."""
    utc_str = to_utc(dt).strftime("%Y-%m-%d %H:%M:%S")
    # -u: give it UTC, so it's right whatever the Pi's own time zone is set to.
    # -n: fail instead of hanging on a password prompt when running as a service
    subprocess.run(['sudo', '-n', 'date', '-u', '-s', utc_str], check=True,
                   capture_output=True, timeout=10)
    logger.info(f"System time set to: {dt:%Y-%m-%d %H:%M:%S} local ({utc_str} UTC)")


class ScheduleManager:
    """
    Manage recording schedule using RTC.
    """
    
    def __init__(self, rtc: Optional[DS3231] = None):
        """
        Initialize schedule manager.
        
        Args:
            rtc: DS3231 instance, or None to use the system clock
        """
        self.rtc = rtc
        self.start_time: Optional[dt_time] = None
        self.stop_time: Optional[dt_time] = None
        self.enabled = False
        self._rtc_failed = False

        logger.info("Schedule manager initialized" + ("" if rtc else " (using system clock)"))

    def now(self) -> datetime:
        """Current time from the RTC, falling back to the system clock if the RTC can't be read."""
        if self.rtc is not None:
            try:
                current = self.rtc.get_datetime()
                if self._rtc_failed:
                    logger.info("RTC readable again")
                    self._rtc_failed = False
                return current
            except Exception as e:
                if not self._rtc_failed:
                    logger.warning(f"RTC read failed ({e}); using system clock until it recovers")
                    self._rtc_failed = True
        return local_now()

    @property
    def clock_source(self) -> str:
        """'rtc' or 'system' - where the schedule's time currently comes from."""
        return "rtc" if self.rtc is not None and not self._rtc_failed else "system"
    
    def set_schedule(self, start_time: str, stop_time: str, enabled: bool = True):
        """
        Set recording schedule.
        
        Args:
            start_time: Start time in "HH:MM" format
            stop_time: Stop time in "HH:MM" format
            enabled: Whether schedule is enabled
        """
        # Parse times
        start_parts = start_time.split(':')
        stop_parts = stop_time.split(':')

        self.start_time = dt_time(int(start_parts[0]), int(start_parts[1]))
        self.stop_time = dt_time(int(stop_parts[0]), int(stop_parts[1]))
        self.enabled = bool(enabled)
        
        logger.info(f"Schedule set: {start_time} - {stop_time} (enabled={enabled})")
    
    def is_within_schedule(self) -> bool:
        """
        Check if current time is within recording schedule.
        
        Returns:
            True if should be recording, False otherwise
        """
        if not self.enabled or self.start_time is None or self.stop_time is None:
            return False

        # Get current time from RTC
        current_time = self.now().time()

        # Window is [start, stop): recording ends as soon as the stop minute begins
        if self.start_time < self.stop_time:
            # Normal schedule (e.g., 08:00 - 17:00)
            return self.start_time <= current_time < self.stop_time
        elif self.start_time > self.stop_time:
            # Crosses midnight (e.g., 18:00 - 06:00)
            return current_time >= self.start_time or current_time < self.stop_time
        else:
            # Start == stop: treat as "record around the clock"
            return True
    
    def get_time_until_start(self) -> Optional[int]:
        """
        Get seconds until next start time.
        
        Returns:
            Seconds until start, or None if within schedule or disabled
        """
        if not self.enabled or self.start_time is None:
            return None

        if self.is_within_schedule():
            return None

        current_dt = self.now()
        current_time = current_dt.time()
        
        # Calculate next start datetime
        start_dt = datetime.combine(current_dt.date(), self.start_time)
        
        # If start time has passed today, use tomorrow
        if current_time > self.start_time:
            start_dt = start_dt + timedelta(days=1)
        
        # Calculate difference
        diff = start_dt - current_dt
        return int(diff.total_seconds())
    
    def get_time_until_stop(self) -> Optional[int]:
        """
        Get seconds until stop time.
        
        Returns:
            Seconds until stop, or None if not within schedule or disabled
        """
        if not self.enabled or self.start_time is None or self.stop_time is None:
            return None

        if not self.is_within_schedule() or self.start_time == self.stop_time:
            return None

        current_dt = self.now()

        # Next occurrence of the stop time
        stop_dt = datetime.combine(current_dt.date(), self.stop_time)
        if stop_dt <= current_dt:
            stop_dt = stop_dt + timedelta(days=1)
        
        # Calculate difference
        diff = stop_dt - current_dt
        return max(0, int(diff.total_seconds()))
    
    def get_schedule_info(self) -> dict:
        """Get schedule information."""
        return {
            'enabled': self.enabled,
            'start_time': self.start_time.strftime('%H:%M') if self.start_time else None,
            'stop_time': self.stop_time.strftime('%H:%M') if self.stop_time else None,
            'is_within_schedule': self.is_within_schedule(),
            'time_until_start': self.get_time_until_start(),
            'time_until_stop': self.get_time_until_stop(),
            'current_time': self.now().strftime('%Y-%m-%d %H:%M:%S'),
            'clock_source': self.clock_source,
            'timezone': timezone_label()
        }


def test_rtc():
    """Test RTC functionality."""
    print("Testing DS3231 RTC...")
    
    try:
        rtc = DS3231()
        
        # Read current time
        current_time = rtc.get_datetime()
        print(f"Current RTC time: {current_time}")
        
        # Read temperature
        temp = rtc.get_temperature()
        print(f"RTC temperature: {temp:.2f}°C")
        
        # Test schedule manager
        print("\nTesting Schedule Manager...")
        scheduler = ScheduleManager(rtc)
        
        # Set a test schedule
        scheduler.set_schedule("18:00", "06:00", enabled=True)
        
        info = scheduler.get_schedule_info()
        print(f"Schedule info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✓ RTC test completed!")
        
        rtc.close()
        
    except Exception as e:
        print(f"\n✗ RTC test failed: {e}")
        print("\nNote: This test requires DS3231 hardware connected via I2C.")
        print("Enable I2C: sudo raspi-config -> Interface Options -> I2C")


if __name__ == "__main__":
    test_rtc()
