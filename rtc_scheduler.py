"""
DS3231 Real-Time Clock interface for scheduling.
Provides accurate timekeeping and schedule management for autonomous operation.
"""

from typing import Any, Optional

try:
    import smbus2 as _smbus2  # type: ignore[import-not-found]
except ImportError:
    _smbus2 = None

smbus2: Any = _smbus2

from datetime import datetime, time as dt_time, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        """
        Read current date and time from RTC.
        
        Returns:
            datetime object with current RTC time
        """
        if self.bus is None:
            raise RuntimeError("DS3231 bus is not initialized")

        try:
            # Read time registers
            seconds = self._bcd_to_dec(self.bus.read_byte_data(self.ADDRESS, self.REG_SECONDS))
            minutes = self._bcd_to_dec(self.bus.read_byte_data(self.ADDRESS, self.REG_MINUTES))
            hours = self._bcd_to_dec(self.bus.read_byte_data(self.ADDRESS, self.REG_HOURS) & 0x3F)
            
            # Read date registers
            date = self._bcd_to_dec(self.bus.read_byte_data(self.ADDRESS, self.REG_DATE))
            month = self._bcd_to_dec(self.bus.read_byte_data(self.ADDRESS, self.REG_MONTH) & 0x1F)
            year = self._bcd_to_dec(self.bus.read_byte_data(self.ADDRESS, self.REG_YEAR)) + 2000
            
            return datetime(year, month, date, hours, minutes, seconds)
        
        except Exception as e:
            logger.error(f"Failed to read datetime from RTC: {e}")
            raise
    
    def set_datetime(self, dt: datetime):
        """
        Set RTC date and time.
        
        Args:
            dt: datetime object to set
        """
        if self.bus is None:
            raise RuntimeError("DS3231 bus is not initialized")

        try:
            self.bus.write_byte_data(self.ADDRESS, self.REG_SECONDS, self._dec_to_bcd(dt.second))
            self.bus.write_byte_data(self.ADDRESS, self.REG_MINUTES, self._dec_to_bcd(dt.minute))
            self.bus.write_byte_data(self.ADDRESS, self.REG_HOURS, self._dec_to_bcd(dt.hour))
            self.bus.write_byte_data(self.ADDRESS, self.REG_DAY, self._dec_to_bcd(dt.weekday() + 1))
            self.bus.write_byte_data(self.ADDRESS, self.REG_DATE, self._dec_to_bcd(dt.day))
            self.bus.write_byte_data(self.ADDRESS, self.REG_MONTH, self._dec_to_bcd(dt.month))
            self.bus.write_byte_data(self.ADDRESS, self.REG_YEAR, self._dec_to_bcd(dt.year - 2000))
            
            logger.info(f"RTC time set to: {dt}")
        
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
            msb = self.bus.read_byte_data(self.ADDRESS, self.REG_TEMP_MSB)
            lsb = self.bus.read_byte_data(self.ADDRESS, self.REG_TEMP_MSB + 1)
            
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
            import subprocess
            rtc_time = self.get_datetime()
            time_str = rtc_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Set system time (requires sudo)
            subprocess.run(['sudo', 'date', '-s', time_str], check=True)
            logger.info(f"System time synced with RTC: {time_str}")
        
        except Exception as e:
            logger.error(f"Failed to sync system time: {e}")
    
    def close(self):
        """Close I2C bus."""
        if self.bus:
            self.bus.close()


class ScheduleManager:
    """
    Manage recording schedule using RTC.
    """
    
    def __init__(self, rtc: Optional[DS3231] = None):
        """
        Initialize schedule manager.
        
        Args:
            rtc: DS3231 instance (creates new one if None)
        """
        self.rtc = rtc or DS3231()
        self.start_time: Optional[dt_time] = None
        self.stop_time: Optional[dt_time] = None
        self.enabled = False
        
        logger.info("Schedule manager initialized")
    
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
        self.enabled = enabled
        
        logger.info(f"Schedule set: {start_time} - {stop_time} (enabled={enabled})")
    
    def is_within_schedule(self) -> bool:
        """
        Check if current time is within recording schedule.
        
        Returns:
            True if should be recording, False otherwise
        """
        if not self.enabled or not self.start_time or not self.stop_time:
            return False
        
        # Get current time from RTC
        current_dt = self.rtc.get_datetime()
        current_time = current_dt.time()
        
        # Handle schedules that cross midnight
        if self.start_time <= self.stop_time:
            # Normal schedule (e.g., 08:00 - 17:00)
            return self.start_time <= current_time <= self.stop_time
        else:
            # Crosses midnight (e.g., 18:00 - 06:00)
            return current_time >= self.start_time or current_time <= self.stop_time
    
    def get_time_until_start(self) -> Optional[int]:
        """
        Get seconds until next start time.
        
        Returns:
            Seconds until start, or None if within schedule or disabled
        """
        if not self.enabled or not self.start_time:
            return None
        
        if self.is_within_schedule():
            return None
        
        current_dt = self.rtc.get_datetime()
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
        if not self.enabled or not self.start_time or not self.stop_time:
            return None
        
        if not self.is_within_schedule():
            return None
        
        current_dt = self.rtc.get_datetime()
        current_time = current_dt.time()
        
        # Calculate next stop datetime
        stop_dt = datetime.combine(current_dt.date(), self.stop_time)
        
        # If schedule crosses midnight and we're before midnight
        if self.start_time > self.stop_time and current_time >= self.start_time:
            stop_dt = stop_dt + timedelta(days=1)
        
        # If stop time has passed today but we're still in schedule (crossed midnight)
        if current_time > self.stop_time and self.start_time > self.stop_time:
            # We're past midnight, stop_dt is already correct
            pass
        elif current_time < self.stop_time:
            # Normal case, stop is later today
            pass
        
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
            'current_time': self.rtc.get_datetime().strftime('%Y-%m-%d %H:%M:%S')
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
