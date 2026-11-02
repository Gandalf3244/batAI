"""
Configuration management for Bat Activity Monitor.
Handles loading/saving settings for email, schedule, WiFi, and system parameters.
"""

import copy
import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

class ConfigManager:
    """Manages configuration persistence for the bat monitoring system."""

    # Model shipped before the retrained bat_model.tflite (kept as a fallback)
    ORIGINAL_MODEL = "12_29_both_species.tflite"

    DEFAULT_CONFIG = {
        "email": {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "",
            "sender_password": "",
            "recipients": [],
            "trendline_slope": 6.4899,
            "trendline_intercept": 40.2899
        },
        "schedule": {
            "start_time": "18:00",  # 6 PM
            "stop_time": "06:00",   # 6 AM next day
            "enabled": True
        },
        "audio": {
            "sample_rate": 48000,
            "channels": 2,
            "chunk_duration": 1.0,  # seconds of audio handed to the detector at a time
            "min_vocalization_duration": 0.5,
            "max_vocalization_duration": 10.0,
            "silence_duration": 0.3,
            "energy_threshold": 0.001
        },
        "model": {
            "model_path": "bat_model.tflite",  # retrained 2026-10-24 (training/README.md)
            "label_encoder_path": "label_encoder.pkl",
            "input_shape": [1, 451, 360],
            "max_length": 451,
            "n_mels": 120,
            "n_fft": 2048,
            "hop_length": 512,
            "sample_rate": 16000,  # rate the model was trained on; audio is resampled to it
            "confidence_threshold": 0.6,
            "classes": [
                "Rods_Fighting",
                "Straws_Fighting",
                "Straws_Talking",
                "Straws_Want_Food"
            ],
            "val_macro_f1": 0.9751
        },
        "system": {
            "timezone": "America/New_York",
            "log_level": "INFO",
            "watchdog_timeout": 300,  # seconds
            "email_retry_attempts": 3,
            "email_retry_delay": 60  # seconds
        },
        "wifi": {
            "ssid": "",
            "password": "",
            "auto_connect": True,
            "setup_hotspot": True,              # no WiFi ~90 s after boot -> start "BatPi-Setup"
            "hotspot_password": "batsetup123"   # change this for a unit in a public place
        },
        "display": {
            "rotate_180": True,        # the screen module is mounted upside down
            "encoder_reverse": False,  # swap knob direction
            "encoder_half_step": False,  # knob type, detected automatically
            "timeout": 60              # seconds without input before the screen turns off (0 = never)
        },
        "web": {
            "enabled": True,
            "host": "127.0.0.1",  # localhost only: reach it through an SSH tunnel
            "port": 8080
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to config file. Defaults to config.json in same dir as script.
        """
        if config_path is None:
            script_dir = Path(__file__).parent
            self.config_path = script_dir / "config.json"
        else:
            self.config_path = Path(config_path)

        # Config is shared by the main loop, the TFT UI thread and the web UI
        self._lock = threading.RLock()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, or create default if not exists."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)

                # Merge with defaults to handle new fields
                config = self._deep_merge(copy.deepcopy(self.DEFAULT_CONFIG), loaded)
                if self._migrate(config, loaded):
                    # Persist, so a one-time migration isn't redone (or undone) on every load
                    self.save_config(config)
                return config

            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")
                return copy.deepcopy(self.DEFAULT_CONFIG)
        else:
            # Create default config file
            config = copy.deepcopy(self.DEFAULT_CONFIG)
            self.save_config(config)
            return config

    def _migrate(self, config: Dict[str, Any], loaded: Dict[str, Any]) -> bool:
        """Carry settings written by older versions over to their current keys.

        Returns True if anything changed."""
        changed = False
        # Older TFT UI saved the trendline under "display", but the email
        # report has always read it from "email".
        old_display = loaded.get("display", {})
        old_email = loaded.get("email", {})
        for key in ("trendline_slope", "trendline_intercept"):
            if key in old_display and key not in old_email:
                config["email"][key] = old_display[key]
            if config["display"].pop(key, None) is not None:
                changed = True
        # Unused old display settings
        for key in ("contrast", "flip"):
            if config["display"].pop(key, None) is not None:
                changed = True

        # Switch once from the original model to the retrained one when it's
        # installed. The flag means a later manual switch back is respected.
        model = config["model"]
        if (model.get("model_path") == self.ORIGINAL_MODEL
                and not model.get("upgraded_from_original")
                and self.resolve_path(self.DEFAULT_CONFIG["model"]["model_path"]).exists()):
            model["model_path"] = self.DEFAULT_CONFIG["model"]["model_path"]
            model["upgraded_from_original"] = True
            print(f"Config: switched model from {self.ORIGINAL_MODEL} to {model['model_path']}")
            changed = True
        return changed

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge update dict into base dict."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """Save configuration to file (atomically, so a power cut can't corrupt it)."""
        with self._lock:
            if config is None:
                config = self.config

            tmp_path = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
            try:
                with open(tmp_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.config_path)
            except Exception as e:
                raise IOError(f"Failed to save config to {self.config_path}: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., "email.smtp_host")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        with self._lock:
            value = self.config

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default

            return copy.deepcopy(value)

    def set(self, key_path: str, value: Any, save: bool = True):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., "email.smtp_host")
            value: Value to set
            save: Whether to save config immediately
        """
        keys = key_path.split('.')
        with self._lock:
            config = self.config

            # Navigate to parent
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]

            # Set value
            config[keys[-1]] = value

            if save:
                self.save_config()

    def update_section(self, section: str, values: Dict[str, Any], save: bool = True):
        """Update several keys of one section at once (single save)."""
        with self._lock:
            self.config.setdefault(section, {}).update(values)
            if save:
                self.save_config()

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the whole configuration."""
        with self._lock:
            return copy.deepcopy(self.config)

    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration for sending reports."""
        return self.get("email", {})

    def set_email_config(self, smtp_host: Optional[str] = None, smtp_port: Optional[int] = None,
                        sender_email: Optional[str] = None, sender_password: Optional[str] = None,
                        recipients: Optional[List[str]] = None, save: bool = True):
        """Update email configuration."""
        updates: Dict[str, Any] = {}
        if smtp_host is not None:
            updates["smtp_host"] = smtp_host
        if smtp_port is not None:
            updates["smtp_port"] = smtp_port
        if sender_email is not None:
            updates["sender_email"] = sender_email
        if sender_password is not None:
            updates["sender_password"] = sender_password
        if recipients is not None:
            updates["recipients"] = recipients

        self.update_section("email", updates, save=save)

    def get_schedule(self) -> Dict[str, Any]:
        """Get recording schedule configuration."""
        return self.get("schedule", {})

    def set_schedule(self, start_time: Optional[str] = None, stop_time: Optional[str] = None,
                    enabled: Optional[bool] = None, save: bool = True):
        """
        Update recording schedule.

        Args:
            start_time: Start time in "HH:MM" format
            stop_time: Stop time in "HH:MM" format
            enabled: Whether schedule is enabled
            save: Whether to save immediately
        """
        updates: Dict[str, Any] = {}
        if start_time is not None:
            updates["start_time"] = self.normalize_time(start_time)
        if stop_time is not None:
            updates["stop_time"] = self.normalize_time(stop_time)
        if enabled is not None:
            updates["enabled"] = bool(enabled)

        self.update_section("schedule", updates, save=save)

    @staticmethod
    def normalize_time(time_str: str) -> str:
        """Validate an "H:MM"/"HH:MM" string and return it as zero-padded "HH:MM"."""
        try:
            parts = time_str.strip().split(':')
            if len(parts) != 2:
                raise ValueError
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")
        return f"{hour:02d}:{minute:02d}"

    def _validate_time_format(self, time_str: str):
        """Validate time format is HH:MM."""
        self.normalize_time(time_str)

    def get_wifi_config(self) -> Dict[str, str]:
        """Get WiFi configuration."""
        return self.get("wifi", {})

    def set_wifi_config(self, ssid: Optional[str] = None, password: Optional[str] = None,
                       auto_connect: Optional[bool] = None, save: bool = True):
        """Update WiFi configuration."""
        updates: Dict[str, Any] = {}
        if ssid is not None:
            updates["ssid"] = ssid
        if password is not None:
            updates["password"] = password
        if auto_connect is not None:
            updates["auto_connect"] = auto_connect

        self.update_section("wifi", updates, save=save)

    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio capture configuration."""
        return self.get("audio", {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get("model", {})

    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self.get("system", {})

    def get_web_config(self) -> Dict[str, Any]:
        """Get web settings UI configuration."""
        return self.get("web", {})

    def resolve_path(self, path: str) -> Path:
        """Resolve a config path (e.g. model file) relative to the config file's folder."""
        p = Path(path)
        return p if p.is_absolute() else self.config_path.parent / p

    def reset_to_defaults(self, save: bool = True):
        """Reset configuration to defaults."""
        with self._lock:
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            if save:
                self.save_config()

    def export_config(self, path: str):
        """Export configuration to a specific file."""
        with open(path, 'w') as f:
            json.dump(self.snapshot(), f, indent=2)

    def import_config(self, path: str, save: bool = True):
        """Import configuration from a file."""
        with open(path, 'r') as f:
            imported = json.load(f)

        with self._lock:
            self.config = self._deep_merge(copy.deepcopy(self.DEFAULT_CONFIG), imported)
            self._migrate(self.config, imported)

            if save:
                self.save_config()


# Convenience functions for quick access
def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Load configuration manager."""
    return ConfigManager(config_path)


if __name__ == "__main__":
    # Test configuration manager
    print("Testing ConfigManager...")

    config = ConfigManager("test_config.json")

    # Test setting values
    config.set("email.sender_email", "test@example.com")
    config.set("schedule.start_time", "20:00")
    config.set_email_config(recipients=["keeper1@zoo.com", "keeper2@zoo.com"])

    # Test getting values
    print(f"Email: {config.get('email.sender_email')}")
    print(f"Start time: {config.get('schedule.start_time')}")
    print(f"Recipients: {config.get('email.recipients')}")

    # Test email config
    email_cfg = config.get_email_config()
    print(f"\nEmail config: {email_cfg}")

    # Test schedule
    schedule = config.get_schedule()
    print(f"Schedule: {schedule}")

    # Defaults must not be modified by edits
    assert ConfigManager.DEFAULT_CONFIG["email"]["recipients"] == []

    # Clean up test file
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")

    print("\n✓ ConfigManager test completed successfully!")
