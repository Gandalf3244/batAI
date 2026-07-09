"""
Configuration management for Bat Activity Monitor.
Handles loading/saving settings for email, schedule, WiFi, and system parameters.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import time

class ConfigManager:
    """Manages configuration persistence for the bat monitoring system."""
    
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
            "chunk_duration": 10.0,  # seconds per processing chunk
            "min_vocalization_duration": 0.5,
            "max_vocalization_duration": 10.0,
            "silence_duration": 0.3,
            "energy_threshold": 0.001
        },
        "model": {
            "model_path": "12_29_both_species.tflite",
            "label_encoder_path": "label_encoder.pkl",
            "input_shape": [1, 451, 120],
            "max_length": 451,
            "n_mels": 120,
            "n_fft": 2048,
            "hop_length": 512,
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
            "auto_connect": True
        },
        "display": {
            "contrast": 255,
            "flip": False,
            "timeout": 60  # seconds before dimming
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

        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, or create default if not exists."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                
                # Merge with defaults to handle new fields
                config = self._deep_merge(self.DEFAULT_CONFIG.copy(), loaded)
                return config
                
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            config = self.DEFAULT_CONFIG.copy()
            self.save_config(config)
            return config
    
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
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
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
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any, save: bool = True):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., "email.smtp_host")
            value: Value to set
            save: Whether to save config immediately
        """
        keys = key_path.split('.')
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
    
    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration for sending reports."""
        return self.config.get("email", {})
    
    def set_email_config(self, smtp_host: Optional[str] = None, smtp_port: Optional[int] = None,
                        sender_email: Optional[str] = None, sender_password: Optional[str] = None,
                        recipients: Optional[List[str]] = None, save: bool = True):
        """Update email configuration."""
        if smtp_host is not None:
            self.config["email"]["smtp_host"] = smtp_host
        if smtp_port is not None:
            self.config["email"]["smtp_port"] = smtp_port
        if sender_email is not None:
            self.config["email"]["sender_email"] = sender_email
        if sender_password is not None:
            self.config["email"]["sender_password"] = sender_password
        if recipients is not None:
            self.config["email"]["recipients"] = recipients
        
        if save:
            self.save_config()
    
    def get_schedule(self) -> Dict[str, Any]:
        """Get recording schedule configuration."""
        return self.config.get("schedule", {})
    
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
        if start_time is not None:
            self._validate_time_format(start_time)
            self.config["schedule"]["start_time"] = start_time
        if stop_time is not None:
            self._validate_time_format(stop_time)
            self.config["schedule"]["stop_time"] = stop_time
        if enabled is not None:
            self.config["schedule"]["enabled"] = enabled
        
        if save:
            self.save_config()
    
    def _validate_time_format(self, time_str: str):
        """Validate time format is HH:MM."""
        try:
            parts = time_str.split(':')
            if len(parts) != 2:
                raise ValueError
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")
    
    def get_wifi_config(self) -> Dict[str, str]:
        """Get WiFi configuration."""
        return self.config.get("wifi", {})
    
    def set_wifi_config(self, ssid: Optional[str] = None, password: Optional[str] = None,
                       auto_connect: Optional[bool] = None, save: bool = True):
        """Update WiFi configuration."""
        if ssid is not None:
            self.config["wifi"]["ssid"] = ssid
        if password is not None:
            self.config["wifi"]["password"] = password
        if auto_connect is not None:
            self.config["wifi"]["auto_connect"] = auto_connect
        
        if save:
            self.save_config()
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio capture configuration."""
        return self.config.get("audio", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get("model", {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self.config.get("system", {})
    
    def reset_to_defaults(self, save: bool = True):
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        if save:
            self.save_config()
    
    def export_config(self, path: str):
        """Export configuration to a specific file."""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def import_config(self, path: str, save: bool = True):
        """Import configuration from a file."""
        with open(path, 'r') as f:
            imported = json.load(f)
        
        self.config = self._deep_merge(self.DEFAULT_CONFIG.copy(), imported)
        
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
    
    # Clean up test file
    import os
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("\n✓ ConfigManager test completed successfully!")
