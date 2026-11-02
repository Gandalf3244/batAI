#!/usr/bin/env python3
"""
Real-Time Bat Activity Monitoring System
Main application for Raspberry Pi Zero 2 W

Captures audio from I2S microphone, classifies bat vocalizations in real-time,
tracks activity, and sends email reports - all without storing audio files.
"""

import os
import sys
import time
import signal
import shutil
import socket
import logging
import threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from math import gcd
from typing import Any, Dict, Optional
import numpy as np
from scipy.signal import resample_poly

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import all components
from config_manager import ConfigManager
from audio_capture import I2SAudioCapture, VocalizationDetector
from feature_extraction import SpectrogramExtractor
from model_inference import BatClassifier
from activity_tracker import ActivityTracker
from email_sender import EmailSender, EmailQueue
from rtc_scheduler import DS3231, ScheduleManager, set_system_time
import wifi_manager

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / 'bat_monitor.log'
REPORT_DIR = SCRIPT_DIR / 'reports'

# Configure logging. force=True is required: the imported modules already call
# basicConfig(), which would otherwise make this a no-op (and no log file).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True,
)
logger = logging.getLogger(__name__)


class BatMonitorSystem:
    """
    Main system controller for real-time bat monitoring.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize bat monitoring system.

        Args:
            config_path: Path to config file (None = default)
        """
        logger.info("=" * 60)
        logger.info("Bat Activity Monitoring System Starting")
        logger.info("=" * 60)

        # Load configuration
        self.config = ConfigManager(config_path) if config_path is not None else ConfigManager()
        self._apply_log_level()
        logger.info("Configuration loaded")

        # Initialize components
        self.rtc: Optional[DS3231] = None
        self.scheduler: Optional[ScheduleManager] = None
        self.ui: Optional[Any] = None
        self.web_ui: Optional[Any] = None
        self.audio_capture: Optional[I2SAudioCapture] = None
        self.detector: Optional[VocalizationDetector] = None
        self.feature_extractor: Optional[SpectrogramExtractor] = None
        self.model_sample_rate = 16000
        self._capture_rate = 48000          # sample rate of the current session
        self.classifier: Optional[BatClassifier] = None
        self.activity_tracker: Optional[ActivityTracker] = None
        self.email_sender: Optional[EmailSender] = None
        self.email_queue: Optional[EmailQueue] = None
        self.wifi_watchdog: Optional[wifi_manager.SetupHotspotWatchdog] = None

        # System state
        self.is_monitoring = False
        self.should_stop = False
        self._stop_event = threading.Event()
        self._monitor_lock = threading.RLock()
        self._last_start_attempt = 0.0
        self._start_retry_delay = 30.0  # seconds before retrying after a failed start
        # Started by hand outside the schedule window: don't auto-stop it
        self.manual_mode = False
        # Stopped by hand inside the schedule window: don't auto-restart until the window ends
        self._skip_current_window = False
        self._detection_dirty = False
        self.started_at = time.time()
        self.last_error: Optional[str] = None
        self.last_session: Optional[Dict[str, Any]] = None
        self.session_stats = {"segments": 0, "low_confidence": 0}
        self.audio_level = {"rms": 0.0, "peak": 0.0, "loudest": 0.0, "floor": 0.0, "updated": None}

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize all system components."""
        try:
            # Initialize RTC and scheduler. A missing/faulty RTC is not fatal:
            # the schedule falls back to the system clock.
            logger.info("Initializing RTC...")
            try:
                self.rtc = DS3231()
                self.rtc.get_datetime()
            except Exception as e:
                logger.warning(f"DS3231 RTC not available ({e}); using system clock")
                if self.rtc:
                    self.rtc.close()
                self.rtc = None
            self.scheduler = ScheduleManager(self.rtc)

            # Load schedule from config
            schedule_cfg = self.config.get_schedule()
            self.scheduler.set_schedule(
                schedule_cfg['start_time'],
                schedule_cfg['stop_time'],
                schedule_cfg['enabled']
            )

            # Initialize UI (optional: the system also runs headless / over SSH)
            logger.info("Initializing UI...")
            try:
                from ui_controller import UIController
                self.ui = UIController(self.config, rtc=self.rtc)
                self.ui.on_start_monitoring = self._ui_start_monitoring
                self.ui.on_stop_monitoring = self._ui_stop_monitoring
                self.ui.is_monitoring = lambda: self.is_monitoring
                self.ui.on_test_email = self._ui_test_email
                self.ui.on_settings_changed = self.apply_config_changes
            except Exception as e:
                logger.warning(f"TFT display not available ({e}); running headless")
                self.ui = None

            # Initialize model
            logger.info("Loading AI model...")
            model_cfg = self.config.get_model_config()
            self.classifier = BatClassifier(
                str(self.config.resolve_path(model_cfg['model_path'])),
                str(self.config.resolve_path(model_cfg['label_encoder_path']))
            )

            # Initialize feature extractor
            audio_cfg = self.config.get_audio_config()
            model_input_shape = tuple(int(x) for x in self.classifier.input_details[0]['shape'])
            model_cfg_input_shape = tuple(model_cfg.get('input_shape', []))
            if model_cfg_input_shape and model_cfg_input_shape != model_input_shape:
                logger.warning(
                    "Config model.input_shape %s does not match TFLite input %s; using TFLite input shape",
                    model_cfg_input_shape,
                    model_input_shape,
                )
            # Spectrograms must be computed at the rate the model was trained on
            # (the training clips are 16 kHz; the microphone records at 48 kHz)
            self.model_sample_rate = int(model_cfg.get('sample_rate', audio_cfg['sample_rate']))
            self.feature_extractor = SpectrogramExtractor(
                sample_rate=self.model_sample_rate,
                n_mels=int(model_cfg.get('n_mels', 120)),
                n_fft=int(model_cfg.get('n_fft', 2048)),
                hop_length=int(model_cfg.get('hop_length', 512)),
                target_length=int(model_input_shape[1]),
                model_feature_dim=int(model_input_shape[2]),
            )

            # Initialize email sender
            logger.info("Initializing email sender...")
            email_cfg = self.config.get_email_config()
            system_cfg = self.config.get_system_config()
            self.email_sender = EmailSender(email_cfg)
            self.email_queue = EmailQueue(
                self.email_sender,
                max_retries=int(system_cfg.get('email_retry_attempts', 3)),
                retry_delay=int(system_cfg.get('email_retry_delay', 60)),
                outbox_dir=str(REPORT_DIR / "outbox"),     # unsent reports wait here, never deleted
            )
            self.email_queue.start()

            # Web settings UI (reach it from your computer through an SSH tunnel)
            self._start_web_ui()

            # Never get locked out: no WiFi shortly after boot -> setup hotspot
            self.wifi_watchdog = wifi_manager.SetupHotspotWatchdog(
                enabled=lambda: bool(self.config.get('wifi.setup_hotspot', True)),
                on_change=self._wifi_changed,
                password=lambda: str(self.config.get('wifi.hotspot_password') or wifi_manager.HOTSPOT_PASSWORD))
            self.wifi_watchdog.start()

            logger.info("System initialization complete!")
            if self.ui:
                self.ui.show_message("System Ready!", duration=2.0)

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            if self.ui:
                self.ui.show_message(f"Init Error: {str(e)[:60]}", duration=30.0)
                time.sleep(5)   # leave it on screen long enough to read before exiting
            raise

    def _start_web_ui(self):
        web_cfg = self.config.get_web_config()
        if not web_cfg.get('enabled', True):
            logger.info("Web settings UI disabled in config")
            return
        try:
            from web_ui import WebUI
            self.web_ui = WebUI(self, host=web_cfg.get('host', '127.0.0.1'),
                                port=int(web_cfg.get('port', 8080)))
            self.web_ui.start()
        except Exception as e:
            logger.warning(f"Web settings UI could not start: {e}")
            self.web_ui = None

    def _wifi_changed(self, message: str):
        logger.info(message)
        if self.ui:
            self.ui.show_message(message, duration=15.0)

    def _apply_log_level(self):
        level = str(self.config.get('system.log_level', 'INFO')).upper()
        logging.getLogger().setLevel(getattr(logging, level, logging.INFO))

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals (the main loop does the actual shutdown)."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.should_stop = True
        self._stop_event.set()

    # -----------------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------------

    def apply_config_changes(self):
        """Push edited settings into the running components (no restart needed)."""
        try:
            if self.scheduler:
                sched = self.config.get_schedule()
                self.scheduler.set_schedule(sched['start_time'], sched['stop_time'], sched['enabled'])
            if self.email_sender:
                self.email_sender.update_config(self.config.get_email_config())
            if self.email_queue:
                system_cfg = self.config.get_system_config()
                self.email_queue.max_retries = int(system_cfg.get('email_retry_attempts', 3))
                self.email_queue.retry_delay = int(system_cfg.get('email_retry_delay', 60))
            self._apply_log_level()
            # Detection thresholds are picked up by the audio thread on its next chunk
            self._detection_dirty = True
            logger.info("Settings applied")
        except Exception as e:
            logger.error(f"Failed to apply settings: {e}")

    def _detector_params(self) -> Dict[str, float]:
        audio_cfg = self.config.get_audio_config()
        return {
            "min_duration": float(audio_cfg.get('min_vocalization_duration', 0.5)),
            "max_duration": float(audio_cfg.get('max_vocalization_duration', 10.0)),
            "silence_duration": float(audio_cfg.get('silence_duration', 0.3)),
            "energy_threshold": float(audio_cfg.get('energy_threshold', 0.001)),
        }

    def set_clock(self, dt: datetime) -> str:
        """Set the RTC and system clock. Returns a human-readable result."""
        results = []
        if self.rtc:
            try:
                self.rtc.set_datetime(dt)
                results.append("RTC set")
            except Exception as e:
                results.append(f"RTC failed: {e}")
        try:
            set_system_time(dt)
            results.append("system clock set")
        except Exception as e:
            results.append("system clock not set (needs passwordless sudo)" if self.rtc
                           else f"system clock not set: {e}")
        return "; ".join(results)

    # -----------------------------------------------------------------------
    # UI callbacks
    # -----------------------------------------------------------------------

    def _ui_start_monitoring(self):
        """Start monitoring from UI."""
        if self.is_monitoring:
            return "Already monitoring"
        return self.start_monitoring(manual=True)

    def _ui_stop_monitoring(self):
        """Stop monitoring from UI."""
        return self.stop_monitoring(manual=True)

    def _ui_test_email(self) -> str:
        """Send a test email from UI."""
        logger.info("Sending test email...")
        if not self.email_sender:
            return "Email sender not initialized"

        return self.email_sender.send_test_email()

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    def start_monitoring(self, manual: bool = False) -> str:
        """
        Start real-time monitoring.

        Args:
            manual: Started by a person (UI/web) rather than by the schedule

        Returns:
            Empty string on success, error message on failure
        """
        with self._monitor_lock:
            if self.is_monitoring:
                logger.warning("Monitoring already active")
                return "Monitoring already active"

            logger.info("=" * 60)
            logger.info("Starting Real-Time Monitoring" + (" (manual)" if manual else ""))
            logger.info("=" * 60)

            try:
                # Initialize activity tracker
                self.activity_tracker = ActivityTracker(start_time=datetime.now())
                self.session_stats = {"segments": 0, "low_confidence": 0}

                # Initialize audio capture
                audio_cfg = self.config.get_audio_config()
                self.detector = VocalizationDetector(sample_rate=audio_cfg['sample_rate'],
                                                     **self._detector_params())
                self._detection_dirty = False
                self._capture_rate = int(audio_cfg['sample_rate'])
                self.audio_capture = I2SAudioCapture(
                    sample_rate=audio_cfg['sample_rate'],
                    channels=audio_cfg['channels'],
                    chunk_duration=audio_cfg['chunk_duration']
                )

                # Set up audio callback for real-time processing
                self.audio_capture.set_callback(self._process_audio_chunk)

                # Start audio capture
                self.audio_capture.start(blocking=False)
                self.is_monitoring = True
                self.manual_mode = manual and not (self.scheduler and self.scheduler.is_within_schedule())
                self._skip_current_window = False
                self._last_start_attempt = 0.0  # Reset so next failure can retry after delay
                self.last_error = None

                logger.info("Monitoring started successfully")
                if self.ui:
                    self.ui.show_message("Monitoring Active", duration=2.0)
                return ""

            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                self.last_error = f"Start failed: {e}"
                if self.ui:
                    self.ui.show_message(f"Start Error: {str(e)[:40]}", duration=5.0)
                self.is_monitoring = False
                self.activity_tracker = None
                return str(e)

    def _process_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """
        Process audio chunk in real-time.

        Args:
            audio_chunk: Audio data
            timestamp: Chunk timestamp
        """
        try:
            if not self.feature_extractor or not self.classifier or not self.detector:
                logger.warning("Processing skipped: classifier pipeline not initialized")
                return

            if self._detection_dirty:
                self._detection_dirty = False
                self.detector.configure(**self._detector_params())

            # Only sounds louder than the energy threshold are classified. The model
            # always picks one of its classes, even for silence, so classifying every
            # chunk would count background noise as bat calls.
            segments = self.detector.process_chunk(audio_chunk, timestamp)

            if audio_chunk.size:
                self.audio_level = {
                    "rms": float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))),
                    "peak": float(np.max(np.abs(audio_chunk))),
                    "loudest": self.detector.last_rms,
                    "floor": self.detector.noise_floor,
                    "updated": time.time(),
                }

            for segment, start_ts in segments:
                self._classify_segment(segment, start_ts)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    def _classify_segment(self, segment: np.ndarray, start_ts: float):
        if not self.feature_extractor or not self.classifier:
            return
        self.session_stats["segments"] += 1

        capture_rate = self._capture_rate
        if capture_rate != self.model_sample_rate:
            g = gcd(capture_rate, self.model_sample_rate)
            segment = resample_poly(segment, self.model_sample_rate // g, capture_rate // g).astype(np.float32)

        # Extract features
        features = self.feature_extractor.extract_features(segment)

        # Classify
        predicted_class, confidence, _probabilities = self.classifier.predict(features)

        # Only track high-confidence predictions
        threshold = float(self.config.get('model.confidence_threshold', 0.6))
        tracker = self.activity_tracker
        if predicted_class.lower() in ("other", "noise", "background"):
            # A non-bat class (if the model has one): not a vocalization
            self.session_stats["low_confidence"] += 1
            logger.debug(f"Ignored non-bat sound ({confidence:.2%})")
        elif confidence >= threshold and tracker:
            tracker.add_classification(
                predicted_class,
                confidence,
                datetime.fromtimestamp(start_ts)
            )
            logger.debug(f"Classified: {predicted_class} ({confidence:.2%})")
        else:
            self.session_stats["low_confidence"] += 1
            logger.debug(f"Ignored low-confidence {predicted_class} ({confidence:.2%})")

    def stop_monitoring(self, manual: bool = False) -> str:
        """Stop monitoring and send report."""
        with self._monitor_lock:
            if not self.is_monitoring:
                logger.warning("Monitoring not active")
                return "Monitoring not active"

            logger.info("Stopping monitoring...")

            try:
                # Stop audio capture (processes audio still in the queue)
                if self.audio_capture:
                    self.audio_capture.stop()
                    self.audio_capture = None

                # Classify a vocalization that was still in progress
                if self.detector:
                    for segment, start_ts in self.detector.flush():
                        try:
                            self._classify_segment(segment, start_ts)
                        except Exception as e:
                            logger.error(f"Could not classify the final sound: {e}")

                self.is_monitoring = False
                self.manual_mode = False
                if manual and self.scheduler and self.scheduler.is_within_schedule():
                    # Stay stopped for the rest of tonight's window
                    self._skip_current_window = True

                tracker = self.activity_tracker
                if tracker:
                    tracker.finish()
                    self.last_session = {
                        "start": tracker.start_time.isoformat(timespec='seconds'),
                        "end": tracker.end_time.isoformat(timespec='seconds') if tracker.end_time else None,
                        "total": tracker.get_total_vocalizations(),
                        "by_class": self._class_counts(tracker),
                    }

                # Generate and send report
                if tracker and tracker.get_total_vocalizations() > 0:
                    self._send_report()
                else:
                    logger.info("No vocalizations detected, skipping report")
                    self.activity_tracker = None

                if self.ui:
                    self.ui.show_message("Monitoring Stopped", duration=2.0)
                return ""

            except Exception as e:
                logger.error(f"Error stopping monitoring: {e}")
                self.is_monitoring = False
                return str(e)

    @staticmethod
    def _class_counts(tracker: ActivityTracker) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _ts, cls, _conf in list(tracker.classifications):
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def _send_report(self):
        """Generate graphs and send email report."""
        logger.info("Generating activity report...")

        try:
            if not self.activity_tracker:
                logger.warning("No activity tracker available for report generation")
                return
            if not self.email_queue:
                logger.warning("Email queue is not initialized")
                return

            # Generate graphs
            REPORT_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timeline_path = str(REPORT_DIR / f"timeline_{timestamp}.png")
            breakdown_path = str(REPORT_DIR / f"breakdown_{timestamp}.png")

            # Close the figures after saving, or matplotlib keeps them in memory forever.
            # If drawing fails, the report is still sent - just without graphs.
            graph_paths = []
            for make, path in ((self.activity_tracker.generate_timeline_graph, timeline_path),
                               (self.activity_tracker.generate_breakdown_graph, breakdown_path)):
                try:
                    plt.close(make(path))
                    graph_paths.append(path)
                except Exception as e:
                    logger.error(f"Could not draw graph {path}: {e}")
                    plt.close("all")

            # Print summary to log
            summary = self.activity_tracker.generate_summary_table()
            logger.info(f"\n{summary}")

            # Queue email (files will be cleaned up after successful send)
            recording_date = self.activity_tracker.start_time.strftime("%m/%d/%Y")
            self.email_queue.enqueue(
                recording_date,
                self.activity_tracker,
                graph_paths=graph_paths,
                summary_text=summary
            )

            logger.info("Report queued for email delivery")
            if self.ui:
                self.ui.show_message("Report Queued", duration=2.0)

            # The email queue holds its own reference to the tracker for sending
            self.activity_tracker = None

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            self.last_error = f"Report failed: {e}"
            if self.ui:
                self.ui.show_message(f"Report Error: {str(e)[:40]}", duration=5.0)

    # -----------------------------------------------------------------------
    # Status (used by the web UI)
    # -----------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Snapshot of everything the dashboard shows."""
        status: Dict[str, Any] = {
            "monitoring": self.is_monitoring,
            "mode": ("manual" if self.manual_mode else "scheduled") if self.is_monitoring else "idle",
            "skip_window": self._skip_current_window,
            "last_error": self.last_error,
            "hostname": socket.gethostname(),
            "uptime_s": int(time.time() - self.started_at),
            "tft_display": self.ui is not None,
        }
        if self.classifier:
            status["model"] = {"file": self.classifier.model_path.name,
                               "count_correction": self.classifier.class_bias is not None,
                               "classes": self.classifier.class_names}

        tracker = self.activity_tracker
        if self.is_monitoring and tracker:
            status["session"] = {
                "start": tracker.start_time.isoformat(timespec='seconds'),
                "duration_s": int(tracker.get_recording_duration().total_seconds()),
                "total": tracker.get_total_vocalizations(),
                "by_class": self._class_counts(tracker),
                "want_food_rate": round(tracker.get_want_food_rate("Straws"), 2),
                "segments": self.session_stats["segments"],
                "low_confidence": self.session_stats["low_confidence"],
                "recent": [
                    {"time": ts.strftime('%H:%M:%S'), "class": cls, "confidence": round(conf, 3)}
                    for ts, cls, conf in tracker.get_recent_classifications(15)
                ],
            }
        status["last_session"] = self.last_session

        level = dict(self.audio_level)
        level["threshold"] = float(self.config.get('audio.energy_threshold', 0.001))
        level["fresh"] = bool(level["updated"] and time.time() - level["updated"] < 10 and self.is_monitoring)
        status["audio"] = level

        if self.scheduler:
            try:
                status["schedule"] = self.scheduler.get_schedule_info()
            except Exception as e:
                status["schedule"] = {"error": str(e)}

        clock: Dict[str, Any] = {"system_time": datetime.now().isoformat(timespec='seconds'),
                                 "rtc_present": self.rtc is not None}
        if self.rtc:
            try:
                clock["rtc_time"] = self.rtc.get_datetime().isoformat(timespec='seconds')
                clock["rtc_temp_c"] = round(self.rtc.get_temperature(), 2)
            except Exception as e:
                clock["rtc_error"] = str(e)
        status["clock"] = clock

        email_cfg = self.config.get_email_config()
        status["email"] = {
            "configured": bool(email_cfg.get('sender_email') and email_cfg.get('sender_password')),
            "recipients": len(email_cfg.get('recipients', [])),
            "queued": self.email_queue.get_queue_size() if self.email_queue else 0,
        }

        status["system"] = self._system_health()
        status["network"] = wifi_manager.current_status()
        return status

    @staticmethod
    def _system_health() -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                health["cpu_temp_c"] = round(int(f.read().strip()) / 1000, 1)
        except Exception:
            pass
        try:
            health["load"] = round(os.getloadavg()[0], 2)
        except (AttributeError, OSError):
            pass
        try:
            with open('/proc/meminfo') as f:
                info = {line.split(':')[0]: int(line.split()[1]) for line in f if ':' in line}
            health["mem_used_pct"] = round(100 * (1 - info["MemAvailable"] / info["MemTotal"]), 1)
        except Exception:
            pass
        try:
            usage = shutil.disk_usage(SCRIPT_DIR)
            health["disk_free_gb"] = round(usage.free / 1e9, 2)
            health["disk_used_pct"] = round(100 * usage.used / usage.total, 1)
        except Exception:
            pass
        return health

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        """Main run loop."""
        logger.info("Entering main run loop")

        try:
            while not self.should_stop:
                try:
                    self._tick()
                except Exception:
                    logger.exception("Error in main loop (continuing)")

                # Sleep briefly (wakes immediately on shutdown signal)
                self._stop_event.wait(1.0)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            self.shutdown()

    def _tick(self):
        """One pass of the main loop: start/stop monitoring on schedule."""
        if not self.scheduler:
            logger.error("Scheduler not initialized")
            return

        if self.scheduler.is_within_schedule():
            if self.is_monitoring:
                # A manual session that runs into the window becomes tonight's session
                self.manual_mode = False
            elif not self._skip_current_window:
                now = time.time()
                if now - self._last_start_attempt >= self._start_retry_delay:
                    logger.info("Within schedule window - starting monitoring")
                    self._last_start_attempt = now
                    self.start_monitoring()
                else:
                    remaining = int(self._start_retry_delay - (now - self._last_start_attempt))
                    logger.debug(f"Waiting {remaining}s before retrying monitoring start")
        else:
            self._skip_current_window = False
            if self.is_monitoring and not self.manual_mode:
                logger.info("Outside schedule window - stopping monitoring")
                self.stop_monitoring()

    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down system...")

        # Stop monitoring (queues tonight's report)
        if self.is_monitoring:
            self.stop_monitoring()

        # Stop email queue (tries to deliver queued reports first)
        if self.email_queue:
            self.email_queue.stop(drain_timeout=60.0)

        if self.web_ui:
            self.web_ui.stop()

        if self.wifi_watchdog:
            self.wifi_watchdog.stopped = True

        # Clean up UI
        if self.ui:
            self.ui.cleanup()

        # Close RTC
        if self.rtc:
            self.rtc.close()

        logger.info("System shutdown complete")


def main():
    """Main entry point."""
    # Check if running on Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' not in model:
                logger.warning(f"Not running on Raspberry Pi (detected: {model.strip()})")
    except FileNotFoundError:
        logger.warning("Cannot detect hardware - /proc/device-tree/model not found")

    # Create and run system
    try:
        system = BatMonitorSystem()
        system.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
