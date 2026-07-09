#!/usr/bin/env python3
"""
Real-Time Bat Activity Monitoring System
Main application for Raspberry Pi Zero 2 W

Captures audio from I2S microphone, classifies bat vocalizations in real-time,
tracks activity, and sends email reports - all without storing audio files.
"""

import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

# Import all components
from config_manager import ConfigManager
from audio_capture import I2SAudioCapture
from feature_extraction import SpectrogramExtractor
from model_inference import BatClassifier
from activity_tracker import ActivityTracker
from email_sender import EmailSender, EmailQueue
from rtc_scheduler import DS3231, ScheduleManager
from ui_controller import UIController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bat_monitor.log'),
        logging.StreamHandler()
    ]
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
        logger.info("Configuration loaded")
        
        # Initialize components
        self.rtc: Optional[DS3231] = None
        self.scheduler: Optional[ScheduleManager] = None
        self.ui: Optional[UIController] = None
        self.audio_capture: Optional[I2SAudioCapture] = None
        self.feature_extractor: Optional[SpectrogramExtractor] = None
        self.classifier: Optional[BatClassifier] = None
        self.activity_tracker: Optional[ActivityTracker] = None
        self.email_sender: Optional[EmailSender] = None
        self.email_queue: Optional[EmailQueue] = None
        
        # System state
        self.is_monitoring = False
        self.should_stop = False
        self._last_start_attempt = 0.0
        self._start_retry_delay = 30.0  # seconds before retrying after a failed start
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            # Initialize RTC and scheduler
            logger.info("Initializing RTC...")
            self.rtc = DS3231()
            self.scheduler = ScheduleManager(self.rtc)
            
            # Load schedule from config
            schedule_cfg = self.config.get_schedule()
            self.scheduler.set_schedule(
                schedule_cfg['start_time'],
                schedule_cfg['stop_time'],
                schedule_cfg['enabled']
            )
            
            # Initialize UI
            logger.info("Initializing UI...")
            self.ui = UIController(self.config)
            self.ui.on_start_monitoring = self._ui_start_monitoring
            self.ui.on_test_email = self._ui_test_email
            
            # Initialize model
            logger.info("Loading AI model...")
            model_cfg = self.config.get_model_config()
            self.classifier = BatClassifier(
                model_cfg['model_path'],
                model_cfg['label_encoder_path']
            )
            
            # Initialize feature extractor
            audio_cfg = self.config.get_audio_config()
            model_input_shape = tuple(self.classifier.input_details[0]['shape'])
            model_cfg_input_shape = tuple(model_cfg.get('input_shape', []))
            if model_cfg_input_shape and model_cfg_input_shape != model_input_shape:
                logger.warning(
                    "Config model.input_shape %s does not match TFLite input %s; using TFLite input shape",
                    model_cfg_input_shape,
                    model_input_shape,
                )
            self.feature_extractor = SpectrogramExtractor(
                sample_rate=audio_cfg['sample_rate'],
                n_mels=int(model_cfg.get('n_mels', 120)),
                n_fft=int(model_cfg.get('n_fft', 2048)),
                hop_length=int(model_cfg.get('hop_length', 512)),
                target_length=int(model_input_shape[1]),
                model_feature_dim=int(model_input_shape[2]),
            )
            
            # Initialize email sender
            logger.info("Initializing email sender...")
            email_cfg = self.config.get_email_config()
            self.email_sender = EmailSender(email_cfg)
            self.email_queue = EmailQueue(self.email_sender)
            self.email_queue.start()
            
            logger.info("System initialization complete!")
            if self.ui:
                self.ui.show_message("System Ready!", duration=2.0)
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            if self.ui:
                self.ui.show_message(f"Init Error: {str(e)[:40]}", duration=5.0)
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.should_stop = True
        self.stop_monitoring()
    
    def _ui_start_monitoring(self):
        """Start monitoring from UI."""
        if not self.ui:
            logger.warning("UI is not initialized")
            return

        if self.is_monitoring:
            self.ui.show_message("Already monitoring!", duration=2.0)
        else:
            self.start_monitoring()
    
    def _ui_test_email(self) -> str:
        """Test email from UI."""
        logger.info("Testing email connection...")
        if not self.email_sender:
            return "Email sender not initialized"

        error = self.email_sender.test_connection()
        
        if error:
            logger.error(f"Email test failed: {error}")
        else:
            logger.info("Email test successful")
        
        return error
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        logger.info("=" * 60)
        logger.info("Starting Real-Time Monitoring")
        logger.info("=" * 60)
        
        try:
            # Initialize activity tracker
            self.activity_tracker = ActivityTracker(start_time=datetime.now())
            
            # Initialize audio capture
            audio_cfg = self.config.get_audio_config()
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
            self._last_start_attempt = 0.0  # Reset so next failure can retry after delay
            
            logger.info("Monitoring started successfully")
            if self.ui:
                self.ui.show_message("Monitoring Active", duration=2.0)
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            if self.ui:
                self.ui.show_message(f"Start Error: {str(e)[:40]}", duration=5.0)
            self.is_monitoring = False
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """
        Process audio chunk in real-time.
        
        Args:
            audio_chunk: Audio data
            timestamp: Chunk timestamp
        """
        try:
            if not self.feature_extractor or not self.classifier:
                logger.warning("Processing skipped: classifier pipeline not initialized")
                return

            # Extract features
            features = self.feature_extractor.extract_features(audio_chunk)
            
            # Classify
            predicted_class, confidence, _probabilities = self.classifier.predict(features)
            
            # Only track high-confidence predictions
            if confidence > 0.6 and self.activity_tracker:  # Confidence threshold
                self.activity_tracker.add_classification(
                    predicted_class,
                    confidence,
                    datetime.fromtimestamp(timestamp)
                )
                
                logger.debug(f"Classified: {predicted_class} ({confidence:.2%})")
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring and send report."""
        if not self.is_monitoring:
            logger.warning("Monitoring not active")
            return
        
        logger.info("Stopping monitoring...")
        
        try:
            # Stop audio capture
            if self.audio_capture:
                self.audio_capture.stop()
            
            self.is_monitoring = False
            
            # Generate and send report
            if self.activity_tracker and self.activity_tracker.get_total_vocalizations() > 0:
                self._send_report()
            else:
                logger.info("No vocalizations detected, skipping report")
            
            if self.ui:
                self.ui.show_message("Monitoring Stopped", duration=2.0)
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timeline_path = f"timeline_{timestamp}.png"
            breakdown_path = f"breakdown_{timestamp}.png"
            
            self.activity_tracker.generate_timeline_graph(timeline_path)
            self.activity_tracker.generate_breakdown_graph(breakdown_path)
            
            # Print summary to log
            summary = self.activity_tracker.generate_summary_table()
            logger.info(f"\n{summary}")
            
            # Queue email (files will be cleaned up after successful send)
            recording_date = self.activity_tracker.start_time.strftime("%m/%d/%Y")
            self.email_queue.enqueue(
                recording_date,
                self.activity_tracker,
                graph_paths=[timeline_path, breakdown_path],
                summary_text=summary
            )
            
            logger.info("Report queued for email delivery")
            if self.ui:
                self.ui.show_message("Report Queued", duration=2.0)
            
            # Clear activity tracker data to free memory
            # The email queue holds a reference to the tracker for sending,
            # but we don't need to keep accumulating data after queuing
            logger.info("Clearing activity tracker data to save memory")
            self.activity_tracker = None
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            if self.ui:
                self.ui.show_message(f"Report Error: {str(e)[:40]}", duration=5.0)
    
    def run(self):
        """Main run loop."""
        logger.info("Entering main run loop")
        
        try:
            while not self.should_stop:
                # Check schedule
                if not self.scheduler:
                    logger.error("Scheduler not initialized")
                    time.sleep(1.0)
                    continue

                if self.scheduler.is_within_schedule():
                    if not self.is_monitoring:
                        now = time.time()
                        if now - self._last_start_attempt >= self._start_retry_delay:
                            logger.info("Within schedule window - starting monitoring")
                            self._last_start_attempt = now
                            self.start_monitoring()
                        else:
                            remaining = int(self._start_retry_delay - (now - self._last_start_attempt))
                            logger.debug(f"Waiting {remaining}s before retrying monitoring start")
                else:
                    if self.is_monitoring:
                        logger.info("Outside schedule window - stopping monitoring")
                        self.stop_monitoring()
                
                # Sleep briefly
                time.sleep(1.0)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down system...")
        
        # Stop monitoring
        if self.is_monitoring:
            self.stop_monitoring()
        
        # Stop email queue
        if self.email_queue:
            self.email_queue.stop()
        
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
