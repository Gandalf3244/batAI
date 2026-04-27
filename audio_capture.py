"""
I2S audio capture for INMP441 microphone on Raspberry Pi.
Provides streaming audio interface for real-time bat vocalization capture.
"""

import numpy as np
from typing import Any, Callable, Optional
import queue
import threading
import time
from pathlib import Path
import logging

try:
    import sounddevice as _sd  # type: ignore[import-not-found]
except ImportError:
    _sd = None

sd: Any = _sd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class I2SAudioCapture:
    """
    Real-time I2S audio capture from INMP441 microphone.
    
    Hardware Connections (BCM pin numbering):
    - SCK (Serial Clock): GPIO 18 (BCM)
    - WS (Word Select/LRCLK): GPIO 19 (BCM)
    - SD (Serial Data): GPIO 21 (BCM)
    - VDD: 3.3V
    - GND: Ground
    - L/R: GND (for left channel)
    """
    
    def __init__(self, sample_rate: int = 48000, channels: int = 2,
                 chunk_duration: float = 1.0, buffer_size: int = 100):
        """
        Initialize I2S audio capture.
        
        Args:
            sample_rate: Sample rate in Hz (INMP441 supports up to 48kHz)
            channels: Number of channels (2 required for I2S hardware)
            chunk_duration: Duration of each audio chunk in seconds
            buffer_size: Maximum number of chunks to buffer
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.is_capturing = False
        self.stream = None
        self.capture_thread = None
        
        # Callbacks
        self.on_audio_chunk: Optional[Callable[[np.ndarray, float], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Statistics
        self.chunks_captured = 0
        self.chunks_dropped = 0
        self.start_time = None
        
        logger.info(f"I2S Audio Capture initialized: {sample_rate}Hz, {channels}ch, {chunk_duration}s chunks")
    
    def _find_i2s_device(self) -> Optional[int]:
        """
        Find I2S audio device.
        
        Returns:
            Device index or None if not found
        """
        if sd is None:
            raise RuntimeError("sounddevice module is not installed")

        devices = sd.query_devices()
        
        # Look for Google voiceHAT / I2S device
        i2s_keywords = ['googlevoicehat', 'google', 'voicehat', 'i2s', 'inmp441', 'mems', 'dmic', 'snd_rpi']
        
        for idx, device in enumerate(devices):
            device_name = device['name'].lower()
            if any(keyword in device_name for keyword in i2s_keywords):
                if device['max_input_channels'] > 0:
                    logger.info(f"Found I2S device: {device['name']} (index {idx})")
                    return idx
        
        # Fallback: use default input device
        default_device = sd.default.device[0]
        if default_device is not None:
            logger.warning(f"I2S device not found by name. Using default input device: {devices[default_device]['name']}")
            return default_device
        
        logger.error("No input audio device found!")
        return None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream (called from audio thread)."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Copy audio data and convert int32 to float32
        audio_chunk = indata.copy().astype(np.float32)
        
        # Normalize int32 to float32 range [-1, 1]
        audio_chunk = audio_chunk / 2147483648.0
        
        try:
            # Non-blocking put
            self.audio_queue.put_nowait((audio_chunk, time.time()))
            self.chunks_captured += 1
        except queue.Full:
            self.chunks_dropped += 1
            if self.chunks_dropped % 10 == 0:
                logger.warning(f"Audio buffer full! Dropped {self.chunks_dropped} chunks total")
    
    def _processing_loop(self):
        """Process audio chunks from queue."""
        logger.info("Audio processing loop started")
        
        while self.is_capturing:
            try:
                # Get chunk with timeout
                audio_chunk, timestamp = self.audio_queue.get(timeout=1.0)
                
                # Convert stereo to mono by taking left channel only
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk[:, 0]
                
                # Flatten to 1D array
                audio_chunk = audio_chunk.flatten()
                
                # Call callback if registered
                if self.on_audio_chunk:
                    try:
                        self.on_audio_chunk(audio_chunk, timestamp)
                    except Exception as e:
                        logger.error(f"Error in audio chunk callback: {e}")
                        if self.on_error:
                            self.on_error(e)
                
            except queue.Empty:
                # No data available, continue
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if self.on_error:
                    self.on_error(e)
        
        logger.info("Audio processing loop stopped")
    
    def start(self, blocking: bool = False):
        """
        Start audio capture.
        
        Args:
            blocking: If True, blocks until stop() is called. If False, runs in background.
        """
        if self.is_capturing:
            logger.warning("Audio capture already running")
            return
        
        # Find I2S device
        device_idx = self._find_i2s_device()
        if device_idx is None:
            raise RuntimeError("No audio input device available")
        
        self.is_capturing = True
        self.start_time = time.time()
        self.chunks_captured = 0
        self.chunks_dropped = 0
        
        # Start audio stream with int32 format (required by INMP441 via I2S)
        try:
            self.stream = sd.InputStream(
                device=device_idx,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                dtype='int32'
            )
            self.stream.start()
            logger.info(f"Audio stream started on device {device_idx}")
        except Exception as e:
            self.is_capturing = False
            raise RuntimeError(f"Failed to start audio stream: {e}")
        
        # Start processing thread
        self.capture_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Audio capture started")
        
        if blocking:
            try:
                while self.is_capturing:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()
    
    def stop(self):
        """Stop audio capture."""
        if not self.is_capturing:
            return
        
        logger.info("Stopping audio capture...")
        self.is_capturing = False
        
        # Stop stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait for processing thread
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
            self.capture_thread = None
        
        duration = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Audio capture stopped. Duration: {duration:.1f}s, "
                   f"Chunks: {self.chunks_captured}, Dropped: {self.chunks_dropped}")
        
        if self.chunks_captured > 0:
            drop_rate = (self.chunks_dropped / (self.chunks_captured + self.chunks_dropped)) * 100
            logger.info(f"Drop rate: {drop_rate:.2f}%")
    
    def set_callback(self, callback: Callable[[np.ndarray, float], None]):
        """
        Set callback for audio chunks.
        
        Args:
            callback: Function called with (audio_chunk, timestamp) for each chunk
        """
        self.on_audio_chunk = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set callback for errors."""
        self.on_error = callback
    
    def get_statistics(self) -> dict:
        """Get capture statistics."""
        duration = 0
        if self.start_time:
            duration = time.time() - self.start_time
        
        return {
            'is_capturing': self.is_capturing,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'chunk_duration': self.chunk_duration,
            'duration_seconds': duration,
            'chunks_captured': self.chunks_captured,
            'chunks_dropped': self.chunks_dropped,
            'queue_size': self.audio_queue.qsize(),
            'drop_rate_percent': (self.chunks_dropped / max(1, self.chunks_captured + self.chunks_dropped)) * 100
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class VocalizationDetector:
    """
    Detects bat vocalizations from audio stream using energy-based detection.
    Extracts individual vocalization segments from continuous audio.
    """
    
    def __init__(self, sample_rate: int = 44100, min_duration: float = 0.5,
                 max_duration: float = 10.0, silence_duration: float = 0.3,
                 energy_threshold: float = 0.001, padding: float = 0.1):
        """
        Initialize vocalization detector.
        
        Args:
            sample_rate: Audio sample rate
            min_duration: Minimum vocalization duration (seconds)
            max_duration: Maximum vocalization duration (seconds)
            silence_duration: Silence duration to end vocalization (seconds)
            energy_threshold: RMS energy threshold for detection
            padding: Padding to add before/after vocalization (seconds)
        """
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_duration = silence_duration
        self.energy_threshold = energy_threshold
        self.padding = padding
        
        # Detection parameters
        self.window_size = int(0.02 * sample_rate)  # 20ms windows
        self.hop_size = self.window_size // 2
        self.silence_frames_needed = int(silence_duration / (self.hop_size / sample_rate))
        self.max_frames = int(max_duration / (self.hop_size / sample_rate))
        self.padding_samples = int(padding * sample_rate)
        
        # State
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_start_sample = None
        self.consecutive_silence = 0
        self.current_duration_frames = 0
        self.total_samples_processed = 0
        
        # Callback for detected vocalizations
        self.on_vocalization: Optional[Callable[[Optional[np.ndarray], float, float], None]] = None
        
        # Statistics
        self.vocalizations_detected = 0
        
        logger.info(f"Vocalization detector initialized: {min_duration}-{max_duration}s, threshold={energy_threshold}")
    
    def process_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """
        Process audio chunk and detect vocalizations.
        
        Args:
            audio_chunk: Audio data array
            timestamp: Timestamp of chunk
        """
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Process in sliding windows
        while len(self.audio_buffer) >= self.window_size:
            window = self.audio_buffer[:self.window_size]
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(window ** 2))
            
            if rms > self.energy_threshold:
                # Sound detected
                if self.current_start_sample is None:
                    # Start new vocalization
                    self.current_start_sample = max(0, self.total_samples_processed - self.padding_samples)
                    self.current_duration_frames = 0
                else:
                    self.current_duration_frames += 1
                
                self.consecutive_silence = 0
                
                # Check if vocalization is too long
                if self.current_duration_frames >= self.max_frames:
                    self._end_vocalization()
            else:
                # Silence
                if self.current_start_sample is not None:
                    self.consecutive_silence += 1
                    
                    # Check if silence is long enough to end vocalization
                    if self.consecutive_silence >= self.silence_frames_needed:
                        self._end_vocalization()
            
            # Move to next hop
            self.audio_buffer = self.audio_buffer[self.hop_size:]
            self.total_samples_processed += self.hop_size
    
    def _end_vocalization(self):
        """End current vocalization and extract segment."""
        if self.current_start_sample is None:
            return
        
        end_sample = self.total_samples_processed + self.padding_samples
        duration = (end_sample - self.current_start_sample) / self.sample_rate
        
        # Check if duration is valid
        if duration >= self.min_duration:
            start_time = self.current_start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            self.vocalizations_detected += 1
            
            if self.on_vocalization:
                self.on_vocalization(None, start_time, end_time)
        
        # Reset state
        self.current_start_sample = None
        self.current_duration_frames = 0
        self.consecutive_silence = 0
    
    def set_callback(self, callback: Callable[[Optional[np.ndarray], float, float], None]):
        """Set callback for detected vocalizations."""
        self.on_vocalization = callback
    
    def reset(self):
        """Reset detector state."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_start_sample = None
        self.consecutive_silence = 0
        self.current_duration_frames = 0


if __name__ == "__main__":
    import sys
    
    print("Testing I2S Audio Capture...")
    print("Available audio devices:")
    print(sd.query_devices())
    
    chunk_count = [0]
    
    def audio_callback(chunk, timestamp):
        chunk_count[0] += 1
        rms = np.sqrt(np.mean(chunk ** 2))
        print(f"Chunk {chunk_count[0]}: {len(chunk)} samples, RMS: {rms:.6f}, Time: {timestamp:.2f}")
    
    try:
        capture = I2SAudioCapture(sample_rate=44100, channels=2, chunk_duration=1.0)
        capture.set_callback(audio_callback)
        
        print("\nStarting capture for 10 seconds...")
        capture.start()
        time.sleep(10)
        capture.stop()
        
        stats = capture.get_statistics()
        print(f"\nCapture statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✓ Audio capture test completed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
