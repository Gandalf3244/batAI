"""
I2S audio capture for INMP441 microphone on Raspberry Pi.
Provides streaming audio interface for real-time bat vocalization capture.
"""

import numpy as np
from typing import Any, Callable, List, Optional, Tuple
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
        
        # Convert int32 to float32 in range [-1, 1] (astype copies the buffer)
        audio_chunk = indata.astype(np.float32) / 2147483648.0
        
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
        
        # Keep going after stop() until already-captured audio has been handled
        while self.is_capturing or not self.audio_queue.empty():
            try:
                # Get chunk with timeout
                audio_chunk, timestamp = self.audio_queue.get(timeout=0.5)
                
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
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
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

    Segments may span chunk boundaries; audio is kept only as long as needed
    (the open segment plus a little pre-roll padding).
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
        self.window_size = int(0.02 * sample_rate)  # 20ms windows
        self.hop_size = self.window_size // 2
        self.configure(min_duration, max_duration, silence_duration, energy_threshold, padding)
        
        # Callback for detected vocalizations: (audio, start_timestamp, end_timestamp)
        self.on_vocalization: Optional[Callable[[np.ndarray, float, float], None]] = None
        
        # Statistics
        self.vocalizations_detected = 0
        
        self.reset()
        logger.info(f"Vocalization detector initialized: {min_duration}-{max_duration}s, threshold={energy_threshold}")

    def configure(self, min_duration: float, max_duration: float, silence_duration: float,
                  energy_threshold: float, padding: float = 0.1):
        """Update detection parameters (safe to call between chunks)."""
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.silence_duration = float(silence_duration)
        self.energy_threshold = float(energy_threshold)
        self.padding = float(padding)

        self.min_samples = int(self.min_duration * self.sample_rate)
        self.max_samples = max(self.window_size, int(self.max_duration * self.sample_rate))
        self.silence_samples = int(self.silence_duration * self.sample_rate)
        self.padding_samples = int(self.padding * self.sample_rate)
    
    def process_chunk(self, audio_chunk: np.ndarray, timestamp: float) -> List[Tuple[np.ndarray, float]]:
        """
        Process audio chunk and detect vocalizations.
        
        Args:
            audio_chunk: Audio data array (mono)
            timestamp: Wall-clock time at the END of the chunk (as given by I2SAudioCapture)

        Returns:
            List of (segment_audio, segment_start_timestamp) for completed vocalizations
        """
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).ravel()
        if self._time_origin is None:
            self._time_origin = timestamp - len(audio_chunk) / self.sample_rate

        self._buffer = np.concatenate([self._buffer, audio_chunk])
        buffer_end = self._buffer_start + len(self._buffer)
        segments: List[Tuple[np.ndarray, float]] = []

        n_frames = (buffer_end - self._next_frame - self.window_size) // self.hop_size + 1
        if n_frames > 0:
            rel = self._next_frame - self._buffer_start
            span = self._buffer[rel:rel + (n_frames - 1) * self.hop_size + self.window_size]
            frames = np.lib.stride_tricks.sliding_window_view(span, self.window_size)[::self.hop_size]
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            self.last_rms = float(rms.max())
            self.noise_floor = float(np.percentile(rms, 20))

            for i, loud in enumerate(rms > self.energy_threshold):
                frame_start = self._next_frame + i * self.hop_size
                frame_end = frame_start + self.window_size
                if loud:
                    if self._seg_start is None:
                        self._seg_start = max(self._buffer_start, frame_start - self.padding_samples)
                    self._last_active = frame_end
                    if frame_end - self._seg_start >= self.max_samples:
                        self._emit(frame_end, segments)
                elif self._seg_start is not None and frame_start - self._last_active >= self.silence_samples:
                    self._emit(min(self._last_active + self.padding_samples, frame_end), segments)

            self._next_frame += n_frames * self.hop_size

        # Drop audio that can no longer be part of a segment
        keep_from = self._seg_start if self._seg_start is not None else self._next_frame - self.padding_samples
        keep_from = max(self._buffer_start, min(keep_from, self._next_frame))
        self._buffer = self._buffer[keep_from - self._buffer_start:]
        self._buffer_start = keep_from

        if self.on_vocalization:
            for seg, start_ts in segments:
                self.on_vocalization(seg, start_ts, start_ts + len(seg) / self.sample_rate)
        return segments

    def flush(self) -> List[Tuple[np.ndarray, float]]:
        """End any open vocalization (e.g. when monitoring stops) and return it."""
        segments: List[Tuple[np.ndarray, float]] = []
        if self._seg_start is not None:
            buffer_end = self._buffer_start + len(self._buffer)
            self._emit(min(self._last_active + self.padding_samples, buffer_end), segments)
        return segments
    
    def _emit(self, end_sample: int, segments: List[Tuple[np.ndarray, float]]):
        """End current vocalization and extract segment."""
        if self._seg_start is None:
            return
        
        if end_sample - self._seg_start >= self.min_samples:
            seg = self._buffer[self._seg_start - self._buffer_start:end_sample - self._buffer_start].copy()
            start_ts = (self._time_origin or 0.0) + self._seg_start / self.sample_rate
            segments.append((seg, start_ts))
            self.vocalizations_detected += 1
        
        # Reset state
        self._seg_start = None
        self._last_active = 0
    
    def set_callback(self, callback: Callable[[np.ndarray, float, float], None]):
        """Set callback for detected vocalizations."""
        self.on_vocalization = callback
    
    def reset(self):
        """Reset detector state."""
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_start = 0       # absolute sample index of _buffer[0]
        self._next_frame = 0         # absolute sample index of next analysis frame
        self._seg_start: Optional[int] = None
        self._last_active = 0        # absolute sample index where the last loud frame ended
        self._time_origin: Optional[float] = None
        self.last_rms = 0.0          # loudest 20ms frame in the last chunk
        self.noise_floor = 0.0       # typical quiet-frame level in the last chunk


if __name__ == "__main__":
    import sys
    
    print("Testing I2S Audio Capture...")
    if sd is None:
        print("✗ sounddevice is not installed")
        sys.exit(1)
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
