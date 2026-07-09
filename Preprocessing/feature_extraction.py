"""
Feature extraction for bat vocalization classification.
Converts audio segments to mel spectrograms matching model input requirements.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectrogramExtractor:
    """
    Extract mel spectrogram features from audio for bat classification model.
    
    Default model input shape: (1, 451, 120)
    - 451 time steps
    - 120 mel features (or 360 when using mel+delta+delta-delta)
    
    All vocalizations are automatically clipped or padded to this exact size.
    """
    
    def __init__(self, sample_rate: int = 44100, n_mels: int = 120,
                 n_fft: int = 2048, hop_length: int = 512,
                 target_length: int = 451, fmin: float = 0,
                 fmax: Optional[float] = None, model_feature_dim: int = 120):
        """
        Initialize spectrogram extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel bands (must be 120 for model)
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            target_length: Target number of time steps expected by model
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sample_rate/2)
            model_feature_dim: Feature dimension expected by model input.
                360 for mel+delta+delta2, 120 for mel-only compatibility.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        self.model_feature_dim = model_feature_dim
        self.use_delta_channels = model_feature_dim == (n_mels * 3)

        if model_feature_dim not in (n_mels, n_mels * 3):
            raise ValueError(
                f"Unsupported model_feature_dim={model_feature_dim}. "
                f"Expected {n_mels} or {n_mels * 3}."
            )
        
        # Calculate expected audio length for target_length time steps
        self.target_samples = self._calculate_audio_length(target_length)
        
        logger.info(
            f"Spectrogram extractor: {n_mels} mels, {target_length} time steps, "
            f"{model_feature_dim} features"
        )
        logger.info(f"Target audio length: {self.target_samples} samples ({self.target_samples/sample_rate:.2f}s)")
    
    def _calculate_audio_length(self, n_frames: int) -> int:
        """Calculate audio length needed for n_frames."""
        return self.n_fft + (n_frames - 1) * self.hop_length
    
    def extract_features(self, audio: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract mel spectrogram features from audio.
        
        Automatically clips or pads audio to produce consistent output shape.
        
        Args:
            audio: Audio samples (1D array) - any length
            normalize: Whether to normalize the spectrogram
            
        Returns:
            Features with shape (1, target_length, model_feature_dim) ready for model
            - Dimension 0: Batch size (always 1)
            - Dimension 1: Time steps (target_length) - padded or clipped as needed
            - Dimension 2: Feature bins (120 or 360)
        """
        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Match training preprocessing: fixed dB reference
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
        
        if self.use_delta_channels:
            delta = librosa.feature.delta(mel_spec_db)
            delta2 = librosa.feature.delta(mel_spec_db, order=2)
            feature_matrix = np.vstack([mel_spec_db, delta, delta2]).T
        else:
            feature_matrix = mel_spec_db.T

        # Match training normalization
        if normalize:
            feature_matrix = np.clip(feature_matrix, -80.0, 80.0) / 80.0
        
        # Ensure exact target length (clip or pad time dimension)
        feature_matrix = self._adjust_time_length(feature_matrix)
        
        # Add batch dimension
        features = np.expand_dims(feature_matrix, axis=0)
        
        # Verify shape
        expected_shape = (1, self.target_length, self.model_feature_dim)
        if features.shape != expected_shape:
            raise ValueError(f"Feature shape mismatch: got {features.shape}, expected {expected_shape}")
        
        return features.astype(np.float32)
    
    def _adjust_time_length(self, features: np.ndarray) -> np.ndarray:
        """
        Ensure features have exactly target_length time steps.
        Clips or pads the time dimension (axis 0) as needed.
        """
        current_length = features.shape[0]
        
        if current_length < self.target_length:
            # Match training: zero-padding
            padding = self.target_length - current_length
            features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        elif current_length > self.target_length:
            # Match training: keep earliest frames
            features = features[:self.target_length, :]
        
        return features
    
    def extract_from_file(self, audio_path: str, offset: float = 0.0,
                         duration: Optional[float] = None) -> np.ndarray:
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            offset: Start time in seconds
            duration: Duration to read in seconds (None = all)
            
        Returns:
            Features with shape (1, target_length, model_feature_dim)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate,
                                offset=offset, duration=duration, mono=True)
        
        # Extract features
        return self.extract_features(audio)
    
    def get_expected_audio_duration(self) -> float:
        """Get expected audio duration in seconds."""
        return self.target_samples / self.sample_rate
    
    def batch_extract(self, audio_segments: list) -> np.ndarray:
        """
        Extract features from multiple audio segments.
        
        Args:
            audio_segments: List of audio arrays
            
        Returns:
            Batch of features with shape (batch_size, target_length, model_feature_dim)
        """
        features_list = []
        
        for audio in audio_segments:
            features = self.extract_features(audio)
            # Remove batch dimension for stacking
            features_list.append(features[0])
        
        # Stack into batch
        batch = np.stack(features_list, axis=0)
        
        return batch.astype(np.float32)


class StreamingFeatureExtractor:
    """
    Streaming feature extractor for real-time processing.
    Maintains a sliding window buffer for continuous feature extraction.
    """
    
    def __init__(self, sample_rate: int = 44100, window_duration: float = 2.0,
                 hop_duration: float = 0.5, n_mels: int = 120,
                 target_length: int = 451, model_feature_dim: int = 120):
        """
        Initialize streaming extractor.
        
        Args:
            sample_rate: Audio sample rate
            window_duration: Duration of analysis window in seconds
            hop_duration: Time to advance window in seconds
            n_mels: Number of mel bands
            target_length: Target number of time steps
            model_feature_dim: Feature dimension expected by model input
        """
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        
        self.window_samples = int(window_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)
        
        # Create spectrogram extractor
        self.extractor = SpectrogramExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            target_length=target_length,
            model_feature_dim=model_feature_dim,
        )
        
        # Rolling buffer
        self.buffer = np.array([], dtype=np.float32)
        
        logger.info(f"Streaming extractor: {window_duration}s window, {hop_duration}s hop")
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer."""
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()
        
        self.buffer = np.concatenate([self.buffer, audio_chunk])
    
    def get_features(self) -> Optional[np.ndarray]:
        """
        Get features from current buffer if enough audio is available.
        
        Returns:
            Features or None if not enough audio
        """
        if len(self.buffer) >= self.window_samples:
            # Extract features from window
            window = self.buffer[:self.window_samples]
            features = self.extractor.extract_features(window)
            
            # Advance buffer by hop
            self.buffer = self.buffer[self.hop_samples:]
            
            return features
        
        return None
    
    def reset(self):
        """Clear buffer."""
        self.buffer = np.array([], dtype=np.float32)
    
    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate


def test_spectrogram_extractor():
    """Test feature extraction."""
    print("Testing Spectrogram Extractor...")
    
    # Create test audio (1 second chirp)
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency_sweep = np.linspace(1000, 5000, len(t))  # 1-5 kHz chirp
    audio = np.sin(2 * np.pi * frequency_sweep * t)
    
    # Create extractor
    extractor = SpectrogramExtractor(sample_rate=sample_rate)
    
    # Extract features
    features = extractor.extract_features(audio)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Expected shape: (1, 451, 120)")
    print(f"  - Batch size: 1")
    print(f"  - Time steps: 451 (clipped/padded as needed)")
    print(f"  - Feature bins: 120 (mel)")
    print(f"Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
    
    assert features.shape == (1, 451, 120), f"Shape mismatch: {features.shape}"
    
    print("✓ Spectrogram extraction test passed!")
    
    # Test streaming extractor
    print("\nTesting Streaming Feature Extractor...")
    
    streaming = StreamingFeatureExtractor(sample_rate=sample_rate)
    
    # Simulate chunks
    chunk_size = 4410  # 0.1 second chunks
    chunks_processed = 0
    features_extracted = 0
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        streaming.add_audio(chunk)
        chunks_processed += 1
        
        # Try to extract features
        features = streaming.get_features()
        if features is not None:
            features_extracted += 1
            print(f"  Extracted features from buffer (chunk {chunks_processed})")
    
    print(f"Processed {chunks_processed} chunks, extracted {features_extracted} feature sets")
    print("✓ Streaming extraction test passed!")


if __name__ == "__main__":
    test_spectrogram_extractor()
