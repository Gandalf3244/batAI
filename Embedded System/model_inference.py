"""
TFLite model inference for real-time bat vocalization classification.
Loads and runs the converted TFLite model with label decoding.
"""

import hashlib
import json
import numpy as np
import pickle

# Any of these TFLite runtimes can run the model; use whichever is installed
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found,no-redef]
    except ImportError:
        import tensorflow as _tf  # type: ignore[import-not-found]
        Interpreter = _tf.lite.Interpreter  # type: ignore[misc]

from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTIVE_CLASS_ORDER = [
    "Rods_Fighting",
    "Straws_Fighting",
    "Straws_Talking",
    "Straws_Want_Food",
]

CLASS_LABEL_MAP = {
    "rods_fighting": ("Rods", "Fighting"),
    "straws_fighting": ("Straws", "Fighting"),
    "straws_talking": ("Straws", "Talking"),
    "straws_want_food": ("Straws", "Want_Food"),
}


def apply_class_bias(probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Shift class probabilities by per-class log-offsets and renormalise.

    Used to correct a model trained with balanced class weights, which otherwise
    over-predicts the rare classes in real (mostly Straws-fighting) audio."""
    logp = np.log(np.clip(probabilities, 1e-9, 1.0)) + bias
    logp = logp - logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


def calibration_path(model_path) -> Path:
    """bat_model.tflite -> bat_model.calibration.json"""
    model_path = Path(model_path)
    return model_path.with_name(model_path.stem + ".calibration.json")


def file_sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_")


def _extract_species_behavior(class_name: str) -> Tuple[str, str]:
    normalized = _normalize_class_name(class_name)
    if normalized in CLASS_LABEL_MAP:
        return CLASS_LABEL_MAP[normalized]

    parts = class_name.replace("_", " ").split(maxsplit=1)
    species = parts[0] if parts else "Unknown"
    behavior = parts[1] if len(parts) > 1 else "Unknown"
    return species, behavior


class BatClassifier:
    """
    Real-time bat vocalization classifier using TFLite model.
    """
    
    def __init__(self, model_path: str, label_encoder_path: str):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to TFLite model file
            label_encoder_path: Path to label encoder pickle file
        """
        self.model_path = Path(model_path)
        self.label_encoder_path = Path(label_encoder_path)
        
        # Load model
        self.interpreter = self._load_model()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load label encoder
        self.label_encoder = self._load_label_encoder()
        self.class_names = [str(name) for name in self.label_encoder.classes_]
        self.logged_class_names = [name for name in ACTIVE_CLASS_ORDER if name in self.class_names]

        # Optional count correction that belongs to this exact model file
        self.class_bias = self._load_calibration()
        
        # Log model info
        logger.info(f"Model loaded: {self.model_path}")
        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Output shape: {self.output_details[0]['shape']}")
        logger.info(f"Classes: {len(self.logged_class_names) or len(self.class_names)} - {self.logged_class_names or self.class_names}")
        
        # Statistics
        self.total_inferences = 0
    
    def _load_model(self):
        """Load TFLite model using ai-edge-litert."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            interpreter = Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            logger.info("✓ Model loaded successfully")
            return interpreter
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_calibration(self) -> Optional[np.ndarray]:
        """Load <model>.calibration.json if present and made for this model file."""
        path = calibration_path(self.model_path)
        if not path.exists():
            return None
        try:
            cal = json.loads(path.read_text())
            if cal.get("model_sha1") and cal["model_sha1"] != file_sha1(self.model_path):
                logger.warning(f"Ignoring {path.name}: it was made for a different model file")
                return None
            bias = np.array([float(cal["class_bias"].get(c, 0.0)) for c in self.class_names], np.float32)
            logger.info("Count correction loaded: " + ", ".join(
                f"{c} {b:+.2f}" for c, b in zip(self.class_names, bias)))
            return bias
        except Exception as e:
            logger.warning(f"Ignoring {path.name}: {e}")
            return None

    def _load_label_encoder(self):
        """Load label encoder from pickle file."""
        if not self.label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found: {self.label_encoder_path}")
        
        with open(self.label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info(f"Label encoder loaded: {len(label_encoder.classes_)} classes")
        return label_encoder
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify bat vocalization features.
        
        Args:
            features: Spectrogram features with shape matching model input
                     (expected for current model config: (1, 451, 120))
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities_dict)
        """
        # Verify input shape
        expected_shape = tuple(self.input_details[0]['shape'])
        if features.shape != expected_shape:
            raise ValueError(f"Input shape mismatch. Expected {expected_shape}, got {features.shape}")
        
        # Ensure correct dtype
        features = features.astype(self.input_details[0]['dtype'])
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output[0]  # Remove batch dimension
        if self.class_bias is not None:
            probabilities = apply_class_bias(probabilities, self.class_bias)
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create probability dictionary
        prob_dict = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        self.total_inferences += 1
        
        return predicted_class, confidence, prob_dict
    
    def predict_batch(self, features_batch: np.ndarray) -> list:
        """
        Classify multiple vocalizations.
        
        Args:
            features_batch: Batch of features
            
        Returns:
            List of (predicted_class, confidence, probabilities) tuples
        """
        results = []
        
        for i in range(len(features_batch)):
            # Extract single sample and add batch dimension
            features = np.expand_dims(features_batch[i], axis=0)
            result = self.predict(features)
            results.append(result)
        
        return results
    
    def get_species_from_class(self, class_name: str) -> str:
        """
        Extract species name from class label.
        
        Examples:
            "straws_want_food" -> "Straws"
            "rods_fighting" -> "Rods"
        """
        species, _ = _extract_species_behavior(class_name)
        return species
    
    def get_behavior_from_class(self, class_name: str) -> str:
        """
        Extract behavior from class label.
        
        Examples:
            "straws_want_food" -> "Want_Food"
            "rods_fighting" -> "Fighting"
        """
        _, behavior = _extract_species_behavior(class_name)
        return behavior
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            'total_inferences': self.total_inferences,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_path': str(self.model_path),
            'input_shape': self.input_details[0]['shape'].tolist(),
            'output_shape': self.output_details[0]['shape'].tolist()
        }


class RealtimeClassificationPipeline:
    """
    Complete pipeline for real-time classification.
    Combines feature extraction and model inference.
    """
    
    def __init__(self, model_path: str, label_encoder_path: str,
                 sample_rate: int = 44100):
        """
        Initialize pipeline.
        
        Args:
            model_path: Path to TFLite model
            label_encoder_path: Path to label encoder
            sample_rate: Audio sample rate
        """
        from feature_extraction import SpectrogramExtractor
        
        # Initialize components
        self.classifier = BatClassifier(model_path, label_encoder_path)
        model_input_shape = tuple(self.classifier.input_details[0]['shape'])
        self.feature_extractor = SpectrogramExtractor(
            sample_rate=sample_rate,
            n_mels=120,
            n_fft=2048,
            hop_length=512,
            target_length=int(model_input_shape[1]),
            model_feature_dim=int(model_input_shape[2]),
        )
        
        logger.info("Real-time classification pipeline initialized")
    
    def process_audio(self, audio: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Process audio and classify.
        
        Args:
            audio: Audio samples
            
        Returns:
            Classification results (class, confidence, probabilities)
        """
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Classify
        return self.classifier.predict(features)
    
    def process_audio_file(self, audio_path: str, offset: float = 0.0,
                          duration: Optional[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Process audio file and classify.
        
        Args:
            audio_path: Path to audio file
            offset: Start time in seconds
            duration: Duration to read (None = all)
            
        Returns:
            Classification results
        """
        # Extract features from file
        features = self.feature_extractor.extract_from_file(audio_path, offset, duration)
        
        # Classify
        return self.classifier.predict(features)


def test_classifier():
    """Test the classifier with random data."""
    print("Testing Bat Classifier...")
    
    # Check if model files exist
    model_path = "12_29_both_species.tflite"
    label_encoder_path = "label_encoder.pkl"
    
    if not Path(model_path).exists():
        print(f"✗ Model file not found: {model_path}")
        return
    
    if not Path(label_encoder_path).exists():
        print(f"✗ Label encoder not found: {label_encoder_path}")
        return
    
    try:
        # Initialize classifier
        classifier = BatClassifier(model_path, label_encoder_path)
        
        # Create random input matching model shape
        input_shape = tuple(classifier.input_details[0]['shape'])
        print(f"Model input shape: {input_shape}")
        
        random_features = np.random.random(input_shape).astype(np.float32)
        
        # Run prediction
        print("\nRunning inference...")
        predicted_class, confidence, probabilities = classifier.predict(random_features)
        
        print(f"\nPredicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nAll class probabilities:")
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name:30s}: {prob:6.2%}")
        
        # Test species/behavior extraction
        species = classifier.get_species_from_class(predicted_class)
        behavior = classifier.get_behavior_from_class(predicted_class)
        print(f"\nExtracted - Species: {species}, Behavior: {behavior}")
        
        # Statistics
        stats = classifier.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            if key != 'class_names':
                print(f"  {key}: {value}")
        
        print("\n✓ Classifier test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_classifier()
