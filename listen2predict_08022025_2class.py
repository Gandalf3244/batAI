import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Parameters
SAMPLE_RATE = 20000
MAX_PAD_LEN = 79     # Must match the training script (training_07312025.py)
# IMPORTANT: Update this to the path of your newly trained 3-class model.
# Example: 'model_08022025_3class_acc92.keras'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_08022025_2class_acc86.keras')
LE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'le_3class.pkl')

model = load_model(MODEL_PATH)
# Load the label encoder to dynamically get class names
with open(LE_PATH, 'rb') as f:
    le = pickle.load(f)
CONTEXT_LABELS = le.classes_

def extract_features(audio):
    # This function must match the feature extraction process used for training the model.
    audio = audio.astype(float)
    # Extract MFCCs, deltas, and delta-deltas to match the training script's features.
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40, n_fft=512)

    # The delta function requires a minimum number of frames (default width=9).
    # If an audio clip is too short after trimming, it can cause an error.
    # We pad the MFCCs with the edge values if they are shorter than the required width.
    min_delta_width = 9
    if mfccs.shape[1] < min_delta_width:
        pad_width = min_delta_width - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='edge')
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate([mfccs, delta, delta2], axis=0)
    # Return features transposed to have shape (time_steps, num_features)
    return features.T

def main():
    while True:
        wav_file = input("Enter the path to a .wav file (or type 'q' to quit): ").strip()
        if wav_file.lower() == 'q':
            print("Exiting.")
            break
        if not os.path.isfile(wav_file):
            print("File not found. Try again.")
            continue
        try:
            # Load audio, limiting to 2 seconds to match the training data processing
            audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True, duration=2)
        except Exception as e:
            print(f"Error loading file: {e}")
            continue

        # Preprocessing must EXACTLY match the training script (training_07312025.py)
        # 1. Trim silence with the same setting (top_db=25)
        audio, _ = librosa.effects.trim(audio, top_db=25)

        features = extract_features(audio)
        # Pad the sequence to the fixed length the model expects (MAX_PAD_LEN)
        # The input to pad_sequences should be a list of sequences
        features_padded = pad_sequences([features], maxlen=MAX_PAD_LEN, padding='post', truncating='post', dtype='float32')
        prediction = model.predict(features_padded)
        confidence = float(np.max(prediction))
        label_idx = int(np.argmax(prediction, axis=1)[0])
        # Map prediction index to class and context
        # For a 2-class model: 0 = feeding, 1 = fighting
        if label_idx == 0:
            predicted_class = 'feeding'
            context = 'feeding'
        elif label_idx == 1:
            predicted_class = 'fighting'
            context = 'fighting'
        else:
            predicted_class = 'unknown'
            context = 'unknown'

        if confidence < 0.5:
            print(f"Predicted class: unknown (confidence: {confidence:.2f})")
            print(f"Predicted context: unknown")
        else:
            print(f"Predicted class: {predicted_class} (confidence: {confidence:.2f})")
            print(f"Predicted context: {context}")

        # Show spectrogram
        S = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
