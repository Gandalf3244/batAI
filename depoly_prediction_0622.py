import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import os

# Function to extract audio features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Prepare test data from audio_300
test_annotations = pd.read_csv(r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_300\annotations300.csv')
test_data_path = r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_300'

# Pick one .wav file from the test set
row = test_annotations.iloc[0]
file_path = os.path.join(test_data_path, row['File Name'])
features = extract_features(file_path)
X_single = np.array(features).reshape(1, -1, 1)

# Prepare label encoder using all possible classes
all_contexts = test_annotations['Context'].unique()
le = LabelEncoder()
le.fit(all_contexts)
# Load the trained model
model = load_model('bat_speech_model_06222025.keras')

# Predict
prediction = model.predict(X_single)
predicted_class_index = np.argmax(prediction, axis=1)[0]
predicted_class = le.inverse_transform([predicted_class_index])[0]

print(f"File: {row['File Name']}")
print(f"True label: {row['Context']}")
print(f"Predicted label: {predicted_class}")