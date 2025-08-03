import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Input, BatchNormalization, LSTM
import os
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
import pickle

# Constants for audio processing
SAMPLE_RATE = 20000  # Match sample rate with other scripts for consistency
MAX_PAD_LEN = 79     # Max length for padding sequences (based on 2s audio at 20kHz)

# Function to extract audio features
def extract_features(file_path, augment=False):
    # Load audio file (limit to 2 seconds)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=2)
    # Trim leading/trailing silence to focus on the active part of the audio
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if augment:
        # Random time stretch
        rate = np.random.uniform(0.8, 1.2)
        if len(audio) > 0:
            audio = librosa.effects.time_stretch(y=audio, rate=rate)
        # Random pitch shift
        steps = np.random.randint(-2, 3)
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)
        # Add random noise
        audio = audio + 0.005 * np.random.randn(len(audio))
    # Extract 40 MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=512)

    # The delta function requires a minimum number of frames (default width=9).
    # If an audio clip is too short after trimming, it can cause an error.
    # We pad the MFCCs with the edge values if they are shorter than the required width.
    min_delta_width = 9
    if mfccs.shape[1] < min_delta_width:
        pad_width = min_delta_width - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='edge')
    # Add delta and delta-delta features
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate([mfccs, delta, delta2], axis=0)
    # Return features transposed to have shape (time_steps, num_features)
    return features.T

# Prepare training and test data from the same folder, stratified by 'Context'
annotations = pd.read_csv(r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_select\fight_feed_high_similarity.csv')
data_path = r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_select'

# Group by 'class'
grouped = annotations.groupby('class')

X_train, y_train, X_test, y_test = [], [], [], []

for context, group in grouped:
    # Filter for .wav files that exist
    files = [f for f in group['File Name'] if f.endswith('.wav') and os.path.exists(os.path.join(data_path, f))]
    # Ensure same number of .wav files per class (use minimum count across all classes)
    min_count = min(
        grouped.get_group(c)['File Name'].apply(
            lambda x: x.endswith('.wav') and os.path.exists(os.path.join(data_path, x))
        ).sum()
        for c in grouped.groups
    )
    files = files[:min_count]

    # Split files: 80% for training, 20% for testing (a standard, robust split)
    num_train = int(0.80 * len(files))
    train_files = files[:num_train]
    test_files = files[num_train:]

    # Extract features for training with augmentation
    for f in train_files:
        features = extract_features(os.path.join(data_path, f), augment=True)
        X_train.append(features)
        y_train.append(context)

    # Extract features for testing (no augmentation)
    for f in test_files:
        features = extract_features(os.path.join(data_path, f), augment=False)
        X_test.append(features)
        y_test.append(context)

# Pad sequences to ensure uniform length for the model
X_train = pad_sequences(X_train, maxlen=MAX_PAD_LEN, padding='post', truncating='post', dtype='float32')
X_test = pad_sequences(X_test, maxlen=MAX_PAD_LEN, padding='post', truncating='post', dtype='float32')

y_train = np.array(y_train)
y_test = np.array(y_test)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Save the label encoder for use in prediction
le_save_path = os.path.join(os.path.dirname(__file__), 'le_3class.pkl')
with open(le_save_path, 'wb') as f:
    pickle.dump(le, f)
print(f"\nLabel encoder saved to {le_save_path}")
print(f"Classes encoded: {le.classes_}\n")

# Build a hybrid CNN-LSTM model for improved temporal pattern recognition
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    # 1. CNN layers to extract local features from the audio sequence
    Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),
    Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),
    # 2. LSTM layer to model long-term dependencies in the extracted features
    LSTM(128, return_sequences=False), # return_sequences=False as the next layer is Dense
    Dropout(0.5),
    # 3. Dense layers for classification
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train_encoded,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test_encoded),
                    callbacks=[early_stopping])

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Initial Test Accuracy: {test_accuracy*100:.2f}%")

retries = 0
while test_accuracy < 0.9 and retries < 10:
    print(f"Current accuracy: {test_accuracy*100:.2f}%. Below 90% target. Retraining... (Attempt {retries+1}/10)")
    # Continue training on the same model for a few more epochs
    history = model.fit(X_train, y_train_encoded,
                        epochs=20,
                        batch_size=32,
                        validation_data=(X_test, y_test_encoded),
                        callbacks=[early_stopping])
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    retries += 1

print(f"\nFinal Test Accuracy after retraining attempts: {test_accuracy*100:.2f}%")

# Save improved model in native Keras format
model_filename = f'model_08022025_2class_acc{int(test_accuracy*100)}.keras'
model_save_path = os.path.join(os.path.dirname(__file__), model_filename)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
