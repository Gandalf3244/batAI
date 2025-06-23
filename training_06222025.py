import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import os
from tensorflow.keras.callbacks import EarlyStopping

# Function to extract audio features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Prepare training data from audio_1000
train_annotations = pd.read_csv(r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_1000\annotations1000.csv')
X_train = []
y_train = []
train_data_path = r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_1000'

for index, row in train_annotations.iterrows():
    file_path = os.path.join(train_data_path, row['File Name'])
    if os.path.exists(file_path):
        features = extract_features(file_path)
        X_train.append(features)
        y_train.append(row['Context'])

# Prepare test data from audio_300
test_annotations = pd.read_csv(r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_300\annotations300.csv')
X_test = []
y_test = []
test_data_path = r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio_300'

for index, row in test_annotations.iterrows():
    file_path = os.path.join(test_data_path, row['File Name'])
    if os.path.exists(file_path):
        features = extract_features(file_path)
        X_test.append(features)
        y_test.append(row['Context'])

# Convert to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load existing model
model = load_model('bat_speech_model_06172025.keras')

# Optionally, recompile if needed (e.g., if optimizer state is not saved)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Continue training with new data
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train_encoded,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test_encoded),
                    callbacks=[early_stopping])

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_pred_labels = le.inverse_transform(y_pred)

print("\nPrediction results (first 10 samples):")
for i in range(10):
    print(f"True: {y_test[i]}, Predicted: {y_pred_labels[i]}")

# Save improved model in native Keras format, achieved accuracy 99.39%
model.save('bat_speech_model_06222025.keras')



