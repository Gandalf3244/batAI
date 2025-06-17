import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import os

# Read annotation file
annotations = pd.read_csv(r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\annotations.csv')
# Function to extract audio features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Prepare data
X = []
y = []
data_path = r'C:\Users\zbgbmf\Documents\AI\egyptian_fruit_bats_long\audio'

for index, row in annotations.iterrows():
    file_path = os.path.join(data_path, row['File Name'])
    if os.path.exists(file_path):
        features = extract_features(file_path)
        X.append(features)
        y.append(row['Context'])  # 'Label' is more commonly used, but verify the actual column name in your CSV

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training (10%) and remaining
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.9, random_state=42)

# Split remaining data to get 5% test set
X_test, X_unused, y_test, y_unused = train_test_split(X_temp, y_temp, test_size=0.944, random_state=42)

# Reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build model
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

# Save model
model.save('bat_speech_model.h5')