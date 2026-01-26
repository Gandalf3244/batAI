# Train audio model from labeled clips in a spreadsheet
# Handles both new training and fine-tuning existing models

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


def extract_audio_features(file_path, n_mfcc=40, max_length=79):
    # pulls out mel-spectrogram features, outputs shape (79, 120) for model
    try:
        # Load audio
        audio, sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=120,  # Match model input (120 features)
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, features)
        log_mel_spec = log_mel_spec.T
        
        # Pad or truncate to fixed length (79 time frames)
        if log_mel_spec.shape[0] < max_length:
            pad_width = max_length - log_mel_spec.shape[0]
            log_mel_spec = np.pad(log_mel_spec, ((0, pad_width), (0, 0)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:max_length, :]
        
        return log_mel_spec
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_data_from_spreadsheet(excel_file, max_samples=None):
    # reads excel file and processes all the audio files into features
    # expects columns: Filename, Label, Full_Path
    # max_samples lets you limit data if you want faster testing
    print(f"Loading data from: {excel_file}")
    
    # Read spreadsheet
    df = pd.read_excel(excel_file)
    print(f"Found {len(df)} entries in spreadsheet")
    print()
    
    # make sure we have the right columns
    required_cols = ['Filename', 'Label', 'Full_Path']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing column '{col}'")
            print(f"Available columns: {list(df.columns)}")
            return None, None, None
    
    # grab random subset if needed
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42)
        print(f"Using {max_samples} random samples")
    
    # process each audio file
    print("Extracting features from audio files...")
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        file_path = row['Full_Path']
        label = row['Label']
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        # pull features from this file
        feat = extract_audio_features(file_path)
        if feat is not None:
            features.append(feat)
            labels.append(label)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} files...")
    
    print(f"  Processed {len(df)}/{len(df)} files")
    print(f"Successfully extracted features from {len(features)} files")
    print()
    
    if len(features) == 0:
        print("Error: No features extracted!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"Feature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Class distribution: {np.bincount(y)}")
    print()
    
    return X, y, label_encoder


def train_model(X, y, label_encoder, model_path=None, epochs=50, batch_size=32):
    # trains a new model or continues training an existing one
    # if model_path is given, it'll fine-tune that model instead
    print("Preparing training...")
    
    # see how many samples each class has
    class_counts = np.bincount(y)
    min_samples = np.min(class_counts)
    
    if min_samples < 2:
        print(f"Warning: Some classes have only {min_samples} sample(s)")
        print("Removing classes with fewer than 2 samples...")
        
        # drop classes that don't have enough samples
        valid_mask = np.array([class_counts[label] >= 2 for label in y])
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Remaining samples: {len(X)}")
        print(f"Remaining classes: {np.unique(y)}")
        print()
    
    # split into train/val sets, try to keep class balance
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # couldn't balance classes, just do random split
        print("Warning: Cannot use stratified split, using random split")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print()
    
    # either load existing model or make a new one
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = keras.models.load_model(model_path)
        print("Model loaded! Fine-tuning...")
    else:
        print("Creating new model...")
        model = create_model(num_classes=len(label_encoder.classes_))
    
    print()
    model.summary()
    print()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nFinal evaluation:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print()
    
    # Save model
    output_name = f"trained_model_acc{int(val_acc*100)}.keras"
    model.save(output_name)
    print(f"✓ Model saved to: {output_name}")
    
    # Save label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved to: label_encoder.pkl")
    
    return model, history


def create_model(num_classes=2):
    # builds the neural net - conv layers + LSTM for sequence learning
    model = models.Sequential([
        layers.Input(shape=(79, 120)),
        
        # first conv layer
        layers.Conv1D(64, 5, padding='same', activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),
        
        # second conv layer
        layers.Conv1D(128, 5, padding='same', activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),
        
        # LSTM
        layers.LSTM(128),
        layers.Dropout(0.5),
        
        # final layers
        layers.Dense(128, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Audio Model Training")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python train_audio_model.py <spreadsheet.xlsx> [existing_model.keras] [epochs] [max_samples]")
        print()
        print("Arguments:")
        print("  spreadsheet.xlsx      : Excel file with columns: Filename, Label, Full_Path")
        print("  existing_model.keras  : Optional - existing model to fine-tune")
        print("  epochs                : Number of epochs (default: 50)")
        print("  max_samples           : Max samples to use (default: all)")
        print()
        print("Examples:")
        print("  python train_audio_model.py Labeled_clips.xlsx")
        print("  python train_audio_model.py Labeled_clips.xlsx model_08022025_2class_acc86.keras")
        print("  python train_audio_model.py Labeled_clips.xlsx model.keras 100 500")
        print()
        sys.exit(1)
    
    spreadsheet = sys.argv[1]
    existing_model = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith('.keras') else None
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 50
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    if not os.path.exists(spreadsheet):
        print(f"Error: Spreadsheet '{spreadsheet}' not found")
        sys.exit(1)
    
    # Load data
    X, y, label_encoder = load_data_from_spreadsheet(spreadsheet, max_samples)
    
    if X is None:
        sys.exit(1)
    
    # Train model
    model, history = train_model(X, y, label_encoder, existing_model, epochs)
