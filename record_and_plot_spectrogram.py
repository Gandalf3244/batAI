import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import os

# --- Configuration ---
RECORD_SECONDS = 3.5     # Duration of the recording
SAMPLE_RATE = 20000      # Sample rate (consistent with your other scripts)
FILENAME = "my_speech.wav" # Filename to save the recording

# --- Internal Constants ---
FORMAT = pyaudio.paInt16  # Audio format (16-bit integers)
CHANNELS = 1             # Mono audio
CHUNK = 1024             # Buffer size for reading audio stream

def record_audio():
    """Records audio from the microphone for a fixed duration."""
    p = pyaudio.PyAudio()

    print("\nStarting recording in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("* Recording... Speak now!")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    for i in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert the recorded frames to a numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    # Normalize to float between -1 and 1, which librosa prefers
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    return audio_float

def plot_spectrogram(audio_data, sr):
    """Generates and displays a spectrogram for the given audio data."""
    if audio_data is None or len(audio_data) == 0:
        print("No audio data to plot.")
        return

    # Generate spectrogram using Short-Time Fourier Transform (STFT)
    S = librosa.stft(audio_data)
    # Convert amplitude to decibels
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Plotting the spectrogram
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram of Your Speech (Sample Rate: {sr} Hz)')
    plt.tight_layout()
    plt.show()

def main():
    """Records audio once and plots the spectrogram."""
    input("Press Enter to start recording...")
    recorded_audio = record_audio()
    plot_spectrogram(recorded_audio, sr=SAMPLE_RATE)
    print("\nSpectrogram generated. You can close the plot window to exit the program.")

if __name__ == "__main__":
    main()