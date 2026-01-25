"""
GUI version of bat behavior timeline analyzer.

Process:
1. Select audio file
2. Select output spreadsheet location
3. Process vocalizations and classify with AI model
4. Display timeline graphs for both species
5. Option to save graphs

Usage:
    python behavior_timeline_gui.py
"""

import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import soundfile as sf
import librosa
from tensorflow import keras  # type: ignore
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import functions from original script
def extract_vocalizations(audio_file, min_duration=1.0, max_duration=10.0,
                         silence_duration=0.5, energy_threshold=0.002, padding=0.1):
    """
    Extract individual vocalization clips from continuous recording.
    Uses the same logic as extract_bat_clips.py
    Returns list of (start_time, end_time, audio_segment) tuples.
    """
    print(f"Loading audio file: {audio_file}")

    # Get file info
    info = sf.info(audio_file)
    sr = info.samplerate
    total_duration = info.duration

    print(f"Duration: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"Sample rate: {sr} Hz")
    print(f"Clip length: {min_duration}s - {max_duration}s")
    print(f"Silence threshold: {silence_duration}s")

    # Detection parameters (same as extract_bat_clips.py)
    window_size = int(0.02 * sr)  # 20ms windows
    hop_size = window_size // 2
    silence_frames_needed = int(silence_duration / (hop_size / sr))
    max_frames = int(max_duration / (hop_size / sr))
    padding_samples = int(padding * sr)

    print("\nDetecting vocalization segments...")

    # Process in chunks
    chunk_duration = 60
    chunk_samples = int(chunk_duration * sr)

    segments = []  # (start_time, end_time)
    current_start = None
    consecutive_silence = 0
    current_duration = 0
    time_offset = 0

    with sf.SoundFile(audio_file) as f:
        while True:
            audio_chunk = f.read(chunk_samples)
            if len(audio_chunk) == 0:
                break

            # Convert to mono for analysis
            if audio_chunk.ndim > 1:
                audio_mono = np.mean(audio_chunk, axis=1)
            else:
                audio_mono = audio_chunk

            # Calculate energy
            for i in range(0, len(audio_mono) - window_size, hop_size):
                window = audio_mono[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))

                current_time = time_offset + (i / sr)

                if rms > energy_threshold:
                    # Sound detected
                    if current_start is None:
                        current_start = current_time
                        current_duration = 0
                    else:
                        current_duration += 1
                    consecutive_silence = 0

                    # Check if clip is getting too long - force end
                    if current_duration >= max_frames:
                        segment_end = current_time
                        segment_duration = segment_end - current_start

                        if segment_duration >= min_duration:
                            segments.append((current_start, segment_end))

                        current_start = None
                        current_duration = 0
                else:
                    # Silence
                    if current_start is not None:
                        consecutive_silence += 1

                        if consecutive_silence >= silence_frames_needed:
                            # Long silence - end segment
                            segment_end = current_time - silence_duration
                            segment_duration = segment_end - current_start

                            if segment_duration >= min_duration:
                                segments.append((current_start, segment_end))

                            current_start = None
                            current_duration = 0
                            consecutive_silence = 0

            time_offset += len(audio_mono) / sr
            progress = (time_offset / total_duration) * 100
            print(f"  Progress: {time_offset/60:.1f}/{total_duration/60:.1f} min ({progress:.1f}%) - {len(segments)} clips", end='\r')

            if len(audio_chunk) < chunk_samples:
                break

    # Handle final segment
    if current_start is not None:
        segment_end = time_offset
        segment_duration = segment_end - current_start
        if segment_duration >= min_duration and segment_duration <= max_duration:
            segments.append((current_start, segment_end))

    print(f"\n\nFound {len(segments)} vocalization clips")

    # Now extract audio for each segment
    print("\nLoading audio segments...")
    all_segments = []

    with sf.SoundFile(audio_file) as f:
        for idx, (start_time, end_time) in enumerate(segments):
            if (idx + 1) % 100 == 0:
                print(f"  Loaded {idx+1}/{len(segments)} clips...")

            # Add padding
            padded_start = max(0, start_time - padding)
            padded_end = min(total_duration, end_time + padding)

            # Read audio segment
            start_sample = int(padded_start * sr)
            end_sample = int(padded_end * sr)

            f.seek(start_sample)
            segment_audio = f.read(end_sample - start_sample)

            # Convert to mono if stereo
            if segment_audio.ndim > 1:
                segment_audio = np.mean(segment_audio, axis=1)

            all_segments.append((start_time, end_time, segment_audio))

    print(f"Loaded {len(all_segments)} audio segments\n")
    return all_segments, sr

def extract_features(audio, sr, target_frames=79, n_mels=120):
    """Extract mel-spectrogram features from audio."""
    try:
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (time, frequency)
        mel_spec_db = mel_spec_db.T

        # Pad or truncate to target length
        if mel_spec_db.shape[0] < target_frames:
            pad_width = target_frames - mel_spec_db.shape[0]
            mel_spec_db = np.pad(mel_spec_db, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:target_frames, :]

        return mel_spec_db
    except Exception as e:
        return None

def classify_vocalizations(segments, sr, model, label_encoder):
    """
    Classify each vocalization using the trained AI model.
    Returns list of (time, label, confidence) tuples.
    """
    print("Classifying vocalizations with AI model...")
    classifications = []

    # Process in batches for efficiency
    batch_size = 32
    for batch_start in range(0, len(segments), batch_size):
        batch_end = min(batch_start + batch_size, len(segments))
        batch_segments = segments[batch_start:batch_end]

        if batch_start % 100 == 0:
            print(f"  Classified {batch_start}/{len(segments)} clips...")

        # Extract features for batch
        batch_features = []
        batch_times = []

        for start_time, end_time, segment_audio in batch_segments:
            features = extract_features(segment_audio, sr)
            if features is not None:
                batch_features.append(features)
                batch_times.append(start_time)

        if not batch_features:
            continue

        # Predict batch
        batch_features = np.array(batch_features)
        predictions = model(batch_features, training=False).numpy()

        # Process predictions
        for time, pred in zip(batch_times, predictions):
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            label = label_encoder.inverse_transform([predicted_class])[0]

            classifications.append({
                'time': time,
                'label': label,
                'confidence': confidence
            })

    print(f"Successfully classified {len(classifications)} clips\n")
    return classifications

def create_timeline_data(classifications, total_duration, bin_minutes=10):
    """
    Create timeline data for graphs (modified from create_timeline_graphs).
    Returns data for plotting.
    """
    print("Creating timeline data...")

    # Calculate bins
    bin_seconds = bin_minutes * 60
    num_bins = int(np.ceil(total_duration / bin_seconds))

    # Initialize behavior counts (separate by species)
    rods_behaviors = {}
    straws_behaviors = {}

    # Count behaviors in each bin
    for classification in classifications:
        time = classification['time']
        label = classification['label']
        bin_index = int(time / bin_seconds)

        if bin_index >= num_bins:
            bin_index = num_bins - 1

        # Separate by species and behavior
        if label.startswith('Rods_'):
            behavior = label.replace('Rods_', '')
            if behavior not in rods_behaviors:
                rods_behaviors[behavior] = np.zeros(num_bins)
            rods_behaviors[behavior][bin_index] += 1
        elif label.startswith('Straws_'):
            behavior = label.replace('Straws_', '')
            if behavior not in straws_behaviors:
                straws_behaviors[behavior] = np.zeros(num_bins)
            straws_behaviors[behavior][bin_index] += 1

    # Create time labels
    time_labels = []
    for i in range(num_bins):
        minutes = i * bin_minutes
        hours = minutes // 60
        mins = minutes % 60
        time_labels.append(f"{hours:02d}:{mins:02d}")

    return rods_behaviors, straws_behaviors, time_labels, num_bins, bin_minutes

def save_summary_spreadsheet(classifications, total_duration, spreadsheet_file, audio_filename, bin_minutes=10):
    """
    Save summary statistics to Excel spreadsheet.
    Preserves all existing data and appends new row.
    """
    rods_behaviors, straws_behaviors, _, _, _ = create_timeline_data(classifications, total_duration, bin_minutes)

    # Prepare data for spreadsheet
    summary_data = {'Filename': [audio_filename]}

    # Add Rods behaviors
    for behavior in sorted(rods_behaviors.keys()):
        total = int(np.sum(rods_behaviors[behavior]))
        rate = total / (total_duration / 3600)
        summary_data[f'Rods_{behavior}_Total_Calls'] = [total]
        summary_data[f'Rods_{behavior}_Calls_Per_Hour'] = [round(rate, 1)]

    # Add Straws behaviors
    for behavior in sorted(straws_behaviors.keys()):
        total = int(np.sum(straws_behaviors[behavior]))
        rate = total / (total_duration / 3600)
        summary_data[f'Straws_{behavior}_Total_Calls'] = [total]
        summary_data[f'Straws_{behavior}_Calls_Per_Hour'] = [round(rate, 1)]

    # Add total duration
    summary_data['Total_Duration_Minutes'] = [round(total_duration / 60, 1)]
    summary_data['Total_Duration_Hours'] = [round(total_duration / 3600, 1)]

    # Create DataFrame for new row
    new_df = pd.DataFrame(summary_data)

    # Check if spreadsheet already exists
    if os.path.exists(spreadsheet_file):
        try:
            # Read existing data and append
            existing_df = pd.read_excel(spreadsheet_file, engine='openpyxl')
            print(f"✓ Found existing spreadsheet with {len(existing_df)} rows")

            # Make sure columns match (add missing columns with NaN if needed)
            for col in new_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = 0

            for col in existing_df.columns:
                if col not in new_df.columns:
                    new_df[col] = 0

            # Reorder columns to match existing spreadsheet
            new_df = new_df[existing_df.columns]

            # Append new row
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save with explicit engine to ensure compatibility
            combined_df.to_excel(spreadsheet_file, index=False, engine='openpyxl')
            print(f"✓ Appended new row (now {len(combined_df)} total rows): {spreadsheet_file}")
        except Exception as e:
            print(f"Warning: Could not read existing spreadsheet ({e})")
            print(f"Creating new spreadsheet instead: {spreadsheet_file}")
            new_df.to_excel(spreadsheet_file, index=False, engine='openpyxl')
    else:
        # Create new spreadsheet
        new_df.to_excel(spreadsheet_file, index=False, engine='openpyxl')
        print(f"✓ Created new spreadsheet: {spreadsheet_file}")

class BatBehaviorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bat Behavior Timeline Analyzer")
        self.root.geometry("1200x800")

        # Variables
        self.audio_file = None
        self.spreadsheet_file = None
        self.classifications = None
        self.total_duration = None
        self.bin_minutes = 10

        # Model and encoder
        self.model = None
        self.label_encoder = None

        # Create UI
        self.create_widgets()

        # Load model
        self.load_model()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Bat Behavior Timeline Analyzer",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Audio file selection
        ttk.Label(main_frame, text="Step 1: Select Audio File").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.audio_button = ttk.Button(main_frame, text="Select Audio File",
                                     command=self.select_audio_file)
        self.audio_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.audio_label = ttk.Label(main_frame, text="No file selected")
        self.audio_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))

        # Spreadsheet selection
        ttk.Label(main_frame, text="Step 2: Select Output Spreadsheet").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        self.spreadsheet_button = ttk.Button(main_frame, text="Select Spreadsheet Location",
                                           command=self.select_spreadsheet)
        self.spreadsheet_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.spreadsheet_label = ttk.Label(main_frame, text="No location selected")
        self.spreadsheet_label.grid(row=4, column=1, sticky=tk.W, padx=(10, 0))

        # Process button
        self.process_button = ttk.Button(main_frame, text="Process Audio",
                                       command=self.process_audio, state=tk.DISABLED)
        self.process_button.grid(row=5, column=0, columnspan=2, pady=(20, 10))

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))

        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        self.results_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Graph canvas
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Save graphs button
        self.save_graphs_button = ttk.Button(self.results_frame, text="Save Graphs",
                                           command=self.save_graphs, state=tk.DISABLED)
        self.save_graphs_button.grid(row=1, column=0, pady=(10, 0))

        # Status label
        self.status_label = ttk.Label(self.results_frame, text="")
        self.status_label.grid(row=1, column=1, pady=(10, 0))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

    def load_model(self):
        try:
            print("Loading AI model...")
            self.model = keras.models.load_model("12_29_both_species.keras")

            print("Loading label encoder...")
            with open("label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            print(f"Model loaded! Classes: {self.label_encoder.classes_}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model files: {str(e)}")
            self.root.quit()

    def select_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if file_path:
            self.audio_file = file_path
            self.audio_label.config(text=os.path.basename(file_path))
            self.check_ready()

    def select_spreadsheet(self):
        file_path = filedialog.asksaveasfilename(
            title="Select Spreadsheet Location",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            self.spreadsheet_file = file_path
            self.spreadsheet_label.config(text=os.path.basename(file_path))
            self.check_ready()

    def check_ready(self):
        if self.audio_file and self.spreadsheet_file:
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)

    def process_audio(self):
        if not self.audio_file or not self.spreadsheet_file:
            return

        # Disable buttons
        self.process_button.config(state=tk.DISABLED)
        self.audio_button.config(state=tk.DISABLED)
        self.spreadsheet_button.config(state=tk.DISABLED)

        # Start progress
        self.progress.start()
        self.status_label.config(text="Processing...")

        try:
            # Extract vocalizations
            segments, sr = extract_vocalizations(self.audio_file)

            if not segments:
                messagebox.showerror("Error", "No vocalizations detected! Try a different audio file.")
                self.reset_ui()
                return

            self.total_duration = segments[-1][1] if segments else 0

            # Classify
            self.classifications = classify_vocalizations(segments, sr, self.model, self.label_encoder)

            if not self.classifications:
                messagebox.showerror("Error", "No clips could be classified!")
                self.reset_ui()
                return

            # Save spreadsheet
            save_summary_spreadsheet(self.classifications, self.total_duration,
                                   self.spreadsheet_file, os.path.basename(self.audio_file),
                                   self.bin_minutes)

            # Display graphs
            self.display_graphs()

            # Enable save graphs button
            self.save_graphs_button.config(state=tk.NORMAL)
            self.status_label.config(text="Processing complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.reset_ui()

        finally:
            self.progress.stop()

    def display_graphs(self):
        rods_behaviors, straws_behaviors, time_labels, num_bins, bin_minutes = create_timeline_data(
            self.classifications, self.total_duration, self.bin_minutes)

        # Clear previous plots
        self.fig.clear()

        # Create subplots
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)

        # Color schemes
        colors_rods = {
            'Fighting': '#e74c3c',
            'Fighting_Talking': '#c0392b',
            'Talking': '#3498db',
            'Want_Food': '#16a085'
        }

        colors_straws = {
            'Fighting': '#f39c12',
            'Fighting_Talking': '#9b59b6',
            'Talking': '#2ecc71',
            'Want_Food': '#e91e63'
        }

        # Plot Rods behaviors
        for behavior, counts in sorted(rods_behaviors.items()):
            color = colors_rods.get(behavior, '#95a5a6')
            ax1.plot(range(num_bins), counts, label=behavior.replace('_', ' '),
                    linewidth=2.5, color=color, marker='o', markersize=5, alpha=0.8)

        ax1.set_xlabel('Time of Day (HH:MM)', fontsize=12)
        ax1.set_ylabel(f'Vocal Activity (calls / {bin_minutes} min)', fontsize=12)
        ax1.set_title('Rods (Species 1) Behaviors', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, num_bins - 1)

        # Plot Straws behaviors
        for behavior, counts in sorted(straws_behaviors.items()):
            color = colors_straws.get(behavior, '#95a5a6')
            ax2.plot(range(num_bins), counts, label=behavior.replace('_', ' '),
                    linewidth=2.5, color=color, marker='o', markersize=5, alpha=0.8)

        ax2.set_xlabel('Time of Day (HH:MM)', fontsize=12)
        ax2.set_ylabel(f'Vocal Activity (calls / {bin_minutes} min)', fontsize=12)
        ax2.set_title('Straws (Species 2) Behaviors', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, num_bins - 1)

        # Set x-axis labels
        n = max(1, num_bins // 12)
        tick_positions = list(range(0, num_bins, n))
        tick_labels = [time_labels[i] for i in tick_positions]

        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

        self.fig.tight_layout()
        self.canvas.draw()

    def save_graphs(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Graphs",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Graphs saved to {file_path}")

    def reset_ui(self):
        self.process_button.config(state=tk.NORMAL)
        self.audio_button.config(state=tk.NORMAL)
        self.spreadsheet_button.config(state=tk.NORMAL)
        self.save_graphs_button.config(state=tk.DISABLED)
        self.status_label.config(text="")
        self.progress.stop()

def main():
    root = tk.Tk()
    app = BatBehaviorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()