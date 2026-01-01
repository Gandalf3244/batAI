"""
Analyze bat behavior timeline from continuous recording.

Process: 
1. Extract vocalization clips from recording
2. Classify each clip with trained AI model
3. Graph behaviors over time (separate graphs for Rods vs Straws)

Usage:
    python behavior_timeline.py <audio_file> [--bin-minutes 10] [--output timeline.png]
    
Examples:
    python behavior_timeline.py "recording.wav"
    python behavior_timeline.py "recording.wav" --bin-minutes 5
    python behavior_timeline.py "recording.wav" --output my_timeline.png
"""

import sys
import os
import argparse
import numpy as np
import soundfile as sf
import librosa
from tensorflow import keras  # type: ignore
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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

def create_timeline_graphs(classifications, total_duration, bin_minutes=10, output_file='behavior_timeline.png', audio_filename='recording.wav'):
    """
    Create separate timeline graphs for Rods and Straws behaviors.
    Also saves summary statistics to Excel spreadsheet.
    """
    print("Creating timeline graphs...")
    
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
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Bat Behavior Timeline Analysis', fontsize=16, fontweight='bold')
    
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
    
    # Set x-axis labels (show every nth label to avoid crowding)
    n = max(1, num_bins // 12)
    tick_positions = list(range(0, num_bins, n))
    tick_labels = [time_labels[i] for i in tick_positions]
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total duration: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"Time bins: {num_bins} bins of {bin_minutes} minutes each")
    
    print(f"\nRODS BEHAVIORS:")
    for behavior in sorted(rods_behaviors.keys()):
        total = int(np.sum(rods_behaviors[behavior]))
        rate = total / (total_duration / 3600)  # calls per hour
        print(f"  {behavior.replace('_', ' '):20s}: {total:4d} calls ({rate:.1f} calls/hour)")
    
    print(f"\nSTRAWS BEHAVIORS:")
    for behavior in sorted(straws_behaviors.keys()):
        total = int(np.sum(straws_behaviors[behavior]))
        rate = total / (total_duration / 3600)  # calls per hour
        print(f"  {behavior.replace('_', ' '):20s}: {total:4d} calls ({rate:.1f} calls/hour)")
    
    print("="*60)
    
    # Save summary to spreadsheet
    spreadsheet_file = output_file.replace('.png', '_summary.xlsx')
    print(f"\nSaving summary to spreadsheet: {spreadsheet_file}")
    
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
        # Read existing data and append
        existing_df = pd.read_excel(spreadsheet_file)
        
        # Make sure columns match (add missing columns with 0s if needed)
        for col in new_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = 0
        
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = 0
        
        # Reorder columns to match
        new_df = new_df[existing_df.columns]
        
        # Append new row
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_excel(spreadsheet_file, index=False)
        print(f"✓ Appended to existing spreadsheet: {spreadsheet_file}")
    else:
        # Create new spreadsheet
        new_df.to_excel(spreadsheet_file, index=False)
        print(f"✓ Created new spreadsheet: {spreadsheet_file}")
    
    # Automatically open the spreadsheet
    try:
        os.startfile(spreadsheet_file)
        print(f"✓ Opening spreadsheet in Excel...")
    except:
        print(f"  (Could not auto-open - please open manually)")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze bat behavior timeline from continuous recording',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Process:
  1. Extract vocalization clips from recording
  2. Classify each clip with trained AI model  
  3. Graph behaviors over time (separate for Rods vs Straws)

Examples:
  python behavior_timeline.py "recording.wav"
  python behavior_timeline.py "recording.wav" --bin-minutes 5
  python behavior_timeline.py "recording.wav" --output my_timeline.png
  python behavior_timeline.py "recording.wav" --energy-threshold 0.001
        """
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--bin-minutes', type=int, default=10,
                       help='Time bin size in minutes (default: 10)')
    parser.add_argument('--output', default='behavior_timeline.png',
                       help='Output graph filename (default: behavior_timeline.png)')
    parser.add_argument('--energy-threshold', type=float, default=0.002,
                       help='Energy threshold for detection (default: 0.002)')
    parser.add_argument('--min-duration', type=float, default=1.0,
                       help='Minimum clip duration in seconds (default: 1.0)')
    parser.add_argument('--max-duration', type=float, default=10.0,
                       help='Maximum clip duration in seconds (default: 10.0)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found!")
        sys.exit(1)
    
    if not os.path.exists("12_29_both_species.keras"):
        print("Error: 12_29_both_species.keras not found!")
        print("Make sure you're in the directory with the trained model.")
        sys.exit(1)
    
    if not os.path.exists("label_encoder.pkl"):
        print("Error: label_encoder.pkl not found!")
        sys.exit(1)
    
    # Load model and label encoder
    print("Loading AI model...")
    model = keras.models.load_model("12_29_both_species.keras")
    
    print("Loading label encoder...")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded! Classes: {label_encoder.classes_}\n")
    print("="*60)
    print("STEP 1: EXTRACT VOCALIZATION CLIPS")
    print("="*60)
    
    # Step 1: Extract vocalizations (1-10 second clips)
    segments, sr = extract_vocalizations(
        args.audio_file,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        energy_threshold=args.energy_threshold
    )
    
    if not segments:
        print("No vocalizations detected! Try lowering --energy-threshold")
        sys.exit(1)
    
    total_duration = segments[-1][1] if segments else 0
    
    print("="*60)
    print("STEP 2: CLASSIFY WITH AI MODEL")
    print("="*60)
    
    # Step 2: Classify with AI model
    classifications = classify_vocalizations(segments, sr, model, label_encoder)
    
    if not classifications:
        print("No clips could be classified!")
        sys.exit(1)
    
    print("="*60)
    print("STEP 3: CREATE TIMELINE GRAPHS")
    print("="*60)
    
    # Step 3: Create graphs
    create_timeline_graphs(classifications, total_duration, 
                          bin_minutes=args.bin_minutes, 
                          output_file=args.output,
                          audio_filename=os.path.basename(args.audio_file))
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
