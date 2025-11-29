"""
Extract Bat Vocalization Clips
Takes filtered audio and creates clips of continuous vocalizations (10+ seconds)
Cuts out silence/minimal noise between vocalizations
"""

import os
import numpy as np
import soundfile as sf


def extract_vocalization_clips(input_file, output_dir="bat_vocalizations",
                               min_clip_duration=0.2, max_clip_duration=3.0,
                               silence_duration=0.5, energy_threshold=0.002,
                               padding=0.1):
    """
    Extract individual bat vocalization clips from filtered audio.
    
    Args:
        input_file: Filtered audio file (voice removed)
        output_dir: Output directory for clips
        min_clip_duration: Minimum clip length in seconds
        max_clip_duration: Maximum clip length in seconds
        silence_duration: Seconds of silence to end a clip
        energy_threshold: RMS threshold for detecting sound
        padding: Seconds to add before/after each clip
    """
    print(f"Loading: {input_file}")
    
    info = sf.info(input_file)
    sr = info.samplerate
    duration = info.duration
    channels = info.channels
    
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Sample rate: {sr} Hz")
    print(f"Channels: {channels}")
    print(f"Clip length: {min_clip_duration}s - {max_clip_duration}s")
    print(f"Silence threshold: {silence_duration}s")
    
    # Detection parameters
    window_size = int(0.02 * sr)  # 20ms windows
    hop_size = window_size // 2
    silence_frames_needed = int(silence_duration / (hop_size / sr))
    max_frames = int(max_clip_duration / (hop_size / sr))
    padding_samples = int(padding * sr)
    
    print("\nDetecting vocalization segments...")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Process in chunks
    chunk_duration = 60
    chunk_samples = int(chunk_duration * sr)
    
    segments = []  # (start_time, end_time)
    current_start = None
    consecutive_silence = 0
    current_duration = 0
    time_offset = 0
    
    with sf.SoundFile(input_file) as f:
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
                        
                        if segment_duration >= min_clip_duration:
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
                            
                            if segment_duration >= min_clip_duration:
                                segments.append((current_start, segment_end))
                            
                            current_start = None
                            current_duration = 0
                            consecutive_silence = 0
            
            time_offset += len(audio_mono) / sr
            progress = (time_offset / duration) * 100
            print(f"  Progress: {time_offset/60:.1f}/{duration/60:.1f} min ({progress:.1f}%) - {len(segments)} clips", end='\r')
            
            if len(audio_chunk) < chunk_samples:
                break
    
    # Handle final segment
    if current_start is not None:
        segment_end = time_offset
        segment_duration = segment_end - current_start
        if segment_duration >= min_clip_duration and segment_duration <= max_clip_duration:
            segments.append((current_start, segment_end))
    
    print(f"\n\nFound {len(segments)} vocalization clips")
    
    if len(segments) == 0:
        print("No clips found! Try lowering energy_threshold or min_clip_duration")
        return 0
    
    # Save clips
    print("\nSaving clips...")
    
    with sf.SoundFile(input_file) as f:
        for idx, (start_time, end_time) in enumerate(segments, 1):
            clip_duration = end_time - start_time
            
            # Add padding
            padded_start = max(0, start_time - padding)
            padded_end = min(duration, end_time + padding)
            
            # Read audio
            start_sample = int(padded_start * sr)
            end_sample = int(padded_end * sr)
            
            f.seek(start_sample)
            clip_audio = f.read(end_sample - start_sample)
            
            # Save
            output_file = os.path.join(output_dir, 
                                      f"{base_name}_clip_{idx:03d}.wav")
            sf.write(output_file, clip_audio, sr)
            
            if idx <= 10 or idx % 50 == 0:
                print(f"  Clip {idx}: {clip_duration:.1f}s at {start_time/60:.2f}min")
    
    print(f"\n✓ Saved {len(segments)} clips to '{output_dir}'")
    
    # Statistics
    durations = [end - start for start, end in segments]
    total_clip_time = sum(durations)
    print(f"\nStatistics:")
    print(f"  Total clips: {len(segments)}")
    print(f"  Total vocalization time: {total_clip_time/60:.1f} minutes")
    print(f"  Average clip length: {np.mean(durations):.1f}s")
    print(f"  Shortest clip: {np.min(durations):.1f}s")
    print(f"  Longest clip: {np.max(durations):.1f}s")
    
    return len(segments)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_bat_clips.py <filtered_audio.wav> [output_dir] [min_seconds]")
        print("\nExample:")
        print("  python extract_bat_clips.py no_voice_recording.wav bat_clips 10")
        print("\nCreates clips of continuous bat vocalizations (10+ seconds by default)")
        print("Removes silence/quiet sections between vocalizations")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "bat_vocalizations"
    min_duration = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    extract_vocalization_clips(
        input_file,
        output_dir,
        min_clip_duration=1.0,     # At least 1 second
        max_clip_duration=10.0,    # At most 10 seconds per clip
        silence_duration=0.5,      # 500ms of silence ends a clip
        energy_threshold=0.002,    # Adjust if needed
        padding=0.1                # 100ms padding before/after
    )
