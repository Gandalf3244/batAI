"""
Human Voice Remover
Removes human speech frequencies from audio, keeping only non-human sounds (like bat calls)
"""

import os
import numpy as np
import soundfile as sf


def apply_frequency_filter(audio, sr, low_cut, high_cut):
    """
    Apply frequency filter using FFT.
    Removes frequencies between low_cut and high_cut Hz.
    """
    # FFT
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    
    # Create mask to zero out human voice frequencies
    mask = np.ones_like(fft)
    mask[(freqs >= low_cut) & (freqs <= high_cut)] = 0
    
    # Apply mask and inverse FFT
    fft_filtered = fft * mask
    filtered = np.fft.irfft(fft_filtered, n=len(audio))
    
    return filtered


def remove_human_voice(input_file, output_file="cleaned_audio.wav"):
    """
    Remove human voice frequencies from audio.
    Human speech is typically 85-255 Hz (fundamental) with harmonics up to ~5kHz.
    Bat calls are often 20-80kHz (ultrasonic) but recording may capture 5-16kHz range.
    
    Args:
        input_file: Input WAV file
        output_file: Output cleaned WAV file
    """
    print(f"Loading: {input_file}")
    
    # Get file info
    info = sf.info(input_file)
    sr = info.samplerate
    duration = info.duration
    channels = info.channels
    
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Sample rate: {sr} Hz")
    print(f"Channels: {channels}")
    
    print("\nDesigning voice removal filter...")
    print("  Removing frequencies: 80-4000 Hz (human voice range)")
    print("  Keeping: >4000 Hz (high frequencies)")
    
    # Filter parameters
    low_cut = 80    # Remove above this
    high_cut = 4000  # Remove below this
    
    print("\nProcessing audio (streaming to output file)...")
    
    # Process in chunks to handle large files
    chunk_duration = 30  # Process 30 seconds at a time
    chunk_samples = int(chunk_duration * sr)
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    total_samples = 0
    
    # Open output file for streaming write
    with sf.SoundFile(input_file) as f_in:
        with sf.SoundFile(output_file, 'w', sr, channels, 'PCM_16') as f_out:
            chunk_num = 0
            while True:
                audio_chunk = f_in.read(chunk_samples)
                if len(audio_chunk) == 0:
                    break
                
                chunk_num += 1
                
                # Apply FFT-based filtering
                if audio_chunk.ndim > 1:
                    # Process each channel
                    filtered = np.zeros_like(audio_chunk)
                    for ch in range(audio_chunk.shape[1]):
                        filtered[:, ch] = apply_frequency_filter(audio_chunk[:, ch], sr, low_cut, high_cut)
                else:
                    filtered = apply_frequency_filter(audio_chunk, sr, low_cut, high_cut)
                
                # Normalize chunk to prevent clipping
                max_val = np.abs(filtered).max()
                if max_val > 0:
                    filtered = filtered * (0.95 / max_val)
                
                # Write directly to output file
                f_out.write(filtered)
                total_samples += len(filtered)
                
                # Progress
                time_processed = total_samples / sr
                progress = (time_processed / duration) * 100
                print(f"  Progress: {time_processed/60:.1f}/{duration/60:.1f} min ({progress:.1f}%)", end='\r')
                
                if len(audio_chunk) < chunk_samples:
                    break
    
    print(f"\n\n✓ Voice-removed audio saved!")
    print(f"  Output: {output_file}")
    print(f"  Duration: {total_samples/sr/60:.1f} minutes")
    print(f"  Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python voice_remover.py <input.wav> [output.wav]")
        print("\nExample:")
        print("  python voice_remover.py 20250912104642.WAV cleaned_recording.wav")
        print("\nThis removes human voice frequencies (80-4000 Hz)")
        print("and keeps high-frequency sounds like bat calls.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "cleaned_audio.wav"
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    remove_human_voice(input_file, output_file)
