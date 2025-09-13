import torch
import torchaudio
import platform

def play_sound(file_path):
    """Cross-platform sound playback function"""
    try:
        if platform.system() == 'Windows':
            # Windows - use winsound
            import winsound
            winsound.PlaySound(str(file_path), winsound.SND_FILENAME)
        else:
            # Linux/Docker - use system commands
            import subprocess
            import os
            
            file_path = str(file_path)
            if not os.path.exists(file_path):
                print(f"Audio file not found: {file_path}")
                return
                
            # Try different audio players available in Linux
            players = ['aplay', 'paplay', 'ffplay', 'play']
            
            for player in players:
                try:
                    if player == 'ffplay':
                        subprocess.run([player, '-nodisp', '-autoexit', file_path], 
                                     check=True, capture_output=True, timeout=10)
                    else:
                        subprocess.run([player, file_path], 
                                     check=True, capture_output=True, timeout=10)
                    print(f"Played audio using {player}")
                    return
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            print("No audio player available - audio playback skipped")
            
    except Exception as e:
        print(f"Error playing sound: {e}")
import json
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import config
import os
import pandas as pd
import jiwer

def normalize_text(input):
    # remove_chars = r"[?!’–—‘\-\.:;()“”\"]"
    # text = input.lower()
    # text = text.replace('"', ' ')
    # text = re.sub(remove_chars, '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', input)
    text = text.lower()

    return text

def mean_norm(spectrogram):
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
    return spectrogram

def rms_normalize(waveform):
    rms = waveform.pow(2).mean().sqrt()
    gain = 0.1 / rms
    if waveform.numel() == 0:
        print("Waveform is empty.")
        return None
    
    if rms < 1e-7:
        print(f"RMS is too low: {rms.item()}")
        return None
    return waveform * gain

def plot_waveforms(waveform, trimmed):
    plt.figure(figsize=(12, 4))

    # waveform = waveform[0, :1000]
    # trimmed = trimmed[0, :1000]
    plt.subplot(1, 2, 1)
    plt.plot(waveform.squeeze().numpy())
    plt.title(f"Waveform 1 - {len(waveform.squeeze())/16000:.2f}s")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(trimmed.squeeze().numpy())
    plt.title(f"Waveform 2 - {len(trimmed.squeeze())/16000:.2f}s")
    plt.xlabel("Samples")

    plt.tight_layout()
    plt.show()

def plot_spectrogram(orig, normalized, sample_rate=16000):
    
    orig = orig.detach()
    normalized = normalized.detach()
    
    # Plot the raw (unnormalized) spectrogram
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert tensors to numpy arrays
    orig = orig.squeeze(0).cpu().numpy()
    normalized = normalized.squeeze(0).cpu().numpy()
    
    # Plot the raw (unnormalized) spectrogram
    im0 = axs[0, 0].imshow(orig, aspect='auto', origin='lower', cmap='inferno')
    axs[0, 0].set_title('Raw Mel Spectrogram')
    axs[0, 0].set_xlabel('Time Frames')
    axs[0, 0].set_ylabel('Mel Bands')
    fig.colorbar(im0, ax=axs[0, 0], format="%+2.0f dB")

    # Plot the normalized spectrogram
    im1 = axs[0, 1].imshow(normalized, aspect='auto', origin='lower', cmap='inferno')
    axs[0, 1].set_title('Normalized Mel Spectrogram')
    axs[0, 1].set_xlabel('Time Frames')
    axs[0, 1].set_ylabel('Mel Bands')
    fig.colorbar(im1, ax=axs[0, 1], format="%+2.0f dB")

    # Plot the distribution (histogram) of the raw spectrogram values
    axs[1, 0].hist(orig.flatten(), bins=80, color='blue', alpha=0.7)
    axs[1, 0].set_title('Distribution of Raw Mel Spectrogram')
    axs[1, 0].set_xlabel('Amplitude')
    axs[1, 0].set_ylabel('Frequency')

    # Plot the distribution (histogram) of the normalized spectrogram values
    axs[1, 1].hist(normalized.flatten(), bins=80, color='green', alpha=0.7)
    axs[1, 1].set_title('Distribution of Normalized Mel Spectrogram')
    axs[1, 1].set_xlabel('Amplitude')
    axs[1, 1].set_ylabel('Frequency')

    # Adjust layout to avoid overlap
    plt.savefig(config.OUTPUT_DIR / f"spectrogram_{time.strftime('%Y%m%d-%H%M%S')}.png")
    plt.tight_layout()
    plt.show()

def double_vad(audio, sample_rate=16000):
    # Apply VAD from the front
    trimmed_front = F.vad(audio, sample_rate=sample_rate, trigger_level=7.0)
    
    if trimmed_front.size(1) < 4:
        return audio
    # Reverse the waveform along the time dimension
    reversed_audio = torch.flip(trimmed_front, dims=[-1])
    
    # Apply VAD on reversed audio (trims end of original)
    trimmed_back = F.vad(reversed_audio, sample_rate=sample_rate, trigger_level=7.0)
    
    # Flip back to original orientation
    final_audio = torch.flip(trimmed_back, dims=[-1])
    
    return final_audio

def get_bucket_duration(waveform, sr=16000):
    # Define non-overlapping buckets: (min (inclusive), max (exclusive), target_duration)
    duration = waveform.shape[1] / sr

    buckets = [
        (0.0, 1.0, 1.0),      # 0.0 ≤ x < 0.1 → 0.5
        (1.0, 1.5, 1.5),      # 0.5 ≤ x < 1.0 → 1.0
        (1.5, 2.0, 2),      # 1.0 ≤ x < 1.5 → 1.5
        (2.0, 2.5, 2.5),      # 2.0 ≤ x < 2.5 → 2.5
        (2.5, 3.0, 3),      # 2.5 ≤ x < 3.0 → 3.0
        (3.0, 3.5, 3.5),      # 3.0 ≤ x < 3.5 → 3.5
        (3.5, 4.0, 4),      # 3.0 ≤ x < 4.5 → 4.5
        (4.0, 4.5, 4.5),      # 4.0 ≤ x < 4.5 → 4.5
        (4.5, 5.0, 5),      # 4.5 ≤ x < 5.0 → 5.0
        (5.0, 5.5, 5.5),      # 5.0 ≤ x < 5.5 → 5.5
        (5.5, 6.0, 6),      # 5.5 ≤ x < 6.0 → 6.0
        (6.0, 6.5, 6.5),      # 6.0 ≤ x < 6.5 → 6.5
        (6.5, 7.0, 7),      # 6.5 ≤ x < 7.0 → 7.0
        (7.0, 7.5, 7.5),      # 7.0 ≤ x < 7.5 → 7.5
        (7.5, 8.0, 8),      # 7.5 ≤ x < 8.0 → 8.0
        (8.0, 8.5, 8.5),      # 8.0 ≤ x < 8.5 → 8.5
        (8.5, 9.0, 9),      # 8.5 ≤ x < 9.0 → 9.0
        (9.0, 9.5, 9.5),      # 9.0 ≤ x < 9.5 → 9.5
        (9.5, 10.0, 10),     # 9.5 ≤ x < 10.0 → 10.0
        (10.0, 15.0, 15),
        (15.0, 20.0, 20)  # 10.0 ≤ x < 10.5 → 10.5
    ]
    
    # Check buckets first
    for min_dur, max_dur, target in buckets:
        if min_dur <= duration < max_dur:
            return target

    return 30

def ctc_decoder(preds):
    decoded = []
    prev_char = None
    for char_idx in preds:
        if char_idx != 0 and char_idx != prev_char:
            decoded.append(char_idx)
        prev_char = char_idx
    return decoded

def wer(ref, hyp):
    pass

def audio_sanity_check(loader, speech_module, device):
    """
    Comprehensive sanity check for audio data with playback and visualization
    Args:
        loader: DataLoader to test
        speech_module: SpeechModule containing original data
        device: torch device for tensor operations
    """
    import random
    import tempfile
    import os
    import winsound
    from tools import language_corpus as lc
    
    print("\n" + "="*50)
    print("🔍 AUDIO SANITY CHECK")
    print("="*50)
    
    for batch in loader:
        # Unpack 6 elements from LibriSpeech dataset
        inputs, labels, input_len, labels_len, file_name, unpadded_specs = batch
        inputs, labels = inputs.to(device), labels.to(device)
        random_idx = random.randint(0, inputs.shape[0] - 1)

        # Basic validation
        if inputs is None or labels is None:
            raise ValueError("Inputs or labels are None.")
        if len(inputs) == 0 or len(labels) == 0:
            raise ValueError("Inputs or labels are empty.")
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError("Batch size mismatch between inputs and labels.")
        if input_len.shape[0] != labels_len.shape[0]:
            raise ValueError("Batch size mismatch between input lengths and label lengths.")

        print(f"📊 Batch Information:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Input lengths shape: {input_len.shape}")
        print(f"  Label lengths shape: {labels_len.shape}")
        print(f"  Input lengths: {input_len}")
        print(f"  Label lengths: {labels_len}")

        print(f"\n🎯 Sample Details (Index {random_idx}):")
        print(f"  File name: {file_name[random_idx]}")
        print(f"  Input Shape: {inputs[random_idx].shape}")
        print(f"  Label Shape: {labels[random_idx].shape}")
        print(f"  Input length: {input_len[random_idx]}")
        print(f"  Label length: {labels_len[random_idx]}")
        print(f"  Decoded Label: '{lc.decode(labels[random_idx].tolist())}'")

        # Spectrogram information
        print(f"\n📈 Spectrogram Details:")
        print(f"  Unpadded spec shape: {unpadded_specs[random_idx].shape}")
        print(f"  Padded spec shape: {inputs[random_idx].shape}")
        print(f"  Spec Stats: Min: {inputs[random_idx].min():.3f} | Max: {inputs[random_idx].max():.3f} | Mean: {inputs[random_idx].mean():.3f} | Std: {inputs[random_idx].std():.3f}")
        
        # Audio playback
        print(f"\n🎵 Audio Playback:")
        print(f"  Playing audio for: {file_name[random_idx]}")
        print(f"  📝 Expected transcription: '{lc.decode(labels[random_idx].tolist())}'")
        print(f"  👂 Listen carefully and verify the audio matches!")
        
        # Find and play audio
        audio_played = False
        if hasattr(speech_module, 'train_data'):
            original_data = speech_module.train_data
            for item in original_data:
                if item['file_name'] == file_name[random_idx]:
                    try:
                        # Use waveform data from memory (safer than file paths)
                        waveform = item['waveform']
                        sample_rate = item['sample_rate']
                        
                        # Create temporary WAV file in system temp directory
                        # This is completely separate from original files
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                            temp_wav_path = temp_wav.name
                            
                        print(f"  📁 Temp file location: {temp_wav_path}")
                        print(f"  ⏳ Creating temporary WAV for playback...")
                        
                        # Save waveform as WAV and play
                        torchaudio.save(temp_wav_path, waveform, sample_rate)
                        print(f"  🔊 Playing audio now...")
                        winsound.PlaySound(temp_wav_path, winsound.SND_FILENAME)
                        
                        # Clean up temporary file immediately
                        os.unlink(temp_wav_path)
                        print(f"  ✅ Audio played successfully! Temp file cleaned up.")
                        audio_played = True
                        
                    except Exception as e:
                        print(f"  ❌ Error playing audio: {e}")
                        try:
                            if os.path.exists(temp_wav_path):
                                os.unlink(temp_wav_path)
                        except:
                            pass
                    break
            
        if not audio_played:
            if not hasattr(speech_module, 'train_data'):
                print(f"  ⚠️ Speech module data not available for audio playback")
            else:
                print(f"  ⚠️ Could not find audio data for {file_name[random_idx]}")
        
        # Spectrogram visualization
        print(f"\n📊 Displaying spectrogram comparison...")
        print(f"  🔍 Compare unpadded vs padded spectrograms")
        plot_spectrogram(unpadded_specs[random_idx], inputs[random_idx])
        
        print(f"\n✅ Sanity check completed!")
        print("="*50)
        return

if __name__ == "__main__":
    # with open(os.path.join(config.OUTPUT_DIR / 'buckets', 'bucket_3.0.json'), 'r') as f:
    #         data = json.load(f)

 
    normalize_text("Hello, world! 231 123")
    