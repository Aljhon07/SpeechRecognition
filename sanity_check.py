"""
Dataset Sanity Checker for LibriSpeech Data

Concise sanity checking including:
1. Dataloader validation
2. Spectrogram comparison (padded vs unpadded)
3. Audio playback with winsound.PlaySound
4. Transcription comparison using CTC decoder
"""

# Fix OpenMP conflict
import os

from src.preprocess import WhisperLogMelSpectrogram
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchaudio
import tempfile
import random
import winsound
from pathlib import Path

import config
from src.dataset import SpeechModule
from tools.utils import plot_spectrogram, ctc_decoder
from tools import language_corpus as lc


class DatasetSanityChecker:
    def __init__(self, batch_size=None):
        self.batch_size = batch_size or config.H_PARAMS["BATCH_SIZE"]
        self.speech_module = None
        self.loaders = None
        self.logmel = WhisperLogMelSpectrogram()
    def setup_dataloader(self):
        try:
            self.speech_module = SpeechModule(use_precomputed=True)
            self.speech_module.load_librispeech_data(subsets=config.LIBRISPEECH_SUBSETS)
            self.loaders = self.speech_module.create_dataloader(batch_size=self.batch_size)
            return True
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def check_sample(self, batch, idx=0):
        """Check a single sample with all validations"""
        specs, labels, spec_lens, label_lens, file_names, unpadded_specs = batch
        
        # Basic info
        file_name = file_names[idx]
        # Fix: Use actual label length to trim padding
        actual_labels = labels[idx].tolist()
        transcription = lc.decode(actual_labels)
        
        print(f"\n🔍 SAMPLE: {file_name}")
        print(f"📊 Shapes - Padded: {specs[idx].shape}, Unpadded: {unpadded_specs[idx].shape}")
        print(f"📝 Transcription: '{transcription}'")
        
        # 1. Use dataloader ✓ (already using it)
        
        # 2. Compare spectrograms
        print("🎨 Showing spectrogram comparison...")
        
        # 3. Play audio using winsound.PlaySound with waveform
        spectrogram = self.play_audio_from_waveform(file_name, transcription)
        print(f"Spectrogram shape from waveform: {spectrogram.shape}")
        print(f"Spectrogram shape from dataloader: {specs[idx].shape}")
        plot_spectrogram(spectrogram, unpadded_specs[idx])
        # 4. Print details
        print(f"📐 Spec lengths: {spec_lens[idx].item()}, Label length: {label_lens[idx].item()}")
        print(f"📊 Stats - Min: {specs[idx].min():.3f}, Max: {specs[idx].max():.3f}, Mean: {specs[idx].mean():.3f}")
        
        # 5. Compare transcriptions using CTC decoder
        ctc_decoded = ctc_decoder(actual_labels)
        ctc_text = lc.decode(ctc_decoded)
        match = transcription == ctc_text
        print(f"🔄 CTC comparison: {'✅ MATCH' if match else '❌ DIFF'}")
        if not match:
            print(f"   Original: '{transcription}'")
            print(f"   CTC: '{ctc_text}'")
        
        return True
    
    def play_audio_from_waveform(self, file_name, transcription):
        """Play audio using winsound.PlaySound from waveform data"""
        print(f"🔊 Playing audio: {file_name}")
        print(f"📝 Expected: '{transcription}'")
        
        # Find original audio file to get waveform
        audio_paths = [
            config.LIBRISPEECH_PATH / "LibriSpeech" / "test-clean",
        ]
        
        waveform = None
        sample_rate = 16000
        
        # Look for audio file
        for base_path in audio_paths:
            if not base_path.exists():
                continue
                
            parts = file_name.split('-')
            if len(parts) >= 3:
                speaker_id, chapter_id = parts[0], parts[1]
                audio_file = base_path / speaker_id / chapter_id / f"{file_name}.flac"
                
                if audio_file.exists():
                    try:
                        waveform, sample_rate = torchaudio.load(str(audio_file))
                        print(f"📁 Loaded: {audio_file}")
                        break
                    except Exception as e:
                        print(f"❌ Load error: {e}")
        
        if waveform is not None:
            # Create temporary WAV file from waveform
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Save waveform as WAV
                torchaudio.save(temp_path, waveform, sample_rate)
                
                # Play using winsound
                print("� Playing...")
                winsound.PlaySound(temp_path, winsound.SND_FILENAME)
                
                # Cleanup
                os.unlink(temp_path)
                print("✅ Audio played successfully!")
                spec, _ = self.logmel(waveform)
                return _.squeeze(0)  # (Time, Mel)
            except Exception as e:
                print(f"❌ Playback error: {e}")
                try:
                    os.unlink(temp_path)
                except:
                    pass
        else:
            print("⚠️ Audio file not found, playing beep...")
            winsound.Beep(800, 500)
    
    def run_sanity_check(self, num_samples=1):
        """Run the complete sanity check on the dataset."""
        print("🚀 DATASET SANITY CHECK")
        print("=" * 40)
        
        if not self.setup_dataloader():
            return False
        
        for split_name, dataloader in self.loaders.items():
            print(f"\n📊 Checking {split_name} split...")
            
            for batch_idx, batch in enumerate(dataloader):
                # Check first few samples
                batch_size = batch[0].shape[0]
                samples_to_check = min(num_samples, batch_size)
                
                for i in range(samples_to_check):
                    self.check_sample(batch, i)
                
                break  # Only check first batch
            
            break  # Only check first split for brevity
        
        print("\n✅ Sanity check completed!")
        return True


def main():
    """Main function to run the dataset sanity checker."""
    print("🔍 LibriSpeech Dataset Sanity Checker")
    
    checker = DatasetSanityChecker(batch_size=2)
    checker.run_sanity_check(num_samples=1)


if __name__ == "__main__":
    main()
