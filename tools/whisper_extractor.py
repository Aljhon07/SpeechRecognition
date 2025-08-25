import os
import torchaudio
import matplotlib.pyplot as plt
from tools.audio import to_wav
from src.preprocess import WhisperLogMelSpectrogram
import config
from transformers import WhisperModel, AutoFeatureExtractor
import torch

class WhisperFeatureExtractor:
    def __init__(self, sr=16000):
        self.feature_extractor = WhisperLogMelSpectrogram(sr=sr)
        self.encoder = WhisperModel.from_pretrained("openai/whisper-tiny").encoder

    def extract_features(self, audio_path):
        # Convert to WAV if necessary
        wav_path = audio_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            to_wav(audio_path, wav_path)

        # Load audio file
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # Resample if necessary
        if sample_rate != self.feature_extractor.sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.feature_extractor.sr
            )
            waveform = resampler(waveform)
            sample_rate = self.feature_extractor.sr

        # Extract mel features using WhisperLogMelSpectrogram
        features = self.feature_extractor(waveform)
        return features, waveform, sample_rate

    def plot_features(self, features, waveform, sample_rate, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)

        # Plot waveform
        plt.figure(figsize=(10, 4))
        plt.plot(waveform.t().numpy())
        plt.title("Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.savefig(os.path.join(output_dir, "waveform.png"))
        plt.close()

        # Plot mel features
        plt.figure(figsize=(10, 4))
        plt.imshow(features[0].numpy(), aspect="auto", origin="lower")
        plt.title("Mel Features")
        plt.xlabel("Time")
        plt.ylabel("Mel Channels")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "mel_features.png"))
        plt.close()

        # Plot histogram of features
        plt.figure(figsize=(10, 4))
        plt.hist(features[0].numpy().flatten(), bins=50, color='blue', alpha=0.7)
        plt.title("Feature Histogram")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, "feature_histogram.png"))
        plt.close()

    def process_single_file(self, audio_file, output_dir="output"):
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return

        print(f"Processing {audio_file}...")
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_file)

        # Ensure the waveform is resampled to the correct sampling rate
        if sample_rate != self.feature_extractor.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.feature_extractor.sr)
            waveform = resampler(waveform)

        # Pass the waveform directly to WhisperLogMelSpectrogram
        features = self.feature_extractor(waveform)

        # Pass features through the encoder
        print(f"Encoder input shape: {features.shape}")
        with torch.no_grad():
            encoder_outputs = self.encoder(features)

        # Print the shape of the encoder input and output
        print(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")

        self.plot_features(features, waveform, sample_rate, output_dir=output_dir)

if __name__ == "__main__":
  
    extractor = WhisperFeatureExtractor()

    audio_file = config.WAVS_PATH / "common_voice_en_42165090.wav"
    output_dir = config.OUTPUT_DIR / "test_output"

    extractor.process_single_file(audio_file, output_dir)
