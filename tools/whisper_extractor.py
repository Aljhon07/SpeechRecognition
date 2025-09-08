import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Resample
from transformers import WhisperFeatureExtractor
import torchaudio
import os
import pandas as pd
import config

class WhisperLogMelSpectrogram(nn.Module):
    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        assert self.feature_extractor.sampling_rate == sr, f"Expected {sr}, got {self.feature_extractor.sampling_rate}"

    def forward(self, x: torch.Tensor):
        waveform = x[0].detach().cpu().numpy()  # Extract single-channel waveform
        unpadded_features = self.feature_extractor(waveform, sampling_rate=self.sr, return_tensors="pt", padding=False )
        unpadded_features = unpadded_features["input_features"]
        
        features = self.feature_extractor(waveform, sampling_rate=self.sr, return_tensors="pt" )
        input_features = features["input_features"]
        # Return the features and their length
        return input_features, unpadded_features
    

class LibriSpeechDataset(Dataset):
    def __init__(self, root, tsv_path, split="test-clean", sr=16000):
        """
        Dataset for LibriSpeech that preprocesses audio using WhisperLogMelSpectrogram.
        Args:
            root (str): Path to the LibriSpeech dataset.
            split (str): Dataset split to use (e.g., "test-clean").
            sr (int): Sampling rate for audio.
        """

        # Initialize the dataset
        self.data = pd.read_csv(tsv_path, sep="\t").head(100)
        self.sr = sr
        self.resampler = Resample(orig_freq=16000, new_freq=sr) if sr != 16000 else None
        self.feature_extractor = WhisperLogMelSpectrogram(sr=sr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load audio and transcription from LibriSpeech
        row = self.data.iloc[idx]
        audio_path = os.path.join(str(config.WAVS_PATH), f"{row['file_name']}.wav")

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if self.resampler:
            waveform = self.resampler(waveform)

        # Extract log mel spectrogram features
        features, unpadded_features = self.feature_extractor(waveform)
        features = features.squeeze(0)
        unpadded_features = unpadded_features.squeeze(0)
        print(f"Features shape: {features.shape}")
        print(f"Features length: {unpadded_features.shape}")

        return {
            "features": features,  
            "features_len": unpadded_features.shape[-1],
            "transcription": row['transcription']
        }

# Create a DataLoader for the LibriSpeech dataset
def create_dataloader(root, tsv_path, split="test-clean", batch_size=4, sr=16000):
    dataset = LibriSpeechDataset(root=root, tsv_path=tsv_path, split=split, sr=sr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

# Collate function for padding and batching
def collate_fn(batch):
    features = [item["features"] for item in batch]
    features_len = [item["features_len"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]

    # Pad features to the maximum length in the batch
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    return {
        "features": features_padded,
        "features_len": torch.tensor(features_len),
        "transcriptions": transcriptions
    }

if __name__ == "__main__":
    # Example usage
    tsv_path = config.OUTPUT_DIR / 'train.tsv'
    root = config.COMMON_VOICE_PATH # Path to store the LibriSpeech dataset

    dataloader = create_dataloader(root=root, tsv_path=tsv_path, split="test-clean", batch_size=4, sr=16000)

    for batch in dataloader:
        print("Features shape:", batch["features"].shape)
        print("Features lengths:", batch["features_len"])
        print("Transcriptions:", batch["transcriptions"])
        break