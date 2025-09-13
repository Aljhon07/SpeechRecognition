import os
import torch
import torch.nn as nn
import torchaudio
from tools import language_corpus as lc
import config
from tools.utils import normalize_text
import pandas as pd
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
import logging

# Set up logging for shape tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    

class Preprocessor:
    def __init__(self, sr=16000):
        self.output_dir = config.OUTPUT_DIR
        self.librispeech_path = config.LIBRISPEECH_PATH

        os.makedirs(self.output_dir, exist_ok=True)
    def preprocess(self, subsets=None, vocab_size=5000):
        if subsets is None:
            subsets = config.LIBRISPEECH_SUBSETS  # Default
            
        all_transcriptions = []
        
        for subset in subsets:
            print(f"Processing {subset}...")
            dataset = torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url=subset, download=True)
            
            for i in tqdm(range(len(dataset)), desc=f"Processing {subset}"):
                waveform, sample_rate, transcript, _, _, _ = dataset[i]

                normalized_transcript = normalize_text(transcript)
                all_transcriptions.append(normalized_transcript)
        
        # save all transcriptions to a text file
        with open(os.path.join(self.output_dir, f"{config.LANGUAGE}_sentences.txt"), "w", encoding="utf-8") as f:
            for transcript in all_transcriptions:
                f.write(f"{transcript}\n")
        
        print(f"Saved {len(all_transcriptions)} transcriptions to {config.LANGUAGE}_sentences.txt")
        
        # Train SentencePiece model
        lc.train(vocab_size=vocab_size, model_prefix=config.LANGUAGE)
        print(f"SentencePiece model trained and saved to {self.output_dir}")

if __name__ == "__main__":
    preprocessor = Preprocessor(sr=16000)
    
    # Multiple datasets example:
    preprocessor.preprocess( vocab_size=5000)