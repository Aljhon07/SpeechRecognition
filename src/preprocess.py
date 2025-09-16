import os
import torch
import torch.nn as nn
import torchaudio
from tools import language_corpus as lc
import config
from tools.utils import normalize_text
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
import logging
from pathlib import Path

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
        self.log_mel = WhisperLogMelSpectrogram(sr=sr)
        self.precomputed_dir = {
            'train': os.path.join(config.PRECOMPUTED_DIR, 'train'),
            'dev': os.path.join(config.PRECOMPUTED_DIR, 'dev'),
            'test': os.path.join(config.PRECOMPUTED_DIR, 'test'),
        }
       
        os.makedirs(self.librispeech_path, exist_ok=True)
        for key in self.precomputed_dir:
            os.makedirs(self.precomputed_dir[key], exist_ok=True)
            tsv_path = os.path.join(self.output_dir, f"{key}.tsv")
            # Initialize empty TSV files using pandas
            df = pd.DataFrame(columns=["file_name", "transcript", "sample_rate", "duration", "audio_path", "speaker_id", "chapter_id", "utterance_id"])
            df.to_csv(tsv_path, sep='\t', index=False)

    def preprocess(self, subsets=None, vocab_size=5000):
        if subsets is None:
            subsets = config.LIBRISPEECH_SUBSETS  # Default
            
        all_transcriptions = []
        data_entries = {
            'train': [],
            'dev': [],
            'test': []
        }
        for subset in subsets:
            print(f"Processing {subset}...")
            dataset = torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url=subset, download=True)
            for i in tqdm(range(len(dataset)), desc=f"Processing {subset}"):

                precomputed_data = self.precompute(subset, dataset[i])

                if precomputed_data is None:
                    continue
                metadata, features = precomputed_data
                file_name, audio_path, transcript, duration = metadata
                waveform, padded_spec, unpadded_spec = features

                
                all_transcriptions.append(transcript)
               
                # Save precomputed features
                split = 'train' if 'train' in subset else 'dev' if 'dev' in subset else 'train'

                data_entries[split].append({
                    "file_name": file_name,
                    "transcript": transcript,
                    "sample_rate": 16000,
                    "duration": duration,
                    "audio_path": audio_path,
                    "speaker_id": file_name.split('-')[0],
                    "chapter_id": file_name.split('-')[1],
                    "utterance_id": file_name.split('-')[2]
                })
                
                # Process spectrogram for saving
                if padded_spec.dim() == 3 and padded_spec.shape[0] == 1:
                    padded_spec = padded_spec.squeeze(0).contiguous()
                
                # Transpose for model compatibility: (n_mels, time) -> (time, n_mels)
                final_spec = padded_spec.transpose(0, 1).contiguous()
                spec_len = unpadded_spec.shape[-1]
                
                if os.path.exists(os.path.join(self.precomputed_dir['train' if 'train' in subset else 'dev' if 'dev' in subset else 'test'], f"{file_name}.pt")):
                    continue
                # Save ONLY spectrograms in .pt files
                torch.save({
                    'spectrogram': final_spec,  # (time, n_mels)
                    'unpadded_spectrogram': unpadded_spec,
                }, os.path.join(self.precomputed_dir[split], f"{file_name}.pt"))

        # Save data entries to TSV files
        for split, entries in data_entries.items():
            if entries:
                df = pd.DataFrame(entries)
                tsv_path = os.path.join(self.output_dir, f"{split}.tsv")
                df.to_csv(tsv_path, sep='\t', index=False)
                print(f"Saved {len(entries)} entries to {tsv_path}")
            else:
                print(f"No entries to save for {split} split.")
        
        # save all transcriptions to a text file
        with open(os.path.join(self.output_dir, f"transcriptions.txt"), "w", encoding="utf-8") as f:
            for transcript in all_transcriptions:
                f.write(f"{transcript}\n")
        
        print(f"Saved {len(all_transcriptions)} transcriptions to transcriptions.txt")
        
        # Train SentencePiece model
        lc.train(vocab_size=vocab_size, model_prefix=config.LANGUAGE)
        print(f"SentencePiece model trained and saved to {self.output_dir}")

    def precompute(self, subset, item):

        ''' Precompute log-Mel spectrograms and return metadata 
            Returns: 
            metadata (Tuple): file_name, audio_path, transcript, duration
            features (Tuple): waveform, padded_spec, unpadded_spec
        '''
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        file_name = f"{speaker_id}-{chapter_id}-{utterance_id:04d}"
                
        # Calculate duration
        duration = waveform.shape[1] / sample_rate

        if duration >= 30.0:
            return None
        
        # Get audio file path (LibriSpeech uses .flac files)
        audio_path = config.LIBRISPEECH_PATH / "LibriSpeech" / subset / str(speaker_id) / str(chapter_id) / f"{file_name}.flac"
        normalized_transcript = normalize_text(transcript)

        padded_spec, unpadded_spec = self.log_mel(waveform)

        metadata = (
            file_name,
            str(audio_path),
            normalized_transcript,
            duration
        )

        features = (
            waveform,
            padded_spec,
            unpadded_spec
        )
        return metadata, features

        # Create data item

if __name__ == "__main__":
    preprocessor = Preprocessor(sr=16000)
    # Multiple datasets example:
    preprocessor.preprocess( vocab_size=5000)