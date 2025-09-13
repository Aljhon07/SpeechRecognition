import os
# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import Dataset, DataLoader
from src.preprocess import WhisperLogMelSpectrogram
import config
import json
import random
import torchaudio
import torch
import winsound
from tools.utils import plot_spectrogram
from tools import language_corpus as lc
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torchaudio.transforms as T
import logging
from pathlib import Path

# Set up logging for dataset shape tracking
logger = logging.getLogger(__name__)

class SpeechDataset(Dataset):
    def __init__(self, data, augmented=False, augmented_prob=0.5, epoch_progress=0.0):
        self.data = data
        self.augmented = augmented
        self.augmented_prob = augmented_prob
        self.logmel = WhisperLogMelSpectrogram()
        self.total_duration = sum(item['duration'] for item in data) / (60 * 60)
        self.apply_mask = nn.Sequential(
            T.TimeMasking(time_mask_param=15),
            T.FrequencyMasking(freq_mask_param=8))
        self.verbose = config.H_PARAMS["VERBOSE"]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        # destucture the data
        file_name = data['file_name']
        waveform = data['waveform']
        sr = data['sample_rate']
        file_name = data['file_name']
        labels = lc.encode(data['transcription'])
        labels_len = len(labels)

        if sr != config.AUDIO_PARAMS["SAMPLE_RATE"]:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=config.AUDIO_PARAMS["SAMPLE_RATE"])
            
        spec, unpadded_spec = self.logmel(waveform)
        if self.verbose:
            print(f"Dataset[{idx}] - Loaded waveform shape: {waveform.shape}, sr: {sr}")
            print(f"Dataset[{idx}] - After logmel shape: {spec.shape}")

        if spec.dim() == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0).contiguous()  # Now shape is (n_mels, time_frames)
            if self.verbose:
                print(f"Dataset[{idx}] - After squeeze shape: {spec.shape}")

        if self.augmented and random.random() < self.augmented_prob:
            if self.verbose:
                print(f"Dataset[{idx}] - Applying augmentation")
            spec = self.apply_mask(spec)
            
        # spec should now be (n_mels, time_frames)
        spec_len = unpadded_spec.shape[-1]  # time dimension is now at index 1
        if self.verbose:
            print(f"Dataset[{idx}] - Spec length: {spec_len}")

        if self.verbose:
            print(f"Dataset[{idx}] - Labels length: {labels_len}")

        # Transpose for model compatibility: (n_mels, time) -> (time, n_mels)
        final_spec = spec.transpose(0, 1).contiguous()
        if self.verbose:
            print(f"Dataset[{idx}] - Final spec shape (time, n_mels): {final_spec.shape}")

        if self.verbose:
            self.verbose = False  # Only log for the first item
        labels = torch.tensor(labels, dtype=torch.long)
        return final_spec, labels, torch.tensor(spec_len, dtype=torch.long), torch.tensor(labels_len, dtype=torch.long), file_name, unpadded_spec
    
class SpeechModule:
    def __init__(self, data=None):
        self.train_data = None
        self.dev_data = None
        self.loaders = {}
        self.datasets = {}
        
    def load_librispeech_data(self, subsets=None):
        """Load LibriSpeech data and separate into train/dev based on subset names"""
        if subsets is None:
            subsets = config.LIBRISPEECH_SUBSETS
            
        train_data = []
        dev_data = []
        
        for subset in subsets:
            print(f"Loading LibriSpeech subset: {subset}")
            dataset = torchaudio.datasets.LIBRISPEECH(
                root=config.LIBRISPEECH_PATH, 
                url=subset, 
                download=True
            )
            
            subset_data = []
            for i in range(len(dataset)):
                waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[i]
                
                # Create file identifier
                file_name = f"{speaker_id}_{chapter_id}_{utterance_id:04d}"
                
                # Get audio file path (LibriSpeech uses .flac files)
                audio_path = config.LIBRISPEECH_PATH / "LibriSpeech" / subset / str(speaker_id) / str(chapter_id) / f"{file_name}.flac"
                
                # Calculate duration
                duration = waveform.shape[1] / sample_rate

                if duration >= 30.0:
                    continue  
                
                # make the data item tuple
                data_item = {
                    'file_name': file_name,
                    'waveform': waveform,
                    'sample_rate': sample_rate,
                    'audio_path': str(audio_path),
                    'transcription': transcript.lower(),  # LibriSpeech transcripts are lowercase
                    'duration': duration,  # Add duration field
                    'subset': subset
                }
                subset_data.append(data_item)
            
            # Separate data based on subset name
            if 'train' in subset.lower():
                train_data.extend(subset_data)
                print(f"  Added {len(subset_data)} samples to training set")
            elif 'dev' in subset.lower():
                dev_data.extend(subset_data)
                print(f"  Added {len(subset_data)} samples to development set")
            # elif 'test' in subset.lower():
                # For test sets, we can add to dev for validation or keep separate
                train_data.extend(subset_data)
                print(f"  Added {len(subset_data)} samples to development set (from test subset)")
            else:
                # Default fallback - add to train
                train_data.extend(subset_data)
                print(f"  Added {len(subset_data)} samples to training set (default)")
        
        # Set the data
        self.train_data = train_data
        self.dev_data = dev_data
        
        print(f"\nData loading complete:")
        print(f"Training samples: {len(train_data)}")
        print(f"Development samples: {len(dev_data)}")
        
        return train_data, dev_data
    
    def create_dataloader(self, batch_size=config.H_PARAMS["BATCH_SIZE"]):
        # Ensure we have data loaded
        if self.train_data is None or self.dev_data is None:
            raise ValueError("No data loaded. Please call load_librispeech_data() first or provide data manually.")

        train_dataset = SpeechDataset(self.train_data, augmented=True)
        val_dataset = SpeechDataset(self.dev_data, augmented=False)

        self.datasets = {
            'train': train_dataset,
            'val': val_dataset
        }

        self.loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=self.collate_fn),
            'val': DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False, collate_fn=self.collate_fn)
        }
    
        self.get_dataset_stats()
        return self.loaders
    
    def get_dataset_stats(self):
        """Print dataset statistics"""
        if self.train_data and self.dev_data:
            train_duration = sum(item['duration'] for item in self.train_data) / 3600  # hours
            val_duration = sum(item['duration'] for item in self.dev_data) / 3600  # hours
            
            print(f"\nDataset Statistics:")
            print(f"Training samples: {len(self.train_data)} ({train_duration:.2f} hours)")
            print(f"Validation samples: {len(self.dev_data)} ({val_duration:.2f} hours)")
            print(f"Total samples: {len(self.train_data) + len(self.dev_data)} ({train_duration + val_duration:.2f} hours)")
    
    def load_and_create_dataloaders(self, subsets=None, batch_size=config.H_PARAMS["BATCH_SIZE"]):
        self.load_librispeech_data(subsets)
        
        # Create dataloaders
        return self.create_dataloader(batch_size)

    def collate_fn(self, batch):
        specs, labels, spec_lens, label_lens, file_name, unpadded_specs = zip(*batch)
     
        specs = list(specs)
        labels = list(labels)
        specs = pad_sequence(specs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)

        # Transpose specs from (batch, time, n_mels) to (batch, n_mels, time) for model input
        final_specs = specs.transpose(1, 2)

        return final_specs, labels, torch.tensor(spec_lens, dtype=torch.long), torch.tensor(label_lens, dtype=torch.long), file_name, unpadded_specs


if __name__ == "__main__":
    # Simple test to verify dataset loading and dataloader functionality
    speech_module = SpeechModule()
    loaders = speech_module.load_and_create_dataloaders(
        subsets=config.LIBRISPEECH_SUBSETS,  # Uses config settings
        batch_size=config.H_PARAMS["BATCH_SIZE"]
    )
    
    # Test the current dataloader
    if 'train' in loaders and len(speech_module.train_data) > 0:
        for batch in loaders['train']:
            specs, labels, spec_lens, label_lens, file_names, unpadded_specs = batch
            print(f"Train batch - Specs: {specs.shape}, Labels: {labels.shape}")
            break  # Just test one batch
     
    if 'val' in loaders and len(speech_module.dev_data) > 0:
        for batch in loaders['val']:
            specs, labels, spec_lens, label_lens, file_names, unpadded_specs = batch
            print(f"Val batch - Specs: {specs.shape}, Labels: {labels.shape}")
            print(f"Sample file names: {file_names[:3]}")
            break