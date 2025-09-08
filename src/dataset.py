from torch.utils.data import Dataset, DataLoader
from src.preprocess import BucketAudio, WhisperLogMelSpectrogram
import os
import config
import json
import random
import torchaudio
import torch
from tools.utils import double_vad
from tools import language_corpus as lc
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torchaudio.transforms as T
import logging

# Set up logging for dataset shape tracking
logger = logging.getLogger(__name__)

class SpeechDataset(Dataset):
    def __init__(self, data, augmented=False, augmented_prob=0.5, epoch_progress=0.0):
        self.data = data
        self.augmented = augmented
        self.augmented_prob = augmented_prob
        self.epoch_progress = epoch_progress
        # Use WhisperLogMelSpectrogram for consistent preprocessing with model
        self.logmel = WhisperLogMelSpectrogram()
        self.total_duration = sum(item['duration'] for item in data) / (60 * 60)
        self.apply_mask = nn.Sequential(
            T.TimeMasking(time_mask_param=15),
            T.FrequencyMasking(freq_mask_param=8))
        self.verbose = True
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        file_name = data['file_name']
        waveform, sr = torchaudio.load(config.WAVS_PATH / f"{file_name}.wav")
        spec, unpadded_spec = self.logmel(waveform)
        if self.verbose:
            print(f"Dataset[{idx}] - Loaded waveform shape: {waveform.shape}, sr: {sr}")
            print(f"Dataset[{idx}] - After logmel shape: {spec.shape}")

        # WhisperLogMelSpectrogram returns shape (1, n_mels, time_frames)
        # Remove the batch dimension for single samples
        if spec.dim() == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0).contiguous()  # Now shape is (n_mels, time_frames)
            if self.verbose:
                print(f"Dataset[{idx}] - After squeeze shape: {spec.shape}")

        if self.augmented and random.random() < self.augmented_prob:
            if self.verbose:
                print(f"Dataset[{idx}] - Applying augmentation")
            # Progressive SpecAugment based on epoch progress
            # Start with stronger augmentation, reduce as training progresses
            spec = self.apply_mask(spec)
            
        # spec should now be (n_mels, time_frames)
        spec_len = unpadded_spec.shape[-1]  # time dimension is now at index 1
        if self.verbose:
            print(f"Dataset[{idx}] - Spec length: {spec_len}")

        transcription = data['transcription']
        labels = lc.encode(transcription)
        labels_len = len(labels)
        if self.verbose:
            print(f"Dataset[{idx}] - Labels length: {labels_len}")

        # Transpose for model compatibility: (n_mels, time) -> (time, n_mels)
        final_spec = spec.transpose(0, 1).contiguous()
        if self.verbose:
            print(f"Dataset[{idx}] - Final spec shape (time, n_mels): {final_spec.shape}")

        if self.verbose:
            self.verbose = False  # Only log for the first item
        labels = torch.tensor(labels, dtype=torch.long)
        return final_spec, labels, torch.tensor(spec_len, dtype=torch.long), torch.tensor(labels_len, dtype=torch.long), file_name
    
    
    def preprocess(self, audio):
        audio = self.logmel(audio)
        return audio
    
class SpeechModule:
    def __init__(self, data=None, excluded_buckets=['0.0', '15.0', '20.0', '30.0']):
        self.train_data = None
        self.dev_data = None
        self.bucket = BucketAudio()
        self.loaders = {}
        self.datasets = {}
        self.excluded_buckets = excluded_buckets

        self.load_data()
        self.create_dataloader()
    
    def load_data(self):
        # Check if buckets exist, if not create them
        if not os.path.exists(config.OUTPUT_DIR / 'buckets' / 'train') or not os.path.exists(config.OUTPUT_DIR / 'buckets' / 'dev'):
            self.bucket.init()
        
        # Load train and dev data separately
        self.train_data = self.bucket.load_buckets('train')
        self.dev_data = self.bucket.load_buckets('dev')
    
    def create_dataloader(self, batch_size=config.H_PARAMS["BATCH_SIZE"]):
        if self.train_data is None or self.dev_data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        # Get all available bucket keys from train data
        train_keys = set(self.train_data.keys())
        dev_keys = set(self.dev_data.keys())
        
        # Use intersection to ensure we have both train and dev data for each bucket
        common_keys = train_keys.intersection(dev_keys)
        keys = sorted(list(common_keys), key=float)
        
        print(f"Available buckets: {sorted(list(train_keys), key=float)}")
        print(f"Train buckets: {sorted(list(train_keys), key=float)}")
        print(f"Dev buckets: {sorted(list(dev_keys), key=float)}")
        print(f"Common buckets: {keys}")
        
        for key in keys:
            if key in self.excluded_buckets:
                print(f"Excluding bucket {key}")
                continue
                
            # Use train data for training and dev data for validation
            train_items = self.train_data[key][:100]
            dev_items = self.dev_data[key][:50]

            train_dataset = SpeechDataset(train_items, augmented=True)
            val_dataset = SpeechDataset(dev_items, augmented=False)
            
            self.datasets[key] = {
                'train': train_dataset,
                'val': val_dataset
            }

            self.loaders[key] = {
                'train': DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=self.collate_fn),
                'val': DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False, collate_fn=self.collate_fn)
            }
        
        self.get_dataset_stats()
        return self.loaders

    def collate_fn(self, batch):
        specs, labels, spec_lens, label_lens, file_name = zip(*batch)
     
        specs = list(specs)
        labels = list(labels)
        specs = pad_sequence(specs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)

        # Transpose specs from (batch, time, n_mels) to (batch, n_mels, time) for model input
        final_specs = specs.transpose(1, 2)

        return final_specs, labels, torch.tensor(spec_lens, dtype=torch.long), torch.tensor(label_lens, dtype=torch.long), file_name
    
    def get_dataset_stats(self) -> None:
        if self.datasets is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        # Separate stats for train and dev
        train_duration = 0
        train_samples = 0
        dev_duration = 0
        dev_samples = 0
        excluded_train_duration = 0
        excluded_train_samples = 0
        excluded_dev_duration = 0
        excluded_dev_samples = 0
        
        print(f"{'='*80}")
        print(f"{'DATASET STATISTICS':^80}")
        print(f"{'='*80}")
        print(f"{'Bucket':<10} {'Train Samples':<15} {'Train Hours':<12} {'Dev Samples':<15} {'Dev Hours':<12} {'Status':<10}")
        print(f"{'-'*80}")
        
        for key, dataset in self.datasets.items():
            train_bucket_duration = dataset['train'].total_duration
            train_bucket_samples = len(dataset['train'])
            dev_bucket_duration = dataset['val'].total_duration  # 'val' contains dev data
            dev_bucket_samples = len(dataset['val'])
            
            status = "EXCLUDED" if key in self.excluded_buckets else "INCLUDED"
            
            print(f"{key:<10} {train_bucket_samples:<15} {train_bucket_duration:<12.2f} {dev_bucket_samples:<15} {dev_bucket_duration:<12.2f} {status:<10}")
            
            if key in self.excluded_buckets:
                excluded_train_duration += train_bucket_duration
                excluded_train_samples += train_bucket_samples
                excluded_dev_duration += dev_bucket_duration
                excluded_dev_samples += dev_bucket_samples
            else:
                train_duration += train_bucket_duration
                train_samples += train_bucket_samples
                dev_duration += dev_bucket_duration
                dev_samples += dev_bucket_samples
        
        print(f"{'-'*80}")
        print(f"{'TRAINING SET SUMMARY':^80}")
        print(f"{'-'*80}")
        print(f"{'Active Train Samples:':<30} {train_samples:>10,}")
        print(f"{'Active Train Hours:':<30} {train_duration:>10.2f}")
        print(f"{'Excluded Train Samples:':<30} {excluded_train_samples:>10,}")
        print(f"{'Excluded Train Hours:':<30} {excluded_train_duration:>10.2f}")
        print(f"{'Total Train Samples:':<30} {train_samples + excluded_train_samples:>10,}")
        print(f"{'Total Train Hours:':<30} {train_duration + excluded_train_duration:>10.2f}")
        
        print(f"\n{'DEVELOPMENT SET SUMMARY':^80}")
        print(f"{'-'*80}")
        print(f"{'Active Dev Samples:':<30} {dev_samples:>10,}")
        print(f"{'Active Dev Hours:':<30} {dev_duration:>10.2f}")
        print(f"{'Excluded Dev Samples:':<30} {excluded_dev_samples:>10,}")
        print(f"{'Excluded Dev Hours:':<30} {excluded_dev_duration:>10.2f}")
        print(f"{'Total Dev Samples:':<30} {dev_samples + excluded_dev_samples:>10,}")
        print(f"{'Total Dev Hours:':<30} {dev_duration + excluded_dev_duration:>10.2f}")
        
        print(f"\n{'OVERALL SUMMARY':^80}")
        print(f"{'-'*80}")
        print(f"{'Active Training Data:':<30} {train_samples:>7,} samples, {train_duration:>6.2f} hours")
        print(f"{'Active Validation Data:':<30} {dev_samples:>7,} samples, {dev_duration:>6.2f} hours")
        print(f"{'Total Active Data:':<30} {train_samples + dev_samples:>7,} samples, {train_duration + dev_duration:>6.2f} hours")
        print(f"{'Excluded Data:':<30} {excluded_train_samples + excluded_dev_samples:>7,} samples, {excluded_train_duration + excluded_dev_duration:>6.2f} hours")
        print(f"{'Grand Total:':<30} {train_samples + dev_samples + excluded_train_samples + excluded_dev_samples:>7,} samples, {train_duration + dev_duration + excluded_train_duration + excluded_dev_duration:>6.2f} hours")
        print(f"{'='*80}")
            
if __name__ == '__main__':
    speech_module = SpeechModule()

    speech_module.create_dataloader()
    speech_module.get_dataset_stats()
    loaders = speech_module.loaders
    print(loaders)
    # for batch in loaders['2.0']['train']:
    #     specs, labels, spec_lens, label_lens, file_name = batch
    #     random_idx = random.randint(0, len(specs) - 1)
    #     print(f"Specs Stats: {specs[random_idx].shape} | Min: {specs[random_idx].min()} | Max: {specs[random_idx].max()} | Mean: {specs[random_idx].mean()} | Std: {specs[random_idx].std()}")
    #     print(f"Transcription: {lc.decode(labels[random_idx].tolist())}")
    #     winsound.PlaySound(config.WAVS_PATH / f"{file_name[random_idx]}.wav", winsound.SND_FILENAME)
    #     plot_spectrogram(specs[random_idx], specs[random_idx], sample_rate=16000)



        