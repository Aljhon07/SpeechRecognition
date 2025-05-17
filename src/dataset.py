from torch.utils.data import Dataset, DataLoader
from src.preprocess import BucketAudio
import os
import config
import json
import random
import torchaudio
import torch
from src.preprocess import LogMelSpectrogram
from tools.utils import double_vad
import winsound
from tools import language_corpus as lc
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torchaudio.transforms as T

class SpeechDataset(Dataset):
    def __init__(self, data, augmented=False, augmented_prob=0.5):
        self.data = data
        self.augmented = augmented
        self.augmented_prob = augmented_prob
        self.logmel = LogMelSpectrogram()
        self.total_duration = sum(item['duration'] for item in data) / (60 * 60)
        self.apply_mask = nn.Sequential(
            T.TimeMasking(time_mask_param=25),
            T.FrequencyMasking(freq_mask_param=15))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        file_name = data['file_name']
        waveform, sr = torchaudio.load(config.WAVS_PATH / f"{file_name}.wav")
        spec = self.logmel(waveform)

        if self.augmented and random.random() < self.augmented_prob:
            spec = self.apply_mask(spec)

        spec_len = spec.shape[2]
        transcription = data['transcription']
        labels = lc.encode(transcription)
        labels_len = len(labels)

        return spec.squeeze(0).transpose(0,1).contiguous(), torch.tensor(labels, dtype=torch.long), torch.tensor(spec_len, dtype=torch.long), torch.tensor(labels_len, dtype=torch.long), file_name
    
    def preprocess(self, audio):
        audio = double_vad(audio)
        audio = self.logmel(audio)
        return audio
    
class SpeechModule:
    def __init__(self, data=None, excluded_buckets=['0.0', '30.0']):
        self.data = data
        self.bucket = BucketAudio()
        self.loaders = {}
        self.datasets = {}
        self.excluded_buckets = excluded_buckets

        self.load_data()
        self.create_dataloader()
    
    def load_data(self):
        if not os.path.exists(config.OUTPUT_DIR / 'buckets'):
            self.bucket.init()
        self.data = self.bucket.load_buckets()
    
    def create_dataloader(self, batch_size=config.H_PARAMS["BATCH_SIZE"], val_split=0.1):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        keys = list(self.data.keys())
        keys = sorted(keys, key=float)
        for idx, key in enumerate(keys):
            if key in self.excluded_buckets:
                continue
            items = self.data[key]
            random.shuffle(items)
            split_idx = int(len(items) * (1 - val_split))
            train_data = items[:split_idx]
            val_data = items[split_idx:]

            train_dataset = SpeechDataset(train_data, augmented=True)
            val_dataset = SpeechDataset(val_data, augmented=False)
            
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

        specs = pad_sequence(specs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)

        return specs.transpose(1, 2), labels, torch.tensor(spec_lens, dtype=torch.long), torch.tensor(label_lens, dtype=torch.long), file_name
    
    def get_dataset_stats(self):
        if self.datasets is None:
            raise ValueError("Data not loaded. Please load data first.")
        
        overall_duration = 0
        overall_samples = 0
        invalidated_durations = 0
        invalidated_samples = 0
        for key, dataset in self.datasets.items():

            dataset_duration = dataset['train'].total_duration + dataset['val'].total_duration
            dataset_sample_size = len(dataset['train']) + len(dataset['val'])
            print(f"Dataset {key}: {dataset_sample_size} samples, {dataset_duration:.2f} hours")
            if key in self.excluded_buckets:
                invalidated_durations += dataset_duration
                invalidated_samples += dataset_sample_size
            overall_duration += dataset_duration
            overall_samples += dataset_sample_size

        print(f"{"="*100}\nOverall dataset duration: {overall_duration:.2f} hours")
        print(f"Overall dataset size: {overall_samples} samples\n{"="*100}")
            
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



        