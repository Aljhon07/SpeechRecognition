from torch.utils.data import Dataset, DataLoader
import config
import random
import torch
import pandas as pd
from tools import language_corpus as lc
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torchaudio.transforms as T
import logging
from pathlib import Path

# Set up logging for dataset shape tracking
logger = logging.getLogger(__name__)

class SpeechDataset(Dataset):
    def __init__(self, tsv_path, precomputed_dir, augmented=False, augmented_prob=0.5):
        """
        Initialize dataset with TSV file path and precomputed .pt files
        
        Args:
            tsv_path: Path to TSV file with metadata
            precomputed_dir: Directory containing precomputed .pt files
            augmented: Whether to apply data augmentation
            augmented_prob: Probability of applying augmentation
        """
        self.tsv_path = Path(tsv_path)
        self.precomputed_dir = Path(precomputed_dir)
        self.augmented = augmented
        self.augmented_prob = augmented_prob
        self.apply_mask = nn.Sequential(
            T.TimeMasking(time_mask_param=15),
            T.FrequencyMasking(freq_mask_param=8))
        self.verbose = config.H_PARAMS["VERBOSE"]
        
        # Load metadata only once to get length and duration
        self.metadata_df = pd.read_csv(self.tsv_path, sep='\t').head(10)
        self.total_duration = self.metadata_df['duration'].sum() / 3600  # Convert to hours
        

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # Get metadata from DataFrame (TSV)
        row = self.metadata_df.iloc[idx]
        file_name = row['file_name']
        transcript = row['transcript']  # Get transcript from TSV
        
        # Load ONLY precomputed spectrograms from .pt file
        pt_file = self.precomputed_dir / f"{file_name}.pt"
        data = torch.load(pt_file, map_location='cpu')
        
        # Extract precomputed spectrograms
        spec = data['spectrogram']  # Already (time, n_mels)
        unpadded_spec = data['unpadded_spectrogram']
        spec_len = unpadded_spec.shape[-1]
        
        # Encode transcript from TSV using SentencePiece
        labels = lc.encode(transcript)
        labels_len = len(labels)
                
        if self.verbose:
            print(f"Dataset[{idx}] - Loaded precomputed data: {file_name}")
            print(f"Dataset[{idx}] - Spec shape: {spec.shape}")
            print(f"Dataset[{idx}] - Spec length: {spec_len}")
            print(f"Dataset[{idx}] - Labels length: {labels_len}")

        # Apply augmentation if enabled
        if self.augmented and random.random() < self.augmented_prob and False:
            if self.verbose:
                print(f"Dataset[{idx}] - Applying augmentation")
            
            # For augmentation, we need (n_mels, time) format
            spec_for_aug = spec.transpose(0, 1)  # (time, n_mels) -> (n_mels, time)
            spec_for_aug = self.apply_mask(spec_for_aug)
            spec = spec_for_aug.transpose(0, 1)  # Back to (time, n_mels)
            
        if self.verbose:
            print(f"Dataset[{idx}] - Final spec shape (time, n_mels): {spec.shape}")
            self.verbose = False  # Only log for the first item

        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        return spec, labels, torch.tensor(spec_len, dtype=torch.long), torch.tensor(labels_len, dtype=torch.long), file_name, unpadded_spec
    
class SpeechModule:
    def __init__(self, use_precomputed=True):
        self.use_precomputed = use_precomputed
        self.loaders = {}
        self.datasets = {}
        
    def _check_precomputed_data(self):
        """Check if any precomputed data exists"""
        precomputed_base = Path(config.PRECOMPUTED_DIR)
        
        # Check if at least one split has both TSV and precomputed directory
        found_any = False
        for split in ['train', 'dev', 'test']:
            tsv_path = config.OUTPUT_DIR / f"{split}.tsv"
            split_dir = precomputed_base / split
            
            if tsv_path.exists() and split_dir.exists():
                found_any = True
                print(f"Found precomputed data for {split} split")
                
        return found_any
        
    def load_librispeech_data(self, subsets=None):
        """Check if LibriSpeech precomputed data exists"""
        if subsets is None:
            subsets = config.LIBRISPEECH_SUBSETS
            
        # Check if precomputed data exists
        if self.use_precomputed and self._check_precomputed_data():
            print("Precomputed data found and ready to use!")
            return True
        else:
            print("Precomputed data not found. Please run preprocessing first:")
            print("python -m src.preprocess")
            raise FileNotFoundError("Precomputed data not available. Run preprocessing first.")
    
    def create_dataloader(self, batch_size=config.H_PARAMS["BATCH_SIZE"]):
        """Create dataloaders from TSV files and precomputed data"""
        precomputed_base = Path(config.PRECOMPUTED_DIR)
        
        datasets = {}
        loaders = {}
        
        # Define split configurations
        split_configs = {
            'train': {'dir': 'train', 'augmented': True, 'shuffle': True},
            'dev': {'dir': 'dev', 'augmented': False, 'shuffle': False},
            'test': {'dir': 'train', 'augmented': False, 'shuffle': False}
        }
        
        # Create datasets and loaders for available splits
        for split_name, config_dict in split_configs.items():
            tsv_path = config.OUTPUT_DIR / f"{config_dict['dir']}.tsv"
            if tsv_path.exists():
                dataset = SpeechDataset(
                    tsv_path, 
                    precomputed_base / config_dict['dir'], 
                    augmented=config_dict['augmented']
                )
                
                # Only create dataloader if we have enough samples
                if len(dataset) > 0:
                    datasets[split_name] = dataset
                    
                    # Adjust drop_last based on dataset size
                    drop_last = len(dataset) >= batch_size
                    
                    loaders[split_name] = DataLoader(
                        dataset, 
                        batch_size=min(batch_size, len(dataset)), 
                        drop_last=drop_last, 
                        shuffle=config_dict['shuffle'], 
                        collate_fn=self.collate_fn
                    )
                    print(f"Created {split_name} dataloader with {len(dataset)} samples")
                else:
                    print(f"Skipping {split_name} - no samples found")
            else:
                print(f"Skipping {split_name} - TSV file not found: {tsv_path}")

        self.datasets = datasets
        self.loaders = loaders
        
        self.get_dataset_stats()
        return loaders
    
    def get_dataset_stats(self):
        """Print dataset statistics"""
        print(f"\nDataset Statistics:")
        
        total_samples = 0
        total_duration = 0
        
        for split_name, dataset in self.datasets.items():
            duration = dataset.total_duration
            samples = len(dataset)
            print(f"{split_name.capitalize()} samples: {samples} ({duration:.2f} hours)")
            total_samples += samples
            total_duration += duration
            
        print(f"Total samples: {total_samples} ({total_duration:.2f} hours)")
    
    def load_and_create_dataloaders(self, subsets=None, batch_size=config.H_PARAMS["BATCH_SIZE"]):
        """Check data availability and create dataloaders"""
        self.load_librispeech_data(subsets)
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
    # Test the refactored dataset loading
    try:
        speech_module = SpeechModule(use_precomputed=True)
        
        # Load data splits
        data_splits = speech_module.load_librispeech_data(subsets=config.LIBRISPEECH_SUBSETS)
        
        # Create dataloaders
        loaders = speech_module.create_dataloader(batch_size=config.H_PARAMS["BATCH_SIZE"])
        
        print("Successfully created dataloaders with precomputed data!")
        
        # Test the dataloaders
        for split_name, dataloader in loaders.items():
            if len(dataloader) > 0:
                for batch in dataloader:
                    specs, labels, spec_lens, label_lens, file_names, unpadded_specs = batch
                    print(f"{split_name.capitalize()} batch - Specs: {specs.shape}, Labels: {labels.shape}")
                    print(f"Sample file names: {file_names[:min(3, len(file_names))]}")

                    random_idx = random.randint(0, specs.size(0) - 1)
                    print(f"Random sample from batch - Spec shape: {specs[random_idx].shape}, Label length: {label_lens[random_idx]}")
                    print(f"Unpadded spec shape: {unpadded_specs[random_idx].shape}")
                    print(f"Label: {labels[random_idx]}")
                    print(f"File name: {file_names[random_idx]}")

                    break  # Just test one batch per split
                    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo fix this, run preprocessing first:")
        print("python -m src.preprocess")