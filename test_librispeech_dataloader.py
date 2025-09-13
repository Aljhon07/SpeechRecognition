#!/usr/bin/env python3
"""
Test script to demonstrate LibriSpeech DataLoader usage.
This shows how to load LibriSpeech data and create DataLoaders for training.
"""

from src.dataset import SpeechModule
import config

def test_librispeech_dataloader():
    """Test LibriSpeech DataLoader functionality"""
    
    print("=== LibriSpeech DataLoader Test ===")
    
    # Initialize the speech module
    speech_module = SpeechModule()
    
    # Example 1: Using only test-clean (current config)
    print("\n--- Example 1: Using test-clean only ---")
    loaders = speech_module.load_and_create_dataloaders(
        subsets=["test-clean"],  # Will automatically go to dev set
        batch_size=8
    )
    
    # Example 2: Using multiple subsets (you would uncomment these in config.py)
    print("\n--- Example 2: Multiple subsets (hypothetical) ---")
    print("To use this example, uncomment the desired subsets in config.py:")
    print("LIBRISPEECH_SUBSETS = [")
    print('    "train-clean-100",  # -> Training set')
    print('    "dev-clean",        # -> Development set') 
    print('    "test-clean",       # -> Development set')
    print("]")
    
    # Example 3: Manual subset specification
    print("\n--- Example 3: Manual subset specification ---")
    try:
        # This will only work if you have these subsets downloaded
        speech_module_manual = SpeechModule()
        # Uncomment the line below to test with multiple subsets:
        # loaders_manual = speech_module_manual.load_and_create_dataloaders(
        #     subsets=["train-clean-100", "dev-clean", "test-clean"],
        #     batch_size=16
        # )
        print("Manual subset loading would separate data as:")
        print("  - 'train-clean-100' -> Training DataLoader")
        print("  - 'dev-clean' -> Development DataLoader") 
        print("  - 'test-clean' -> Development DataLoader")
        
    except Exception as e:
        print(f"Manual loading not available: {e}")
    
    # Test the current dataloader
    print("\n--- Testing Current DataLoader ---")
    if 'train' in loaders and len(speech_module.train_data) > 0:
        print("Training DataLoader available")
        train_batch = next(iter(loaders['train']))
        specs, labels, spec_lens, label_lens, file_names = train_batch
        print(f"Train batch - Specs: {specs.shape}, Labels: {labels.shape}")
        
    if 'val' in loaders and len(speech_module.dev_data) > 0:
        print("Validation DataLoader available")
        val_batch = next(iter(loaders['val']))
        specs, labels, spec_lens, label_lens, file_names = val_batch
        print(f"Val batch - Specs: {specs.shape}, Labels: {labels.shape}")
        print(f"Sample file names: {file_names[:3]}")
    
    return loaders

def usage_example():
    """Show how to use LibriSpeech DataLoader in your training script"""
    
    print("\n=== Usage Example for Training ===")
    print("""
# In your training script:

from src.dataset import SpeechModule
import config

# 1. Create speech module
speech_module = SpeechModule()

# 2. Load data and create dataloaders
dataloaders = speech_module.load_and_create_dataloaders(
    subsets=config.LIBRISPEECH_SUBSETS,  # Uses config settings
    batch_size=config.H_PARAMS["BATCH_SIZE"]
)

# 3. Use in training loop
for epoch in range(config.H_PARAMS["TOTAL_EPOCH"]):
    # Training phase
    if len(speech_module.train_data) > 0:
        for batch in dataloaders['train']:
            specs, labels, spec_lens, label_lens, file_names = batch
            # Your training code here
            pass
    
    # Validation phase  
    if len(speech_module.dev_data) > 0:
        for batch in dataloaders['val']:
            specs, labels, spec_lens, label_lens, file_names = batch
            # Your validation code here
            pass

# 4. Access dataset statistics
print(f"Training samples: {len(speech_module.train_data)}")
print(f"Validation samples: {len(speech_module.dev_data)}")
    """)

if __name__ == "__main__":
    test_librispeech_dataloader()
    usage_example()