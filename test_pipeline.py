#!/usr/bin/env python3
"""
Comprehensive test script for LibriSpeech dataset pipeline.
Run this script to validate that your LibriSpeech implementation is working correctly.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SpeechModule, test_librispeech_pipeline, quick_test_dataloader
import config
from tools import language_corpus as lc


def main():
    """Main test runner"""
    print("🧪 LibriSpeech Dataset Pipeline Test")
    print("=" * 60)
    
    # Check if language corpus model exists
    model_path = config.OUTPUT_DIR / f"{config.LANGUAGE}.model"
    if not model_path.exists():
        print("⚠️  Warning: SentencePiece model not found!")
        print(f"Expected: {model_path}")
        print("Run the following command first:")
        print("python -m src.preprocess")
        return False
    
    # Run comprehensive test
    print("\n🔍 Running comprehensive pipeline test...")
    success = test_librispeech_pipeline()
    
    if success:
        print("\n🚀 Running quick demonstration...")
        try:
            quick_test_dataloader()
            print("\n✅ All tests completed successfully!")
            print("\n📋 Summary:")
            print("   - LibriSpeech data loading: ✅")
            print("   - Train/Dev separation: ✅")
            print("   - SpeechDataset creation: ✅")
            print("   - DataLoader batching: ✅")
            print("   - Audio preprocessing: ✅")
            print("   - Text tokenization: ✅")
            
            print("\n📚 How to use in your training:")
            print("""
from src.dataset import SpeechModule

# Create dataloaders
speech_module = SpeechModule()
loaders = speech_module.load_and_create_dataloaders(
    subsets=["train-clean-100", "dev-clean", "test-clean"],
    batch_size=32
)

# Use in training loop
for epoch in range(num_epochs):
    for batch in loaders['train']:
        specs, labels, spec_lens, label_lens, file_names = batch
        # Your training code here
        pass
            """)
            
        except Exception as e:
            print(f"⚠️  Demo failed: {e}")
            success = False
    else:
        print("\n❌ Pipeline test failed!")
        print("\n🔧 Troubleshooting steps:")
        print("1. Ensure you have run: python -m src.preprocess")
        print("2. Check that LibriSpeech data exists in:", config.LIBRISPEECH_PATH)
        print("3. Verify config.LIBRISPEECH_SUBSETS contains valid subset names")
        print("4. Check that all required dependencies are installed")
    
    return success


def test_individual_components():
    """Test individual components separately for debugging"""
    print("\n🔧 Testing Individual Components")
    print("=" * 40)
    
    try:
        # Test 1: Config validation
        print("1. Testing configuration...")
        print(f"   LibriSpeech path: {config.LIBRISPEECH_PATH}")
        print(f"   Subsets: {config.LIBRISPEECH_SUBSETS}")
        print(f"   Output dir: {config.OUTPUT_DIR}")
        print("   ✅ Config OK")
        
        # Test 2: Language corpus
        print("\n2. Testing language corpus...")
        test_text = "hello world"
        encoded = lc.encode(test_text)
        decoded = lc.decode(encoded)
        print(f"   Original: '{test_text}'")
        print(f"   Encoded: {encoded}")
        print(f"   Decoded: '{decoded}'")
        print("   ✅ Language corpus OK")
        
        # Test 3: SpeechModule init
        print("\n3. Testing SpeechModule...")
        speech_module = SpeechModule()
        print("   ✅ SpeechModule initialization OK")
        
        # Test 4: LibriSpeech dataset access
        print("\n4. Testing LibriSpeech dataset access...")
        import torchaudio
        test_subset = config.LIBRISPEECH_SUBSETS[0] if config.LIBRISPEECH_SUBSETS else "test-clean"
        
        try:
            dataset = torchaudio.datasets.LIBRISPEECH(
                root=config.LIBRISPEECH_PATH,
                url=test_subset,
                download=False  # Don't download, just check if exists
            )
            print(f"   Dataset size: {len(dataset)}")
            if len(dataset) > 0:
                waveform, sr, transcript, speaker_id, chapter_id, utterance_id = dataset[0]
                print(f"   Sample audio shape: {waveform.shape}")
                print(f"   Sample rate: {sr}")
                print(f"   Sample transcript: '{transcript[:50]}...'")
            print("   ✅ LibriSpeech access OK")
        except Exception as e:
            print(f"   ❌ LibriSpeech access failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting LibriSpeech pipeline validation...\n")
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        # Run main test
        success = main()
        exit_code = 0 if success else 1
    else:
        print("\n❌ Component tests failed. Cannot proceed with main test.")
        exit_code = 1
    
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)