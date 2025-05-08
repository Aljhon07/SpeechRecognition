from pathlib import Path

# Get the absolute path to the current config file's directory
BASE_DIR = Path(__file__).parent.resolve()
LANGUAGE = "en"

COMMON_VOICE_PATH = BASE_DIR / "commonvoice" / LANGUAGE
OUTPUT_PATH = BASE_DIR / "output" / LANGUAGE
WAVS_PATH = COMMON_VOICE_PATH / "wavs"
LOG_DIR = BASE_DIR / "logs"

AUDIO_PARAMS = {
    "SAMPLE_RATE": 16000,
    "N_MELS": 80,
    "HOP_LENGTH": 160
}

H_PARAMS = {
    "BASE_LR": 0.001,
    "TOTAL_EPOCH": 100,
    "VOCAB_SIZE": 31,
    "N_FEATS": 80,
    "VERBOSE": False
}
