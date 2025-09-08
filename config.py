import os
from pathlib import Path
import logging

# Get the absolute path to the current config file's directory
BASE_DIR = Path(__file__).parent.resolve()
LANGUAGE = "en"
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "AIzaSyCy9YBM9s8K3jTnk4jvs7jMVz8ln5CJnZM")

LIBRISPEECH_PATH = BASE_DIR / "librispeech"
COMMON_VOICE_PATH = BASE_DIR / "commonvoice" / LANGUAGE
OUTPUT_DIR = BASE_DIR / "output" / LANGUAGE
WAVS_PATH = COMMON_VOICE_PATH / "wavs"
LOG_DIR = BASE_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
UPLOAD_DIR = BASE_DIR / "uploads"
SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / 'inference' / "models"

AUDIO_PARAMS = {
    "SAMPLE_RATE": 16000,  # Whisper standard
    "N_MELS": 80,          # Whisper standard 
    "HOP_LENGTH": 160      # Whisper standard (10ms hop at 16kHz)
}

H_PARAMS = {
    "BASE_LR": 0.003,
    "TOTAL_EPOCH": 30,
    "VOCAB_SIZE": 5000,
    "N_FEATS": 80,
    "VERBOSE": True,
    "BATCH_SIZE": 8
}

# Logging configuration for shape tracking
LOGGING_CONFIG = {
    "SHAPE_TRACKING": True,  # Enable/disable shape tracking logs
    "LEVEL": "INFO",         # Options: DEBUG, INFO, WARNING, ERROR
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# How to use logging levels:
# - DEBUG: Very detailed shape info at every step (use for debugging)
# - INFO: Key shape transformations and model I/O (recommended for monitoring)
# - WARNING: Only important warnings
# - ERROR: Only errors
# 
# To change logging level during runtime:
# import logging
# logging.getLogger().setLevel(logging.DEBUG)  # or INFO, WARNING, ERROR

# Configure logging based on settings
def setup_logging():
    level = getattr(logging, LOGGING_CONFIG["LEVEL"])
    logging.basicConfig(
        level=level,
        format=LOGGING_CONFIG["FORMAT"],
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(LOG_DIR / "shape_tracking.log", mode='a')  # File output
        ]
    )
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

# Call setup_logging when config is imported
if LOGGING_CONFIG["SHAPE_TRACKING"]:
    setup_logging()
