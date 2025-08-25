# Speech Recognition with Whisper Integration

This project implements a speech recognition system using a hybrid approach that combines OpenAI's Whisper tiny model as a feature extractor with a custom BiGRU neural network for transcription.

## Architecture

- **Feature Extractor**: Frozen Whisper tiny encoder (384-dim features)
- **Adaptation Layer**: Linear layer to bridge Whisper output to BiGRU input
- **Sequence Model**: Bidirectional GRU with CTC loss for sequence-to-sequence learning
- **Web Interface**: Flask API for audio transcription with mobile app support

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for Docker

### 1. Environment Setup

```bash
# Copy the environment template
cp .env.example .env

# Edit .env file with your Google GenAI API key
# GENAI_API_KEY=your-actual-api-key-here
```

### 2. Build the Application

```bash
# Build the Docker image
docker-compose build

# Or use the convenience script (Windows)
docker-run.bat build
```

### 3. Run the Application

#### Production Mode

```bash
# Start the speech recognition service
docker-compose up -d speech-recognition

# Access the web interface at http://localhost:8080
```

#### Development Mode

```bash
# Start with live code reloading
docker-compose -f docker-compose.dev.yml up speech-recognition-dev
```

#### Training Mode

```bash
# Preprocess data only
docker-compose --profile preprocess up preprocess

# Full training pipeline
docker-compose --profile training up training
```

## Dataset Gathering

## Data Preprocessing

1.  **Divide Datasets**: Divide datasets into 80/10/10 for training/dev/testing respectively.
2.  **Convert to Appropriate Format**: 16000 sample rate, and wav format.
3.  **Adding Noise**: Random noise is added to the audio.
4.  **Time Stretching**: The audio is stretched by a factor of
5.  **Pitch Shifting**: The pitch of the audio is shifted by 2 steps.
6.  **Extract Feature**: extract Mel-spectogram features from the audio files and saved as `.npy` files for model training.
7.  **Tokenization of Transcriptions**: Tokenize text....

## Model Training

Includes callbacks for checkpointing and early stopping.

Model Arhictecture and Training Configuration:

1.  CNN
2.  RNN (BiGRU)
3.  Set up an optimizer (AdamW)
4.  CTC Loss
5.  Implement scheduler

## Evaluation

## Inference

## Finetune

## Export

## Requirements

- Python 3.x
- librosa
- numpy
- pandas
- PyTorch
- sentencepiece

<<<<<<< HEAD

## Setup

1. pip install -r requirements.txt
2. Install FFMPEG - https://www.youtube.com/watch?v=JR36oH35Fgg
3. Install Dataset - https://commonvoice.mozilla.org/en/datasets - ung Common Voice Corpus 1
