import torch
import torch.nn.functional as F
from inference.models.librispeechv2.LightWeightModel import LightWeightModel as Model
from src.preprocess import WhisperLogMelSpectrogram
import os
from tools import audio, utils, language_corpus as lc
import torchaudio
import config
import uuid
from google import genai
import logging

# Set up logging for inference shape tracking
logger = logging.getLogger(__name__)

client = genai.Client(api_key=config.GENAI_API_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
LOCAL_MODEL_PATH = config.MODEL_DIR / 'librispeechv2'
checkpoint_path = LOCAL_MODEL_PATH / 'checkpoint_epoch_15_val_0.3760.pth'

# Load checkpoint once
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict']) 
model.eval()
model.to(device)

# Use WhisperLogMelSpectrogram for consistent preprocessing
log_mel = WhisperLogMelSpectrogram()

def inference(file_path, use_post_processing=False):
    # print(f"Using Model: {checkpoint_path}")
    # print(f"Loading audio file: {file_path}")

    # Play the audio file for preview
    # utils.play_sound(file_path)

    id = uuid.uuid4().hex
    converted_file = audio.to_wav(file_path,  config.UPLOAD_DIR / f"{id}.flac")
    if converted_file is None:
        print(f"Error converting audio file: {converted_file}")
        return None
    
    waveform, sample_rate = torchaudio.load(converted_file)
    # logger.info(f"Inference - Loaded waveform shape: {waveform.shape}, sample_rate: {sample_rate}")
    
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        # logger.info(f"Inference - Resampled waveform shape: {waveform.shape}")
        
    # WhisperLogMelSpectrogram returns shape (1, n_mels, time)
    spectrogram, unpadded_spec = log_mel(waveform)
    spectrogram = spectrogram.to(device)    
    unpadded_len = unpadded_spec.shape[-1]
    lengths = torch.tensor([unpadded_len], dtype=torch.long)
    # logger.info(f"Inference - Spectrogram shape after WhisperLogMel: {spectrogram.shape}")
    
    # Remove batch dimension and transpose to (time, n_mels) for model input
    # if spectrogram.dim() == 3 and spectrogram.shape[0] == 1:
        # spectrogram = spectrogram.squeeze(0).transpose(0, 1).unsqueeze(0)  # (1, time, n_mels)
        # logger.info(f"Inference - Spectrogram shape after reshape: {spectrogram.shape}")

    # os.remove(file_path)
    os.remove(converted_file)
    with torch.no_grad():
        # logger.info(f"Inference - Input to model shape: {spectrogram.shape}")
        # logger.info(f"Inference - Input lengths: {lengths}")
        output, hidden = model(spectrogram, lengths=lengths // 2)
        # logger.info(f"Inference - Model output shape: {output.shape}")
        output = F.log_softmax(output, dim=-1)
        # print(f"Output shape: {output.shape}")
        predicted_ids = torch.argmax(output, dim=-1).transpose(0, 1)
        # print(f"Predicted IDs shape: {predicted_ids.shape}")
        raw_prediction = utils.ctc_decoder(predicted_ids.tolist())
        # print(raw_prediction)
        decoded_pred = lc.decode(raw_prediction, str(LOCAL_MODEL_PATH / f"{config.LANGUAGE}.model"))

        pred = decoded_pred

        if os.getenv("POST_PROCESS_WITH_AI", str(use_post_processing)).lower() == "true":
            print(f"Using Post Processing with AI")
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Correct the grammar and spelling of this speech recognition output so that it makes sense: '{decoded_pred}'. Return only the corrected text without explanations.",
                )
                pred = response.text
                print(f"Orig: {decoded_pred}")
                print(f"AI Enhanced: {pred}")
            except Exception as e:
                print(f"Error during AI enhancement: {e}")
                pred = decoded_pred
                return pred

        return pred

if __name__ == '__main__':
    # path = config.COMMON_VOICE_PATH / 'clips' / 'common_voice_en_16759015.mp3'
    
    # use the librispeech dataset sample file for testing
    path = "D:/Projects/SpeechRecognition/librispeech/LibriSpeech/dev-clean/174/84280/174-84280-0007.flac"
    print(f"Using Model: {checkpoint_path}")
    inference(path)
