import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.models.OneCycle_20_30.LightWeightModel import LightWeightModel as Model
from src.preprocess import LogMelSpectrogram
import os
from tools import audio, utils, language_corpus as lc
import torchaudio
import torch.nn.functional as F
import config
import uuid
from google import genai


client = genai.Client(api_key=os.getenv("GENAI_API_KEY", config.GENAI_API_KEY))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
LOCAL_MODEL_PATH = config.MODEL_DIR / 'OneCycle_20_30'
checkpoint_path = LOCAL_MODEL_PATH / 'checkpoint_epoch_29_val_1.7049.pth'

# Load checkpoint once
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict']) 
model.eval()
model.to(device)

log_mel = LogMelSpectrogram()

def inference(file_path):
    # print(f"Using Model: {checkpoint_path}")
    # print(f"Loading audio file: {file_path}")

    id = uuid.uuid4().hex
    converted_file = audio.to_wav(file_path,  config.UPLOAD_DIR / f"{id}.wav")
    if converted_file is None:
        print(f"Error converting audio file: {converted_file}")
        return None
    
    waveform, sample_rate = torchaudio.load(converted_file)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
    spectrogram = log_mel(waveform).to(device)

    # os.remove(file_path)
    os.remove(converted_file)
    with torch.no_grad():
        output, _ = model(spectrogram)
        output = F.log_softmax(output, dim=-1)
        output[:,:,0] -= 1
        # print(f"Output shape: {output.shape}")
        predicted_ids = torch.argmax(output, dim=-1).transpose(0, 1)
        # print(f"Predicted IDs shape: {predicted_ids.shape}")
        raw_prediction = utils.ctc_decoder(predicted_ids.tolist())
        # print(raw_prediction)
        decoded_pred = lc.decode(raw_prediction, str(LOCAL_MODEL_PATH / f"{config.LANGUAGE}.model"))
        pred = decoded_pred

        if os.getenv("POST_PROCESS_WITH_AI", "false").lower() == "true":
            print("ENHANCEMENT WITH AI ENABLED")
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Correct the grammar and spelling of this speech recognition output so that it makes sense: '{decoded_pred}'. Return only the corrected text without explanations.",
                )
                pred = response.text
            except Exception as e:
                print(f"Error during AI enhancement: {e}")
                pred = decoded_pred

            print(f"Orig: {decoded_pred}")
            print(f"AI Enhanced: {pred}")
            return pred
        
        return pred

if __name__ == '__main__':
    # path = config.COMMON_VOICE_PATH / 'clips' / 'common_voice_en_16759015.mp3'
    path = config.OUTPUT_DIR / 'a.wav'
    result = inference(path)
    print(f"Decoded prediction: {result}")
