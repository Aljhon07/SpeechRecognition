import torch
import torch.nn as nn
import torch.nn.functional as F
from src.neural_net.LightWeightModel import LightWeightModel as Model
from src.preprocess import LogMelSpectrogram
from tools import audio
import torchaudio
import torch.nn.functional as F
class InferenceModel(nn.Module):
    def __init__(self, model_path=None, device=None):
        super(InferenceModel, self).__init__()
        self.device = device
        self.model = Model()

        if model_path:
            self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        x = LogMelSpectrogram()(waveform)
        x = x.transpose(1, 2).contiguous()
        x = self.model(x)
        x = F.log_softmax(x, dim=-1)
        x = x.argmax(dim=-1)
        return x


