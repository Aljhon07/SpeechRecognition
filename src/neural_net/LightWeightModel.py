import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import WhisperModel
import config 
verbose = config.H_PARAMS["VERBOSE"]

class LightWeightModel(nn.Module):

    def __init__(self, hidden_size=512, num_classes=config.H_PARAMS['VOCAB_SIZE'], n_feats=80, num_layers=3, dropout=0.2):
        super(LightWeightModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Replace CNN with frozen Whisper tiny encoder
        self.whisper_encoder = WhisperModel.from_pretrained("openai/whisper-tiny").encoder
        
        # Freeze all Whisper parameters
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        
        # Whisper tiny outputs 384-dimensional features
        whisper_output_dim = 384
        
        # Adaptation layer to connect Whisper output to BiGRU
        self.adaptation = nn.Sequential(
            nn.Linear(whisper_output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.bigru = nn.GRU(input_size=128, hidden_size=512,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=True)
        
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size * 2, num_classes)
        self.final_fc.bias.data[0] = -5.0

    def _init_hidden(self, batch_size, device):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 2, batch_size, hs, device=device)

    def forward(self, x, hidden=None):
        # Input should be (batch, n_mels=80, time)
        if verbose:
            print(f"Input Shape: {x.shape}")
        
        # Ensure input is (batch, n_mels, time) format for Whisper
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove channel dimension if present
        
        # Forward through frozen Whisper encoder
        with torch.no_grad():
            whisper_outputs = self.whisper_encoder(x)
            whisper_features = whisper_outputs.last_hidden_state  # (batch, time, 384)
        
        if verbose:
            print(f"Whisper Output Shape: {whisper_features.shape}")
        
        # Adapt features for BiGRU
        x = self.adaptation(whisper_features)  # (batch, time, 128)
        
        if verbose:
            print(f"After Adaptation Shape: {x.shape}")
        
        # Transpose for GRU: (time, batch, feature)
        x = x.transpose(0, 1)
        
        if verbose:
            print(f"After Transpose Shape: {x.shape}")
        
        out, hidden = self.bigru(x, hidden)
        
        if verbose:
            print(f"After GRU Shape: {out.shape}")
        
        x = self.dropout2(F.gelu(self.layer_norm2(out)))
        
        if verbose:
            print(f"After Layer Norm Shape: {x.shape}")
        
        return self.final_fc(x), hidden # (time, batch, n_class)

