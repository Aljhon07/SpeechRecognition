import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import WhisperModel
import config 
import logging
import torch.nn.utils.rnn as rnn_utils

logger = logging.getLogger(__name__)
verbose = config.H_PARAMS['VERBOSE']  

class LightWeightModel(nn.Module):

    def __init__(self, hidden_size=512, num_classes=config.H_PARAMS['VOCAB_SIZE'], n_feats=80, num_layers=3, dropout=0.0):
        super(LightWeightModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.whisper_encoder = WhisperModel.from_pretrained("openai/whisper-tiny").encoder
        
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        
        whisper_output_dim = 384
        
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
        # self.final_fc.bias.data[0] = -0.5

    def _init_hidden(self, batch_size, device):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 2, batch_size, hs, device=device)

    def forward(self, x, lengths, hidden=None):
        # Input should be (batch, n_mels=80, time) - from collate_fn
        if verbose:
            print(f"Model forward - Input shape: {x.shape}")

        with torch.no_grad():
            whisper_outputs = self.whisper_encoder(x)
            whisper_features = whisper_outputs.last_hidden_state  # (batch, time, 384)

        if verbose:
            print(f"Whisper Output Shape: {whisper_features.shape}")

        x = self.adaptation(whisper_features)  # (batch, time, 128)

        if verbose:
            print(f"After adaptation shape: {x.shape}")

        # Transpose for GRU: (time, batch, feature)
        x = x.transpose(0, 1)

        if verbose:
            print(f"After transpose for GRU shape: {x.shape}")

        # Pack the sequence for GRU
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, enforce_sorted=False)
        packed_out, hidden = self.bigru(packed_x, hidden)

        # Unpack the sequence
        out, _ = rnn_utils.pad_packed_sequence(packed_out)

        if verbose:
            print(f"After BiGRU shape: {out.shape}")

        x = self.dropout2(F.gelu(self.layer_norm2(out)))

        if verbose:
            print(f"After layer norm/dropout shape: {x.shape}")

        final_output = self.final_fc(x)

        if verbose:
            print(f"Final output shape (before log_softmax): {final_output.shape}")
        return final_output, hidden  # (time, batch, n_class)

