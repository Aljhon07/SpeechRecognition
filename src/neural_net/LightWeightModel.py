import torch
import torch.nn as nn
from torch.nn import functional as F
import config 
verbose = config.H_PARAMS["VERBOSE"]
class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        x = x.transpose(1, 2)
        # x = self.norm(self.dropout(F.gelu(x)))
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x

class LightWeightModel(nn.Module):

    def __init__(self, hidden_size=512, num_classes=config.H_PARAMS['VOCAB_SIZE'], n_feats=80, num_layers=1, dropout=0.2):
        super(LightWeightModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, 128, 7, 2, padding=7//2),
            ActDropNormCNN1D(128, dropout, keep_shape=True),
            nn.Conv1d(128, 256, 3, 1, padding=3//2),
            ActDropNormCNN1D(256, dropout, keep_shape=True),
            nn.Conv1d(256, 128, 3, 1, padding=3//2),
            ActDropNormCNN1D(128, dropout),
        )

        self.dense = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.bigru = nn.GRU(input_size=128, hidden_size=512,
                            num_layers=num_layers, dropout=0.0,
                            bidirectional=True)
        
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size * 2, num_classes)
        self.final_fc.bias.data[0] = -5.0

    def _init_hidden(self, batch_size, device):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 2, batch_size, hs, device=device)

    def forward(self, x, hidden=None):
        x = x.squeeze(1).contiguous()  # batch, feature, time
        if verbose:
            print(f"Input Shape: {x.shape} | Contiguous: {x.is_contiguous()}")
        x = self.cnn(x) # batch, time, feature
        if verbose:
            print(f"After CNN Shape: {x.shape} | Contiguous: {x.is_contiguous()}")
        x = self.dense(x) # batch, time, feature
        if verbose:
            print(f"After Dense Shape: {x.shape} | Contiguous: {x.is_contiguous()}")
        x = x.transpose(0, 1) # time, batch, feature
        if verbose:
            print(f"After Transpose Shape: {x.shape} | Contiguous: {x.is_contiguous()}")
        out, hidden = self.bigru(x, hidden)
        if verbose:
            print(f"After LSTM Shape: {out.shape} | Contiguous: {out.is_contiguous()}")
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        if verbose:
            print(f"After Layer Norm Shape: {x.shape} | Contiguous: {x.is_contiguous()}")
        return self.final_fc(x), hidden # (time, batch, n_class)

