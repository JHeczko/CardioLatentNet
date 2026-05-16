import torch
from torch import nn


class LSTMConvProcessBlock(nn.Module):
    """Blok przetwarzający bez downsamplingu — do używania między EncoderBlockami."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim // 2,
            bidirectional=True,
            num_layers=1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: (B, seq_len, dim)
        x = self.norm(x)
        identity = x
        x, _ = self.lstm(x)
        x = self.dropout(x)
        alpha = torch.sigmoid(self.alpha)
        return identity + alpha * x