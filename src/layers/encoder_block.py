import torch
from torch import nn

from . import MultiLayerPerceptron


class EncoderBlock(nn.Module):
    def __init__(self, dim_hidden, num_heads, dropout=0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = MultiLayerPerceptron(
            dim_in=dim_hidden,
            dim_hidden=dim_hidden * 3
        )

        self.att_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

    # x = (batch_size, seq_len, dim_hidden)
    # dim_hidden stays the same through whole encoder/decoder block
    def forward(self, x):
        identity = x
        # x = (batch_size, seq_len, dim_embedded)
        x = self.norm1(x)
        # x = (batch_size, seq_len, dim_embedded)
        x, _ = self.attention(x, x, x, need_weights=False)
        # x = (batch_size, seq_len, dim_embedded)
        x = self.att_dropout(x)
        x = x + identity

        identity = x
        # x = (batch_size, seq_len, dim_embedded)
        x = self.norm2(x)
        # x = (batch_size, seq_len, dim_embedded)
        x = self.mlp(x)
        # x = (batch_size, seq_len, dim_embedded)
        x = self.mlp_dropout(x)
        x = x + identity

        return x