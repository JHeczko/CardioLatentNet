import torch
from torch import nn

from . import MultiLayerPerceptron


class EncoderBlock(nn.Module):
    def __init__(self, dim_embedded, num_heads, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim_embedded)
        self.norm2 = nn.LayerNorm(dim_embedded)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim_embedded,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = MultiLayerPerceptron(
            dim_in=dim_embedded,
            dim_hidden=dim_embedded * 3
        )

        self.att_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # --- Attention block ---
        identity = x
        x = self.norm1(x)

        x, _ = self.attention(x, x, x, need_weights=False)
        x = self.att_dropout(x)

        x = x + identity

        # --- MLP block ---
        identity = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = self.mlp_dropout(x)

        x = x + identity

        return x