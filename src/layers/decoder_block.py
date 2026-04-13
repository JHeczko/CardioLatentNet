import torch
from torch import nn

from . import MultiLayerPerceptron

class DecoderBlock(nn.Module):
    def __init__(self, dim_hidden, num_heads, att_dropout=0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)
        self.norm3 = nn.LayerNorm(dim_hidden)

        self.self_att = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            dropout=att_dropout,
            batch_first=True)

        self.cross_att = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            dropout=att_dropout,
            batch_first=True
        )

        self.mlp = MultiLayerPerceptron(dim_in=dim_hidden, dim_hidden=dim_hidden * 3)

    # x = (batch_size, seq_len, dim_hidden)
    # enc_out = (batch_size, compressed_seq_len, dim_hidden)
    def forward(self, x, enc_out):
        identity = x
        x = self.norm1(x)
        x = self.self_att(x, x, x, need_weights=False)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.cross_att(x, enc_out, enc_out, need_weights=False)
        x = x + identity

        identity = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = x + identity