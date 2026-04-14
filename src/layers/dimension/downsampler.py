import torch
from torch import nn

class Downsampler(nn.Module):
    def __init__(self, hidden_dim, stride = 2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            stride=stride,
            kernel_size=kernel_size,
            padding=kernel_size//2)

    # x = (B, seq_len, hidden_dim)
    def forward(self, x):
        # x = (B, hidden_dim, seq_len)
        x = x.transpose(2, 1)
        # x = (B, hidden_dim, seq_len/stride)
        x = self.conv(x)
        # x = (B, seq_len/stride, hidden_dim)
        x = x.transpose(2, 1)

        return x


