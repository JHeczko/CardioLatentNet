import torch
from torch import nn

class Upsampler(nn.Module):
    def __init__(self, hidden_dim, stride = 2, kernel_size=3):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            stride=stride,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            output_padding=stride - 1)
    # x = (B, seq_len, hidden_dim)
    def forward(self, x):
        # x = (B, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        # x = (B, hidden_dim, seq_len*stride)
        x = self.conv(x)
        # x = (B, seq_len*stride, hidden_dim)
        x = x.transpose(1, 2)

        return x

if __name__ == '__main__':
    # x = (B, seq_len, hidden_dim)
    hidden_dim = 128
    seq_len = 60
    kernel_size = 7
    stride = 5

    t = torch.ones(3, seq_len, hidden_dim)
    conv1d = nn.ConvTranspose1d(hidden_dim, hidden_dim, stride=stride, kernel_size=kernel_size, padding=kernel_size//2, output_padding=stride-1)
    t_out = conv1d(t.transpose(2,1))

    print(f"Before: {t.shape}\nAfter: {t_out.transpose(2,1).shape}")