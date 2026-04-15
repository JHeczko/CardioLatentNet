import torch
from torch import nn

class Upsampler(nn.Module):
    """Upsamples temporal sequences using 1D transposed convolution.

        This module increases the sequence length by a factor defined by the stride,
        reconstructing higher-resolution temporal features from compressed representations.

        Args:
            hidden_dim (int): The number of input and output channels.
            stride (int, optional): The stride of the transposed convolution,
                which determines the upsampling factor. Defaults to 2.
            kernel_size (int, optional): The size of the sliding window. Defaults to 3.

        Attributes:
            conv (nn.ConvTranspose1d): 1D transposed convolutional layer for upsampling.
    """
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
        """Processes the input to increase its temporal dimension.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).

                Returns:
                    torch.Tensor: Upsampled tensor of shape
                        (batch_size, seq_len * stride, hidden_dim).
        """

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