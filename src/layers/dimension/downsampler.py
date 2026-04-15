import torch
from torch import nn

class Downsampler(nn.Module):
    """Downsamples temporal sequences using 1D convolution.

        This module reduces the sequence length by a factor defined by the stride,
        effectively performing a feature-preserving spatial/temporal reduction.

        Args:
            hidden_dim (int): The number of input and output channels.
            stride (int, optional): The stride of the convolution, which determines
                the downsampling factor. Defaults to 2.
            kernel_size (int, optional): The size of the sliding window. Defaults to 3.

        Attributes:
            conv (nn.Conv1d): 1D convolutional layer used for downsampling.
    """

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
        """Processes the input to reduce its temporal dimension.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).

                Returns:
                    torch.Tensor: Downsampled tensor of shape
                        (batch_size, seq_len // stride, hidden_dim).
        """
        # x = (B, hidden_dim, seq_len)
        x = x.transpose(2, 1)
        # x = (B, hidden_dim, seq_len/stride)
        x = self.conv(x)
        # x = (B, seq_len/stride, hidden_dim)
        x = x.transpose(2, 1)

        return x


