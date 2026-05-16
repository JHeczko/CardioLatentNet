from torch import nn


class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, stride=2):
        super().__init__()

        kernel_size = 4 if stride == 2 else 3

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual = (in_channels==out_channels and stride == 1 )

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return out