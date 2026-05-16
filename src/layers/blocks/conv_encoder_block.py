from torch import nn


class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, stride=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return out

