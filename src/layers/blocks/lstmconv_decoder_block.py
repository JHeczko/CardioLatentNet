import torch
from torch import nn
from torch.nn import functional as F



class LSTMConvDecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim//2,
            bidirectional=True,
            num_layers=1,
            batch_first=True
        )

        self.norm_lstm = nn.LayerNorm(input_dim)
        self.dropout_lstm = nn.Dropout(dropout)

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1
        )

        self.norm_conv = nn.GroupNorm(8, output_dim)
        self.dropout_conv = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.tensor(0.0))


    def forward(self, x):

        x = self.norm_lstm(x)
        identity = x
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x)

        alpha = torch.sigmoid(self.alpha)
        x = identity + alpha * x

        # (B, C, T)
        x = x.transpose(1, 2)

        # upsample
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        # conv
        x = self.conv(x)
        x = self.norm_conv(x)
        self.dropout_conv(x)
        x = F.gelu(x)

        # back
        x = x.transpose(1, 2)


        return x


if __name__ == '__main__':
    t = torch.randn(7, 30, 64)

    block = LSTMConvDecoderBlock(64,32)

    t_out = block(t)

    print(t_out.shape)