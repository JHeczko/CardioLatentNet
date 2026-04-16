import torch
from torch import nn
from torch.nn import functional as F
from typing import List,Tuple,Optional,Literal

from torch.nn.functional import dropout


class LSTMConvEncoderBlock(nn.Module):
    def __init__(self,input_dim, output_dim, dropout=0.2):
        super().__init__()

        assert output_dim % 2 == 0

        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=output_dim//2,
            bidirectional=True,
            num_layers=1,
            batch_first=True)

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1,
            stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=5,
            padding=2,
            stride=2)
        self.conv3 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=7,
            padding=3,
            stride=2)

        self.channel_summer = nn.Conv1d(in_channels=output_dim*3, out_channels=output_dim, kernel_size=1, stride=1)

        self.norm_conv = nn.GroupNorm(num_groups=8, num_channels=output_dim)
        self.norm_merge = nn.GroupNorm(num_groups=8, num_channels=output_dim)
        self.norm_lstm = nn.LayerNorm(output_dim)

        self.lstm_dropout = nn.Dropout(p=dropout)
        self.conv_dropout = nn.Dropout(p=dropout)


        self.alpha = nn.Parameter(torch.tensor(0.1))

        self._weight_init()

    def _weight_init(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self,x):
        # input: x = (B, seq_len, channels)

        # x = (B, channels, seq_len)
        x = x.transpose(1, 2)

        # x_conv = (B, output_dim, seq_len/2)
        x_conv1 = self.conv1(x)
        x_conv1 = self.norm_conv(x_conv1)
        x_conv1 = F.gelu(x_conv1)

        x_conv2 = self.conv2(x)
        x_conv2 = self.norm_conv(x_conv2)
        x_conv2 = F.gelu(x_conv2)

        x_conv3 = self.conv3(x)
        x_conv3 = self.norm_conv(x_conv3)
        x_conv3 = F.gelu(x_conv3)


        # x = (B, output_dim*3, seq_len/2)
        x = torch.concat([x_conv1, x_conv2, x_conv3], dim=1)
        x = x * (1 / 3 ** 0.5)

        # x = (B, output_dim, seq_len/2)
        x = self.channel_summer(x)
        x = self.norm_merge(x)
        x = self.conv_dropout(x)
        x = F.gelu(x)

        # x = (B, seq_len/2, output_dim)
        x = x.transpose(1, 2)

        # x = (B, seq_len/2, output_dim)
        x = self.norm_lstm(x)
        identity = x
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        alpha = torch.sigmoid(self.alpha)
        x = x*alpha + identity

        x = self.norm_lstm(x)
        return x


if __name__ == '__main__':
    t = torch.randn(7, 60, 12)

    block = LSTMConvEncoderBlock(12,64, dropout=0.2)
    block2 = LSTMConvEncoderBlock(64,128,dropout=0.2)
    t_out = block(t)
    t2_out = block2(t_out)

    print(t_out.shape)
    print(t2_out.shape)