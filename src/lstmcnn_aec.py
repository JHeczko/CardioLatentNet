import math
import torch
from torch import nn
from torch.nn import functional as F

from src.layers.blocks import LSTMConvDecoderBlock, LSTMConvEncoderBlock, VariationalBlock

class LstmCnnAEC(nn.Module):
    def __init__(self, blocks, latent_dim, seq_len, ecg_channels, dropout):
        super().__init__()

        current_channel_size = 32
        current_seq_len = seq_len

        first = True
        self.encoder_blocks = nn.ModuleList()
        for i in range(blocks):
            if first:
                self.encoder_blocks.append(LSTMConvEncoderBlock(input_dim=ecg_channels, output_dim=current_channel_size, dropout=dropout))
                first = False
                current_seq_len = math.ceil(current_seq_len / 2)
            else:
                self.encoder_blocks.append(LSTMConvEncoderBlock(input_dim=current_channel_size, output_dim=current_channel_size*2, dropout=dropout))
                current_channel_size = current_channel_size*2
                current_seq_len = math.ceil(current_seq_len/2)

        print(current_seq_len, current_channel_size)
        self.flatten = nn.Flatten()
        self.variational_latent = VariationalBlock(input_dim=current_channel_size*current_seq_len,latent_dim=latent_dim)
        self.projection = nn.Linear(latent_dim, current_channel_size*current_seq_len)

        self.decoder_blocks = nn.ModuleList()
        for i in range(blocks):
            self.decoder_blocks.append(LSTMConvDecoderBlock(current_channel_size, current_channel_size//2, dropout=dropout))
            current_channel_size = current_channel_size//2

        self.out_projection = nn.Linear(current_channel_size, ecg_channels)


    def forward(self, x):

        for block in self.encoder_blocks:
            x = block(x)

        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x, mu, logvar = self.variational_latent(x)
        x = self.projection(x)
        x = x.view(x.shape[0], )
        print(x.shape)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        x = self.out_projection(x)

        return x, mu, logvar


if __name__ == "__main__":
    t = torch.randn(9, 60, 12)

    vae = LstmCnnAEC(3, 20, 60, 12, 0.2)

    t_out = vae(t)

    print(t_out.shape)