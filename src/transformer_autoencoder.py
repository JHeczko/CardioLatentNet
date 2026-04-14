import torch
from torch import nn

import warnings

from layers.transformer import EncoderBlock, DecoderBlock
from layers.encoding import PositionalEncoding
from layers.dimension import Upsampler, Downsampler

class TransformerAutoEncoder(nn.Module):
    def __init__(self, num_encoders, num_decoders, num_att_heads, input_dim, hidden_dim, seq_len):
        super().__init__()

        if num_encoders % 2  != 0:
            warnings.warn("num_encoders should be divisible by 2. Other wise, there will be block without downsample")

        # ============== ENCODER PART ==============
        self.encoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_pos_enc = PositionalEncoding(max_context_length=seq_len, dim_embedded=hidden_dim)

        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(num_encoders):
            self.encoders.append(EncoderBlock(dim_hidden=hidden_dim, num_heads=num_att_heads, dropout=0.2))
            if((i+1)%2 == 0 ):
                self.downsamplers.append(Downsampler(hidden_dim=hidden_dim, stride=2,kernel_size=3))

        # ============== DECODER PART ==============
        self.query = nn.Parameter(torch.randn(seq_len, hidden_dim))

        self.decoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.decoder_pos_enc = PositionalEncoding(max_context_length=seq_len, dim_embedded=hidden_dim)

        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in range(num_decoders):
            self.decoders.append(DecoderBlock(dim_hidden=hidden_dim, num_heads=num_att_heads, dropout=0.2))
            self.upsamplers.append(Upsampler(hidden_dim=hidden_dim, stride=2,kernel_size=3))

    # INPUT - x = (batch_size, seq_len, channels)
    def forward(self, x):

        # ----- ENCODER PASS -----
        # x = (batch_size, seq_len, hidden_dim)
        enc_out = self.encoder_embedding(x)
        # pos_enc = (batch_size, seq_len, hidden_dim)
        enc_pos_enc = self.encoder_pos_enc(x)
        # adding learned position encoding
        enc_out = enc_out + enc_pos_enc

        # x = (batch_size, seq_len/(num_encoder//2), hidden_dim)
        downsamplers_iter = iter(self.downsamplers)
        for i,encoder_block in enumerate(self.encoders):
            enc_out = encoder_block(enc_out)
            if ((i + 1) % 2 == 0):
                downsampler = next(downsamplers_iter)
                enc_out = downsampler(enc_out)


        # ----- DECODER PASS -----

        dec_out = enc_out
        for decoder, upsampler in zip(self.decoders, self.upsamplers):
            dec_out = decoder(dec_out, enc_out)
            dec_out = upsampler(dec_out)

        return dec_out

if __name__ == "__main__":
    model = TransformerAutoEncoder(num_encoders=4, num_decoders=2, num_att_heads=2, input_dim=12, hidden_dim=128, seq_len=60)

    x = torch.ones(6, 60, 12)

    x_melt = model(x)

    print(x_melt.shape)