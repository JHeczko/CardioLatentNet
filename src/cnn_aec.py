import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

from src.layers import AttentionPooling
from src.layers.blocks import ConvEncoderBlock as EncoderBlock
from src.layers.blocks import ConvDecoderBlock as DecoderBlock
from src.utils.config.model import CnnAecConfig

class CnnAec(nn.Module):
    def __init__(self, config: CnnAecConfig):
        super().__init__()

        self.config = config

        # ===== ENCODER =====
        self.encoder_blocks = nn.ModuleList()
        self.enc_per_block = config.enc_dec_ratio[0]
        in_channels = config.input_dim

        for i in range(config.blocks):
            out_channels = config.hidden_channels * (2 ** i)
            for _ in range(self.enc_per_block - 1):
                self.encoder_blocks.append(EncoderBlock(in_channels, in_channels, config.dropout, stride=1))
            self.encoder_blocks.append(EncoderBlock(in_channels, out_channels, config.dropout, stride=2))
            in_channels = out_channels

        self.final_channels = in_channels
        self.final_seq_len = config.seq_len // (2 ** config.blocks)

        # ===== LATENT =====
        self.attn_pool = AttentionPooling(self.final_channels)

        self.to_latent = nn.Linear(self.final_channels, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, self.final_channels * self.final_seq_len)

        # ===== DECODER =====
        self.decoder_blocks = nn.ModuleList()
        self.dec_per_block = config.enc_dec_ratio[1]
        in_channels = self.final_channels

        for i in reversed(range(config.blocks)):
            out_channels = config.hidden_channels * (2 ** i) if i > 0 else config.input_dim
            for _ in range(self.dec_per_block - 1):
                self.decoder_blocks.append(DecoderBlock(in_channels, out_channels, config.dropout, stride=1))
            self.decoder_blocks.append(DecoderBlock(in_channels, out_channels, config.dropout, stride=2))
            in_channels = out_channels

        self.out_projection = nn.Linear(config.input_dim, config.input_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():

            # ===== CONV =====
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # ===== LINEAR =====
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # ===== BATCHNORM =====
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # ===== ENCODER =====
        # (B, seq_len, input_dim) → (B, input_dim, seq_len)
        x = x.transpose(1, 2)

        for block in self.encoder_blocks:
            x = block(x)

        # ===== LATENT =====
        # (B, final_channels, final_seq_len) → (B, final_seq_len, final_channels)
        x = x.transpose(1, 2)

        latent = self.attn_pool(x)       # (B, final_channels)
        latent = self.to_latent(latent)  # (B, latent_dim)

        x = self.from_latent(latent)                               # (B, final_channels * final_seq_len)
        x = x.view(B, self.final_channels, self.final_seq_len)     # (B, final_channels, final_seq_len)

        # ===== DECODER =====
        for block in self.decoder_blocks:
            x = block(x)

        # (B, input_dim, seq_len) → (B, seq_len, input_dim)
        x = x.transpose(1, 2)

        # wyrównaj seq_len jeśli się rozjechał przez stride
        if x.shape[1] != self.config.seq_len:
            x = F.interpolate(
                x.transpose(1, 2),
                size=self.config.seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # ===== OUTPUT =====
        x = self.out_projection(x)

        return x, latent

    @torch.no_grad()
    def encode(self, x):
        self.eval()

        # ===== ENCODER =====
        x = x.transpose(1, 2)

        for block in self.encoder_blocks:
            x = block(x)

        # ===== LATENT =====
        x = x.transpose(1, 2)

        latent = self.attn_pool(x)
        latent = self.to_latent(latent)

        return latent


if __name__ == "__main__":
    cfg = CnnAecConfig(input_dim=12, seq_len=60, hidden_channels=64, latent_dim=32, blocks=3)
    model = CnnAec(cfg)

    x = torch.randn(8, 60, 12)
    x_hat, latent = model(x)

    print(f"Input:   {x.shape}")
    print(f"Latent:  {latent.shape}")
    print(f"Output:  {x_hat.shape}")