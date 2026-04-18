import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass


# ========================
# Config
# ========================

@dataclass
class CnnAECConfig:
    # architecture
    input_dim: int = 12
    seq_len: int = 60
    hidden_channels: int = 64
    latent_dim: int = 32
    blocks: int = 3
    dropout: float = 0.1


# ========================
# Blocks
# ========================

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


# ========================
# Model
# ========================

class CnnAEC(nn.Module):
    def __init__(self, config: CnnAECConfig):
        super().__init__()

        self.config = config

        # ===== ENCODER =====
        self.encoder_blocks = nn.ModuleList()
        in_channels = config.input_dim

        for i in range(config.blocks):
            out_channels = config.hidden_channels * (2 ** i)
            self.encoder_blocks.append(EncoderBlock(in_channels, out_channels, config.dropout))
            in_channels = out_channels

        self.final_channels = in_channels
        self.final_seq_len = config.seq_len // (2 ** config.blocks)

        # ===== LATENT =====
        self.flatten = nn.Flatten()

        flat_dim = self.final_channels * self.final_seq_len

        self.to_latent = nn.Linear(flat_dim, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, flat_dim)

        # ===== DECODER =====
        self.decoder_blocks = nn.ModuleList()
        in_channels = self.final_channels

        for i in reversed(range(config.blocks)):
            out_channels = config.hidden_channels * (2 ** i) if i > 0 else config.input_dim
            self.decoder_blocks.append(DecoderBlock(in_channels, out_channels, config.dropout))
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
        x = self.flatten(x)
        latent = self.to_latent(x)

        x = self.from_latent(latent)
        x = x.view(B, self.final_channels, self.final_seq_len)

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
        x = self.flatten(x)
        latent = self.to_latent(x)

        return latent