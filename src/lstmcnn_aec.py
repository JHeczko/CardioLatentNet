import math
import torch
from torch import nn
from torch.nn import functional as F

from .utils.config.model import LstmVaeConfig
from .layers.blocks import LSTMConvDecoderBlock, LSTMConvEncoderBlock, VariationalBlock, LSTMConvProcessBlock


class LstmVae(nn.Module):
    def __init__(self, config: LstmVaeConfig):
        super().__init__()

        blocks = config.blocks
        latent_dim = config.latent_dim
        seq_len = config.seq_len
        ecg_channels = config.ecg_channels
        starting_channel_size = config.starting_channel_size
        dropout = config.dropout


        # ===== ENCODER =====
        self.enc_per_block = config.enc_dec_ratio[0]
        current_channel_size = starting_channel_size
        current_seq_len = seq_len

        first = True
        self.encoder_blocks = nn.ModuleList()

        for i in range(blocks):
            if first:
                self.encoder_blocks.append(
                    LSTMConvEncoderBlock(
                        input_dim=ecg_channels,
                        output_dim=current_channel_size,
                        dropout=dropout
                    )
                )
                first = False
            else:
                combined_block = nn.Sequential()

                for _ in range(self.enc_per_block - 1):
                    combined_block.append(LSTMConvProcessBlock(dim=current_channel_size, dropout=dropout))

                combined_block.append(
                    LSTMConvEncoderBlock(
                        input_dim=current_channel_size,
                        output_dim=current_channel_size * 2,
                        dropout=dropout
                    )
                )

                self.encoder_blocks.append(combined_block)
                current_channel_size *= 2

            current_seq_len = math.ceil(current_seq_len / 2)

        # zapisujemy do późniejszego reshape
        self.final_seq_len = current_seq_len
        self.final_channel_size = current_channel_size

        # ===== LATENT =====
        self.flatten = nn.Flatten()

        self.variational_latent = VariationalBlock(
            input_dim=self.final_seq_len * self.final_channel_size,
            latent_dim=latent_dim
        )

        self.projection = nn.Linear(
            latent_dim,
            self.final_seq_len * self.final_channel_size
        )

        # ===== DECODER =====
        self.dec_per_block = config.enc_dec_ratio[1]
        self.decoder_blocks = nn.ModuleList()

        for i in range(blocks):
            block_combined = nn.Sequential()

            for _ in range(self.dec_per_block - 1):
                block_combined.append(LSTMConvProcessBlock(dim=current_channel_size, dropout=dropout))

            block_combined.append(
                LSTMConvDecoderBlock(
                    input_dim=current_channel_size,
                    output_dim=current_channel_size // 2,
                    dropout=dropout
                )
            )

            self.decoder_blocks.append(block_combined)

            current_channel_size //= 2

        self.out_projection = nn.Linear(current_channel_size, ecg_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():

            # ===== CONV =====
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # ===== LINEAR =====
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # ===== GROUPNORM =====
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            # ===== LAYERNORM =====
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            # ===== LSTM =====
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        B = x.shape[0]

        # ===== ENCODER =====
        encoder_lengths = []

        for block in self.encoder_blocks:
            x = block(x)
            encoder_lengths.append(x.shape[1])  # zapisujemy po każdym bloku

        encoder_lengths = list(reversed(encoder_lengths))

        # ===== LATENT =====
        x = self.flatten(x)

        z, mu, logvar = self.variational_latent(x)

        x = self.projection(z)
        x = x * (1 / (self.final_channel_size ** 0.5))
        x = x.view(B, self.final_seq_len, self.final_channel_size)

        # ===== DECODER =====
        for i, block in enumerate(self.decoder_blocks):
            target_len = encoder_lengths[i]

            if x.shape[1] != target_len:
                x = F.interpolate(
                    x.transpose(1, 2),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            x = block(x)

        # ===== OUTPUT =====
        x = self.out_projection(x)

        return x, mu, logvar

    @torch.no_grad()
    def encode(self, x):
        self.eval()

        # ===== ENCODER =====
        for block in self.encoder_blocks:
            x = block(x)

        # ===== LATENT =====q
        x = self.flatten(x)

        _, mu, _ = self.variational_latent(x)

        return mu

if __name__ == "__main__":
    t = torch.randn(9, 60, 12)

    vae = LstmVae(3, 20, 60, 12, 0.2)

    t_out, mu, logvar = vae(t)

    print(t_out.shape)