from dataclasses import dataclass

@dataclass
class LstmVaeConfig:
    # architecture
    blocks: int = 3
    enc_dec_ratio: tuple = (1, 1)
    latent_dim: int = 128
    seq_len: int = 60
    ecg_channels: int = 12
    starting_channel_size: int = 64
    dropout: float = 0.25