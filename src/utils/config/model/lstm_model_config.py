from dataclasses import dataclass

@dataclass
class LSTMConfig:
    # architecture
    blocks: int = 3
    latent_dim: int = 32
    seq_len: int = 60
    ecg_channels: int = 12
    starting_channel_size: int = 32
    dropout: float = 0.2