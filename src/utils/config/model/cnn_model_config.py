from dataclasses import dataclass

@dataclass
class CnnAecConfig:
    # architecture
    input_dim: int = 12
    seq_len: int = 60
    hidden_channels: int = 128
    latent_dim: int = 64
    blocks: int = 3
    dropout: float = 0.2