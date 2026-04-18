from dataclasses import dataclass

@dataclass
class TransformerUAECConfig:
    # architecture
    blocks: int = 4
    enc_dec_ratio: tuple = (1, 1)
    num_att_heads: int = 4
    input_dim: int = 12
    hidden_dim: int = 128
    latent_dim: int = 32
    seq_len: int = 60
    dropout: float = 0.2
    gradient_checkpointing: bool = False