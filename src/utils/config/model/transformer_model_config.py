from dataclasses import dataclass

@dataclass
class TransformerAecConfig:
    # architecture
    blocks: int = 4
    enc_dec_ratio: tuple = (1, 1)
    num_att_heads: int = 8
    input_dim: int = 12
    hidden_dim: int = 256
    latent_dim: int = 64
    seq_len: int = 60 # heartbeat lenght with sampling 100hz
    dropout: float = 0.2
    gradient_checkpointing: bool = True