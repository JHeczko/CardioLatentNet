from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainerConfig:
    # training
    max_iters: int = 100_000
    log_every: int = 100

    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # VAE
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_iters: int = 10_000

    # precision
    use_amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "bf16"

    # device
    device: str = "cuda"