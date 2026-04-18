from dataclasses import dataclass
from typing import Literal


@dataclass
class TransformerTrainerConfig:
    # training
    max_iters: int = 100_000
    log_every: int = 100
    eval_every: int = 2_000

    # optimization
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # scheduler
    warmup_iters: int = 2_000

    # AMP
    use_amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "bf16"

    # device
    device: str = "cuda"

    # checkpointing
    checkpoint_every: int = 2_000
    checkpoint_dir: str = "./checkpoints_transformer"