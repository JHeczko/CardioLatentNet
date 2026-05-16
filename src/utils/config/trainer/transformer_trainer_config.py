from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TransformerTrainerConfig:
    '''
    Gradient Accumulation Logic:
    ----------------------------
    Effective batch size for optimizer.step() = accumulation_step * batch_size
    Example: accumulation_step=8, batch_size=8 → effective batch = 64

    max_iters and warmup_iters are specified in terms of optimizer updates,
    __post_init__ automatically scales them to raw loop steps.

    [!] log_every, eval_every, checkpoint_every must be multiples of accumulation_step
        — __post_init__ handles this automatically by rounding down.
    '''
    # training
    max_iters: int = 100_000
    log_every: int = 100
    eval_every: int = 1_000
    accumulation_step: int = 1

    # optimization
    lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # scheduler
    warmup_iters: int = 4_000

    # early stopper
    early_stopper_patience: int = 25

    # AMP
    use_amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "bf16"

    # device
    device: str = "cuda"

    # checkpointing
    checkpoint_every: int = 1_000
    checkpoint_dir: str = "./checkpoints_transformer"

    def __post_init__(self):
        acc = self.accumulation_step
        self.log_every = max(acc, (self.log_every // acc) * acc)
        self.eval_every = max(acc, (self.eval_every // acc) * acc)
        self.checkpoint_every = max(acc, (self.checkpoint_every // acc) * acc)

        self.max_iters = self.max_iters * acc
        self.warmup_iters = self.warmup_iters * acc