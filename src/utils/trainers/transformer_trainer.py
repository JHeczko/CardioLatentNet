import os
import json
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from ..config.trainer import TransformerTrainerConfig

class TransformerUAECTrainer:
    def __init__(self, model: nn.Module, dataloader, config: TransformerTrainerConfig, val_dataloader=None):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        # device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # loss — zwykły MSE, brak VAE
        self.recon_loss_fn = nn.MSELoss()

        # dataloader iterator
        self.data_iter = iter(self.dataloader)

        # ===== AMP CONFIG =====
        self.use_amp = False
        self.amp_dtype = torch.float32

        if self.device.type == "cuda" and config.use_amp:
            self.use_amp = True

            if config.amp_dtype == "bf16":
                self.amp_dtype = torch.bfloat16
            elif config.amp_dtype == "fp16":
                self.amp_dtype = torch.float16

        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler(
            device=self.device.type,
            enabled=self.use_scaler
        )

        # scheduler
        self.base_lr = config.lr

        # dirs
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # history
        self.history = []

        # step (for resume)
        self.start_step = 1

    # ========================
    # LR Scheduler
    # ========================
    def _get_lr(self, step):
        if step < self.config.warmup_iters:
            return self.base_lr * step / self.config.warmup_iters

        progress = (step - self.config.warmup_iters) / (
            self.config.max_iters - self.config.warmup_iters
        )

        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
        return self.config.min_lr + (self.base_lr - self.config.min_lr) * cosine_decay.item()

    # ========================
    # Get batch
    # ========================
    def _get_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        return batch.to(self.device)

    # ========================
    # Train step
    # ========================
    def train_step(self, step):
        self.model.train()

        x = self._get_batch()

        # LR update
        lr = self._get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        with autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp
        ):
            x_hat = self.model(x)
            loss = self.recon_loss_fn(x_hat, x)

        self.optimizer.zero_grad(set_to_none=True)

        if self.use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        metrics = {
            "step": step,
            "loss": loss.item(),
            "lr": lr
        }

        self.history.append(metrics)

        return metrics

    # ========================
    # Evaluation
    # ========================
    @torch.no_grad()
    def evaluate(self, max_batches=20):
        if self.val_dataloader is None:
            return None

        self.model.eval()

        losses = []

        for i, batch in enumerate(self.val_dataloader):
            if i >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            x = batch.to(self.device)
            x_hat = self.model(x)
            loss = self.recon_loss_fn(x_hat, x)
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"[EVAL] Recon Loss: {avg_loss:.4f}")

        return avg_loss

    # ========================
    # Checkpoint Save
    # ========================
    def _save_checkpoint(self, step):
        path = f"{self.config.checkpoint_dir}/transformer_step_{step}.pt"
        path_newest = f"{self.config.checkpoint_dir}/transformer_newest.pt"

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "history": self.history
        }, path)

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "history": self.history
        }, path_newest)

    # ========================
    # Checkpoint Load
    # ========================
    def load_checkpoint(self, path):
        if path is None:
            checkpoint = torch.load(f"{self.config.checkpoint_dir}/transformer_newest.pt", map_location=self.device)
        else:
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_step = checkpoint["step"] + 1
        self.history = checkpoint.get("history", [])

        print(f"Loaded checkpoint from step {checkpoint['step']}")

    # ========================
    # Save history
    # ========================
    def _save_history(self):
        path = f"{self.config.checkpoint_dir}/transformer_history.json"

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # ========================
    # Train loop
    # ========================
    def train(self):
        for step in range(self.start_step, self.config.max_iters + 1):

            metrics = self.train_step(step)

            if step % self.config.log_every == 0:
                print(
                    f"[Step {step}] "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"LR: {metrics['lr']:.6f}"
                )

            if self.val_dataloader and step % self.config.eval_every == 0:
                self.evaluate()

            if step % self.config.checkpoint_every == 0:
                self._save_checkpoint(step)
                self._save_history()