import os
import json
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from ..config.trainer import LstmTrainerConfig


class LstmVaeTrainer:
    def __init__(self, model: nn.Module, config: LstmTrainerConfig, dataloader, val_dataloader=None):
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

        # loss
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
        self.history_val = []

        # step (for resume)
        self.start_step = 1

        # ===== EARLY STOPPER =====
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    # ========================
    # Early Stopper
    # ========================
    def _early_stopper(self, val_loss):
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            self._save_best()
            return False  # nie zatrzymuj

        self._patience_counter += 1
        if self._patience_counter >= self.config.early_stopper_patience:
            print(f"[EARLY STOP] No improvement for {self.config.early_stopper_patience} evals. Best val loss: {self._best_val_loss:.4f}")
            return True  # zatrzymaj

        return False

    # ========================
    # Save best
    # ========================
    def _save_best(self):
        path = f"{self.config.checkpoint_dir}/lstm_best.pt"

        torch.save(self.model.state_dict(), path)

    # ========================
    # MMD Loss
    # ========================
    def _mmd_loss(self, z):
        z_prior = torch.randn_like(z)

        def rbf_kernel(x, y):
            x = x.unsqueeze(1)  # (B, 1, D)
            y = y.unsqueeze(0)  # (1, B, D)
            return torch.exp(-((x - y) ** 2).sum(-1) / z.size(1))

        k_zz = rbf_kernel(z, z)
        k_pp = rbf_kernel(z_prior, z_prior)
        k_zp = rbf_kernel(z, z_prior)

        mmd = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
        return self.config.mmd_weight * mmd

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
            x_hat, mu, logvar = self.model(x)

            recon_loss = self.recon_loss_fn(x_hat, x)
            reg_loss = self._mmd_loss(mu)
            loss = recon_loss + reg_loss

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
            "recon_loss": recon_loss.item(),
            "reg_loss": reg_loss.item(),
            "lr": lr
        }

        return metrics

    # ========================
    # Evaluation
    # ========================
    @torch.no_grad()
    def evaluate(self, max_batches=1000):
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
            x_hat, _, _ = self.model(x)
            losses.append(self.recon_loss_fn(x_hat, x).item())

        avg_loss = sum(losses) / len(losses)
        print(f"[EVAL] Recon Loss: {avg_loss:.4f}")

        return avg_loss

    @torch.no_grad()
    def test(self, test_loader):
        if test_loader is None:
            return None

        self.model.eval()

        losses = []

        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            x = batch.to(self.device)
            x_hat, _, _ = self.model(x)
            losses.append(self.recon_loss_fn(x_hat, x).item())

        avg_loss = sum(losses) / len(losses)
        print(f"[TEST] Recon Loss: {avg_loss:.4f}")

        return avg_loss

    # ========================
    # Checkpoint Save
    # ========================
    def _save_checkpoint(self, step):
        #path = f"{self.config.checkpoint_dir}/lstm_step_{step}.pt"
        path_newest = f"{self.config.checkpoint_dir}/lstm_newest.pt"
        path_model = f"{self.config.checkpoint_dir}/lstm_model.pt"

        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "history": self.history,
            "history_val": self.history_val
        }

        #torch.save(state, path)
        torch.save(state, path_newest)
        torch.save(self.model.state_dict(), path_model)

    # ========================
    # Checkpoint Load
    # ========================
    def load_checkpoint(self, path=None):
        if path is None:
            path = f"{self.config.checkpoint_dir}/lstm_newest.pt"

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_step = checkpoint["step"] + 1
        self.history = checkpoint.get("history", [])
        self.history_val = checkpoint.get("history_val", [])

        print(f"Loaded checkpoint from step {checkpoint['step']}")

    # ========================
    # Save history
    # ========================
    def _save_history(self):
        path = f"{self.config.checkpoint_dir}/lstm_history.json"
        path_val = f"{self.config.checkpoint_dir}/lstm_history_val.json"

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        with open(path_val, "w") as f:
            json.dump(self.history_val, f, indent=2)

    # ========================
    # Train loop
    # ========================
    def train(self):
        for step in range(self.start_step, self.config.max_iters + 1):

            metrics = self.train_step(step)
            self.history.append(metrics)

            if step % self.config.log_every == 0:
                print(
                    f"[Step {step}] "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Recon: {metrics['recon_loss']:.4f} | "
                    f"MMD: {metrics['reg_loss']:.4f} | "
                    f"LR: {metrics['lr']:.6f}"
                )

            if self.val_dataloader and step % self.config.eval_every == 0:
                loss_val = self.evaluate()
                self.history_val.append({
                    "step": step,
                    "val_loss": loss_val,
                })

                if self._early_stopper(loss_val):
                    self._save_checkpoint(step)
                    self._save_history()
                    return self.history, self.history_val


            if step % self.config.checkpoint_every == 0:
                self._save_checkpoint(step)
                self._save_history()

        return self.history, self.history_val