import logging
import os
from typing import Optional

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .losses import InfoNCE

logger = logging.getLogger(__name__)


class SSLPreTrainer:
    """Self-supervised pretraining loop."""

    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        device,
        epochs=100,
        save_dir="checkpoints/ssl",
        checkpoint_every_n_epochs=10,
        resume_from: Optional[str] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        wandb_run=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.checkpoint_every_n_epochs = max(1, int(checkpoint_every_n_epochs or 1))
        self.grad_accum_steps = max(1, int(grad_accum_steps or 1))
        self.use_amp = use_amp and device.type == "cuda"
        self.wandb_run = wandb_run

        self.criterion = InfoNCE(temperature=0.1).to(device)
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = 1.0
        self.start_epoch = 0
        self.global_step = 0

        os.makedirs(self.save_dir, exist_ok=True)

        if resume_from:
            self._load_checkpoint(resume_from)

    def _save_checkpoint(self, epoch, avg_loss, filename):
        ckpt_path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                "model": self.model.state_dict(),
                "backbone": self.model.backbone.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "global_step": self.global_step,
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint: %s", ckpt_path)

    def _load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning("Resume checkpoint not found: %s", path)
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get("model", checkpoint))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
            except Exception:
                logger.warning("Failed to load AMP scaler state; continuing without it.")
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        logger.info("Resumed SSL pretraining from %s (epoch %s)", path, self.start_epoch)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.dataloader, desc=f"SSL Epoch {epoch + 1}/{self.epochs}")
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            view1 = batch["view1"].to(self.device, non_blocking=True)
            view2 = batch["view2"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                out1 = self.model(view1)
                out2 = self.model(view2)
                loss = self.criterion(out1["fused_proj"], out2["fused_proj"])
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.grad_accum_steps
            self.global_step += 1
            pbar.set_postfix({"Loss": f"{loss.item() * self.grad_accum_steps:.4f}", "LR": f"{self.scheduler.get_last_lr()[0]:.2e}"})

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"ssl/loss": loss.item() * self.grad_accum_steps, "lr": self.scheduler.get_last_lr()[0]},
                    step=self.global_step,
                )

        return total_loss / len(self.dataloader)

    def train(self):
        logger.info("Starting SSL Pretraining (AMP=%s)...", self.use_amp)
        for epoch in range(self.start_epoch, self.epochs):
            avg_loss = self.train_epoch(epoch)
            self.scheduler.step()
            logger.info("Epoch %s/%s - SSL Loss: %.4f", epoch + 1, self.epochs, avg_loss)

            if self.wandb_run is not None:
                self.wandb_run.log({"ssl/epoch_loss": avg_loss, "epoch": epoch + 1})

            self._save_checkpoint(epoch + 1, avg_loss, "last.pth")

            if (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1, avg_loss, f"ssl_encoder_ep{epoch + 1}.pth")