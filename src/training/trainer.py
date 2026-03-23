import logging
import os
import math
from typing import Optional

import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .losses import MultiTaskLoss

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """Multi-task fine-tuning loop."""

    def __init__(self, model, optimizer, train_loader, val_loader, device, epochs=50, save_dir="checkpoints/multitask", checkpoint_every_n_epochs=5, resume_from: Optional[str] = None, grad_accum_steps: int = 1, use_amp: bool = True, early_stop_patience: Optional[int] = None, wandb_run=None, loss_kwargs: Optional[dict] = None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.checkpoint_every_n_epochs = max(1, int(checkpoint_every_n_epochs or 1))
        self.grad_accum_steps = max(1, int(grad_accum_steps or 1))
        self.use_amp = use_amp and device.type == "cuda"
        self.early_stop_patience = early_stop_patience
        self.wandb_run = wandb_run

        self.criterion = MultiTaskLoss(**(loss_kwargs or {})).to(device)
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        scaler_device = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = GradScaler(scaler_device, enabled=self.use_amp)
        self.max_grad_norm = 1.0
        self.start_epoch = 0
        self.global_step = 0

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.no_improve_epochs = 0

        if resume_from:
            self._load_checkpoint(resume_from)

    def _save_checkpoint(self, epoch, val_loss, filename):
        ckpt_path = os.path.join(self.save_dir, filename)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }, ckpt_path)
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
                logger.warning("Failed to load AMP scaler state.")
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        self.global_step = int(checkpoint.get("global_step", 0))
        logger.info("Resumed from %s (epoch %s)", path, self.start_epoch)

    def _check_loss(self, loss, loss_dict, step_info=""):
        """Validate loss is finite."""
        if not math.isfinite(loss.item()):
            logger.error("NaN/Inf loss at %s: %s", step_info, loss_dict)
            raise RuntimeError(f"Non-finite loss: {loss.item()}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}/{self.epochs}")
        self.optimizer.zero_grad(set_to_none=True)

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "image"}

            if "mask" not in targets:
                logger.warning("No mask in batch %d — segmentation loss skipped", batch_idx)

            with autocast(amp_device, enabled=self.use_amp):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)
                loss = loss / self.grad_accum_steps

            self._check_loss(loss, loss_dict, f"epoch {epoch+1} step {batch_idx}")

            if "segmentation" in preds and preds["segmentation"].sum() == 0:
                logger.warning("Empty segmentation output at epoch %d step %d", epoch + 1, batch_idx)

            if "classification" in preds:
                conf = torch.softmax(preds["classification"].detach(), dim=1).max(dim=1).values.mean()
                if conf < 0.1:
                    logger.warning("Low confidence %.4f at epoch %d step %d", conf.item(), epoch + 1, batch_idx)

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.grad_accum_steps
            self.global_step += 1

            pbar.set_postfix({
                "Loss": f"{loss.item() * self.grad_accum_steps:.4f}",
                "C": f"{loss_dict.get('class_loss', 0):.2f}",
                "S": f"{loss_dict.get('seg_loss', 0):.2f}",
            })

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "train/loss": loss.item() * self.grad_accum_steps,
                    "train/class_loss": float(loss_dict.get("class_loss", 0)),
                    "train/seg_loss": float(loss_dict.get("seg_loss", 0)),
                    "lr": self.scheduler.get_last_lr()[0],
                }, step=self.global_step)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        if self.val_loader is None or len(self.val_loader) == 0:
            return float("inf")

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        total_loss = 0.0
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "image"}

            with autocast(amp_device, enabled=self.use_amp):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            total_loss += loss.item()

            if "segmentation" in preds:
                seg = preds["segmentation"]
                if seg.sum() == 0:
                    logger.warning("Empty segmentation mask in validation")
                cls_conf = torch.softmax(preds["classification"], dim=1).max(dim=1).values
                if cls_conf.mean() < 0.1:
                    logger.warning("Low classification confidence: %.4f", cls_conf.mean().item())

        return total_loss / len(self.val_loader)

    def train(self):
        logger.info("Starting Multi-Task Fine-Tuning (AMP=%s)...", self.use_amp)
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            logger.info("Epoch %s/%s - Train: %.4f | Val: %.4f", epoch + 1, self.epochs, train_loss, val_loss)

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "val/loss": val_loss,
                    "epoch": epoch + 1,
                }, step=self.global_step)

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                self._save_checkpoint(epoch + 1, val_loss, "best_multi_task_model.pth")
            else:
                self.no_improve_epochs += 1

            self._save_checkpoint(epoch + 1, val_loss, "last.pth")

            if (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1, val_loss, f"multitask_ep{epoch + 1}.pth")

            if self.early_stop_patience is not None and self.no_improve_epochs >= self.early_stop_patience:
                logger.info("Early stopping after %s epochs.", self.no_improve_epochs)
                break