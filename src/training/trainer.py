import logging
import os
import math
from typing import Optional
from collections import Counter

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
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
        self.best_val_f1 = 0.0
        self.no_improve_epochs = 0
        self.hard_sample_indices = []

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

        if not hasattr(self, "epoch_preds"):
            self.epoch_preds = []

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
                cls_detach = preds["classification"].detach()
                conf = torch.softmax(cls_detach, dim=1).max(dim=1).values.mean()
                if conf < 0.1:
                    logger.warning("Low confidence %.4f at epoch %d step %d", conf.item(), epoch + 1, batch_idx)
                self.epoch_preds.extend(cls_detach.argmax(dim=1).cpu().tolist())

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

        if epoch < 5 and self.epoch_preds:
            counts = [self.epoch_preds.count(i) for i in range(3)]
            logger.info("Epoch %d prediction counts (0, 1, 2): %s", epoch + 1, counts)
            if max(counts) >= 0.9 * len(self.epoch_preds):
                dom = counts.index(max(counts))
                logger.warning("COLLAPSE DETECTED: Model predicted class %d for %d/%d samples in Epoch %d!", dom, max(counts), len(self.epoch_preds), epoch + 1)
        self.epoch_preds = []

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        if self.val_loader is None or len(self.val_loader) == 0:
            return float("inf"), 0.0

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        total_loss = 0.0
        val_preds, val_labels = [], []

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "image"}

            with autocast(amp_device, enabled=self.use_amp):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            total_loss += loss.item()

            if "classification" in preds and "label" in targets:
                val_preds.extend(preds["classification"].argmax(dim=1).cpu().tolist())
                val_labels.extend(targets["label"].cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) if val_labels else 0.0

        pred_dist = Counter(val_preds)
        logger.info("Val prediction distribution: %s | F1: %.4f", dict(pred_dist), val_f1)

        return avg_loss, val_f1

    def train(self):
        logger.info("Starting Multi-Task Fine-Tuning (AMP=%s)...", self.use_amp)
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.validate()
            self.scheduler.step()

            logger.info("Epoch %s/%s - Train: %.4f | Val: %.4f | F1: %.4f", epoch + 1, self.epochs, train_loss, val_loss, val_f1)

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "val/loss": val_loss,
                    "val/f1": val_f1,
                    "epoch": epoch + 1,
                }, step=self.global_step)

            # Save best by F1 (primary) and val_loss (secondary)
            f1_improved = val_f1 > self.best_val_f1
            loss_improved = val_loss < self.best_val_loss

            if f1_improved:
                self.best_val_f1 = val_f1
                self.no_improve_epochs = 0
                self._save_checkpoint(epoch + 1, val_loss, "best_multi_task_model.pth")
                logger.info("New best F1: %.4f", val_f1)
            elif loss_improved:
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            if loss_improved:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch + 1, val_loss, "last.pth")

            if (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1, val_loss, f"multitask_ep{epoch + 1}.pth")

            if self.early_stop_patience is not None and self.no_improve_epochs >= self.early_stop_patience:
                logger.info("Early stopping after %s epochs.", self.no_improve_epochs)
                break