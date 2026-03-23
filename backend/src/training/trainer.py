import logging
import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from .losses import MultiTaskLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class MultiTaskTrainer:
    """Multi-task fine-tuning with early stopping and label smoothing."""
    def __init__(self, model, optimizer, train_loader, val_loader, device,
                 epochs=50, save_dir="checkpoints/multitask",
                 label_smoothing=0.1, patience=10, min_delta=0.001):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = MultiTaskLoss(label_smoothing=label_smoothing).to(device)
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.max_grad_norm = 1.0
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{self.epochs}")
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "image"}

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(self.device.type == "cuda")):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "C": f"{loss_dict.get('class_loss', 0):.2f}",
                "D": f"{loss_dict.get('detect_loss', 0):.2f}"
            })

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        if self.val_loader is None or len(self.val_loader) == 0:
            return float("inf")

        total_loss = 0.0

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "image"}

            with autocast("cuda", enabled=(self.device.type == "cuda")):
                preds = self.model(images)
                loss, _ = self.criterion(preds, targets)

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        logger.info("Starting Multi-Task Fine-Tuning (AMP enabled)...")
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            logger.info(f"Epoch {epoch+1}/{self.epochs} - Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                ckpt_path = os.path.join(self.save_dir, "best_multi_task_model.pth")
                torch.save({
                    "model": self.model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss
                }, ckpt_path)
                logger.info(f"Saved best model: {ckpt_path}")

            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
