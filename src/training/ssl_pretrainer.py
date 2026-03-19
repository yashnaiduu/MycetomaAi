import logging
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from .losses import InfoNCE

logger = logging.getLogger(__name__)

class SSLPreTrainer:
    """Self-supervised pretraining loop."""
    def __init__(self, model, optimizer, dataloader, device, epochs=100, save_dir="checkpoints/ssl"):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = InfoNCE(temperature=0.1).to(device)
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        self.max_grad_norm = 1.0
        
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.dataloader, desc=f"SSL Epoch {epoch+1}/{self.epochs}")
        for batch_idx, batch in enumerate(pbar):
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            
            self.optimizer.zero_grad()
            
            out1 = self.model(view1)
            out2 = self.model(view2)
            
            loss = self.criterion(out1["fused_proj"], out2["fused_proj"])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{self.scheduler.get_last_lr()[0]:.2e}"})
            
        return total_loss / len(self.dataloader)
        
    def train(self):
        logger.info("Starting SSL Pretraining...")
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)
            self.scheduler.step()
            logger.info(f"Epoch {epoch+1}/{self.epochs} - SSL Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(self.save_dir, f"ssl_encoder_ep{epoch+1}.pth")
                torch.save({
                    "model": self.model.state_dict(),
                    "backbone": self.model.backbone.state_dict(),
                    "epoch": epoch + 1,
                    "loss": avg_loss
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
