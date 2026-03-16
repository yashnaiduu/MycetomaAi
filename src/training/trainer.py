import torch
from tqdm import tqdm
import os
from .losses import MultiTaskLoss

class MultiTaskTrainer:
    """
    Main Training Loop for Multi-Task Fine-Tuning.
    Takes the SSL-pretrained encoder and fine-tunes the Class, Detection, and Subtype heads.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, device, epochs=50, save_dir="checkpoints/multitask"):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = MultiTaskLoss().to(device)
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{self.epochs}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != "image"}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            preds = self.model(images)
            
            # Compute Multi-Task Loss
            loss, loss_dict = self.criterion(preds, targets)
            
            loss.backward()
            self.optimizer.step()
            
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
        total_loss = 0.0
        
        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != "image"}
            
            preds = self.model(images)
            loss, _ = self.criterion(preds, targets)
            
            total_loss += loss.item()
            
        return total_loss / len(self.val_loader)

    def train(self):
        print("Starting Multi-Task Fine-Tuning...")
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                ckpt_path = os.path.join(self.save_dir, "best_multi_task_model.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved new best model: {ckpt_path}")
