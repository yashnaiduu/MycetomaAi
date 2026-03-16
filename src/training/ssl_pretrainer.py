import torch
from tqdm import tqdm
import os
from .losses import InfoNCE

class SSLPreTrainer:
    """
    Self-Supervised Pretraining Loop.
    Uses Contrastive Loss on unclassified histopathology slides to enforce
    morphological pattern recognition in the encoder.
    """
    def __init__(self, model, optimizer, dataloader, device, epochs=100, save_dir="checkpoints/ssl"):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = InfoNCE(temperature=0.1).to(device)
        
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.dataloader, desc=f"SSL Epoch {epoch+1}/{self.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Batch contains two views of the same image from SimCLRTransform
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Since this is a hybrid encoder, we run both views through the simclr path
            # Assume model.forward() handles returning the fused projections
            # (In a true implementation, we split the feature extraction: ResNet backbone -> SSL Encoder)
            # Here we assume model is ResNet50 + HybridSSLEncoder chained together.
            out1 = self.model(view1)
            out2 = self.model(view2)
            
            # Compute Contrastive Loss
            loss = self.criterion(out1["fused_proj"], out2["fused_proj"])
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        return total_loss / len(self.dataloader)
        
    def train(self):
        print("Starting SSL Pretraining...")
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.epochs} - SSL Loss: {avg_loss:.4f}")
            
            # Checkpointing
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(self.save_dir, f"ssl_encoder_ep{epoch+1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
