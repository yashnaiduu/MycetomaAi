import torch
import torch.nn as nn
from transformers import AutoModel

class HybridSSLEncoder(nn.Module):
    """SimCLR + DINOv2 fusion encoder."""
    def __init__(self, use_vits14: bool = True, simclr_dim: int = 2048, projection_dim: int = 128):
        super(HybridSSLEncoder, self).__init__()
        
        self.simclr_proj = nn.Sequential(
            nn.Linear(simclr_dim, simclr_dim),
            nn.BatchNorm1d(simclr_dim),
            nn.ReLU(),
            nn.Linear(simclr_dim, projection_dim)
        )
        
        model_name = "facebook/dinov2-small" if use_vits14 else "facebook/dinov2-base"
        self.dinov2 = AutoModel.from_pretrained(model_name)
        
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        dino_dim = self.dinov2.config.hidden_size
        
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_dim, dino_dim // 2),
            nn.BatchNorm1d(dino_dim // 2),
            nn.ReLU(),
            nn.Linear(dino_dim // 2, projection_dim)
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x_simclr_features, x_dino_images):
        """Fuse SimCLR + DINO projections."""
        simclr_z = self.simclr_proj(x_simclr_features)
        
        dino_outputs = self.dinov2(x_dino_images)
        dino_cls = dino_outputs.last_hidden_state[:, 0, :]
        dino_z = self.dino_proj(dino_cls)
        
        fused = torch.cat((simclr_z, dino_z), dim=1)
        output = self.fusion_head(fused)
        
        return {
            "simclr_proj": simclr_z,
            "dino_proj": dino_z,
            "fused_proj": output
        }
