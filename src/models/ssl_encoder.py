import torch
import torch.nn as nn
from transformers import AutoModel

class HybridSSLEncoder(nn.Module):
    """
    Hybrid Self-Supervised Encoder.
    Fuses SimCLR and DINOv2 representations to learn robust morphology features
    without requiring pixel-level annotations for training.
    """
    def __init__(self, use_vits14: bool = True, simclr_dim: int = 2048, projection_dim: int = 128):
        """
        Args:
            use_vits14 (bool): Whether to use ViT-S/14 DINOv2 weights.
            simclr_dim (int): Base dimensionality of the SimCLR backbone (e.g. ResNet50 output).
            projection_dim (int): Final output dimensionality for contrastive loss.
        """
        super(HybridSSLEncoder, self).__init__()
        
        # SimCLR projection head components
        # (Assuming the main backbone is injected from elsewhere or trained jointly)
        self.simclr_proj = nn.Sequential(
            nn.Linear(simclr_dim, simclr_dim),
            nn.BatchNorm1d(simclr_dim),
            nn.ReLU(),
            nn.Linear(simclr_dim, projection_dim)
        )
        
        # DINOv2 Base Model (Frozen for pure feature extraction, or fine-tuned)
        model_name = "facebook/dinov2-small" if use_vits14 else "facebook/dinov2-base"
        self.dinov2 = AutoModel.from_pretrained(model_name)
        
        # Disable gradients on DINO initially
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        dino_dim = self.dinov2.config.hidden_size # usually 384 for ViT-S
        
        # DINO projection head to match simclr_projection
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_dim, dino_dim // 2),
            nn.BatchNorm1d(dino_dim // 2),
            nn.ReLU(),
            nn.Linear(dino_dim // 2, projection_dim)
        )
        
        # Fusion projection head
        self.fusion_head = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x_simclr_features, x_dino_images):
        """
        Forward generic feature extraction.
        Args:
            x_simclr_features (torch.Tensor): Output feature maps from ResNet/EffNet backbone. [B, simclr_dim]
            x_dino_images (torch.Tensor): Raw image tensors normalized for DINO. [B, 3, H, W]
        """
        simclr_z = self.simclr_proj(x_simclr_features)
        
        # Extract DINO features (CLS token)
        dino_outputs = self.dinov2(x_dino_images)
        dino_cls = dino_outputs.last_hidden_state[:, 0, :]
        dino_z = self.dino_proj(dino_cls)
        
        # Concatenate features
        fused = torch.cat((simclr_z, dino_z), dim=1)
        output = self.fusion_head(fused)
        
        return {
            "simclr_proj": simclr_z,
            "dino_proj": dino_z,
            "fused_proj": output
        }
