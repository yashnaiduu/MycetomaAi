import torch
import torch.nn as nn
from src.models.backbone import ResNet50CBAM
from src.models.ssl_encoder import HybridSSLEncoder
from src.models.multi_task_head import MultiTaskHeads

class MycetomaAIModel(nn.Module):
    """
    Unified entry point for Mycetoma AI architecture.
    Can be configured for 'pretrain' (SSL) or 'finetune' (Multi-task logic).
    """
    def __init__(self, mode='finetune', pretrained_backbone=True):
        super(MycetomaAIModel, self).__init__()
        self.mode = mode
        
        # Core Backbone (Feature Extractor)
        self.backbone = ResNet50CBAM(pretrained=pretrained_backbone)
        
        if mode == 'pretrain':
            # SSL specific components
            self.ssl_head = HybridSSLEncoder(simclr_dim=2048, projection_dim=128)
        else:
            # Multi-task heads for downstream diagnosis
            self.task_heads = MultiTaskHeads(in_features=2048, num_classes=3, num_subtypes=10)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, x_dino=None):
        """
        Args:
            x (torch.Tensor): Main image tensor [B, 3, H, W]
            x_dino (torch.Tensor): Specifically normalized image for DINO if in pretrain mode.
        """
        features = self.backbone(x) # Feature map [B, 2048, H', W']
        
        if self.mode == 'pretrain':
            # In pretrain, we care about the projection outputs for contrastive loss
            # We flatten spatial features for the SSL head
            pooled_features = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(features), 1)
            return self.ssl_head(pooled_features, x_dino if x_dino is not None else x)
        
        else:
            # In finetune, we pass features to the multi-task heads
            return self.task_heads(features)

    def load_backbone(self, checkpoint_path):
        """
        Utility to load pretrained backbone weights.
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle cases where the backbone is nested in a larger state_dict
        if 'backbone' in state_dict:
            self.backbone.load_state_dict(state_dict['backbone'])
        else:
            self.backbone.load_state_dict(state_dict)
        print(f"✅ Loaded pretrained backbone from {checkpoint_path}")
