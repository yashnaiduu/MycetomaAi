import logging
import torch
import torch.nn as nn
from src.models.backbone import ResNet50CBAM
from src.models.ssl_encoder import HybridSSLEncoder
from src.models.multi_task_head import MultiTaskHeads

logger = logging.getLogger(__name__)

class MycetomaAIModel(nn.Module):
    """Unified pretrain/finetune model."""
    def __init__(self, mode='finetune', pretrained_backbone=True):
        super().__init__()
        self.mode = mode
        
        self.backbone = ResNet50CBAM(pretrained=pretrained_backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if mode == 'pretrain':
            self.ssl_head = HybridSSLEncoder(simclr_dim=2048, projection_dim=128)
        else:
            self.task_heads = MultiTaskHeads(in_features=2048, num_classes=3, num_subtypes=10)

    def forward(self, x, x_dino=None):
        features = self.backbone(x)
        
        if self.mode == 'pretrain':
            pooled_features = torch.flatten(self.pool(features), 1)
            dino_input = x_dino if x_dino is not None else x
            return self.ssl_head(pooled_features, dino_input)
        else:
            return self.task_heads(features)

    def load_backbone(self, checkpoint_path):
        """Load pretrained backbone weights."""
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'backbone' in state_dict:
            self.backbone.load_state_dict(state_dict['backbone'])
        else:
            self.backbone.load_state_dict(state_dict)
        logger.info(f"Loaded pretrained backbone from {checkpoint_path}")
