import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHeads(nn.Module):
    """Classification, detection, subtype heads."""
    def __init__(self, in_features=2048, num_classes=3, num_subtypes=10):
        super(MultiTaskHeads, self).__init__()
        self.in_features = in_features
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.detection_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
        
        self.subtype_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_subtypes)
        )

    def forward(self, x):
        """x: backbone output [B, 2048, H', W']"""
        pooled = self.pool(x)
        features = torch.flatten(pooled, 1)
        
        class_preds = self.classification_head(features)
        bbox_preds = self.detection_head(features)
        subtype_preds = self.subtype_head(features)
        
        return {
            "classification": class_preds,
            "detection": bbox_preds,
            "subtype": subtype_preds
        }
