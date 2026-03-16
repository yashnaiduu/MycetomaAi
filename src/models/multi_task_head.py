import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHeads(nn.Module):
    """
    Multi-Task Learning Heads extending from the main ResNet backbone.
    Tasks:
    1. Classification: Fungal (Eumycetoma) vs Bacterial (Actinomycetoma)
    2. Detection: Bounding logic coordinates [x_min, y_min, x_max, y_max]
    3. Subtype: Granular strain prediction
    """
    def __init__(self, in_features=2048, num_classes=3, num_subtypes=10):
        """
        Args:
            in_features: Channel mapping from backbone (e.g. 2048 for ResNet50)
            num_classes: e.g. 0=Background, 1=Eumycetoma, 2=Actinomycetoma
            num_subtypes: Number of specific pathogen types
        """
        super(MultiTaskHeads, self).__init__()
        self.in_features = in_features
        
        # Shared pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Task 1: Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Task 2: Detection Head (Bounding Box Coordinates)
        self.detection_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4), # x_min, y_min, x_max, y_max
            nn.Sigmoid() # Normalize coordinates to [0,1]
        )
        
        # Task 3: Subtype Classification Head
        self.subtype_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_subtypes)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Output from the CBAM backbone [B, 2048, H', W']
        Returns dict containing predictions for each task
        """
        pooled = self.pool(x)
        features = torch.flatten(pooled, 1) # [B, 2048]
        
        class_preds = self.classification_head(features)
        bbox_preds = self.detection_head(features)
        subtype_preds = self.subtype_head(features)
        
        return {
            "classification": class_preds,
            "detection": bbox_preds,
            "subtype": subtype_preds
        }
