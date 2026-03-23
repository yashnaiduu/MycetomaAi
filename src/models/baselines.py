import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Baseline(nn.Module):
    """Plain ResNet50 classifier."""
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.resnet50(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return {"classification": self.net(x)}


class DenseNet121Baseline(nn.Module):
    """DenseNet121 classifier."""
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.densenet121(weights=weights)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def forward(self, x):
        return {"classification": self.net(x)}
