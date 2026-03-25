import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLRModel(nn.Module):
    def __init__(self, base_model=resnet50, out_dim=128):
        super().__init__()
        self.encoder = base_model(weights=ResNet50_Weights.DEFAULT)
        
        in_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projection_head = ProjectionHead(in_dim, in_dim, out_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z
