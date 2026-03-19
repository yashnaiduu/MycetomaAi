import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    """Few-shot prototypical network head."""
    def __init__(self, in_features=2048, hidden_dim=512, z_dim=128):
        super(PrototypicalNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, features):
        """Encode into embedding space."""
        return self.encoder(features)
    
    @staticmethod
    def euclidean_dist(x, y):
        """Pairwise euclidean distance."""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
        
    def loss(self, query_features, support_features, support_labels, query_labels):
        """Compute prototypical loss."""
        classes = torch.unique(support_labels)
        prototypes = []
        
        for c in classes:
            class_mask = support_labels == c
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
            
        prototypes = torch.stack(prototypes)
        
        dists = self.euclidean_dist(query_features, prototypes)
        
        log_p_y = F.log_softmax(-dists, dim=1)
        
        target_inds = torch.tensor([
            (classes == y).nonzero(as_tuple=True)[0][0] for y in query_labels
        ], dtype=torch.long, device=query_labels.device)
        
        loss = F.nll_loss(log_p_y, target_inds)
        
        return loss, log_p_y
