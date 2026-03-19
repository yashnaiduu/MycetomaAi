import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    """Contrastive loss for SimCLR."""
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """z1, z2: projections [B, D]"""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        z = torch.cat((z1, z2), dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        
        mask = (~torch.eye(z.size(0), z.size(0), dtype=bool)).float().to(z.device)
        sim_matrix = sim_matrix * mask
        
        B = z1.size(0)
        positives = torch.cat((torch.diag(sim_matrix, B), torch.diag(sim_matrix, -B)), dim=0)
        
        negatives_sum = sim_matrix.sum(dim=1)
        loss = -torch.log(positives / negatives_sum).mean()
        
        return loss

class MultiTaskLoss(nn.Module):
    """Weighted supervised multi-task loss."""
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, delta=0.2):
        super(MultiTaskLoss, self).__init__()
        self.class_criterion = nn.CrossEntropyLoss()
        self.detect_criterion = nn.SmoothL1Loss()
        self.subtype_criterion = nn.CrossEntropyLoss()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, preds, targets):
        """preds/targets: task prediction dicts."""
        class_loss = self.class_criterion(preds["classification"], targets["label"])
        
        detect_loss = 0.0
        if "bbox" in targets and targets["bbox"] is not None:
             detect_loss = self.detect_criterion(preds["detection"], targets["bbox"])
             
        subtype_loss = 0.0
        if "subtype" in targets and targets["subtype"] is not None and "subtype" in preds:
             subtype_loss = self.subtype_criterion(preds["subtype"], targets["subtype"])
             
        total_loss = (self.alpha * class_loss) + \
                     (self.beta * detect_loss) + \
                     (self.gamma * subtype_loss)
                     
        return total_loss, {
            "class_loss": class_loss.item(),
            "detect_loss": detect_loss.item() if isinstance(detect_loss, torch.Tensor) else 0.0,
            "subtype_loss": subtype_loss.item() if isinstance(subtype_loss, torch.Tensor) else 0.0
        }
