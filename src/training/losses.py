import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    """
    Contrastive Loss for SimCLR representations.
    Maximizes agreement between differently augmented views of the same sample.
    """
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """
        z1: Projection from view 1 [B, D]
        z2: Projection from view 2 [B, D]
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Calculate similarity [2B, 2B] matrix
        z = torch.cat((z1, z2), dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        
        # Mask out self-similarity
        mask = (~torch.eye(z.size(0), z.size(0), dtype=bool)).float().to(z.device)
        sim_matrix = sim_matrix * mask
        
        # Positives are the pairs (z1_i, z2_i) and (z2_i, z1_i)
        B = z1.size(0)
        positives = torch.cat((torch.diag(sim_matrix, B), torch.diag(sim_matrix, -B)), dim=0)
        
        # Calculate loss
        negatives_sum = sim_matrix.sum(dim=1)
        loss = -torch.log(positives / negatives_sum).mean()
        
        return loss

class MultiTaskLoss(nn.Module):
    """
    Computes total weighted loss for the supervised training phase.
    Total Loss = ClassLoss + DetectionLoss + SubtypeLoss + FewShotLoss
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, delta=0.2):
        super(MultiTaskLoss, self).__init__()
        self.class_criterion = nn.CrossEntropyLoss()
        self.detect_criterion = nn.SmoothL1Loss() # for bounding box regression
        self.subtype_criterion = nn.CrossEntropyLoss()
        
        self.alpha = alpha # Weight for base classification
        self.beta = beta   # Weight for detection box loss
        self.gamma = gamma # Weight for subtype classification
        self.delta = delta # Weight for few-shot proto-loss (handled externally or passed in if aggregated)

    def forward(self, preds, targets):
        """
        preds: Dict of predictions from MultiTaskHead
        targets: Dict of ground truth labels
        """
        # 1. Base classification loss (Fungal vs Bacterial vs Normal)
        class_loss = self.class_criterion(preds["classification"], targets["label"])
        
        # 2. Detection Loss (Bounding box)
        # Note: BBox loss only computed if ground truth box exists,
        # otherwise 0. Typically requires mask/indexing.
        detect_loss = 0.0
        if "bbox" in targets and targets["bbox"] is not None:
             detect_loss = self.detect_criterion(preds["detection"], targets["bbox"])
             
        # 3. Subtype Loss
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
