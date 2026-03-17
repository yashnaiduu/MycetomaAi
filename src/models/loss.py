import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used for SimCLR style self-supervised learning.
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Args:
            z_i, z_j: Projections of two augmented views [B, dim]
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        mask = self.mask_correlated_samples(batch_size).to(z.device)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(z.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        loss = self.criterion(logits, labels)
        return loss / N

class MultiTaskLoss(nn.Module):
    """
    Weighted Multi-Task Loss for Mycetoma Diagnosis.
    Combines:
    1. Classification (CrossEntropy)
    2. Detection (Smooth L1 / MSE)
    3. Subtype (CrossEntropy)
    """
    def __init__(self, weights={'cls': 1.0, 'det': 1.0, 'sub': 0.5}):
        super(MultiTaskLoss, self).__init__()
        self.weights = weights
        self.cls_criterion = nn.CrossEntropyLoss()
        self.det_criterion = nn.SmoothL1Loss()
        self.sub_criterion = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: Dict containing 'classification', 'detection', 'subtype'
            targets: Dict containing 'classification', 'detection', 'subtype'
        """
        loss_cls = self.cls_criterion(preds['classification'], targets['classification'])
        loss_det = self.det_criterion(preds['detection'], targets['detection'])
        loss_sub = self.sub_criterion(preds['subtype'], targets['subtype'])

        total_loss = (self.weights['cls'] * loss_cls + 
                      self.weights['det'] * loss_det + 
                      self.weights['sub'] * loss_sub)
        
        return {
            "total": total_loss,
            "classification": loss_cls,
            "detection": loss_det,
            "subtype": loss_sub
        }
