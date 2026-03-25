import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        z = torch.cat([z1, z2], dim=0)

        sim_matrix = torch.matmul(z, z.T) / self.temperature

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -9e15)

        loss = self.criterion(sim_matrix, labels)
        return loss / (2 * batch_size)

class DiceLoss(nn.Module):
    """Soft Dice loss."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.dice = DiceLoss(smooth)

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='mean')
        return bce + self.dice(pred, target)

class MultiTaskLoss(nn.Module):
    """Joint classification + segmentation loss."""
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, delta=0.5,
                 label_smoothing=0.1, class_weights=None):
        super().__init__()
        weight_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.class_criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
        self.detect_criterion = nn.SmoothL1Loss()
        self.subtype_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.seg_criterion = DiceBCELoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, preds, targets):
        class_loss = self.class_criterion(preds["classification"], targets["label"])

        detect_loss = torch.tensor(0.0, device=class_loss.device)
        if "bbox" in targets and targets["bbox"] is not None:
            detect_loss = self.detect_criterion(preds["detection"], targets["bbox"])

        subtype_loss = torch.tensor(0.0, device=class_loss.device)
        if "subtype" in targets and targets["subtype"] is not None and "subtype" in preds:
            subtype_loss = self.subtype_criterion(preds["subtype"], targets["subtype"])

        seg_loss = torch.tensor(0.0, device=class_loss.device)
        if "segmentation" in preds and "mask" in targets and targets["mask"] is not None:
            seg_loss = self.seg_criterion(preds["segmentation"], targets["mask"])

        total = (self.alpha * class_loss + self.beta * detect_loss +
                 self.gamma * subtype_loss + self.delta * seg_loss)

        return total, {
            "class_loss": class_loss.item(),
            "detect_loss": detect_loss.item(),
            "subtype_loss": subtype_loss.item(),
            "seg_loss": seg_loss.item(),
        }
