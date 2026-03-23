import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob=None):
    """Classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            auc = np.nan

    cm = confusion_matrix(y_true, y_pred)

    sensitivities = []
    specificities = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return {
        "Accuracy": acc,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Mean_Sensitivity": np.mean(sensitivities),
        "Mean_Specificity": np.mean(specificities),
        "Confusion_Matrix": cm.tolist(),
    }

def bbox_iou(box1, box2):
    """Bounding box IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def dice_score(pred, target, smooth=1.0):
    """Segmentation Dice score."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """Segmentation IoU (Jaccard)."""
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = target.astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    return (intersection + smooth) / (union + smooth)

