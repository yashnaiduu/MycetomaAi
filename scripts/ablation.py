"""Ablation study: compare model variants under identical conditions."""
import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.dataset import MycetomaDataset, get_image_paths, infer_labels_from_folders
from src.data.transforms import get_supervised_transforms
from src.models.backbone import ResNet50CBAM
from src.models.baselines import ResNet50Baseline, DenseNet121Baseline
from src.models.model import MycetomaAIModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class CBAMClassifier(torch.nn.Module):
    """ResNet50+CBAM classification only."""
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = ResNet50CBAM(pretrained=pretrained)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        out = torch.flatten(self.pool(feat), 1)
        return {"classification": self.fc(out)}


def build_model(variant, device, num_classes=3):
    if variant == "resnet50":
        return ResNet50Baseline(num_classes=num_classes, pretrained=False).to(device)
    elif variant == "densenet121":
        return DenseNet121Baseline(num_classes=num_classes, pretrained=False).to(device)
    elif variant == "cbam":
        return CBAMClassifier(num_classes=num_classes, pretrained=False).to(device)
    elif variant == "cbam_aug":
        return CBAMClassifier(num_classes=num_classes, pretrained=False).to(device)
    elif variant == "full":
        return MycetomaAIModel(mode="finetune", pretrained_backbone=False).to(device)
    raise ValueError(f"Unknown variant: {variant}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds["classification"], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=3):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        logits = model(images)["classification"].cpu()
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.numpy())
        all_preds.extend(logits.argmax(1).tolist())
        all_labels.extend(labels.tolist())

    all_probs = np.vstack(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4)}


def run_experiment(variant, data_dir, epochs, batch_size, device):
    logger.info("=== %s ===", variant)
    paths, labels, class_map = infer_labels_from_folders(data_dir)
    num_classes = len(class_map) if class_map else 3

    if not paths:
        paths = get_image_paths(data_dir)
        if not paths:
            logger.error("No images in %s", data_dir)
            return {}
        labels = [0] * len(paths)
        num_classes = 1
        logger.warning("No class folders found — single-class fallback")

    logger.info("Found %d images, %d classes", len(paths), num_classes)

    use_aug = variant in ("cbam_aug", "full")
    transforms = get_supervised_transforms()
    labels_arr = np.array(labels)

    n_splits = min(3, len(set(labels)))
    if n_splits < 2:
        logger.warning("Insufficient classes for cross-validation")
        return {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        logger.info("  Fold %d/%d", fold + 1, n_splits)
        paths_arr = np.array(paths)

        t_key = "train" if use_aug else "val"
        train_ds = MycetomaDataset(
            paths_arr[train_idx].tolist(), labels_arr[train_idx].tolist(),
            transform=transforms[t_key], generate_masks=False,
        )
        val_ds = MycetomaDataset(
            paths_arr[val_idx].tolist(), labels_arr[val_idx].tolist(),
            transform=transforms["val"], generate_masks=False,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        model = build_model(variant, device, num_classes=max(num_classes, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for ep in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        metrics = evaluate(model, val_loader, device)
        fold_metrics.append(metrics)
        logger.info("  Fold %d → acc=%.4f f1=%.4f auc=%.4f",
                     fold + 1, metrics["accuracy"], metrics["f1"], metrics["auc"])

    avg = {
        k: round(np.mean([m[k] for m in fold_metrics]), 4)
        for k in ("accuracy", "f1", "auc")
    }
    logger.info("  Mean → acc=%.4f f1=%.4f auc=%.4f", avg["accuracy"], avg["f1"], avg["auc"])
    return avg


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--data_dir", type=str, default="data/finetune")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = ["resnet50", "densenet121", "cbam", "cbam_aug", "full"]

    results = {}
    for v in variants:
        results[v] = run_experiment(v, args.data_dir, args.epochs, args.batch_size, device)

    os.makedirs("results", exist_ok=True)
    out_path = "results/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    print("\n" + "=" * 55)
    print(f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 55)
    for v, m in results.items():
        if m:
            print(f"{v:<15} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
