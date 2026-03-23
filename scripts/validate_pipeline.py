"""End-to-end pipeline validation: train, evaluate, verify outputs."""
import json
import logging
import os
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.create_sample_data import create_dataset
from src.data.dataset import MycetomaDataset
from src.data.transforms import get_supervised_transforms
from src.evaluation.metrics import compute_metrics, dice_score, iou_score
from src.models.model import MycetomaAIModel
from src.training.losses import MultiTaskLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def validate_training(data_dir, epochs=5, batch_size=4, device=None):
    device = device or torch.device("cpu")

    dataset, class_map = MycetomaDataset.from_directory(
        data_dir,
        transform=get_supervised_transforms()["train"],
        generate_masks=True,
    )

    num_imgs = len(dataset)
    if num_imgs == 0:
        logger.error("No images found in %s", data_dir)
        return False

    logger.info("Loaded %d images, classes: %s", num_imgs, class_map)

    split = max(1, int(0.8 * num_imgs))
    train_paths = dataset.image_paths[:split]
    train_labels = dataset.labels[:split]
    val_paths = dataset.image_paths[split:]
    val_labels = dataset.labels[split:]

    transforms = get_supervised_transforms()
    train_ds = MycetomaDataset(train_paths, train_labels, transform=transforms["train"], generate_masks=True)
    val_ds = MycetomaDataset(val_paths, val_labels, transform=transforms["val"], generate_masks=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = max(len(class_map), 3)
    model = MycetomaAIModel(mode="finetune", pretrained_backbone=False).to(device)
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epoch_losses = []
    passed = True

    logger.info("=== Training %d epochs ===", epochs)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "image"}

            preds = model(images)
            loss, loss_dict = criterion(preds, targets)

            if not torch.isfinite(loss):
                logger.error("NaN/Inf loss at epoch %d", epoch + 1)
                passed = False
                break

            if "mask" not in targets:
                logger.error("Mask missing from batch — segmentation will not train")
                passed = False

            seg_loss = loss_dict.get("seg_loss", 0)
            if seg_loss == 0 and "mask" in targets:
                logger.warning("Segmentation loss is zero despite masks present")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        logger.info("Epoch %d/%d — Loss: %.4f (cls=%.3f seg=%.3f)",
                     epoch + 1, epochs, avg_loss,
                     loss_dict.get("class_loss", 0), loss_dict.get("seg_loss", 0))

    if len(epoch_losses) >= 2 and epoch_losses[-1] >= epoch_losses[0]:
        logger.warning("Loss did not decrease: %.4f → %.4f", epoch_losses[0], epoch_losses[-1])

    logger.info("=== Evaluating ===")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    seg_dices, seg_ious = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            preds = model(images)

            logits = preds["classification"].cpu()
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.numpy())
            all_preds.extend(logits.argmax(1).tolist())
            all_labels.extend(labels.tolist())

            seg = preds["segmentation"]
            if seg.sum() == 0:
                logger.warning("Empty segmentation mask")

            cls_conf = probs.max(dim=1).values.mean()
            if cls_conf < 0.1:
                logger.warning("Very low classification confidence: %.4f", cls_conf.item())

            if "mask" in batch:
                pred_mask = seg.cpu().numpy()
                gt_mask = batch["mask"].numpy()
                for i in range(pred_mask.shape[0]):
                    seg_dices.append(float(dice_score(pred_mask[i], gt_mask[i])))
                    seg_ious.append(float(iou_score(pred_mask[i], gt_mask[i])))

    all_probs = np.vstack(all_probs) if all_probs else np.array([])
    metrics = compute_metrics(all_labels, all_preds, all_probs if len(all_probs) > 0 else None)

    if seg_dices:
        metrics["Mean_Dice"] = float(np.mean(seg_dices))
        metrics["Mean_IoU"] = float(np.mean(seg_ious))

    logger.info("=== Results ===")
    for k, v in metrics.items():
        if k != "Confusion_Matrix":
            logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)
        else:
            logger.info("  Confusion Matrix:\n%s", np.array(v))

    os.makedirs("results", exist_ok=True)
    results = {
        "training_losses": epoch_losses,
        "loss_decreased": epoch_losses[-1] < epoch_losses[0] if len(epoch_losses) >= 2 else False,
        **metrics,
    }
    with open("results/validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to results/validation_results.json")

    if not passed:
        logger.error("VALIDATION FAILED")
    else:
        logger.info("VALIDATION PASSED")

    return passed


def main():
    data_dir = "data/finetune"

    has_classes = False
    if os.path.exists(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        for sd in subdirs:
            sd_path = os.path.join(data_dir, sd)
            has_images = any(
                f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))
                for f in os.listdir(sd_path)
                if os.path.isfile(os.path.join(sd_path, f))
            )
            if has_images:
                has_classes = True
                break

    if not has_classes:
        logger.info("No class-structured data found, generating synthetic data...")
        create_dataset(data_dir, images_per_class=10, size=224, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    success = validate_training(data_dir, epochs=5, batch_size=4, device=device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
