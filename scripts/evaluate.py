"""Model evaluation with full metric suite and visualization."""
import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MycetomaDataset, infer_labels_from_folders, get_image_paths
from src.data.transforms import get_supervised_transforms
from src.models.model import MycetomaAIModel
from src.evaluation.metrics import compute_metrics, dice_score, iou_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    seg_dices, seg_ious = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        preds = model(images)
        logits = preds["classification"].cpu()
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.numpy())
        all_preds.extend(logits.argmax(1).tolist())
        all_labels.extend(labels.tolist())

        if "segmentation" in preds and "mask" in batch:
            pred_mask = preds["segmentation"].cpu().numpy()
            gt_mask = batch["mask"].numpy()
            for i in range(pred_mask.shape[0]):
                seg_dices.append(dice_score(pred_mask[i], gt_mask[i]))
                seg_ious.append(iou_score(pred_mask[i], gt_mask[i]))

        if "segmentation" in preds:
            seg = preds["segmentation"]
            if seg.sum() == 0:
                logger.warning("Empty segmentation output detected")

    all_probs = np.vstack(all_probs)
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    if seg_dices:
        metrics["Mean_Dice"] = float(np.mean(seg_dices))
        metrics["Mean_IoU"] = float(np.mean(seg_ious))

    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = MycetomaAIModel(mode="finetune", pretrained_backbone=False).to(device)

    if not os.path.exists(args.model_path):
        logger.error("Checkpoint not found: %s", args.model_path)
        return

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    logger.info("Loaded: %s", args.model_path)

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        logger.error("Data dir not found: %s", data_dir)
        return

    paths, labels, class_map = infer_labels_from_folders(data_dir)
    if not paths:
        paths = get_image_paths(data_dir)
        labels = [0] * len(paths)
        class_map = {0: "unknown"}

    if not paths:
        logger.error("No images found in %s", data_dir)
        return

    logger.info("Evaluating on %d images", len(paths))
    transforms = get_supervised_transforms()
    ds = MycetomaDataset(
        paths, labels,
        transform=transforms["test"],
        generate_masks=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    metrics = run_evaluation(model, loader, device)

    for k, v in metrics.items():
        if k != "Confusion_Matrix":
            logger.info("%s: %.4f", k, v if v is not None else 0.0)
        else:
            logger.info("Confusion Matrix:\n%s", np.array(v))

    os.makedirs("results", exist_ok=True)

    cm = np.array(metrics.get("Confusion_Matrix", []))
    if cm.size > 0:
        class_names = [class_map.get(i, str(i)) for i in range(cm.shape[0])]
        plot_confusion_matrix(cm, class_names, "results/confusion_matrix.png")

    out_path = "results/evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved to %s", out_path)


def cli():
    parser = argparse.ArgumentParser(description="Mycetoma AI Evaluation")
    parser.add_argument("--model_path", type=str, default="checkpoints/multitask/best_multi_task_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/finetune")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
