"""Model evaluation with full metric suite."""
import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MycetomaDataset, get_image_paths
from src.data.transforms import get_supervised_transforms
from src.models.model import MycetomaAIModel
from src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        preds = model(images)
        logits = preds["classification"].cpu()
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.numpy())
        all_preds.extend(logits.argmax(1).tolist())
        all_labels.extend(labels.tolist())

    all_probs = np.vstack(all_probs)
    return compute_metrics(all_labels, all_preds, all_probs)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = MycetomaAIModel(mode="finetune", pretrained_backbone=False).to(device)

    if not os.path.exists(args.model_path):
        logger.error("Checkpoint not found: %s", args.model_path)
        return

    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    logger.info("Loaded: %s", args.model_path)

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        logger.error("Data dir not found: %s", data_dir)
        return

    paths = get_image_paths(data_dir)
    # Placeholder labels — replace with CSV
    labels = [0] * len(paths)

    transforms = get_supervised_transforms()
    ds = MycetomaDataset(paths, labels, transform=transforms["test"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    metrics = run_evaluation(model, loader, device)

    for k, v in metrics.items():
        if k != "Confusion_Matrix":
            logger.info("%s: %.4f", k, v)
        else:
            logger.info("Confusion Matrix:\n%s", np.array(v))

    os.makedirs("results", exist_ok=True)
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
