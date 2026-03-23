import argparse
import datetime
import json
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import MycetomaDataset, MultiDatasetWrapper, get_image_paths
from src.data.transforms import get_supervised_transforms, SimCLRTransform
from src.models.model import MycetomaAIModel
from src.training.trainer import MultiTaskTrainer
from src.training.ssl_pretrainer import SSLPreTrainer


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


def configure_logging(log_dir: str, run_id: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def resolve_resume_path(resume_from: Optional[str], checkpoint_dir: str) -> Optional[str]:
    if not resume_from:
        return None
    if resume_from != "auto":
        return resume_from
    last_ckpt = os.path.join(checkpoint_dir, "last.pth")
    return last_ckpt if os.path.exists(last_ckpt) else None


def init_wandb(config: Dict[str, Any]):
    if not config.get("wandb", False):
        return None
    try:
        import wandb
        return wandb.init(
            project=config.get("wandb_project", "mycetoma-ai"),
            name=config.get("wandb_name"),
            tags=config.get("wandb_tags"),
            config=config,
            resume="allow",
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("W&B init failed: %s", exc)
        return None


def discover_pretrain_roots(pretrain_data_dir: str) -> list:
    roots = []
    if os.path.isdir(pretrain_data_dir):
        for entry in sorted(os.listdir(pretrain_data_dir)):
            full = os.path.join(pretrain_data_dir, entry)
            if os.path.isdir(full):
                roots.append(full)
    if not roots:
        roots = [pretrain_data_dir]
    return roots


def load_finetune_data(config, logger):
    finetune_dir = config.get("finetune_data_dir", "data/finetune")
    csv_path = config.get("annotations_csv")
    gen_masks = bool(config.get("generate_masks", True))
    use_macenko = bool(config.get("use_macenko", True))
    transforms = get_supervised_transforms(int(config.get("image_size", 224)))

    if csv_path and os.path.exists(csv_path):
        logger.info("Loading from CSV: %s", csv_path)
        dataset = MycetomaDataset.from_csv(
            csv_path, finetune_dir,
            transform=transforms["train"],
            generate_masks=gen_masks,
            use_macenko=use_macenko,
        )
        num_imgs = len(dataset)
        if num_imgs == 0:
            return None, None, 0
        indices = list(range(num_imgs))
        random.shuffle(indices)
        split = int(0.8 * num_imgs) or 1
        train_ds = torch.utils.data.Subset(dataset, indices[:split])
        val_ds = torch.utils.data.Subset(dataset, indices[split:])
        return train_ds, val_ds, num_imgs

    if not os.path.exists(finetune_dir):
        logger.error("Finetune directory not found: %s", finetune_dir)
        return None, None, 0

    dataset, class_map = MycetomaDataset.from_directory(
        finetune_dir,
        transform=transforms["train"],
        generate_masks=gen_masks,
        use_macenko=use_macenko,
    )

    num_imgs = len(dataset)
    if num_imgs == 0:
        logger.error("No images found in %s", finetune_dir)
        return None, None, 0

    logger.info("Loaded %d images, classes: %s", num_imgs, class_map)

    indices = list(range(num_imgs))
    random.shuffle(indices)
    split = int(0.8 * num_imgs) or 1

    train_paths = [dataset.image_paths[i] for i in indices[:split]]
    train_labels = [dataset.labels[i] for i in indices[:split]]
    val_paths = [dataset.image_paths[i] for i in indices[split:]]
    val_labels = [dataset.labels[i] for i in indices[split:]]

    train_ds = MycetomaDataset(
        train_paths, train_labels,
        transform=transforms["train"],
        generate_masks=gen_masks,
        use_macenko=use_macenko,
    )
    val_ds = MycetomaDataset(
        val_paths, val_labels,
        transform=transforms["val"],
        generate_masks=gen_masks,
        use_macenko=use_macenko,
    )

    return train_ds, val_ds, num_imgs


def main(args):
    config: Dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)

    cli_override = {
        "stage": args.stage,
        "pretrain_data_dir": args.pretrain_data_dir,
        "finetune_data_dir": args.finetune_data_dir,
        "annotations_csv": args.annotations_csv,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "samples_per_dataset": args.samples_per_dataset,
        "lr": args.lr,
        "verify_data": args.verify_data,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_every_n_epochs": args.checkpoint_every_n_epochs,
        "resume_from": args.resume_from,
        "grad_accum_steps": args.grad_accum_steps,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "wandb_tags": args.wandb_tags,
        "run_id": args.run_id,
        "log_dir": args.log_dir,
        "early_stop_patience": args.early_stop_patience,
        "freeze_backbone": args.freeze_backbone,
    }

    config = merge_config(config, cli_override)

    if "run_id" not in config or not config["run_id"]:
        config["run_id"] = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if "log_dir" not in config or not config["log_dir"]:
        config["log_dir"] = "logs"

    log_path = configure_logging(config["log_dir"], config["run_id"])
    logger = logging.getLogger(__name__)

    if "stage" not in config or not config["stage"]:
        raise ValueError("stage is required (pretrain or finetune)")

    if not config.get("checkpoint_dir"):
        config["checkpoint_dir"] = "checkpoints/ssl" if config["stage"] == "pretrain" else "checkpoints/multitask"

    seed = int(config.get("seed", 42))
    set_seed(seed, bool(config.get("deterministic", False)))

    precision = config.get("precision", "fp16")
    if precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Run: %s", device, config["run_id"])

    wandb_run = init_wandb(config)

    if config["stage"] == "pretrain":
        logger.info("=== Self-Supervised Pretraining ===")
        pretrain_dir = config.get("pretrain_data_dir", "data/pretrain")
        root_dirs = discover_pretrain_roots(pretrain_dir)

        datasets = []
        for rd in root_dirs:
            img_paths = get_image_paths(rd)
            if img_paths:
                logger.info("  %s: %d images", os.path.basename(rd), len(img_paths))
                datasets.append(
                    MycetomaDataset(img_paths, is_ssl=True, transform=SimCLRTransform())
                )

        if not datasets:
            logger.error("No images found for pretraining.")
            return

        dataset = MultiDatasetWrapper(datasets, samples_per_dataset=config.get("samples_per_dataset"))
        logger.info("SSL dataset size: %d", len(dataset))

        if config.get("verify_data"):
            sample = dataset[0]
            logger.info("View1 shape: %s", sample["view1"].shape)
            return

        dataloader = DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 16)),
            shuffle=True,
            num_workers=int(config.get("num_workers", 4)),
            pin_memory=True,
        )

        model = MycetomaAIModel(mode="pretrain").to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("lr", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )

        resume_path = resolve_resume_path(config.get("resume_from"), config["checkpoint_dir"])

        pretrainer = SSLPreTrainer(
            model, optimizer, dataloader, device,
            epochs=int(config.get("epochs", 100)),
            save_dir=config["checkpoint_dir"],
            checkpoint_every_n_epochs=int(config.get("checkpoint_every_n_epochs", 10)),
            resume_from=resume_path,
            grad_accum_steps=int(config.get("grad_accum_steps", 1)),
            use_amp=(precision != "fp32"),
            wandb_run=wandb_run,
        )
        pretrainer.train()

    elif config["stage"] == "finetune":
        logger.info("=== Multi-Task Finetuning ===")

        train_ds, val_ds, num_imgs = load_finetune_data(config, logger)
        if train_ds is None or num_imgs == 0:
            return

        train_loader = DataLoader(
            train_ds,
            batch_size=int(config.get("batch_size", 16)),
            shuffle=True,
            num_workers=int(config.get("num_workers", 4)),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(config.get("batch_size", 16)),
            shuffle=False,
            num_workers=int(config.get("num_workers", 4)),
        )

        if config.get("verify_data"):
            sample = next(iter(train_loader))
            logger.info("Image: %s", sample["image"].shape)
            logger.info("Label: %s", sample.get("label"))
            logger.info("Mask: %s", sample["mask"].shape if "mask" in sample else "N/A")
            return

        pretrained_bb = bool(config.get("pretrained_backbone", True))
        model = MycetomaAIModel(mode="finetune", pretrained_backbone=pretrained_bb).to(device)

        if config.get("checkpoint"):
            model.load_backbone(config["checkpoint"])

        if config.get("freeze_backbone", False):
            for param in model.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.get("lr", 1e-4)) * 0.1,
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )

        loss_kwargs = {
            "alpha": float(config.get("loss_alpha", 1.0)),
            "beta": float(config.get("loss_beta", 0.5)),
            "gamma": float(config.get("loss_gamma", 0.5)),
            "delta": float(config.get("loss_delta", 0.5)),
            "label_smoothing": float(config.get("label_smoothing", 0.1)),
        }

        resume_path = resolve_resume_path(config.get("resume_from"), config["checkpoint_dir"])

        trainer = MultiTaskTrainer(
            model, optimizer, train_loader, val_loader, device,
            epochs=int(config.get("epochs", 50)),
            save_dir=config["checkpoint_dir"],
            checkpoint_every_n_epochs=int(config.get("checkpoint_every_n_epochs", 5)),
            resume_from=resume_path,
            grad_accum_steps=int(config.get("grad_accum_steps", 1)),
            use_amp=(precision != "fp32"),
            early_stop_patience=config.get("early_stop_patience"),
            wandb_run=wandb_run,
            loss_kwargs=loss_kwargs,
        )
        trainer.train()

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    manifest = {
        "run_id": config["run_id"],
        "stage": config["stage"],
        "log": log_path,
        "checkpoint_dir": config.get("checkpoint_dir"),
        "config": {k: v for k, v in config.items() if not k.startswith("wandb")},
    }
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    manifest_path = os.path.join(config["checkpoint_dir"], "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)


def cli():
    parser = argparse.ArgumentParser(description="Mycetoma AI Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], default=None)
    parser.add_argument("--pretrain_data_dir", type=str, default=None)
    parser.add_argument("--finetune_data_dir", type=str, default=None)
    parser.add_argument("--annotations_csv", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples_per_dataset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--verify_data", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()