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
    if os.path.exists(last_ckpt):
        return last_ckpt
    return None

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
    except Exception as exc:  # pragma: no cover - best-effort logging
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

def main(args):
    config: Dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)

    cli_override = {
        "stage": args.stage,
        "pretrain_data_dir": args.pretrain_data_dir,
        "finetune_data_dir": args.finetune_data_dir,
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

    if config.get("save_config") and args.config is None:
        os.makedirs("configs", exist_ok=True)
        config_path = os.path.join("configs", f"run_{config['run_id']}.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        logger.info("Saved config: %s", config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed, bool(config.get("deterministic", False)))

    precision = config.get("precision", "fp16")
    if precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Run ID: %s", config["run_id"])

    wandb_run = init_wandb(config)

    if config["stage"] == "pretrain":
        logger.info("=== Self-Supervised Pretraining ===")
        pretrain_dir = config.get("pretrain_data_dir", "data/pretrain")
        root_dirs = discover_pretrain_roots(pretrain_dir)
        logger.info("Pretraining directories: %s", root_dirs)

        datasets = []
        for rd in root_dirs:
            img_paths = get_image_paths(rd)
            if len(img_paths) > 0:
                logger.info("  Found %s images in %s", len(img_paths), os.path.basename(rd))
                datasets.append(
                    MycetomaDataset(img_paths, is_ssl=True, transform=SimCLRTransform())
                )

        if not datasets:
            logger.error("No images found in pretraining directories.")
            return

        dataset = MultiDatasetWrapper(datasets, samples_per_dataset=config.get("samples_per_dataset"))
        logger.info("Total balanced SSL dataset size: %s", len(dataset))

        if config.get("verify_data"):
            sample = dataset[0]
            logger.info("Data OK. Sample view1 shape: %s", sample["view1"].shape)
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
            model,
            optimizer,
            dataloader,
            device,
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

        finetune_dir = config.get("finetune_data_dir", "data/finetune/MyData")
        if not os.path.exists(finetune_dir):
            logger.error("Finetune data directory '%s' not found.", finetune_dir)
            return

        img_paths = get_image_paths(finetune_dir)
        num_imgs = len(img_paths)
        logger.info("Loading %s images from %s", num_imgs, finetune_dir)

        # Placeholder labels until Mycetoma annotations arrive
        labels = [random.choice([0, 1, 2]) for _ in range(num_imgs)]
        boxes = [[0.1, 0.1, 0.8, 0.8] for _ in range(num_imgs)]
        subtypes = [random.randint(0, 9) for _ in range(num_imgs)]

        transforms = get_supervised_transforms()

        random.shuffle(img_paths)
        split_idx = int(0.8 * num_imgs)
        if split_idx == 0 and num_imgs > 0:
            split_idx = 1

        train_ds = MycetomaDataset(
            img_paths[:split_idx],
            labels[:split_idx],
            boxes[:split_idx],
            subtypes[:split_idx],
            transform=transforms["train"],
        )
        val_ds = MycetomaDataset(
            img_paths[split_idx:],
            labels[split_idx:],
            boxes[split_idx:],
            subtypes[split_idx:],
            transform=transforms["val"],
        )

        if len(train_ds) > 0:
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
        else:
            logger.error("No images found for training split.")
            return

        model = MycetomaAIModel(mode="finetune").to(device)

        if config.get("checkpoint"):
            model.load_backbone(config["checkpoint"])

        if config.get("verify_data"):
            logger.info("Data verification OK for finetuning.")
            return

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("lr", 1e-4)) * 0.1,
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )

        resume_path = resolve_resume_path(config.get("resume_from"), config["checkpoint_dir"])

        trainer = MultiTaskTrainer(
            model,
            optimizer,
            train_loader,
            val_loader,
            device,
            epochs=int(config.get("epochs", 50)),
            save_dir=config["checkpoint_dir"],
            checkpoint_every_n_epochs=int(config.get("checkpoint_every_n_epochs", 5)),
            resume_from=resume_path,
            grad_accum_steps=int(config.get("grad_accum_steps", 1)),
            use_amp=(precision != "fp32"),
            early_stop_patience=config.get("early_stop_patience"),
            wandb_run=wandb_run,
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
    }
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    manifest_path = os.path.join(config["checkpoint_dir"], "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mycetoma AI Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], default=None)
    parser.add_argument("--pretrain_data_dir", type=str, default=None)
    parser.add_argument("--finetune_data_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples_per_dataset", type=int, default=None, help="Balance factor for SSL")
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

    args = parser.parse_args()
    main(args)