import argparse
import logging
import random
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MycetomaDataset, MultiDatasetWrapper, get_image_paths
from src.data.transforms import get_supervised_transforms, SimCLRTransform
from src.models.model import MycetomaAIModel
from src.training.trainer import MultiTaskTrainer
from src.training.ssl_pretrainer import SSLPreTrainer

import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if args.stage == "pretrain":
        logger.info("=== Self-Supervised Pretraining ===")
        root_dirs = []
        if os.path.exists(args.pretrain_data_dir):
            for d in os.listdir(args.pretrain_data_dir):
                full_path = os.path.join(args.pretrain_data_dir, d)
                if os.path.isdir(full_path):
                    root_dirs.append(full_path)
                    
        logger.info(f"Pretraining directories: {root_dirs}")
        
        datasets = []
        for rd in root_dirs:
            img_paths = get_image_paths(rd)
            if len(img_paths) > 0:
                logger.info(f"  Found {len(img_paths)} images in {os.path.basename(rd)}")
                datasets.append(MycetomaDataset(img_paths, is_ssl=True, transform=SimCLRTransform()))
        
        if not datasets:
            logger.error("No images found in pretraining directories.")
            return

        dataset = MultiDatasetWrapper(datasets, samples_per_dataset=args.samples_per_dataset)
        logger.info(f"Total balanced SSL dataset size: {len(dataset)}")

        if args.verify_data:
            sample = dataset[0]
            logger.info(f"Data OK. Sample view1 shape: {sample['view1'].shape}")
            return
            
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        model = MycetomaAIModel(mode='pretrain').to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        pretrainer = SSLPreTrainer(model, optimizer, dataloader, device, epochs=args.epochs)
        pretrainer.train()
        
    elif args.stage == "finetune":
        logger.info("=== Multi-Task Finetuning ===")
        
        if not os.path.exists(args.finetune_data_dir):
            logger.error(f"Finetune data directory '{args.finetune_data_dir}' not found.")
            return

        img_paths = get_image_paths(args.finetune_data_dir)
        num_imgs = len(img_paths)
        logger.info(f"Loading {num_imgs} images from {args.finetune_data_dir}")
        
        # Dummy labels for verification
        labels = [random.choice([0, 1, 2]) for _ in range(num_imgs)]
        boxes = [[0.1, 0.1, 0.8, 0.8] for _ in range(num_imgs)]
        subtypes = [random.randint(0, 9) for _ in range(num_imgs)]
        
        transforms = get_supervised_transforms()
        
        # 80/20 train/val split
        random.shuffle(img_paths)
        split_idx = int(0.8 * num_imgs)
        
        if split_idx == 0 and num_imgs > 0:
            split_idx = 1
            
        train_ds = MycetomaDataset(img_paths[:split_idx], labels[:split_idx], boxes[:split_idx], subtypes[:split_idx], transform=transforms['train'])
        val_ds = MycetomaDataset(img_paths[split_idx:], labels[split_idx:], boxes[split_idx:], subtypes[split_idx:], transform=transforms['val'])
        
        if len(train_ds) > 0:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        else:
            logger.error("No images found for training split.")
            return
        
        model = MycetomaAIModel(mode='finetune').to(device)
        
        if args.checkpoint: 
            model.load_backbone(args.checkpoint)
        
        if args.verify_data:
            logger.info("Data verification OK for finetuning.")
            return

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-2)
        
        trainer = MultiTaskTrainer(model, optimizer, train_loader, val_loader, device, epochs=args.epochs)
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mycetoma AI Training Pipeline")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--pretrain_data_dir", type=str, default="data/pretrain")
    parser.add_argument("--finetune_data_dir", type=str, default="data/finetune/MyData")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to pretrained checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples_per_dataset", type=int, default=None, help="Balance factor for SSL")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--verify_data", action="store_true")
    
    args = parser.parse_args()
    main(args)
