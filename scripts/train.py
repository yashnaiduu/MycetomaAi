import argparse
import random
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MycetomaDataset
from src.data.transforms import get_supervised_transforms, SimCLRTransform
from src.models.backbone import ResNet50CBAM
from src.models.multi_task_head import MultiTaskHeads
from src.models.ssl_encoder import HybridSSLEncoder
from src.training.trainer import MultiTaskTrainer
from src.training.ssl_pretrainer import SSLPreTrainer

import os

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.stage == "pretrain":
        print("=== Initiating Self-Supervised Pretraining ===")
        # Get immediate subdirectories in pretrain_data_dir
        root_dirs = []
        if os.path.exists(args.pretrain_data_dir):
            for d in os.listdir(args.pretrain_data_dir):
                full_path = os.path.join(args.pretrain_data_dir, d)
                if os.path.isdir(full_path):
                    root_dirs.append(full_path)
                    
        print(f"Loading data from pretraining directories: {root_dirs}")
        
        dataset = MycetomaDataset.from_ssl_directories(root_dirs, transform=SimCLRTransform())
        
        if args.verify_data:
            print(f"Verification: Found {len(dataset)} SSL images.")
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample view1 shape: {sample['view1'].shape}")
            return
            
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        # Assuming we build the model and start training in full implementation
        # model = nn.Sequential(...) 
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # trainer = SSLPreTrainer(model, optimizer, dataloader, device, epochs=args.epochs)
        # trainer.train()
        print("Pretraining pipeline initialized. (Model not fully scaffolded for actual training yet)")
        
    elif args.stage == "finetune":
        print("=== Initiating Multi-Task Finetuning ===")
        
        if not os.path.exists(args.finetune_data_dir):
            print(f"Warning: Finetune data directory '{args.finetune_data_dir}' not found.")
            if args.verify_data:
                return

        from src.data.dataset import get_image_paths
        img_paths = get_image_paths(args.finetune_data_dir)
        num_imgs = len(img_paths)
        print(f"Loading {num_imgs} images for finetuning from {args.finetune_data_dir}.")
        
        if args.verify_data:
            print("Verification successful for finetune loader.")
            return
            
        # dummy labels for now until MyData labels.csv parser is added
        labels = [random.choice([0, 1, 2]) for _ in range(num_imgs)]
        boxes = [[0.1, 0.1, 0.8, 0.8] for _ in range(num_imgs)]
        subtypes = [random.randint(0, 9) for _ in range(num_imgs)]
        
        transforms = get_supervised_transforms()
        
        # Split 80/20 train/val
        split_idx = int(0.8 * num_imgs)
        train_ds = MycetomaDataset(img_paths[:split_idx], labels[:split_idx], boxes[:split_idx], subtypes[:split_idx], transform=transforms['train'])
        val_ds = MycetomaDataset(img_paths[split_idx:], labels[split_idx:], boxes[split_idx:], subtypes[split_idx:], transform=transforms['val'])
        
        if len(train_ds) > 0:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        # Build composite network
        class FullModel(torch.nn.Module):
            def __init__(self):
                 super().__init__()
                 self.backbone = ResNet50CBAM()
                 self.heads = MultiTaskHeads(in_features=2048, num_classes=3, num_subtypes=10)
                 
            def forward(self, x):
                 features = self.backbone(x)
                 out = self.heads(features)
                 return out
                 
        # model = FullModel().to(device)
        if args.checkpoint: 
             print(f"Loading checkpoint: {args.checkpoint}")
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        # trainer = MultiTaskTrainer(model, optimizer, train_loader, val_loader, device, epochs=args.epochs)
        # trainer.train()
        print("Finetuning pipeline initialized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mycetoma AI Training Pipeline")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], required=True, help="Training stage to execute")
    parser.add_argument("--pretrain_data_dir", type=str, default="data/pretrain", help="Directory containing pretraining datasets")
    parser.add_argument("--finetune_data_dir", type=str, default="data/finetune/MyData", help="Directory containing finetuning dataset")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to pretrained checkpoint (for finetune stage)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--verify-data", action="store_true", help="Run a quick check of the dataloaders without training")
    
    args = parser.parse_args()
    main(args)
