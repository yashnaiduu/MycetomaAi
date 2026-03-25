import os
import sys
import torch
import torch.optim as optim
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.simclr_dataset import setup_simclr_dataloader
from src.models.simclr import SimCLRModel
from src.training.losses import NTXentLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/pretrain_ready")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4) # Adam specific initialization
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="models/checkpoints")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    dataset, dataloader = setup_simclr_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    logging.info(f"Initialized Dataset with {len(dataset)} items.")

    model = SimCLRModel().to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")

    logging.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            
            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = criterion(z1, z2)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logging.info(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss:.4f}")
        
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }
        
        torch.save(checkpoint, os.path.join(args.out_dir, "simclr_latest.pth"))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(args.out_dir, "simclr_best.pth"))
            logging.info(f"Saved best model. Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
