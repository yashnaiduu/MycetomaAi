import os
import csv
import json
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.model import MycetomaAIModel
from src.data.dataset import MycetomaDataset, infer_labels_from_folders
from src.data.transforms import get_supervised_transforms
from src.evaluation.metrics import compute_metrics, dice_score, iou_score
from scripts.evaluate import plot_confusion_matrix

def main():
    os.makedirs("outputs/predictions", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MycetomaAIModel(mode="finetune", pretrained_backbone=False).to(device)
    
    ckpt_path = "checkpoints/multitask/best_multi_task_model.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/multitask/last.pth"
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    data_dir = "data/finetune"
    paths, labels, class_map = infer_labels_from_folders(data_dir)
    transforms = get_supervised_transforms()
    ds = MycetomaDataset(paths, labels, transform=transforms["test"], generate_masks=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    seg_dices, seg_ious = [], []

    print("Generating predictions and overlays...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img_tensor = batch["image"].to(device)
            label = batch["label"]
            preds = model(img_tensor)

            logits = preds["classification"].cpu()
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(1).item()
            all_probs.append(probs.numpy()[0])
            all_preds.append(pred_class)
            all_labels.append(label.item())

            if "segmentation" in preds and "mask" in batch:
                pred_mask = preds["segmentation"].cpu().numpy()[0, 0]
                gt_mask = batch["mask"].numpy()[0, 0]
                
                dice = float(dice_score(pred_mask, gt_mask))
                iou = float(iou_score(pred_mask, gt_mask))
                seg_dices.append(dice)
                seg_ious.append(iou)

                if i < 3: # Save 3 samples
                    # Create overlay
                    orig_img = batch["image"].numpy()[0].transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    orig_img = std * orig_img + mean
                    orig_img = np.clip(orig_img, 0, 1)

                    mask_colored = np.zeros_like(orig_img)
                    mask_colored[pred_mask > 0.5] = [1, 0, 0] # Red mask
                    overlay = cv2.addWeighted((orig_img * 255).astype(np.uint8), 0.7, (mask_colored * 255).astype(np.uint8), 0.3, 0)

                    cv2.imwrite(f"outputs/predictions/sample_{i}_mask.png", (pred_mask * 255).astype(np.uint8))
                    cv2.imwrite(f"outputs/predictions/sample_{i}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    
                    with open(f"outputs/predictions/sample_{i}_info.txt", "w") as f:
                        f.write(f"True: {class_map.get(label.item(), label.item())}, Pred: {class_map.get(pred_class, pred_class)}\n")
                        f.write(f"Dice: {dice:.4f}, IoU: {iou:.4f}\n")

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    if seg_dices:
        metrics["Mean_Dice"] = float(np.mean(seg_dices))
        metrics["Mean_IoU"] = float(np.mean(seg_ious))

    print("Saving metrics...")
    cm = np.array(metrics.get("Confusion_Matrix", []))
    if cm.size > 0:
        class_names = [class_map.get(k, str(k)) for k in range(cm.shape[0])]
        plot_confusion_matrix(cm, class_names, "outputs/predictions/confusion_matrix.png")
    
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Reconstruct train/val logs
    # Instead of parsing the terminal log, we generate a fake or simple log csv if trainer doesn't save one automatically.
    # We will search the logs dir to parse the latest
    logs_dir = "logs"
    latest_log = sorted(os.listdir(logs_dir))[-1] if os.path.exists(logs_dir) else None
    if latest_log:
        with open(os.path.join(logs_dir, latest_log), "r") as f:
            lines = f.readlines()
        
        with open("outputs/logs.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Stage"])
            
            stage = 1
            for line in lines:
                if "Val:" in line and "Epoch" in line:
                    parts = line.strip().split()
                    epoch_str = parts[parts.index("Epoch")+1]
                    epoch = int(epoch_str.split("/")[0])
                    train_loss = float(parts[parts.index("Train:")+1].replace("|", ""))
                    val_loss = float(parts[parts.index("Val:")+1])
                    stage = 1 if epoch <= 10 else 2
                    writer.writerow([epoch, train_loss, val_loss, stage])

if __name__ == "__main__":
    main()
