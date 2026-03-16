import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import MycetomaDataset
from src.data.transforms import get_supervised_transforms
from src.models.backbone import ResNet50CBAM
from src.models.multi_task_head import MultiTaskHeads
from src.evaluation.metrics import compute_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # 1. Build composite network structure mirroring training script
    class FullModel(torch.nn.Module):
        def __init__(self):
             super().__init__()
             self.backbone = ResNet50CBAM(pretrained=False) # Load locally instead
             self.heads = MultiTaskHeads(in_features=2048, num_classes=3, num_subtypes=10)
             
        def forward(self, x):
             features = self.backbone(x)
             return self.heads(features)
             
    model = FullModel().to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model weights from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {args.model_path}")
        print("Please provide a valid path or run scripts/train.py first.")
        return
        
    model.eval()
    
    # 2. Setup Data Loading (Dummy for now until real data is supplied)
    print("Initializing test dataset mapping...")
    dummy_paths = [f"test_dummy_{i}.jpg" for i in range(20)]
    labels = np.random.randint(0, 3, 20).tolist()
    boxes = [[0.1, 0.1, 0.8, 0.8]] * 20
    subtypes = np.random.randint(0, 10, 20).tolist()
    
    transforms = get_supervised_transforms()
    # test_ds = MycetomaDataset(dummy_paths, labels, boxes, subtypes, transform=transforms['test'])
    # test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    print("Beginning inference pass over the test set...")
    all_preds_class = []
    all_targets_class = []
    all_probs_class = []
    
    # with torch.no_grad():
    #     for batch in test_loader:
    #         images = batch["image"].to(device)
    #         class_targs = batch["label"].to(device)
    #
    #         preds = model(images)
    #         logits = preds["classification"]
    #         probs = torch.softmax(logits, dim=1)
    #         predictions = torch.argmax(probs, dim=1)
    #
    #         all_preds_class.extend(predictions.cpu().numpy())
    #         all_targets_class.extend(class_targs.cpu().numpy())
    #         all_probs_class.extend(probs.cpu().numpy())
            
    print("Aggregating multi-task evaluation metrics...")
    # metrics = compute_metrics(
    #     y_true=all_targets_class, 
    #     y_pred=all_preds_class, 
    #     y_prob=np.array(all_probs_class)
    # )
    
    # print("\n=== Clinical Evaluation Results ===")
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")
        
    print("Done. Update the mock data generator with actual paths to yield true clinical results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mycetoma AI Evaluation Pipeline")
    parser.add_argument("--model_path", type=str, default="checkpoints/multitask/best_multi_task_model.pth", 
                        help="Path to trained PyTorch .pth file")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
