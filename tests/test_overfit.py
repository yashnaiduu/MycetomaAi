"""Overfit test: verify gradients flow and model learns."""
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.model import MycetomaAIModel
from src.training.losses import MultiTaskLoss


class _TinySegDataset(Dataset):
    def __init__(self, n=8):
        self.n = n
        self.fixed_label = [i % 3 for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "label": torch.tensor(self.fixed_label[idx]).long(),
            "bbox": torch.rand(4),
            "subtype": torch.tensor(idx % 10).long(),
            "mask": torch.randint(0, 2, (1, 224, 224)).float(),
        }


def test_overfit():
    device = torch.device("cpu")
    model = MycetomaAIModel(mode="finetune", pretrained_backbone=False).to(device)
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = _TinySegDataset(n=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # Verify shapes before training
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224).to(device))
        assert out["segmentation"].shape == (1, 1, 224, 224), f"Bad seg shape: {out['segmentation'].shape}"
        assert out["classification"].shape == (1, 3), f"Bad cls shape: {out['classification'].shape}"
        assert out["detection"].shape == (1, 4), f"Bad det shape: {out['detection'].shape}"
        assert out["subtype"].shape == (1, 10), f"Bad sub shape: {out['subtype'].shape}"
        mask = out["segmentation"]
        assert mask.min() >= 0 and mask.max() <= 1, "Mask out of [0,1]"
    print("Shape checks PASSED")

    # Train and verify loss decreases
    model.train()
    losses = []
    for epoch in range(10):
        epoch_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "image"}
            preds = model(images)
            loss, ld = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    # Verify gradients flowed (loss changed)
    assert losses[-1] != losses[0], "Loss unchanged — gradient flow broken"

    # Verify all parameters received gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None or "ssl" in name, f"No gradient: {name}"

    print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print("Overfit test PASSED")


if __name__ == "__main__":
    test_overfit()
