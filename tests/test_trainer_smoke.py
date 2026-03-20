"""Smoke test: one forward/backward step with MultiTaskTrainer on CPU."""
import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.backbone import ResNet50CBAM
from src.models.multi_task_head import MultiTaskHeads
from src.training.trainer import MultiTaskTrainer


class _TinyModel(torch.nn.Module):
    """Lightweight stand-in for MycetomaAIModel (no pretrained download)."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet50CBAM(pretrained=False)
        self.heads = MultiTaskHeads(in_features=2048, num_classes=3, num_subtypes=10)

    def forward(self, x):
        return self.heads(self.backbone(x))


class _TinyDataset(Dataset):
    """Four random images with classification, detection and subtype labels."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 64, 64),
            "label": torch.randint(0, 3, ()).long(),
            "bbox": torch.rand(4),
            "subtype": torch.randint(0, 10, ()).long(),
        }


def test_trainer_one_step_cpu():
    """One forward/backward pass on CPU and checkpoint directory creation."""
    device = torch.device("cpu")
    model = _TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = _TinyDataset(n=4)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=1,
            save_dir=tmpdir,
            use_amp=False,
        )
        trainer.train()

        assert os.path.exists(os.path.join(tmpdir, "last.pth")), "last.pth not created"
        assert os.path.exists(os.path.join(tmpdir, "best_multi_task_model.pth")), (
            "best_multi_task_model.pth not created"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer_one_step_gpu():
    """One forward/backward pass on GPU (skipped when no CUDA)."""
    device = torch.device("cuda")
    model = _TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = _TinyDataset(n=4)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=1,
            save_dir=tmpdir,
            use_amp=True,
        )
        trainer.train()

        assert os.path.exists(os.path.join(tmpdir, "last.pth")), "last.pth not created"
