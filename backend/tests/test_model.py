import torch
import pytest
from backend.src.models.model import MycetomaAIModel
from backend.src.models.backbone import ResNet50CBAM
from backend.src.models.multi_task_head import MultiTaskHeads


class TestBackbone:
    def test_output_shape(self):
        model = ResNet50CBAM(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 2048, 7, 7)

    def test_cbam_applied(self):
        model = ResNet50CBAM(pretrained=False)
        assert hasattr(model, "cbam4")


class TestMultiTaskHeads:
    def test_output_keys(self):
        heads = MultiTaskHeads(in_features=2048, num_classes=3, num_subtypes=10)
        x = torch.randn(2, 2048, 7, 7)
        out = heads(x)
        assert "classification" in out
        assert "detection" in out
        assert "subtype" in out

    def test_output_shapes(self):
        heads = MultiTaskHeads(in_features=2048, num_classes=3, num_subtypes=10)
        x = torch.randn(2, 2048, 7, 7)
        out = heads(x)
        assert out["classification"].shape == (2, 3)
        assert out["detection"].shape == (2, 4)
        assert out["subtype"].shape == (2, 10)


class TestMycetomaAIModel:
    def test_finetune_mode(self):
        model = MycetomaAIModel(mode="finetune", pretrained_backbone=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert "classification" in out
        assert out["classification"].shape == (1, 3)

    def test_pretrain_mode(self):
        model = MycetomaAIModel(mode="pretrain", pretrained_backbone=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert "fused_proj" in out
