import io
import logging
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path

from backend.src.models.model import MycetomaAIModel
from backend.src.data.stain_normalization import apply_macenko
from backend.src.data.transforms import get_supervised_transforms
from backend.src.evaluation.xai import CAMExplainer
from backend.src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Eumycetoma", "Actinomycetoma", "Normal"]
SUBTYPE_NAMES = [
    "Madurella mycetomatis", "Madurella grisea", "Leptosphaeria senegalensis",
    "Pseudallescheria boydii", "Acremonium sp.", "Streptomyces somaliensis",
    "Actinomadura madurae", "Actinomadura pelletieri", "Nocardia brasiliensis",
    "Other"
]


class _ClassificationWrapper(torch.nn.Module):
    """Wraps multi-task model for GradCAM compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["classification"]


class InferenceEngine:
    def __init__(self, checkpoint_path: str = None):
        api_cfg = load_config("api")
        model_cfg = load_config("model")

        self.device = get_device(api_cfg.get("inference", {}).get("device", "auto"))
        self.image_size = api_cfg.get("inference", {}).get("image_size", 224)

        self.model = MycetomaAIModel(mode="finetune", pretrained_backbone=False)

        ckpt_path = checkpoint_path or api_cfg.get("inference", {}).get("model_path")
        if ckpt_path and Path(ckpt_path).exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint: {ckpt_path}")
        else:
            logger.warning("No checkpoint loaded, using random weights")

        self.model.to(self.device)
        self.model.eval()

        self.transforms = get_supervised_transforms(self.image_size)["test"]

        cam_model = _ClassificationWrapper(self.model)
        target_layers = [self.model.backbone.cbam4]
        self.cam_explainer = CAMExplainer(cam_model, target_layers)

    def preprocess(self, image_bytes: bytes) -> tuple:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
        img_normalized = apply_macenko(img_array)

        augmented = self.transforms(image=img_normalized)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        rgb_float = img_normalized.astype(np.float32) / 255.0
        rgb_resized = cv2.resize(rgb_float, (self.image_size, self.image_size))

        return tensor, rgb_resized

    def predict(self, image_bytes: bytes) -> dict:
        tensor, rgb_img = self.preprocess(image_bytes)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor)

            class_logits = preds["classification"]
            probs = torch.softmax(class_logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

            bbox = preds["detection"][0].cpu().numpy().tolist()

            subtype_logits = preds["subtype"]
            subtype_probs = torch.softmax(subtype_logits, dim=1)[0]
            pred_subtype = torch.argmax(subtype_probs).item()

        # GradCAM needs gradients
        heatmap_vis, heatmap_raw = self.cam_explainer.generate_heatmap(
            tensor, rgb_img, target_class=pred_class
        )

        return {
            "class_name": CLASS_NAMES[pred_class],
            "class_id": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))
            },
            "bounding_box": bbox,
            "subtype": SUBTYPE_NAMES[pred_subtype],
            "subtype_id": pred_subtype,
            "heatmap": heatmap_vis,
            "heatmap_raw": heatmap_raw,
        }

