import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class CAMExplainer:
    """Grad-CAM explainability."""
    def __init__(self, model, target_layers, use_cuda=False, cam_type='gradcam'):
        self.model = model
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate_heatmap(self, input_tensor, rgb_img, target_class=None):
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return visualization, grayscale_cam
