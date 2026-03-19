import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class CAMExplainer:
    """Grad-CAM explainability visualizations."""
    def __init__(self, model, target_layers, use_cuda=True, cam_type='gradcam'):
        self.model = model
        self.target_layers = target_layers
        
        if cam_type == 'gradcam':
            self.cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        elif cam_type == 'scorecam':
            self.cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        else:
            raise ValueError("cam_type must be gradcam or scorecam")
            
    def generate_heatmap(self, input_tensor, rgb_img, target_class=None):
        """Generate activation heatmap overlay."""
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return visualization, grayscale_cam
