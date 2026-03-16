import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class CAMExplainer:
    """
    Explainable AI Visualizations using Grad-CAM.
    Allows clinicians to see which regions of the histopathology slide (grains)
    triggered the classification decision.
    """
    def __init__(self, model, target_layers, use_cuda=True, cam_type='gradcam'):
        """
        Args:
            model: The PyTorch network (e.g., our full MultiTask model)
            target_layers: list of nn.Module. Typically the last conv block 
                           e.g. [model.backbone.layer4[-1]]
            cam_type: 'gradcam' or 'scorecam'
        """
        self.model = model
        self.target_layers = target_layers
        
        if cam_type == 'gradcam':
            self.cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        elif cam_type == 'scorecam':
            self.cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        else:
            raise ValueError("cam_type must be gradcam or scorecam")
            
    def generate_heatmap(self, input_tensor, rgb_img, target_class=None):
        """
        Generate activation map overlaid on the original image.
        Args:
            input_tensor: [1, C, H, W] Torch tensor for inference.
            rgb_img: [H, W, 3] Numpy array of the original image normalized to [0,1].
            target_class: Integer ID for the target class. 
                          If None, uses the highest scoring class.
        Returns:
            visualization: [H, W, 3] Numpy array RGB image with heatmap.
            grayscale_cam: [H, W] Raw probabilities for diffusion refinement.
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        # Generates CAM
        # For multi-task, make sure pytorch_grad_cam can handle dict outputs
        # Often requires a wrapper model returning just the classification logit tensor
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Take the first image in batch
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return visualization, grayscale_cam
