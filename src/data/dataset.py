import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Tuple

from .stain_normalization import apply_macenko

class MycetomaDataset(Dataset):
    """
    Multi-task PyTorch dataset for Mycetoma microscopic images.
    Returns images, and labels for multiple tasks (classification, detection, subtype).
    """
    def __init__(self, 
                 image_paths: list, 
                 labels: Optional[list] = None, 
                 bounding_boxes: Optional[list] = None,
                 subtypes: Optional[list] = None,
                 transform: Optional[Callable] = None,
                 use_macenko: bool = True,
                 is_ssl: bool = False):
        """
        Args:
            image_paths (list): List of absolute file paths to the images.
            labels (list, optional): List of integer labels (0=Background, 1=Fungal, 2=Bacterial).
            bounding_boxes (list, optional): List of [x_min, y_min, x_max, y_max] arrays for the grains.
            subtypes (list, optional): List of integer labels identifying specific pathogen strains for few-shot learning.
            transform (callable, optional): Albumentations or Torchvision transforms to apply.
            use_macenko (bool): Whether to apply Macenko stain normalization.
            is_ssl (bool): If true, ignores labels and uses SimCLR self-supervised representation learning output transforms.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.subtypes = subtypes
        self.transform = transform
        self.use_macenko = use_macenko
        self.is_ssl = is_ssl
        
        # Validations for supervised multi-task runs
        if not is_ssl and labels is not None:
            assert len(labels) == len(image_paths)
            if bounding_boxes is not None:
                assert len(bounding_boxes) == len(image_paths)
            if subtypes is not None:
                assert len(subtypes) == len(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        # Load image via cv2 to apply numpy-based Macenko
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_macenko:
            image = apply_macenko(image)

        if self.is_ssl:
            # SSL Transform (SimCLR block expects PIL images)
            pil_image = Image.fromarray(image)
            view1, view2 = self.transform(pil_image)
            return {"view1": view1, "view2": view2}

        # Multi-task Supervised Flow
        out = {}
        
        # apply albumentations
        if self.transform:
            augmented = self.transform(image=image)
            out["image"] = augmented["image"]
        else:
            out["image"] = torch.tensor(image.transpose(2,0,1), dtype=torch.float32) / 255.0

        if self.labels is not None:
            out["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        if self.bounding_boxes is not None:
            # Box structure expected: [x_min, y_min, x_max, y_max]
            out["bbox"] = torch.tensor(self.bounding_boxes[idx], dtype=torch.float32)
            
        if self.subtypes is not None:
            out["subtype"] = torch.tensor(self.subtypes[idx], dtype=torch.long)
            
        return out
