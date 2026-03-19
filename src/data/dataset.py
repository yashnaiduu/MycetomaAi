import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Tuple

from .stain_normalization import apply_macenko

def get_image_paths(directory: str) -> list:
    """Recursively find all images."""
    valid_exts = {'.jpg', '.png', '.jpeg', '.tif', '.tiff'}
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                paths.append(os.path.join(root, file))
    return paths

class MycetomaDataset(Dataset):
    """Multi-task dataset for Mycetoma images."""
    def __init__(self, 
                 image_paths: list, 
                 labels: Optional[list] = None, 
                 bounding_boxes: Optional[list] = None,
                 subtypes: Optional[list] = None,
                 transform: Optional[Callable] = None,
                 use_macenko: bool = True,
                 is_ssl: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.subtypes = subtypes
        self.transform = transform
        self.use_macenko = use_macenko
        self.is_ssl = is_ssl
        
        if not is_ssl and labels is not None:
            assert len(labels) == len(image_paths)
            if bounding_boxes is not None:
                assert len(bounding_boxes) == len(image_paths)
            if subtypes is not None:
                assert len(subtypes) == len(image_paths)

    @classmethod
    def from_ssl_directories(cls, root_dirs: list, **kwargs):
        """Build SSL dataset from directories."""
        all_paths = []
        for d in root_dirs:
            if os.path.isdir(d):
                all_paths.extend(get_image_paths(d))
            else:
                print(f"Warning: SSL directory {d} not found.")
                
        kwargs['is_ssl'] = True
        return cls(image_paths=all_paths, labels=None, bounding_boxes=None, subtypes=None, **kwargs)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_macenko:
            image = apply_macenko(image)

        if self.is_ssl:
            pil_image = Image.fromarray(image)
            view1, view2 = self.transform(pil_image)
            return {"view1": view1, "view2": view2}

        out = {}
        
        if self.transform:
            augmented = self.transform(image=image)
            out["image"] = augmented["image"]
        else:
            out["image"] = torch.tensor(image.transpose(2,0,1), dtype=torch.float32) / 255.0

        if self.labels is not None:
            out["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        if self.bounding_boxes is not None:
            out["bbox"] = torch.tensor(self.bounding_boxes[idx], dtype=torch.float32)
            
        if self.subtypes is not None:
            out["subtype"] = torch.tensor(self.subtypes[idx], dtype=torch.long)
            
        return out

class MultiDatasetWrapper(Dataset):
    """Balanced sampling across datasets."""
    def __init__(self, datasets: list, samples_per_dataset: Optional[int] = None):
        self.datasets = datasets
        self.samples_per_dataset = samples_per_dataset
        
        if samples_per_dataset is not None:
            self.length = len(datasets) * samples_per_dataset
        else:
            self.length = sum(len(d) for d in datasets)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.samples_per_dataset is not None:
            dataset_idx = idx // self.samples_per_dataset
            sample_idx = idx % self.samples_per_dataset
            # Loop if dataset smaller
            sample_idx = sample_idx % len(self.datasets[dataset_idx])
            return self.datasets[dataset_idx][sample_idx]
        else:
            curr_idx = idx
            for d in self.datasets:
                if curr_idx < len(d):
                    return d[curr_idx]
                curr_idx -= len(d)
            raise IndexError("Index out of bounds")
