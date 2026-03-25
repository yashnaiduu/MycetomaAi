import os
import csv
import logging

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, List, Tuple

from .stain_normalization import apply_macenko

logger = logging.getLogger(__name__)

VALID_EXTS = {'.jpg', '.png', '.jpeg', '.tif', '.tiff'}


def get_image_paths(directory: str) -> list:
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(root, file))
    return sorted(paths)


def load_annotations_csv(csv_path: str) -> Dict[str, dict]:
    """Load annotations from CSV: filename,label[,subtype][,bbox][,mask_path]."""
    annotations = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']
            ann = {'label': int(row['label'])}
            if 'subtype' in row and row['subtype']:
                ann['subtype'] = int(row['subtype'])
            if 'mask_path' in row and row['mask_path']:
                ann['mask_path'] = row['mask_path']
            if 'bbox' in row and row['bbox']:
                ann['bbox'] = [float(x) for x in row['bbox'].split(';')]
            annotations[fname] = ann
    return annotations


def infer_labels_from_folders(data_dir: str) -> Tuple[List[str], List[int], Dict[int, str]]:
    """Infer labels from subfolder names (class-per-folder structure)."""
    paths, labels = [], []
    class_map = {}

    subdirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])

    if not subdirs:
        return [], [], {}

    has_images_in_subdirs = False
    for idx, subdir in enumerate(subdirs):
        class_map[idx] = subdir
        subdir_path = os.path.join(data_dir, subdir)
        imgs = get_image_paths(subdir_path)
        if imgs:
            has_images_in_subdirs = True
        for img in imgs:
            paths.append(img)
            labels.append(idx)

    if not has_images_in_subdirs:
        return [], [], {}

    logger.info("Class mapping: %s", class_map)
    return paths, labels, class_map


def generate_pseudo_mask(image: np.ndarray) -> np.ndarray:
    """Generate weak supervision mask via Otsu + Canny edge fusion."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu branch
    thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu_mask = cv2.threshold(blurred, max(0, thresh - 15), 255, cv2.THRESH_BINARY_INV)

    # Canny edge branch for boundary sharpness
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    edge_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    # Fuse: union of Otsu region and edge-detected boundaries
    mask = cv2.bitwise_or(otsu_mask, edge_dilated)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Smooth to reduce jagged edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if np.sum(mask) < 0.05 * mask.shape[0] * mask.shape[1] * 255:
        h, w = mask.shape
        cv2.circle(mask, (w//2, h//2), int(min(h, w)*0.35), 255, -1)

    return (mask / 255.0).astype(np.float32)


class MycetomaDataset(Dataset):
    """Multi-task dataset with folder-based or CSV-based labels."""
    def __init__(
        self,
        image_paths: list,
        labels: Optional[list] = None,
        bounding_boxes: Optional[list] = None,
        subtypes: Optional[list] = None,
        mask_paths: Optional[list] = None,
        transform: Optional[Callable] = None,
        use_macenko: bool = True,
        is_ssl: bool = False,
        generate_masks: bool = True,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.subtypes = subtypes
        self.mask_paths = mask_paths
        self.transform = transform
        self.use_macenko = use_macenko
        self.is_ssl = is_ssl
        self.generate_masks = generate_masks
        self.target_size = target_size

        if not is_ssl and labels is not None:
            assert len(labels) == len(image_paths)

    @classmethod
    def from_directory(cls, data_dir: str, **kwargs):
        """Build dataset from class-per-folder or flat directory."""
        paths, labels, class_map = infer_labels_from_folders(data_dir)
        if paths:
            logger.info("Loaded %d images across %d classes from folders", len(paths), len(class_map))
            return cls(image_paths=paths, labels=labels, **kwargs), class_map

        flat_paths = get_image_paths(data_dir)
        if flat_paths:
            logger.warning("Flat directory detected — assigning label 0 to all %d images", len(flat_paths))
            return cls(image_paths=flat_paths, labels=[0] * len(flat_paths), **kwargs), {0: "unknown"}

        logger.error("No images found in %s", data_dir)
        return cls(image_paths=[], labels=[], **kwargs), {}

    @classmethod
    def from_csv(cls, csv_path: str, data_dir: str, **kwargs):
        """Build dataset from CSV annotations."""
        annotations = load_annotations_csv(csv_path)
        paths, labels, subtypes_list, mask_paths_list, bboxes = [], [], [], [], []

        for fname, ann in annotations.items():
            img_path = os.path.join(data_dir, fname)
            if not os.path.exists(img_path):
                continue
            paths.append(img_path)
            labels.append(ann['label'])
            subtypes_list.append(ann.get('subtype'))
            mask_paths_list.append(ann.get('mask_path'))
            bboxes.append(ann.get('bbox'))

        has_subtypes = any(s is not None for s in subtypes_list)
        has_masks = any(m is not None for m in mask_paths_list)
        has_bboxes = any(b is not None for b in bboxes)

        return cls(
            image_paths=paths,
            labels=labels,
            subtypes=subtypes_list if has_subtypes else None,
            mask_paths=mask_paths_list if has_masks else None,
            bounding_boxes=bboxes if has_bboxes else None,
            **kwargs,
        )

    @classmethod
    def from_ssl_directories(cls, root_dirs: list, **kwargs):
        all_paths = []
        for d in root_dirs:
            if os.path.isdir(d):
                all_paths.extend(get_image_paths(d))
        kwargs['is_ssl'] = True
        return cls(image_paths=all_paths, labels=None, **kwargs)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_mask(self, idx: int, image: np.ndarray) -> Optional[torch.Tensor]:
        """Load real mask or generate pseudo-mask."""
        h, w = self.target_size

        if self.mask_paths and self.mask_paths[idx]:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (w, h))
                mask = (mask / 255.0).astype(np.float32)
                return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.generate_masks:
            mask = generate_pseudo_mask(image)
            mask = cv2.resize(mask, (w, h))
            return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load: {img_path}")
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
            out["image"] = torch.tensor(
                image.transpose(2, 0, 1), dtype=torch.float32
            ) / 255.0

        if self.labels is not None:
            out["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.bounding_boxes and self.bounding_boxes[idx] is not None:
            out["bbox"] = torch.tensor(self.bounding_boxes[idx], dtype=torch.float32)

        if self.subtypes and self.subtypes[idx] is not None:
            out["subtype"] = torch.tensor(self.subtypes[idx], dtype=torch.long)

        mask = self._load_mask(idx, image)
        if mask is not None:
            out["mask"] = mask

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
            sample_idx = sample_idx % len(self.datasets[dataset_idx])
            return self.datasets[dataset_idx][sample_idx]
        else:
            curr_idx = idx
            for d in self.datasets:
                if curr_idx < len(d):
                    return d[curr_idx]
                curr_idx -= len(d)
            raise IndexError("Index out of bounds")
