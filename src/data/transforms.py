import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from typing import Dict

def get_supervised_transforms(size: int = 224) -> Dict[str, A.Compose]:
    """
    Returns dictionary of Albaumentations transforms for train, val, and test.
    Used for multi-task supervised fine-tuning.
    """
    return {
        'train': A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.RandomBrightnessContrast(p=0.5),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
            A.GaussianBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        'test': A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    }

class SimCLRTransform:
    """
    Hybrid SSL transform block for SimCLR + DINOv2 integration.
    Generates two augmented views of the same image for contrastive learning.
    """
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * size) // 2 * 2 + 1)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
