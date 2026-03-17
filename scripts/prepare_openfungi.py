import os
import random
import shutil
from pathlib import Path
import kagglehub

def prepare_openfungi_subset(target_dir: str, sample_ratio: float = 0.25):
    """
    Downloads the OpenFungi dataset via KaggleHub and extracts a stratified subset.
    """
    print("🚀 Initiating OpenFungi Download via KaggleHub...")
    
    # Download latest version. KaggleHub caches this, so repeated runs won't redownload.
    cache_path = kagglehub.dataset_download("deepanshugupta1501/openfungi")
    source_path = Path(cache_path)
    print(f"✅ Download complete. Cached at: {source_path}")
    
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"⚙️ Starting stratified subsetting (Target ~{sample_ratio * 100}%)...")
    
    # Kaggle dataset often contains a parent folder or direct class folders.
    # Let's find directories that contain files directly.
    class_dirs = [d for d in source_path.rglob("*") if d.is_dir() and any(f.is_file() for f in d.iterdir())]

    if not class_dirs:
        print("❌ Error: Could not find any directories containing files in the downloaded dataset.")
        return

    # To avoid matching intermediate directories like the root folder if it has files, we filter out parents of other matched dirs.
    # But for an image dataset, usually leaf directories are the classes.
    leaf_dirs = []
    for d in class_dirs:
        has_subdir = any(sub.is_dir() for sub in d.iterdir())
        if not has_subdir:
            leaf_dirs.append(d)
    
    if not leaf_dirs:
        # Fallback to class_dirs if no leaf dirs were found somehow
        leaf_dirs = class_dirs

    total_copied = 0
    total_found = 0

    for class_dir in leaf_dirs:
        # Create corresponding class folder in our project subset directory
        target_class_dir = target_path / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        # Get all images for this class
        images = [f for f in class_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        total_found += len(images)
        
        if not images:
            continue

        # Calculate samples required for this class
        num_samples = max(1, int(len(images) * sample_ratio))
        
        # Randomly select the images (Stratified Sampling)
        sampled_images = random.sample(images, num_samples)

        for img_path in sampled_images:
            target_file = target_class_dir / img_path.name
            if not target_file.exists():
                shutil.copy2(img_path, target_file)
                total_copied += 1

    print(f"✅ Subset preparation complete!")
    print(f"📊 Total images found in full dataset: {total_found}")
    print(f"📦 Total images copied to subset: {total_copied}")
    print(f"📁 Subset location: {target_path.absolute()}")

if __name__ == "__main__":
    # Ensure random seed is fixed for reproducibility
    random.seed(42)
    
    # Define target path according to the DATASETS.md folder structure
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SUBSET_TARGET_DIR = PROJECT_ROOT / "data" / "pretrain" / "OpenFungi_Subset_2GB"
    
    prepare_openfungi_subset(target_dir=str(SUBSET_TARGET_DIR), sample_ratio=0.25)
