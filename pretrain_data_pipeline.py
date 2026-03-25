import os
import glob
import cv2
import shutil
import random
import logging
from pathlib import Path
import concurrent.futures
from PIL import Image

# setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = "/Users/yashnaidu/Proj/MycetomaAi"
OUT_DIR = os.path.join(BASE_DIR, "data", "pretrain_ready")

# dataset configs
DATASETS = {
    "LC25000": {"pattern": "LC25000/**/*.*"},
    "NuInsSeg": {"pattern": "NuInsSeg/**/tissue images/*.*"},
    "OpenFungi": {"pattern": "openfungi/**/*.*"}
}

def is_valid_file(path_str):
    """Filter valid images."""
    lower = path_str.lower()
    
    # filter extensions
    if not lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return False
        
    # ignore masks/overlays/maps
    for excl in ["mask", "overlay", "annotation", "map", "label"]:
        if excl in lower:
            return False
            
    # OpenFungi paths
    if "openfungi" in lower and "macro" not in lower and "micro" not in lower:
        return False
        
    return True

def compute_dhash(image, hash_size=8):
    """Compute difference hash."""
    img_gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img_gray.getdata())
    
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            diff.append(img_gray.getpixel((col, row)) > img_gray.getpixel((col + 1, row)))
            
    return sum([2 ** i for i, v in enumerate(diff) if v])

def analyze_image(path, dataset_name):
    """Phase 1: quality check."""
    try:
        with Image.open(path) as img:
            # check size
            if img.width < 128 or img.height < 128:
                return None
                
            img_rgb = img.convert("RGB")
            
            # check blur
            cv_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if cv_img is None:
                return None
            if cv2.Laplacian(cv_img, cv2.CV_64F).var() < 25.0:
                return None
                
            hash_val = compute_dhash(img_rgb)
            return {"path": path, "dataset": dataset_name, "hash": hash_val}
            
    except Exception:
        return None

def process_and_save(item, out_path):
    """Phase 4: standardize and save."""
    try:
        with Image.open(item["path"]) as img:
            img_rgb = img.convert("RGB")
            img_rgb.save(out_path, "JPEG", quality=95)
        return True
    except Exception:
        return False

def main():
    logging.info("Starting SimCLR Pretrain Pipeline...")
    
    raw_paths = []
    for ds_name, config in DATASETS.items():
        pattern = os.path.join(BASE_DIR, config["pattern"])
        for p in glob.glob(pattern, recursive=True):
            if is_valid_file(p):
                raw_paths.append((p, ds_name))
                
    logging.info(f"Phase 1: Analyzing {len(raw_paths)} images...")
    
    valid_items = []
    with concurrent.futures.ProcessPoolExecutor() as exc:
        futures = {exc.submit(analyze_image, p, ds): ds for p, ds in raw_paths}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                valid_items.append(res)
                
    logging.info(f"Valid items: {len(valid_items)}")
    
    logging.info("Phase 2: Deduplication...")
    unique_items = []
    seen_hashes = set()
    
    for item in valid_items:
        if item["hash"] not in seen_hashes:
            seen_hashes.add(item["hash"])
            unique_items.append(item)
            
    # grouping
    ds_groups = {k: [] for k in DATASETS.keys()}
    for item in unique_items:
        ds_groups[item["dataset"]].append(item)
        
    logging.info("Phase 3: Domain Balancing...")
    min_count = min([len(v) for v in ds_groups.values() if len(v) > 0])
    
    balanced_items = []
    for ds_name, items in ds_groups.items():
        if len(items) > 0:
            logging.info(f"- {ds_name} downsampled from {len(items)} to {min_count}")
            balanced_items.extend(random.sample(items, min_count))
            
    logging.info("Phase 4: Processing Output...")
    
    # recreate output
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        
    for ds_name in DATASETS.keys():
        os.makedirs(os.path.join(OUT_DIR, ds_name, "images"), exist_ok=True)
        
    success_count = 0
    counters = {k: 0 for k in DATASETS.keys()}
    
    with concurrent.futures.ProcessPoolExecutor() as exc:
        futures = []
        for item in balanced_items:
            ds = item["dataset"]
            idx = counters[ds]
            out_file = os.path.join(OUT_DIR, ds, "images", f"{ds}_{idx}.jpg")
            futures.append(exc.submit(process_and_save, item, out_file))
            counters[ds] += 1
            
        for fut in concurrent.futures.as_completed(futures):
            if fut.result():
                success_count += 1
                
    # final report
    logging.info("\n--- DATASET SUMMARY REPORT ---")
    logging.info(f"Original valid files: {len(raw_paths)}")
    logging.info(f"Failed size/blur checks: {len(raw_paths) - len(valid_items)}")
    logging.info(f"Duplicates removed: {len(valid_items) - len(unique_items)}")
    for ds_name in DATASETS.keys():
        logging.info(f"Final balanced [{ds_name}]: {min_count} images")
    logging.info("------------------------------")
    logging.info(f"Total processed output: {success_count} images")
    
    logging.info("\n--- USAGE EXAMPLE ---")
    logging.info("import torchvision")
    logging.info("import torch")
    logging.info("dataset = torchvision.datasets.ImageFolder('data/pretrain_ready')")
    logging.info("loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)")

if __name__ == "__main__":
    main()
