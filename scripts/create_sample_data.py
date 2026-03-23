"""Generate synthetic histopathology images for pipeline testing."""
import os
import argparse
import numpy as np
from PIL import Image


COLOR_PROFILES = {
    "Eumycetoma": {"base": (180, 120, 160), "grain": (60, 30, 80), "var": 30},
    "Actinomycetoma": {"base": (200, 150, 170), "grain": (100, 40, 60), "var": 25},
    "Normal": {"base": (210, 180, 190), "grain": None, "var": 20},
}


def make_synthetic_image(profile, size=224, rng=None):
    rng = rng or np.random.default_rng()
    base = np.array(profile["base"], dtype=np.float32)
    noise = rng.normal(0, profile["var"], (size, size, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)

    if profile["grain"] is not None:
        n_grains = rng.integers(3, 8)
        for _ in range(n_grains):
            cx, cy = rng.integers(20, size - 20, 2)
            r = rng.integers(8, 25)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            circle = xx**2 + yy**2 <= r**2
            grain_color = np.array(profile["grain"]) + rng.normal(0, 10, 3)
            y_start, y_end = max(0, cy - r), min(size, cy + r + 1)
            x_start, x_end = max(0, cx - r), min(size, cx + r + 1)
            mask_y = slice(y_start - (cy - r), y_end - (cy - r))
            mask_x = slice(x_start - (cx - r), x_end - (cx - r))
            region = circle[mask_y, mask_x]
            img[y_start:y_end, x_start:x_end][region] = np.clip(grain_color, 0, 255).astype(np.uint8)

    return img


def create_dataset(output_dir, images_per_class=10, size=224, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for class_name, profile in COLOR_PROFILES.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(images_per_class):
            img = make_synthetic_image(profile, size=size, rng=rng)
            path = os.path.join(class_dir, f"{class_name.lower()}_{i:03d}.png")
            Image.fromarray(img).save(path)
            total += 1

    print(f"Created {total} images in {output_dir}")
    for cls in COLOR_PROFILES:
        print(f"  {cls}: {images_per_class} images")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument("--output_dir", default="data/finetune", type=str)
    parser.add_argument("--images_per_class", default=10, type=int)
    parser.add_argument("--size", default=224, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    create_dataset(args.output_dir, args.images_per_class, args.size, args.seed)


if __name__ == "__main__":
    main()
