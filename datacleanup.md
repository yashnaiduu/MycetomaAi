
# ROLE
You are a senior ML data engineer preparing a dataset specifically for self-supervised learning (SimCLR).

You are working with 3 datasets:

- LC25000
- NuInsSeg
- OpenFungi

Your goal is NOT just cleaning — your goal is to make the dataset OPTIMAL for representation learning.

Do everything in ONE PASS.

---

# OBJECTIVE

Create a dataset that is:

- clean
- uniform
- diverse
- balanced across domains
- optimized for SimCLR pretraining

---

# OUTPUT STRUCTURE

data/pretrain_ready/
  LC25000/images/
  NuInsSeg/images/
  OpenFungi/images/

---

# TASKS

## 1. EXTRACT ONLY VALID IMAGES

LC25000:
- extract from both:
  Train and Validation Set
  Test Set

NuInsSeg:
- ONLY extract from:
  tissue images/

OpenFungi:
- extract from:
  macro/
  micro/

IGNORE ALL:
- masks
- overlays
- annotations
- maps

---

## 2. IMAGE STANDARDIZATION (CRITICAL)

For each image:

- convert to RGB
- ensure 3 channels
- resize to 224x224
- normalize format to JPG

---

## 3. QUALITY FILTERING

Remove images that are:

- too small (<128px)
- too blurry (low variance)
- corrupted

---

## 4. DEDUPLICATION

- detect duplicate images
- remove duplicates

---

## 5. DOMAIN BALANCING (VERY IMPORTANT)

Ensure:

- roughly equal number of images from:
  - LC25000
  - NuInsSeg
  - OpenFungi

If one dataset is larger:
- downsample it

---

## 6. FILE RENAMING

Rename all images:

<dataset>_<index>.jpg

---

## 7. FAST PROCESSING

- use batch processing
- avoid loading all images into memory
- process efficiently for large datasets (~12GB)

---

## 8. DATA VALIDATION

Ensure:

- all images load correctly
- all shapes are consistent (224x224x3)

---

## 9. LOGGING

Generate report:

- total images per dataset
- removed images
- duplicates removed
- final counts

---

## 10. TRAINING COMPATIBILITY

Ensure output is directly usable by:

- SimCLR dataset loader
- PyTorch DataLoader

---

## 11. CLEAN CODE RULES

- comments: 3–4 words only
- no unnecessary comments

---

# OUTPUT

Return:

1. pretrain_data_pipeline.py
2. cleaned dataset structure
3. dataset summary report
4. usage example for training

---

# CONSTRAINTS

- DO NOT keep original structure
- DO NOT keep labels
- DO NOT mix datasets
- DO NOT skip normalization

---

# FINAL GOAL

A dataset that is:

- perfectly structured for SimCLR
- balanced across domains
- consistent and clean
- ready for immediate pretraining


DO NOT resize images during preprocessing.

All resizing and cropping must be performed dynamically during training
using augmentation pipelines (SimCLR-style random crops).
