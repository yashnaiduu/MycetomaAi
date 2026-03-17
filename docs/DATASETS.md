# Mycetoma AI: Dataset Guide

This guide outlines the datasets required to train the Mycetoma AI models and how to structure them for the multi-stage training pipeline.

## 1. Required Datasets

To achieve research-grade performance, we use a two-stage training approach with multiple datasets:

### Stage 1: Pretraining Datasets
Used for self-supervised learning (SSL) to teach the model general histopathology, nuclei segmentation, and fungal morphology features.

1. **LC25000 (Lung and Colon Cancer Histopathological Images)**
   - **Purpose:** General histopathology feature pretraining.
   - **Source:** [arXiv](https://arxiv.org/abs/1912.12142) or kaggle.
   - **Size:** 25,000 images

2. **OpenFungi**
   - **Purpose:** Fungal morphology learning.
   - **Source:** [MDPI](https://www.mdpi.com/2075-1729/15/7/1132)
   - **Size:** 1,249 images

3. **NuInsSeg**
   - **Purpose:** Tissue structure and nuclei features.
   - **Source:** [Kaggle / Public Repositories]
   - **Size:** 665 histology images

### Stage 2: Finetuning Dataset
Used to train the multi-task heads for classification (Eumycetoma vs Actinomycetoma), detection, and subtyping.

1. **MyData (Mycetoma Tissue Microscopic Images)**
   - **Purpose:** Final classification and evaluation.
   - **Source:** [Zenodo](https://zenodo.org/records/13655082) or [Mycetoma Research Centre](https://mycetoma.edu.sd/?p=5242)
   - **Size:** 864 images

## 2. Directory Structure

Download the datasets and extract them into a `data/` directory at the root of the project. The training scripts expect the following structure:

```text
MycetomaAi/
├── data/
│   ├── pretrain/                  # Stage 1 Datasets
│   │   ├── LC25000/
│   │   │   ├── lung_n/            # Folders containing images
│   │   │   ├── colon_n/
│   │   │   └── ...
│   │   ├── OpenFungi/
│   │   │   ├── images/
│   │   │   └── ...
│   │   └── NuInsSeg/
│   │       ├── images/
│   │       └── ...
│   │
│   └── finetune/                  # Stage 2 Dataset
│       └── MyData/
│           ├── images/            # Directory containing the 864 images
│           ├── labels.csv         # File containing classification/bounding max info
│           └── ...
```

## 3. Running Training

Once you have arranged the images according to the structure above, you can run the pipelines:

**Stage 1: Self-Supervised Pretraining**
```bash
python scripts/train.py --stage pretrain --pretrain_data_dir data/pretrain --epochs 100
```

**Stage 2: Multi-Task Finetuning**
```bash
python scripts/train.py --stage finetune --finetune_data_dir data/finetune/MyData --checkpoint checkpoints/ssl/ssl_encoder_ep100.pth --epochs 50
```
