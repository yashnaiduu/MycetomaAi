# Mycetoma AI Diagnostics

A research-grade AI system capable of diagnosing Mycetoma from histopathology microscopic images and identifying sub-types of causative organisms (Eumycetoma vs Actinomycetoma).

This architecture implements a multi-component deep learning pipeline engineered specifically to handle the "small dataset problem" common with Neglected Tropical Diseases.

## 🧠 System Architecture

The model combines Self-Supervised Learning (SSL), Weak Localization, Diffusion Masking, and Graph Neural Networks.

1. **Pre-Processing Component**:
   - `Macenko Stain Normalization` to reduce inter-laboratory visual variance.
   - Heavy spatial, morphological, and color augmentations tailored for histopathology.

2. **Self-Supervised Encoder (HybridSSLEncoder)**:
   - Combines structural learning from **SimCLR** with semantic awareness from **DINOv2**.
   - Solves the problem of missing large annotated datasets by learning underlying image distributions first.

3. **Attention-Enhanced Backbone (ResNet50CBAM)**:
   - ResNet50 core with injects of Channel and Spatial Attention Modules (CBAM).
   - Forces the network to ignore benign background human tissue and focus heavily on fungal/bacterial "grain" structures (which usually occupy < 20% of an image).

4. **Weakly Supervised Grain Localization & Diffusion Segmentation**:
   - Grad-CAM heatmaps generate rough class activation regions based purely on the classification signal.
   - A lightweight 2D Diffusion Model (`DiffusionSegmentationRefiner`) denoises this probability map to produce a refined pixel-perfect structural segmentation mask—eliminating the need for manual polygon labeling.

5. **Multi-Task & Few-Shot Learning**:
   - Instead of separate networks, a single `MultiTaskHead` jointly optimizes:
     - Fungal / Bacterial / Normal Classification (Cross Entropy)
     - Target coordinates detection (Smooth L1)
     - Granular strain subtype identification
   - A parallel `PrototypicalNetwork` allows extreme few-shot identification of exceptionally rare pathogens using Euclidean distances in an embedding space instead of standard linear classifiers.

6. **Graph Morphology Classification**:
   - `MorphologyGNN` processes extracted nodes from segmented grain boundaries, translating the structural branching patterns into a classification signal.

## 🛠 Project Structure

```bash
Mycetoma-AI/
├── README.md
├── requirements.txt
├── notebooks/
│   └── xai_exploration.ipynb   # Interactive analysis of Model Heatmaps
├── scripts/
│   ├── train.py                # Main CLI for Model pretraining and fine-tuning
│   └── evaluate.py             # CLI for testing models and generating evaluation metrics
└── src/
    ├── data/                   # Macenko Normalization, PyTorch DataLoaders, and Augmentations
    ├── evaluation/             # ROC-AUC, Sensitivity, Specificity, and Grad-CAM Explainer
    ├── models/                 # Architecture (Backbones, GNNs, Prototypical Nets, Diffusers)
    └── training/               # Custom Multi-Task Losses and Train loops
```

## 🚀 Usage Guide

### 1. Setup

```bash
pip install -r requirements.txt
```

> **Note**: For data preparation, you must configure the mock structures inside `scripts/train.py` to point to your `MyData` (~864 images) local directory structure. The input shapes expect square tensor crops post-normalization.

### 2. Self-Supervised Pretraining
Before providing any labels, let the encoder learn the visual domain.

```bash
python scripts/train.py --ssl --batch_size 16 --epochs 100
```
This forces the model to align views using SimCLR contrastive `InfoNCE` losses combined with DINOv2 frozen features.

### 3. Multi-Task Supervised Finetuning
Once pretrained, execute the supervised loop leveraging your subtype parameters, bounding boxes, and image labels.

```bash
python scripts/train.py --batch_size 16 --epochs 50 --lr 1e-4
```

### 4. Evaluation & XAI Visualization
To evaluate clinical thresholds globally on the test dataset:
```bash
python scripts/evaluate.py --model_path checkpoints/multitask/best_multi_task_model.pth
```
To visually inspect *why* the model made a decision, launch Jupyter and use `notebooks/xai_exploration.ipynb` to see the Grad-CAM activation overlays pinpointing the target fungus/bacteria.

---
*Built for deployment in clinical environments as offline diagnostic tooling and lightweight inference edge systems.*
