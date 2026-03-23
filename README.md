# Mycetoma AI Diagnostics

AI system for diagnosing Mycetoma from histopathology images. Multi-task deep learning pipeline handling medical data scarcity via self-supervised learning and attention mechanisms.

## Research Contributions

**Self-Supervised Pretraining (SimCLR + DINOv2)** — Hybrid contrastive encoder fusing SimCLR projections with frozen DINOv2 features. Learns histopathology representations from unlabeled images (~864 images).

**Attention-Based Classification (CBAM)** — Channel + spatial attention integrated into every ResNet50 block. Guides the network toward grain morphology patterns.

**Multi-Task Learning** — Joint optimization of classification (CrossEntropy), segmentation (DiceBCE), detection (SmoothL1), and subtype classification through a shared backbone with configurable loss weights.

**Weakly Supervised Segmentation** — Otsu-based pseudo-mask generation enables segmentation training without pixel-level annotations.

**Ablation Study** — Systematic comparison of 5 model variants (ResNet50, DenseNet121, +CBAM, +Aug, +SSL) via stratified k-fold cross-validation.

## Project Structure

```
├── configs/default.yaml            # All tunable parameters
├── scripts/
│   ├── train.py                    # SSL pretrain / multi-task finetune
│   ├── evaluate.py                 # Metrics + confusion matrix
│   ├── ablation.py                 # Ablation study (5 variants)
│   ├── create_sample_data.py       # Synthetic test data
│   └── validate_pipeline.py        # End-to-end validation
├── src/
│   ├── data/                       # Dataset, transforms, stain norm
│   ├── models/                     # Backbone, SSL, heads, segmentation
│   ├── training/                   # Losses, trainer, SSL pretrainer
│   └── evaluation/                 # Metrics, Grad-CAM
├── notebooks/
│   └── mycetoma_training.ipynb     # Complete Colab notebook
├── backend/                        # FastAPI inference server
├── frontend/                       # Next.js diagnostic UI
└── tests/
```

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .

# Generate test data
python scripts/create_sample_data.py --output_dir data/finetune --images_per_class 20

# Train (all params from config)
python scripts/train.py --config configs/default.yaml --stage finetune

# Evaluate
python scripts/evaluate.py --model_path checkpoints/multitask/best_multi_task_model.pth

# Full validation
python scripts/validate_pipeline.py
```

## Training with Real Data

1. Organize images into class folders:
```
data/finetune/
├── Eumycetoma/
├── Actinomycetoma/
└── Normal/
```

2. Edit `configs/default.yaml` — all behavior is configurable:
```yaml
finetune_data_dir: data/finetune
freeze_backbone: false
loss_alpha: 1.0      # classification weight
loss_delta: 0.5      # segmentation weight
generate_masks: true  # Otsu pseudo-masks when annotations unavailable
```

3. Run: `python scripts/train.py --config configs/default.yaml --stage finetune`

## Docker

```bash
docker build -t mycetomai:latest .
docker run --gpus all -v $(pwd)/data:/workspace/data mycetomai:latest \
    python scripts/train.py --config configs/default.yaml --stage finetune
```
