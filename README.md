# Mycetoma AI Diagnostics

A research-grade AI system for diagnosing Mycetoma from histopathology microscopic images. This system implements a multi-component deep learning pipeline engineered specifically to handle medical data scarcity.

## 🚀 Training on Google Colab

Use the provided `notebooks/colab_training_setup.ipynb` for cloud training. 

### 🔧 Critical Fixes for Colab Sessions
To resolve common path and permission errors:

1. **Kaggle Dataset Download:**
   The dataset is owned by the project service account. Use this path:
   ```bash
   !kaggle datasets download -d supporoot/mycetoma-ai-pretraining-data --unzip -p data/
   ```

2. **PYTHONPATH Configuration:**
   Before running any training scripts, tell Python where the `src` module is located:
   ```bash
   %env PYTHONPATH=.:$PYTHONPATH
   ```

3. **Working Directory:**
   Ensure you are inside the repository root:
   ```bash
   %cd /content/MycetomaAi
   ```

## 🧠 System Architecture

- **Attention-Enhanced Backbone (ResNet50CBAM):** Focuses on fungal/bacterial grain morphology.
- **Hybrid Self-Supervised Encoder:** Fuses SimCLR and DINOv2 for robust feature extraction without labels.
- **Multi-Task Diagnostic Head:** Jointly optimizes classification, detection, and subtype identification.

## 🛠 Project Structure

```bash
MycetomaAi/
├── README.md
├── requirements.txt
├── kaggle.json             # (Local only, Git-ignored)
├── notebooks/
│   └── colab_training_setup.ipynb
├── scripts/
│   ├── train.py            # Main entry point (Pretrain/Finetune)
│   └── evaluate.py         # Model evaluation
└── src/
    ├── data/               # DataLoaders & StainNormalization
    ├── models/             # Core AI Architectures
    └── training/           # Loss functions & Trainer loops
```

## 🛠 Usage

### SSL Pretraining
```bash
python scripts/train.py --stage pretrain --epochs 100
```

### Supervised Finetuning
```bash
python scripts/train.py --stage finetune --checkpoint checkpoints/ssl/ssl_encoder_ep100.pth
```

---
*Optimized for high-performance training on NVIDIA T4/A100 GPUs via Google Colab.*
