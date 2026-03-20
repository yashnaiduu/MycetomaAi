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
├── pyproject.toml
├── Dockerfile
├── configs/
│   └── default.yaml
├── notebooks/
│   └── colab_training_setup.ipynb
├── scripts/
│   ├── train.py            # Main entry point (Pretrain/Finetune)
│   └── evaluate.py         # Model evaluation
├── src/
│   ├── data/               # DataLoaders & StainNormalization
│   ├── models/             # Core AI Architectures
│   └── training/           # Loss functions & Trainer loops
└── tests/
    └── test_trainer_smoke.py
```

## 📦 Installation (pip / editable)

```bash
git clone https://github.com/yashnaiduu/MycetomaAi.git
cd MycetomaAi

# GPU (CUDA 12.1)
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Once installed in editable mode the console scripts are available:

```bash
# SSL pre-training
mycetomai-train --stage pretrain --pretrain_data_dir data/pretrain --epochs 100

# Supervised fine-tuning
mycetomai-train --stage finetune --finetune_data_dir data/finetune \
    --checkpoint checkpoints/ssl/ssl_encoder_ep100.pth

# Evaluation
mycetomai-eval --model_path checkpoints/multitask/best_multi_task_model.pth
```

You can also load the default config and override specific keys:

```bash
mycetomai-train --config configs/default.yaml --epochs 30 --batch_size 64
```

## 🐳 Docker (GPU)

```bash
# Build
docker build -t mycetomai:latest .

# Run training (mount data and checkpoints)
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    mycetomai:latest \
    python scripts/train.py --config configs/default.yaml --stage finetune
```

## ⚙️ Running CI Locally

Install [act](https://github.com/nektos/act) and then run:

```bash
act push
```

Or run the individual steps manually:

```bash
# Install CPU PyTorch
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
grep -vE '^(torch|torchvision|torchaudio|--extra-index-url)' requirements.txt \
    | pip install -r /dev/stdin
pip install -e . ruff pytest

# Lint
ruff check src/ scripts/ tests/

# Tests
pytest -v tests/
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
