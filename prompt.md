# 🚀 **RESEARCH UPGRADE PROMPT (COPY THIS FULL MD)**

```md id="mycetoma_research_upgrade_prompt_v1"

# ROLE
You are a senior AI researcher (medical imaging), deep learning expert, and MLOps engineer.

You are improving an existing Mycetoma AI system that is already production-ready but lacks research-level depth.

You MUST upgrade it to a **publishable research system**.

Do everything in ONE PASS. No questions.

---

# CONTEXT

Current system has:

- ResNet50 + CBAM
- K-fold training
- FastAPI backend
- Next.js frontend
- GradCAM explainability

BUT missing:

- self-supervised learning (SimCLR)
- true multi-task learning (segmentation not real)
- ablation studies
- baseline comparisons
- proper packaging (uses PYTHONPATH hack)

---

# OBJECTIVE

Upgrade system to:

- research-paper ready
- experimentally validated
- enterprise-grade packaging
- cost-efficient LLM usage

---

# TASKS

## 1. ADD SELF-SUPERVISED PRETRAINING (CRITICAL)

Implement SimCLR:

- encoder: ResNet50 backbone
- projection head (MLP)
- contrastive loss (NT-Xent)

Add:

- augmentations:
  - rotation
  - color jitter
  - gaussian blur
  - crop

Pipeline:

Pretrain → Save encoder → Fine-tune classifier

Ensure:

- modular design
- reusable encoder weights

---

## 2. FIX MULTI-TASK LEARNING (REAL IMPLEMENTATION)

Current system uses GradCAM (not real localization).

Implement TRUE multi-task model:

- shared encoder
- classification head
- segmentation head (UNet-style or decoder)

Train jointly:

Loss =
- classification loss
- segmentation loss

Output:

- class prediction
- segmentation mask

---

## 3. LLM COST OPTIMIZATION (STRICT)

Audit and fix LLM usage:

- ensure NO per-request repeated calls
- implement caching (hash input → reuse output)
- enforce short prompts
- deterministic outputs (temperature=0)

Add:

- max token limit
- fallback logic (if cache hit)

---

## 4. ADD ABLATION STUDY (MANDATORY)

Implement experiments:

Train and compare:

1. baseline ResNet50
2. + CBAM
3. + augmentation
4. + SimCLR

Output:

- accuracy
- F1 score
- ROC-AUC

Generate:

- comparison table
- structured results

---

## 5. ADD BENCHMARK MODELS

Implement:

- plain ResNet50
- DenseNet121

Train under same conditions.

Compare against your model.

---

## 6. METRICS (RESEARCH STANDARD)

Add:

- Accuracy
- F1 Score
- ROC-AUC
- Confusion Matrix

Ensure:

- logging during training
- evaluation script

---

## 7. FIX PYTHON PACKAGING (CRITICAL)

REMOVE PYTHONPATH dependency.

Restructure:

```

backend/
src/
mycetoma_ai/
**init**.py
models/
training/
inference/
api/

```

---

Create:

pyproject.toml:

```

[project]
name = "mycetoma_ai"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["src"]

```

---

Enable:

pip install -e .

Ensure:

- imports use package name
- no relative hacks
- IDE works without errors

---

## 8. UPDATE README (PROFESSIONAL)

Add:

### Installation

```

cd backend
pip install -e .

```

---

Add:

### Research Contributions

- self-supervised learning
- attention-based classification
- multi-task learning

---

## 9. CLEAN CODE RULES (STRICT)

- comments: 3–4 words only
- commit messages: 2 words only
- no unnecessary comments
- no verbose explanations

---

## 10. OUTPUT FORMAT

Return:

1. SimCLR implementation (code)
2. multi-task model code
3. ablation study pipeline
4. benchmark models
5. metrics evaluation code
6. updated folder structure
7. packaging files (pyproject.toml)
8. updated README section

---

# CONSTRAINTS

- ONE RESPONSE ONLY
- NO QUESTIONS
- MINIMAL TOKEN USAGE
- MAXIMUM QUALITY

---

# FINAL GOAL

Transform system into:

- publishable research project
- enterprise-grade codebase
- reproducible experiment pipeline
- cost-efficient AI system

```

---

# 🔥 Why this works

This prompt forces the model to:

✅ Add **actual research novelty (SimCLR)**
✅ Fix **fake explainability → real multi-task learning**
✅ Add **experiments (critical for paper)**
✅ Fix **packaging (enterprise standard)**
✅ Prevent **LLM cost explosion**

---

# 💡 How to use (IMPORTANT)

👉 Run this AFTER your first prompt output
👉 Paste only:

* model files
* training pipeline
* explanation service



