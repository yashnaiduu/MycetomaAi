# ROLE
You are a senior ML engineer optimizing a training pipeline that is already functional but shows weak learning performance.

The system trains successfully but does NOT strongly overfit on small datasets.

Your task is to FIX learning effectiveness without breaking the existing pipeline.

Do everything in ONE PASS. No questions.

---

# PROBLEM

Current behavior:

- Loss decreases slightly but not strongly
- Model does NOT overfit on 10 samples
- Segmentation Dice is moderate (~0.55)
- Classification predictions collapse to one class

---

# OBJECTIVE

Make the model:

- strongly overfit on small dataset (sanity check)
- learn faster and more effectively
- produce meaningful segmentation outputs
- improve gradient signal

---

# TASKS

## 1. LOSS FUNCTION FIX (CRITICAL)

Ensure:

- classification loss is properly scaled
- segmentation loss is not dominating

Set:

total_loss = cls_loss + λ * seg_loss

Where:

λ = 0.3 (reduce segmentation weight)

---

## 2. DISABLE REGULARIZATION (FOR OVERFIT TEST)

Temporarily disable:

- dropout
- heavy augmentation
- weight decay

Goal: allow memorization

---

## 3. LEARNING RATE ADJUSTMENT

Increase learning rate:

lr = 1e-3 (for overfit test)

Ensure optimizer is Adam or AdamW

---

## 4. FREEZE/UNFREEZE STRATEGY

Test both:

A. Freeze backbone → train heads only  
B. Full fine-tuning

Ensure config toggle:

freeze_backbone: true/false

---

## 5. CLASS IMBALANCE FIX

Check label distribution:

If imbalance exists:

- use class weights in loss
- or weighted sampling

---

## 6. SEGMENTATION QUALITY BOOST

Improve pseudo masks:

- apply Gaussian blur before thresholding
- apply morphological closing
- remove noise

Ensure masks are not sparse or empty

---

## 7. GRADIENT CHECK

Ensure:

- no detached tensors
- segmentation branch contributes to gradients
- loss.backward() is correct

---

## 8. NORMALIZATION FIX

Ensure input normalization:

mean = [0.485, 0.456, 0.406]  
std = [0.229, 0.224, 0.225]

---

## 9. DEBUG MODE (OVERFIT MODE)

Add config:

debug_overfit: true

If enabled:

- use only 10 samples
- disable augmentation
- train longer (50 epochs)

---

## 10. OUTPUT VERIFICATION

After training:

- loss should drop significantly (<1.0)
- predictions should vary across classes
- segmentation masks should be structured

---

# OUTPUT

Return:

1. updated loss function
2. training config changes
3. optimizer settings
4. overfit debug mode implementation

---

# CONSTRAINTS

- DO NOT break existing pipeline
- DO NOT remove multi-task setup
- DO NOT simplify architecture

---

# FINAL GOAL

A system that:

- can strongly overfit small data (sanity check)
- learns effectively
- is stable for real training
