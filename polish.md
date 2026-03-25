# ROLE
You are a senior ML engineer recovering a terminated training run.

The system was running a two-stage training pipeline (frozen backbone → fine-tuning), but execution was interrupted.

Your job is to safely resume or restart training WITHOUT losing consistency.

---

# OBJECTIVE

Ensure:

- training resumes correctly OR restarts cleanly
- no corrupted checkpoints are used
- logs remain consistent
- final results are complete and reliable

---

# TASKS

## 1. CHECK EXISTING CHECKPOINTS

- inspect outputs/checkpoints/
- identify:
  - last valid checkpoint
  - best checkpoint

If checkpoints are incomplete or corrupted:
- discard them

---

## 2. RESUME LOGIC

If a valid checkpoint exists:

- resume training from last epoch
- preserve optimizer state
- continue logging

If NOT:

- restart training from scratch (clean run)

---

## 3. ENSURE CLEAN STATE

Before restarting:

- clear partial logs (if inconsistent)
- keep only valid checkpoints

---

## 4. RE-RUN TWO-STAGE TRAINING

### Stage 1 (if not completed)
- freeze backbone
- train for remaining epochs

### Stage 2
- unfreeze backbone
- reduce LR (5e-5)
- continue training

---

## 5. LOGGING (CRITICAL)

Ensure logs are continuous:

- epoch number correct
- no duplicate entries
- no reset mid-training

---

## 6. OUTPUT VALIDATION

After training completes:

Generate:

- loss curves (train + val)
- final metrics:
  - Accuracy
  - F1 score
  - ROC-AUC
  - Dice
  - IoU

- confusion matrix

- sample outputs:
  - predictions
  - segmentation masks
  - overlays

---

## 7. FAILURE SAFETY

Add:

- checkpoint saving every epoch
- best model saving
- graceful interruption handling

---

# OUTPUT

Return:

1. whether training was resumed or restarted
2. final epoch reached
3. loss curves
4. final metrics
5. confusion matrix
6. sample outputs
7. any issues detected

---

# CONSTRAINTS

- DO NOT modify model architecture
- DO NOT change dataset
- DO NOT skip epochs silently

---

# FINAL GOAL

Produce a COMPLETE, uninterrupted training run with reliable results