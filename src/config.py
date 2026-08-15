"""
Frozen experiment configuration.

These values were fixed in the study's preregistration (see PREREGISTRATION.md)
before any results were observed, with one documented exception: STEPS_PER_SAMPLE,
MIN_STEPS, and MAX_STEPS were introduced after a post-hoc convergence diagnostic
revealed that a fixed 30-step training budget under-trained the from-scratch
condition at larger fine-tuning sizes. This deviation is disclosed in the paper's
Methods section and in PREREGISTRATION.md.
"""

import torch

# --- Virtual patient population ---
SEED_LIST = [0, 1, 2, 3, 4]
N_PATIENTS = 100
N_PRETRAIN_PATIENTS = 80
N_HOLDOUT_PATIENTS = 20
PATIENT_GENERATION_SEED = SEED_LIST[0]

# --- Fine-tuning sweep ---
FINETUNE_SIZES = [5, 10, 20, 50, 100, 200]
REFERENCE_SIZE = 200
N_EVAL_IMAGES = 20

# --- Image / encoder ---
IMG_SIZE = 32

# --- Training budget (steps-scaled; see module docstring) ---
STEPS_PER_SAMPLE = 3
MIN_STEPS = 30
MAX_STEPS = 600
PRETRAIN_STEPS = 100
PRETRAIN_POOL_SIZE = 500

# --- Optimization ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 8

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Paths (override via CLI args in scripts/ if needed) ---
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
