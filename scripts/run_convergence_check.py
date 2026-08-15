"""Post-hoc convergence diagnostic.

Verifies that the training step budget used in the main sweep (steps_for_size,
capped at MAX_STEPS) is sufficient for both the pretrained and from-scratch
conditions to reach a stable loss plateau at the largest fine-tuning size.

This script reproduces the diagnostic that originally caught two bugs during
the study's development:
  1. A fixed 30-step budget under-trained the from-scratch condition at large
     fine-tuning sizes (confounding the data-efficiency comparison).
  2. After introducing a steps-scaled budget, MAX_STEPS=300 under-trained the
     pretrained condition specifically (it was still improving ~30% past the
     cutoff); MAX_STEPS was raised to 600 as a result.

Usage:
    python -m scripts.run_convergence_check
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.patients import generate_virtual_patients, split_patients
from src.simulator import build_model_for_patient, compute_electrode_templates
from src.encoder import StimulusEncoder
from src.training import train_encoder

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import random


def main(n_check_steps: int = 600, size: int = 200):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    virtual_patients = generate_virtual_patients(config.N_PATIENTS, seed=config.PATIENT_GENERATION_SEED)
    pretrain_patients, holdout_patients = split_patients(
        virtual_patients, config.N_PRETRAIN_PATIENTS, config.N_HOLDOUT_PATIENTS
    )

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor()
    ])
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)

    pretrained_path = os.path.join(config.RESULTS_DIR, "pretrained_encoder.pt")
    checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)

    check_patient = holdout_patients[0]
    implant, model = build_model_for_patient(check_patient)
    template = compute_electrode_templates(implant, model)

    random.seed(0)
    ft_idx = random.sample(range(1000, 2000), size)

    encoder_scratch = StimulusEncoder(n_electrodes=len(implant.electrode_names))
    encoder_scratch, losses_scratch = train_encoder(encoder_scratch, mnist, ft_idx, template, n_steps=n_check_steps)

    encoder_pretrained = StimulusEncoder(n_electrodes=len(implant.electrode_names))
    encoder_pretrained.load_state_dict(checkpoint["state_dict"])
    encoder_pretrained, losses_pretrained = train_encoder(encoder_pretrained, mnist, ft_idx, template, n_steps=n_check_steps)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses_scratch, label="scratch", color="tab:orange")
    ax.plot(losses_pretrained, label="pretrained", color="tab:blue")
    ax.axvline(config.MAX_STEPS, linestyle="--", color="gray", label=f"MAX_STEPS ({config.MAX_STEPS})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(config.FIGURES_DIR, "convergence_check.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Saved {fig_path}")

    for name, losses in [("scratch", losses_scratch), ("pretrained", losses_pretrained)]:
        loss_at_cutoff = losses[config.MAX_STEPS - 1]
        loss_final = losses[-1]
        total_drop = losses[0] - loss_final
        remaining_pct = ((loss_at_cutoff - loss_final) / total_drop * 100) if total_drop > 0 else 0
        print(f"{name}: loss@{config.MAX_STEPS}={loss_at_cutoff:.4f}, loss@{n_check_steps}={loss_final:.4f}, "
              f"remaining improvement past cutoff: {remaining_pct:.1f}%")


if __name__ == "__main__":
    main()
