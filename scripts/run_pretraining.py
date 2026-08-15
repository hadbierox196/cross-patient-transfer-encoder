"""Pretrain a stimulus encoder on a single representative patient's electrode
geometry, using a pool of MNIST images.

The pretrained encoder's state_dict is saved to results/pretrained_encoder.pt
for reuse by run_full_sweep.py.

See Discussion (paper) for the justification of single-patient pretraining
templates: the pretraining phase teaches generalizable image-to-stimulation
mapping strategies, while patient-specific adaptation is left to fine-tuning.

Usage:
    python -m scripts.run_pretraining
"""

import os
import sys
import time

import torch
from torchvision import transforms
from torchvision.datasets import MNIST

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.patients import generate_virtual_patients, split_patients
from src.simulator import build_model_for_patient, compute_electrode_templates
from src.encoder import StimulusEncoder
from src.training import train_encoder


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("Generating virtual patients...")
    virtual_patients = generate_virtual_patients(config.N_PATIENTS, seed=config.PATIENT_GENERATION_SEED)
    pretrain_patients, holdout_patients = split_patients(
        virtual_patients, config.N_PRETRAIN_PATIENTS, config.N_HOLDOUT_PATIENTS
    )

    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)

    print("Building simulator for pretraining-pool patient 0...")
    implant0, model0 = build_model_for_patient(pretrain_patients[0])
    template0 = compute_electrode_templates(implant0, model0)

    print(f"Pretraining encoder ({config.PRETRAIN_STEPS} steps, "
          f"{config.PRETRAIN_POOL_SIZE} image pool)...")
    encoder = StimulusEncoder(n_electrodes=len(implant0.electrode_names))
    pretrain_idx = list(range(config.PRETRAIN_POOL_SIZE))
    encoder, losses = train_encoder(
        encoder, mnist, pretrain_idx, template0, n_steps=config.PRETRAIN_STEPS
    )

    out_path = os.path.join(config.RESULTS_DIR, "pretrained_encoder.pt")
    torch.save({
        "state_dict": encoder.state_dict(),
        "n_electrodes": len(implant0.electrode_names),
        "final_loss": losses[-1],
    }, out_path)

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Final pretraining loss: {losses[-1]:.6f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
