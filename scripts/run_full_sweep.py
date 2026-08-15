"""Full experiment sweep: pretrained vs. from-scratch encoders, evaluated across
20 holdout patients x 6 fine-tuning sizes x 5 random seeds x 2 conditions.

Saves incrementally after every (patient, size) combination, so an interrupted
run can be resumed without repeating completed work. Requires
run_pretraining.py to have been run first.

Usage:
    python -m scripts.run_full_sweep
    python -m scripts.run_full_sweep --resume   # skip already-completed rows
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.patients import generate_virtual_patients, split_patients
from src.simulator import build_model_for_patient, compute_electrode_templates
from src.encoder import StimulusEncoder
from src.training import train_encoder, steps_for_size
from src.evaluate import evaluate_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                         help="Skip (patient, size) combos already present in the partial results file.")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    partial_path = os.path.join(config.RESULTS_DIR, "results_full_sweep_partial.csv")
    progress_log = os.path.join(config.RESULTS_DIR, "progress_log.txt")

    print("Generating virtual patients...")
    virtual_patients = generate_virtual_patients(config.N_PATIENTS, seed=config.PATIENT_GENERATION_SEED)
    _, holdout_patients = split_patients(
        virtual_patients, config.N_PRETRAIN_PATIENTS, config.N_HOLDOUT_PATIENTS
    )

    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)

    print("Loading pretrained encoder...")
    pretrained_path = os.path.join(config.RESULTS_DIR, "pretrained_encoder.pt")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"{pretrained_path} not found. Run `python -m scripts.run_pretraining` first."
        )
    checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)

    # --- Determine already-completed work, if resuming ---
    results = []
    done_combos = set()
    if args.resume and os.path.exists(partial_path):
        existing = pd.read_csv(partial_path)
        results = existing.to_dict("records")
        done_combos = set(zip(existing["patient_id"], existing["finetune_size"]))
        print(f"Resuming: {len(existing)} rows already saved "
              f"({len(done_combos)} (patient, size) combos complete).")

    total_combos = len(holdout_patients) * len(config.FINETUNE_SIZES)
    combo_counter = len(done_combos)
    t_start = time.time()

    for patient in holdout_patients:
        pid = patient["id"]
        implant, model = build_model_for_patient(patient)
        template = compute_electrode_templates(implant, model)

        rng = np.random.default_rng(pid)  # deterministic eval set per patient
        eval_idx = rng.choice(range(500, 1000), size=config.N_EVAL_IMAGES, replace=False).tolist()

        for size in config.FINETUNE_SIZES:
            if (pid, size) in done_combos:
                continue

            n_steps = steps_for_size(size)

            for seed in config.SEED_LIST:
                import random
                random.seed(seed)
                ft_idx = random.sample(range(1000, 2000), size)

                encoder_scratch = StimulusEncoder(n_electrodes=len(implant.electrode_names))
                encoder_scratch, _ = train_encoder(encoder_scratch, mnist, ft_idx, template, n_steps=n_steps)

                encoder_pretrained = StimulusEncoder(n_electrodes=len(implant.electrode_names))
                encoder_pretrained.load_state_dict(checkpoint["state_dict"])
                encoder_pretrained, _ = train_encoder(encoder_pretrained, mnist, ft_idx, template, n_steps=n_steps)

                for condition, enc in [("pretrained", encoder_pretrained), ("scratch", encoder_scratch)]:
                    ssim_mean, ssim_sd = evaluate_encoder(enc, mnist, implant, model, eval_idx)
                    results.append({
                        "patient_id": pid, "condition": condition, "finetune_size": size,
                        "n_steps_used": n_steps, "seed": seed,
                        "ssim_mean": ssim_mean, "ssim_sd": ssim_sd,
                        "n_eval_images": config.N_EVAL_IMAGES,
                    })

            combo_counter += 1
            elapsed = time.time() - t_start
            avg_per_combo = elapsed / max(1, combo_counter - len(done_combos))
            remaining = (total_combos - combo_counter) * avg_per_combo
            print(f"[{combo_counter}/{total_combos}] patient={pid} size={size} steps={n_steps} "
                  f"| elapsed={elapsed/60:.1f}min | est. remaining={remaining/60:.1f}min")

            pd.DataFrame(results).to_csv(partial_path, index=False)

        with open(progress_log, "a") as f:
            f.write(f"{datetime.now()}: patient {pid} done, total elapsed {(time.time()-t_start)/60:.1f} min\n")

    final_df = pd.DataFrame(results)
    final_path = os.path.join(config.RESULTS_DIR, "results_full_sweep_final.csv")
    final_df.to_csv(final_path, index=False)

    expected_rows = len(holdout_patients) * len(config.FINETUNE_SIZES) * len(config.SEED_LIST) * 2
    print(f"\nSweep complete in {(time.time()-t_start)/60:.1f} min")
    print(f"Final shape: {final_df.shape} (expected: {expected_rows} rows)")
    print(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
