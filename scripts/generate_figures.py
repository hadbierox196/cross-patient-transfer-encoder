"""Generate Figures 1-3 for the manuscript from sweep results.

Usage:
    python -m scripts.generate_figures
"""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.datasets import MNIST

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.patients import generate_virtual_patients, split_patients
from src.simulator import build_model_for_patient, compute_electrode_templates, render_percept
from src.encoder import StimulusEncoder
from src.training import train_encoder, steps_for_size


def fig1_data_efficiency_curve(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, color in [("pretrained", "tab:blue"), ("scratch", "tab:orange")]:
        sub = df[df["condition"] == condition]
        grouped = sub.groupby("finetune_size")["ssim_mean"]
        means, sds = grouped.mean(), grouped.std()
        ax.plot(means.index, means.values, marker="o", linewidth=2, label=condition, color=color)
        ax.fill_between(means.index, means.values - sds.values, means.values + sds.values, alpha=0.15, color=color)
    ax.set_xlabel("Fine-tuning dataset size (samples)", fontsize=12)
    ax.set_ylabel("SSIM (mean ± SD across patients)", fontsize=12)
    ax.set_title("Data-efficiency curve: pretrained vs. from-scratch encoders", fontsize=13)
    ax.set_xscale("log")
    ax.set_xticks(config.FINETUNE_SIZES)
    ax.set_xticklabels(config.FINETUNE_SIZES)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "fig1_data_efficiency_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def fig2_relative_improvement(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    pct = summary_df.set_index("finetune_size")["pct_improvement_vs_scratch"]
    bars = ax.bar([str(s) for s in pct.index], pct.values, color="tab:blue", alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%", ha="center", fontsize=10)
    ax.set_xlabel("Fine-tuning dataset size (samples)", fontsize=12)
    ax.set_ylabel("Relative SSIM improvement over from-scratch (%)", fontsize=12)
    ax.set_title("Pretraining advantage diminishes as calibration data increases", fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "fig2_relative_improvement.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def fig3_qualitative_examples(example_size: int = 50, example_image_idx: int = 505, n_patients: int = 3):
    virtual_patients = generate_virtual_patients(config.N_PATIENTS, seed=config.PATIENT_GENERATION_SEED)
    _, holdout_patients = split_patients(virtual_patients, config.N_PRETRAIN_PATIENTS, config.N_HOLDOUT_PATIENTS)
    example_patients = holdout_patients[:n_patients]

    transform = transforms.Compose([transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor()])
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)

    checkpoint = torch.load(os.path.join(config.RESULTS_DIR, "pretrained_encoder.pt"), map_location=config.DEVICE)

    fig, axes = plt.subplots(len(example_patients), 3, figsize=(9, 3 * len(example_patients)))
    n_steps = steps_for_size(example_size)

    for row, patient in enumerate(example_patients):
        implant, model = build_model_for_patient(patient)
        template = compute_electrode_templates(implant, model)

        random.seed(0)
        ft_idx = random.sample(range(1000, 2000), example_size)

        enc_scratch = StimulusEncoder(n_electrodes=len(implant.electrode_names))
        enc_scratch, _ = train_encoder(enc_scratch, mnist, ft_idx, template, n_steps=n_steps)

        enc_pre = StimulusEncoder(n_electrodes=len(implant.electrode_names))
        enc_pre.load_state_dict(checkpoint["state_dict"])
        enc_pre, _ = train_encoder(enc_pre, mnist, ft_idx, template, n_steps=n_steps)

        target_img = mnist[example_image_idx][0].squeeze(0).numpy()
        with torch.no_grad():
            img_tensor = mnist[example_image_idx][0].unsqueeze(0).to(config.DEVICE)

            amp_scratch = enc_scratch(img_tensor).cpu().numpy()[0]
            stim_scratch = {name: amp_scratch[j] for j, name in enumerate(implant.electrode_names)}
            percept_scratch = render_percept(implant, model, stim_scratch)
            ssim_scratch = ssim(target_img, percept_scratch, data_range=percept_scratch.max() - percept_scratch.min() + 1e-8)

            amp_pre = enc_pre(img_tensor).cpu().numpy()[0]
            stim_pre = {name: amp_pre[j] for j, name in enumerate(implant.electrode_names)}
            percept_pre = render_percept(implant, model, stim_pre)
            ssim_pre = ssim(target_img, percept_pre, data_range=percept_pre.max() - percept_pre.min() + 1e-8)

        axes[row, 0].set_ylabel(f"Patient {patient['id']}", fontsize=11)
        axes[row, 0].imshow(target_img, cmap="gray")
        axes[row, 1].imshow(percept_scratch, cmap="gray")
        axes[row, 2].imshow(percept_pre, cmap="gray")
        axes[row, 1].set_title(f"SSIM={ssim_scratch:.3f}", fontsize=9)
        axes[row, 2].set_title(f"SSIM={ssim_pre:.3f}", fontsize=9)
        if row == 0:
            axes[row, 0].set_title("Target", fontsize=11)
            axes[row, 1].set_title(f"From-scratch\nSSIM={ssim_scratch:.3f}", fontsize=10)
            axes[row, 2].set_title(f"Pretrained\nSSIM={ssim_pre:.3f}", fontsize=10)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    plt.suptitle(f"Qualitative examples at {example_size}-sample fine-tuning", fontsize=13)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "fig3_qualitative_examples.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "results_full_sweep_final.csv"))
    summary_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "finetuning_size_statistical_analysis.csv"))

    fig1_data_efficiency_curve(df)
    fig2_relative_improvement(summary_df)
    fig3_qualitative_examples()


if __name__ == "__main__":
    main()
