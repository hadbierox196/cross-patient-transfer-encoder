"""Evaluation: SSIM scoring of trained encoders and statistical comparison."""

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, ttest_rel, shapiro, normaltest
from skimage.metrics import structural_similarity as ssim
from statsmodels.stats.multitest import multipletests

from . import config
from .simulator import render_percept


def evaluate_encoder(encoder, mnist_dataset, implant, model, eval_indices: list[int],
                      device: torch.device = config.DEVICE) -> tuple[float, float]:
    """Compute mean and SD of SSIM across a fixed evaluation image set for one encoder."""
    encoder.eval()
    scores = []
    with torch.no_grad():
        for i in eval_indices:
            img = mnist_dataset[i][0].unsqueeze(0).to(device)
            amp = encoder(img).cpu().numpy()[0]
            stim_dict = {name: amp[j] for j, name in enumerate(implant.electrode_names)}
            percept = render_percept(implant, model, stim_dict)
            target = mnist_dataset[i][0].squeeze(0).numpy()
            score = ssim(target, percept, data_range=percept.max() - percept.min() + 1e-8)
            scores.append(score)
    return float(np.mean(scores)), float(np.std(scores))


def compute_statistical_summary(results_df: pd.DataFrame, finetune_sizes: list[int]) -> pd.DataFrame:
    """Paired comparison (pretrained vs. scratch) at each fine-tuning size.

    Primary test: Wilcoxon signed-rank (paired, non-parametric — chosen because
    Shapiro-Wilk / D'Agostino-Pearson reject normality of paired differences;
    see PREREGISTRATION.md). Paired t-test reported as a supplementary check.
    Multiple-comparisons correction: Holm-Bonferroni across all tested sizes.
    """
    rows = []
    for size in finetune_sizes:
        sub = results_df[results_df["finetune_size"] == size]
        pivot = sub.groupby(["patient_id", "condition"])["ssim_mean"].mean().unstack()
        a = pivot["pretrained"].values
        b = pivot["scratch"].values
        diff = a - b

        t_stat, p_ttest = ttest_rel(a, b)
        w_stat, p_wilcoxon = wilcoxon(a, b)
        cohens_dz = diff.mean() / diff.std(ddof=1)

        rng = np.random.default_rng(0)
        boot_diffs = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(2000)]
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

        rows.append({
            "finetune_size": size, "n_patients": len(a),
            "pretrained_mean_ssim": a.mean(), "scratch_mean_ssim": b.mean(),
            "mean_difference": diff.mean(),
            "ci95_low": ci_low, "ci95_high": ci_high,
            "pct_improvement_vs_scratch": (diff.mean() / b.mean()) * 100,
            "paired_t": t_stat, "p_ttest": p_ttest,
            "wilcoxon_W": w_stat, "p_wilcoxon": p_wilcoxon,
            "cohens_dz": cohens_dz,
            "n_pretrained_better": int((diff > 0).sum()),
            "n_scratch_better": int((diff < 0).sum()),
            "n_equal": int((diff == 0).sum()),
        })

    summary_df = pd.DataFrame(rows)
    _, p_t_holm, _, _ = multipletests(summary_df["p_ttest"], method="holm")
    _, p_w_holm, _, _ = multipletests(summary_df["p_wilcoxon"], method="holm")
    summary_df["p_ttest_holm"] = p_t_holm
    summary_df["p_wilcoxon_holm"] = p_w_holm
    summary_df["significant_ttest_holm"] = p_t_holm < 0.05
    summary_df["significant_wilcoxon_holm"] = p_w_holm < 0.05
    return summary_df


def check_normality(results_df: pd.DataFrame, finetune_sizes: list[int]) -> pd.DataFrame:
    """Shapiro-Wilk and D'Agostino-Pearson normality tests on paired differences,
    used to justify the choice of Wilcoxon (non-parametric) as the primary test.
    """
    rows = []
    for size in finetune_sizes:
        sub = results_df[results_df["finetune_size"] == size]
        pivot = sub.groupby(["patient_id", "condition"])["ssim_mean"].mean().unstack()
        diff = (pivot["pretrained"] - pivot["scratch"]).values
        shapiro_w, shapiro_p = shapiro(diff)
        dagostino_k2, dagostino_p = normaltest(diff)
        rows.append({
            "finetune_size": size,
            "shapiro_W": shapiro_w, "shapiro_p": shapiro_p,
            "dagostino_K2": dagostino_k2, "dagostino_p": dagostino_p,
        })
    return pd.DataFrame(rows)


def leave_one_out_robustness(results_df: pd.DataFrame, finetune_sizes: list[int]) -> pd.DataFrame:
    """Recompute the primary Wilcoxon comparison with each patient iteratively excluded,
    to confirm the effect is not driven by any single patient.
    """
    rows = []
    for size in finetune_sizes:
        sub = results_df[results_df["finetune_size"] == size]
        pivot = sub.groupby(["patient_id", "condition"])["ssim_mean"].mean().unstack()
        patient_ids = pivot.index.tolist()
        for excluded in patient_ids:
            remaining = pivot.drop(index=excluded)
            a, b = remaining["pretrained"].values, remaining["scratch"].values
            _, p = wilcoxon(a, b)
            rows.append({
                "finetune_size": size, "excluded_patient": excluded,
                "p_wilcoxon": p, "significant": p < 0.05,
                "mean_difference": (a - b).mean(),
            })
    return pd.DataFrame(rows)
