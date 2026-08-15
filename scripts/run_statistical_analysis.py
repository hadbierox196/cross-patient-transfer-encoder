"""Run the full statistical analysis pipeline on sweep results.

Produces:
  - finetuning_size_statistical_analysis.csv  (Table 2 in the paper)
  - normality_check.csv                        (justifies Wilcoxon over t-test)
  - leave_one_out_robustness.csv                (robustness check)

Usage:
    python -m scripts.run_statistical_analysis
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.evaluate import compute_statistical_summary, check_normality, leave_one_out_robustness


def main():
    results_path = os.path.join(config.RESULTS_DIR, "results_full_sweep_final.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found. Run run_full_sweep.py first.")

    df = pd.read_csv(results_path)

    print("Computing paired statistical comparison per fine-tuning size...")
    summary_df = compute_statistical_summary(df, config.FINETUNE_SIZES)
    summary_path = os.path.join(config.RESULTS_DIR, "finetuning_size_statistical_analysis.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string())

    print("\nChecking normality of paired differences...")
    normality_df = check_normality(df, config.FINETUNE_SIZES)
    normality_path = os.path.join(config.RESULTS_DIR, "normality_check.csv")
    normality_df.to_csv(normality_path, index=False)
    print(normality_df.to_string())

    print("\nRunning leave-one-patient-out robustness check...")
    loo_df = leave_one_out_robustness(df, config.FINETUNE_SIZES)
    loo_path = os.path.join(config.RESULTS_DIR, "leave_one_out_robustness.csv")
    loo_df.to_csv(loo_path, index=False)
    print(f"All leave-one-out comparisons significant: {loo_df['significant'].all()}")

    print(f"\nSaved: {summary_path}, {normality_path}, {loo_path}")


if __name__ == "__main__":
    main()
