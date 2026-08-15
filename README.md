# Cross-Patient Transfer Learning for Stimulus Encoders

Reproducibility repository for the manuscript *"Cross-Patient Transfer Learning
for Stimulus Encoders"*. This repo contains all code used to generate virtual
patients, pretrain and fine-tune stimulus encoders, run the statistical
analysis, and produce the figures reported in the paper.

## Overview

We use the [pulse2percept](https://github.com/pulse2percept/pulse2percept)
simulator to generate 100 virtual patients with heterogeneous retinal and
implant parameters, pretrain a convolutional stimulus encoder on one
representative patient, and evaluate how well the pretrained encoder
transfers to 20 held-out patients across a range of per-patient calibration
(fine-tuning) dataset sizes, compared against training from scratch.

See `PREREGISTRATION.md` for the frozen analysis plan (including one
documented deviation, discovered via the convergence diagnostic described
below).

## Repository structure

```
├── src/                        # Core library code
│   ├── config.py                # Frozen experiment parameters
│   ├── patients.py              # Virtual patient generation
│   ├── simulator.py             # pulse2percept wrapper
│   ├── encoder.py                # StimulusEncoder architecture
│   ├── training.py               # NNLS-based training targets + loop
│   └── evaluate.py               # SSIM scoring + statistical analysis
├── scripts/                    # Runnable pipeline stages
│   ├── run_pretraining.py        # Stage 1: pretrain encoder
│   ├── run_full_sweep.py          # Stage 2: main experiment (checkpointed)
│   ├── run_convergence_check.py   # Diagnostic: verify training budget is sufficient
│   ├── run_statistical_analysis.py# Stage 3: stats (Wilcoxon, Holm, bootstrap CI)
│   └── generate_figures.py        # Stage 4: Figures 1-3
├── notebooks/
│   └── demo.ipynb                 # Minimal exploratory notebook (not used for results)
├── results/                    # Output CSVs (generated; not tracked in git except .gitkeep)
├── figures/                    # Output figures (generated; not tracked in git except .gitkeep)
└── PREREGISTRATION.md          # Frozen analysis plan + documented deviation
```

## Setup

```bash
git clone <this-repo-url>
cd cross-patient-transfer-encoder
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+, on both CPU and CUDA (T4) runtimes in Google Colab.
pulse2percept version used for the reported results: **[PIN EXACT VERSION HERE
— run `python -c "import pulse2percept; print(pulse2percept.__version__)"`
and record it before submission]**.

## Reproducing the results

Run in order:

```bash
# 1. Pretrain the encoder (~few seconds on GPU, ~tens of seconds on CPU)
python -m scripts.run_pretraining

# 2. Run the full sweep: 20 patients x 6 sizes x 5 seeds x 2 conditions
#    (~1-4 hours depending on hardware; safe to interrupt and resume)
python -m scripts.run_full_sweep
# if interrupted, resume with:
python -m scripts.run_full_sweep --resume

# 3. Verify the training step budget was sufficient (optional but recommended)
python -m scripts.run_convergence_check

# 4. Compute statistics (Table 1, Table 2 in the paper)
python -m scripts.run_statistical_analysis

# 5. Generate figures (Fig. 1-3 in the paper)
python -m scripts.generate_figures
```

All intermediate and final results are written to `results/`, and figures to
`figures/`.

## Reproducibility notes

- **Random seeds:** patient generation is seeded (`config.PATIENT_GENERATION_SEED`);
  each fine-tuning run uses one of 5 fixed seeds (`config.SEED_LIST`), averaged
  in the final analysis.
- **Non-differentiable simulator:** pulse2percept's percept-rendering function
  is not differentiable, so training targets are derived via non-negative
  least-squares (NNLS) regression against per-patient electrode templates,
  rather than by backpropagating an image-quality loss through the simulator
  directly. See `src/training.py` docstring for details and the rationale.
- **Training budget:** fine-tuning step count scales with dataset size
  (`steps_for_size` in `src/training.py`), rather than using a fixed step
  count. This was a correction made after a post-hoc convergence diagnostic
  revealed that a fixed 30-step budget under-trained the from-scratch
  condition at larger fine-tuning sizes, and that a subsequent scaled budget
  capped at 300 steps under-trained the pretrained condition. The final
  budget (capped at 600 steps) was verified via `run_convergence_check.py`.
  This is a disclosed deviation from the original preregistration — see
  `PREREGISTRATION.md`.
- **Incremental checkpointing:** `run_full_sweep.py` saves results after every
  (patient, fine-tuning size) combination, so long runs can be safely
  interrupted and resumed with `--resume` without repeating completed work.


This work depends on pulse2percept; please also cite:

> Beyeler, M., Boynton, G. M., Fine, I., & Rokem, A. (2017). pulse2percept: A
> Python-based simulation framework for bionic vision. *Proceedings of the
> 16th Python in Science Conference*, 81–88.

## License

See LICENSE

## Data availability

This study uses entirely synthetic data (simulated virtual patients) and the
publicly available MNIST dataset (downloaded automatically via `torchvision`).
No human subjects or protected health information were used.
