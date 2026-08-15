# Preregistration

## Cross-Patient Transfer Learning for Stimulus Encoders

This document records the frozen analysis plan committed to before results
were observed (also registered on OSF: **[insert OSF registration URL]**),
along with a disclosed deviation discovered during data collection.

### Research question / hypothesis

Can pretraining a stimulus encoder across a population of virtual patients
reduce the amount of per-patient calibration data needed to reach reference-
level image quality, compared to training an encoder from scratch for each
patient? We hypothesized that pretrained encoders would outperform
from-scratch encoders at a given calibration data size, with the advantage
being largest when calibration data is scarce.

### Frozen parameters

| Parameter | Value |
|---|---|
| Simulator | pulse2percept, AxonMapModel + ArgusII |
| N virtual patients | 100 (80 pretrain pool / 20 holdout) |
| Fine-tuning sizes tested | 5, 10, 20, 50, 100, 200 |
| Primary metric | SSIM |
| Statistical test | Wilcoxon signed-rank (paired), chosen in advance pending normality check |
| Multiple comparisons correction | Holm-Bonferroni across 6 sizes |
| Random seeds | 5 (0-4), averaged |
| Evaluation images per patient | 20, fixed |

### Deviation from preregistration

**Original plan:** define a "% calibration data reduction" headline metric,
based on the fine-tuning size at which the pretrained encoder's SSIM first
matched the from-scratch encoder's performance at a fixed reference size
(200 samples), assumed to represent a performance plateau.

**What happened:** from-scratch performance did not plateau within the
tested range (5-200 samples) — it continued improving substantially even at
200 samples (mean SSIM 0.126 to 0.261). As a result, the pretrained encoder's
performance never fully closed the gap to the from-scratch condition's
200-sample performance within the tested range, making the originally
planned "% reduction to reach reference" metric unusable as defined.

**Resolution:** the primary result is instead reported as the relative SSIM
improvement of pretrained over from-scratch encoders at each tested
fine-tuning size (a full curve, rather than a single matched point). This was
decided after observing that the reference-based metric returned 0% despite
a large, statistically significant, and consistent advantage for pretrained
encoders at every tested size — indicating the original metric did not
capture the actual effect present in the data. All statistical tests
(Wilcoxon, effect sizes, confidence intervals, Holm correction) specified in
the original plan were retained and applied to this reframed primary
metric.

A related deviation: the training step budget was originally intended to be
fixed (30 steps per fine-tuning run, matching an early implementation). A
post-hoc convergence diagnostic (`scripts/run_convergence_check.py`) revealed
this fixed budget under-trained the from-scratch condition at larger
fine-tuning sizes, and a naive fix (fixed 30 steps regardless of size) risked
confounding data efficiency with training-budget adequacy. The budget was
therefore changed to scale with fine-tuning size (`steps_for_size` in
`src/training.py`, capped at 600 steps), with the cap itself verified via the
same diagnostic script. This change was made before the final reported sweep
was run, and is disclosed here and in the manuscript's Methods section.

### What was NOT changed

- The statistical test selection process (checking normality via Shapiro-Wilk
  / D'Agostino-Pearson before choosing Wilcoxon over a paired t-test) was
  followed exactly as planned.
- Sample size (N=20 holdout patients), number of seeds (5), and fine-tuning
  sizes tested (5-200) were unchanged.
- Multiple comparisons correction (Holm-Bonferroni) was applied exactly as
  planned.
