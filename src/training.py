"""Encoder training via NNLS-derived per-patient targets.

pulse2percept's percept-rendering function is not differentiable, so an
image-quality loss cannot be backpropagated through the simulator directly.
Instead, for each patient we precompute a linear electrode-template matrix
(see simulator.compute_electrode_templates) and, for each training image,
solve a non-negative least-squares problem for the amplitude vector that
best reconstructs that image through the patient's real electrode templates.
The encoder is then trained with ordinary MSE regression toward these
NNLS-derived targets.

An earlier version of this pipeline instead regressed encoder output against
raw flattened pixel values. Because MNIST images are mostly black background,
this caused the encoder to collapse to near-zero output (trivially minimizing
the loss without learning anything about image encoding). This module
implements the corrected, patient-physics-aware target computation.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import nnls

from . import config


def get_image_batch(mnist_dataset, indices: list[int]) -> torch.Tensor:
    return torch.stack([mnist_dataset[i][0] for i in indices])


def compute_target_amplitudes(template_matrix: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    """Solve NNLS for the amplitude vector that best reconstructs target_image
    via the patient's electrode templates. Amplitudes are normalized to [0, 1].
    """
    amps, _ = nnls(template_matrix, target_image.flatten())
    if amps.max() > 0:
        amps = amps / amps.max()
    return amps


def train_encoder(encoder: nn.Module, mnist_dataset, image_indices: list[int],
                   template_matrix: np.ndarray, n_steps: int = 50,
                   lr: float = config.LEARNING_RATE, batch_size: int = config.BATCH_SIZE,
                   device: torch.device = config.DEVICE):
    """Train (or fine-tune) an encoder against NNLS-derived targets.

    Returns:
        (trained encoder, list of per-step training losses)
    """
    encoder.to(device)
    opt = optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    target_cache = {
        idx: compute_target_amplitudes(template_matrix, mnist_dataset[idx][0].squeeze(0).numpy())
        for idx in image_indices
    }

    losses = []
    for _ in range(n_steps):
        idx_batch = random.sample(image_indices, min(batch_size, len(image_indices)))
        imgs = get_image_batch(mnist_dataset, idx_batch).to(device)
        amps_pred = encoder(imgs)
        amps_target = torch.tensor(
            np.stack([target_cache[i] for i in idx_batch]), dtype=torch.float32
        ).to(device)
        loss = loss_fn(amps_pred, amps_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return encoder, losses


def steps_for_size(finetune_size: int) -> int:
    """Map a fine-tuning dataset size to a training step budget.

    Steps are scaled proportionally to dataset size (STEPS_PER_SAMPLE), clipped
    to [MIN_STEPS, MAX_STEPS]. See config.py docstring for why this replaced an
    earlier fixed 30-step budget (confirmed via post-hoc convergence diagnostic
    in scripts/run_convergence_check.py).
    """
    return int(np.clip(finetune_size * config.STEPS_PER_SAMPLE, config.MIN_STEPS, config.MAX_STEPS))
