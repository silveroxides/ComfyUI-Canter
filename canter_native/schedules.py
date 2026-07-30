from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats

LOGSNR_SOLVER_START_EPS = 1.0e-4


def apply_log_snr_shift(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
    if not math.isfinite(shift):
        raise ValueError("log-SNR shift must be finite")
    if abs(shift) < 1.0e-12:
        return timesteps
    zero, one = timesteps == 0, timesteps == 1
    clipped = timesteps.clamp(1.0e-12, 1.0 - torch.finfo(timesteps.dtype).eps)
    shifted = torch.sigmoid(torch.logit(clipped) - 0.5 * shift)
    shifted = torch.where(zero, torch.zeros_like(shifted), shifted)
    return torch.where(one, torch.ones_like(shifted), shifted)


def build_schedule(kind: str, steps: int, shift: float = 0.0, finite_start=False) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    if kind == "linear":
        ascending = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    elif kind == "beta":
        probabilities = torch.linspace(0, 1, steps + 1, dtype=torch.float64).numpy()
        ascending = torch.from_numpy(
            np.asarray(stats.beta.ppf(probabilities, 0.6, 0.6), dtype=np.float32)
        )
    else:
        raise ValueError(f"unsupported Canter schedule: {kind}")
    if finite_start:
        candidate = torch.tensor(1.0 - LOGSNR_SOLVER_START_EPS)
        if ascending[-2] >= candidate:
            candidate = torch.nextafter(ascending[-2], torch.tensor(1.0))
        ascending[-1] = candidate
    return apply_log_snr_shift(ascending, shift).flip(0).contiguous()
