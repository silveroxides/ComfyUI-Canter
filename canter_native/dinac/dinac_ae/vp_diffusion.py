"""VP diffusion math: logSNR schedules, alpha/sigma computation, noise construction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def alpha_sigma_from_logsnr(lmb: Tensor) -> tuple[Tensor, Tensor]:
    """Compute (alpha, sigma) from logSNR in float32.

    VP constraint: alpha^2 + sigma^2 = 1.
    """
    lmb32 = lmb.to(dtype=torch.float32)
    alpha = torch.exp(0.5 * F.logsigmoid(lmb32))
    sigma = torch.exp(0.5 * F.logsigmoid(-lmb32))
    return alpha, sigma


def broadcast_time_like(coeff: Tensor, x: Tensor) -> Tensor:
    """Broadcast [B] coefficient to match x for per-sample scaling."""
    view_shape = (int(x.shape[0]),) + (1,) * (x.dim() - 1)
    return coeff.view(view_shape)


def _cosine_interpolated_params(
    logsnr_min: float, logsnr_max: float
) -> tuple[float, float]:
    """Compute (a, b) for cosine-interpolated logSNR schedule.

    logsnr(t) = -2 * log(tan(a*t + b))
    logsnr(0) = logsnr_max, logsnr(1) = logsnr_min
    """
    b = math.atan(math.exp(-0.5 * logsnr_max))
    a = math.atan(math.exp(-0.5 * logsnr_min)) - b
    return a, b


def cosine_interpolated_logsnr_from_t(
    t: Tensor, *, logsnr_min: float, logsnr_max: float
) -> Tensor:
    """Map t in [0,1] to logSNR via cosine-interpolated schedule. Always float32."""
    a, b = _cosine_interpolated_params(logsnr_min, logsnr_max)
    t32 = t.to(dtype=torch.float32)
    a_t = torch.tensor(a, device=t32.device, dtype=torch.float32)
    b_t = torch.tensor(b, device=t32.device, dtype=torch.float32)
    u = a_t * t32 + b_t
    return -2.0 * torch.log(torch.tan(u))


def shifted_cosine_interpolated_logsnr_from_t(
    t: Tensor,
    *,
    logsnr_min: float,
    logsnr_max: float,
    log_change_high: float = 0.0,
    log_change_low: float = 0.0,
) -> Tensor:
    """SiD2 "shifted cosine" schedule: logSNR with resolution-dependent shifts.

    lambda(t) = (1-t) * (base(t) + log_change_high) + t * (base(t) + log_change_low)
    """
    base = cosine_interpolated_logsnr_from_t(
        t, logsnr_min=logsnr_min, logsnr_max=logsnr_max
    )
    t32 = t.to(dtype=torch.float32)
    high = base + float(log_change_high)
    low = base + float(log_change_low)
    return (1.0 - t32) * high + t32 * low


def get_schedule(schedule_type: str, num_steps: int) -> Tensor:
    """Generate a descending t-schedule in [0, 1] for VP diffusion sampling.

    ``num_steps`` is the number of function evaluations (NFE = decoder forward
    passes).  Internally the schedule has ``num_steps + 1`` time points
    (including both endpoints).

    Args:
        schedule_type: "linear" or "cosine".
        num_steps: Number of decoder forward passes (NFE), >= 1.

    Returns:
        Descending 1D tensor with ``num_steps + 1`` elements from ~1.0 to ~0.0.
    """
    if int(num_steps) < 1:
        raise ValueError("num_steps must be at least 1")
    n = int(num_steps) + 1
    match schedule_type:
        case "linear":
            base = torch.linspace(0.0, 1.0, n)
        case "cosine":
            i = torch.arange(n, dtype=torch.float32)
            base = 0.5 * (1.0 - torch.cos(math.pi * (i / (n - 1))))
        case _ as unreachable:
            raise ValueError(
                f"Unsupported schedule type: {unreachable!r}. "
                "Use 'linear' or 'cosine'."
            )
    # Descending: high t (noisy) -> low t (clean)
    return torch.flip(base, dims=[0])


def make_initial_state(
    *,
    noise: Tensor,
    t_start: Tensor,
    logsnr_min: float,
    logsnr_max: float,
    log_change_high: float = 0.0,
    log_change_low: float = 0.0,
) -> Tensor:
    """Construct VP initial state x_t0 = sigma_start * noise (since x0=0).

    All math in float32.
    """
    batch = int(noise.shape[0])
    lmb_start = shifted_cosine_interpolated_logsnr_from_t(
        t_start.expand(batch).to(dtype=torch.float32),
        logsnr_min=logsnr_min,
        logsnr_max=logsnr_max,
        log_change_high=log_change_high,
        log_change_low=log_change_low,
    )
    _alpha_start, sigma_start = alpha_sigma_from_logsnr(lmb_start)
    sigma_view = broadcast_time_like(sigma_start, noise)
    return sigma_view * noise.to(dtype=torch.float32)


def sample_noise(
    shape: tuple[int, ...],
    *,
    noise_std: float = 1.0,
    seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample Gaussian noise with optional seeding. CPU-seeded for reproducibility."""
    if seed is None:
        noise = torch.randn(
            shape, device=device or torch.device("cpu"), dtype=torch.float32
        )
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        noise = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
    noise = noise.mul(float(noise_std))
    target_device = device if device is not None else torch.device("cpu")
    return noise.to(device=target_device, dtype=dtype)
