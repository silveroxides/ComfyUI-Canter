"""Sinusoidal timestep embedding with MLP projection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _log_spaced_frequencies(
    half: int, max_period: float, *, device: torch.device | None = None
) -> Tensor:
    """Log-spaced frequencies for sinusoidal embedding."""
    return torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(float(half - 1), 1.0)
    )


def sinusoidal_time_embedding(
    t: Tensor,
    dim: int,
    *,
    max_period: float = 10000.0,
    scale: float | None = None,
    freqs: Tensor | None = None,
) -> Tensor:
    """Sinusoidal timestep embedding (DDPM/DiT-style). Always float32."""
    t32 = t.to(torch.float32)
    if scale is not None:
        t32 = t32 * float(scale)
    half = dim // 2
    if freqs is not None:
        freqs = freqs.to(device=t32.device, dtype=torch.float32)
    else:
        freqs = _log_spaced_frequencies(half, max_period, device=t32.device)
    angles = t32[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SinusoidalTimeEmbeddingMLP(nn.Module):
    """Sinusoidal time embedding followed by Linear -> SiLU -> Linear."""

    def __init__(
        self,
        dim: int,
        *,
        freq_dim: int = 256,
        hidden_mult: float = 1.0,
        time_scale: float = 1000.0,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.freq_dim = int(freq_dim)
        self.time_scale = float(time_scale)
        self.max_period = float(max_period)
        hidden_dim = max(int(round(int(dim) * float(hidden_mult))), 1)

        freqs = _log_spaced_frequencies(self.freq_dim // 2, self.max_period)
        self.register_buffer("freqs", freqs, persistent=True)

        self.proj_in = nn.Linear(self.freq_dim, hidden_dim)
        self.act = nn.SiLU()
        self.proj_out = nn.Linear(hidden_dim, self.dim)

    def forward(self, t: Tensor) -> Tensor:
        freqs: Tensor = self.freqs  # type: ignore[assignment]
        emb_freq = sinusoidal_time_embedding(
            t.to(torch.float32),
            self.freq_dim,
            max_period=self.max_period,
            scale=self.time_scale,
            freqs=freqs,
        )
        dtype_in = self.proj_in.weight.dtype
        hidden = self.proj_in(emb_freq.to(dtype_in))
        hidden = self.act(hidden)
        if hidden.dtype != self.proj_out.weight.dtype:
            hidden = hidden.to(self.proj_out.weight.dtype)
        return self.proj_out(hidden)
