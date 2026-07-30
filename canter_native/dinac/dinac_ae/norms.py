"""Channel-wise RMSNorm for NCHW tensors."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelWiseRMSNorm(nn.Module):
    """Channel-wise RMSNorm with float32 reduction for numerical stability.

    Normalizes across channels per spatial position. Supports optional
    per-channel affine weight and bias.
    """

    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.channels: int = int(channels)
        self._eps: float = float(eps)
        if affine:
            self.weight = nn.Parameter(torch.ones(self.channels))
            self.bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() < 2:
            return x
        # Float32 accumulation for stability
        ms = torch.mean(torch.square(x), dim=1, keepdim=True, dtype=torch.float32)
        inv_rms = torch.rsqrt(ms + self._eps)
        y = x * inv_rms.to(dtype=x.dtype)
        if self.weight is not None:
            shape = (1, -1) + (1,) * (x.dim() - 2)
            y = y * self.weight.view(shape).to(dtype=x.dtype)
            if self.bias is not None:
                y = y + self.bias.view(shape).to(dtype=x.dtype)
        return y
