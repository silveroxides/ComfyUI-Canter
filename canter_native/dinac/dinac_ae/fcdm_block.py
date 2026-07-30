"""FCDM block: ConvNeXt-style conv block with GRN and scale+gate AdaLN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .norms import ChannelWiseRMSNorm


class GRN(nn.Module):
    """Global Response Normalization for NCHW tensors."""

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = float(eps)
        c = int(channels)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1), dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.linalg.vector_norm(x, ord=2, dim=(2, 3), keepdim=True)
        g_fp32 = g.to(dtype=torch.float32)
        n = (g_fp32 / (g_fp32.mean(dim=1, keepdim=True) + self.eps)).to(dtype=x.dtype)
        gamma = self.gamma.to(device=x.device, dtype=x.dtype)
        beta = self.beta.to(device=x.device, dtype=x.dtype)
        return gamma * (x * n) + beta + x


class FCDMBlock(nn.Module):
    """ConvNeXt-style block with scale+gate AdaLN and GRN.

    Two modes:
    - Unconditioned (encoder): uses learned layer-scale for near-identity init.
    - External AdaLN (decoder): receives packed [B, 2*C] modulation (scale, gate).
      The gate is applied raw (no tanh).
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: float,
        *,
        depthwise_kernel_size: int = 7,
        use_external_adaln: bool = False,
        norm_eps: float = 1e-6,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.channels: int = int(channels)
        self.mlp_ratio: float = float(mlp_ratio)

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=depthwise_kernel_size,
            padding=depthwise_kernel_size // 2,
            stride=1,
            groups=channels,
            bias=True,
        )
        self.norm = ChannelWiseRMSNorm(channels, eps=float(norm_eps), affine=False)
        hidden = max(int(float(channels) * float(mlp_ratio)), 1)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.grn = GRN(hidden, eps=1e-6)
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        if not use_external_adaln:
            self.layer_scale = nn.Parameter(
                torch.full((channels,), float(layer_scale_init))
            )
        else:
            self.register_parameter("layer_scale", None)

    def forward(self, x: Tensor, *, adaln_m: Tensor | None = None) -> Tensor:
        b, c, _, _ = x.shape

        if adaln_m is not None:
            m = adaln_m.to(device=x.device, dtype=x.dtype)
            scale, gate = m.chunk(2, dim=-1)
        else:
            scale = gate = None

        h = self.dwconv(x)
        h = self.norm(h)

        if scale is not None:
            h = h * (1.0 + scale.view(b, c, 1, 1))

        h = self.pwconv1(h)
        h = F.gelu(h)
        h = self.grn(h)
        h = self.pwconv2(h)

        if gate is not None:
            gate_view = gate.view(b, c, 1, 1)
        else:
            gate_view = self.layer_scale.view(1, c, 1, 1).to(  # type: ignore[union-attr]
                device=h.device, dtype=h.dtype
            )

        return x + gate_view * h
