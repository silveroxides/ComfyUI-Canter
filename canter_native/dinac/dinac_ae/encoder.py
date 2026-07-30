"""Encoder matching the exported mixed DitBlock/FCDM architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..dit.axial_rope2d import (
    AxialRoPE2D,
    AxialRoPE2DConfig,
    AxialRoPE2DCoordMode,
    AxialRoPE2DDimLayout,
    AxialRoPE2DNormalizeCoords,
)
from ..dit.blocks import DitBlock
from ..dit.body_config import DiTConditioning
from ..dit.mlp_types import MLPType
from ..dit.position_encoding import DiTPositionEncoding

from .straight_through_encoder import Patchify

_ENCODER_HEAD_DIM = 64


def _resolve_encoder_mlp_type(name: str) -> MLPType:
    """Return the encoder DiT MLP enum for the serialized config value."""

    match str(name):
        case "gelu":
            return MLPType.GELU
        case "silu":
            return MLPType.SILU
        case "relu":
            return MLPType.RELU
        case _ as unreachable:
            raise ValueError(
                f"Unsupported encoder_mlp_type for DinacAE export: {unreachable!r}"
            )


@dataclass(frozen=True)
class EncoderPosterior:
    """VP-parameterized diagonal Gaussian posterior."""

    mean: Tensor
    logsnr: Tensor

    @property
    def alpha(self) -> Tensor:
        """Return the VP signal coefficient."""

        logsnr_fp32 = self.logsnr.to(torch.float32)
        return torch.exp(0.5 * F.logsigmoid(logsnr_fp32))

    @property
    def sigma(self) -> Tensor:
        """Return the VP noise coefficient."""

        logsnr_fp32 = self.logsnr.to(torch.float32)
        return torch.exp(0.5 * F.logsigmoid(-logsnr_fp32))

    def mode(self) -> Tensor:
        """Return the posterior mode in token space."""

        return self.alpha * self.mean.to(torch.float32)

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Sample from the posterior."""

        mean_fp32 = self.mean.to(torch.float32)
        eps = torch.randn(
            mean_fp32.shape,
            device=mean_fp32.device,
            dtype=torch.float32,
            generator=generator,
        )
        return self.alpha * mean_fp32 + self.sigma * eps


class Encoder(nn.Module):
    """Residual-patchify plus DitBlock encoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        patch_size: int,
        model_dim: int,
        depth: int,
        bottleneck_dim: int,
        mlp_ratio: float,
        mlp_type: str,
        bottleneck_posterior_kind: str,
        bottleneck_norm_mode: str,
    ) -> None:
        super().__init__()
        if int(model_dim) % int(_ENCODER_HEAD_DIM) != 0:
            raise ValueError("model_dim must be divisible by encoder head dim")
        self.bottleneck_dim: int = int(bottleneck_dim)
        self.bottleneck_posterior_kind: str = str(bottleneck_posterior_kind)
        self.bottleneck_norm_mode: str = str(bottleneck_norm_mode)
        if self.bottleneck_norm_mode != "disabled":
            raise ValueError("DINAC-AE export requires disabled bottleneck norm")
        self.patchify = Patchify(
            in_channels,
            patch_size,
            model_dim,
        )
        self.blocks = nn.ModuleList(
            [
                DitBlock(
                    d_model=int(model_dim),
                    n_heads=int(model_dim) // int(_ENCODER_HEAD_DIM),
                    mlp_ratio=float(mlp_ratio),
                    mlp_type=_resolve_encoder_mlp_type(mlp_type),
                    block_index=int(index),
                    use_norms=True,
                    position_encoding=DiTPositionEncoding.ROPE_2D_AXIAL_UNNORMALIZED,
                    conditioning=DiTConditioning.UNCOND,
                )
                for index in range(int(depth))
            ]
        )
        self.rope = AxialRoPE2D(
            head_dim=int(_ENCODER_HEAD_DIM),
            cfg=AxialRoPE2DConfig(
                base=10_000.0,
                min_period=None,
                max_period=None,
                coord_mode=AxialRoPE2DCoordMode.PATCH_INDICES,
                normalize_coords=AxialRoPE2DNormalizeCoords.MAX,
                dim_layout=AxialRoPE2DDimLayout.PAIR_INTERLEAVED,
                angle_multiplier=1.0,
                coord_offset=0.0,
                frequency_aware=None,
                beta_warp=None,
                alpha_warp=None,
            ),
        )
        match self.bottleneck_posterior_kind:
            case "deterministic":
                output_channels = int(bottleneck_dim)
            case "diagonal_gaussian":
                output_channels = 2 * int(bottleneck_dim)
            case _ as unreachable:
                raise RuntimeError(
                    f"Unsupported bottleneck_posterior_kind: {unreachable}"
                )
        self.to_bottleneck = nn.Conv2d(
            int(model_dim),
            output_channels,
            kernel_size=1,
            bias=True,
        )

    def _encode_projection(self, images: Tensor) -> Tensor:
        """Encode images to the raw bottleneck projection."""

        z = self.patchify(images)
        batch, channels, height, width = z.shape
        cond = torch.zeros(
            (int(batch), int(channels)),
            device=z.device,
            dtype=z.dtype,
        )
        rope_sincos = self.rope(H=int(height), W=int(width), scales=None)
        y = z
        for block in self.blocks:
            y = block(
                y,
                hw=(int(height), int(width)),
                cond_vec=cond,
                adaln_m=None,
                rope_sincos=rope_sincos,
                generator=None,
            )
        return self.to_bottleneck(y)

    def _apply_bottleneck_norm(self, z: Tensor) -> Tensor:
        """Return the unnormalized bottleneck mean."""

        return z

    def encode_posterior(self, images: Tensor) -> EncoderPosterior:
        """Encode images and return the posterior."""

        if self.bottleneck_posterior_kind != "diagonal_gaussian":
            raise RuntimeError(
                "encode_posterior requires bottleneck_posterior_kind=diagonal_gaussian"
            )
        projection = self._encode_projection(images)
        mean, logsnr = projection.chunk(2, dim=1)
        mean = self._apply_bottleneck_norm(mean)
        return EncoderPosterior(mean=mean, logsnr=logsnr)

    def forward(self, images: Tensor) -> Tensor:
        """Encode images to latent tokens."""

        projection = self._encode_projection(images)
        match self.bottleneck_posterior_kind:
            case "diagonal_gaussian":
                mean, logsnr = projection.chunk(2, dim=1)
                mean = self._apply_bottleneck_norm(mean)
                logsnr_fp32 = logsnr.to(torch.float32)
                alpha = torch.exp(0.5 * F.logsigmoid(logsnr_fp32))
                return (alpha * mean.to(torch.float32)).to(dtype=mean.dtype)
            case "deterministic":
                return self._apply_bottleneck_norm(projection)
            case _ as unreachable:
                raise RuntimeError(
                    f"Unsupported bottleneck_posterior_kind: {unreachable}"
                )
