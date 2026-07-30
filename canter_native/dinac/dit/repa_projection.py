"""DINO token/class alignment head used by the DINAC-AE export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ..common.norms import RMSNorm
from ..dit.axial_rope2d import (
    AxialRoPE2D,
    AxialRoPE2DConfig,
    AxialRoPE2DCoordMode,
    AxialRoPE2DDimLayout,
    AxialRoPE2DNormalizeCoords,
)
from ..dit.blocks import DitBlock
from ..dit.body_config import DiTConditioning
from ..dit.position_encoding import DiTPositionEncoding
from ..dit.xattn_blocks import CrossAttentionBlock, CrossAttentionConfig

if TYPE_CHECKING:
    from ..dit.mlp_types import MLPType


@dataclass(frozen=True)
class DinoTokenAlignmentOutput:
    """Predicted DINO class token and spatial patch tokens."""

    class_token: Tensor
    spatial_tokens: Tensor


def _prepend_identity_rope_prefix(
    *,
    rope_sincos: tuple[Tensor, Tensor],
    prefix_token_count: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Prepend no-op RoPE entries for class/register prefix tokens."""

    sin, cos = rope_sincos
    prefix_shape = (int(prefix_token_count), int(sin.shape[-1]))
    prefix_sin = torch.zeros(prefix_shape, device=device, dtype=sin.dtype)
    prefix_cos = torch.ones(prefix_shape, device=device, dtype=cos.dtype)
    match sin.dim():
        case 2:
            return (
                torch.cat([prefix_sin, sin.to(device=device)], dim=0),
                torch.cat([prefix_cos, cos.to(device=device)], dim=0),
            )
        case 3:
            batch = int(sin.shape[0])
            return (
                torch.cat(
                    [
                        prefix_sin.unsqueeze(0).expand(batch, -1, -1),
                        sin.to(device=device),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        prefix_cos.unsqueeze(0).expand(batch, -1, -1),
                        cos.to(device=device),
                    ],
                    dim=1,
                ),
            )
        case _ as unreachable:
            raise ValueError(f"Unsupported RoPE tensor rank: {int(unreachable)}")


class DinoTokenAlignmentHead(nn.Module):
    """Predict DINOv3 spatial tokens and a class token from latent grids."""

    in_channels: int
    feature_dim: int
    model_dim: int
    register_token_count: int
    in_proj: nn.Conv2d
    initial_class_token: nn.Parameter
    register_tokens: nn.Parameter
    block: DitBlock
    spatial_output_norm: RMSNorm
    class_readout: CrossAttentionBlock
    class_output_norm: RMSNorm
    _axial_rope2d: AxialRoPE2D

    def __init__(
        self,
        *,
        in_channels: int,
        feature_dim: int,
        model_dim: int,
        head_dim: int,
        mlp_ratio: float,
        mlp_activation: MLPType,
        block_index: int,
        register_token_count: int,
    ) -> None:
        super().__init__()
        if int(feature_dim) != int(model_dim):
            raise ValueError("DINAC-AE class head requires feature_dim == model_dim")
        if int(register_token_count) != 4:
            raise ValueError("DINAC-AE class head requires four register tokens")
        self.register_token_count = int(register_token_count)
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)
        self.model_dim = int(model_dim)
        self.in_proj = nn.Conv2d(
            self.in_channels,
            self.model_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
        )
        self.initial_class_token = nn.Parameter(torch.empty((1, self.model_dim)))
        self.register_tokens = nn.Parameter(
            torch.empty((self.register_token_count, self.model_dim))
        )
        nn.init.normal_(self.initial_class_token, mean=0.0, std=0.02)
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        conditioning = DiTConditioning.UNCOND
        self.block = DitBlock(
            d_model=self.model_dim,
            n_heads=int(self.model_dim // int(head_dim)),
            mlp_ratio=float(mlp_ratio),
            mlp_type=mlp_activation,
            block_index=int(block_index),
            use_norms=True,
            position_encoding=DiTPositionEncoding.ROPE_2D_AXIAL_UNNORMALIZED,
            conditioning=conditioning,
        )
        self.spatial_output_norm = RMSNorm(self.model_dim, affine=False)
        self.class_readout = CrossAttentionBlock(
            query_dim=self.model_dim,
            context_dim=self.model_dim,
            cfg=CrossAttentionConfig(
                n_heads=int(self.model_dim // int(head_dim)),
                head_dim=int(head_dim),
                query_extra_dim=0,
                context_extra_dim=0,
                mlp_ratio=float(mlp_ratio),
                attn_dropout=0.0,
                mlp_type=mlp_activation,
                activation_config=None,
                use_norms=True,
                block_index=int(block_index) + 1,
                use_attn_residual=True,
            ),
        )
        self.class_output_norm = RMSNorm(self.model_dim, affine=False)
        self._axial_rope2d = AxialRoPE2D(
            head_dim=int(head_dim),
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

    def compile_for_training(self, *, fullgraph: bool, dynamic: bool) -> None:
        """No-op hook kept for source API compatibility."""

        _ = fullgraph, dynamic

    def compile_for_eval(self, *, fullgraph: bool, dynamic: bool) -> None:
        """No-op hook kept for source API compatibility."""

        _ = fullgraph, dynamic

    def forward(self, latents: Tensor, *, t: Tensor) -> DinoTokenAlignmentOutput:
        """Return predicted class and spatial DINO tokens."""

        y = self.in_proj(latents)
        batch, _channels, height, width = y.shape
        spatial_tokens = y.flatten(2).transpose(1, 2)
        class_token = self.initial_class_token.to(device=y.device, dtype=y.dtype)
        class_token = class_token.unsqueeze(0).expand(int(batch), -1, -1)
        register_tokens = self.register_tokens.to(device=y.device, dtype=y.dtype)
        register_tokens = register_tokens.unsqueeze(0).expand(int(batch), -1, -1)
        tokens = torch.cat([class_token, register_tokens, spatial_tokens], dim=1)
        rope_sincos = _prepend_identity_rope_prefix(
            rope_sincos=self._axial_rope2d(H=int(height), W=int(width), scales=None),
            prefix_token_count=int(1 + self.register_token_count),
            device=y.device,
        )
        _ = t
        cond = torch.zeros(
            (int(batch), self.model_dim),
            device=y.device,
            dtype=y.dtype,
        )
        tokens = self.block(
            tokens,
            hw=(int(height), int(width)),
            cond_vec=cond,
            adaln_m=None,
            rope_sincos=rope_sincos,
            generator=None,
        )
        class_query = tokens[:, :1, :]
        context = tokens[:, 1:, :]
        class_output = self.class_readout(class_query, context)[:, 0, :]
        class_output = self.class_output_norm(class_output)
        prefix_token_count = int(1 + self.register_token_count)
        predicted_spatial = self.spatial_output_norm(tokens[:, prefix_token_count:, :])
        return DinoTokenAlignmentOutput(
            class_token=class_output,
            spatial_tokens=predicted_spatial,
        )


__all__ = ["DinoTokenAlignmentHead", "DinoTokenAlignmentOutput"]
