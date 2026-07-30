"""Dense unconditional DiT blocks used by the DINAC-AE export."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common.norms import RMSNorm
from ..common.rope import Rope1D
from ..dit.attention_blocks import DitSelfAttentionCore
from ..dit.body_config import DiTConditioning
from ..dit.mlp import build_dit_mlp, reset_module_parameters
from ..dit.mlp_types import MLPType
from ..dit.position_encoding import DiTPositionEncoding


def _flatten_tokens(
    x: Tensor, hw: tuple[int, int] | None
) -> tuple[Tensor, tuple[int, int], bool]:
    """Return dense tokens plus spatial metadata."""

    if x.dim() == 4:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        return tokens, (int(height), int(width)), True
    return x, hw if hw is not None else (int(x.shape[1]), 1), False


def _restore_spatial(tokens: Tensor, hw: tuple[int, int]) -> Tensor:
    """Restore dense tokens to NCHW features."""

    batch, _sequence_length, width = tokens.shape
    height, spatial_width = hw
    return tokens.transpose(1, 2).reshape(batch, width, height, spatial_width)


class TransformerBlock(nn.Module):
    """Dense pre-norm transformer block kept for import compatibility."""

    d_model: int
    n_heads: int
    attn_norm: RMSNorm | None
    mlp_norm: RMSNorm | None
    self_attn: DitSelfAttentionCore
    rope_1d: Rope1D | None
    mlp: nn.Module

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        mlp_type: MLPType,
        activation_config: object | None = None,
        block_index: int = 0,
        use_norms: bool = True,
        position_encoding: DiTPositionEncoding = DiTPositionEncoding.NONE,
        rope_theta: float | None = None,
        rope_max_position_embeddings: int | None = None,
        operations=None,
    ) -> None:
        super().__init__()
        if operations is None:
            import comfy.ops

            operations = comfy.ops.manual_cast
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.attn_norm = RMSNorm(self.d_model) if bool(use_norms) else None
        self.mlp_norm = RMSNorm(self.d_model) if bool(use_norms) else None
        self.self_attn = DitSelfAttentionCore(
            d_model=self.d_model,
            n_heads=self.n_heads,
            position_encoding=position_encoding,
            operations=operations,
        )
        self.rope_1d = self._build_rope_1d(
            position_encoding=position_encoding,
            rope_theta=rope_theta,
            rope_max_position_embeddings=rope_max_position_embeddings,
        )
        self.mlp = build_dit_mlp(
            mlp_type=mlp_type,
            in_features=self.d_model,
            hidden_budget=int(round(float(mlp_ratio) * self.d_model)),
            activation_config=activation_config,
            block_index=int(block_index),
            bias_up=False,
            bias_down=False,
            operations=operations,
        )

    def reset_parameters(self) -> None:
        """Reset attention and MLP parameters."""

        self.self_attn.reset_parameters()
        reset_module_parameters(self.mlp)

    def _build_rope_1d(
        self,
        *,
        position_encoding: DiTPositionEncoding,
        rope_theta: float | None,
        rope_max_position_embeddings: int | None,
    ) -> Rope1D | None:
        """Build 1D RoPE for sequence-only transformer blocks."""

        match position_encoding:
            case DiTPositionEncoding.NONE:
                return None
            case DiTPositionEncoding.ROPE_1D:
                if rope_theta is None or rope_max_position_embeddings is None:
                    raise ValueError("ROPE_1D requires theta and max positions")
                return Rope1D(
                    dim=int(self.d_model // self.n_heads),
                    max_position_embeddings=int(rope_max_position_embeddings),
                    base=float(rope_theta),
                )
            case _ as unreachable:
                raise ValueError(f"Unsupported TransformerBlock RoPE: {unreachable}")

    def forward(self, tokens: Tensor, *, generator: torch.Generator | None) -> Tensor:  # type: ignore[override]
        """Apply dense self-attention and MLP to token sequences."""

        _ = generator
        attn_in = self.attn_norm(tokens) if self.attn_norm is not None else tokens
        rope_sincos = self._build_rope_sincos(attn_in)
        x = tokens + self.self_attn(attn_in, rope_sincos=rope_sincos)
        mlp_in = self.mlp_norm(x) if self.mlp_norm is not None else x
        return x + self.mlp(mlp_in)

    def _build_rope_sincos(self, tokens: Tensor) -> tuple[Tensor, Tensor] | None:
        """Return dense 1D RoPE sin/cos buffers."""

        rope = self.rope_1d
        if rope is None:
            return None
        batch = int(tokens.shape[0])
        seqlen = int(tokens.shape[1])
        position_ids = torch.arange(
            seqlen,
            device=tokens.device,
            dtype=torch.int64,
        ).unsqueeze(0)
        position_ids = position_ids.expand(batch, seqlen)
        dummy = tokens.new_empty(batch, self.n_heads, seqlen, rope.dim)
        cos, sin = rope(dummy, position_ids)
        return sin, cos


class DitBlock(nn.Module):
    """Dense unconditional DiT self-attention block."""

    d: int
    h: int
    dh: int
    hidden_budget: int
    position_encoding: DiTPositionEncoding
    conditioning: DiTConditioning
    adaln: object | None
    gate_attn: nn.Parameter | None
    gate_mlp: nn.Parameter | None
    use_norms: bool
    attn_norm1: RMSNorm
    attn_norm2: RMSNorm
    mlp_norm1: RMSNorm
    mlp_norm2: RMSNorm
    attn_core: DitSelfAttentionCore
    qkv: nn.Linear
    proj_out: nn.Linear
    mlp: nn.Module

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        *,
        adaln: object | None = None,
        mlp_type: MLPType = MLPType.GELU,
        activation_config: object | None = None,
        block_index: int = 0,
        use_norms: bool = True,
        position_encoding: DiTPositionEncoding = DiTPositionEncoding.NONE,
        conditioning: DiTConditioning = DiTConditioning.UNCOND,
        operations=None,
    ) -> None:
        super().__init__()
        if operations is None:
            import comfy.ops

            operations = comfy.ops.manual_cast
        if conditioning is not DiTConditioning.UNCOND or adaln is not None:
            raise ValueError("DINAC-AE export only supports unconditional DitBlock")
        self.d = int(d_model)
        self.h = int(n_heads)
        self.dh = int(self.d // self.h)
        self.hidden_budget = int(float(mlp_ratio) * self.d)
        self.position_encoding = position_encoding
        self.conditioning = conditioning
        self.adaln = None
        self.gate_attn = None
        self.gate_mlp = None
        self.use_norms = bool(use_norms)
        self.attn_norm1 = RMSNorm(self.d)
        self.attn_norm2 = RMSNorm(self.d)
        self.mlp_norm1 = RMSNorm(self.d)
        self.mlp_norm2 = RMSNorm(self.d)
        self.attn_core = DitSelfAttentionCore(
            d_model=self.d,
            n_heads=self.h,
            position_encoding=position_encoding,
            operations=operations,
        )
        self.qkv = self.attn_core.qkv
        self.proj_out = self.attn_core.proj_out
        self.mlp = build_dit_mlp(
            mlp_type=mlp_type,
            in_features=self.d,
            hidden_budget=self.hidden_budget,
            activation_config=activation_config,
            block_index=int(block_index),
            bias_up=False,
            bias_down=False,
            operations=operations,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset attention and MLP parameters."""

        self.attn_core.reset_parameters()
        reset_module_parameters(self.mlp)

    def compile_for_training(self, *, fullgraph: bool, dynamic: bool) -> None:
        """No-op hook kept for API compatibility."""

        _ = fullgraph, dynamic

    def compile_for_eval(self, *, fullgraph: bool, dynamic: bool) -> None:
        """No-op hook kept for API compatibility."""

        _ = fullgraph, dynamic

    def forward(
        self,
        x: Tensor,
        hw: tuple[int, int],
        cond_vec: Tensor,
        adaln_m: Tensor | None = None,
        *,
        rope_sincos: tuple[Tensor, Tensor] | None = None,
        generator: torch.Generator | None = None,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Apply the dense unconditional block to spatial features or tokens."""

        _ = cond_vec, adaln_m, generator
        tokens, hw_tokens, was_spatial = _flatten_tokens(x, hw)
        attn_in = self.attn_norm1(tokens) if self.use_norms else tokens
        y = self.attn_core(
            attn_in,
            rope_sincos=rope_sincos,
            transformer_options=transformer_options,
        )
        attn_out = self.attn_norm2(y) if self.use_norms else y
        tokens = tokens + attn_out
        mlp_in = self.mlp_norm1(tokens) if self.use_norms else tokens
        mlp_out = self.mlp(mlp_in)
        mlp_out = self.mlp_norm2(mlp_out) if self.use_norms else mlp_out
        tokens = tokens + mlp_out
        if was_spatial:
            return _restore_spatial(tokens, hw_tokens)
        return tokens


__all__ = ["DitBlock", "TransformerBlock"]
