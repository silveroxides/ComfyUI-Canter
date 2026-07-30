"""Dense SDPA attention blocks used by the DINAC-AE export."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from comfy.ldm.modules.attention import optimized_attention

from ..common.norms import RMSNorm
from ..common.rope import rotate_half, rotate_half_adjacent
from ..dit.position_encoding import DiTPositionEncoding


def _axial_rope_rotate_fn(
    position_encoding: DiTPositionEncoding,
) -> Callable[[Tensor], Tensor]:
    """Return the head-dimension rotation matching the configured RoPE layout."""

    match position_encoding:
        case (
            DiTPositionEncoding.ROPE_2D_AXIAL_DILATED
            | DiTPositionEncoding.ROPE_2D_AXIAL_NORMALIZED
            | DiTPositionEncoding.ROPE_2D_AXIAL_FREQ_AWARE
            | DiTPositionEncoding.ROPE_1D
        ):
            return rotate_half
        case (
            DiTPositionEncoding.ROPE_2D_AXIAL_UNNORMALIZED
            | DiTPositionEncoding.ROPE_2D_AXIAL_UNNORMALIZED_DILATED
            | DiTPositionEncoding.ROPE_2D_AXIAL_BETA_WARP
            | DiTPositionEncoding.ROPE_2D_AXIAL_ALPHA_WARP
            | DiTPositionEncoding.ROPE_3D_ZIMAGE
        ):
            return rotate_half_adjacent
        case _ as unreachable:
            raise ValueError(f"Unsupported RoPE position encoding: {unreachable}")


class DitSelfAttentionCore(nn.Module):
    """Dense self-attention core with optional axial RoPE on Q/K."""

    d_model: int
    n_heads: int
    head_dim: int
    position_encoding: DiTPositionEncoding
    qkv: nn.Linear
    proj_out: nn.Linear
    q_norm: RMSNorm
    k_norm: RMSNorm

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        position_encoding: DiTPositionEncoding,
        operations,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(self.d_model // self.n_heads)
        self.position_encoding = position_encoding
        self.qkv = operations.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj_out = operations.Linear(
            self.d_model, self.d_model, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def reset_parameters(self) -> None:
        """Reset projections to their initialization."""

        if self.qkv.weight is not None:
            nn.init.xavier_uniform_(self.qkv.weight)
        if self.proj_out.weight is not None:
            nn.init.xavier_uniform_(self.proj_out.weight)

    def forward(
        self,
        tokens: Tensor,
        *,
        rope_sincos: tuple[Tensor, Tensor] | None,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Apply dense self-attention to ``[B, N, D]`` tokens."""

        batch, sequence_length, _width = tokens.shape
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())
        q, k = self._apply_axial_rope_dense(q, k, rope_sincos=rope_sincos)
        attn = optimized_attention(
            q, k, v, self.n_heads, skip_reshape=True, skip_output_reshape=True,
            transformer_options=transformer_options or {},
        )
        attn = (
            attn.transpose(1, 2).contiguous().view(batch, sequence_length, self.d_model)
        )
        return self.proj_out(attn)

    def _apply_axial_rope_dense(
        self,
        q: Tensor,
        k: Tensor,
        *,
        rope_sincos: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply axial RoPE to dense Q/K tensors."""

        if rope_sincos is None:
            return q, k
        sin, cos = rope_sincos
        rope_len = int(sin.shape[-2])
        rope_dtype = sin.dtype
        q_dtype = q.dtype
        k_dtype = k.dtype
        q_rope = q.to(dtype=rope_dtype)
        k_rope = k.to(dtype=rope_dtype)
        match sin.dim():
            case 2:
                sin_b = sin.view(1, 1, rope_len, self.head_dim)
                cos_b = cos.view(1, 1, rope_len, self.head_dim)
            case 3:
                sin_b = sin.view(int(q.shape[0]), 1, rope_len, self.head_dim)
                cos_b = cos.view(int(q.shape[0]), 1, rope_len, self.head_dim)
            case _ as unreachable:
                raise ValueError(f"Unsupported RoPE tensor rank: {int(unreachable)}")
        rotate = _axial_rope_rotate_fn(self.position_encoding)
        q_span = q_rope[:, :, :rope_len, :]
        k_span = k_rope[:, :, :rope_len, :]
        q_head = (q_span * cos_b) + (rotate(q_span) * sin_b)
        k_head = (k_span * cos_b) + (rotate(k_span) * sin_b)
        q_rope = torch.cat([q_head, q_rope[:, :, rope_len:, :]], dim=2)
        k_rope = torch.cat([k_head, k_rope[:, :, rope_len:, :]], dim=2)
        return q_rope.to(dtype=q_dtype), k_rope.to(dtype=k_dtype)


class CrossAttentionCore(nn.Module):
    """Dense cross-attention core used by the class-token readout."""

    query_dim: int
    context_dim: int
    context_extra_dim: int
    key_extra_dim: int
    n_heads: int
    head_dim: int
    attn_dim: int
    context_in_dim: int
    attn_dropout: float
    kv_proj: nn.Linear
    k_extra_proj: nn.Linear | None
    out_proj: nn.Linear
    q_norm_heads: RMSNorm
    k_norm_heads: RMSNorm

    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        n_heads: int,
        head_dim: int,
        context_extra_dim: int = 0,
        key_extra_dim: int = 0,
        attn_dropout: float = 0.0,
        operations=None,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)
        self.context_extra_dim = int(context_extra_dim)
        self.key_extra_dim = int(key_extra_dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.attn_dim = int(self.n_heads * self.head_dim)
        self.context_in_dim = int(self.context_dim + self.context_extra_dim)
        self.attn_dropout = float(attn_dropout)
        if operations is None:
            import comfy.ops

            operations = comfy.ops.manual_cast
        self.kv_proj = operations.Linear(
            self.context_in_dim, 2 * self.attn_dim, bias=False
        )
        if self.key_extra_dim == 0:
            self.k_extra_proj = None
        else:
            self.k_extra_proj = operations.Linear(
                self.key_extra_dim, self.attn_dim, bias=False
            )
        self.out_proj = operations.Linear(
            self.attn_dim, self.query_dim, bias=False
        )
        self.q_norm_heads = RMSNorm(self.head_dim)
        self.k_norm_heads = RMSNorm(self.head_dim)

    def reset_parameters(self) -> None:
        """Reset projections to their initialization."""

        if self.kv_proj.weight is not None:
            nn.init.xavier_uniform_(self.kv_proj.weight)
        if self.k_extra_proj is not None and self.k_extra_proj.weight is not None:
            nn.init.xavier_uniform_(self.k_extra_proj.weight)
        if self.out_proj.weight is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, sequence_length, _width = x.shape
        return x.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(
            1, 2
        )

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _heads, sequence_length, _head_dim = x.shape
        return (
            x.transpose(1, 2).contiguous().view(batch, sequence_length, self.attn_dim)
        )

    def forward(
        self,
        q_tokens: Tensor,
        kv_tokens: Tensor,
        *,
        training: bool,
        key_extra: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Apply dense cross-attention to query and context tokens."""

        kv = self.kv_proj(kv_tokens)
        k, v = kv.chunk(2, dim=-1)
        if self.k_extra_proj is not None and key_extra is not None:
            k = k + self.k_extra_proj(key_extra)
        q = self.q_norm_heads(self._split_heads(q_tokens).contiguous())
        k = self.k_norm_heads(self._split_heads(k).contiguous())
        v = self._split_heads(v).contiguous()
        if key_padding_mask is None:
            attn_mask = None
        else:
            attn_mask = (~key_padding_mask).to(dtype=q.dtype)
            attn_mask = attn_mask.view(
                key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1]
            )
            attn_mask = attn_mask.masked_fill(attn_mask > 0, float("-inf"))
        attn = optimized_attention(
            q, k, v, self.n_heads, mask=attn_mask,
            skip_reshape=True, skip_output_reshape=True,
            transformer_options=transformer_options or {},
        )
        return self.out_proj(self._merge_heads(attn))


__all__ = ["CrossAttentionCore", "DitSelfAttentionCore"]
