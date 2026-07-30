"""Lean inference blocks used by the frozen Canter architecture."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from comfy.ldm.modules.attention import optimized_attention

from .runtime import configure_jagged_compilation


class TextAttentionBackend(Enum):
    """Text tensor layout selected at the public inference boundary."""

    DENSE = "dense"
    JAGGED = "jagged"


class JaggedTensor(Protocol):
    """Typed subset of the PyTorch jagged NestedTensor interface."""

    _maybe_min_seqlen: int | None
    _maybe_max_seqlen: int | None

    def offsets(self) -> Tensor:
        """Return packed row offsets on the values device."""

    def values(self) -> Tensor:
        """Return packed token values."""


def _jagged(tensor: Tensor) -> JaggedTensor:
    """Return a statically typed jagged-tensor view for a validated input."""

    return cast("JaggedTensor", tensor)


def _jagged_rope(
    tokens: Tensor,
    heads: int,
    head_dim: int,
    theta: float,
) -> tuple[Tensor, Tensor]:
    """Build packed text RoPE with Canter's frozen Rope1D operation sequence."""

    nested = _jagged(tokens)
    offsets = nested.offsets().to(dtype=torch.int64)
    values = nested.values()
    lengths = offsets[1:] - offsets[:-1]
    total = values.shape[0]
    packed = torch.arange(total, device=offsets.device, dtype=torch.int64)
    starts = torch.repeat_interleave(offsets[:-1], lengths)
    positions = (packed - starts).unsqueeze(0)
    dummy = values.new_empty(1, heads, total, head_dim)
    sine, cosine = _text_rope_from_positions(
        positions,
        head_dim=head_dim,
        theta=theta,
        dtype=dummy.dtype,
        device=dummy.device,
    )
    return sine.squeeze(0), cosine.squeeze(0)


def _text_rope_from_positions(
    positions: Tensor,
    *,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Evaluate the frozen Rope1D math for explicit position rows."""

    inverse = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / float(head_dim)
        )
    )
    inverse_expanded = (
        inverse[None, :, None]
        .float()
        .expand(
            positions.shape[0],
            -1,
            1,
        )
    )
    positions_expanded = positions[:, None, :].float() / 1.0
    with torch.autocast(device_type=device.type, enabled=False):
        frequencies = (inverse_expanded.float() @ positions_expanded.float()).transpose(
            1, 2
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos()
        sine = embedding.sin()
    return sine.to(dtype=dtype), cosine.to(dtype=dtype)


def rotate_adjacent_pairs(x: Tensor) -> Tensor:
    """Rotate adjacent channel pairs for Canter's axial image RoPE."""

    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    return torch.stack((-second, first), dim=-1).reshape_as(x)


def rotate_halves(x: Tensor) -> Tensor:
    """Rotate channel halves for the text encoder's one-dimensional RoPE."""

    midpoint = x.shape[-1] // 2
    first = x[..., :midpoint]
    second = x[..., midpoint:]
    return torch.cat((-second, first), dim=-1)


class RMSNorm(nn.Module):
    """RMS normalization with Canter's explicit AMP behavior."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self._eps = 1e-6
        self.weight = nn.Parameter(torch.ones(self.width))

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> RMSNorm:
        """Move the module without rounding its FP32 affine weight."""

        weight = self.weight.detach().float()
        super()._apply(fn, recurse=recurse)
        self.weight.data = weight.to(device=self.weight.device)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the final dimension and preserve the input dtype."""

        output = F.rms_norm(
            x,
            (self.width,),
            self.weight.to(dtype=x.dtype),
            self._eps,
        )
        return output if output.dtype == x.dtype else output.to(x.dtype)


class GELUMLP(nn.Module):
    """Bias-free GELU feed-forward network."""

    def __init__(self, width: int, hidden_width: int, operations) -> None:
        super().__init__()
        self.up = operations.Linear(width, hidden_width, bias=False)
        self.down = operations.Linear(hidden_width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two feed-forward projections."""

        return self.down(F.gelu(self.up(x)))


class LowRankAdaLN(nn.Module):
    """Bias-free low-rank delta added to a shared AdaLN projection."""

    def __init__(
        self, input_width: int, output_width: int, rank: int, operations
    ) -> None:
        super().__init__()
        self.down = operations.Linear(input_width, rank, bias=False)
        self.up = operations.Linear(rank, output_width, bias=False)

    def forward(self, activated_condition: Tensor) -> Tensor:
        """Project an already SiLU-activated conditioning vector."""

        return self.up(self.down(activated_condition))


class SharedAdaLN(nn.Module):
    """Shared SiLU-linear AdaLN base used by all image blocks."""

    def __init__(self, width: int, output_width: int, operations) -> None:
        super().__init__()
        self.proj = operations.Linear(width, output_width, bias=True)

    def activated(self, condition: Tensor) -> Tensor:
        """Return the common SiLU activation used by base and deltas."""

        return F.silu(condition)

    def project(self, activated_condition: Tensor) -> Tensor:
        """Project an activated condition in the parameter compute dtype."""

        return self.proj(activated_condition)


def project_adaln(
    base: SharedAdaLN,
    delta: LowRankAdaLN,
    condition: Tensor,
) -> Tensor:
    """Sum the shared and per-block AdaLN projections in float32."""

    activated = base.activated(condition)
    return base.project(activated).float() + delta(activated).float()


def apply_scale(normed: Tensor, scale: Tensor) -> Tensor:
    """Apply sample-wise AdaLN scale in float32 and restore token dtype."""

    normed_fp32 = normed.to(dtype=torch.float32)
    scale_fp32 = scale.to(dtype=torch.float32).unsqueeze(1)
    value = normed_fp32 * (1.0 + scale_fp32)
    return value.to(dtype=normed.dtype)


def gated_residual(gate: Tensor, value: Tensor) -> Tensor:
    """Apply Canter's float32 tanh residual gate."""

    gate_fp32 = torch.tanh(gate.to(dtype=torch.float32)).unsqueeze(1)
    value_fp32 = value.to(dtype=torch.float32)
    result = gate_fp32 * value_fp32
    return result.to(dtype=value.dtype)


class ImageSelfAttention(nn.Module):
    """Canter image self-attention with QK norm and axial 2D RoPE."""

    def __init__(self, width: int, heads: int, operations) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.qkv = operations.Linear(self.width, 3 * self.width, bias=False)
        self.proj_out = operations.Linear(self.width, self.width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attention_query_scale = 1.0

    def forward(
        self,
        tokens: Tensor,
        *,
        rope_sincos: tuple[Tensor, Tensor],
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Apply dense scaled-dot-product self-attention."""

        batch, length, _ = tokens.shape
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q = (
            q.view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            k.view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            v.view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        sin, cos = rope_sincos
        q_dtype = q.dtype
        k_dtype = k.dtype
        q_rope = q.to(dtype=sin.dtype)
        k_rope = k.to(dtype=sin.dtype)
        sin = sin.view(1, 1, length, self.head_dim)
        cos = cos.view(1, 1, length, self.head_dim)
        q_span = q_rope[:, :, :length, :]
        k_span = k_rope[:, :, :length, :]
        q_head = q_span * cos + rotate_adjacent_pairs(q_span) * sin
        k_head = k_span * cos + rotate_adjacent_pairs(k_span) * sin
        q_rope = torch.cat((q_head, q_rope[:, :, length:, :]), dim=2)
        k_rope = torch.cat((k_head, k_rope[:, :, length:, :]), dim=2)
        q = q_rope.to(dtype=q_dtype)
        k = k_rope.to(dtype=k_dtype)
        q = q * float(self.attention_query_scale)
        attended = optimized_attention(
            q, k, v, self.heads, skip_reshape=True, skip_output_reshape=True,
            transformer_options=transformer_options or {},
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.width)
        return self.proj_out(attended)


class DiTBlock(nn.Module):
    """One frozen Canter image transformer block."""

    def __init__(self, width: int, heads: int, mlp_ratio: int, operations) -> None:
        super().__init__()
        self.attn_norm1 = RMSNorm(width)
        self.attn_norm2 = RMSNorm(width)
        self.mlp_norm1 = RMSNorm(width)
        self.mlp_norm2 = RMSNorm(width)
        self.mlp = GELUMLP(width, width * mlp_ratio, operations)
        self.attention = ImageSelfAttention(width, heads, operations)
        self._implementation: Callable[..., Tensor] = self._forward_impl
        self._compiled = False

    def forward(
        self,
        tokens: Tensor,
        hw: tuple[int, int],
        condition: Tensor,
        modulation: Tensor,
        *,
        rope_sincos: tuple[Tensor, Tensor],
        generator: torch.Generator | None,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Dispatch to the eager or setup-time compiled implementation."""

        return self._implementation(
            tokens,
            hw,
            condition,
            modulation,
            rope_sincos=rope_sincos,
            generator=generator,
            transformer_options=transformer_options,
        )

    def _forward_impl(
        self,
        tokens: Tensor,
        hw: tuple[int, int],
        condition: Tensor,
        modulation: Tensor,
        *,
        rope_sincos: tuple[Tensor, Tensor],
        generator: torch.Generator | None,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Apply attention and MLP residuals with packed AdaLN modulation."""

        del hw, condition, generator
        scale_a, gate_a, scale_m, gate_m = self._split_modulation(modulation)
        attn_in = apply_scale(self.attn_norm1(tokens), scale_a)
        y = self.attention(
            attn_in,
            rope_sincos=rope_sincos,
            transformer_options=transformer_options,
        )
        attn_out = self.attn_norm2(y)
        tokens = tokens + gated_residual(gate_a, attn_out)
        mlp_in = apply_scale(self.mlp_norm1(tokens), scale_m)
        mlp_out = self.mlp(mlp_in)
        mlp_out = self.mlp_norm2(mlp_out)
        return tokens + gated_residual(gate_m, mlp_out)

    @staticmethod
    def _split_modulation(
        modulation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split the frozen packed image AdaLN values."""

        scale_attn, gate_attn, scale_mlp, gate_mlp = modulation.chunk(4, dim=-1)
        return scale_attn, gate_attn, scale_mlp, gate_mlp

    def compile_for_inference(self, *, fullgraph: bool, dynamic: bool) -> None:
        """Compile the block once; compiler errors propagate to the caller."""

        if self._compiled:
            raise RuntimeError("DiTBlock is already compiled.")
        self._implementation = torch.compile(
            self._forward_impl,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self._compiled = True


class TransitionDiTBlock(DiTBlock):
    """Always-on boundary DiT block with a learned modulation gate."""

    def __init__(self, width: int, heads: int, mlp_ratio: int, operations) -> None:
        super().__init__(width, heads, mlp_ratio, operations)
        self.adaln_modulation_scale = nn.Parameter(torch.zeros(()))

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> TransitionDiTBlock:
        """Move the block without rounding its FP32 modulation gate."""

        scale = self.adaln_modulation_scale.detach().float()
        super()._apply(fn, recurse=recurse)
        self.adaln_modulation_scale.data = scale.to(
            device=self.adaln_modulation_scale.device
        )
        return self

    def _split_modulation(
        self,
        modulation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Scale every packed modulation component by the transition gate."""

        scale_attn, gate_attn, scale_mlp, gate_mlp = super()._split_modulation(
            modulation
        )
        multiplier = self.adaln_modulation_scale.to(dtype=scale_attn.dtype)
        return (
            scale_attn * multiplier,
            gate_attn * multiplier,
            scale_mlp * multiplier,
            gate_mlp * multiplier,
        )


def text_rope(
    length: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build one dense row of Rope1D values in float32."""

    positions = torch.arange(
        length,
        device=device,
        dtype=torch.int64,
    ).unsqueeze(0)
    return _text_rope_from_positions(
        positions,
        head_dim=head_dim,
        theta=theta,
        dtype=torch.float32,
        device=device,
    )


class TextSelfAttention(nn.Module):
    """Text refinement self-attention with QK norm and 1D RoPE."""

    def __init__(self, width: int, heads: int, theta: float, operations) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.theta = float(theta)
        self.qkv = operations.Linear(width, 3 * width, bias=False)
        self.proj_out = operations.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def dense(
        self, tokens: Tensor, mask: Tensor, transformer_options: dict | None = None
    ) -> Tensor:
        """Apply dense padded self-attention."""

        batch, length, _ = tokens.shape
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q = (
            q.view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            k.view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            v.view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        sin, cos = text_rope(length, self.head_dim, self.theta, tokens.device)
        sin = sin.to(dtype=q.dtype).view(1, 1, length, self.head_dim)
        cos = cos.to(dtype=q.dtype).view(1, 1, length, self.head_dim)
        q = q * cos + rotate_halves(q) * sin
        k = k * cos + rotate_halves(k) * sin
        q = q * 1.0
        attention_mask = mask[:, None, None, :]
        attended = optimized_attention(
            q, k, v, self.heads, mask=attention_mask,
            skip_reshape=True, skip_output_reshape=True,
            transformer_options=transformer_options or {},
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.width)
        return self.proj_out(attended)

    def jagged(
        self,
        tokens: Tensor,
        rope: tuple[Tensor, Tensor],
        min_length: int,
        max_length: int,
    ) -> Tensor:
        """Apply packed FlashAttention to jagged text."""

        nested = _jagged(tokens)
        values = nested.values()
        offsets = nested.offsets()
        total = values.shape[0]
        q, k, v = self.qkv(values).chunk(3, dim=-1)
        q = self.q_norm(q.view(total, self.heads, self.head_dim)).contiguous()
        k = self.k_norm(k.view(total, self.heads, self.head_dim)).contiguous()
        v = v.view(total, self.heads, self.head_dim).contiguous()
        sin, cos = rope
        sin = sin.view(total, 1, self.head_dim)
        cos = cos.view(total, 1, self.head_dim)
        rope_dtype = sin.dtype
        query_dtype = q.dtype
        key_dtype = k.dtype
        query_rope = q.to(dtype=rope_dtype)
        key_rope = k.to(dtype=rope_dtype)
        query_rope = query_rope * cos + rotate_halves(query_rope) * sin
        key_rope = key_rope * cos + rotate_halves(key_rope) * sin
        q = query_rope.to(dtype=query_dtype)
        k = key_rope.to(dtype=key_dtype)
        q = q * 1.0
        offsets_i32 = offsets.to(dtype=torch.int32)
        lengths_i32 = offsets_i32[1:] - offsets_i32[:-1]
        del lengths_i32
        flash = cast(
            "Callable[..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]",
            torch.ops.aten._flash_attention_forward,
        )
        attended, _, _, _, _ = flash(
            q,
            k,
            v,
            offsets_i32,
            offsets_i32,
            max_length,
            max_length,
            0.0,
            False,
            False,
        )
        output = self.proj_out(attended.contiguous().view(total, self.width))
        return torch.nested.nested_tensor_from_jagged(
            output,
            offsets,
            jagged_dim=1,
            min_seqlen=min_length,
            max_seqlen=max_length,
        )


class TextTransformerBlock(nn.Module):
    """Pre-norm text transformer with compiled dense and jagged kernels."""

    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: int,
        theta: float,
        backend: TextAttentionBackend,
        operations,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.mlp_norm = RMSNorm(width)
        self.self_attn = TextSelfAttention(width, heads, theta, operations)
        self.mlp = GELUMLP(width, width * mlp_ratio, operations)
        self._dense_implementation: Callable[..., Tensor] = self._forward_dense
        self._jagged_implementation: Callable[..., Tensor] = self._forward_jagged
        self._jagged_singleton_implementation: Callable[..., Tensor] = (
            self._forward_jagged_singleton
        )
        self._implementation: Callable[..., Tensor]
        self._backend = backend
        self._compiled = False
        self.select_backend(backend)

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        min_length: int,
        max_length: int,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Dispatch to the eager or setup-time compiled implementation."""

        return self._implementation(
            tokens, mask, min_length, max_length, transformer_options
        )

    def _forward_dense(
        self,
        tokens: Tensor,
        mask: Tensor,
        min_length: int,
        max_length: int,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Refine dense padded text tokens."""

        del min_length, max_length
        tokens = tokens + self.self_attn.dense(
            self.attn_norm(tokens), mask, transformer_options
        )
        return tokens + self.mlp(self.mlp_norm(tokens))

    def _forward_jagged(
        self,
        tokens: Tensor,
        mask: Tensor,
        min_length: int,
        max_length: int,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Refine packed text with FlashAttention."""

        del mask, transformer_options
        normalized = self.attn_norm(tokens)
        rope = _jagged_rope(
            normalized,
            self.self_attn.heads,
            self.self_attn.head_dim,
            self.self_attn.theta,
        )
        attention = self.self_attn.jagged(
            normalized,
            rope,
            min_length,
            max_length,
        )
        tokens = tokens + attention
        return tokens + self.mlp(self.mlp_norm(tokens))

    def forward_jagged_singleton(self, tokens: Tensor) -> Tensor:
        """Run the optimized one-token path for every batch row."""

        return self._jagged_singleton_implementation(tokens)

    def _forward_jagged_singleton(self, tokens: Tensor) -> Tensor:
        """Refine the learned one-token unconditional conditioning."""

        nested = _jagged(tokens)
        values = nested.values()
        offsets = nested.offsets()
        normalized = self.attn_norm(values)
        _, _, value = self.self_attn.qkv(normalized).chunk(3, dim=-1)
        values = values + self.self_attn.proj_out(value)
        values = values + self.mlp(self.mlp_norm(values))
        return torch.nested.nested_tensor_from_jagged(
            values,
            offsets,
            jagged_dim=1,
            min_seqlen=1,
            max_seqlen=1,
        )

    def select_backend(self, backend: TextAttentionBackend) -> None:
        """Select one prebuilt implementation without branching in ``forward``."""

        if self._compiled and backend is not self._backend:
            raise RuntimeError(
                "Text backend cannot change after compilation. Select the backend "
                "before calling compile_for_inference()."
            )
        match backend:
            case TextAttentionBackend.DENSE:
                self._implementation = self._dense_implementation
            case TextAttentionBackend.JAGGED:
                self._implementation = self._jagged_implementation
            case _ as unreachable:
                raise RuntimeError(f"Unsupported text backend: {unreachable}")
        self._backend = backend

    def compile_for_inference(self, *, fullgraph: bool, dynamic: bool) -> None:
        """Compile only the selected text backend."""

        if self._compiled:
            raise RuntimeError("TextTransformerBlock is already compiled.")
        match self._backend:
            case TextAttentionBackend.DENSE:
                self._dense_implementation = torch.compile(
                    self._forward_dense,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
            case TextAttentionBackend.JAGGED:
                configure_jagged_compilation()
                self._jagged_implementation = torch.compile(
                    self._forward_jagged,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
                self._jagged_singleton_implementation = torch.compile(
                    self._forward_jagged_singleton,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
            case _ as unreachable:
                raise RuntimeError(f"Unsupported text backend: {unreachable}")
        self._compiled = True
        self.select_backend(self._backend)


class ImageTextCrossAttention(nn.Module):
    """Image-query cross-attention with dense and jagged text kernels."""

    def __init__(
        self,
        image_width: int,
        text_width: int,
        heads: int,
        head_dim: int,
        backend: TextAttentionBackend,
        operations,
    ) -> None:
        super().__init__()
        self.image_width = int(image_width)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.attention_width = self.heads * self.head_dim
        self.image_norm = RMSNorm(image_width)
        self.text_norm = RMSNorm(text_width)
        self.q_proj = operations.Linear(
            image_width, self.attention_width, bias=False
        )
        self.kv_proj = operations.Linear(
            text_width, 2 * self.attention_width, bias=False
        )
        self.out_proj = operations.Linear(
            self.attention_width, image_width, bias=False
        )
        self.q_norm_heads = RMSNorm(head_dim)
        self.k_norm_heads = RMSNorm(head_dim)
        self.attention_query_scale = 1.0
        self._dense_implementation: Callable[..., Tensor] = self._forward_dense
        self._jagged_implementation: Callable[..., Tensor] = self._forward_jagged
        self._implementation: Callable[..., Tensor]
        self._backend = backend
        self._compiled = False
        self.select_backend(backend)

    def forward(  # noqa: PLR0917 - frozen cross-attention tensor interface
        self,
        image: Tensor,
        text: Tensor,
        text_mask: Tensor,
        text_min_length: int,
        text_max_length: int,
        modulation: Tensor,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Dispatch to the eager or setup-time compiled implementation."""

        return self._implementation(
            image,
            text,
            text_mask,
            text_min_length,
            text_max_length,
            modulation,
            transformer_options,
        )

    def _forward_dense(  # noqa: PLR0917 - frozen cross-attention tensor interface
        self,
        image: Tensor,
        text: Tensor,
        text_mask: Tensor,
        text_min_length: int,
        text_max_length: int,
        modulation: Tensor,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Apply dense cross-attention with an explicit padding mask."""

        del text_min_length, text_max_length
        scale, gate = modulation.chunk(2, dim=-1)
        query_input = apply_scale(self.image_norm(image), scale)
        batch, image_length, _ = image.shape
        text_length = text.shape[1]
        q = self.q_proj(query_input).view(
            batch, image_length, self.heads, self.head_dim
        )
        k, v = self.kv_proj(self.text_norm(text)).chunk(2, dim=-1)
        k = k.view(batch, text_length, self.heads, self.head_dim)
        v = v.view(batch, text_length, self.heads, self.head_dim)
        q = self.q_norm_heads(q).transpose(1, 2)
        k = self.k_norm_heads(k).transpose(1, 2)
        v = v.transpose(1, 2)
        attended = optimized_attention(
            q, k, v, self.heads, mask=text_mask[:, None, None, :],
            skip_reshape=True, skip_output_reshape=True,
            transformer_options=transformer_options or {},
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, image_length, self.attention_width)
        )
        attended = self.out_proj(attended)
        return image + gated_residual(gate, attended)

    def _forward_jagged(  # noqa: PLR0917 - frozen cross-attention tensor interface
        self,
        image: Tensor,
        text: Tensor,
        text_mask: Tensor,
        text_min_length: int,
        text_max_length: int,
        modulation: Tensor,
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Apply NestedTensor SDPA cross-attention over packed text."""

        del text_mask, transformer_options
        scale, gate = modulation.chunk(2, dim=-1)
        query_input = apply_scale(self.image_norm(image), scale)
        batch, image_length, _ = image.shape
        query = self.q_proj(query_input)
        query = (
            query.view(batch, image_length, self.heads, self.head_dim)
            .contiguous()
            .view(batch * image_length, self.heads, self.head_dim)
        )
        nested = _jagged(text)
        text_values = nested.values().to(dtype=query.dtype)
        text_offsets = nested.offsets()
        key, value = self.kv_proj(self.text_norm(text_values)).chunk(2, dim=-1)
        query = self.q_norm_heads(query) * float(self.attention_query_scale)
        key = self.k_norm_heads(
            key.view(key.shape[0], self.heads, self.head_dim)
        ).contiguous()
        value = value.view(value.shape[0], self.heads, self.head_dim).contiguous()
        query_offsets = torch.arange(
            0,
            (batch + 1) * image_length,
            image_length,
            device=image.device,
            dtype=torch.int64,
        )
        query_nested = torch.nested.nested_tensor_from_jagged(
            query,
            query_offsets,
            jagged_dim=1,
            min_seqlen=image_length,
            max_seqlen=image_length,
        ).transpose(1, 2)
        key_nested = torch.nested.nested_tensor_from_jagged(
            key,
            text_offsets,
            jagged_dim=1,
            min_seqlen=text_min_length,
            max_seqlen=text_max_length,
        ).transpose(1, 2)
        value_nested = torch.nested.nested_tensor_from_jagged(
            value,
            text_offsets,
            jagged_dim=1,
            min_seqlen=text_min_length,
            max_seqlen=text_max_length,
        ).transpose(1, 2)
        attended_nested = F.scaled_dot_product_attention(
            query_nested,
            key_nested,
            value_nested,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        attended_values = _jagged(attended_nested).values()
        attended_values = attended_values.reshape(
            attended_values.shape[0],
            self.attention_width,
        )
        attended = self.out_proj(attended_values).view(
            batch,
            image_length,
            self.image_width,
        )
        return image + gated_residual(gate, attended)

    def select_backend(self, backend: TextAttentionBackend) -> None:
        """Select one compiled/eager backend without a hot-path branch."""

        if self._compiled and backend is not self._backend:
            raise RuntimeError(
                "Text backend cannot change after compilation. Select the backend "
                "before calling compile_for_inference()."
            )
        match backend:
            case TextAttentionBackend.DENSE:
                self._implementation = self._dense_implementation
            case TextAttentionBackend.JAGGED:
                self._implementation = self._jagged_implementation
            case _ as unreachable:
                raise RuntimeError(f"Unsupported text backend: {unreachable}")
        self._backend = backend

    def compile_for_inference(self, *, fullgraph: bool, dynamic: bool) -> None:
        """Compile only the selected text layout."""

        if self._compiled:
            raise RuntimeError("ImageTextCrossAttention is already compiled.")
        match self._backend:
            case TextAttentionBackend.DENSE:
                self._dense_implementation = torch.compile(
                    self._forward_dense,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
            case TextAttentionBackend.JAGGED:
                configure_jagged_compilation()
                self._jagged_implementation = torch.compile(
                    self._forward_jagged,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
            case _ as unreachable:
                raise RuntimeError(f"Unsupported text backend: {unreachable}")
        self._compiled = True
        self.select_backend(self._backend)


def axial_rope(
    height: int,
    width: int,
    head_dim: int,
    base: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build exact FP32 axial RoPE values without a BF16 round trip."""

    quarter = head_dim // 4
    exponents = (
        2.0
        * torch.arange(quarter, device=device, dtype=torch.float32)
        / float(head_dim // 2)
    )
    periods = torch.tensor(base, device=device, dtype=torch.float32) ** exponents
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    coordinates = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
    coordinates = coordinates.flatten(0, 1)
    angles = coordinates[:, :, None] / periods[None, None, :]
    angles = angles.repeat_interleave(2, dim=-1).flatten(1, 2)
    return torch.sin(angles), torch.cos(angles)


def normalized_position_features(
    height: int,
    width: int,
    feature_width: int,
    device: torch.device,
) -> Tensor:
    """Build Canter's normalized additive 2D sin/cos features."""

    denominator = float(max(height, width))
    rows = torch.arange(0.5, height + 0.5, device=device, dtype=torch.float32)
    cols = torch.arange(0.5, width + 0.5, device=device, dtype=torch.float32)
    rows = 2.0 * rows / denominator - 1.0
    cols = 2.0 * cols / denominator - 1.0
    coordinates = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
    coordinates = coordinates.flatten(0, 1)
    bands = feature_width // 4
    periods = torch.tensor(100.0, device=device, dtype=torch.float32) ** (
        torch.arange(bands, device=device, dtype=torch.float32) / float(bands)
    )
    angles = 2.0 * math.pi * coordinates[:, :, None] / periods[None, None, :]
    y_angles = angles[:, 0]
    x_angles = angles[:, 1]
    return torch.cat(
        (
            torch.sin(y_angles),
            torch.cos(y_angles),
            torch.sin(x_angles),
            torch.cos(x_angles),
        ),
        dim=-1,
    )
