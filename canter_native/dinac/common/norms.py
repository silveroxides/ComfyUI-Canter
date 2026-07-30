from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "ChannelWiseRMSNorm",
    "GlobalRMSNorm",
    "GroupNormF32",
    "LayerNorm",
    "LayerNorm2d",
    "RMSNorm",
    "global_rms_norm",
    "row_norm",
]


_HALF_PRECISION_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)


def _cast_to_float32(x: Tensor) -> tuple[Tensor, torch.dtype]:
    """Return tensor cast to fp32 for compute along with the original dtype."""
    dtype = x.dtype
    if dtype in _HALF_PRECISION_DTYPES:
        return x.float(), dtype
    return x, dtype


def _restore_dtype(x: Tensor, dtype: torch.dtype) -> Tensor:
    return x if x.dtype == dtype else x.to(dtype)


class RMSNorm(nn.Module):
    """Thin wrapper around ``torch.nn.RMSNorm`` that preserves our API.

    - Keeps an ``_eps`` attribute used by tests.
    - Maps ``affine`` -> ``elementwise_affine``.
    - Delegates all compute to the native implementation.

    Notes on precision
    - PyTorch ≥ 2.8 computes RMSNorm reductions in ``opmath`` dtype
      (float32 for float16/bfloat16) internally, then restores the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self._eps: float = float(eps)
        self._impl: nn.RMSNorm = nn.RMSNorm(
            dim, eps=self._eps, elementwise_affine=affine
        )
        self._dim: int = int(dim)

    @property
    def weight(self) -> Tensor | None:  # expose for tests/compat
        return self._impl.weight

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Apply RMSNorm while avoiding dtype-mismatch warnings under AMP.

        When inputs are bfloat16/float16 under autocast and the stored affine
        weight is float32 (common when model weights remain FP32), PyTorch emits
        a warning about mismatched dtypes and disables the fused path.

        We pass a view of the weight cast to the input dtype into the functional
        RMSNorm to enable the fused implementation without changing the
        parameter storage dtype (which remains FP32 for stability).
        """
        # Prefer functional to control the weight dtype for the kernel
        w: Tensor | None = self._impl.weight
        w_cast = w.to(dtype=x.dtype) if w is not None else None
        # Bias is not present in RMSNorm; functional takes (input, shape, weight, eps)
        return F.rms_norm(x, (self._dim,), w_cast, self._eps)


class LayerNorm(nn.LayerNorm):
    """Thin wrapper over ``torch.nn.LayerNorm`` with an ``_eps`` attribute.

    Notes on precision
    - Native LayerNorm kernels accumulate statistics in ``opmath`` dtype
      (float32 for float16/bfloat16) before casting results back.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        shape: int | list[int]
        match normalized_shape:
            case int() as dim:
                shape = dim
            case _:
                shape = [int(v) for v in normalized_shape]
        super().__init__(shape, eps=eps, elementwise_affine=elementwise_affine)
        self._eps: float = float(eps)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Delegate to native LayerNorm
        return super().forward(x)


# Prefer numerically stable GroupNormF32 below.


class GroupNormF32(nn.GroupNorm):
    """Thin wrapper over ``torch.nn.GroupNorm`` with an ``_eps`` attribute.

    Notes on precision
    - Native GroupNorm uses ``opmath`` accumulation (float32 for
      float16/bfloat16) for statistics and fused scale/bias math; results
      are cast back to the input dtype.
    - Despite the class name, this wrapper does not force a cast; it
      delegates to the native implementation.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
        self._eps: float = float(eps)


class ChannelWiseRMSNorm(nn.Module):
    """Channel-wise RMSNorm for NCHW tensors (fast NCHW path).

    - Normalizes across channels per spatial position without reshaping, using
      a float32 reduction for numerical stability and keeping elementwise ops
      in input dtype for throughput.
    - Supports optional per-channel affine ``weight`` and ``bias``.
    """

    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.channels: int = int(channels)
        self._eps: float = float(eps)
        self.affine: bool = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.channels))
            self.bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if x.dim() < 2:
            return x
        C = x.size(1)
        if self.channels != C:
            raise ValueError(f"ChannelWiseRMSNorm expected C={self.channels}, got {C}")
        # Keep only the reductions in fp32; scale/apply in the input dtype.
        ms = torch.mean(torch.square(x), dim=1, keepdim=True, dtype=torch.float32)
        inv_rms = torch.rsqrt(ms + self._eps)  # float32
        y = x * inv_rms.to(dtype=x.dtype)
        if self.affine and self.weight is not None:
            shape = (1, -1) + (1,) * (x.dim() - 2)
            y = y * self.weight.view(shape).to(dtype=x.dtype)
            if self.bias is not None:
                y = y + self.bias.view(shape).to(dtype=x.dtype)
        return y


def global_rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Project each sample to unit RMS across all non-batch dimensions.

    This is equivalent to RMSNorm with ``normalized_shape=x.shape[1:]`` and no
    affine parameters. Delegating to the native functional keeps the fast fused
    CUDA path and the same opmath accumulation behavior as ``torch.nn.RMSNorm``.
    """

    if x.dim() < 2:
        return x
    normalized_shape = tuple(int(dim) for dim in x.shape[1:])
    return F.rms_norm(x, normalized_shape, None, eps)


class GlobalRMSNorm(nn.Module):
    """RMSNorm across all dims except batch — sphere projection for NCHW tensors.

    Unlike :class:`ChannelWiseRMSNorm` (which normalizes per spatial position
    over channels), this normalizes the *entire* feature volume jointly,
    projecting each sample onto a hypersphere.  No learnable parameters.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self._eps: float = float(eps)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return global_rms_norm(x, eps=self._eps)


class LayerNorm2d(nn.LayerNorm):
    """Channel-wise LayerNorm using native ``F.layer_norm`` on a reshaped view.

    - Normalizes over channels only for each spatial location (B, h, w).
    - Weight and bias follow the base class semantics (shape [C]).

    Notes on precision
    - ``F.layer_norm`` calls the native LayerNorm kernel which accumulates in
      ``opmath`` dtype (float32 for float16/bfloat16), then casts back.
    """

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if x.dim() < 3:
            return super().forward(x)
        B, C = x.shape[:2]
        spatial = x.shape[2:]
        x_view = x.permute(0, *range(2, x.dim()), 1).contiguous().view(-1, C)
        y = F.layer_norm(x_view, (C,), self.weight, self.bias, self.eps)
        y = y.view(B, *spatial, C).permute(0, x.dim() - 1, *range(1, x.dim() - 1))
        return y.contiguous()


def row_norm(W: Tensor, eps: float = 1e-6) -> Tensor:
    """Row-normalise weight matrices along the last dimension.

    Precision and performance
    - Accumulates the squared sum in float32 without materializing a full fp32
      copy of ``W`` via ``sum(..., dtype=torch.float32)``.
    - Uses ``rsqrt`` and clamps the inverse norm via ``clamp_max(1/eps)`` to
      match ``clamp_min(eps)`` on the denominator.
    - Scales in the input dtype for throughput; callers relying on exact
      float32 scaling should cast explicitly.
    """
    # Sum of squares in fp32 for stability
    ss = torch.sum(torch.square(W), dim=-1, keepdim=True, dtype=torch.float32)
    inv = torch.rsqrt(ss).clamp_max(1.0 / float(eps))  # float32
    return W * inv.to(dtype=W.dtype)
