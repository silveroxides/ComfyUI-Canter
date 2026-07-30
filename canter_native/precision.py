"""Frozen parameter-storage precision policy for Canter inference."""

from __future__ import annotations

from enum import Enum

import torch


class CanterComponent(Enum):
    """Standalone weight components governed by the precision policy."""

    MODEL = "model"
    TEXT_ENCODER = "text_encoder"


_FP32_MODEL_PARAMETERS = frozenset(
    {
        "input_position.proj.weight",
        "output_projection.weight",
        "output_projection.bias",
        "unconditional_token",
    }
)
_FP32_TEXT_PARAMETERS = frozenset({"embed_tokens.weight"})
_FP32_SCALE_SUFFIXES = (
    "adaln_modulation_scale",
    "cross_modulation_scale",
)


def _is_norm_weight(name: str) -> bool:
    """Return whether a frozen parameter is a normalization affine weight."""

    return name.endswith(".weight") and any(
        "norm" in component for component in name.split(".")
    )


def parameter_storage_dtype(
    component: CanterComponent,
    name: str,
    base_dtype: torch.dtype,
) -> torch.dtype:
    """Return the exact storage dtype for one frozen parameter.

    Float32 releases store every parameter in float32. Bfloat16 releases retain
    the small parameter islands whose AMP consumers operate in float32.
    """

    if base_dtype is torch.float32:
        return torch.float32
    if base_dtype is not torch.bfloat16:
        raise ValueError(f"Unsupported Canter base storage dtype: {base_dtype}")
    match component:
        case CanterComponent.MODEL:
            preserve = (
                name in _FP32_MODEL_PARAMETERS
                or _is_norm_weight(name)
                or name.endswith(_FP32_SCALE_SUFFIXES)
            )
        case CanterComponent.TEXT_ENCODER:
            preserve = name in _FP32_TEXT_PARAMETERS or _is_norm_weight(name)
        case _ as unreachable:
            raise RuntimeError(f"Unsupported Canter component: {unreachable}")
    return torch.float32 if preserve else torch.bfloat16
