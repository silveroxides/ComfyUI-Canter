"""Frozen mixed-precision storage policy for DINAC-AE inference."""

from __future__ import annotations

import torch
from torch import nn


def _is_norm_parameter(name: str) -> bool:
    """Return whether a parameter belongs to a normalization module."""

    return any("norm" in component for component in name.split("."))


def _is_gate_parameter(name: str) -> bool:
    """Return whether a parameter is a learned residual or GRN gate."""

    return (
        name.endswith(".layer_scale")
        or name.endswith(".grn.gamma")
        or name.endswith(".grn.beta")
        or name.endswith("decoder.path_drop_mask_feature")
    )


def _is_decoder_output_parameter(name: str) -> bool:
    """Return whether a parameter belongs to the final pixel projection."""

    return ".decoder.out_proj." in f".{name}."


def parameter_storage_dtype(name: str, base_dtype: torch.dtype) -> torch.dtype:
    """Return storage dtype for one frozen DINAC-AE parameter."""

    if base_dtype is torch.float32:
        return torch.float32
    if base_dtype is not torch.bfloat16:
        raise ValueError(f"Unsupported DINAC-AE parameter storage dtype: {base_dtype}")
    preserve = (
        _is_norm_parameter(name)
        or _is_gate_parameter(name)
        or _is_decoder_output_parameter(name)
    )
    return torch.float32 if preserve else torch.bfloat16


def validate_inference_storage(
    module: nn.Module,
    *,
    base_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Require the exact mixed parameter and FP32-buffer inference policy."""

    def _device_matches(actual: torch.device) -> bool:
        """Return whether an actual device satisfies the requested device."""

        return actual.type == device.type and (
            device.index is None or actual.index == device.index
        )

    parameter_mismatches = tuple(
        name
        for name, parameter in module.named_parameters()
        if not _device_matches(parameter.device)
        or parameter.dtype is not parameter_storage_dtype(name, base_dtype)
    )
    if parameter_mismatches:
        raise RuntimeError(
            "DINAC-AE parameter storage does not match its inference policy: "
            f"{parameter_mismatches}"
        )
    buffer_mismatches = tuple(
        name
        for name, buffer in module.named_buffers()
        if not _device_matches(buffer.device)
        or (buffer.is_floating_point() and buffer.dtype is not torch.float32)
    )
    if buffer_mismatches:
        raise RuntimeError(
            "DINAC-AE floating inference buffers must retain FP32 storage: "
            f"{buffer_mismatches}"
        )
