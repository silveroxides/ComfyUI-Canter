"""Fail-fast runtime checks for the standalone Canter package."""

from __future__ import annotations

import sys
from enum import Enum

import torch

CANTER_AMP_DTYPE = torch.bfloat16
CANTER_DYNAMO_RECOMPILE_LIMIT = 4096


class OperatingSystem(Enum):
    """Operating systems supported by the Canter CUDA package."""

    LINUX = "linux"
    WINDOWS = "win32"


def configure_inference_runtime() -> None:
    """Apply Canter's frozen TF32 and compiler-cache policy.

    Canter uses TF32 for explicit float32 CUDA matrix multiplications and a
    4096-entry Dynamo recompile/cache limit. This must run before model
    construction or compilation.
    """

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    try:
        import torch._dynamo as dynamo
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch._dynamo is required to configure Canter inference."
        ) from exc
    try:
        dynamo.config.recompile_limit = CANTER_DYNAMO_RECOMPILE_LIMIT  # ty: ignore[invalid-assignment]
        dynamo.config.cache_size_limit = CANTER_DYNAMO_RECOMPILE_LIMIT
        dynamo.config.accumulated_cache_size_limit = max(
            int(dynamo.config.accumulated_cache_size_limit),
            CANTER_DYNAMO_RECOMPILE_LIMIT,
        )
    except AttributeError as exc:  # pragma: no cover
        raise RuntimeError(
            "The installed PyTorch version lacks the Dynamo cache settings "
            "required by Canter."
        ) from exc


def validate_common_runtime(device: torch.device, dtype: torch.dtype) -> None:
    """Validate the shared CUDA, OS, and inference-compute requirements."""

    try:
        OperatingSystem(sys.platform)
    except ValueError as exc:
        raise RuntimeError(
            "Canter supports Linux and Windows CUDA environments; "
            f"the current platform is {sys.platform!r}."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Canter inference requires a CUDA-enabled PyTorch installation and "
            "an available NVIDIA GPU."
        )
    if device.type != "cuda":
        raise RuntimeError(
            "Canter inference requires CUDA; move the model to a CUDA device "
            "before compiling it."
        )
    if dtype is not CANTER_AMP_DTYPE:
        raise RuntimeError(
            "Canter inference compute must use torch.bfloat16 AMP; "
            f"received {dtype}. Weight storage may be bfloat16 or float32."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The selected CUDA device does not support the bfloat16 compute "
            "required by Canter inference."
        )


def validate_jagged_runtime(device: torch.device, dtype: torch.dtype) -> None:
    """Validate the FlashAttention and NestedTensor jagged dependencies."""

    validate_common_runtime(device, dtype)
    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError(
            "This PyTorch CUDA build does not include FlashAttention. Install a "
            "supported CUDA PyTorch build or select the dense backend."
        )
    try:
        _ = torch.ops.aten._flash_attention_forward
    except AttributeError as exc:
        raise RuntimeError(
            "This PyTorch build lacks aten._flash_attention_forward, which the "
            "exact Canter jagged backend requires."
        ) from exc
    try:
        _ = torch.nested.nested_tensor_from_jagged
    except AttributeError as exc:
        raise RuntimeError(
            "This PyTorch build lacks jagged NestedTensor support, which the "
            "Canter jagged backend requires."
        ) from exc


def configure_jagged_compilation() -> None:
    """Enable dynamic-output tracing required by the jagged kernels."""

    try:
        import torch._dynamo as dynamo
    except ImportError as exc:
        raise RuntimeError(
            "torch._dynamo is required to compile the Canter jagged backend."
        ) from exc
    try:
        dynamo.config.capture_dynamic_output_shape_ops = True
    except AttributeError as exc:
        raise RuntimeError(
            "The installed PyTorch version lacks the Dynamo dynamic-output "
            "setting required by the Canter jagged backend."
        ) from exc
