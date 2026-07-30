from __future__ import annotations

from pathlib import Path

import torch

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .adapter import CanterModelAdapter
from .blocks import TextAttentionBackend
from .modeling_canter import CanterModel
from .precision import CanterComponent, parameter_storage_dtype
from .text import CanterCLIP, CanterTextRefiner, LiteralTokenizer, TextBundle

def validate_denoiser_state(state):
    required = {
        "input_projection.weight": (2048, 128),
        "output_projection.weight": (128, 2048),
        "mask_token": (1, 1, 2048),
        "unconditional_token": (1, 1024),
    }
    for key, shape in required.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"Not a Canter v0001 checkpoint: invalid or missing {key}")
    if len(state) != 652:
        raise ValueError(f"Incomplete Canter v0001 checkpoint: expected 652 tensors, found {len(state)}")
    unsupported = {value.dtype for value in state.values()} - {torch.bfloat16, torch.float32}
    if unsupported:
        raise TypeError(f"Canter supports BF16/FP32 checkpoints, found {sorted(map(str, unsupported))}")
    _validate_storage_policy(state, CanterComponent.MODEL)


def validate_text_state(state):
    if len(state) != 217 or tuple(state.get("embed_tokens.weight", ()).shape) != (49152, 960):
        raise ValueError("Not the Canter SmolLM2 24-layer text subset")
    unsupported = {value.dtype for value in state.values()} - {torch.bfloat16, torch.float32}
    if unsupported:
        raise TypeError(f"Canter text supports BF16/FP32 weights, found {sorted(map(str, unsupported))}")
    _validate_storage_policy(state, CanterComponent.TEXT_ENCODER)


def _validate_storage_policy(state, component):
    base = torch.bfloat16 if any(value.dtype is torch.bfloat16 for value in state.values()) else torch.float32
    mismatched = [
        key for key, value in state.items()
        if value.dtype is not parameter_storage_dtype(component, key, base)
    ]
    if mismatched:
        raise TypeError(
            f"Canter {component.value} mixed-precision policy mismatch: {mismatched[:8]}"
        )


def load_model_and_clip(denoiser_path: str, text_path: str, package_root: Path):
    denoiser_state = comfy.utils.load_torch_file(denoiser_path, safe_load=True)
    text_state = comfy.utils.load_torch_file(text_path, safe_load=True)
    validate_denoiser_state(denoiser_state)
    validate_text_state(text_state)

    with torch.device("meta"):
        model = CanterModel(TextAttentionBackend.DENSE)
    missing, unexpected = model.load_state_dict(denoiser_state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Canter state mismatch: missing={missing}, unexpected={unexpected}")

    refiner = CanterTextRefiner(model)
    del model.text_projection_low
    del model.text_projection_middle
    del model.text_projection_high
    del model.text_blocks
    del model.unconditional_token
    adapter = CanterModelAdapter(model)
    model_patcher = comfy.model_patcher.CoreModelPatcher(
        adapter, mm.get_torch_device(), mm.unet_offload_device()
    )

    operations = comfy.ops.manual_cast
    with torch.device("meta"):
        backbone_bundle = TextBundle(operations, refiner)
    bundle = backbone_bundle
    missing, unexpected = bundle.backbone.load_state_dict(text_state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Canter text state mismatch: missing={missing}, unexpected={unexpected}")
    bundle_patcher = comfy.model_patcher.CoreModelPatcher(
        bundle, mm.text_encoder_device(), mm.text_encoder_offload_device()
    )
    clip = CanterCLIP(bundle_patcher, LiteralTokenizer(package_root / "tokenizer.json"), bundle)
    return model_patcher, clip
