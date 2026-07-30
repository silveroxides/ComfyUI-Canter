from __future__ import annotations

import torch

import comfy.model_management as mm
import comfy.model_patcher

from .dinac.dinac_ae.config import DinacAEConfig, DinacAEInferenceConfig
from .dinac.dinac_ae.model import DinacAE

DINAC_CONFIG = DinacAEConfig(
    in_channels=3,
    patch_size=16,
    model_dim=896,
    encoder_depth=8,
    decoder_depth=8,
    decoder_start_blocks=2,
    decoder_end_blocks=2,
    bottleneck_dim=128,
    mlp_ratio=4.0,
    encoder_mlp_type="gelu",
    depthwise_kernel_size=7,
    adaln_low_rank_rank=128,
    bottleneck_posterior_kind="diagonal_gaussian",
    bottleneck_norm_mode="disabled",
    logsnr_min=-10.0,
    logsnr_max=10.0,
    pixel_noise_std=0.558,
    latent_running_stats_eps=0.0001,
    class_head_feature_dim=768,
    class_head_model_dim=768,
    class_head_head_dim=64,
    class_head_mlp_ratio=4.0,
    class_head_mlp_type="gelu",
    class_head_register_token_count=4,
)
IGNORED_PREFIX = "dino_token_alignment_head."


class CanterVAE:
    def __init__(self, model, patcher):
        self.first_stage_model = model
        self.patcher = patcher
        self.downscale_ratio = 16
        self.upscale_ratio = 16
        self.latent_channels = 128

    def get_models(self):
        return [self.patcher]

    def encode(self, pixels):
        mm.load_models_gpu([self.patcher])
        pixels = pixels.movedim(-1, 1)
        pixels = pixels.to(self.patcher.load_device)
        return self.first_stage_model.encode(pixels).float()

    def _decode(self, samples, *, seed, steps, sampler, schedule, pdg_scale, strength=1.0):
        mm.load_models_gpu([self.patcher])
        samples = samples.to(self.patcher.load_device)
        height, width = int(samples.shape[-2]) * 16, int(samples.shape[-1]) * 16
        inference = DinacAEInferenceConfig(
            num_steps=int(steps),
            sampler=sampler,
            schedule=schedule,
            pdg=float(pdg_scale) != 1.0,
            pdg_strength=float(pdg_scale),
            strength=float(strength),
            seed=int(seed),
        )
        try:
            decoded = self.first_stage_model.decode(
                samples.float(), height, width, inference_config=inference
            )
            return decoded.movedim(1, -1)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "DINAC uses global attention and cannot be tiled. Reduce image size or batch."
            ) from error

    def decode(self, samples):
        return self._decode(
            samples, seed=0, steps=1, sampler="ddim",
            schedule="linear", pdg_scale=1.0,
        )

    def decode_advanced(self, samples, **settings):
        return self._decode(samples, **settings)

    def decode_tiled(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("DINAC global attention is not tile-equivalent; use ordinary VAEDecode")


def load_dinac(state):
    ignored = {key: state.pop(key) for key in tuple(state) if key.startswith(IGNORED_PREFIX)}
    with torch.device("meta"):
        model = DinacAE(DINAC_CONFIG)
    del model.dino_token_alignment_head
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Not a complete DINAC-AE-D2 checkpoint: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape={mismatched[:8]}"
        )
    unsupported = {value.dtype for value in state.values()} - {torch.bfloat16, torch.float32}
    if unsupported:
        raise TypeError(f"DINAC supports BF16/FP32 weights, found {sorted(map(str, unsupported))}")
    model.load_state_dict(state, strict=True, assign=True)
    model._canter_intentionally_ignored_keys = tuple(sorted(ignored))
    patcher = comfy.model_patcher.CoreModelPatcher(
        model, mm.vae_device(), mm.vae_offload_device()
    )
    return CanterVAE(model, patcher)
