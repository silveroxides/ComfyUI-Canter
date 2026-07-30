"""Standalone mixed DitBlock/FCDM diffusion autoencoder export."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from ..dit.mlp_types import MLPType
from ..dit.repa_projection import DinoTokenAlignmentHead

from .config import DinacAEConfig, DinacAEInferenceConfig
from .decoder import Decoder
from .encoder import Encoder, EncoderPosterior
from .precision import parameter_storage_dtype, validate_inference_storage
from .samplers import run_ddim, run_dpmpp_2m
from .vp_diffusion import get_schedule, make_initial_state, sample_noise


def _resolve_class_head_mlp_type(name: str) -> MLPType:
    """Return the token-head MLP enum for the serialized config value."""

    match str(name):
        case "gelu":
            return MLPType.GELU
        case "silu":
            return MLPType.SILU
        case "relu":
            return MLPType.RELU
        case _ as unreachable:
            raise ValueError(
                f"Unsupported class_head_mlp_type for DinacAE export: {unreachable!r}"
            )


class DinacAE(nn.Module):
    """Exported DINAC-AE wrapper with encode/decode/predict_class APIs."""

    def __init__(self, config: DinacAEConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "latent_norm_running_mean",
            torch.zeros((config.latent_channels,), dtype=torch.float32),
        )
        self.register_buffer(
            "latent_norm_running_var",
            torch.ones((config.latent_channels,), dtype=torch.float32),
        )
        self.encoder = Encoder(
            in_channels=int(config.in_channels),
            patch_size=int(config.patch_size),
            model_dim=int(config.model_dim),
            depth=int(config.encoder_depth),
            bottleneck_dim=int(config.bottleneck_dim),
            mlp_ratio=float(config.mlp_ratio),
            mlp_type=str(config.encoder_mlp_type),
            bottleneck_posterior_kind=str(config.bottleneck_posterior_kind),
            bottleneck_norm_mode=str(config.bottleneck_norm_mode),
        )
        self.decoder = Decoder(
            in_channels=int(config.in_channels),
            patch_size=int(config.patch_size),
            model_dim=int(config.model_dim),
            depth=int(config.decoder_depth),
            start_block_count=int(config.decoder_start_blocks),
            end_block_count=int(config.decoder_end_blocks),
            bottleneck_dim=int(config.bottleneck_dim),
            mlp_ratio=float(config.mlp_ratio),
            depthwise_kernel_size=int(config.depthwise_kernel_size),
            adaln_low_rank_rank=int(config.adaln_low_rank_rank),
        )
        self.dino_token_alignment_head = DinoTokenAlignmentHead(
            in_channels=int(config.bottleneck_dim),
            feature_dim=int(config.class_head_feature_dim),
            model_dim=int(config.class_head_model_dim),
            head_dim=int(config.class_head_head_dim),
            mlp_ratio=float(config.class_head_mlp_ratio),
            mlp_activation=_resolve_class_head_mlp_type(config.class_head_mlp_type),
            block_index=10_001,
            register_token_count=int(config.class_head_register_token_count),
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> DinacAE:
        """Move the model without rounding FP32 parameter or buffer islands."""

        preserved_parameters = {
            name: parameter.detach().float()
            for name, parameter in self.named_parameters()
            if parameter_storage_dtype(name, torch.bfloat16) is torch.float32
        }
        preserved_buffers = {
            name: buffer.detach().float()
            for name, buffer in self.named_buffers()
            if buffer.is_floating_point()
        }
        super()._apply(fn, recurse=recurse)
        for name, parameter in self.named_parameters():
            preserved = preserved_parameters.get(name)
            if preserved is not None:
                parameter.data = preserved.to(device=parameter.device)
        for name, buffer in self.named_buffers():
            preserved = preserved_buffers.get(name)
            if preserved is not None:
                buffer.data = preserved.to(device=buffer.device)
        return self

    @property
    def ordinary_weight_storage_dtype(self) -> torch.dtype:
        """Return the ordinary-weight storage dtype."""

        return self.decoder.patchify.proj.weight.dtype

    def validate_inference_storage(self) -> None:
        """Require the loaded model's complete mixed-precision policy."""

        device = self.decoder.patchify.proj.weight.device
        validate_inference_storage(
            self,
            base_dtype=self.ordinary_weight_storage_dtype,
            device=device,
        )

    def _latent_norm_stats(self) -> tuple[Tensor, Tensor]:
        """Return ``(mean, std)`` tensors for latent whitening."""

        mean = self.latent_norm_running_mean.view(1, -1, 1, 1)
        var = self.latent_norm_running_var.view(1, -1, 1, 1)
        std = torch.sqrt(
            var.to(torch.float32) + float(self.config.latent_running_stats_eps)
        )
        return mean.to(torch.float32), std

    def _require_image_size_divisible(self, height: int, width: int) -> None:
        """Require image dimensions compatible with the exported patch size."""

        patch = int(self.config.effective_patch_size)
        if int(height) % patch != 0 or int(width) % patch != 0:
            raise ValueError(
                f"Image height={height} and width={width} must be divisible by "
                f"effective_patch_size={patch}"
            )

    def whiten(self, latents: Tensor) -> Tensor:
        """Whiten raw latents using exported running stats."""

        z = latents.to(torch.float32)
        mean, std = self._latent_norm_stats()
        return (z - mean.to(device=z.device)) / std.to(device=z.device)

    def dewhiten(self, latents: Tensor) -> Tensor:
        """Undo latent whitening back to the raw decoder scale."""

        z = latents.to(torch.float32)
        mean, std = self._latent_norm_stats()
        return z * std.to(device=z.device) + mean.to(device=z.device)

    def encode(self, images: Tensor) -> Tensor:
        """Encode images to the exported whitened latent space."""

        self._require_image_size_divisible(
            height=int(images.shape[2]),
            width=int(images.shape[3]),
        )
        device = self.decoder.patchify.proj.weight.device
        compute_dtype = self.ordinary_weight_storage_dtype
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=compute_dtype is torch.bfloat16,
        ):
            latents = self.encoder(images.to(device=device, dtype=compute_dtype))
        return self.whiten(latents)

    def encode_posterior(self, images: Tensor) -> EncoderPosterior:
        """Encode images and return the raw posterior."""

        self._require_image_size_divisible(
            height=int(images.shape[2]),
            width=int(images.shape[3]),
        )
        device = self.decoder.patchify.proj.weight.device
        compute_dtype = self.ordinary_weight_storage_dtype
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=compute_dtype is torch.bfloat16,
        ):
            return self.encoder.encode_posterior(
                images.to(device=device, dtype=compute_dtype)
            )

    def predict_class(self, latents: Tensor) -> Tensor:
        """Predict the exported DINO class token from whitened latents."""

        device = self.decoder.patchify.proj.weight.device
        dewhitened = self.dewhiten(latents).to(
            device=device,
            dtype=torch.float32,
        )
        t_zero = torch.zeros(
            (int(latents.shape[0]),),
            device=device,
            dtype=torch.float32,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
        ):
            out = self.dino_token_alignment_head(
                dewhitened,
                t=t_zero,
            )
        return out.class_token.to(torch.float32)

    def decode(
        self,
        latents: Tensor,
        height: int,
        width: int,
        *,
        inference_config: DinacAEInferenceConfig | None = None,
    ) -> Tensor:
        """Decode exported whitened latents to images via VP diffusion."""

        cfg = (
            inference_config
            if inference_config is not None
            else DinacAEInferenceConfig()
        )
        self._require_image_size_divisible(height=int(height), width=int(width))
        batch = int(latents.shape[0])
        device = latents.device
        decoder_latents = self.dewhiten(latents).to(
            device=device,
            dtype=torch.float32,
        )
        noise = sample_noise(
            (batch, int(self.config.in_channels), int(height), int(width)),
            noise_std=float(self.config.pixel_noise_std),
            seed=cfg.seed,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        schedule = get_schedule(cfg.schedule, cfg.num_steps).to(device=device)
        strength = float(cfg.strength)
        if not 0.0 < strength <= 1.0:
            raise ValueError("DINAC decoder strength must lie in (0, 1]")
        schedule = schedule * strength
        initial_state = make_initial_state(
            noise=noise.to(device=device),
            t_start=schedule[0:1],
            logsnr_min=float(self.config.logsnr_min),
            logsnr_max=float(self.config.logsnr_max),
        )
        compute_dtype = self.ordinary_weight_storage_dtype
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=compute_dtype is torch.bfloat16,
        ):

            def _forward_fn(
                x_t: Tensor,
                t: Tensor,
                latents_in: Tensor,
                *,
                drop_middle_blocks: bool = False,
                mask_latent_tokens: bool = False,
            ) -> Tensor:
                _ = mask_latent_tokens
                return self.decoder(
                    x_t.to(dtype=torch.float32),
                    t,
                    latents_in.to(dtype=torch.float32),
                    drop_middle_blocks=bool(drop_middle_blocks),
                )

            match cfg.sampler:
                case "ddim":
                    sampler_fn = run_ddim
                case "dpmpp_2m":
                    sampler_fn = run_dpmpp_2m
                case _ as unreachable:
                    raise ValueError(f"Unsupported sampler: {unreachable!r}")
            pdg_mode = "path_drop" if bool(cfg.pdg) else "disabled"
            return sampler_fn(
                forward_fn=_forward_fn,
                initial_state=initial_state,
                schedule=schedule,
                latents=decoder_latents,
                logsnr_min=float(self.config.logsnr_min),
                logsnr_max=float(self.config.logsnr_max),
                pdg_mode=pdg_mode,
                pdg_strength=float(cfg.pdg_strength),
                device=device,
            )

    def reconstruct(
        self,
        images: Tensor,
        *,
        inference_config: DinacAEInferenceConfig | None = None,
    ) -> Tensor:
        """Encode then decode one image batch."""

        latents = self.encode(images)
        _batch, _channels, height, width = images.shape
        return self.decode(
            latents,
            height=int(height),
            width=int(width),
            inference_config=inference_config,
        )
