from __future__ import annotations

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
from comfy.model_sampling import CONST

from .modeling_canter import CanterPath, PreparedText


class RawVelocitySampling(CONST, torch.nn.Module):
    sigma_min = 0.0
    sigma_max = 1.0

    def __init__(self):
        torch.nn.Module.__init__(self)

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return timestep

    def percent_to_sigma(self, percent):
        return 1.0 - percent

    def calculate_denoised(self, sigma, model_output, model_input):
        return model_input - sigma.view(-1, 1, 1, 1) * model_output

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        del max_denoise
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class CanterModelAdapter(comfy.model_base.BaseModel):
    def __init__(self, diffusion_model):
        torch.nn.Module.__init__(self)
        self.diffusion_model = diffusion_model
        self.model_sampling = RawVelocitySampling()
        self.latent_format = CanterLatentFormat()
        self.model_type = "canter_raw_velocity"
        self.manual_cast_dtype = None
        self.device = torch.device("cpu")
        self.current_patcher = None
        self.concat_keys = ()
        self.memory_usage_factor = 4.0
        self.memory_usage_factor_conds = ()
        self.memory_usage_shape_process = {}

    def get_dtype(self):
        return next(self.diffusion_model.parameters()).dtype

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            out["attention_mask"] = comfy.conds.CONDRegular(attention_mask)
        return out

    def _apply_model(
        self,
        x,
        sigma,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        del c_concat, control
        options = (transformer_options or {}).get("canter", {})
        mask = kwargs.get("attention_mask")
        if mask is None:
            mask = torch.ones(c_crossattn.shape[:2], device=c_crossattn.device, dtype=torch.bool)
        lengths = mask.sum(1)
        text = PreparedText(c_crossattn, mask, int(lengths.min()), int(lengths.max()))
        path = CanterPath(options.get("path", "full"))
        generator = None
        if path is CanterPath.THREE_QUARTER:
            generator = options.get("generator")
            if not isinstance(generator, torch.Generator):
                raise RuntimeError(
                    "Three-quarter SPRINT requires the run-scoped Canter generator"
                )
        compute_dtype = self.get_dtype()
        with torch.autocast(
            device_type=x.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype is torch.bfloat16,
        ):
            velocity = self.diffusion_model(
                x, sigma.float(), text, path=path,
                self_attention_gain=float(options.get("self_attention_gain", 0.0)),
                generator=generator,
                transformer_options=transformer_options or {},
            )
        return self.model_sampling.calculate_denoised(sigma, velocity.float(), x)


class CanterLatentFormat(comfy.latent_formats.LatentFormat):
    latent_channels = 128
    latent_dimensions = 2
    scale_factor = 1.0
    spacial_downscale_ratio = 16
