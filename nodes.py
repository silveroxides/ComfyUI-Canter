from __future__ import annotations

from pathlib import Path

import torch
from typing_extensions import override

import comfy.samplers
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .canter_native.guidance import (
    PDG_MODES,
    inside,
    mode_uses_cfg,
    pdg_scales,
    resolve_window,
)
from .canter_native.loading import load_model_and_clip
from .canter_native.schedules import build_schedule
from .canter_native.solvers import (
    sample_native,
)
from .canter_native.vae import load_dinac

PACKAGE_ROOT = Path(__file__).parent
SOLVERS = ("euler", "euler_maruyama", "er_sde", "dpmpp_2m", "abm2")


class CanterModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CanterModelLoader", category="loaders/canter",
            inputs=[
                io.Combo.Input("diffusion_model", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("text_encoder", options=folder_paths.get_filename_list("text_encoders")),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, diffusion_model, text_encoder):
        denoiser = folder_paths.get_full_path_or_raise("diffusion_models", diffusion_model)
        text = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        return io.NodeOutput(*load_model_and_clip(denoiser, text, PACKAGE_ROOT))


class CanterVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CanterVAELoader", category="loaders/canter",
            inputs=[io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae"))],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        state = comfy.utils.load_torch_file(path, safe_load=True)
        return io.NodeOutput(load_dinac(state))


class EmptyCanterLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyCanterLatent", category="latent/canter",
            inputs=[
                io.Int.Input("width", default=832, min=16, max=16384, step=16),
                io.Int.Input("height", default=1216, min=16, max=16384, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, batch_size):
        if width % 16 or height % 16:
            raise ValueError("Canter width and height must be divisible by 16")
        samples = torch.zeros((batch_size, 128, height // 16, width // 16), dtype=torch.float32)
        return io.NodeOutput({"samples": samples})


class CanterCFGGuider(comfy.samplers.CFGGuider):
    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                     callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        if not getattr(sampler, "extra_options", {}).get("canter_native", False):
            raise ValueError(
                "Canter uses raw velocity. Select sampler and sigmas with CanterSamplerScheduler."
            )
        steps = len(sigmas) - 1
        options = self.model_options["transformer_options"]["canter"]
        options["cfg_window"] = resolve_window(
            *options["cfg_window"], steps, "CFG"
        )
        options["pdg_window"] = resolve_window(
            *options["pdg_window"], steps, "PDG"
        )
        options["steps"] = steps
        return super().outer_sample(
            noise, latent_image, sampler, sigmas, denoise_mask, callback,
            disable_pbar, seed, latent_shapes,
        )

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        del seed
        options = model_options["transformer_options"]["canter"]
        schedule_index = int(options["schedule_index"])
        steps = int(options["steps"])
        step = min(schedule_index, steps - 1)
        cfg_active = options["cfg_enabled"] and inside(step, *options["cfg_window"])
        pdg_active = options["pdg_mode"] != "none" and inside(
            step, *options["pdg_window"]
        )
        mode = options["pdg_mode"]
        if mode == "alternate_pdg_first" and pdg_active:
            cfg_active, pdg_active = step % 2 == 1, step % 2 == 0
        elif mode == "alternate_cfg_first" and pdg_active:
            cfg_active, pdg_active = step % 2 == 0, step % 2 == 1
        elif mode == "combined_cfg_pdg" and pdg_active:
            cfg_active = True
        elif mode == "pdg_with_alternating_cfg" and pdg_active:
            cfg_active = step % 2 == 1
        elif mode == "cfg_to_pdg":
            cfg_active = step < options["pdg_window"][0]
            pdg_active = inside(step, *options["pdg_window"])

        positive = self.conds["positive"]
        negative = self.conds.get("negative")

        def predict(role, path, gain):
            branch_options = comfy.model_patcher.create_model_options_clone(model_options)
            canter = branch_options.setdefault("transformer_options", {}).setdefault("canter", {})
            canter.update(path=path, self_attention_gain=gain)
            cond = positive if role == "positive" else negative
            return comfy.samplers.calc_cond_batch(
                self.inner_model, [cond], x, timestep, branch_options
            )[0]

        main = predict("positive", "full", options["self_attention_gain"])
        if not cfg_active and not pdg_active:
            return main
        scales = pdg_scales(
            options["pdg_curve"], options["pdg_noisy"], options["pdg_clean"],
            options["pdg_power"], steps,
        )
        weak_path = "three_quarter" if mode == "three_quarter" else "skip_middle"
        if cfg_active and pdg_active and mode in {"combined_cfg_pdg", "pdg_with_alternating_cfg"}:
            cfg_tweak = predict("negative", "full", 0.0)
            pdg_tweak = predict("negative", weak_path, 0.0)
            return main + 0.5 * (
                self.cfg * (main - cfg_tweak) + scales[step] * (main - pdg_tweak)
            )
        if pdg_active:
            tweak_role = (
                "negative"
                if mode in {"combined_cfg_pdg", "pdg_with_alternating_cfg"}
                else "positive"
            )
            tweak = predict(tweak_role, weak_path, 0.0)
            return tweak + scales[step] * (main - tweak)
        tweak = predict("negative", "full", 0.0)
        return tweak + self.cfg * (main - tweak)


class CanterGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CanterGuider", category="sampling/canter",
            inputs=[
                io.Model.Input("model"), io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative", optional=True),
                io.Boolean.Input("cfg_enabled", default=False),
                io.Float.Input("cfg_scale", default=1.0, min=0.0, max=100.0),
                io.Int.Input("cfg_start", default=0, min=0),
                io.Int.Input("cfg_stop", default=-1, min=-1),
                io.Combo.Input("pdg_mode", options=list(PDG_MODES), default="full"),
                io.Combo.Input(
                    "pdg_curve", options=["constant", "linear", "power"],
                    default="constant",
                ),
                io.Float.Input("pdg_noisy", default=2.5, min=0.0),
                io.Float.Input("pdg_clean", default=2.5, min=0.0),
                io.Float.Input("pdg_power", default=3.0, min=0.01),
                io.Int.Input("pdg_start", default=0, min=0),
                io.Int.Input("pdg_stop", default=-1, min=-1),
                io.Float.Input("self_attention_gain", default=-0.03, min=-10.0, max=10.0),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(cls, model, positive, cfg_enabled, cfg_scale, cfg_start, cfg_stop,
                pdg_mode, pdg_curve, pdg_noisy, pdg_clean, pdg_power, pdg_start,
                pdg_stop, self_attention_gain, negative=None):
        pdg_scales(pdg_curve, pdg_noisy, pdg_clean, pdg_power, 1)
        if (cfg_enabled or mode_uses_cfg(pdg_mode)) and negative is None:
            raise ValueError("Connect negative conditioning: the selected Canter mode can execute CFG")
        if negative is None:
            negative = positive
        guider = CanterCFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg_scale)
        guider.model_options = comfy.model_patcher.create_model_options_clone(guider.model_options)
        guider.model_options.setdefault("transformer_options", {})["canter"] = {
            "cfg_enabled": cfg_enabled,
            "cfg_window": (cfg_start, cfg_stop),
            "pdg_mode": pdg_mode,
            "pdg_curve": pdg_curve,
            "pdg_noisy": pdg_noisy, "pdg_clean": pdg_clean, "pdg_power": pdg_power,
            "pdg_window": (pdg_start, pdg_stop), "self_attention_gain": self_attention_gain,
        }
        return io.NodeOutput(guider)


class CanterSamplerScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CanterSamplerScheduler", category="sampling/canter",
            inputs=[
                io.Combo.Input("solver", options=list(SOLVERS), default="abm2"),
                io.Combo.Input("schedule", options=["linear", "beta"], default="beta"),
                io.Int.Input("steps", default=50, min=1, max=1000),
                io.Float.Input("log_snr_shift", default=0.0, min=-20.0, max=20.0),
                io.Float.Input("euler_maruyama_multiplier", default=1.0, min=0.0),
                io.Float.Input("er_sde_noise_multiplier", default=1.0, min=0.0),
            ],
            outputs=[io.Sampler.Output(), io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, solver, schedule, steps, log_snr_shift,
                euler_maruyama_multiplier, er_sde_noise_multiplier):
        finite = solver in {"dpmpp_2m", "er_sde"}
        sigmas = build_schedule(schedule, steps, log_snr_shift, finite)
        options = {"solver": solver, "canter_native": True}
        if solver == "euler_maruyama":
            options["noise_multiplier"] = euler_maruyama_multiplier
        if solver == "er_sde":
            options["noise_multiplier"] = er_sde_noise_multiplier
        return io.NodeOutput(comfy.samplers.KSAMPLER(sample_native, options), sigmas)


class CanterVAEDecodeAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CanterVAEDecodeAdvanced", category="latent/canter",
            inputs=[
                io.Vae.Input("vae"), io.Latent.Input("samples"),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff,
                    control_after_generate=True,
                ),
                io.Int.Input("steps", default=1, min=1, max=1000),
                io.Combo.Input("sampler", options=["ddim", "dpmpp_2m"], default="ddim"),
                io.Combo.Input("schedule", options=["linear", "cosine"], default="linear"),
                io.Boolean.Input("pdg_enabled", default=False),
                io.Float.Input("pdg_strength", default=2.0, min=0.0),
                io.Float.Input("strength", default=1.0, min=0.0001, max=1.0),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, vae, samples, seed, steps, sampler, schedule, pdg_enabled,
                pdg_strength, strength):
        if not 0.0 < float(strength) <= 1.0:
            raise ValueError("DINAC decoder strength must lie in (0, 1]")
        return io.NodeOutput(vae.decode_advanced(
            samples["samples"], seed=seed, steps=steps, sampler=sampler,
            schedule=schedule, pdg_enabled=pdg_enabled,
            pdg_strength=pdg_strength, strength=strength,
        ))


class CanterExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            CanterModelLoader, CanterVAELoader, EmptyCanterLatent,
            CanterGuider, CanterSamplerScheduler, CanterVAEDecodeAdvanced,
        ]


async def comfy_entrypoint():
    return CanterExtension()
