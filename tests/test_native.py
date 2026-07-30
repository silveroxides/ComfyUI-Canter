from __future__ import annotations

from pathlib import Path

import torch
import comfy.ops
import comfy.model_base
from safetensors import safe_open

from comfyui_canter.canter_native.blocks import TextAttentionBackend
from comfyui_canter.canter_native.adapter import CanterLatentFormat, CanterModelAdapter
from comfyui_canter.canter_native.guidance import inside, pdg_scales, route
from comfyui_canter.canter_native.modeling_canter import CanterModel
from comfyui_canter.canter_native.precision import CanterComponent, parameter_storage_dtype
from comfyui_canter.canter_native.schedules import build_schedule
from comfyui_canter.canter_native.solver_core import Solver, solve
from comfyui_canter.canter_native.text import CanterTextBackbone
from comfyui_canter.canter_native.text import CanterTextRefiner

ROOT = Path(__file__).parents[1]


def test_schedules_are_float32_descending_and_shifted():
    for kind in ("linear", "beta"):
        schedule = build_schedule(kind, 12, 0.7)
        assert schedule.dtype == torch.float32
        assert schedule.shape == (13,)
        assert torch.all(schedule[:-1] > schedule[1:])
        assert schedule[0] == 1 and schedule[-1] == 0


def test_logsnr_solvers_get_finite_noisy_endpoint():
    schedule = build_schedule("beta", 10, finite_start=True)
    assert 0 < schedule[0] < 1
    assert schedule[-1] == 0


def test_guidance_windows_and_routes_are_inclusive():
    assert inside(2, 2, 4)
    assert inside(4, 2, 4)
    assert not inside(5, 2, 4)
    assert route("alternate_cfg_first", 0, True, True).combine == "cfg"
    assert route("alternate_cfg_first", 1, True, True).combine == "pdg"
    assert route("alternate_pdg_first", 0, True, True).combine == "pdg"


def test_pdg_curves():
    assert pdg_scales("constant", 2.5, 2.5, 3, 2) == (2.5, 2.5, 2.5)
    assert pdg_scales("linear", 1, 3, 3, 2) == (1.0, 2.0, 3.0)
    assert pdg_scales("power", 1, 3, 2, 2) == (1.0, 1.5, 3.0)


def test_all_solvers_shape_dtype_and_determinism():
    schedule = build_schedule("linear", 4, finite_start=True)
    initial = torch.ones((1, 2, 2, 2), dtype=torch.float32)

    def velocity(state, time, index):
        del time, index
        return state * 0.1

    for solver in Solver:
        def run():
            generator = torch.Generator().manual_seed(4)
            return solve(
                velocity, initial.clone(), schedule, solver=solver,
                generator=generator, euler_maruyama_multiplier=0.2,
                er_sde_noise_multiplier=0.2,
            )
        first, second = run(), run()
        assert first.shape == initial.shape
        assert first.dtype == torch.float32
        assert torch.equal(first, second)


def _assert_checkpoint_coverage(path, module):
    with safe_open(str(path), framework="pt") as checkpoint:
        keys = set(checkpoint.keys())
        shapes = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in keys}
    state = module.state_dict()
    assert keys == set(state)
    assert not [key for key in keys if tuple(state[key].shape) != shapes[key]]


def test_actual_checkpoint_headers_match_production_meta_graphs():
    snapshot = ROOT.parents[2] / "canter"
    with torch.device("meta"):
        denoiser = CanterModel(TextAttentionBackend.DENSE)
        text = CanterTextBackbone(comfy.ops.manual_cast)
    _assert_checkpoint_coverage(snapshot / "model-00001-of-00001.safetensors", denoiser)
    _assert_checkpoint_coverage(
        snapshot / "text_encoder" / "model-00001-of-00001.safetensors", text
    )


def test_actual_checkpoint_storage_policy():
    snapshot = ROOT.parents[2] / "canter"
    paths = (
        (snapshot / "model-00001-of-00001.safetensors", CanterComponent.MODEL),
        (
            snapshot / "text_encoder" / "model-00001-of-00001.safetensors",
            CanterComponent.TEXT_ENCODER,
        ),
    )
    names = {"BF16": torch.bfloat16, "F32": torch.float32}
    for path, component in paths:
        with safe_open(str(path), framework="pt") as checkpoint:
            actual = {key: names[checkpoint.get_slice(key).get_dtype()] for key in checkpoint.keys()}
        base = torch.bfloat16 if torch.bfloat16 in actual.values() else torch.float32
        assert not [
            key for key, dtype in actual.items()
            if dtype is not parameter_storage_dtype(component, key, base)
        ]


def test_refiner_mixed_precision_boundary_uses_compute_autocast():
    class Block(torch.nn.Module):
        def forward(self, tokens, mask, minimum, maximum):
            del mask, minimum, maximum
            return tokens

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_projection_low = torch.nn.Linear(4, 4, bias=False).bfloat16()
            self.text_projection_middle = torch.nn.Linear(4, 4, bias=False).bfloat16()
            self.text_projection_high = torch.nn.Linear(4, 4, bias=False).bfloat16()
            self.text_blocks = torch.nn.ModuleList([Block()])
            self.unconditional_token = torch.nn.Parameter(torch.empty(1, 4))

    refiner = CanterTextRefiner(Source())
    taps = [torch.ones((1, 2, 4), dtype=torch.float32) for _ in range(3)]
    mask = torch.ones((1, 2), dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = refiner.project_text_features(*taps, mask)
    assert result.tokens.shape == (1, 2, 4)
    assert result.tokens.dtype == torch.float32


def test_canter_latent_format_implements_core_contract():
    latent = CanterLatentFormat()
    assert latent.latent_channels == 128
    assert latent.latent_dimensions == 2
    assert latent.spacial_downscale_ratio == 16
    assert latent.temporal_downscale_ratio == 1
    assert latent.taesd_decoder_name is None
    assert latent.latent_rgb_factors is None
    assert latent.latent_rgb_factors_bias is None
    sample = torch.randn((1, 128, 2, 3))
    assert torch.equal(latent.process_in(sample), sample)
    assert torch.equal(latent.process_out(sample), sample)


def test_canter_adapter_is_core_base_model():
    with torch.device("meta"):
        denoiser = CanterModel(TextAttentionBackend.DENSE)
    adapter = CanterModelAdapter(denoiser)
    assert isinstance(adapter, comfy.model_base.BaseModel)
    assert adapter.extra_conds_shapes() == {}
    assert adapter.process_latent_in(torch.ones(1, 128, 1, 1)).shape == (1, 128, 1, 1)


def test_canter_adapter_apply_model_signature_matches_core():
    import inspect

    signature = inspect.signature(CanterModelAdapter._apply_model)
    assert list(signature.parameters)[:7] == [
        "self",
        "x",
        "sigma",
        "c_concat",
        "c_crossattn",
        "control",
        "transformer_options",
    ]
