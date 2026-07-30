from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import comfy.ops
import comfy.model_base
import folder_paths
from safetensors import safe_open

from comfyui_canter.canter_native.blocks import TextAttentionBackend
from comfyui_canter.canter_native.adapter import CanterLatentFormat, CanterModelAdapter
from comfyui_canter.canter_native.guidance import (
    inside,
    pdg_scales,
    resolve_window,
    route,
)
from comfyui_canter.canter_native.modeling_canter import CanterModel
from comfyui_canter.canter_native.precision import CanterComponent, parameter_storage_dtype
from comfyui_canter.canter_native.schedules import build_schedule
from comfyui_canter.canter_native.solver_core import Solver, solve
from comfyui_canter.canter_native.solvers import sample_native
from comfyui_canter.canter_native.text import CanterTextBackbone
from comfyui_canter.canter_native.text import CanterTextRefiner
from comfyui_canter.canter_native.vae import CanterVAE, DINAC_CONFIG
from comfyui_canter.canter_native.dinac.dinac_ae.model import DinacAE
from comfyui_canter.canter_native.dinac.dinac_ae.adaln import (
    AdaLNScaleGateZeroLowRankDelta,
    AdaLNScaleGateZeroProjector,
)
from comfyui_canter.canter_native.dinac.dit.attention_blocks import (
    CrossAttentionCore,
    DitSelfAttentionCore,
)
from comfyui_canter.canter_native.dinac.dit.mlp import SimpleActivationMLP
from comfyui_canter.canter_native.dinac.dit.position_encoding import (
    DiTPositionEncoding,
)
from comfyui_canter.nodes import CanterGuider, CanterVAEDecodeAdvanced
from comfyui_canter.nodes import CanterCFGGuider

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
    assert resolve_window(0, -1, 7, "PDG") == (0, 6)
    assert resolve_window(2, 4, 7, "CFG") == (2, 4)
    for window in ((3, 2, 7), (0, 7, 7), (0, -2, 7)):
        try:
            resolve_window(*window, "CFG")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid guidance window accepted: {window}")


def test_pdg_curves():
    assert pdg_scales("constant", 2.5, 2.5, 3, 2) == (2.5, 2.5, 2.5)
    assert pdg_scales("linear", 1, 3, 3, 2) == (1.0, 2.0, 3.0)
    assert pdg_scales("power", 1, 3, 2, 2) == (1.0, 1.5, 3.0)


@pytest.mark.parametrize(
    ("mode", "step", "pdg_start", "expected", "branches"),
    (
        ("none", 0, 0, 37.0, (("positive", "full"), ("negative", "full"))),
        ("full", 0, 0, 22.0, (("positive", "full"), ("positive", "skip_middle"))),
        (
            "three_quarter",
            0,
            0,
            20.5,
            (("positive", "full"), ("positive", "three_quarter")),
        ),
        (
            "alternate_pdg_first",
            0,
            0,
            22.0,
            (("positive", "full"), ("positive", "skip_middle")),
        ),
        (
            "alternate_pdg_first",
            1,
            0,
            37.0,
            (("positive", "full"), ("negative", "full")),
        ),
        (
            "alternate_cfg_first",
            0,
            0,
            37.0,
            (("positive", "full"), ("negative", "full")),
        ),
        (
            "alternate_cfg_first",
            1,
            0,
            22.0,
            (("positive", "full"), ("positive", "skip_middle")),
        ),
        (
            "combined_cfg_pdg",
            0,
            0,
            43.0,
            (
                ("positive", "full"),
                ("negative", "full"),
                ("negative", "skip_middle"),
            ),
        ),
        (
            "pdg_with_alternating_cfg",
            0,
            0,
            28.0,
            (("positive", "full"), ("negative", "skip_middle")),
        ),
        (
            "pdg_with_alternating_cfg",
            1,
            0,
            43.0,
            (
                ("positive", "full"),
                ("negative", "full"),
                ("negative", "skip_middle"),
            ),
        ),
        (
            "cfg_to_pdg",
            0,
            2,
            37.0,
            (("positive", "full"), ("negative", "full")),
        ),
        (
            "cfg_to_pdg",
            2,
            2,
            22.0,
            (("positive", "full"), ("positive", "skip_middle")),
        ),
    ),
)
def test_guider_executes_every_guidance_mode(
    monkeypatch, mode, step, pdg_start, expected, branches
):
    calls = []
    values = {
        ("positive", "full"): 10.0,
        ("positive", "skip_middle"): 2.0,
        ("positive", "three_quarter"): 3.0,
        ("negative", "full"): 1.0,
        ("negative", "skip_middle"): -2.0,
    }

    def calc_cond_batch(inner_model, conditions, x, timestep, model_options):
        del inner_model, timestep
        options = model_options["transformer_options"]["canter"]
        key = (conditions[0], options["path"])
        calls.append((key, options["self_attention_gain"]))
        return (torch.full_like(x, values[key]),)

    monkeypatch.setattr("comfy.samplers.calc_cond_batch", calc_cond_batch)
    canter_options = {
        "schedule_index": step,
        "steps": 4,
        "cfg_enabled": True,
        "cfg_window": (0, 3),
        "pdg_mode": mode,
        "pdg_window": (pdg_start, 3),
        "pdg_curve": "constant",
        "pdg_noisy": 2.5,
        "pdg_clean": 2.5,
        "pdg_power": 3.0,
        "self_attention_gain": -0.03,
    }
    guider = SimpleNamespace(
        conds={"positive": "positive", "negative": "negative"},
        inner_model=object(),
        cfg=4.0,
    )
    result = CanterCFGGuider.predict_noise(
        guider,
        torch.zeros((1, 1, 1, 1)),
        torch.ones(1),
        {"transformer_options": {"canter": canter_options}},
    )
    assert result.item() == pytest.approx(expected)
    assert tuple(key for key, gain in calls) == branches
    assert calls[0][1] == -0.03
    assert all(gain == 0.0 for _, gain in calls[1:])


def test_guider_clamps_corrected_state_index_to_final_guidance_step(monkeypatch):
    paths = []

    def calc_cond_batch(inner_model, conditions, x, timestep, model_options):
        del inner_model, conditions, timestep
        paths.append(model_options["transformer_options"]["canter"]["path"])
        return (torch.ones_like(x),)

    monkeypatch.setattr("comfy.samplers.calc_cond_batch", calc_cond_batch)
    options = {
        "schedule_index": 4,
        "steps": 4,
        "cfg_enabled": False,
        "cfg_window": (0, 3),
        "pdg_mode": "full",
        "pdg_window": (3, 3),
        "pdg_curve": "constant",
        "pdg_noisy": 2.5,
        "pdg_clean": 2.5,
        "pdg_power": 3.0,
        "self_attention_gain": -0.03,
    }
    guider = SimpleNamespace(
        conds={"positive": "positive", "negative": "negative"},
        inner_model=object(),
        cfg=1.0,
    )
    CanterCFGGuider.predict_noise(
        guider,
        torch.zeros((1, 1, 1, 1)),
        torch.zeros(1),
        {"transformer_options": {"canter": options}},
    )
    assert paths == ["full", "skip_middle"]


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


def test_solver_indices_match_upstream_evaluation_contract():
    schedule = build_schedule("linear", 4, finite_start=True)
    initial = torch.ones((1, 1, 1, 1), dtype=torch.float32)

    for solver in Solver:
        indices = []

        def velocity(state, time, index):
            del time
            indices.append(index)
            return torch.zeros_like(state)

        solve(
            velocity,
            initial.clone(),
            schedule,
            solver=solver,
            generator=torch.Generator().manual_seed(2),
            euler_maruyama_multiplier=0.0,
            er_sde_noise_multiplier=0.0,
        )
        expected = (
            [0, 1, 2, 2, 3, 3, 4, 4]
            if solver is Solver.ABM2
            else [0, 1, 2, 3]
        )
        assert indices == expected


def test_native_sampler_passes_one_generator_and_current_callback_state():
    sigmas = build_schedule("linear", 3)
    initial = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    evaluations = []
    callbacks = []

    def model(state, time, **extra):
        options = extra["model_options"]["transformer_options"]["canter"]
        draw = torch.rand((), generator=options["generator"], device=state.device)
        evaluations.append((options["schedule_index"], id(options["generator"]), draw))
        return state - time.view(-1, 1, 1, 1)

    result = sample_native(
        model,
        initial.clone(),
        sigmas,
        extra_args={"model_options": {"transformer_options": {"canter": {}}}},
        callback=callbacks.append,
        solver="euler",
        seed=17,
    )
    assert [item[0] for item in evaluations] == [0, 1, 2]
    assert len({item[1] for item in evaluations}) == 1
    assert len({float(item[2]) for item in evaluations}) == len(evaluations)
    assert len(callbacks) == 3
    assert torch.equal(callbacks[-1]["x"], result)
    assert not torch.equal(callbacks[0]["x"], initial)
    assert callbacks[-1]["denoised"].shape == initial.shape

    repeated = []

    def repeated_model(state, time, **extra):
        generator = extra["model_options"]["transformer_options"]["canter"]["generator"]
        repeated.append(torch.rand((), generator=generator, device=state.device))
        return state - time.view(-1, 1, 1, 1)

    sample_native(
        repeated_model,
        initial.clone(),
        sigmas,
        extra_args={"model_options": {"transformer_options": {"canter": {}}}},
        solver="euler",
        seed=17,
    )
    assert torch.equal(
        torch.stack([item[2] for item in evaluations]), torch.stack(repeated)
    )


def _assert_checkpoint_coverage(path, module):
    with safe_open(str(path), framework="pt") as checkpoint:
        keys = set(checkpoint.keys())
        shapes = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in keys}
    state = module.state_dict()
    assert keys == set(state)
    assert not [key for key in keys if tuple(state[key].shape) != shapes[key]]


def _registered_checkpoint(category, name):
    path = folder_paths.get_full_path(category, name)
    if path is None:
        pytest.skip(f"{name} is not installed in ComfyUI's {category} registry")
    return Path(path)


def test_actual_checkpoint_headers_match_production_meta_graphs():
    denoiser_path = _registered_checkpoint(
        "diffusion_models", "canter_v0001.safetensors"
    )
    text_path = _registered_checkpoint(
        "text_encoders", "canter_smol_lm2_360m.safetensors"
    )
    with torch.device("meta"):
        denoiser = CanterModel(TextAttentionBackend.DENSE)
        text = CanterTextBackbone(comfy.ops.manual_cast)
    _assert_checkpoint_coverage(denoiser_path, denoiser)
    _assert_checkpoint_coverage(text_path, text)


def test_actual_checkpoint_storage_policy():
    paths = (
        (
            _registered_checkpoint(
                "diffusion_models", "canter_v0001.safetensors"
            ),
            CanterComponent.MODEL,
        ),
        (
            _registered_checkpoint(
                "text_encoders", "canter_smol_lm2_360m.safetensors"
            ),
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
        def forward(self, tokens, mask, minimum, maximum, transformer_options=None):
            del mask, minimum, maximum, transformer_options
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


def _assert_large_weighted_modules_are_core_stageable(module):
    failures = []
    for name, child in module.named_modules():
        weight = dict(child.named_parameters(recurse=False)).get("weight")
        if weight is None or weight.numel() * weight.element_size() <= 16 * 1024:
            continue
        if not hasattr(child, "comfy_cast_weights"):
            failures.append((name, type(child).__name__, tuple(weight.shape)))
    assert failures == []


def test_production_graphs_use_core_stageable_weighted_operations():
    with torch.device("meta"):
        canter = CanterModel(TextAttentionBackend.DENSE, comfy.ops.manual_cast)
        dinac = DinacAE(DINAC_CONFIG, comfy.ops.manual_cast)
    del dinac.dino_token_alignment_head
    _assert_large_weighted_modules_are_core_stageable(canter)
    _assert_large_weighted_modules_are_core_stageable(dinac)


def test_dinac_initializers_accept_core_lazy_operation_parameters():
    class LazyLinear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.comfy_need_lazy_init_bias = bias

    class LazyOperations:
        Linear = LazyLinear

    mlp = SimpleActivationMLP(
        8,
        16,
        activation=torch.nn.functional.gelu,
        activation_name="gelu",
        bias_up=False,
        bias_down=False,
        operations=LazyOperations,
    )
    attention = DitSelfAttentionCore(
        8,
        2,
        position_encoding=DiTPositionEncoding.NONE,
        operations=LazyOperations,
    )
    cross_attention = CrossAttentionCore(
        query_dim=8,
        context_dim=8,
        n_heads=2,
        head_dim=4,
        operations=LazyOperations,
    )
    projector = AdaLNScaleGateZeroProjector(8, 4, LazyOperations)
    delta = AdaLNScaleGateZeroLowRankDelta(
        d_model=8,
        d_cond=4,
        rank=2,
        operations=LazyOperations,
    )
    assert all(
        module.weight is None
        for module in (
            mlp.up,
            mlp.down,
            attention.qkv,
            attention.proj_out,
            cross_attention.kv_proj,
            cross_attention.out_proj,
            projector.proj,
            delta.down,
            delta.up,
        )
    )


def test_node_defaults_are_explicit_and_step_independent():
    guider = {item.id: item for item in CanterGuider.define_schema().inputs}
    assert guider["cfg_enabled"].default is False
    assert guider["cfg_stop"].default == -1
    assert guider["pdg_mode"].default == "full"
    assert guider["pdg_curve"].default == "constant"
    assert guider["pdg_stop"].default == -1

    decoder = {
        item.id: item for item in CanterVAEDecodeAdvanced.define_schema().inputs
    }
    assert decoder["seed"].default == 0
    assert decoder["sampler"].default == "ddim"
    assert decoder["schedule"].default == "linear"
    assert decoder["pdg_enabled"].default is False
    assert decoder["pdg_strength"].default == 2.0
    assert decoder["strength"].min > 0.0


def test_text_backbone_casts_fp32_embeddings_to_layer_compute_dtype():
    backbone = CanterTextBackbone.__new__(CanterTextBackbone)
    torch.nn.Module.__init__(backbone)

    class Embedding(torch.nn.Module):
        def forward(self, ids):
            return torch.ones((*ids.shape, 4), dtype=torch.float32)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SimpleNamespace(
                q_proj=SimpleNamespace(
                    weight=torch.empty((), dtype=torch.bfloat16)
                )
            )
            self.seen_dtype = None

        def forward(self, value, mask, transformer_options=None):
            del mask, transformer_options
            self.seen_dtype = value.dtype
            return value

    backbone.embed_tokens = Embedding()
    backbone.layers = torch.nn.ModuleList(Layer() for _ in range(24))
    taps = backbone(
        torch.ones((1, 3), dtype=torch.int64),
        torch.ones((1, 3), dtype=torch.bool),
    )
    assert all(layer.seen_dtype is torch.bfloat16 for layer in backbone.layers)
    assert all(tap.dtype is torch.bfloat16 for tap in taps)


def test_vae_converts_between_comfy_and_dinac_pixel_ranges(monkeypatch):
    class Model:
        def encode(self, pixels, transformer_options):
            self.encoded = pixels
            self.transformer_options = transformer_options
            return pixels[:, :1]

        def decode(self, samples, height, width, inference_config):
            del samples, height, width, inference_config
            return torch.tensor(
                [[[[-2.0, -1.0, 0.0, 1.0, 2.0]]]]
            )

    model = Model()
    patcher = SimpleNamespace(
        load_device=torch.device("cpu"),
        model_options={"transformer_options": {"fixture": True}},
    )
    vae = CanterVAE(model, patcher)
    monkeypatch.setattr(
        "comfyui_canter.canter_native.vae.mm.load_models_gpu",
        lambda models: None,
    )

    pixels = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    vae.encode(pixels)
    assert torch.equal(
        model.encoded,
        torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]]),
    )
    assert model.transformer_options == {"fixture": True}

    decoded = vae.decode(torch.zeros((1, 128, 1, 5)))
    assert torch.equal(
        decoded,
        torch.tensor([[[[0.0], [0.0], [0.5], [1.0], [1.0]]]]),
    )


def test_canonical_api_workflow_uses_stock_modular_interfaces():
    workflow = json.loads((ROOT / "examples" / "canter_api.json").read_text())
    node_types = {node["class_type"] for node in workflow.values()}
    assert {
        "CanterModelLoader",
        "CLIPTextEncode",
        "CanterGuider",
        "CanterSamplerScheduler",
        "SamplerCustomAdvanced",
        "CanterVAELoader",
        "VAEDecode",
        "CanterVAEDecodeAdvanced",
    } <= node_types
    guider = workflow["4"]["inputs"]
    assert guider["cfg_stop"] == -1
    assert guider["pdg_stop"] == -1
    assert guider["pdg_mode"] == "full"
    decoder = workflow["11"]["inputs"]
    assert decoder["pdg_enabled"] is False
    assert decoder["pdg_strength"] == 2.0
