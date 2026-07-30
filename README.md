# ComfyUI-Canter

Native ComfyUI integration for Canter v0001 and DINAC-AE-D2. It does not use
`CanterPipeline`, Diffusers, Transformers model wrappers, downloads, or ComfyUI
core patches.

## Models

Download these three official safetensors files. Rename them when saving so
the two upstream files named `model-00001-of-00001.safetensors` remain
unambiguous in ComfyUI.

| Model | Official file | Save as |
|---|---|---|
| Canter v0001 denoiser | [`model-00001-of-00001.safetensors`](https://huggingface.co/data-archetype/canter/blob/v0001/model-00001-of-00001.safetensors) | `ComfyUI/models/diffusion_models/canter_v0001.safetensors` |
| Bundled SmolLM2 subset | [`text_encoder/model-00001-of-00001.safetensors`](https://huggingface.co/data-archetype/canter/blob/v0001/text_encoder/model-00001-of-00001.safetensors) | `ComfyUI/models/text_encoders/canter_smol_lm2_360m.safetensors` |
| DINAC-AE-D2 | [`model.safetensors`](https://huggingface.co/data-archetype/dinac_ae_d2/blob/main/model.safetensors) | `ComfyUI/models/vae/dinac_ae_d2.safetensors` |

Restart ComfyUI or refresh its model list after adding the files. Select the
renamed files in `CanterModelLoader` and `CanterVAELoader`. No config files,
model indexes, tokenizer downloads, copied snapshots, or custom model paths
are required. The loaders validate checkpoint family, tensor count, shapes,
and storage dtype.

## Workflow

Connect `CanterModelLoader` to the stock **CLIP Text Encode** node. Feed the
conditioning and model into `CanterGuider`; connect the paired outputs from
`CanterSamplerScheduler` and the guider to **SamplerCustomAdvanced**. Start
with `EmptyCanterLatent`. Decode with the stock **VAEDecode** node for the
published one-step defaults or `CanterVAEDecodeAdvanced` for multi-step DINAC
decoding.

`CanterGuider` uses inclusive step windows. A stop value of `-1` means the
final step selected in `CanterSamplerScheduler`, so the default remains valid
when the step count changes. Standalone CFG is disabled by default. PDG uses
the published full-path, constant 2.5 configuration.

Advanced DINAC decoding defaults to one-step DDIM, the linear schedule,
strength 1, deterministic seed 0, and decoder PDG disabled. Enabling decoder
PDG exposes its upstream strength default of 2.0. The seed widget uses
ComfyUI's normal control-after-generate behavior, while API workflows remain
reproducible from the serialized integer seed.

Only `CanterSamplerScheduler` implements the raw-velocity contract. Generic
ComfyUI samplers interpret the model output differently and are unsupported.
DINAC uses global transformer attention; tiled encode/decode is deliberately
rejected because it is not numerically equivalent.

Production execution targets CUDA BF16 with published FP32 parameter/math
islands. The denoiser, text refiner, and DINAC weighted layers use ComfyUI
injected operations so Core controls staging, patching, casting, offload, and
Dynamic VRAM. The node does not call comfy-aimdo controls directly. FP32 is
retained for tests and compatible checkpoints.

## Validation status

The package includes deterministic coverage for schedules, all native
solvers, solver evaluation indices, callbacks, run-scoped randomness,
guidance windows, checkpoint structure, mixed precision, node defaults, and
Core-stageable weighted operations. Real native-versus-upstream image parity
is still a separate validation gate and is not claimed here.

The source repository also contains `tools/fit_latent_rgb.py`, a resumable
maintainer utility for fitting optional 128-channel DINAC preview factors.
It resolves the VAE through ComfyUI's standard `vae` registry and takes the
dataset root as an explicit command-line argument. It performs DINAC encoding
only and never runs the Canter denoiser. Dataset paths, manifests, contact
sheets, and fitting state are not stored in the repository.

## Licensing

Canter-derived portions are distributed under ModelGo
Attribution-ShareAlike 2.0; see `LICENSE` and `NOTICE`. SmolLM2 subset,
tokenizer assets, and DINAC-derived portions retain Apache-2.0 attribution;
see `LICENSE-APACHE-2.0` and `ATTRIBUTION.md`.
