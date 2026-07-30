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

Only `CanterSamplerScheduler` implements the raw-velocity contract. Generic
ComfyUI samplers interpret the model output differently and are unsupported.
DINAC uses global transformer attention; tiled encode/decode is deliberately
rejected because it is not numerically equivalent.

Production execution targets CUDA BF16 with published FP32 parameter/math
islands. FP32 is retained for tests and compatible checkpoints. No real model
inference or image-parity claim is made by this implementation.

## Licensing

Canter-derived portions are distributed under ModelGo
Attribution-ShareAlike 2.0; see `LICENSE` and `NOTICE`. SmolLM2 subset,
tokenizer assets, and DINAC-derived portions retain Apache-2.0 attribution;
see `LICENSE-APACHE-2.0` and `ATTRIBUTION.md`.
