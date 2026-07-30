# Attribution

The release bundles a tokenizer, token embedding, and the first 24 decoder layers
derived from
[`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M),
pinned by commit in each `release.json`.

SmolLM2 is published by Hugging Face under the Apache License 2.0. Its model
card identifies the SmolLM2 family and links its accompanying technical paper:

> SmolLM2: When Smol Goes Big - Data-Centric Training of a Small Language Model

The release retains only the subset required for prompt conditioning;
it does not ship the unused final backbone layers or language-model output
head.

Generated latents are designed for the separately versioned
[`data-archetype/dinac_ae_d2`](https://huggingface.co/data-archetype/dinac_ae_d2)
VAE. The loader uses the latest VAE repository revision and records the resolved
immutable commit in runtime metadata; the VAE's own
repository remains the authority for its code, model card, and license.

The original weights, architecture, model-specific code, and
documentation are distributed under the ModelGo Attribution-ShareAlike
License 2.0 (`MG-BY-SA-2.0`). See `LICENSE` and retain `NOTICE` when required
by that license.

The bundled SmolLM2 subset retains its original license. Its Apache
License 2.0 text is retained in `LICENSE-APACHE-2.0`.
