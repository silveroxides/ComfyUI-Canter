from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

import comfy.model_management as mm
from comfy.ldm.modules.attention import optimized_attention


class RMSNorm(torch.nn.Module):
    def __init__(self, width=960):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(width))

    def forward(self, x):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight, 1.0e-5)


class SmolAttention(torch.nn.Module):
    def __init__(self, operations, width=960, heads=15, kv_heads=5):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, width // heads
        self.q_proj = operations.Linear(width, heads * self.head_dim, bias=False)
        self.k_proj = operations.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v_proj = operations.Linear(width, kv_heads * self.head_dim, bias=False)
        self.o_proj = operations.Linear(width, width, bias=False)

    def forward(self, x, mask, transformer_options=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(length, device=x.device, dtype=torch.float32)
        inverse = 100000.0 ** (-torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim)
        angles = positions[:, None] * inverse[None]
        cos = torch.cat((angles.cos(), angles.cos()), -1)[None, None].to(q)
        sin = torch.cat((angles.sin(), angles.sin()), -1)[None, None].to(q)
        def rotate(t):
            half = t.shape[-1] // 2
            return torch.cat((-t[..., half:], t[..., :half]), -1)
        q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        valid = mask[:, None, None, :] & causal[None, None]
        bias = torch.zeros((batch, 1, length, length), device=x.device, dtype=x.dtype)
        bias.masked_fill_(~valid, torch.finfo(x.dtype).min)
        out = optimized_attention(
            q, k, v, self.heads, mask=bias, skip_reshape=True,
            transformer_options=transformer_options or {}, enable_gqa=True,
        )
        return self.o_proj(out)


class SmolMLP(torch.nn.Module):
    def __init__(self, operations, width=960, hidden=2560):
        super().__init__()
        self.gate_proj = operations.Linear(width, hidden, bias=False)
        self.up_proj = operations.Linear(width, hidden, bias=False)
        self.down_proj = operations.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SmolLayer(torch.nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.input_layernorm = RMSNorm()
        self.self_attn = SmolAttention(operations)
        self.post_attention_layernorm = RMSNorm()
        self.mlp = SmolMLP(operations)

    def forward(self, x, mask, transformer_options=None):
        x = x + self.self_attn(self.input_layernorm(x), mask, transformer_options)
        return x + self.mlp(self.post_attention_layernorm(x))


class CanterTextBackbone(torch.nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.embed_tokens = operations.Embedding(49152, 960)
        self.layers = torch.nn.ModuleList(SmolLayer(operations) for _ in range(24))

    def forward(self, ids, mask, transformer_options=None):
        x = self.embed_tokens(ids)
        taps = []
        for index, layer in enumerate(self.layers, 1):
            x = layer(x, mask, transformer_options)
            if index in (8, 16, 24):
                taps.append(x)
        return taps


class LiteralTokenizer:
    def __init__(self, path: Path):
        self.tokenizer = Tokenizer.from_file(str(path))
        self.eos_id = 0
        config_path = path.with_name("tokenizer_config.json")
        if config_path.exists():
            import json
            config = json.loads(config_path.read_text(encoding="utf-8"))
            eos = config.get("eos_token", "<|endoftext|>")
            self.eos_id = self.tokenizer.token_to_id(eos) or 0

    def encode(self, text: str):
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids[:512]
        return ids or [self.eos_id]

    def batch(self, texts):
        rows = [self.encode(text) for text in texts]
        length = max(map(len, rows))
        ids = torch.tensor([row + [self.eos_id] * (length - len(row)) for row in rows])
        mask = torch.tensor([[True] * len(row) + [False] * (length - len(row)) for row in rows])
        return ids, mask


class CanterCLIP:
    def __init__(self, patcher, tokenizer, model):
        self.patcher, self.tokenizer_impl, self.cond_stage_model = patcher, tokenizer, model

    def tokenize(self, text, **kwargs):
        del kwargs
        return [text]

    def encode_from_tokens_scheduled(self, tokens, **kwargs):
        del kwargs
        texts = list(tokens)
        mm.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        if all(not text.strip() for text in texts):
            mask = torch.ones((len(texts), 1), device=device, dtype=torch.bool)
            compute_dtype = self.cond_stage_model.compute_dtype
            with torch.autocast(
                device_type=device.type,
                dtype=compute_dtype,
                enabled=compute_dtype is torch.bfloat16,
            ):
                token = self.cond_stage_model.refine_unconditional(len(texts), device, mask)
            return [[token, {"attention_mask": mask}]]
        ids, mask = self.tokenizer_impl.batch(texts)
        compute_dtype = self.cond_stage_model.compute_dtype
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=compute_dtype is torch.bfloat16,
        ):
            taps = self.cond_stage_model.backbone(ids.to(device), mask.to(device))
            refined = self.cond_stage_model.refine(taps, mask.to(device))
        return [[refined, {"attention_mask": mask.to(device)}]]

    def clone(self):
        return CanterCLIP(self.patcher.clone(), self.tokenizer_impl, self.cond_stage_model)


class TextBundle(torch.nn.Module):
    def __init__(self, operations, refiner):
        super().__init__()
        self.backbone = CanterTextBackbone(operations)
        self.refiner = refiner
        self.unconditional_token = refiner.unconditional_token

    @property
    def compute_dtype(self):
        return self.backbone.layers[0].self_attn.q_proj.weight.dtype

    def refine(self, taps, mask):
        return self.refiner.project_text_features(taps[0], taps[1], taps[2], mask).tokens

    def refine_unconditional(self, batch, device, mask):
        tokens = self.unconditional_token.to(device).unsqueeze(0).expand(batch, -1, -1)
        for block in self.refiner.text_blocks:
            tokens = block(tokens, mask, 1, 1)
        return tokens


class CanterTextRefiner(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.text_projection_low = denoiser.text_projection_low
        self.text_projection_middle = denoiser.text_projection_middle
        self.text_projection_high = denoiser.text_projection_high
        self.text_blocks = denoiser.text_blocks
        self.unconditional_token = denoiser.unconditional_token

    def project_text_features(self, low, middle, high, mask):
        from .modeling_canter import PreparedText
        tokens = (
            self.text_projection_low(low).float()
            + self.text_projection_middle(middle).float()
            + self.text_projection_high(high).float()
        )
        lengths = mask.sum(1)
        for block in self.text_blocks:
            tokens = block(tokens, mask, int(lengths.min()), int(lengths.max()))
        return PreparedText(tokens, mask, int(lengths.min()), int(lengths.max()))
