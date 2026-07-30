"""Frozen 3/24/3 Canter latent-flow denoiser."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import autocast

from .blocks import (
    DiTBlock,
    ImageTextCrossAttention,
    LowRankAdaLN,
    SharedAdaLN,
    TextAttentionBackend,
    TextTransformerBlock,
    TransitionDiTBlock,
    axial_rope,
    normalized_position_features,
    project_adaln,
)
from .configuration_canter import CANTER_CONFIG, CanterConfig
from .runtime import (
    CANTER_AMP_DTYPE,
    validate_common_runtime,
    validate_jagged_runtime,
)


class CanterPath(Enum):
    """The three concrete SPRINT middle paths supported by Canter."""

    FULL = "full"
    THREE_QUARTER = "three_quarter"
    SKIP_MIDDLE = "skip_middle"


@dataclass(frozen=True)
class PreparedText:
    """Text tokens refined once and reused across denoiser evaluations."""

    tokens: Tensor
    mask: Tensor
    min_length: int
    max_length: int


@dataclass(frozen=True)
class _ImageState:
    """Values shared by the three concrete model paths."""

    tokens: Tensor
    condition: Tensor
    rope: tuple[Tensor, Tensor]
    latents: Tensor
    time: Tensor
    height: int
    width: int


class TimeEmbedding(nn.Module):
    """Float32 sinusoidal embedding followed by the learned two-layer MLP."""

    def __init__(self, width: int, frequency_width: int, scale: float) -> None:
        super().__init__()
        self.frequency_width = int(frequency_width)
        self.scale = float(scale)
        self.proj_in = nn.Linear(self.frequency_width, width, bias=True)
        self.proj_out = nn.Linear(width, width, bias=True)

    def forward(self, time: Tensor) -> Tensor:
        """Embed a batch of continuous flow times."""

        half = self.frequency_width // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / float(half - 1)
        )
        angles = time.float().unsqueeze(1) * self.scale * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        hidden = self.proj_in(embedding.to(dtype=self.proj_in.weight.dtype))
        return self.proj_out(F.silu(hidden))


class InputPositionProjection(nn.Module):
    """FP32 projection of fixed normalized 2D sin/cos features."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.proj = nn.Linear(width, width, bias=False)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> InputPositionProjection:
        """Move the module while preserving the learned projection in FP32."""

        weight = self.proj.weight.detach().float()
        super()._apply(fn, recurse=recurse)
        self.proj.weight.data = weight.to(device=self.proj.weight.device)
        return self

    def forward(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return additive image-token position features."""

        features = normalized_position_features(
            height,
            width,
            self.width,
            device,
        )
        with autocast(device_type=device.type, enabled=False):
            projected = F.linear(features, self.proj.weight.float())
        return projected.to(dtype=dtype).unsqueeze(0).expand(batch, -1, -1)


class Float32Linear(nn.Linear):
    """Linear layer whose parameters remain FP32 across module dtype moves."""

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> Float32Linear:
        """Apply device moves while retaining unrounded FP32 parameters."""

        weight = self.weight.detach().float()
        bias = None if self.bias is None else self.bias.detach().float()
        super()._apply(fn, recurse=recurse)
        self.weight.data = weight.to(device=self.weight.device)
        if self.bias is not None and bias is not None:
            self.bias.data = bias.to(device=self.bias.device)
        return self


class ImageLayer(nn.Module):
    """One regular functional image layer and its AdaLN delta."""

    def __init__(self, config: CanterConfig) -> None:
        super().__init__()
        self.dit_block = DiTBlock(
            config.width,
            config.image_heads,
            config.mlp_ratio,
        )
        self.dit_delta = LowRankAdaLN(
            config.width,
            4 * config.width,
            config.adaln_rank,
        )

    def forward(  # noqa: PLR0917 - frozen image-layer tensor interface
        self,
        image: Tensor,
        hw: tuple[int, int],
        condition: Tensor,
        image_adaln: SharedAdaLN,
        rope: tuple[Tensor, Tensor],
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply the functional image layer."""

        modulation = project_adaln(image_adaln, self.dit_delta, condition)
        modulation = modulation.to(device=image.device, dtype=image.dtype)
        return self.dit_block(
            image,
            hw,
            condition,
            modulation,
            rope_sincos=rope,
            generator=generator,
        )

    def compile_for_inference(self, *, fullgraph: bool, dynamic: bool) -> None:
        """Compile the image block at setup time."""

        self.dit_block.compile_for_inference(fullgraph=fullgraph, dynamic=dynamic)


class BoundaryImageLayer(nn.Module):
    """Always-on text-conditioned prefix or suffix boundary layer."""

    def __init__(self, config: CanterConfig, backend: TextAttentionBackend) -> None:
        super().__init__()
        self.dit_block = TransitionDiTBlock(
            config.width,
            config.image_heads,
            config.mlp_ratio,
        )
        self.dit_delta = LowRankAdaLN(
            config.width,
            4 * config.width,
            config.adaln_rank,
        )
        self.cross_modulation_scale = nn.Parameter(torch.zeros(()))
        self.cross_attention = ImageTextCrossAttention(
            config.width,
            config.text_width,
            config.cross_heads,
            config.cross_head_dim,
            backend,
        )
        self.cross_delta = LowRankAdaLN(
            config.width,
            2 * config.width,
            config.adaln_rank,
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> BoundaryImageLayer:
        """Move the layer without rounding its FP32 cross-attention gate."""

        scale = self.cross_modulation_scale.detach().float()
        super()._apply(fn, recurse=recurse)
        self.cross_modulation_scale.data = scale.to(
            device=self.cross_modulation_scale.device
        )
        return self

    def forward(  # noqa: PLR0917 - frozen boundary-layer tensor interface
        self,
        image: Tensor,
        hw: tuple[int, int],
        condition: Tensor,
        text: PreparedText,
        image_adaln: SharedAdaLN,
        cross_adaln: SharedAdaLN,
        rope: tuple[Tensor, Tensor],
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply cross-attention followed by the transition DiT block."""

        cross_modulation = project_adaln(
            cross_adaln,
            self.cross_delta,
            condition,
        )
        cross_modulation = cross_modulation * self.cross_modulation_scale.to(
            dtype=cross_modulation.dtype
        )
        image = self.cross_attention(
            image,
            text.tokens,
            text.mask,
            text.min_length,
            text.max_length,
            cross_modulation,
        )
        dit_modulation = project_adaln(
            image_adaln,
            self.dit_delta,
            condition,
        )
        dit_modulation = dit_modulation.to(device=image.device, dtype=image.dtype)
        return self.dit_block(
            image,
            hw,
            condition,
            dit_modulation,
            rope_sincos=rope,
            generator=generator,
        )

    def select_text_backend(self, backend: TextAttentionBackend) -> None:
        """Select the boundary cross-attention backend."""

        self.cross_attention.select_backend(backend)

    def compile_for_inference(self, *, fullgraph: bool, dynamic: bool) -> None:
        """Compile the boundary image and active text kernels."""

        self.dit_block.compile_for_inference(
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self.cross_attention.compile_for_inference(
            fullgraph=False,
            dynamic=dynamic,
        )


class MiddleImageLayer(nn.Module):
    """One text-conditioned SPRINT-routable middle layer."""

    def __init__(self, config: CanterConfig, backend: TextAttentionBackend) -> None:
        super().__init__()
        self.cross_attention = ImageTextCrossAttention(
            config.width,
            config.text_width,
            config.cross_heads,
            config.cross_head_dim,
            backend,
        )
        self.cross_delta = LowRankAdaLN(
            config.width,
            2 * config.width,
            config.adaln_rank,
        )
        self.dit_block = DiTBlock(
            config.width,
            config.image_heads,
            config.mlp_ratio,
        )
        self.dit_delta = LowRankAdaLN(
            config.width,
            4 * config.width,
            config.adaln_rank,
        )

    def forward(  # noqa: PLR0917 - frozen middle-layer tensor interface
        self,
        image: Tensor,
        hw: tuple[int, int],
        condition: Tensor,
        text: PreparedText,
        image_adaln: SharedAdaLN,
        cross_adaln: SharedAdaLN,
        rope: tuple[Tensor, Tensor],
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply cross-attention and the corresponding image block."""

        cross_modulation = project_adaln(
            cross_adaln,
            self.cross_delta,
            condition,
        )
        image = self.cross_attention(
            image,
            text.tokens,
            text.mask,
            text.min_length,
            text.max_length,
            cross_modulation,
        )
        dit_modulation = project_adaln(
            image_adaln,
            self.dit_delta,
            condition,
        )
        dit_modulation = dit_modulation.to(device=image.device, dtype=image.dtype)
        return self.dit_block(
            image,
            hw,
            condition,
            dit_modulation,
            rope_sincos=rope,
            generator=generator,
        )

    def select_text_backend(self, backend: TextAttentionBackend) -> None:
        """Select the middle cross-attention backend."""

        self.cross_attention.select_backend(backend)

    def compile_for_inference(self, *, dynamic: bool) -> None:
        """Compile the regular middle image and active text kernels."""

        self.dit_block.compile_for_inference(fullgraph=False, dynamic=dynamic)
        self.cross_attention.compile_for_inference(
            fullgraph=False,
            dynamic=dynamic,
        )


def _sample_even_groupwise_indices(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Sample exactly one token from every non-overlapping 2x2 cell."""

    groups_h = height // 2
    groups_w = width // 2
    num_groups = groups_h * groups_w
    rows = torch.arange(groups_h, device=device, dtype=torch.int64)
    columns = torch.arange(groups_w, device=device, dtype=torch.int64)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    base = ((2 * grid_y) * width + (2 * grid_x)).reshape(num_groups, 1)
    offsets = torch.tensor(
        [0, 1, width, width + 1],
        device=device,
        dtype=torch.int64,
    ).view(1, 4)
    groups = base + offsets
    choice = torch.randint(
        0,
        4,
        (batch, num_groups),
        device=device,
        generator=generator,
    )
    selected = groups.unsqueeze(0).expand(batch, -1, -1)
    return selected.gather(2, choice.unsqueeze(-1)).squeeze(-1).sort(dim=-1).values


def _sample_odd_groupwise_indices(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Sample one latent per grid cell when either spatial edge is odd."""

    even_h = height - (height % 2)
    even_w = width - (width % 2)
    groups_h = even_h // 2
    groups_w = even_w // 2
    row_start = torch.zeros((batch,), device=device, dtype=torch.int64)
    column_start = torch.zeros((batch,), device=device, dtype=torch.int64)
    if height % 2:
        row_start = torch.randint(0, 2, (batch,), device=device, generator=generator)
    if width % 2:
        column_start = torch.randint(
            0,
            2,
            (batch,),
            device=device,
            generator=generator,
        )
    rows = torch.arange(groups_h, device=device, dtype=torch.int64)
    columns = torch.arange(groups_w, device=device, dtype=torch.int64)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    base = ((2 * grid_y) * width + (2 * grid_x)).reshape(1, -1)
    base = base + row_start.view(batch, 1) * width + column_start.view(batch, 1)
    offsets = torch.tensor(
        [0, 1, width, width + 1],
        device=device,
        dtype=torch.int64,
    ).view(1, 1, 4)
    groups = base.unsqueeze(-1) + offsets
    choice = torch.randint(
        0,
        4,
        (batch, groups.shape[1]),
        device=device,
        generator=generator,
    )
    interior = groups.gather(2, choice.unsqueeze(-1)).squeeze(-1)
    edges = _odd_edge_candidates(
        batch,
        height,
        width,
        row_start,
        column_start,
        device,
    )
    keep = edges.shape[1] // 2
    if keep == 0:
        return interior.sort(dim=-1).values
    scores = torch.rand(
        (batch, edges.shape[1]),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    edge_positions = torch.topk(scores, k=keep, dim=-1, largest=False).indices
    selected_edges = torch.gather(edges, 1, edge_positions)
    return torch.cat((interior, selected_edges), dim=1).sort(dim=-1).values


def _odd_edge_candidates(  # noqa: PLR0917 - explicit spatial sampling inputs
    batch: int,
    height: int,
    width: int,
    row_start: Tensor,
    column_start: Tensor,
    device: torch.device,
) -> Tensor:
    """Return the odd-edge candidate set for every sample."""

    candidates: list[Tensor] = []
    selected_row = torch.empty((batch,), device=device, dtype=torch.int64)
    if height % 2:
        selected_row = torch.where(
            row_start == 1,
            torch.zeros_like(row_start),
            torch.full_like(row_start, height - 1),
        )
        columns = torch.arange(width, device=device, dtype=torch.int64).view(1, width)
        candidates.append(selected_row.view(batch, 1) * width + columns)
    if width % 2:
        selected_column = torch.where(
            column_start == 1,
            torch.zeros_like(column_start),
            torch.full_like(column_start, width - 1),
        )
        if height % 2:
            rows = torch.arange(height - 1, device=device, dtype=torch.int64).view(
                1,
                height - 1,
            )
            rows = rows + (rows >= selected_row.view(batch, 1)).to(torch.int64)
        else:
            rows = torch.arange(height, device=device, dtype=torch.int64).view(
                1,
                height,
            )
        candidates.append(rows * width + selected_column.view(batch, 1))
    if not candidates:
        raise RuntimeError(
            "Odd-edge sampling requires at least one odd grid dimension."
        )
    return torch.cat(candidates, dim=1)


def sample_three_quarter_indices(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Sample Canter's exact 75% group-wise SPRINT keep pattern."""

    if batch <= 0 or height < 2 or width < 2:
        raise ValueError(
            "Three-quarter SPRINT requires a positive batch and a grid of at "
            "least 2x2 tokens."
        )
    if height % 2 == 0 and width % 2 == 0:
        return _sample_even_groupwise_indices(
            batch,
            height,
            width,
            device,
            generator,
        )
    return _sample_odd_groupwise_indices(
        batch,
        height,
        width,
        device,
        generator,
    )


class CanterModel(nn.Module):
    """The frozen 2B Canter denoiser represented directly as 3/24/3 layers."""

    config: CanterConfig

    def __init__(
        self,
        text_backend: TextAttentionBackend = TextAttentionBackend.JAGGED,
    ) -> None:
        super().__init__()
        config = CANTER_CONFIG
        self.config = config
        self.input_projection = nn.Linear(
            config.latent_channels,
            config.width,
            bias=True,
        )
        self.input_position = InputPositionProjection(config.width)
        self.time_embedding = TimeEmbedding(
            config.width,
            config.time_frequency_width,
            config.time_scale,
        )
        self.image_adaln = SharedAdaLN(config.width, 4 * config.width)
        self.cross_adaln = SharedAdaLN(config.width, 2 * config.width)
        self.prefix_layers = nn.ModuleList(
            (
                ImageLayer(config),
                ImageLayer(config),
                BoundaryImageLayer(config, text_backend),
            )
        )
        self.middle_layers = nn.ModuleList(
            MiddleImageLayer(config, text_backend) for _ in range(config.middle_depth)
        )
        self.suffix_layers = nn.ModuleList(
            (
                BoundaryImageLayer(config, text_backend),
                ImageLayer(config),
                ImageLayer(config),
            )
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.width))
        self.fusion = nn.Linear(2 * config.width, config.width, bias=True)
        self.tail_scale = nn.Linear(config.width, 2 * config.width, bias=True)
        self.text_blocks = nn.ModuleList(
            TextTransformerBlock(
                config.text_width,
                config.text_heads,
                config.mlp_ratio,
                config.text_rope_theta,
                text_backend,
            )
            for _ in range(config.text_refine_depth)
        )
        self.output_projection = Float32Linear(
            config.width,
            config.latent_channels,
            bias=True,
        )
        self.unconditional_token = nn.Parameter(torch.zeros(1, config.text_width))
        self.text_projection_low = nn.Linear(
            config.text_backbone_width,
            config.text_width,
            bias=False,
        )
        self.text_projection_middle = nn.Linear(
            config.text_backbone_width,
            config.text_width,
            bias=False,
        )
        self.text_projection_high = nn.Linear(
            config.text_backbone_width,
            config.text_width,
            bias=False,
        )
        self._text_backend = text_backend
        self._compiled = False

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CanterModel:
        """Move the model without rounding its FP32 unconditional token."""

        token = self.unconditional_token.detach().float()
        super()._apply(fn, recurse=recurse)
        self.unconditional_token.data = token.to(device=self.unconditional_token.device)
        return self

    def compile_for_inference(
        self,
        *,
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> None:
        """Compile every block while compiling only the active text backend."""

        if self._compiled:
            raise RuntimeError("CanterModel is already compiled.")
        device = self.input_projection.weight.device
        match self._text_backend:
            case TextAttentionBackend.DENSE:
                validate_common_runtime(device, CANTER_AMP_DTYPE)
            case TextAttentionBackend.JAGGED:
                validate_jagged_runtime(device, CANTER_AMP_DTYPE)
            case _ as unreachable:
                raise RuntimeError(f"Unsupported text backend: {unreachable}")
        for module in self.prefix_layers:
            match module:
                case ImageLayer() as layer:
                    layer.compile_for_inference(
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                case BoundaryImageLayer() as layer:
                    layer.compile_for_inference(
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                case _ as unreachable:
                    raise RuntimeError(f"Unsupported prefix layer: {unreachable}")
        for module in self.middle_layers:
            layer = cast("MiddleImageLayer", module)
            layer.compile_for_inference(dynamic=dynamic)
        for module in self.suffix_layers:
            match module:
                case ImageLayer() as layer:
                    layer.compile_for_inference(
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                case BoundaryImageLayer() as layer:
                    layer.compile_for_inference(
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                case _ as unreachable:
                    raise RuntimeError(f"Unsupported suffix layer: {unreachable}")
        for module in self.text_blocks:
            block = cast("TextTransformerBlock", module)
            block.compile_for_inference(fullgraph=False, dynamic=dynamic)
        self._compiled = True

    def select_text_backend(self, backend: TextAttentionBackend) -> None:
        """Select the dense or jagged layout before compilation."""

        if self._compiled and backend is not self._text_backend:
            raise RuntimeError(
                "Text backend cannot change after model compilation. Select it "
                "before calling compile_for_inference()."
            )
        for module in self.text_blocks:
            cast("TextTransformerBlock", module).select_backend(backend)
        for module in self.middle_layers:
            cast("MiddleImageLayer", module).select_text_backend(backend)
        cast("BoundaryImageLayer", self.prefix_layers[2]).select_text_backend(backend)
        cast("BoundaryImageLayer", self.suffix_layers[0]).select_text_backend(backend)
        self._text_backend = backend

    def project_text_features(
        self,
        low: Tensor,
        middle: Tensor,
        high: Tensor,
        mask: Tensor,
    ) -> PreparedText:
        """Sum the three SmolLM2 taps and run four refinement blocks."""

        lengths = mask.sum(dim=1, dtype=torch.int64)
        min_length = int(lengths.amin().item())
        max_length = int(lengths.amax().item())
        if min_length <= 0:
            raise ValueError("Every prompt must contain at least one text token.")
        match self._text_backend:
            case TextAttentionBackend.DENSE:
                tokens = (
                    self.text_projection_low(low).to(dtype=torch.float32)
                    + self.text_projection_middle(middle).to(dtype=torch.float32)
                    + self.text_projection_high(high).to(dtype=torch.float32)
                )
            case TextAttentionBackend.JAGGED:
                values = (
                    self.text_projection_low(low[mask]).to(dtype=torch.float32)
                    + self.text_projection_middle(middle[mask]).to(dtype=torch.float32)
                    + self.text_projection_high(high[mask]).to(dtype=torch.float32)
                )
                offsets = torch.zeros(
                    lengths.shape[0] + 1,
                    device=values.device,
                    dtype=torch.int64,
                )
                offsets[1:] = torch.cumsum(lengths, dim=0)
                tokens = torch.nested.nested_tensor_from_jagged(
                    values,
                    offsets,
                    jagged_dim=1,
                    min_seqlen=min_length,
                    max_seqlen=max_length,
                )
            case _ as unreachable:
                raise RuntimeError(f"Unsupported text backend: {unreachable}")
        for block in self.text_blocks:
            tokens = block(tokens, mask, min_length, max_length)
        return PreparedText(tokens, mask, min_length, max_length)

    def unconditional_text(self, batch: int, device: torch.device) -> PreparedText:
        """Return the refined learned one-token unconditional prompt."""

        token = self.unconditional_token.to(device=device)
        mask = torch.ones((batch, 1), device=device, dtype=torch.bool)
        match self._text_backend:
            case TextAttentionBackend.DENSE:
                tokens = token.unsqueeze(0).expand(batch, 1, -1)
            case TextAttentionBackend.JAGGED:
                values = token.expand(batch, -1).contiguous()
                offsets = torch.arange(
                    batch + 1,
                    device=device,
                    dtype=torch.int64,
                )
                tokens = torch.nested.nested_tensor_from_jagged(
                    values,
                    offsets,
                    jagged_dim=1,
                    min_seqlen=1,
                    max_seqlen=1,
                )
            case _ as unreachable:
                raise RuntimeError(f"Unsupported text backend: {unreachable}")
        for module in self.text_blocks:
            block = cast("TextTransformerBlock", module)
            match self._text_backend:
                case TextAttentionBackend.DENSE:
                    tokens = block(tokens, mask, 1, 1)
                case TextAttentionBackend.JAGGED:
                    tokens = block.forward_jagged_singleton(tokens)
                case _ as unreachable:
                    raise RuntimeError(f"Unsupported text backend: {unreachable}")
        return PreparedText(tokens, mask, 1, 1)

    def forward(
        self,
        latents: Tensor,
        time: Tensor,
        text: PreparedText,
        *,
        path: CanterPath = CanterPath.FULL,
        self_attention_gain: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Dispatch to one of the three concrete model paths."""
        try:
            match path:
                case CanterPath.FULL:
                    return self.forward_full(
                        latents,
                        time,
                        text,
                        self_attention_gain=self_attention_gain,
                        generator=generator,
                    )
                case CanterPath.THREE_QUARTER:
                    if generator is None:
                        raise ValueError(
                            "Three-quarter SPRINT requires an explicit torch.Generator."
                        )
                    return self.forward_three_quarter(
                        latents,
                        time,
                        text,
                        generator=generator,
                        self_attention_gain=self_attention_gain,
                    )
                case CanterPath.SKIP_MIDDLE:
                    return self.forward_skip_middle(
                        latents,
                        time,
                        text,
                        self_attention_gain=self_attention_gain,
                        generator=generator,
                    )
                case _ as unreachable:
                    raise RuntimeError(f"Unsupported Canter path: {unreachable}")
        finally:
            self._set_image_attention_gain(0.0)

    def forward_full(
        self,
        latents: Tensor,
        time: Tensor,
        text: PreparedText,
        *,
        self_attention_gain: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Predict velocity with all 24 middle layers and all image tokens."""

        self._set_image_attention_gain(self_attention_gain)
        state = self._prepare_image(latents, time)
        prefix = self._run_prefix(state.tokens, state, text, generator)
        middle = self._run_middle(prefix, state, text, state.rope, generator)
        return self._finish(prefix, middle, state, text, generator)

    def forward_three_quarter(
        self,
        latents: Tensor,
        time: Tensor,
        text: PreparedText,
        *,
        generator: torch.Generator,
        self_attention_gain: float = 0.0,
    ) -> Tensor:
        """Predict velocity with one image token per 2x2 SPRINT cell."""

        self._set_image_attention_gain(self_attention_gain)
        state = self._prepare_image(latents, time)
        prefix = self._run_prefix(state.tokens, state, text, generator)
        keep = sample_three_quarter_indices(
            prefix.shape[0],
            state.height,
            state.width,
            prefix.device,
            generator,
        )
        index = keep.unsqueeze(-1).expand(-1, -1, prefix.shape[-1])
        sparse = torch.gather(prefix, 1, index)
        sparse_rope = self._gather_rope(state.rope, keep)
        sparse = self._run_middle(sparse, state, text, sparse_rope, generator)
        masked = self.mask_token.to(dtype=prefix.dtype).expand_as(prefix).clone()
        masked.scatter_(1, index, sparse)
        return self._finish(prefix, masked, state, text, generator)

    def forward_skip_middle(
        self,
        latents: Tensor,
        time: Tensor,
        text: PreparedText,
        *,
        self_attention_gain: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Predict velocity while replacing the complete middle path by masks."""

        self._set_image_attention_gain(self_attention_gain)
        state = self._prepare_image(latents, time)
        prefix = self._run_prefix(state.tokens, state, text, generator)
        masked = self.mask_token.to(dtype=prefix.dtype).expand_as(prefix)
        return self._finish(prefix, masked, state, text, generator)

    @staticmethod
    def _set_layer_attention_scale(
        layer: ImageLayer | BoundaryImageLayer | MiddleImageLayer,
        scale: float,
    ) -> None:
        """Set one frozen image layer's self-attention query multiplier."""

        layer.dit_block.attention.attention_query_scale = float(scale)

    def _set_image_attention_gain(self, gain: float) -> None:
        """Apply the inference gain only to image self-attention modules."""

        scale = math.exp(float(gain))
        for module in self.prefix_layers:
            match module:
                case ImageLayer() | BoundaryImageLayer() as layer:
                    self._set_layer_attention_scale(layer, scale)
                case _ as unreachable:
                    raise RuntimeError(f"Unsupported prefix layer: {unreachable}")
        for module in self.middle_layers:
            self._set_layer_attention_scale(cast("MiddleImageLayer", module), scale)
        for module in self.suffix_layers:
            match module:
                case ImageLayer() | BoundaryImageLayer() as layer:
                    self._set_layer_attention_scale(layer, scale)
                case _ as unreachable:
                    raise RuntimeError(f"Unsupported suffix layer: {unreachable}")

    def _prepare_image(
        self,
        latents: Tensor,
        time: Tensor,
    ) -> _ImageState:
        """Project latents and construct shared time and position values."""

        batch, _, height, width = latents.shape
        tokens = (
            latents.permute(0, 2, 3, 1)
            .contiguous()
            .view(
                batch,
                height * width,
                self.config.latent_channels,
            )
        )
        tokens = self.input_projection(tokens)
        tokens = tokens + self.input_position(
            batch,
            height,
            width,
            tokens.device,
            tokens.dtype,
        )
        condition = self.time_embedding(time)
        rope = axial_rope(
            height,
            width,
            self.config.width // self.config.image_heads,
            self.config.rope_base,
            tokens.device,
        )
        return _ImageState(
            tokens=tokens,
            condition=condition,
            rope=rope,
            latents=latents,
            time=time,
            height=height,
            width=width,
        )

    def _run_prefix(
        self,
        tokens: Tensor,
        state: _ImageState,
        text: PreparedText,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Run the three public prefix layers."""

        first = cast("ImageLayer", self.prefix_layers[0])
        second = cast("ImageLayer", self.prefix_layers[1])
        boundary = cast("BoundaryImageLayer", self.prefix_layers[2])
        tokens = first(
            tokens,
            (state.height, state.width),
            state.condition,
            self.image_adaln,
            state.rope,
            generator,
        )
        tokens = second(
            tokens,
            (state.height, state.width),
            state.condition,
            self.image_adaln,
            state.rope,
            generator,
        )
        return boundary(
            tokens,
            (state.height, state.width),
            state.condition,
            text,
            self.image_adaln,
            self.cross_adaln,
            state.rope,
            generator,
        )

    def _run_middle(
        self,
        tokens: Tensor,
        state: _ImageState,
        text: PreparedText,
        rope: tuple[Tensor, Tensor],
        generator: torch.Generator | None,
    ) -> Tensor:
        """Run all 24 text-conditioned middle layers."""

        for module in self.middle_layers:
            layer = cast("MiddleImageLayer", module)
            tokens = layer(
                tokens,
                (state.height, state.width),
                state.condition,
                text,
                self.image_adaln,
                self.cross_adaln,
                rope,
                generator,
            )
        return tokens

    def _finish(
        self,
        prefix: Tensor,
        middle: Tensor,
        state: _ImageState,
        text: PreparedText,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Fuse SPRINT streams, run suffix layers, and decode velocity."""

        tokens = self._fuse(prefix, middle, state.condition)
        boundary = cast("BoundaryImageLayer", self.suffix_layers[0])
        second = cast("ImageLayer", self.suffix_layers[1])
        third = cast("ImageLayer", self.suffix_layers[2])
        tokens = boundary(
            tokens,
            (state.height, state.width),
            state.condition,
            text,
            self.image_adaln,
            self.cross_adaln,
            state.rope,
            generator,
        )
        tokens = second(
            tokens,
            (state.height, state.width),
            state.condition,
            self.image_adaln,
            state.rope,
            generator,
        )
        tokens = third(
            tokens,
            (state.height, state.width),
            state.condition,
            self.image_adaln,
            state.rope,
            generator,
        )
        return self._decode(tokens, state)

    @staticmethod
    def _gather_rope(
        rope: tuple[Tensor, Tensor],
        keep: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Gather shared axial RoPE values into a per-sample sparse layout."""

        sin, cos = rope
        index = keep.unsqueeze(-1).expand(-1, -1, sin.shape[-1])
        expanded_sin = sin.unsqueeze(0).expand(keep.shape[0], -1, -1)
        expanded_cos = cos.unsqueeze(0).expand(keep.shape[0], -1, -1)
        return torch.gather(expanded_sin, 1, index), torch.gather(
            expanded_cos,
            1,
            index,
        )

    def _fuse(
        self,
        prefix: Tensor,
        middle: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Apply timestep AdaScale and the learned SPRINT tail projection."""

        concatenated = torch.cat((prefix, middle), dim=-1)
        scale = self.tail_scale(F.silu(condition))
        concatenated = (concatenated.float() * (1.0 + scale.float().unsqueeze(1))).to(
            dtype=concatenated.dtype
        )
        return self.fusion(concatenated).to(dtype=prefix.dtype)

    def _decode(self, tokens: Tensor, state: _ImageState) -> Tensor:
        """Return float32 latent velocity."""

        with autocast(device_type=tokens.device.type, enabled=False):
            residual = F.linear(
                tokens.float(),
                self.output_projection.weight.float(),
                self.output_projection.bias.float(),
            )
            residual = residual.transpose(1, 2).reshape(
                tokens.shape[0],
                self.config.latent_channels,
                state.height,
                state.width,
            )
            time_fp32 = state.time.float()
            one_minus = 1.0 - time_fp32
            scale = time_fp32 / (time_fp32.square() + one_minus.square())
            return residual + scale[:, None, None, None] * state.latents.float()
