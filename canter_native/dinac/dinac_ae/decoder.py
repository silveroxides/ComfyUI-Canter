"""Decoder matching the exported FCDM decoder stack."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .adaln import AdaLNScaleGateZeroLowRankDelta, AdaLNScaleGateZeroProjector
from .fcdm_block import FCDMBlock
from .straight_through_encoder import Patchify
from .time_embed import SinusoidalTimeEmbeddingMLP


class Decoder(nn.Module):
    """VP diffusion decoder conditioned on encoder latents and timestep.

    Architecture (skip-concat, 2+4+2 default):
        Patchify x_t -> Fuse with upsampled z
        -> Start blocks (2) -> Middle blocks (4) -> Skip fuse -> End blocks (2)
        -> Conv1x1 -> PixelShuffle

    Path-Drop Guidance (PDG) at inference:
    - Replace middle block output with ``path_drop_mask_feature`` to create
      an unconditional prediction, then extrapolate.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        model_dim: int,
        depth: int,
        start_block_count: int,
        end_block_count: int,
        bottleneck_dim: int,
        mlp_ratio: float,
        depthwise_kernel_size: int,
        adaln_low_rank_rank: int,
        operations,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.model_dim = int(model_dim)

        self.patchify = Patchify(
            in_channels,
            patch_size,
            model_dim,
            operations,
        )

        self.latent_up = operations.Conv2d(
            bottleneck_dim, model_dim, kernel_size=1, bias=True
        )
        self.fuse_in = operations.Conv2d(
            2 * model_dim, model_dim, kernel_size=1, bias=True
        )

        # Time embedding
        self.time_embed = SinusoidalTimeEmbeddingMLP(
            model_dim, operations=operations
        )

        # 2-way AdaLN: shared base projector + per-block low-rank deltas
        self.adaln_base = AdaLNScaleGateZeroProjector(
            d_model=model_dim, d_cond=model_dim, operations=operations
        )
        self.adaln_deltas = nn.ModuleList(
            [
                AdaLNScaleGateZeroLowRankDelta(
                    d_model=model_dim, d_cond=model_dim,
                    rank=adaln_low_rank_rank, operations=operations,
                )
                for _ in range(depth)
            ]
        )

        # Block layout: start + middle + end
        middle_count = depth - start_block_count - end_block_count
        self._middle_start_idx = start_block_count
        self._end_start_idx = start_block_count + middle_count

        def _make_blocks(count: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    FCDMBlock(
                        model_dim,
                        mlp_ratio,
                        depthwise_kernel_size=depthwise_kernel_size,
                        use_external_adaln=True,
                        operations=operations,
                    )
                    for _ in range(count)
                ]
            )

        self.start_blocks = _make_blocks(start_block_count)
        self.middle_blocks = _make_blocks(middle_count)
        self.fuse_skip = operations.Conv2d(
            2 * model_dim, model_dim, kernel_size=1, bias=True
        )
        self.end_blocks = _make_blocks(end_block_count)

        self.path_drop_mask_feature = nn.Parameter(torch.zeros((1, model_dim, 1, 1)))

        self.out_proj = operations.Conv2d(
            model_dim, in_channels * (patch_size**2), kernel_size=1, bias=True
        )
        self.unpatchify = nn.PixelShuffle(patch_size)

    def _adaln_m_for_layer(self, cond: Tensor, layer_idx: int) -> Tensor:
        """Compute packed AdaLN modulation = shared_base + per-layer delta."""
        act = self.adaln_base.act(cond)
        base_m = self.adaln_base.project_activated(act)
        delta_m = self.adaln_deltas[layer_idx](act)
        return base_m + delta_m

    def _run_blocks(
        self, blocks: nn.ModuleList, x: Tensor, cond: Tensor, start_index: int
    ) -> Tensor:
        """Run a group of decoder blocks with per-block AdaLN modulation."""
        for local_idx, block in enumerate(blocks):
            adaln_m = self._adaln_m_for_layer(cond, layer_idx=start_index + local_idx)
            x = block(x, adaln_m=adaln_m)
        return x

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        latents: Tensor,
        *,
        drop_middle_blocks: bool = False,
    ) -> Tensor:
        """Single decoder forward pass.

        Args:
            x_t: Noised image [B, C, H, W].
            t: Timestep [B] in [0, 1].
            latents: Encoder latents [B, bottleneck_dim, h, w].
            drop_middle_blocks: Replace middle block output with mask feature (PDG).

        Returns:
            x0 prediction [B, C, H, W].
        """
        x_feat = self.patchify(x_t)
        z_up = self.latent_up(latents)

        fused = torch.cat([x_feat, z_up], dim=1)
        fused = self.fuse_in(fused)

        cond = self.time_embed(t.to(torch.float32).to(device=x_t.device))

        start_out = self._run_blocks(self.start_blocks, fused, cond, start_index=0)

        if drop_middle_blocks:
            middle_out = self.path_drop_mask_feature.to(
                device=x_t.device, dtype=x_t.dtype
            ).expand_as(start_out)
        else:
            middle_out = self._run_blocks(
                self.middle_blocks,
                start_out,
                cond,
                start_index=self._middle_start_idx,
            )

        skip_fused = torch.cat([start_out, middle_out], dim=1)
        skip_fused = self.fuse_skip(skip_fused)

        end_out = self._run_blocks(
            self.end_blocks, skip_fused, cond, start_index=self._end_start_idx
        )
        patches = self.out_proj(end_out)
        return self.unpatchify(patches)
