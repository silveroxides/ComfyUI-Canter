"""Frozen architecture constants for the Canter 2B denoiser."""

from __future__ import annotations

from dataclasses import dataclass

CANTER_LICENSE = "MG-BY-SA-2.0"
CANTER_LICENSE_URL = "https://ids.nus.edu.sg/docs/modelgo/v2/MG-BY-SA/LICENSE"
CANTER_DEFAULT_RELEASE = "v0001"


@dataclass(frozen=True)
class CanterConfig:
    """The single architecture supported by the public Canter checkpoint.

    These values are deliberately not constructor options on ``CanterModel``.
    A checkpoint with different dimensions is a different architecture and must
    receive its own export package.
    """

    latent_channels: int = 128
    width: int = 2048
    depth: int = 30
    prefix_depth: int = 3
    suffix_depth: int = 3
    image_heads: int = 16
    adaln_rank: int = 256
    mlp_ratio: int = 4
    text_width: int = 1024
    text_backbone_width: int = 960
    text_refine_depth: int = 4
    text_heads: int = 8
    cross_heads: int = 16
    cross_head_dim: int = 128
    text_rope_theta: float = 160_000.0
    text_max_length: int = 7_999
    time_frequency_width: int = 256
    time_scale: float = 1_000.0
    rope_base: float = 10_000.0

    @property
    def middle_depth(self) -> int:
        """Return the number of SPRINT-routable functional layers."""

        return self.depth - self.prefix_depth - self.suffix_depth


CANTER_CONFIG = CanterConfig()
