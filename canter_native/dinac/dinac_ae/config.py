"""Frozen model architecture and user-tunable inference configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DinacAEConfig:
    """Frozen architecture config stored alongside exported weights."""

    in_channels: int = 3
    patch_size: int = 16
    model_dim: int = 896
    encoder_depth: int = 6
    decoder_depth: int = 8
    decoder_start_blocks: int = 2
    decoder_end_blocks: int = 2
    bottleneck_dim: int = 128
    mlp_ratio: float = 4.0
    encoder_mlp_type: str = "gelu"
    depthwise_kernel_size: int = 7
    adaln_low_rank_rank: int = 128
    bottleneck_posterior_kind: str = "diagonal_gaussian"
    bottleneck_norm_mode: str = "disabled"
    logsnr_min: float = -10.0
    logsnr_max: float = 10.0
    pixel_noise_std: float = 0.558
    latent_running_stats_eps: float = 1e-4
    class_head_feature_dim: int = 768
    class_head_model_dim: int = 768
    class_head_head_dim: int = 64
    class_head_mlp_ratio: float = 4.0
    class_head_mlp_type: str = "gelu"
    class_head_register_token_count: int = 4

    @property
    def latent_channels(self) -> int:
        """Return the exported latent channel width."""

        return int(self.bottleneck_dim)

    @property
    def effective_patch_size(self) -> int:
        """Return the image-to-latent stride."""

        return int(self.patch_size)

    def save(self, path: str | Path) -> None:
        """Save config as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> DinacAEConfig:
        """Load config from JSON."""

        data = json.loads(Path(path).read_text())
        return cls(**data)


@dataclass
class DinacAEInferenceConfig:
    """User-tunable VP diffusion decode settings."""

    num_steps: int = 1
    sampler: str = "ddim"
    schedule: str = "linear"
    pdg: bool = False
    pdg_strength: float = 2.0
    strength: float = 1.0
    seed: int | None = None
