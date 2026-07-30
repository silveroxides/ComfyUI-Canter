"""Small DiT configuration enums required by the DINAC-AE export."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DiTConditioning(Enum):
    """Conditioning strategy for exported DiT blocks."""

    ADALN = auto()
    GATED_UNCOND = auto()
    UNCOND = auto()


class AdaLNSharingMode(Enum):
    """AdaLN sharing modes retained for auxiliary config imports."""

    PER_BLOCK = auto()
    SHARED_BASE_LOW_RANK_DELTA = auto()


@dataclass
class DiTBodyConfig:
    """Minimal body config placeholder for unused auxiliary heads."""

    depth: int = 1
    d_model: int = 768
    n_heads: int = 12


__all__ = ["AdaLNSharingMode", "DiTBodyConfig", "DiTConditioning"]
