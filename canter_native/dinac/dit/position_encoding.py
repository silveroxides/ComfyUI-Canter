"""Position encoding options used by exported dense DiT blocks."""

from __future__ import annotations

from enum import Enum


class DiTPositionEncoding(Enum):
    """Position encoding strategy used inside exported DiT blocks."""

    ROPE_2D_AXIAL_DILATED = "rope_2d_axial_dilated"
    ROPE_2D_AXIAL_UNNORMALIZED_DILATED = "rope_2d_axial_unnormalized_dilated"
    ROPE_2D_AXIAL_NORMALIZED = "rope_2d_axial_normalized"
    ROPE_2D_AXIAL_UNNORMALIZED = "rope_2d_axial_unnormalized"
    ROPE_2D_AXIAL_FREQ_AWARE = "rope_2d_axial_freq_aware"
    ROPE_2D_AXIAL_BETA_WARP = "rope_2d_axial_beta_warp"
    ROPE_2D_AXIAL_ALPHA_WARP = "rope_2d_axial_alpha_warp"
    ROPE_3D_ZIMAGE = "rope_3d_zimage"
    ROPE_1D = "rope_1d"
    NONE = "none"


__all__ = ["DiTPositionEncoding"]
