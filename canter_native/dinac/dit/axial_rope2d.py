from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, cast

import torch
from torch import Tensor, nn

__all__ = [
    "AxialRoPE2D",
    "AxialRoPE2DAlphaWarpConfig",
    "AxialRoPE2DBetaWarpConfig",
    "AxialRoPE2DConfig",
    "AxialRoPE2DCoordMode",
    "AxialRoPE2DDimLayout",
    "AxialRoPE2DDyPE",
    "AxialRoPE2DDyPEConfig",
    "AxialRoPE2DFrequencyAwareConfig",
    "AxialRoPE2DNormalizeCoords",
    "DyPERoPEMethod",
    "build_axial_rope2d_dype",
    "build_axial_rope2d_inference_warp_with_strength",
    "build_axial_rope2d_with_lumina_frequency_warp",
    "lumina_frequency_aware_periods_for_axis",
    "set_axial_rope2d_dype_noise_time",
]


class AxialRoPE2DNormalizeCoords(Enum):
    """Coordinate normalization strategy for axial 2D RoPE (DINOv3-style)."""

    MIN = "min"
    MAX = "max"
    SEPARATE = "separate"


class AxialRoPE2DCoordMode(Enum):
    """Coordinate grid mode for axial 2D RoPE.

    - ``DINOV3_NORMALIZED``: DINOv3-style normalized patch-centre coordinates in
      ``[-1, 1]`` (after normalization).
    - ``PATCH_INDICES``: Standard unnormalized patch-grid coordinates in patch
      units (e.g., ``x in [0, W-1]``, ``y in [0, H-1]``).
    """

    DINOV3_NORMALIZED = "dinov3_normalized"
    PATCH_INDICES = "patch_indices"


class AxialRoPE2DDimLayout(Enum):
    """Layout of angles along the head-dimension.

    The layout must match the rotation convention used when applying RoPE to Q/K.

    - ``HALF_SPLIT``: LLaMA-style layout compatible with ``common.rope.rotate_half``
      (splits last dim into two halves).
    - ``PAIR_INTERLEAVED``: EVA-02 / SpeedrunDiT-style layout compatible with an
      adjacent-pair rotate_half (pairs consecutive dims).

    TODO(refactor): Standardize on ``PAIR_INTERLEAVED`` throughout DiT to reduce
    complexity and avoid layout mismatches, then delete ``HALF_SPLIT`` and any
    related branching once the migration is complete.
    """

    HALF_SPLIT = "half_split"
    PAIR_INTERLEAVED = "pair_interleaved"


class DyPERoPEMethod(Enum):
    """Dynamic position extrapolation method applied to inference RoPE."""

    VISION_YARN = "vision_yarn"
    DY_YARN = "dy_yarn"
    DY_NTK = "dy_ntk"


@dataclass(frozen=True)
class AxialRoPE2DDyPEConfig:
    """Inference-only DyPE controls for axial RoPE.

    Args:
        method: Dynamic extrapolation rule to apply.
        ref_h_tokens: Training/reference token height.
        ref_w_tokens: Training/reference token width.
        lambda_s: Dynamic extrapolation magnitude.
        lambda_t: Dynamic extrapolation noise-time exponent.
        yarn_beta_0: YaRN first-ramp high rotation threshold.
        yarn_beta_1: YaRN first-ramp low rotation threshold.
        yarn_gamma_0: YaRN base-blend high rotation threshold.
        yarn_gamma_1: YaRN base-blend low rotation threshold.
        yarn_attention_scale: Apply YaRN's static attention magnitude correction.
    """

    method: DyPERoPEMethod
    ref_h_tokens: int
    ref_w_tokens: int
    lambda_s: float = 2.0
    lambda_t: float = 2.0
    yarn_beta_0: float = 1.25
    yarn_beta_1: float = 0.75
    yarn_gamma_0: float = 16.0
    yarn_gamma_1: float = 2.0
    yarn_attention_scale: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.method, DyPERoPEMethod):
            raise TypeError("method must be a DyPERoPEMethod")
        if int(self.ref_h_tokens) <= 0 or int(self.ref_w_tokens) <= 0:
            raise ValueError("ref_h_tokens and ref_w_tokens must be positive")
        for name, value in (
            ("lambda_s", self.lambda_s),
            ("lambda_t", self.lambda_t),
            ("yarn_beta_0", self.yarn_beta_0),
            ("yarn_beta_1", self.yarn_beta_1),
            ("yarn_gamma_0", self.yarn_gamma_0),
            ("yarn_gamma_1", self.yarn_gamma_1),
        ):
            v = float(value)
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not isinstance(self.yarn_attention_scale, bool):
            raise TypeError("yarn_attention_scale must be a bool")


@dataclass(frozen=True)
class AxialRoPE2DFrequencyAwareConfig:
    """Lumina/Next-DiT-style frequency-aware RoPE warping for one token grid.

    This config implements a per-axis, per-band frequency warp that depends on
    the input axis length ``L`` relative to a reference length ``L_ref``:

    - Define the axis scale ``s = L / L_ref``.
    - RoPE is parameterized by *periods* (wavelengths in tokens) ``period[d]``.
      In this module's axial parameterization (with patch-index coordinates),
      the angle for coordinate ``p`` and band ``d`` is:

          angle(p, d) = 2π * p / period[d]

      so the wavelength of band ``d`` is exactly ``period[d]`` tokens.

    - Pick a *boundary wavelength* ``L_boundary`` (in tokens), expressed as a
      trainable multiplier around the reference length:

          L_boundary = L_ref * exp(boundary_log_multiplier)

      The scalar ``boundary_log_multiplier`` is shared across H/W axes (and
      initialized by this config).

    - Define a (possibly fractional) boundary band index ``d*`` as the band
      whose wavelength equals ``L_boundary``:

          period(d*) = L_boundary

      In practice we compute ``d*`` by linear interpolation in log-period space
      (periods are geometric for both supported period parametrizations).

    - The Lumina/Next-DiT implicit exponent ramp is then:

          alpha[d] = clamp(d / d*, 0, 1)

      where:
        - high-frequency bands (small d) have alpha≈0 (extrapolation-like),
        - low-frequency bands (large d) have alpha→1 (interpolation-like),
        - alpha is capped at 1 to ensure we never compress a band more than
          plain position interpolation would.

    - Finally, warp the periods per axis:

          period'[d] = period[d] * s ** alpha[d]

      Equivalently, angular frequencies warp as:

          omega'[d] = omega[d] / s ** alpha[d]

    Notes
    -----
    - This warp is only meaningful for patch-index coordinates
      (``AxialRoPE2DCoordMode.PATCH_INDICES``). Mixing it with normalized
      coordinates would create an implicit "gauge switch"; we fail fast.
    - The boundary multiplier is trainable by construction (it is stored as an
      nn.Parameter inside AxialRoPE2D when this config is present).
    """

    ref_h_tokens: int
    ref_w_tokens: int
    boundary_log_multiplier_init: float

    def __post_init__(self) -> None:
        if int(self.ref_h_tokens) <= 0 or int(self.ref_w_tokens) <= 0:
            raise ValueError("ref_h_tokens and ref_w_tokens must be positive")
        init = float(self.boundary_log_multiplier_init)
        if not math.isfinite(init):
            raise ValueError("boundary_log_multiplier_init must be finite")


@dataclass(frozen=True)
class AxialRoPE2DBetaWarpConfig:
    """Trainable beta-curve warping for axial 2D RoPE periods (per token grid).

    This config defines a per-axis period warp that depends on the runtime axis
    length ``L`` relative to a reference length ``L_ref``:

        s = L / L_ref
        period'[d] = period[d] * s ** beta[d]

    where the per-band exponent curve beta(d) is parameterized by three
    trainable u-space scalars (shared across H/W axes):

        beta_hi   = beta_max * tanh(beta_hi_u)    (high-frequency endpoint, d=0)
        beta_lo   = beta_max * tanh(beta_lo_u)    (low-frequency endpoint, d=qtr-1)
        beta_bend = beta_max * tanh(beta_bend_u)  (mid-band bump amplitude)

    and the per-band curve is:

        t = d / (qtr - 1)   in [0, 1]
        beta(t) = lerp(beta_hi, beta_lo, t) + beta_bend * 4*t*(1-t)

    Interpretation
    --------------
    - ``beta(d) == 0``: identity / "extrapolation-like" (no warping; periods do not
      change with axis length).
    - ``beta(d) == 1``: position-interpolation-like for that band
      (``period'[d] = period[d] * s`` so ``omega'[d] = omega[d] / s``).

    This parameterization provides strong and smooth control over the effective
    scaling of each frequency band, including allowing beta<0 (increasing
    frequencies when s>1), which can be important for unnormalized RoPE bases
    (e.g. base=10_000) where some very low-frequency bands barely rotate on
    practical token grids.

    Notes:
      - This warp requires patch-index coordinates (coord_mode=PATCH_INDICES).
      - The u parameters are stored as nn.Parameter inside AxialRoPE2D when this
        config is present.
    """

    ref_h_tokens: int
    ref_w_tokens: int
    beta_max: float
    beta_hi_u_init: float
    beta_lo_u_init: float
    beta_bend_u_init: float

    def __post_init__(self) -> None:
        if int(self.ref_h_tokens) <= 0 or int(self.ref_w_tokens) <= 0:
            raise ValueError("ref_h_tokens and ref_w_tokens must be positive")
        bmax = float(self.beta_max)
        if not math.isfinite(bmax) or bmax <= 0.0:
            raise ValueError("beta_max must be finite and > 0")
        for name, value in (
            ("beta_hi_u_init", self.beta_hi_u_init),
            ("beta_lo_u_init", self.beta_lo_u_init),
            ("beta_bend_u_init", self.beta_bend_u_init),
        ):
            v = float(value)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class AxialRoPE2DAlphaWarpConfig:
    """Per-band power-law warping of axial 2D RoPE frequencies (shared across axes).

    This config warps RoPE frequencies per band using a learned exponent vector
    ``alpha[d]`` shared across H/W axes:

        f'[d] = f[d] * s ** alpha[d]        where s = L / L_ref

    Since this module parameterizes angles via periods ``period[d]`` with
    ``f[d] ∝ 1 / period[d]``, the equivalent period warp implemented in AxialRoPE2D is:

        period'[d] = period[d] / s ** alpha[d]

    Notes:
      - This warp requires patch-index coordinates (coord_mode=PATCH_INDICES).
      - ``alpha`` is stored as an unconstrained nn.Parameter vector of length Q
        (bands per axis), initialized to ``alpha_init`` for all bands.
    """

    ref_h_tokens: int
    ref_w_tokens: int
    alpha_init: float

    def __post_init__(self) -> None:
        if int(self.ref_h_tokens) <= 0 or int(self.ref_w_tokens) <= 0:
            raise ValueError("ref_h_tokens and ref_w_tokens must be positive")
        init = float(self.alpha_init)
        if not math.isfinite(init):
            raise ValueError("alpha_init must be finite")


@dataclass(frozen=True)
class AxialRoPE2DConfig:
    """Configuration for axial 2D RoPE sin/cos generation.

    This module supports two coordinate conventions via ``coord_mode``:
    - ``DINOV3_NORMALIZED``: DINOv3-style normalized patch-centre coordinates in
      ``[-1, 1]`` (after normalization).
    - ``PATCH_INDICES``: Standard unnormalized patch-grid coordinates in patch
      units (e.g., ``x in [0, W-1]``).

    Period parametrization
    ----------------------
    The periods parametrization matches DINOv3:
    - Provide either `base` (and leave `min_period/max_period` unset), or
    - Provide both `min_period` and `max_period` (and set `base=None`).
    """

    base: float | None = 100.0
    min_period: float | None = None
    max_period: float | None = None
    coord_mode: AxialRoPE2DCoordMode = AxialRoPE2DCoordMode.DINOV3_NORMALIZED
    normalize_coords: AxialRoPE2DNormalizeCoords = AxialRoPE2DNormalizeCoords.MAX
    dim_layout: AxialRoPE2DDimLayout = AxialRoPE2DDimLayout.HALF_SPLIT
    angle_multiplier: float = 2.0 * float(math.pi)
    coord_offset: float = 0.5
    frequency_aware: AxialRoPE2DFrequencyAwareConfig | None = None
    beta_warp: AxialRoPE2DBetaWarpConfig | None = None
    alpha_warp: AxialRoPE2DAlphaWarpConfig | None = None

    def __post_init__(self) -> None:
        both_periods = self.min_period is not None and self.max_period is not None
        if (self.base is None and not both_periods) or (
            self.base is not None and both_periods
        ):
            raise ValueError(
                "AxialRoPE2DConfig requires either base!=None, or both min_period and max_period."
            )
        if self.base is not None and float(self.base) <= 0.0:
            raise ValueError("AxialRoPE2DConfig.base must be positive when provided")
        if self.min_period is not None and float(self.min_period) <= 0.0:
            raise ValueError(
                "AxialRoPE2DConfig.min_period must be positive when provided"
            )
        if self.max_period is not None and float(self.max_period) <= 0.0:
            raise ValueError(
                "AxialRoPE2DConfig.max_period must be positive when provided"
            )
        if self.min_period is not None and self.max_period is not None:
            if float(self.max_period) <= float(self.min_period):
                raise ValueError("AxialRoPE2DConfig.max_period must be > min_period")
        if not isinstance(self.coord_mode, AxialRoPE2DCoordMode):
            raise TypeError(
                "AxialRoPE2DConfig.coord_mode must be an AxialRoPE2DCoordMode"
            )
        if not isinstance(self.normalize_coords, AxialRoPE2DNormalizeCoords):
            raise TypeError(
                "AxialRoPE2DConfig.normalize_coords must be an AxialRoPE2DNormalizeCoords"
            )
        if not isinstance(self.dim_layout, AxialRoPE2DDimLayout):
            raise TypeError(
                "AxialRoPE2DConfig.dim_layout must be an AxialRoPE2DDimLayout"
            )
        mult = float(self.angle_multiplier)
        if not math.isfinite(mult) or mult <= 0.0:
            raise ValueError(
                "AxialRoPE2DConfig.angle_multiplier must be finite and > 0"
            )
        off = float(self.coord_offset)
        if not math.isfinite(off):
            raise ValueError("AxialRoPE2DConfig.coord_offset must be finite")
        if self.frequency_aware is not None and not isinstance(
            self.frequency_aware, AxialRoPE2DFrequencyAwareConfig
        ):
            raise TypeError(
                "AxialRoPE2DConfig.frequency_aware must be an AxialRoPE2DFrequencyAwareConfig"
            )
        if self.beta_warp is not None and not isinstance(
            self.beta_warp, AxialRoPE2DBetaWarpConfig
        ):
            raise TypeError(
                "AxialRoPE2DConfig.beta_warp must be an AxialRoPE2DBetaWarpConfig"
            )
        if self.alpha_warp is not None and not isinstance(
            self.alpha_warp, AxialRoPE2DAlphaWarpConfig
        ):
            raise TypeError(
                "AxialRoPE2DConfig.alpha_warp must be an AxialRoPE2DAlphaWarpConfig"
            )
        warp_count = (
            int(self.frequency_aware is not None)
            + int(self.beta_warp is not None)
            + int(self.alpha_warp is not None)
        )
        if warp_count > 1:
            raise ValueError(
                "AxialRoPE2DConfig requires at most one of frequency_aware, beta_warp, or alpha_warp"
            )
        if self.frequency_aware is not None and (
            self.coord_mode is not AxialRoPE2DCoordMode.PATCH_INDICES
        ):
            raise ValueError(
                "AxialRoPE2D frequency-aware warping requires coord_mode=PATCH_INDICES"
            )
        if self.beta_warp is not None and (
            self.coord_mode is not AxialRoPE2DCoordMode.PATCH_INDICES
        ):
            raise ValueError("AxialRoPE2D beta warp requires coord_mode=PATCH_INDICES")
        if self.alpha_warp is not None and (
            self.coord_mode is not AxialRoPE2DCoordMode.PATCH_INDICES
        ):
            raise ValueError("AxialRoPE2D alpha warp requires coord_mode=PATCH_INDICES")


_AXIAL_COORDS_CACHE: dict[
    tuple[
        int, int, torch.device, AxialRoPE2DCoordMode, AxialRoPE2DNormalizeCoords, float
    ],
    Tensor,
] = {}


def _get_dinov3_normalized_coords(
    H: int,
    W: int,
    *,
    device: torch.device,
    normalize: AxialRoPE2DNormalizeCoords,
    offset: float,
) -> Tensor:
    """Return DINOv3-style flattened coords in [-1, 1] with shape [HW, 2]."""
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive for axial RoPE coords")
    key = (
        int(H),
        int(W),
        device,
        AxialRoPE2DCoordMode.DINOV3_NORMALIZED,
        normalize,
        float(offset),
    )
    cached = _AXIAL_COORDS_CACHE.get(key)
    if cached is not None:
        return cached
    start = float(offset)
    end_h = start + float(int(H))
    end_w = start + float(int(W))
    match normalize:
        case AxialRoPE2DNormalizeCoords.MAX:
            denom = float(max(int(H), int(W)))
            coords_h = (
                torch.arange(start, end_h, device=device, dtype=torch.float32) / denom
            )
            coords_w = (
                torch.arange(start, end_w, device=device, dtype=torch.float32) / denom
            )
        case AxialRoPE2DNormalizeCoords.MIN:
            denom = float(min(int(H), int(W)))
            coords_h = (
                torch.arange(start, end_h, device=device, dtype=torch.float32) / denom
            )
            coords_w = (
                torch.arange(start, end_w, device=device, dtype=torch.float32) / denom
            )
        case AxialRoPE2DNormalizeCoords.SEPARATE:
            coords_h = torch.arange(
                start, end_h, device=device, dtype=torch.float32
            ) / float(int(H))
            coords_w = torch.arange(
                start, end_w, device=device, dtype=torch.float32
            ) / float(int(W))
        case _ as unreachable:  # pragma: no cover - defensive
            raise RuntimeError(f"Unsupported normalize_coords: {unreachable}")
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0
    # torch.compile cannot trace `torch.is_inference_mode_enabled()` and should
    # not record Python-side cache mutations in the graph.
    if torch.compiler.is_compiling():
        return coords
    if torch.is_inference_mode_enabled():
        return coords
    _AXIAL_COORDS_CACHE[key] = coords
    return coords


def _get_patch_index_coords(
    H: int,
    W: int,
    *,
    device: torch.device,
    offset: float,
) -> Tensor:
    """Return unnormalized patch-grid coords with shape [HW, 2] and (y, x) columns."""
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive for axial RoPE coords")
    key = (
        int(H),
        int(W),
        device,
        AxialRoPE2DCoordMode.PATCH_INDICES,
        AxialRoPE2DNormalizeCoords.MAX,
        float(offset),
    )
    cached = _AXIAL_COORDS_CACHE.get(key)
    if cached is not None:
        return cached
    start = float(offset)
    end_h = start + float(int(H))
    end_w = start + float(int(W))
    coords_h = torch.arange(start, end_h, device=device, dtype=torch.float32)
    coords_w = torch.arange(start, end_w, device=device, dtype=torch.float32)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    if torch.compiler.is_compiling():
        return coords
    if torch.is_inference_mode_enabled():
        return coords
    _AXIAL_COORDS_CACHE[key] = coords
    return coords


def _lumina_boundary_band_index(
    *,
    periods: Tensor,
    boundary_wavelength: Tensor,
) -> Tensor:
    """Return the fractional boundary band index d* for a given boundary wavelength.

    This implements the Lumina/Next-DiT definition:

        period(d*) = boundary_wavelength

    We compute d* by linear interpolation in log-period space. For the supported
    period parameterizations, periods are geometric and log(period) is linear in
    band index.

    Args:
        periods: 1D float tensor of length Q containing monotonically increasing
            periods in tokens.
        boundary_wavelength: Scalar positive float tensor giving the desired
            boundary wavelength in tokens.

    Returns:
        Scalar float32 tensor giving the (possibly fractional) boundary index d*.

    Raises:
        ValueError: If periods are invalid or the boundary is outside valid range
            for a well-defined positive d*.
    """
    if periods.dim() != 1:
        raise ValueError("periods must be 1D for boundary band index")
    if int(periods.numel()) < 2:
        raise ValueError("periods must have length >= 2 for boundary band index")
    if boundary_wavelength.dim() != 0:
        raise ValueError("boundary_wavelength must be a scalar tensor")
    if not torch.isfinite(boundary_wavelength).item():
        raise ValueError("boundary_wavelength must be finite")
    if float(boundary_wavelength.item()) <= 0.0:
        raise ValueError("boundary_wavelength must be > 0")

    periods_f = periods.to(dtype=torch.float32)
    if not torch.isfinite(periods_f).all().item():
        raise ValueError("periods must be finite for boundary band index")
    if float(periods_f[0].item()) <= 0.0:
        raise ValueError("periods must be positive for boundary band index")
    if not (periods_f[1:] > periods_f[:-1]).all().item():
        raise ValueError("periods must be strictly increasing for boundary band index")

    log_p0 = torch.log(periods_f[0])
    log_p1 = torch.log(periods_f[-1])
    denom = log_p1 - log_p0
    if float(denom.item()) <= 0.0:
        raise ValueError("Invalid periods range for boundary band index")
    log_boundary = torch.log(boundary_wavelength.to(dtype=torch.float32))
    q = int(periods_f.numel())
    d_star = (float(q - 1) * (log_boundary - log_p0)) / denom
    if not torch.isfinite(d_star).item():
        raise ValueError("Computed non-finite boundary band index d*")
    if float(d_star.item()) <= 0.0:
        raise ValueError(
            "Boundary wavelength implies d* <= 0; increase the boundary wavelength "
            "(or its multiplier) to be >= the wavelength of the first non-zero band."
        )
    return d_star


def _lumina_alpha_ramp(
    *,
    qtr: int,
    d_star: Tensor,
    device: torch.device,
) -> Tensor:
    """Return alpha[d] = clamp(d / d*, 0, 1) for d in [0, qtr).

    Args:
        qtr: Number of RoPE bands per axis (Q).
        d_star: Scalar positive float tensor boundary index d*.
        device: Device for the returned alpha tensor.

    Returns:
        Float32 tensor of shape [Q] with values in [0, 1].
    """
    if int(qtr) <= 0:
        raise ValueError("qtr must be positive for alpha ramp")
    if d_star.dim() != 0:
        raise ValueError("d_star must be a scalar tensor for alpha ramp")
    if float(d_star.item()) <= 0.0:
        raise ValueError("d_star must be > 0 for alpha ramp")
    d = torch.arange(int(qtr), device=device, dtype=torch.float32)
    alpha = d / d_star.to(device=device, dtype=torch.float32)
    return torch.clamp(alpha, min=0.0, max=1.0)


def lumina_frequency_aware_periods_for_axis(
    *,
    periods: Tensor,
    axis_len: int,
    ref_axis_len: int,
    boundary_log_multiplier: Tensor,
    angle_multiplier: float,
) -> Tensor:
    """Return Lumina/Next-DiT frequency-aware warped periods for one axis.

    Implements:
      s = axis_len / ref_axis_len
      L_boundary = ref_axis_len * exp(boundary_log_multiplier)
      d* = boundary band index where period(d*) = L_boundary
      alpha[d] = clamp(d / d*, 0, 1)
      period'[d] = period[d] * s**alpha[d]

    Notes on ``angle_multiplier``
    -----------------------------
    This module parameterizes angles as:

        angle(p, d) = angle_multiplier * p / period[d]

    The *wavelength* (period in tokens) is the delta in ``p`` that increases
    the angle by ``2π``:

        wavelength[d] = 2π * period[d] / angle_multiplier

    Lumina/Next-DiT define the boundary by matching *wavelength* to the
    reference axis length. We therefore convert the boundary wavelength
    ``L_boundary`` into a boundary period via:

        period_boundary = (angle_multiplier / 2π) * L_boundary

    When ``angle_multiplier == 2π`` (the DINOv3-style parameterization), this
    reduces to ``period_boundary == L_boundary``.

    Args:
        periods: Base periods ``[Q]`` in tokens (wavelengths).
        axis_len: Input axis length ``L`` in tokens.
        ref_axis_len: Reference axis length ``L_ref`` in tokens.
        boundary_log_multiplier: Scalar tensor; shared trainable log-multiplier.
        angle_multiplier: RoPE angle multiplier used when converting periods to
            physical wavelengths in tokens.

    Returns:
        Warped periods ``[Q]`` as float32.

    Raises:
        ValueError: If inputs are malformed or imply an invalid boundary index.
    """
    if int(axis_len) <= 0:
        raise ValueError("axis_len must be positive for frequency-aware periods")
    if int(ref_axis_len) <= 0:
        raise ValueError("ref_axis_len must be positive for frequency-aware periods")
    if boundary_log_multiplier.dim() != 0:
        raise ValueError("boundary_log_multiplier must be a scalar tensor")
    if not torch.isfinite(boundary_log_multiplier).item():
        raise ValueError("boundary_log_multiplier must be finite")
    mult = float(angle_multiplier)
    if not math.isfinite(mult) or mult <= 0.0:
        raise ValueError("angle_multiplier must be finite and > 0")

    device = periods.device
    qtr = int(periods.numel())
    s = float(int(axis_len)) / float(int(ref_axis_len))
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("axis_len/ref_axis_len must be finite and > 0")
    boundary_wavelength = float(int(ref_axis_len)) * torch.exp(
        boundary_log_multiplier.to(device=device, dtype=torch.float32)
    )
    boundary_period = (mult / (2.0 * float(math.pi))) * boundary_wavelength
    d_star = _lumina_boundary_band_index(
        periods=periods, boundary_wavelength=boundary_period
    )
    alpha = _lumina_alpha_ramp(qtr=qtr, d_star=d_star, device=device)
    scale = torch.pow(torch.tensor(s, device=device, dtype=torch.float32), alpha)
    return periods.to(device=device, dtype=torch.float32) * scale


def build_axial_rope2d_with_lumina_frequency_warp(
    base: AxialRoPE2D,
    *,
    ref_h_tokens: int,
    ref_w_tokens: int,
    boundary_log_multiplier: float | None,
    boundary_band_multiplier: float | None,
) -> AxialRoPE2D:
    """Return an AxialRoPE2D module that applies Lumina-style frequency warping.

    This helper is intended for inference-time experimentation on checkpoints
    that were trained without frequency-aware warping (e.g.
    ``position_encoding=ROPE_2D_AXIAL_UNNORMALIZED``). It constructs a new
    AxialRoPE2D instance that:
      - Keeps the base RoPE periods and layout identical to ``base``.
      - Applies Lumina/Next-DiT per-axis warping based on the runtime token
        lengths ``H`` and ``W`` relative to reference lengths.
      - Uses a fixed (non-trainable) scalar boundary multiplier for inference.

    Args:
        base: Existing AxialRoPE2D instance from a loaded model.
        ref_h_tokens: Reference H token length (L_ref,h).
        ref_w_tokens: Reference W token length (L_ref,w).
        boundary_log_multiplier: Optional log multiplier applied to reference
            lengths to define the boundary wavelength. Use 0.0 for "boundary at
            L_ref". Mutually exclusive with boundary_band_multiplier.
        boundary_band_multiplier: Optional multiplier that directly selects the
            boundary band index d* relative to the lowest-frequency band index
            (qtr-1). Concretely, with qtr bands per axis:

                d* = boundary_band_multiplier * (qtr - 1)

            This lets you move the transition point in frequency space:
            - smaller values => more bands become PI-like (more interpolation)
            - larger values => fewer bands become PI-like (more extrapolation)

            When provided, we compute the implied boundary wavelength and store
            it as boundary_log_multiplier for the module.

    Returns:
        New AxialRoPE2D instance on the same device as ``base``.

    Raises:
        TypeError: If base is not an AxialRoPE2D.
        ValueError: If base uses incompatible coordinates for Lumina warping.
    """
    if not isinstance(base, AxialRoPE2D):
        raise TypeError("base must be an AxialRoPE2D")
    if base.cfg.coord_mode is not AxialRoPE2DCoordMode.PATCH_INDICES:
        raise ValueError(
            "Lumina frequency-aware warping requires coord_mode=PATCH_INDICES"
        )
    if (boundary_log_multiplier is None) == (boundary_band_multiplier is None):
        raise ValueError(
            "Provide exactly one of boundary_log_multiplier or boundary_band_multiplier"
        )

    resolved_log_multiplier: float
    if boundary_band_multiplier is not None:
        if int(ref_h_tokens) != int(ref_w_tokens):
            raise ValueError(
                "boundary_band_multiplier requires ref_h_tokens == ref_w_tokens when using a shared scalar boundary"
            )
        mult = float(boundary_band_multiplier)
        if not math.isfinite(mult) or mult <= 0.0:
            raise ValueError("boundary_band_multiplier must be finite and > 0")
        qtr = int(base.periods.numel())
        if qtr < 2:
            raise ValueError(
                "AxialRoPE2D periods length must be >= 2 for boundary band selection"
            )
        # Solve for the boundary wavelength implied by choosing d* directly.
        #
        # We use geometric interpolation in log-period space:
        #   log(period(d*)) = log(period0) + (d*/(qtr-1)) * (log(period_max) - log(period0))
        # with:
        #   d* = boundary_band_multiplier * (qtr-1)
        #
        # This allows d* outside the trained band range (multiplier > 1), which
        # corresponds to pushing the transition beyond the lowest-frequency band.
        with torch.no_grad():
            periods_f = base.periods.to(dtype=torch.float32, device=torch.device("cpu"))
            if not (periods_f[1:] > periods_f[:-1]).all().item():
                raise ValueError(
                    "base.periods must be strictly increasing for boundary band selection"
                )
            log_p0 = float(torch.log(periods_f[0]).item())
            log_p1 = float(torch.log(periods_f[-1]).item())
        d_star = mult * float(qtr - 1)
        log_boundary_period = log_p0 + (d_star / float(qtr - 1)) * (log_p1 - log_p0)
        boundary_period = math.exp(log_boundary_period)
        angle_mult = float(base.cfg.angle_multiplier)
        if not math.isfinite(angle_mult) or angle_mult <= 0.0:
            raise ValueError("base.cfg.angle_multiplier must be finite and > 0")
        boundary_wavelength = (2.0 * float(math.pi) / angle_mult) * boundary_period
        resolved_log_multiplier = math.log(
            boundary_wavelength / float(int(ref_h_tokens))
        )
    else:
        if boundary_log_multiplier is None:  # pragma: no cover - validated above
            raise RuntimeError("boundary_log_multiplier missing despite validation")
        resolved_log_multiplier = float(boundary_log_multiplier)

    freq_cfg = AxialRoPE2DFrequencyAwareConfig(
        ref_h_tokens=int(ref_h_tokens),
        ref_w_tokens=int(ref_w_tokens),
        boundary_log_multiplier_init=resolved_log_multiplier,
    )
    cfg = replace(base.cfg, frequency_aware=freq_cfg, beta_warp=None, alpha_warp=None)
    device = base.periods.device
    warped = AxialRoPE2D(head_dim=int(base.head_dim), cfg=cfg).to(device=device)
    with torch.no_grad():
        warped.periods.copy_(base.periods.to(device=device, dtype=torch.float32))
        if warped.boundary_log_multiplier is None:  # pragma: no cover - defensive
            raise RuntimeError("Expected boundary_log_multiplier to be initialized")
        warped.boundary_log_multiplier.copy_(
            torch.tensor(resolved_log_multiplier, device=device, dtype=torch.float32)
        )
        warped.boundary_log_multiplier.requires_grad_(False)
    return warped


def build_axial_rope2d_inference_warp_with_strength(
    base: AxialRoPE2D,
    *,
    ref_h_tokens: int,
    ref_w_tokens: int,
    beta_hi_u: float,
    beta_lo_u: float,
    beta_bend_u: float,
    beta_max: float,
) -> AxialRoPE2D:
    """Build an inference-only RoPE warp parameterized by a 3-knob beta(t) curve.

    This helper is meant for notebook experimentation on checkpoints trained
    with patch-index axial RoPE (e.g. ``position_encoding=ROPE_2D_AXIAL_UNNORMALIZED``).

    We warp per-axis RoPE periods (wavelengths, in tokens) as:

        period'[d] = period[d] * s ** beta[d]      where s = L / L_ref

    with a smooth exponent curve beta(d) over bands. Unlike a strict
    interpolation-only exponent (0..1), beta is allowed to be negative or > 1,
    which is important for unnormalized RoPE (e.g. base=10_000) where some very
    low-frequency bands are effectively "dead" on practical token grids unless
    their frequencies can be increased (beta < 0).

    Knobs (bounded via u-space)
    ---------------------------
    We use three unconstrained parameters (u-space) which map to bounded beta
    values via tanh:

        beta_hi   = beta_max * tanh(beta_hi_u)    (high-frequency endpoint, d=0)
        beta_lo   = beta_max * tanh(beta_lo_u)    (low-frequency endpoint, d=qtr-1)
        beta_bend = beta_max * tanh(beta_bend_u)  ("bump" amplitude in the middle)

    Then define the per-band curve over t in [0,1] (high -> low frequency):

        t = d / (qtr - 1)
        beta(t) = lerp(beta_hi, beta_lo, t) + beta_bend * 4*t*(1-t)

    The bump term is 0 at the endpoints and peaks at 1 at t=0.5.

    Notes
    -----
    - This wrapper is inference-only: it is not saved in checkpoints.
    - It requires patch-index coordinates (no normalized "gauge").
    - It preserves the base module's periods and layout exactly.

    Args:
        base: Existing AxialRoPE2D instance from a loaded model.
        ref_h_tokens: Reference H token length (L_ref,h).
        ref_w_tokens: Reference W token length (L_ref,w).
        beta_hi_u: Unconstrained u for beta_hi.
        beta_lo_u: Unconstrained u for beta_lo.
        beta_bend_u: Unconstrained u for beta_bend (mid-band bump).
        beta_max: Maximum absolute beta value (> 0). Higher increases control.

    Returns:
        An AxialRoPE2D instance whose forward applies the inference-only warp.
    """
    if not isinstance(base, AxialRoPE2D):
        raise TypeError("base must be an AxialRoPE2D")
    if base.cfg.coord_mode is not AxialRoPE2DCoordMode.PATCH_INDICES:
        raise ValueError(
            "Inference freq-warp requires base.cfg.coord_mode=PATCH_INDICES"
        )
    if int(ref_h_tokens) <= 0 or int(ref_w_tokens) <= 0:
        raise ValueError("ref_h_tokens and ref_w_tokens must be positive")
    hi_u = float(beta_hi_u)
    lo_u = float(beta_lo_u)
    bend_u = float(beta_bend_u)
    if not math.isfinite(hi_u):
        raise ValueError("beta_hi_u must be finite")
    if not math.isfinite(lo_u):
        raise ValueError("beta_lo_u must be finite")
    if not math.isfinite(bend_u):
        raise ValueError("beta_bend_u must be finite")
    bmax = float(beta_max)
    if not math.isfinite(bmax) or bmax <= 0.0:
        raise ValueError("beta_max must be finite and > 0")

    class _AxialRoPE2DInferenceWarp(AxialRoPE2D):
        """Inference-only axial RoPE variant with beta-curve knobs."""

        def __init__(self, *, device: torch.device) -> None:
            super().__init__(head_dim=int(base.head_dim), cfg=base.cfg)
            self.ref_h_tokens: Final[int] = int(ref_h_tokens)
            self.ref_w_tokens: Final[int] = int(ref_w_tokens)
            # Store as buffers so the notebook can mutate by replacing the module.
            self.register_buffer(
                "beta_hi_u",
                torch.tensor(float(hi_u), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "beta_lo_u",
                torch.tensor(float(lo_u), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "beta_bend_u",
                torch.tensor(float(bend_u), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "beta_max",
                torch.tensor(float(bmax), dtype=torch.float32),
                persistent=False,
            )
            self.to(device=device)
            with torch.no_grad():
                self.periods.copy_(
                    base.periods.detach().to(device=device, dtype=torch.float32)
                )

        def forward(
            self,
            *,
            H: int,
            W: int,
            scales: Tensor | None,
        ) -> tuple[Tensor, Tensor]:
            if scales is not None:
                raise ValueError("Inference freq-warp does not support dilation scales")
            if int(H) <= 0 or int(W) <= 0:
                raise ValueError("H and W must be positive for axial RoPE")
            device = self.periods.device
            offset = float(self.cfg.coord_offset)
            coords = _get_patch_index_coords(
                int(H), int(W), device=device, offset=offset
            )
            if coords.dim() != 2 or coords.shape[1] != 2:
                raise RuntimeError("Axial RoPE coords must have shape [HW, 2]")

            qtr = int(self.periods.numel())
            if qtr <= 0:
                raise RuntimeError("Axial RoPE periods length must be positive")

            beta_max_t = cast("Tensor", self.beta_max).to(
                device=device, dtype=torch.float32
            )
            beta_hi = beta_max_t * torch.tanh(
                cast("Tensor", self.beta_hi_u).to(device=device, dtype=torch.float32)
            )
            beta_lo = beta_max_t * torch.tanh(
                cast("Tensor", self.beta_lo_u).to(device=device, dtype=torch.float32)
            )
            beta_bend = beta_max_t * torch.tanh(
                cast("Tensor", self.beta_bend_u).to(device=device, dtype=torch.float32)
            )

            if qtr == 1:
                beta = beta_hi[None]
            else:
                t = torch.arange(int(qtr), device=device, dtype=torch.float32) / float(
                    qtr - 1
                )
                bump = 4.0 * t * (1.0 - t)
                beta = (1.0 - t) * beta_hi + t * beta_lo + beta_bend * bump

            s_h = float(int(H)) / float(int(self.ref_h_tokens))
            s_w = float(int(W)) / float(int(self.ref_w_tokens))
            if (
                not math.isfinite(s_h)
                or s_h <= 0.0
                or not math.isfinite(s_w)
                or s_w <= 0.0
            ):
                raise ValueError(
                    "H/ref_h_tokens and W/ref_w_tokens must be finite and > 0"
                )

            periods_h = self.periods * torch.pow(
                torch.tensor(s_h, device=device, dtype=torch.float32), beta
            )
            periods_w = self.periods * torch.pow(
                torch.tensor(s_w, device=device, dtype=torch.float32), beta
            )
            axis_periods = torch.stack([periods_h, periods_w], dim=0)  # [2, Q]

            angles = (
                float(self.cfg.angle_multiplier)
                * coords[:, :, None].to(dtype=torch.float32)
                / axis_periods[None, :, :].to(dtype=torch.float32)
            )
            match self.cfg.dim_layout:
                case AxialRoPE2DDimLayout.HALF_SPLIT:
                    angles = angles.flatten(1, 2).repeat(1, 2)
                case AxialRoPE2DDimLayout.PAIR_INTERLEAVED:
                    angles = angles.repeat_interleave(2, dim=-1).flatten(1, 2)
                case _ as unreachable:  # pragma: no cover - defensive
                    raise RuntimeError(f"Unsupported dim_layout: {unreachable}")
            if angles.shape != (int(H) * int(W), int(self.head_dim)):
                raise RuntimeError(
                    "Unexpected angles shape in inference freq-warp: "
                    f"{tuple(angles.shape)} for H={int(H)} W={int(W)}"
                )
            return torch.sin(angles), torch.cos(angles)

    return _AxialRoPE2DInferenceWarp(device=base.periods.device)


class AxialRoPE2D(nn.Module):
    """DINOv3-style axial 2D RoPE sin/cos generator.

    The base periods are fixed by ``AxialRoPE2DConfig``. Optionally, this module
    can include learnable scalar parameters when using:
      - ``frequency_aware`` (boundary_log_multiplier), or
      - ``beta_warp`` (beta_hi_u/beta_lo_u/beta_bend_u), or
      - ``alpha_warp`` (alpha per-band exponents).
    """

    periods: Tensor

    def __init__(self, *, head_dim: int, cfg: AxialRoPE2DConfig) -> None:
        super().__init__()
        if int(head_dim) <= 0:
            raise ValueError("head_dim must be positive for AxialRoPE2D")
        if int(head_dim) % 4 != 0:
            raise ValueError(
                "AxialRoPE2D requires head_dim % 4 == 0 (DINOv3 constraint); "
                f"got head_dim={int(head_dim)}"
            )
        if not isinstance(cfg, AxialRoPE2DConfig):
            raise TypeError("cfg must be an AxialRoPE2DConfig for AxialRoPE2D")
        self.head_dim: Final[int] = int(head_dim)
        self.cfg: Final[AxialRoPE2DConfig] = cfg
        self._d_head: Final[int] = self.head_dim
        self.register_buffer(
            "periods",
            torch.empty(self._d_head // 4, dtype=torch.float32),
            persistent=True,
        )
        if cfg.frequency_aware is None:
            self.register_parameter("boundary_log_multiplier", None)
        else:
            init = float(cfg.frequency_aware.boundary_log_multiplier_init)
            self.boundary_log_multiplier = nn.Parameter(
                torch.tensor(init, dtype=torch.float32),
                requires_grad=True,
            )
        if cfg.beta_warp is None:
            self.register_parameter("beta_hi_u", None)
            self.register_parameter("beta_lo_u", None)
            self.register_parameter("beta_bend_u", None)
        else:
            beta = cfg.beta_warp
            self.beta_hi_u = nn.Parameter(
                torch.tensor(float(beta.beta_hi_u_init), dtype=torch.float32),
                requires_grad=True,
            )
            self.beta_lo_u = nn.Parameter(
                torch.tensor(float(beta.beta_lo_u_init), dtype=torch.float32),
                requires_grad=True,
            )
            self.beta_bend_u = nn.Parameter(
                torch.tensor(float(beta.beta_bend_u_init), dtype=torch.float32),
                requires_grad=True,
            )
        if cfg.alpha_warp is None:
            self.register_parameter("alpha", None)
        else:
            qtr = int(self._d_head) // 4
            if qtr <= 0:  # pragma: no cover - defensive
                raise RuntimeError("AxialRoPE2D periods length must be positive")
            init = float(cfg.alpha_warp.alpha_init)
            if not math.isfinite(init):
                raise RuntimeError("alpha_init must be finite for alpha-warp RoPE")
            self.alpha = nn.Parameter(
                torch.full((int(qtr),), init, dtype=torch.float32),
                requires_grad=True,
            )
        self._init_periods()

    def _apply(self, fn):  # type: ignore[override]
        out = super()._apply(fn)
        with torch.no_grad():
            self.periods.data = self.periods.data.to(dtype=torch.float32)
            if self.boundary_log_multiplier is not None:
                self.boundary_log_multiplier.data = (
                    self.boundary_log_multiplier.data.to(dtype=torch.float32)
                )
            if self.beta_hi_u is not None:
                self.beta_hi_u.data = self.beta_hi_u.data.to(dtype=torch.float32)
            if self.beta_lo_u is not None:
                self.beta_lo_u.data = self.beta_lo_u.data.to(dtype=torch.float32)
            if self.beta_bend_u is not None:
                self.beta_bend_u.data = self.beta_bend_u.data.to(dtype=torch.float32)
            if self.alpha is not None:
                self.alpha.data = self.alpha.data.to(dtype=torch.float32)
        return out

    def _init_periods(self) -> None:
        """Initialize per-dimension periods using DINOv3 formulas."""
        device: torch.device = self.periods.device
        dtype: torch.dtype = self.periods.dtype
        d_head = int(self._d_head)
        qtr = d_head // 4
        if qtr <= 0:
            raise RuntimeError("AxialRoPE2D periods length must be positive")
        if self.cfg.base is not None:
            base = float(self.cfg.base)
            exponents = (
                2.0
                * torch.arange(int(qtr), device=device, dtype=dtype)
                / float(d_head // 2)
            )
            periods = torch.tensor(base, device=device, dtype=dtype) ** exponents
        else:
            if self.cfg.min_period is None or self.cfg.max_period is None:
                raise RuntimeError(
                    "AxialRoPE2DConfig must provide min_period and max_period when base is None"
                )
            min_p = float(self.cfg.min_period)
            max_p = float(self.cfg.max_period)
            base = max_p / min_p
            exponents = torch.linspace(0.0, 1.0, int(qtr), device=device, dtype=dtype)
            periods = torch.tensor(base, device=device, dtype=dtype) ** exponents
            periods = periods / torch.tensor(base, device=device, dtype=dtype)
            periods = periods * torch.tensor(max_p, device=device, dtype=dtype)
        self.periods.data = periods

    def forward(
        self,
        *,
        H: int,
        W: int,
        scales: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return (sin, cos) buffers for axial 2D RoPE.

        Args:
            H: Patch-grid height.
            W: Patch-grid width.
            scales: Optional per-batch dilation scale (scalar tensor). When
                None, returns shared sin/cos shaped ``[HW, head_dim]``. When
                provided, applies the scalar dilation and still returns shared
                sin/cos shaped ``[HW, head_dim]``.
        """
        if int(H) <= 0 or int(W) <= 0:
            raise ValueError("H and W must be positive for AxialRoPE2D forward")
        device = self.periods.device
        offset = float(self.cfg.coord_offset)
        coords: Tensor
        match self.cfg.coord_mode:
            case AxialRoPE2DCoordMode.DINOV3_NORMALIZED:
                coords = _get_dinov3_normalized_coords(
                    int(H),
                    int(W),
                    device=device,
                    normalize=self.cfg.normalize_coords,
                    offset=offset,
                )
            case AxialRoPE2DCoordMode.PATCH_INDICES:
                coords = _get_patch_index_coords(
                    int(H), int(W), device=device, offset=offset
                )
            case _ as unreachable:  # pragma: no cover - defensive
                raise RuntimeError(f"Unsupported coord_mode: {unreachable}")
        if coords.dim() != 2 or coords.shape[1] != 2:
            raise RuntimeError("AxialRoPE2D coords must have shape [HW, 2]")
        if self.cfg.frequency_aware is not None:
            if scales is not None:
                raise ValueError(
                    "frequency-aware axial RoPE does not support dilation scales"
                )
            if self.boundary_log_multiplier is None:
                raise RuntimeError(
                    "boundary_log_multiplier parameter missing for frequency-aware RoPE"
                )
            ref_h = int(self.cfg.frequency_aware.ref_h_tokens)
            ref_w = int(self.cfg.frequency_aware.ref_w_tokens)
            periods_h = lumina_frequency_aware_periods_for_axis(
                periods=self.periods,
                axis_len=int(H),
                ref_axis_len=ref_h,
                boundary_log_multiplier=self.boundary_log_multiplier,
                angle_multiplier=float(self.cfg.angle_multiplier),
            )
            periods_w = lumina_frequency_aware_periods_for_axis(
                periods=self.periods,
                axis_len=int(W),
                ref_axis_len=ref_w,
                boundary_log_multiplier=self.boundary_log_multiplier,
                angle_multiplier=float(self.cfg.angle_multiplier),
            )
            axis_periods = torch.stack([periods_h, periods_w], dim=0)  # [2, Q]
        elif self.cfg.beta_warp is not None:
            if scales is not None:
                raise ValueError(
                    "beta-warp axial RoPE does not support dilation scales"
                )
            if (
                self.beta_hi_u is None
                or self.beta_lo_u is None
                or self.beta_bend_u is None
            ):
                raise RuntimeError("beta warp parameters missing for beta-warp RoPE")
            beta_cfg = self.cfg.beta_warp
            ref_h = int(beta_cfg.ref_h_tokens)
            ref_w = int(beta_cfg.ref_w_tokens)
            qtr = int(self.periods.numel())
            if qtr <= 0:  # pragma: no cover - defensive (checked elsewhere)
                raise RuntimeError("AxialRoPE2D periods length must be positive")
            beta_max = float(beta_cfg.beta_max)
            if not math.isfinite(beta_max) or beta_max <= 0.0:
                raise RuntimeError("beta_max must be finite and > 0")
            beta_max_t = torch.tensor(beta_max, device=device, dtype=torch.float32)
            beta_hi = beta_max_t * torch.tanh(self.beta_hi_u.to(dtype=torch.float32))
            beta_lo = beta_max_t * torch.tanh(self.beta_lo_u.to(dtype=torch.float32))
            beta_bend = beta_max_t * torch.tanh(
                self.beta_bend_u.to(dtype=torch.float32)
            )
            if qtr == 1:
                beta = beta_hi[None]
            else:
                t = torch.arange(int(qtr), device=device, dtype=torch.float32) / float(
                    qtr - 1
                )
                bump = 4.0 * t * (1.0 - t)
                beta = (1.0 - t) * beta_hi + t * beta_lo + beta_bend * bump

            s_h = float(int(H)) / float(ref_h)
            s_w = float(int(W)) / float(ref_w)
            if (
                not math.isfinite(s_h)
                or s_h <= 0.0
                or not math.isfinite(s_w)
                or s_w <= 0.0
            ):
                raise RuntimeError(
                    "Computed invalid axis scale factors for beta-warp RoPE"
                )
            periods_h = self.periods.to(dtype=torch.float32) * torch.pow(
                torch.tensor(s_h, device=device, dtype=torch.float32), beta
            )
            periods_w = self.periods.to(dtype=torch.float32) * torch.pow(
                torch.tensor(s_w, device=device, dtype=torch.float32), beta
            )
            axis_periods = torch.stack([periods_h, periods_w], dim=0)  # [2, Q]
        elif self.cfg.alpha_warp is not None:
            if scales is not None:
                raise ValueError(
                    "alpha-warp axial RoPE does not support dilation scales"
                )
            if self.alpha is None:
                raise RuntimeError("alpha parameter missing for alpha-warp RoPE")
            alpha_cfg = self.cfg.alpha_warp
            ref_h = int(alpha_cfg.ref_h_tokens)
            ref_w = int(alpha_cfg.ref_w_tokens)
            qtr = int(self.periods.numel())
            if int(self.alpha.numel()) != qtr:
                raise RuntimeError(
                    "alpha length must match RoPE periods length for alpha-warp RoPE"
                )
            s_h = float(int(H)) / float(ref_h)
            s_w = float(int(W)) / float(ref_w)
            if (
                not math.isfinite(s_h)
                or s_h <= 0.0
                or not math.isfinite(s_w)
                or s_w <= 0.0
            ):
                raise RuntimeError(
                    "Computed invalid axis scale factors for alpha-warp RoPE"
                )
            alpha = self.alpha.to(device=device, dtype=torch.float32)
            scale_h = torch.pow(
                torch.tensor(s_h, device=device, dtype=torch.float32), alpha
            )
            scale_w = torch.pow(
                torch.tensor(s_w, device=device, dtype=torch.float32), alpha
            )
            periods_h = self.periods.to(dtype=torch.float32) / scale_h
            periods_w = self.periods.to(dtype=torch.float32) / scale_w
            axis_periods = torch.stack([periods_h, periods_w], dim=0)  # [2, Q]
        else:
            axis_periods = self.periods[None, :].expand(2, -1).to(dtype=torch.float32)

        # Angles: angle_multiplier * coords / periods, flattened and tiled.
        angles = (
            float(self.cfg.angle_multiplier)
            * coords[:, :, None].to(dtype=torch.float32)
            / axis_periods[None, :, :].to(dtype=torch.float32)
        )
        match self.cfg.dim_layout:
            case AxialRoPE2DDimLayout.HALF_SPLIT:
                angles = angles.flatten(1, 2).repeat(1, 2)
            case AxialRoPE2DDimLayout.PAIR_INTERLEAVED:
                angles = angles.repeat_interleave(2, dim=-1).flatten(1, 2)
            case _ as unreachable:  # pragma: no cover - defensive
                raise RuntimeError(f"Unsupported dim_layout: {unreachable}")
        if angles.shape != (int(H) * int(W), int(self._d_head)):
            raise RuntimeError(
                "Unexpected angles shape in AxialRoPE2D: "
                f"{tuple(angles.shape)} for H={int(H)} W={int(W)}"
            )
        if scales is not None:
            if scales.dim() != 0:
                raise ValueError(
                    "AxialRoPE2D scales must be a scalar tensor for per-batch dilation; "
                    "per-sample dilation is not supported"
                )
            angles = angles * scales.to(device=device, dtype=torch.float32)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return sin, cos


def _dy_ntk_periods_for_axis(
    *,
    periods: Tensor,
    axis_len: int,
    ref_axis_len: int,
    noise_time: Tensor,
    lambda_s: float,
    lambda_t: float,
) -> Tensor:
    """Return Dy-NTK periods for one spatial axis.

    Raises:
        ValueError: If token lengths or scheduler values are invalid.
    """

    if int(axis_len) <= 0 or int(ref_axis_len) <= 0:
        raise ValueError("axis_len and ref_axis_len must be positive for Dy-NTK")
    qtr = int(periods.numel())
    if qtr <= 0:
        raise ValueError("periods must be non-empty for Dy-NTK")
    scale = float(int(axis_len)) / float(int(ref_axis_len))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Dy-NTK axis scale must be finite and > 0")
    return _dy_ntk_periods_for_scale(
        periods=periods,
        scale=scale,
        noise_time=noise_time,
        lambda_s=float(lambda_s),
        lambda_t=float(lambda_t),
    )


def _dy_ntk_periods_for_scale(
    *,
    periods: Tensor,
    scale: float,
    noise_time: Tensor,
    lambda_s: float,
    lambda_t: float,
) -> Tensor:
    """Return Dy-NTK periods for a precomputed axis scale."""

    axis_scale = float(scale)
    if not math.isfinite(axis_scale) or axis_scale <= 0.0:
        raise ValueError("Dy-NTK scale must be finite and > 0")
    qtr = int(periods.numel())
    if qtr <= 0:
        raise ValueError("periods must be non-empty for Dy-NTK")
    if scale <= 1.0:
        return periods.to(dtype=torch.float32)
    if qtr == 1:
        exponent = torch.zeros((1,), device=periods.device, dtype=torch.float32)
    else:
        exponent = torch.arange(qtr, device=periods.device, dtype=torch.float32) / (
            float(qtr - 1)
        )
    kappa = float(lambda_s) * torch.pow(
        noise_time.to(device=periods.device, dtype=torch.float32),
        float(lambda_t),
    )
    return periods.to(dtype=torch.float32) * torch.pow(
        torch.tensor(axis_scale, device=periods.device, dtype=torch.float32),
        kappa * exponent,
    )


def _dype_dynamic_exponent(
    *, noise_time: float, lambda_s: float, lambda_t: float
) -> float:
    """Return Comfy/DyPE-style dynamic magnitude for normalized noise time."""

    noise = float(noise_time)
    if not math.isfinite(noise):
        raise ValueError("DyPE noise_time must be finite")
    noise = max(0.0, min(1.0, noise))
    scale = float(lambda_s)
    exponent = float(lambda_t)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("DyPE lambda_s must be finite and > 0")
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("DyPE lambda_t must be finite and > 0")
    return scale * (noise**exponent)


def _dype_correction_factor(
    *,
    periods: Tensor,
    rotations: float,
    ref_axis_len: int,
    angle_multiplier: float,
) -> float:
    """Return fractional band index whose wavelength makes ``rotations`` turns."""

    if int(ref_axis_len) <= 0:
        raise ValueError("ref_axis_len must be positive for DyPE correction")
    rot = float(rotations)
    if not math.isfinite(rot) or rot <= 0.0:
        raise ValueError("rotations must be finite and > 0")
    mult = float(angle_multiplier)
    if not math.isfinite(mult) or mult <= 0.0:
        raise ValueError("angle_multiplier must be finite and > 0")
    if int(periods.numel()) < 2:
        return 0.0
    periods_cpu = periods.detach().to(device=torch.device("cpu"), dtype=torch.float32)
    p0 = float(periods_cpu[0].item())
    p1 = float(periods_cpu[-1].item())
    if p0 <= 0.0 or p1 <= p0:
        raise ValueError("periods must be positive and strictly increasing for DyPE")
    boundary_wavelength = float(int(ref_axis_len)) / rot
    boundary_period = (mult / (2.0 * float(math.pi))) * boundary_wavelength
    log_p0 = math.log(p0)
    log_p1 = math.log(p1)
    return float(periods.numel() - 1) * (
        (math.log(boundary_period) - log_p0) / (log_p1 - log_p0)
    )


def _dype_ramp_mask(
    *,
    periods: Tensor,
    threshold_high_rotations: float,
    threshold_low_rotations: float,
    ref_axis_len: int,
    angle_multiplier: float,
) -> Tensor:
    """Return YaRN's high-to-low band mask for one dynamic threshold pair."""

    qtr = int(periods.numel())
    if qtr <= 0:
        raise ValueError("periods must be non-empty for DyPE ramp mask")
    device = periods.device
    if qtr == 1:
        return torch.ones((1,), device=device, dtype=torch.float32)
    low = math.floor(
        _dype_correction_factor(
            periods=periods,
            rotations=float(threshold_high_rotations),
            ref_axis_len=int(ref_axis_len),
            angle_multiplier=float(angle_multiplier),
        )
    )
    high = math.ceil(
        _dype_correction_factor(
            periods=periods,
            rotations=float(threshold_low_rotations),
            ref_axis_len=int(ref_axis_len),
            angle_multiplier=float(angle_multiplier),
        )
    )
    low = max(0, min(qtr - 1, int(low)))
    high = max(0, min(qtr, int(high)))
    if low == high:
        high = min(qtr, low + 1)
    band = torch.arange(qtr, device=device, dtype=torch.float32)
    ramp = (band - float(low)) / float(high - low)
    return 1.0 - torch.clamp(ramp, min=0.0, max=1.0)


def _dy_yarn_periods_for_axis(
    *,
    periods: Tensor,
    linear_scale: float,
    ntk_scale: float,
    ref_axis_len: int,
    noise_time: float,
    lambda_s: float,
    cfg: AxialRoPE2DDyPEConfig,
    angle_multiplier: float,
) -> Tensor:
    """Return Dy-YaRN periods for one spatial axis."""

    if int(ref_axis_len) <= 0:
        raise ValueError("ref_axis_len must be positive for Dy-YaRN")
    linear_s = float(linear_scale)
    ntk_s = float(ntk_scale)
    if (
        not math.isfinite(linear_s)
        or linear_s <= 0.0
        or not math.isfinite(ntk_s)
        or ntk_s <= 0.0
    ):
        raise ValueError("Dy-YaRN axis scales must be finite and > 0")
    periods_f = periods.to(dtype=torch.float32)
    if max(linear_s, ntk_s) <= 1.0:
        return periods_f
    kappa = _dype_dynamic_exponent(
        noise_time=float(noise_time),
        lambda_s=float(lambda_s),
        lambda_t=float(cfg.lambda_t),
    )
    if kappa <= 1e-6:
        return periods_f
    freq_base = float(angle_multiplier) / periods_f
    freq_linear = float(angle_multiplier) / (periods_f * max(1.0, linear_s))
    periods_ntk = _dy_ntk_periods_for_scale(
        periods=periods_f,
        scale=max(1.0, ntk_s),
        noise_time=torch.ones((), device=periods.device, dtype=torch.float32),
        lambda_s=1.0,
        lambda_t=1.0,
    )
    freq_ntk = float(angle_multiplier) / periods_ntk

    beta_mask = _dype_ramp_mask(
        periods=periods_f,
        threshold_high_rotations=float(cfg.yarn_beta_0) ** kappa,
        threshold_low_rotations=float(cfg.yarn_beta_1) ** kappa,
        ref_axis_len=int(ref_axis_len),
        angle_multiplier=float(angle_multiplier),
    )
    freq = freq_linear * (1.0 - beta_mask) + freq_ntk * beta_mask

    gamma_mask = _dype_ramp_mask(
        periods=periods_f,
        threshold_high_rotations=float(cfg.yarn_gamma_0) ** kappa,
        threshold_low_rotations=float(cfg.yarn_gamma_1) ** kappa,
        ref_axis_len=int(ref_axis_len),
        angle_multiplier=float(angle_multiplier),
    )
    freq = freq * (1.0 - gamma_mask) + freq_base * gamma_mask
    return float(angle_multiplier) / freq


class AxialRoPE2DDyPE(AxialRoPE2D):
    """Inference-only axial RoPE wrapper using dynamic position extrapolation."""

    dype_cfg: AxialRoPE2DDyPEConfig
    dype_noise_time: Tensor
    dype_noise_time_values: list[float]

    def __init__(self, *, base: AxialRoPE2D, cfg: AxialRoPE2DDyPEConfig) -> None:
        if not isinstance(base, AxialRoPE2D):
            raise TypeError("base must be an AxialRoPE2D")
        if not isinstance(cfg, AxialRoPE2DDyPEConfig):
            raise TypeError("cfg must be an AxialRoPE2DDyPEConfig")
        if base.cfg.coord_mode is not AxialRoPE2DCoordMode.PATCH_INDICES:
            raise ValueError("DyPE requires patch-index axial RoPE coordinates")
        super().__init__(head_dim=int(base.head_dim), cfg=base.cfg)
        self.dype_cfg = cfg  # ty: ignore[unresolved-attribute]
        self.register_buffer(
            "dype_noise_time",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        self.dype_noise_time_values: list[float] = [1.0]
        with torch.no_grad():
            self.periods.copy_(base.periods.detach().to(dtype=torch.float32))

    def set_dype_noise_time(self, noise_time: float) -> None:
        """Set the current normalized diffusion noise time in ``[0, 1]``."""

        t = float(noise_time)
        if not math.isfinite(t) or t < 0.0 or t > 1.0:
            raise ValueError("DyPE noise_time must be finite and within [0, 1]")
        self.dype_noise_time.fill_(t)
        self.dype_noise_time_values[0] = t

    def _dype_axis_periods(
        self,
        *,
        axis_len: int,
        ref_axis_len: int,
        global_scale: float,
        lambda_s: float,
    ) -> Tensor:
        """Return method-specific periods for one spatial axis."""

        cfg = self.dype_cfg
        axis_scale = float(int(axis_len)) / float(int(ref_axis_len))
        shared_scale = float(global_scale)
        if (
            not math.isfinite(axis_scale)
            or axis_scale <= 0.0
            or not math.isfinite(shared_scale)
            or shared_scale <= 0.0
        ):
            raise ValueError("DyPE axis and global scales must be finite and > 0")
        match cfg.method:
            case DyPERoPEMethod.DY_NTK:
                return _dy_ntk_periods_for_scale(
                    periods=self.periods,
                    scale=shared_scale,
                    noise_time=self.dype_noise_time,
                    lambda_s=float(lambda_s),
                    lambda_t=float(cfg.lambda_t),
                )
            case DyPERoPEMethod.VISION_YARN:
                return _dy_yarn_periods_for_axis(
                    periods=self.periods,
                    linear_scale=axis_scale,
                    ntk_scale=shared_scale,
                    ref_axis_len=int(ref_axis_len),
                    noise_time=float(self.dype_noise_time_values[0]),
                    lambda_s=float(lambda_s),
                    cfg=cfg,
                    angle_multiplier=float(self.cfg.angle_multiplier),
                )
            case DyPERoPEMethod.DY_YARN:
                return _dy_yarn_periods_for_axis(
                    periods=self.periods,
                    linear_scale=shared_scale,
                    ntk_scale=shared_scale,
                    ref_axis_len=int(ref_axis_len),
                    noise_time=float(self.dype_noise_time_values[0]),
                    lambda_s=float(lambda_s),
                    cfg=cfg,
                    angle_multiplier=float(self.cfg.angle_multiplier),
                )
            case _ as unreachable:
                raise RuntimeError(f"Unsupported DyPE method: {unreachable}")

    def forward(
        self,
        *,
        H: int,
        W: int,
        scales: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return timestep-aware DyPE sin/cos buffers."""

        if scales is not None:
            raise ValueError("DyPE axial RoPE does not support dilation scales")
        if int(H) <= 0 or int(W) <= 0:
            raise ValueError("H and W must be positive for DyPE axial RoPE")
        device = self.periods.device
        coords = _get_patch_index_coords(
            int(H), int(W), device=device, offset=float(self.cfg.coord_offset)
        )
        scale_h = float(int(H)) / float(int(self.dype_cfg.ref_h_tokens))
        scale_w = float(int(W)) / float(int(self.dype_cfg.ref_w_tokens))
        global_scale = max(scale_h, scale_w)
        periods_h = self._dype_axis_periods(
            axis_len=int(H),
            ref_axis_len=int(self.dype_cfg.ref_h_tokens),
            global_scale=global_scale,
            lambda_s=float(self.dype_cfg.lambda_s),
        )
        periods_w = self._dype_axis_periods(
            axis_len=int(W),
            ref_axis_len=int(self.dype_cfg.ref_w_tokens),
            global_scale=global_scale,
            lambda_s=float(self.dype_cfg.lambda_s),
        )
        axis_periods = torch.stack([periods_h, periods_w], dim=0)
        angles = (
            float(self.cfg.angle_multiplier)
            * coords[:, :, None].to(dtype=torch.float32)
            / axis_periods[None, :, :].to(dtype=torch.float32)
        )
        match self.cfg.dim_layout:
            case AxialRoPE2DDimLayout.HALF_SPLIT:
                angles = angles.flatten(1, 2).repeat(1, 2)
            case AxialRoPE2DDimLayout.PAIR_INTERLEAVED:
                angles = angles.repeat_interleave(2, dim=-1).flatten(1, 2)
            case _ as unreachable:
                raise RuntimeError(f"Unsupported dim_layout: {unreachable}")
        expected_shape = (int(H) * int(W), int(self.head_dim))
        if angles.shape != expected_shape:
            raise RuntimeError(
                "Unexpected angles shape in DyPE axial RoPE: "
                f"{tuple(angles.shape)} for expected {expected_shape}"
            )
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        if (
            self.dype_cfg.method in (DyPERoPEMethod.VISION_YARN, DyPERoPEMethod.DY_YARN)
            and bool(self.dype_cfg.yarn_attention_scale)
            and global_scale > 1.0
        ):
            match self.dype_cfg.method:
                case DyPERoPEMethod.VISION_YARN:
                    mscale_start = 0.1 * math.log(global_scale) + 1.0
                    kappa = _dype_dynamic_exponent(
                        noise_time=float(self.dype_noise_time_values[0]),
                        lambda_s=1.0,
                        lambda_t=float(self.dype_cfg.lambda_t),
                    )
                    mscale = 1.0 + (mscale_start - 1.0) * kappa
                case DyPERoPEMethod.DY_YARN:
                    mscale = 1.0 + 0.1 * math.log(global_scale) / math.sqrt(
                        global_scale
                    )
                case _ as unreachable:  # pragma: no cover - guarded above
                    raise RuntimeError(
                        f"Unsupported YaRN attention scale: {unreachable}"
                    )
            if mscale > 1.0:
                sin = sin * float(mscale)
                cos = cos * float(mscale)
        return sin, cos


def build_axial_rope2d_dype(
    *, base: AxialRoPE2D, cfg: AxialRoPE2DDyPEConfig
) -> AxialRoPE2DDyPE:
    """Build an inference-only DyPE wrapper for an existing axial RoPE."""

    return AxialRoPE2DDyPE(base=base, cfg=cfg).to(device=base.periods.device)


def set_axial_rope2d_dype_noise_time(module: nn.Module, *, noise_time: float) -> bool:
    """Set DyPE noise time on all axial DyPE modules inside ``module``."""

    updated = False
    for child in module.modules():
        match child:
            case AxialRoPE2DDyPE() as dype:
                dype.set_dype_noise_time(float(noise_time))
                updated = True
            case _:
                pass
    return updated
