from __future__ import annotations

from enum import Enum


class MLPType(Enum):
    """MLP implementation variants for DiT blocks.

    - SWI: baseline SwiGLU MLP
    - SWINE: Sigmoid-gated sine GLU (σ·sin) with trig promoted to float32
    - SWINER: Sigmoid-gated FINER-style chirp (σ·sin(ω₀·((1+|x|)·x)))
    - SPWIDER: sqrt-gated sine GLU (√|a|·sin(ω₀·b))
    - RELU: Plain ReLU-activated feedforward
    - RELU2: ReLU-squared activation (ReLU(x)^2) feedforward
    - SILU: Plain SiLU-activated feedforward
    - GELU: Plain GELU-activated feedforward
    - SIREN: Pure sine-activated MLP
    - SPIDER: Sine with sqrt magnitude (sin(ω₀·x)·√|x|)
    - SINC: Sinc-activated MLP with log-spaced per-channel scales
    - FINER: FINER activation MLP with a fixed global scale (non-learnable)
    - RBF: Low-rank per-patch RBF with Gaussian kernel
    - RBF_ODD: RBF with odd-Gaussian kernel (z·exp(-z^2))
    - RBF_SHARP: RBF with sharpness exponent alpha (exp(-(s·|x-b|)^alpha))
    - RBF_SIREN: RBF using sine basis sin(ω0·(s·(x-b)))
    - RBF_FINER: RBF using FINER (chirp) basis sin(ω0·((1+|z|)·z)), z=s·(x-b)
    - RBF_DAMPED_SINE: RBF using damped sine sin(ω0·z)·exp(-|z|), z=s·(x-b)
    - RBF_SINC: RBF using sinc basis sinc(z)=sin(z)/z with z=s·(x-b)
    """

    SWI = "swi"
    SWINE = "swine"
    SWINER = "swiner"
    SPWIDER = "spwider"
    RELU = "relu"
    RELU2 = "relu2"
    SILU = "silu"
    GELU = "gelu"
    SIREN = "siren"
    SPIDER = "spider"
    SINC = "sinc"
    FINER = "finer"
    RBF = "rbf"
    RBF_ODD = "rbf_odd"
    RBF_SHARP = "rbf_sharp"
    RBF_SIREN = "rbf_siren"
    RBF_FINER = "rbf_finer"
    RBF_DAMPED_SINE = "rbf_damped_sine"
    RBF_SINC = "rbf_sinc"


__all__ = ["MLPType"]
