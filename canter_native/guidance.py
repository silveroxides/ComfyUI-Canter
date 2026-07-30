from __future__ import annotations

from dataclasses import dataclass

import torch

PDG_MODES = (
    "none",
    "full",
    "three_quarter",
    "alternate_pdg_first",
    "alternate_cfg_first",
    "combined_cfg_pdg",
    "pdg_with_alternating_cfg",
    "cfg_to_pdg",
)


def inside(step: int, start: int, stop: int) -> bool:
    return start <= step <= stop


def pdg_scales(curve: str, noisy: float, clean: float, power: float, steps: int):
    if steps < 1 or power <= 0:
        raise ValueError("steps and PDG power must be positive")
    position = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    if curve == "constant":
        if noisy != clean:
            raise ValueError("constant PDG requires equal noisy and clean scales")
        values = torch.full_like(position, noisy)
    elif curve == "linear":
        values = noisy + (clean - noisy) * position
    elif curve == "power":
        values = noisy + (clean - noisy) * position.pow(power)
    else:
        raise ValueError(f"unsupported PDG curve: {curve}")
    return tuple(float(x) for x in values)


def mode_uses_cfg(mode: str) -> bool:
    return mode in {
        "alternate_pdg_first",
        "alternate_cfg_first",
        "combined_cfg_pdg",
        "pdg_with_alternating_cfg",
        "cfg_to_pdg",
    }


@dataclass(frozen=True)
class Route:
    main: str
    tweak: str | None
    combine: str


def route(mode: str, step: int, cfg_active: bool, pdg_active: bool) -> Route:
    if mode not in PDG_MODES:
        raise ValueError(f"unsupported Canter guidance mode: {mode}")
    if not pdg_active or mode == "none":
        return Route("full", "unconditional" if cfg_active else None, "cfg")
    path = "three_quarter" if mode == "three_quarter" else "skip_middle"
    if mode in {"full", "three_quarter"}:
        return Route("full", path, "pdg")
    if mode == "combined_cfg_pdg":
        return Route("full", "both", "combined")
    if mode == "pdg_with_alternating_cfg":
        return Route("full", "unconditional" if step % 2 else path, "cfg" if step % 2 else "pdg")
    if mode == "cfg_to_pdg":
        return Route("full", "unconditional" if cfg_active else path, "cfg" if cfg_active else "pdg")
    cfg_turn = (step % 2 == 0) == (mode == "alternate_cfg_first")
    return Route("full", "unconditional" if cfg_turn else path, "cfg" if cfg_turn else "pdg")


def combine(main, tweak, scale: float):
    return tweak + scale * (main - tweak)
