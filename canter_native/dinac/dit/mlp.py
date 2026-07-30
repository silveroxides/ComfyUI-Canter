"""Small MLP factory for DINAC-AE DiT blocks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import torch.nn.functional as F
from torch import Tensor, nn

from ..dit.mlp_types import MLPType


class Resettable(Protocol):
    """Typing protocol for modules with ``reset_parameters``."""

    def reset_parameters(self) -> None:
        """Reset module parameters."""


def reset_module_parameters(module: nn.Module) -> None:
    """Reset a module that exposes ``reset_parameters``."""

    cast(Resettable, module).reset_parameters()


class SimpleActivationMLP(nn.Module):
    """Feedforward MLP: ``down(activation(up(x)))``."""

    in_features: int
    hidden_features: int
    activation: Callable[[Tensor], Tensor]
    activation_name: str
    up: nn.Linear
    down: nn.Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        activation: Callable[[Tensor], Tensor],
        activation_name: str,
        bias_up: bool,
        bias_down: bool,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.activation = activation
        self.activation_name = str(activation_name)
        self.up = nn.Linear(self.in_features, self.hidden_features, bias=bias_up)
        self.down = nn.Linear(self.hidden_features, self.in_features, bias=bias_down)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset linear projections."""

        nn.init.xavier_uniform_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        nn.init.xavier_uniform_(self.down.weight)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Apply the MLP."""

        return self.down(self.activation(self.up(x)))


def build_dit_mlp(
    *,
    mlp_type: MLPType,
    in_features: int,
    hidden_budget: int,
    activation_config: object | None = None,
    block_index: int = 0,
    bias_up: bool = False,
    bias_down: bool = False,
) -> nn.Module:
    """Build the exported MLP variant."""

    _ = activation_config, block_index
    match mlp_type:
        case MLPType.GELU:
            return SimpleActivationMLP(
                in_features=int(in_features),
                hidden_features=int(hidden_budget),
                activation=F.gelu,
                activation_name="gelu",
                bias_up=bool(bias_up),
                bias_down=bool(bias_down),
            )
        case MLPType.SILU:
            return SimpleActivationMLP(
                in_features=int(in_features),
                hidden_features=int(hidden_budget),
                activation=F.silu,
                activation_name="silu",
                bias_up=bool(bias_up),
                bias_down=bool(bias_down),
            )
        case MLPType.RELU:
            return SimpleActivationMLP(
                in_features=int(in_features),
                hidden_features=int(hidden_budget),
                activation=F.relu,
                activation_name="relu",
                bias_up=bool(bias_up),
                bias_down=bool(bias_down),
            )
        case _ as unreachable:
            raise ValueError(f"Unsupported exported MLP type: {unreachable}")


__all__ = ["SimpleActivationMLP", "build_dit_mlp", "reset_module_parameters"]
