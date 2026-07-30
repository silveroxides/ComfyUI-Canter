"""Scale+Gate AdaLN (2-way) for FCDM decoder blocks."""

from __future__ import annotations

from torch import Tensor, nn


class AdaLNScaleGateZeroProjector(nn.Module):
    """Packed 2-way AdaLN projection (SiLU -> Linear), zero-initialized.

    Outputs [B, 2*d_model] packed as (scale, gate).
    """

    def __init__(self, d_model: int, d_cond: int, operations) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.d_cond: int = int(d_cond)
        self.act: nn.SiLU = nn.SiLU()
        self.proj = operations.Linear(self.d_cond, 2 * self.d_model)
        if self.proj.weight is not None:
            nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def project_activated(self, act_cond: Tensor) -> Tensor:
        """Return packed modulation for a pre-activated conditioning vector."""

        if act_cond.dim() != 2:
            raise ValueError(
                "AdaLNScaleGateZeroProjector expects act_cond with shape [B, d_cond]"
            )
        if act_cond.shape[1] != self.d_cond:
            raise ValueError(
                f"act_cond width {int(act_cond.shape[1])} does not match d_cond={self.d_cond}"
            )
        return self.proj(act_cond)

    def forward(self, cond: Tensor) -> Tensor:
        """Return packed modulation [B, 2*d_model]."""
        if cond.dim() != 2:
            raise ValueError(
                "AdaLNScaleGateZeroProjector expects cond with shape [B, d_cond]"
            )
        if cond.shape[1] != self.d_cond:
            raise ValueError(
                f"cond width {int(cond.shape[1])} does not match d_cond={self.d_cond}"
            )
        return self.project_activated(self.act(cond))


class AdaLNScaleGateZeroLowRankDelta(nn.Module):
    """Low-rank delta for 2-way AdaLN: down(d_cond -> rank) -> up(rank -> 2*d_model).

    Zero-initialized up projection preserves zero-output semantics at init.
    """

    def __init__(
        self, *, d_model: int, d_cond: int, rank: int, operations
    ) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.d_cond: int = int(d_cond)
        self.rank: int = int(rank)
        self.down = operations.Linear(self.d_cond, self.rank, bias=False)
        self.up = operations.Linear(
            self.rank, 2 * self.d_model, bias=False
        )
        if self.down.weight is not None:
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        if self.up.weight is not None:
            nn.init.zeros_(self.up.weight)

    def forward(self, act_cond: Tensor) -> Tensor:
        """Return packed delta modulation [B, 2*d_model]."""
        if act_cond.dim() != 2:
            raise ValueError(
                "AdaLNScaleGateZeroLowRankDelta expects act_cond with shape [B, d_cond]"
            )
        if act_cond.shape[1] != self.d_cond:
            raise ValueError(
                f"act_cond width {int(act_cond.shape[1])} does not match d_cond={self.d_cond}"
            )
        return self.up(self.down(act_cond))
