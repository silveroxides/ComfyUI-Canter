"""Patch embedding used by the exported DINAC-AE model."""

from __future__ import annotations

from typing import Final

from torch import Tensor, nn

__all__ = ["Patchify", "StraightThroughEncoder"]


class StraightThroughEncoder(nn.Module):
    """Project non-overlapping image patches with a stride-patch convolution."""

    def __init__(
        self,
        in_channels: int,
        patch: int,
        out_channels: int,
        operations,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if patch <= 0:
            raise ValueError("patch must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        self.in_channels: Final[int] = int(in_channels)
        self.patch: Final[int] = int(patch)
        self._output_channels: Final[int] = int(out_channels)
        self.proj = operations.Conv2d(
            self.in_channels,
            self._output_channels,
            kernel_size=self.patch,
            stride=self.patch,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Return the patchified token grid."""

        return self.proj(x)

    @property
    def output_channels(self) -> int:
        """Return the output channel width produced by the encoder."""

        return int(self._output_channels)

    @property
    def latent_channels(self) -> int:
        """Alias for ``output_channels`` to match encoder interface shape."""

        return int(self._output_channels)


Patchify = StraightThroughEncoder
