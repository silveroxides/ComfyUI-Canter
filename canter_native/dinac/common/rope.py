from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


class Rope1D(nn.Module):
    """
    Rotary Position Embedding (RoPE) 1D.

    Based on the reference LLaMA implementation (Hugging Face
    `modeling_llama.py`), adapted to this codebase without behavior changes.

    - dim: per-head dimension
    - max_position_embeddings: length used to precompute cached cos/sin (not required
      by forward)
    - base: RoPE base theta

    Forward expects:
      - x: (B, H, T, D)
      - position_ids: (B, T) integer positions
    Returns:
      - cos, sin: (B, T, D)
    """

    inv_freq: torch.Tensor
    _cos_cached: torch.Tensor
    _sin_cached: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: torch.device | None = None,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise AssertionError("head_dim must be even for RoPE")
        self.scaling_factor: float = float(scaling_factor)
        self.dim: int = int(dim)
        self.max_position_embeddings: int = int(max_position_embeddings)
        self.base: float = float(base)
        inv_freq = self._build_inv_freq(device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cached cos/sin (not used in application, but kept for parity with reference)
        self.max_seq_len_cached: int = self.max_position_embeddings
        cos_cached, sin_cached = self._build_cached_trig(device=device)
        self.register_buffer("_cos_cached", cos_cached, persistent=False)
        self.register_buffer("_sin_cached", sin_cached, persistent=False)

    def _build_inv_freq(self, *, device: torch.device | None) -> torch.Tensor:
        """Return the RoPE inverse-frequency vector in float32."""

        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / float(self.dim)
            )
        )

    def _build_cached_trig(
        self, *, device: torch.device | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached RoPE trig tensors in float32."""

        inv_freq = self._build_inv_freq(device=device)
        t = torch.arange(
            self.max_seq_len_cached,
            device=device,
            dtype=torch.float32,
        )
        t = t / self.scaling_factor
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> Rope1D:
        """Apply module moves/casts while preserving fp32 RoPE buffers."""

        out = super()._apply(fn, recurse=recurse)
        with torch.no_grad():
            device = self.inv_freq.device
            self.inv_freq.data = self._build_inv_freq(device=device)
            cos_cached, sin_cached = self._build_cached_trig(device=device)
            self._cos_cached.data = cos_cached
            self._sin_cached.data = sin_cached
        return out

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_tensor = self._build_inv_freq(device=x.device)
        inv_freq_expanded = (
            inv_freq_tensor[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float() / self.scaling_factor
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_adjacent(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs in the last dimension.

    This matches the common EVA-02 / SpeedrunDiT RoPE convention where the last
    dimension is interpreted as pairs ``(x0, x1), (x2, x3), ...``.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("rotate_half_adjacent requires an even last dimension")
    x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LearnableRoPE2D(nn.Module):
    r"""
    Learnable mixed 2D RoPE with axial RoPE2D-compatible initialization.

    - Learnable frequency banks for X and Y.
    - Frequencies can be shared across groups of attention heads (see
      ``rope_param_dim``).
    - Angle per pair: theta = x * fx[g, i] + y * fy[g, i]
    - Initialization matches the axial RoPE2D parameterization used by DiTTrunk
      for ``ROPE_2D_AXIAL_FREQ_AWARE`` (AxialRoPE2DConfig(base=100, dim_layout=HALF_SPLIT)):
        - Angle multiplier ``2π``.
        - Period base ``100`` (DINOv3-style), applied per-axis.
      Each head group starts identically (deterministic init) so the learnable
      variant is functionally identical to axial RoPE2D at step 0.
    - Rotation is implemented with real-valued sin/cos to avoid complex tensors
      (torch.compile/inductor cannot codegen complex dtypes).

    Shapes:
    - Expects q,k of shape (B, H, T, D) with D % 4 == 0.
    - Positions xy: (T, 2) or (B, T, 2), any real dtype (cast to float32).
    - Parameter `freqs`: (2, G, D//2) in float32; index 0 = x, 1 = y.

    Head grouping / parameter budget
    -------------------------------
    ``rope_param_dim`` controls the total number of learned RoPE frequency
    parameters (scalars) for this module.

    Let:
      - ``head_dim = D`` (per-head width)
      - ``num_heads = H``
      - ``rope_param_dim = P``

    Then the module uses:
      - ``num_groups = G = P // D``
      - ``heads_per_group = H // G``

    This is fail-fast: ``P`` must be divisible by ``D`` and ``H`` must be
    divisible by ``G``. When ``rope_param_dim`` is None (default), the module
    uses the classic per-head parameterization with ``P = H * D``.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        num_heads: int,
        rope_param_dim: int | None = None,
        rope_base: float = 100.0,
        angle_multiplier: float = 2.0 * float(math.pi),
        learnable: bool = True,
        persist_buffers: bool = True,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise AssertionError("head_dim must be divisible by 4 for mixed 2D RoPE")
        self.head_dim: int = int(head_dim)
        # Avoid naming collisions with nn.Module.half() (dtype casting helper).
        self.half_dim: int = self.head_dim // 2
        self.num_heads: int = int(num_heads)
        effective_param_dim = (
            int(rope_param_dim)
            if rope_param_dim is not None
            else self.num_heads * self.head_dim
        )
        if effective_param_dim <= 0:
            raise ValueError("rope_param_dim must be positive for LearnableRoPE2D")
        self.rope_param_dim: int = int(effective_param_dim)
        self._learnable: bool = bool(learnable)
        theta = float(rope_base)
        mult = float(angle_multiplier)
        if not math.isfinite(theta) or theta <= 0.0:
            raise ValueError("rope_base must be finite and > 0 for LearnableRoPE2D")
        if not math.isfinite(mult) or mult <= 0.0:
            raise ValueError(
                "angle_multiplier must be finite and > 0 for LearnableRoPE2D"
            )

        if self.rope_param_dim % self.head_dim != 0:
            raise ValueError(
                "rope_param_dim must be divisible by head_dim for LearnableRoPE2D "
                f"(got rope_param_dim={self.rope_param_dim}, head_dim={self.head_dim})"
            )
        self.num_groups: int = self.rope_param_dim // self.head_dim
        if self.num_groups <= 0:
            raise RuntimeError("num_groups must be positive for LearnableRoPE2D")
        if self.num_heads % self.num_groups != 0:
            raise ValueError(
                "num_heads must be divisible by (rope_param_dim / head_dim) for LearnableRoPE2D "
                f"(got num_heads={self.num_heads}, num_groups={self.num_groups}, "
                f"rope_param_dim={self.rope_param_dim}, head_dim={self.head_dim})"
            )
        self.heads_per_group: int = self.num_heads // self.num_groups
        if self.heads_per_group <= 0:
            raise RuntimeError("heads_per_group must be positive for LearnableRoPE2D")

        # Axial-compatible deterministic init:
        # - periods match AxialRoPE2DConfig(base=100, dim_layout=HALF_SPLIT)
        # - angle = 2π * coord / period
        qtr = self.head_dim // 4
        exponents = (
            2.0
            * torch.arange(int(qtr), dtype=torch.float32)
            / float(self.head_dim // 2)
        )
        periods = torch.tensor(theta, dtype=torch.float32) ** exponents  # [qtr]
        axis_freqs = (mult / periods).to(dtype=torch.float32)  # [qtr]

        zeros = torch.zeros_like(axis_freqs)
        # Match AxialRoPE2D(HALF_SPLIT) flatten order: [y-axis, x-axis].
        # Our xy columns are (x, y), so:
        # - x contributes to the second quarter (x-axis part)
        # - y contributes to the first quarter (y-axis part)
        fx_half = torch.cat((zeros, axis_freqs), dim=0)  # [half_dim]
        fy_half = torch.cat((axis_freqs, zeros), dim=0)  # [half_dim]

        freqs_x = fx_half.expand(int(self.num_groups), -1).clone()
        freqs_y = fy_half.expand(int(self.num_groups), -1).clone()
        freqs = torch.stack([freqs_x, freqs_y], dim=0)  # (2, G, half)
        if self._learnable:
            self.freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer("freqs", freqs, persistent=persist_buffers)

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> LearnableRoPE2D:
        """Apply module moves/casts while preserving fp32 frequency tensors."""

        out = super()._apply(fn, recurse=recurse)
        with torch.no_grad():
            self.freqs.data = self.freqs.data.to(dtype=torch.float32)
        return out

    def _apply_rotary_from_trig(
        self,
        x: torch.Tensor,
        *,
        sin: torch.Tensor,
        cos: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate Q/K using precomputed grouped sin/cos buffers (HALF_SPLIT layout).

        This matches AxialRoPE2DConfig(dim_layout=HALF_SPLIT) rotation and keeps
        the learnable variant identical at initialization when combined with
        axial-compatible frequency init.

        Args:
            x: Tensor shaped ``(B, H, T, D)``.
            sin: Sin tensor shaped ``(G, T, D//2)`` or ``(B, G, T, D//2)``.
            cos: Cos tensor shaped ``(G, T, D//2)`` or ``(B, G, T, D//2)``.

        Returns:
            Tensor with the same shape/dtype/device as ``x``.
        """
        if x.dim() != 4:
            raise ValueError("x must be shaped (B, H, T, D)")
        B, H, T, D = x.shape
        if self.num_heads != int(H):
            raise ValueError("num_heads mismatch for LearnableRoPE2D")
        if self.head_dim != int(D):
            raise ValueError("head_dim mismatch for LearnableRoPE2D")

        if sin.dim() == 3 and cos.dim() == 3:
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        if sin.dim() != 4 or cos.dim() != 4:
            raise RuntimeError("Unexpected sin/cos rank for LearnableRoPE2D")
        if int(D) % 2 != 0:
            raise RuntimeError("LearnableRoPE2D requires even head_dim for HALF_SPLIT")
        half = int(D) // 2
        if int(sin.shape[-1]) != half or int(cos.shape[-1]) != half:
            raise RuntimeError(
                "LearnableRoPE2D expected sin/cos last dim == head_dim//2 "
                f"(got sin={tuple(sin.shape)}, cos={tuple(cos.shape)}, head_dim={int(D)})"
            )

        sin = sin[:, :, None, :, :]  # [B, G, 1, T, half]
        cos = cos[:, :, None, :, :]  # [B, G, 1, T, half]

        grouped = x.reshape(
            int(B),
            int(self.num_groups),
            int(self.heads_per_group),
            int(T),
            int(D),
        )
        x1 = grouped[..., :half]
        x2 = grouped[..., half:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        out = torch.cat((out1, out2), dim=-1).reshape(int(B), int(H), int(T), int(D))
        return out.to(dtype=x.dtype)

    def _compute_mixed_cis(self, xy: torch.Tensor) -> torch.Tensor:
        # Returns complex cis angles with shape (G, T, half) or (B, G, T, half)
        if xy.dim() == 2:
            # (T, 2) -> (G, T, half)
            t_x = xy[:, 0].to(dtype=torch.float32)
            t_y = xy[:, 1].to(dtype=torch.float32)
            with torch.autocast(device_type=t_x.device.type, enabled=False):
                # Memory notes:
                # - Avoid materializing both fx and fy; accumulate in-place into angles.
                # - Avoid torch.ones_like(angles) (full-size allocation); a scalar
                #   magnitude broadcasts in torch.polar.
                angles = t_x.unsqueeze(-1).unsqueeze(-1) * self.freqs[0].unsqueeze(
                    0
                )  # (T, G, half)
                angles.add_(
                    t_y.unsqueeze(-1).unsqueeze(-1) * self.freqs[1].unsqueeze(0)
                )
                angles = angles.permute(1, 0, 2)  # (G, T, half)
                cis = torch.polar(
                    torch.ones((), device=angles.device, dtype=angles.dtype), angles
                )
            return cis
        elif xy.dim() == 3:
            # (B, T, 2) -> (B, G, T, half)
            t_x = xy[..., 0].to(dtype=torch.float32)
            t_y = xy[..., 1].to(dtype=torch.float32)
            with torch.autocast(device_type=t_x.device.type, enabled=False):
                angles = t_x.unsqueeze(-1).unsqueeze(-1) * self.freqs[0].unsqueeze(
                    0
                ).unsqueeze(0)
                angles.add_(
                    t_y.unsqueeze(-1).unsqueeze(-1)
                    * self.freqs[1].unsqueeze(0).unsqueeze(0)
                )
                angles = angles.permute(0, 2, 1, 3)  # (B, G, T, half)
                cis = torch.polar(
                    torch.ones((), device=angles.device, dtype=angles.dtype), angles
                )
            return cis
        else:
            raise ValueError("xy must have shape (T,2) or (B,T,2)")

    def _compute_mixed_angles(self, xy: torch.Tensor) -> torch.Tensor:
        """Return mixed RoPE2D angles without applying cis/polar.

        Args:
            xy: XY positions shaped ``(T, 2)`` or ``(B, T, 2)``.

        Returns:
            Float tensor of angles shaped ``(G, T, half)`` or ``(B, G, T, half)``.
        """
        if xy.dim() == 2:
            t_x = xy[:, 0].to(dtype=torch.float32)
            t_y = xy[:, 1].to(dtype=torch.float32)
            with torch.autocast(device_type=t_x.device.type, enabled=False):
                angles = t_x.unsqueeze(-1).unsqueeze(-1) * self.freqs[0].unsqueeze(0)
                angles.add_(
                    t_y.unsqueeze(-1).unsqueeze(-1) * self.freqs[1].unsqueeze(0)
                )
                return angles.permute(1, 0, 2)
        if xy.dim() == 3:
            t_x = xy[..., 0].to(dtype=torch.float32)
            t_y = xy[..., 1].to(dtype=torch.float32)
            with torch.autocast(device_type=t_x.device.type, enabled=False):
                angles = t_x.unsqueeze(-1).unsqueeze(-1) * self.freqs[0].unsqueeze(
                    0
                ).unsqueeze(0)
                angles.add_(
                    t_y.unsqueeze(-1).unsqueeze(-1)
                    * self.freqs[1].unsqueeze(0).unsqueeze(0)
                )
                return angles.permute(0, 2, 1, 3)
        raise ValueError("xy must have shape (T,2) or (B,T,2)")

    def _cos_sin_half_from_xy(
        self,
        xy: torch.Tensor,
        *,
        device: torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Helper used in tests to build real-valued cos/sin tensors.
        cis = self._compute_mixed_cis(xy.to(device=device) if device else xy)
        # Convert complex cis to cos/sin (real/imag) with matching shapes
        if cis.is_complex():
            cos_h = cis.real
            sin_h = cis.imag
        else:
            # Should not happen; torch.polar returns complex64/128
            raise RuntimeError("Expected complex cis tensor from polar")
        if out_dtype is not None:
            cos_h = cos_h.to(dtype=out_dtype)
            sin_h = sin_h.to(dtype=out_dtype)
        return cos_h, sin_h

    def _cos_sin_from_xy(
        self,
        xy: torch.Tensor,
        *,
        device: torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_h, sin_h = self._cos_sin_half_from_xy(
            xy, device=device, out_dtype=out_dtype
        )
        emb_cos = torch.cat((cos_h, cos_h), dim=-1)
        emb_sin = torch.cat((sin_h, sin_h), dim=-1)
        return emb_cos, emb_sin

    def rotate_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError("q,k must be shaped (B,H,T,D)")
        _, H, _, D = q.shape
        if self.num_heads != H:
            raise ValueError("num_heads mismatch for LearnableRoPE2D")
        if self.head_dim != D:
            raise ValueError("head_dim mismatch for LearnableRoPE2D")
        if D % 4 != 0:
            raise AssertionError("head_dim must be divisible by 4 for mixed 2D RoPE")

        # Use real-valued sin/cos rotation to keep torch.compile/inductor on the
        # fast path (inductor cannot codegen complex tensors).
        angles = self._compute_mixed_angles(xy.to(device=q.device))
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        q_out = self._apply_rotary_from_trig(q, sin=sin, cos=cos)
        k_out = self._apply_rotary_from_trig(k, sin=sin, cos=cos)
        return q_out, k_out

    def rotate_qk_with_dilation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        xy: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q/K using mixed 2D RoPE with per-sample isotropic dilation.

        This implements dilation by scaling the RoPE angle, i.e.
        ``theta_dilated = scale * theta_base`` where ``theta_base`` comes from the
        undilated XY coordinates.

        Args:
            q: Query tensor shaped ``(B, H, T, D)``.
            k: Key tensor shaped ``(B, H, T, D)``.
            xy: Base XY coordinates shaped ``(T, 2)`` or ``(B, T, 2)``.
            scales: Per-sample dilation scales shaped ``(B,)``.

        Raises:
            ValueError: If shapes are inconsistent or scales are not 1D.
        """
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError("q,k must be shaped (B,H,T,D)")
        B, H, T, D = q.shape
        if self.num_heads != H:
            raise ValueError("num_heads mismatch for LearnableRoPE2D")
        if self.head_dim != D:
            raise ValueError("head_dim mismatch for LearnableRoPE2D")
        if scales.dim() != 1 or scales.shape[0] != B:
            raise ValueError("scales must have shape (B,) matching q batch size")
        if xy.dim() == 2 and xy.shape[0] != T:
            raise ValueError("xy length must match q sequence length")
        if xy.dim() == 3 and (xy.shape[0] != B or xy.shape[1] != T):
            raise ValueError("xy must have shape (B,T,2) matching q batch/sequence")
        if xy.shape[-1] != 2:
            raise ValueError("xy must have last dimension 2")

        angles = self._compute_mixed_angles(xy.to(device=q.device))
        angles = angles * scales.to(device=q.device, dtype=torch.float32).view(
            B, 1, 1, 1
        )
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        q_out = self._apply_rotary_from_trig(q, sin=sin, cos=cos)
        k_out = self._apply_rotary_from_trig(k, sin=sin, cos=cos)
        return q_out, k_out
