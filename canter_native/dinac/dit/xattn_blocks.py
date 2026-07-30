"""Dense cross-attention block used by the DINAC-AE class-token head."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from ..common.norms import RMSNorm
from ..dit.attention_blocks import CrossAttentionCore
from ..dit.mlp import build_dit_mlp, reset_module_parameters
from ..dit.mlp_types import MLPType


@dataclass
class CrossAttentionConfig:
    """Configuration for the exported dense cross-attention block."""

    n_heads: int = 16
    head_dim: int | None = None
    query_extra_dim: int = 0
    context_extra_dim: int = 0
    key_extra_dim: int = 0
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.0
    mlp_type: MLPType = MLPType.GELU
    activation_config: object | None = None
    use_norms: bool = True
    block_index: int = 0
    use_attn_residual: bool = True


class CrossAttentionBlock(nn.Module):
    """Dense pre-norm cross-attention plus residual MLP."""

    query_dim: int
    context_dim: int
    query_extra_dim: int
    context_extra_dim: int
    key_extra_dim: int
    n_heads: int
    head_dim: int
    attn_dim: int
    use_norms: bool
    attn_dropout: float
    use_attn_residual: bool
    query_norm: RMSNorm | None
    context_norm: RMSNorm | None
    mlp_norm: RMSNorm | None
    q_proj: nn.Linear
    attn_core: CrossAttentionCore
    kv_proj: nn.Linear
    out_proj: nn.Linear
    mlp: nn.Module

    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        cfg: CrossAttentionConfig,
    ) -> None:
        super().__init__()
        n_heads = int(cfg.n_heads)
        if cfg.head_dim is None:
            if query_dim % n_heads != 0:
                raise ValueError("query_dim must be divisible by n_heads")
            head_dim = query_dim // n_heads
        else:
            head_dim = int(cfg.head_dim)
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)
        self.query_extra_dim = int(cfg.query_extra_dim)
        self.context_extra_dim = int(cfg.context_extra_dim)
        self.key_extra_dim = int(cfg.key_extra_dim)
        self.n_heads = n_heads
        self.head_dim = int(head_dim)
        self.attn_dim = int(self.n_heads * self.head_dim)
        self.use_norms = bool(cfg.use_norms)
        self.attn_dropout = float(cfg.attn_dropout)
        if not cfg.use_attn_residual:
            raise ValueError("DINAC-AE export requires attention residuals")
        self.use_attn_residual = True
        self.query_norm = RMSNorm(self.query_dim) if self.use_norms else None
        self.context_norm = RMSNorm(self.context_dim) if self.use_norms else None
        self.mlp_norm = RMSNorm(query_dim) if self.use_norms else None
        self.q_proj = nn.Linear(
            self.query_dim + self.query_extra_dim, self.attn_dim, bias=False
        )
        self.attn_core = CrossAttentionCore(
            query_dim=query_dim,
            context_dim=context_dim,
            context_extra_dim=self.context_extra_dim,
            key_extra_dim=self.key_extra_dim,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            attn_dropout=self.attn_dropout,
        )
        self.kv_proj = self.attn_core.kv_proj
        self.out_proj = self.attn_core.out_proj
        hidden = int(round(cfg.mlp_ratio * query_dim))
        self.mlp = build_dit_mlp(
            mlp_type=cfg.mlp_type,
            in_features=query_dim,
            hidden_budget=hidden,
            activation_config=cfg.activation_config,
            block_index=int(cfg.block_index),
            bias_up=False,
            bias_down=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset projections and MLP parameters."""

        nn.init.xavier_uniform_(self.q_proj.weight)
        self.attn_core.reset_parameters()
        reset_module_parameters(self.mlp)

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        *,
        query_extra: Tensor | None = None,
        context_extra: Tensor | None = None,
        key_extra: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:  # type: ignore[override]
        """Run dense cross-attention followed by the residual MLP."""

        query_tokens = self.query_norm(query) if self.query_norm is not None else query
        if query_extra is not None:
            q_in = query_tokens.new_empty(
                *query_tokens.shape[:-1],
                int(query_tokens.shape[-1]) + int(query_extra.shape[-1]),
            )
            q_in[..., : int(query_tokens.shape[-1])] = query_tokens
            q_in[..., int(query_tokens.shape[-1]) :] = query_extra
        else:
            q_in = query_tokens
        context_tokens = (
            self.context_norm(context) if self.context_norm is not None else context
        )
        if context_extra is not None:
            kv_tokens = context_tokens.new_empty(
                *context_tokens.shape[:-1],
                int(context_tokens.shape[-1]) + int(context_extra.shape[-1]),
            )
            kv_tokens[..., : int(context_tokens.shape[-1])] = context_tokens
            kv_tokens[..., int(context_tokens.shape[-1]) :] = context_extra
        else:
            kv_tokens = context_tokens
        q_attn_tokens = self.q_proj(q_in)
        attn_out = self.attn_core(
            q_attn_tokens,
            kv_tokens,
            training=self.training,
            key_extra=key_extra,
            key_padding_mask=key_padding_mask,
        )
        tokens = query + attn_out
        mlp_in = self.mlp_norm(tokens) if self.mlp_norm is not None else tokens
        return tokens + self.mlp(mlp_in)

    def compile_for_training(self, *, fullgraph: bool, dynamic: bool) -> None:
        """No-op hook kept for the token-alignment head API."""

        _ = fullgraph, dynamic

    def compile_for_eval(self, *, fullgraph: bool, dynamic: bool) -> None:
        """No-op hook kept for the token-alignment head API."""

        _ = fullgraph, dynamic


__all__ = ["CrossAttentionBlock", "CrossAttentionConfig"]
