"""DDIM and DPM++2M samplers for VP diffusion with path-drop PDG support."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .vp_diffusion import (
    alpha_sigma_from_logsnr,
    broadcast_time_like,
    shifted_cosine_interpolated_logsnr_from_t,
)


class DecoderForwardFn(Protocol):
    """Callable that predicts x0 from (x_t, t, latents) with path-drop PDG flag."""

    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        latents: Tensor,
        *,
        drop_middle_blocks: bool = False,
        mask_latent_tokens: bool = False,
    ) -> Tensor: ...


def _reconstruct_eps_from_x0(
    *, x_t: Tensor, x0_hat: Tensor, alpha: Tensor, sigma: Tensor
) -> Tensor:
    """Reconstruct eps_hat from (x_t, x0_hat) under VP parameterization.

    eps_hat = (x_t - alpha * x0_hat) / sigma. All float32.
    """
    alpha_view = broadcast_time_like(alpha, x_t).to(dtype=torch.float32)
    sigma_view = broadcast_time_like(sigma, x_t).to(dtype=torch.float32)
    x_t_f32 = x_t.to(torch.float32)
    x0_f32 = x0_hat.to(torch.float32)
    return (x_t_f32 - alpha_view * x0_f32) / sigma_view


def _ddim_step(
    *,
    x0_hat: Tensor,
    eps_hat: Tensor,
    alpha_next: Tensor,
    sigma_next: Tensor,
    ref: Tensor,
) -> Tensor:
    """DDIM step: x_next = alpha_next * x0_hat + sigma_next * eps_hat."""
    a = broadcast_time_like(alpha_next, ref).to(dtype=torch.float32)
    s = broadcast_time_like(sigma_next, ref).to(dtype=torch.float32)
    return a * x0_hat + s * eps_hat


def _predict_with_pdg(
    forward_fn: DecoderForwardFn,
    state: Tensor,
    t_vec: Tensor,
    latents: Tensor,
    *,
    pdg_mode: str,
    pdg_strength: float,
) -> Tensor:
    """Run decoder forward with optional PDG guidance.

    Args:
        forward_fn: Decoder forward function.
        state: Current noised state [B, C, H, W].
        t_vec: Timestep vector [B].
        latents: Encoder latents.
        pdg_mode: "disabled" or "path_drop".
        pdg_strength: CFG-like strength for PDG.

    Returns:
        x0_hat prediction in float32.
    """
    match pdg_mode:
        case "path_drop":
            x0_uncond = forward_fn(state, t_vec, latents, drop_middle_blocks=True).to(
                torch.float32
            )
            x0_cond = forward_fn(state, t_vec, latents, drop_middle_blocks=False).to(
                torch.float32
            )
            return x0_uncond + pdg_strength * (x0_cond - x0_uncond)
        case "disabled":
            return forward_fn(state, t_vec, latents, drop_middle_blocks=False).to(
                torch.float32
            )
        case _ as unreachable:
            raise ValueError(f"Unsupported PDG mode: {unreachable!r}")


def run_ddim(
    *,
    forward_fn: DecoderForwardFn,
    initial_state: Tensor,
    schedule: Tensor,
    latents: Tensor,
    logsnr_min: float,
    logsnr_max: float,
    log_change_high: float = 0.0,
    log_change_low: float = 0.0,
    pdg_mode: str = "disabled",
    pdg_strength: float = 1.5,
    device: torch.device | None = None,
) -> Tensor:
    """Run DDIM sampling loop with path-drop PDG support.

    Args:
        forward_fn: Decoder forward function (x_t, t, latents) -> x0_hat.
        initial_state: Starting noised state [B, C, H, W] in float32.
        schedule: Descending t-schedule [num_steps] in [0, 1].
        latents: Encoder latents [B, bottleneck_dim, h, w].
        logsnr_min, logsnr_max: VP schedule endpoints.
        log_change_high, log_change_low: Shifted-cosine schedule parameters.
        pdg_mode: "disabled" or "path_drop".
        pdg_strength: CFG-like strength for PDG.
        device: Target device.

    Returns:
        Denoised samples [B, C, H, W] in float32.
    """
    run_device = device or initial_state.device
    batch_size = int(initial_state.shape[0])
    state = initial_state.to(device=run_device, dtype=torch.float32)

    # Precompute logSNR, alpha, sigma for all schedule points
    lmb = shifted_cosine_interpolated_logsnr_from_t(
        schedule.to(device=run_device),
        logsnr_min=logsnr_min,
        logsnr_max=logsnr_max,
        log_change_high=log_change_high,
        log_change_low=log_change_low,
    )
    alpha_sched, sigma_sched = alpha_sigma_from_logsnr(lmb)

    for i in range(int(schedule.numel()) - 1):
        t_i = schedule[i]
        a_t = alpha_sched[i].expand(batch_size)
        s_t = sigma_sched[i].expand(batch_size)
        a_next = alpha_sched[i + 1].expand(batch_size)
        s_next = sigma_sched[i + 1].expand(batch_size)

        # Model prediction with optional PDG
        t_vec = t_i.expand(batch_size).to(device=run_device, dtype=torch.float32)
        x0_hat = _predict_with_pdg(
            forward_fn,
            state,
            t_vec,
            latents,
            pdg_mode=pdg_mode,
            pdg_strength=pdg_strength,
        )

        eps_hat = _reconstruct_eps_from_x0(
            x_t=state, x0_hat=x0_hat, alpha=a_t, sigma=s_t
        )
        state = _ddim_step(
            x0_hat=x0_hat,
            eps_hat=eps_hat,
            alpha_next=a_next,
            sigma_next=s_next,
            ref=state,
        )

    return state


def run_dpmpp_2m(
    *,
    forward_fn: DecoderForwardFn,
    initial_state: Tensor,
    schedule: Tensor,
    latents: Tensor,
    logsnr_min: float,
    logsnr_max: float,
    log_change_high: float = 0.0,
    log_change_low: float = 0.0,
    pdg_mode: str = "disabled",
    pdg_strength: float = 1.5,
    device: torch.device | None = None,
) -> Tensor:
    """Run DPM++2M sampling loop with path-drop PDG support.

    Multi-step solver using exponential integrator formulation in half-lambda space.
    """
    run_device = device or initial_state.device
    batch_size = int(initial_state.shape[0])
    state = initial_state.to(device=run_device, dtype=torch.float32)

    # Precompute logSNR, alpha, sigma, half-lambda for all schedule points
    lmb = shifted_cosine_interpolated_logsnr_from_t(
        schedule.to(device=run_device),
        logsnr_min=logsnr_min,
        logsnr_max=logsnr_max,
        log_change_high=log_change_high,
        log_change_low=log_change_low,
    )
    alpha_sched, sigma_sched = alpha_sigma_from_logsnr(lmb)
    half_lambda = 0.5 * lmb.to(torch.float32)

    x0_prev: Tensor | None = None

    for i in range(int(schedule.numel()) - 1):
        t_i = schedule[i]
        s_t = sigma_sched[i].expand(batch_size)
        a_next = alpha_sched[i + 1].expand(batch_size)
        s_next = sigma_sched[i + 1].expand(batch_size)

        # Model prediction with optional PDG
        t_vec = t_i.expand(batch_size).to(device=run_device, dtype=torch.float32)
        x0_hat = _predict_with_pdg(
            forward_fn,
            state,
            t_vec,
            latents,
            pdg_mode=pdg_mode,
            pdg_strength=pdg_strength,
        )

        lam_t = half_lambda[i].expand(batch_size)
        lam_next = half_lambda[i + 1].expand(batch_size)
        h = (lam_next - lam_t).to(torch.float32)
        phi_1 = torch.expm1(-h)

        sigma_ratio = (s_next / s_t).to(torch.float32)

        if i == 0 or x0_prev is None:
            # First-order step
            state = (
                sigma_ratio.view(-1, *([1] * (state.dim() - 1))) * state
                - broadcast_time_like(a_next, state).to(torch.float32)
                * broadcast_time_like(phi_1, state).to(torch.float32)
                * x0_hat
            )
        else:
            # Second-order step
            lam_prev = half_lambda[i - 1].expand(batch_size)
            h_0 = (lam_t - lam_prev).to(torch.float32)
            r0 = h_0 / h
            d1_0 = (x0_hat - x0_prev) / broadcast_time_like(r0, x0_hat)
            common = broadcast_time_like(a_next, state).to(
                torch.float32
            ) * broadcast_time_like(phi_1, state).to(torch.float32)
            state = (
                sigma_ratio.view(-1, *([1] * (state.dim() - 1))) * state
                - common * x0_hat
                - 0.5 * common * d1_0
            )

        x0_prev = x0_hat

    return state
