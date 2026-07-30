"""Lean float32 solvers for Canter's linear flow-matching path."""

from __future__ import annotations

import math
from enum import Enum
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

LOGSNR_SOLVER_START_EPS = 1.0e-4
_ER_QUADRATURE_POINTS = 16
_ER_NOISE_EXPONENT = 0.3
_ER_NOISE_OFFSET = 10.0
_SCORE_EPS = 1.0e-4


class Solver(Enum):
    """Numerical solvers exposed by the Canter inference API."""

    EULER = "euler"
    EULER_MARUYAMA = "euler_maruyama"
    ER_SDE = "er_sde"
    DPMPP_2M = "dpmpp_2m"
    ABM2 = "abm2"


class VelocityFunction(Protocol):
    """Callable that predicts velocity for one schedule evaluation."""

    def __call__(self, state: Tensor, time: Tensor, schedule_index: int) -> Tensor:
        """Return a velocity tensor matching ``state``."""

        ...


class SolverProgress(Protocol):
    """Callback receiving completed and total solver-update counts."""

    def __call__(self, completed: int, total: int) -> None:
        """Report progress after one completed state update."""

        ...


def _ignore_progress(completed: int, total: int) -> None:
    """Discard solver progress for non-interactive inference."""

    del completed, total


def _time_batch(time: Tensor, batch: int, device: torch.device) -> Tensor:
    """Expand one scalar schedule value to a float32 batch vector."""

    return time.expand(batch).to(device=device, dtype=torch.float32)


def _validate_inputs(state: Tensor, schedule: Tensor) -> None:
    """Validate solver invariants before entering the integration loop."""

    if state.dim() != 4:
        raise ValueError("Canter solver state must have shape [B, C, H, W].")
    if state.dtype is not torch.float32:
        raise ValueError("Canter solver state must use torch.float32.")
    if schedule.dim() != 1 or schedule.numel() < 2:
        raise ValueError("Canter schedules must contain at least two points.")
    if schedule.dtype is not torch.float32:
        raise ValueError("Canter schedules must use torch.float32.")
    if not bool(torch.isfinite(schedule).all().item()):
        raise ValueError("Canter schedules must contain only finite values.")
    if not bool(torch.all(schedule[:-1] > schedule[1:]).item()):
        raise ValueError("Canter inference requires a strictly descending schedule.")


def _euler(
    velocity: VelocityFunction,
    state: Tensor,
    schedule: Tensor,
    progress: SolverProgress,
) -> Tensor:
    """Integrate one first-order Euler update."""

    batch = int(state.shape[0])
    intervals = int(schedule.numel()) - 1
    for index in range(intervals):
        time = _time_batch(schedule[index], batch, state.device)
        step = schedule[index + 1] - schedule[index]
        state = state + step * velocity(state, time, index)
        progress(index + 1, intervals)
    return state


def _score(state: Tensor, velocity: Tensor, time: Tensor) -> Tensor:
    """Convert linear flow velocity to a float32 score estimate."""

    shape = (int(time.shape[0]),) + (1,) * (state.dim() - 1)
    time_view = time.float().view(shape)
    safe_time = time_view.clamp_min(_SCORE_EPS)
    return -(((1.0 - safe_time) * velocity.float()) + state.float()) / safe_time


def _euler_maruyama(
    velocity: VelocityFunction,
    state: Tensor,
    schedule: Tensor,
    *,
    generator: torch.Generator,
    multiplier: float,
    progress: SolverProgress,
) -> Tensor:
    """Integrate the reverse SDE with an Euler-Maruyama update."""

    batch = int(state.shape[0])
    intervals = int(schedule.numel()) - 1
    for index in range(intervals):
        time_value = float(schedule[index].item())
        next_value = float(schedule[index + 1].item())
        step = float(next_value - time_value)
        time = torch.full(
            (batch,),
            time_value,
            device=state.device,
            dtype=torch.float32,
        )
        predicted = velocity(state, time, index).float()
        terminal = index == intervals - 1 and next_value == 0.0
        if terminal:
            state = state - time_value * predicted
            progress(index + 1, intervals)
            continue
        drift = predicted - time.view((batch, 1, 1, 1)) * _score(
            state,
            predicted,
            time,
        )
        noise = torch.randn(
            state.shape,
            device=state.device,
            dtype=torch.float32,
            generator=generator,
        )
        diffusion = 2.0 * time_value
        noise_scale = float(multiplier) * math.sqrt(diffusion) * math.sqrt(abs(step))
        state = state + step * drift + noise_scale * noise
        progress(index + 1, intervals)
    return state


def _prepare_logsnr_schedule(schedule: Tensor, *, solver_name: str) -> Tensor:
    """Require finite evaluation times for a flow log-SNR solver."""

    if bool(((schedule[:-1] <= 0.0) | (schedule[:-1] >= 1.0)).any().item()):
        raise ValueError(
            f"{solver_name} evaluation times must lie strictly inside (0, 1)."
        )
    final_time = float(schedule[-1].item())
    if final_time < 0.0 or final_time >= 1.0:
        raise ValueError(f"{solver_name} final time must lie in [0, 1).")
    return schedule


def _half_log_snr(time: Tensor) -> Tensor:
    """Convert finite flow times to half-log-SNR in float64."""

    time64 = time.to(dtype=torch.float64)
    return torch.log1p(-time64) - torch.log(time64)


def _dpm_update(  # noqa: PLR0917 - explicit multistep solver state
    state: Tensor,
    denoised: Tensor,
    previous_denoised: Tensor | None,
    previous_lambda: Tensor | None,
    current_lambda: Tensor,
    next_lambda: Tensor,
) -> Tensor:
    """Apply one finite half-log-SNR DPM++ 2M update."""

    step = next_lambda - current_lambda
    if previous_denoised is None or previous_lambda is None:
        extrapolated = denoised
    else:
        previous_step = current_lambda - previous_lambda
        ratio = previous_step / step
        extrapolated = (1.0 + 1.0 / (2.0 * ratio)) * denoised - (
            1.0 / (2.0 * ratio)
        ) * previous_denoised
    log_sigma_ratio = F.logsigmoid(-next_lambda) - F.logsigmoid(-current_lambda)
    sigma_ratio = torch.exp(log_sigma_ratio).to(state)
    alpha_next = torch.sigmoid(next_lambda).to(state)
    denoised_coefficient = (alpha_next * (-torch.expm1(-step))).to(state)
    return sigma_ratio * state + denoised_coefficient * extrapolated


def _dpmpp_2m(
    velocity: VelocityFunction,
    state: Tensor,
    schedule: Tensor,
    progress: SolverProgress,
) -> Tensor:
    """Integrate with the flow-matching DPM++ 2M formulation."""

    times = _prepare_logsnr_schedule(schedule, solver_name="DPM++ 2M")
    lambdas = _half_log_snr(times.clamp_min(torch.finfo(times.dtype).tiny))
    batch = int(state.shape[0])
    previous_denoised: Tensor | None = None
    previous_lambda: Tensor | None = None
    intervals = int(times.numel()) - 1
    for index in range(intervals):
        current_time = times[index]
        predicted = velocity(
            state,
            _time_batch(current_time, batch, state.device),
            index,
        )
        denoised = state - current_time.to(state) * predicted
        if float(times[index + 1].item()) == 0.0:
            state = denoised
        else:
            state = _dpm_update(
                state,
                denoised,
                previous_denoised,
                previous_lambda,
                lambdas[index],
                lambdas[index + 1],
            )
        previous_denoised = denoised
        previous_lambda = lambdas[index]
        progress(index + 1, intervals)
    return state


def _er_noise_scaler(er_lambda: Tensor) -> Tensor:
    """Evaluate the paper-selected ER-SDE noise-scaling function."""

    return er_lambda * (torch.exp(er_lambda.pow(_ER_NOISE_EXPONENT)) + _ER_NOISE_OFFSET)


def _er_quadrature_rule(*, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return float32 Gauss-Legendre nodes and weights on the solver device."""

    nodes_np, weights_np = np.polynomial.legendre.leggauss(_ER_QUADRATURE_POINTS)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=torch.float32)
    weights = torch.as_tensor(weights_np, device=device, dtype=torch.float32)
    return nodes, weights


def _er_quadrature_terms(
    er_lambda_s: Tensor,
    er_lambda_t: Tensor,
    nodes: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate the ER-SDE correction integrals with Gauss-Legendre quadrature."""

    midpoint = (er_lambda_s + er_lambda_t) / 2.0
    half_span = (er_lambda_s - er_lambda_t) / 2.0
    positions = midpoint + half_span * nodes
    scaled_positions = _er_noise_scaler(positions)
    second = half_span * torch.sum(weights / scaled_positions)
    third = half_span * torch.sum(
        weights * (positions - er_lambda_s) / scaled_positions
    )
    return second, third


def _er_sde_step(
    *,
    state: Tensor,
    denoised: Tensor,
    old_denoised: Tensor | None,
    old_denoised_derivative: Tensor | None,
    times: Tensor,
    er_lambdas: Tensor,
    index: int,
    nodes: Tensor,
    weights: Tensor,
    generator: torch.Generator,
    noise_multiplier: float,
) -> tuple[Tensor, Tensor | None]:
    """Apply one nonterminal third-stage ER-SDE update."""

    er_lambda_s = er_lambdas[index]
    er_lambda_t = er_lambdas[index + 1]
    alpha_s = times[index] / er_lambda_s
    alpha_t = times[index + 1] / er_lambda_t
    ratio = _er_noise_scaler(er_lambda_t) / _er_noise_scaler(er_lambda_s)
    next_state = (alpha_t / alpha_s) * ratio * state
    next_state = next_state + alpha_t * (1.0 - ratio) * denoised

    denoised_derivative: Tensor | None = None
    stage = min(3, int(index) + 1)
    if stage >= 2:
        if old_denoised is None:
            raise RuntimeError("ER-SDE stage 2 requires denoised history.")
        second, third = _er_quadrature_terms(
            er_lambda_s,
            er_lambda_t,
            nodes,
            weights,
        )
        delta = er_lambda_t - er_lambda_s
        previous_delta = er_lambda_s - er_lambdas[index - 1]
        denoised_derivative = (denoised - old_denoised) / previous_delta
        next_state = (
            next_state
            + alpha_t
            * (delta + second * _er_noise_scaler(er_lambda_t))
            * denoised_derivative
        )
        if stage >= 3:
            if old_denoised_derivative is None:
                raise RuntimeError("ER-SDE stage 3 requires derivative history.")
            derivative_span = (er_lambda_s - er_lambdas[index - 2]) / 2.0
            second_derivative = (
                denoised_derivative - old_denoised_derivative
            ) / derivative_span
            next_state = (
                next_state
                + alpha_t
                * (delta.square() / 2.0 + third * _er_noise_scaler(er_lambda_t))
                * second_derivative
            )

    if float(noise_multiplier) > 0.0:
        noise = torch.randn(
            state.shape,
            device=state.device,
            dtype=torch.float32,
            generator=generator,
        )
        variance = er_lambda_t.square() - er_lambda_s.square() * ratio.square()
        stochastic_scale = alpha_t * torch.sqrt(torch.clamp(variance, min=0.0))
        next_state = next_state + float(noise_multiplier) * stochastic_scale * noise
    return next_state.float(), denoised_derivative


def _er_sde(
    velocity: VelocityFunction,
    state: Tensor,
    schedule: Tensor,
    *,
    generator: torch.Generator,
    noise_multiplier: float,
    progress: SolverProgress,
) -> Tensor:
    """Integrate with third-stage VP ER-SDE and Gauss-Legendre quadrature."""

    times = _prepare_logsnr_schedule(schedule, solver_name="ER-SDE")
    half_log_snr = torch.log1p(-times.float()) - torch.log(times.float())
    er_lambdas = torch.exp(-half_log_snr)
    nodes, weights = _er_quadrature_rule(device=state.device)
    old_denoised: Tensor | None = None
    old_denoised_derivative: Tensor | None = None
    batch = int(state.shape[0])
    intervals = int(times.numel()) - 1
    for index in range(intervals):
        current_time = times[index]
        predicted = velocity(
            state,
            _time_batch(current_time, batch, state.device),
            index,
        ).float()
        denoised = state - current_time * predicted
        if float(times[index + 1].item()) == 0.0:
            state = denoised
        else:
            state, old_denoised_derivative = _er_sde_step(
                state=state,
                denoised=denoised,
                old_denoised=old_denoised,
                old_denoised_derivative=old_denoised_derivative,
                times=times,
                er_lambdas=er_lambdas,
                index=index,
                nodes=nodes,
                weights=weights,
                generator=generator,
                noise_multiplier=float(noise_multiplier),
            )
        old_denoised = denoised
        progress(index + 1, intervals)
    return state


def _ab2_predict(
    state: Tensor,
    current_step: Tensor,
    previous_step: Tensor,
    current_velocity: Tensor,
    previous_velocity: Tensor,
) -> Tensor:
    """Return the variable-step Adams-Bashforth predictor."""

    ratio = current_step / previous_step
    return state + current_step * (
        (1.0 + 0.5 * ratio) * current_velocity - 0.5 * ratio * previous_velocity
    )


def _abm2(
    velocity: VelocityFunction,
    state: Tensor,
    schedule: Tensor,
    progress: SolverProgress,
) -> Tensor:
    """Integrate with ABM2, including corrected-state reevaluation."""

    batch = int(state.shape[0])
    intervals = int(schedule.numel()) - 1
    first_step = schedule[1] - schedule[0]
    previous_velocity = velocity(
        state,
        _time_batch(schedule[0], batch, state.device),
        0,
    )
    state = state + first_step * previous_velocity
    current_velocity = velocity(
        state,
        _time_batch(schedule[1], batch, state.device),
        1,
    )
    progress(1, intervals)
    for index in range(1, intervals):
        previous_step = schedule[index] - schedule[index - 1]
        current_step = schedule[index + 1] - schedule[index]
        predicted_state = _ab2_predict(
            state,
            current_step,
            previous_step,
            current_velocity,
            previous_velocity,
        )
        predicted_velocity = velocity(
            predicted_state,
            _time_batch(schedule[index + 1], batch, state.device),
            index + 1,
        )
        corrected_state = state + 0.5 * current_step * (
            current_velocity + predicted_velocity
        )
        previous_velocity = current_velocity
        state = corrected_state
        current_velocity = velocity(
            state,
            _time_batch(schedule[index + 1], batch, state.device),
            index + 1,
        )
        progress(index + 1, intervals)
    return state


def solve(
    velocity: VelocityFunction,
    initial_state: Tensor,
    schedule: Tensor,
    *,
    solver: Solver,
    generator: torch.Generator,
    euler_maruyama_multiplier: float,
    er_sde_noise_multiplier: float = 1.0,
    progress: SolverProgress | None = None,
) -> Tensor:
    """Integrate Canter velocity predictions over one validated schedule."""

    _validate_inputs(initial_state, schedule)
    if not isinstance(solver, Solver):
        raise TypeError("solver must be a Solver.")
    em_multiplier = float(euler_maruyama_multiplier)
    if not math.isfinite(em_multiplier) or em_multiplier < 0.0:
        raise ValueError("euler_maruyama_multiplier must be finite and non-negative.")
    er_multiplier = float(er_sde_noise_multiplier)
    if not math.isfinite(er_multiplier) or er_multiplier < 0.0:
        raise ValueError("er_sde_noise_multiplier must be finite and non-negative.")
    resolved_progress = _ignore_progress if progress is None else progress
    match solver:
        case Solver.EULER:
            return _euler(velocity, initial_state, schedule, resolved_progress)
        case Solver.EULER_MARUYAMA:
            return _euler_maruyama(
                velocity,
                initial_state,
                schedule,
                generator=generator,
                multiplier=em_multiplier,
                progress=resolved_progress,
            )
        case Solver.ER_SDE:
            return _er_sde(
                velocity,
                initial_state,
                schedule,
                generator=generator,
                noise_multiplier=er_multiplier,
                progress=resolved_progress,
            )
        case Solver.DPMPP_2M:
            return _dpmpp_2m(
                velocity,
                initial_state,
                schedule,
                resolved_progress,
            )
        case Solver.ABM2:
            return _abm2(
                velocity,
                initial_state,
                schedule,
                resolved_progress,
            )
        case _ as unreachable:
            raise RuntimeError(f"Unsupported Canter solver: {unreachable}")
