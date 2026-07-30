from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .solver_core import Solver, solve


def _time(t, batch, device):
    return t.expand(batch).to(device=device, dtype=torch.float32)


def _validate(x, sigmas):
    if x.ndim != 4 or x.dtype != torch.float32:
        raise ValueError("Canter solver state must be float32 [B,C,H,W]")
    if sigmas.ndim != 1 or len(sigmas) < 2 or sigmas.dtype != torch.float32:
        raise ValueError("Canter sigmas must be a float32 vector with at least two points")
    if not torch.all(sigmas[:-1] > sigmas[1:]):
        raise ValueError("Canter sigmas must be strictly descending")


def _velocity(model, x, sigma, index, **extra):
    denoised = model(x, sigma, **extra)
    return (x - denoised) / sigma.view(-1, 1, 1, 1).clamp_min(1.0e-12)


def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    del disable, kwargs
    extra_args = extra_args or {}
    x = x.float()
    _validate(x, sigmas)
    for i in range(len(sigmas) - 1):
        t = _time(sigmas[i], len(x), x.device)
        denoised = model(x, t, **extra_args)
        x = x + (sigmas[i + 1] - sigmas[i]) * ((x - denoised) / sigmas[i])
        if callback:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
    return x


def sample_euler_maruyama(model, x, sigmas, extra_args=None, callback=None,
                          disable=None, noise_multiplier=1.0, seed=None, **kwargs):
    del disable, kwargs
    extra_args = extra_args or {}
    x = x.float()
    _validate(x, sigmas)
    generator = torch.Generator(device=x.device).manual_seed(int(seed or 0))
    for i in range(len(sigmas) - 1):
        t = _time(sigmas[i], len(x), x.device)
        denoised = model(x, t, **extra_args)
        velocity = (x - denoised) / sigmas[i].clamp_min(1.0e-12)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            score = -(((1 - sigmas[i]) * velocity) + x) / sigmas[i].clamp_min(1.0e-4)
            dt = sigmas[i + 1] - sigmas[i]
            drift = velocity - sigmas[i] * score
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=torch.float32)
            x = x + dt * drift + noise_multiplier * math.sqrt(2 * float(sigmas[i])) * math.sqrt(abs(float(dt))) * noise
        if callback:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
    return x


def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    del disable, kwargs
    extra_args = extra_args or {}
    x = x.float()
    _validate(x, sigmas)
    old, old_lambda = None, None
    lambdas = torch.log1p(-sigmas.double()) - torch.log(sigmas.double().clamp_min(1e-30))
    for i in range(len(sigmas) - 1):
        t = _time(sigmas[i], len(x), x.device)
        denoised = model(x, t, **extra_args).float()
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            h = lambdas[i + 1] - lambdas[i]
            extrapolated = denoised
            if old is not None:
                ratio = (lambdas[i] - old_lambda) / h
                extrapolated = (1 + 1 / (2 * ratio)) * denoised - old / (2 * ratio)
            sigma_ratio = torch.exp(F.logsigmoid(-lambdas[i + 1]) - F.logsigmoid(-lambdas[i])).to(x)
            alpha = torch.sigmoid(lambdas[i + 1]).to(x)
            x = sigma_ratio * x + alpha * (-torch.expm1(-h)).to(x) * extrapolated
        old, old_lambda = denoised, lambdas[i]
        if callback:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
    return x


def sample_abm2(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    del disable, kwargs
    extra_args = extra_args or {}
    x = x.float()
    _validate(x, sigmas)
    previous = None
    previous_dt = None
    for i in range(len(sigmas) - 1):
        t = _time(sigmas[i], len(x), x.device)
        denoised = model(x, t, **extra_args)
        velocity = (x - denoised) / sigmas[i].clamp_min(1e-12)
        dt = sigmas[i + 1] - sigmas[i]
        if previous is None:
            predicted = x + dt * velocity
        else:
            ratio = dt / previous_dt
            predicted = x + dt * ((1 + ratio / 2) * velocity - ratio * previous / 2)
        if sigmas[i + 1] == 0:
            x = predicted
        else:
            next_t = _time(sigmas[i + 1], len(x), x.device)
            next_denoised = model(predicted, next_t, **extra_args)
            corrected = (predicted - next_denoised) / sigmas[i + 1]
            x = x + dt * (velocity + corrected) / 2
        previous, previous_dt = velocity, dt
        if callback:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
    return x


def sample_native(model, x, sigmas, extra_args=None, callback=None, disable=None,
                  solver="abm2", noise_multiplier=1.0, seed=None, **kwargs):
    del disable, kwargs
    extra_args = extra_args or {}
    x = x.float()

    def velocity(state, time, schedule_index):
        del schedule_index
        denoised = model(state, time, **extra_args)
        return (state - denoised) / time.view(-1, 1, 1, 1).clamp_min(1.0e-12)

    def progress(completed, total):
        if callback:
            index = completed - 1
            callback({
                "x": x, "i": index, "sigma": sigmas[index],
                "sigma_hat": sigmas[index], "denoised": x,
            })

    generator = torch.Generator(device=x.device).manual_seed(int(seed or 0))
    return solve(
        velocity, x, sigmas.float(), solver=Solver(solver),
        generator=generator,
        euler_maruyama_multiplier=noise_multiplier,
        er_sde_noise_multiplier=noise_multiplier, progress=progress,
    )
