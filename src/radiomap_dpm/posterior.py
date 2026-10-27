"""Measurement-consistent posterior sampling for RadioMap diffusion.

The known building pixels and the paid outdoor samples deliberately take different
paths:

* building values are free, exact observations and are hard-projected at every reverse
  step using their forward-noised marginal;
* noisy outdoor measurements use a variance-aware DPS/pi-GDM likelihood gradient;
* a zero-noise outdoor measurement is promoted to the exact-observation set.

The current scene's known building layout may additionally be bound to a checkpoint's
native environment-condition input.  Transmitter information is never forwarded to the
network or selector.  Building conditioning and noise-free building measurement are kept
separate: the former supplies geometry, while the latter enforces the stipulated exact RM
values at building pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


class DiffusionAdapter(Protocol):
    """Small interface required by :class:`MeasurementConsistentSampler`."""

    device: torch.device
    image_size: int
    alphas_cumprod: torch.Tensor

    def model_predictions(self, x_t: torch.Tensor, t: torch.Tensor) -> Any:
        """Return an object/tuple containing predicted noise and predicted x0."""


@dataclass(frozen=True)
class SparseObservations:
    """Paid outdoor measurements in normalized gain units ``[0, 1]``."""

    idx: np.ndarray
    values: np.ndarray
    sigma2: np.ndarray

    @classmethod
    def empty(cls) -> "SparseObservations":
        return cls(
            idx=np.empty(0, dtype=np.int64),
            values=np.empty(0, dtype=np.float32),
            sigma2=np.empty(0, dtype=np.float32),
        )

    def validated(self, n_pixels: int, building_mask: np.ndarray) -> "SparseObservations":
        idx = np.asarray(self.idx, dtype=np.int64).reshape(-1)
        val = np.asarray(self.values, dtype=np.float32).reshape(-1)
        var = np.asarray(self.sigma2, dtype=np.float32).reshape(-1)
        if not (idx.size == val.size == var.size):
            raise ValueError("idx, values and sigma2 must have the same length")
        if idx.size and (idx.min() < 0 or idx.max() >= n_pixels):
            raise ValueError("outdoor measurement index is out of range")
        if np.unique(idx).size != idx.size:
            raise ValueError("outdoor measurement indices must be unique")
        if np.any(var < 0):
            raise ValueError("measurement variances must be non-negative")
        if np.any(np.asarray(building_mask, dtype=bool).reshape(-1)[idx]):
            raise ValueError("paid outdoor measurements may not lie inside buildings")
        if np.any(~np.isfinite(val)) or np.any(~np.isfinite(var)):
            raise ValueError("measurement values and variances must be finite")
        return SparseObservations(idx=idx, values=val, sigma2=var)


@dataclass(frozen=True)
class PosteriorConfig:
    steps: int = 32
    n_particles: int = 12
    chunk_size: int = 3
    guidance_scale: float = 0.35
    max_grad_norm: float = 1.0
    exact_variance_threshold: float = 1e-12
    enforce_building_observations: bool = True
    soft_constraint_weight: float = 0.35
    seed: int = 0

    def validate(self) -> None:
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if self.n_particles < 1 or self.chunk_size < 1:
            raise ValueError("n_particles and chunk_size must be positive")
        if self.guidance_scale < 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid guidance configuration")
        if not np.isfinite(self.soft_constraint_weight) or self.soft_constraint_weight < 0:
            raise ValueError("soft_constraint_weight must be finite and non-negative")
        if not isinstance(self.enforce_building_observations, (bool, np.bool_)):
            raise ValueError("enforce_building_observations must be boolean")


def _prediction_parts(prediction: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Support namedtuple, attribute object and two-tuple model outputs."""

    if hasattr(prediction, "pred_noise") and hasattr(prediction, "pred_x_start"):
        return prediction.pred_noise, prediction.pred_x_start
    if isinstance(prediction, (tuple, list)) and len(prediction) >= 2:
        return prediction[0], prediction[1]
    raise TypeError("model_predictions must expose pred_noise and pred_x_start")


class MeasurementConsistentSampler:
    """DDIM posterior sampler with exact-building and sparse-data consistency."""

    def __init__(self, prior: DiffusionAdapter):
        self.prior = prior
        self.device = torch.device(prior.device)
        size = prior.image_size
        if isinstance(size, (tuple, list)):
            if len(size) != 2 or int(size[0]) != int(size[1]):
                raise ValueError("only square RadioMap images are supported")
            size = int(size[0])
        self.image_size = int(size)
        self.n_pixels = self.image_size * self.image_size
        self.alphas = prior.alphas_cumprod.to(self.device)

    def sample(
        self,
        building_mask: np.ndarray,
        building_values: np.ndarray,
        observations: SparseObservations | None = None,
        config: PosteriorConfig | None = None,
        soft_constraint: SparseObservations | None = None,
    ) -> np.ndarray:
        """Draw posterior particles as ``(M, H*W)`` float32 arrays in ``[0, 1]``.

        ``building_values`` is the true radio-map value at the known building pixels,
        not the binary building mask. This is the user's stipulated free/noise-free
        observation. The binary mask is used only to identify exact indices.
        """

        cfg = config or PosteriorConfig()
        cfg.validate()
        bmask = np.asarray(building_mask, dtype=bool)
        bvals = np.asarray(building_values, dtype=np.float32)
        expected = (self.image_size, self.image_size)
        if bmask.shape != expected or bvals.shape != expected:
            raise ValueError(f"building arrays must have shape {expected}")
        if np.any(~np.isfinite(bvals)):
            raise ValueError("building values must be finite")
        obs = (observations or SparseObservations.empty()).validated(
            self.n_pixels, bmask
        )
        soft = (soft_constraint or SparseObservations.empty()).validated(
            self.n_pixels, bmask
        )

        # The released model-50 checkpoint was trained with three repeated building
        # channels.  A building-aware adapter binds this scene once and exposes the same
        # condition-free prediction interface to the reverse loop.  Fixed-null adapters
        # remain supported for reproducibility of earlier diagnostics.
        bind = getattr(self.prior, "bind_environment", None)
        scene_prior = bind(bmask) if callable(bind) else self.prior

        pieces: list[np.ndarray] = []
        for start in range(0, cfg.n_particles, cfg.chunk_size):
            count = min(cfg.chunk_size, cfg.n_particles - start)
            pieces.append(
                self._sample_chunk(
                    count=count,
                    building_mask=bmask,
                    building_values=bvals,
                    observations=obs,
                    soft_constraint=soft,
                    config=cfg,
                    seed=cfg.seed + 1_000_003 * start,
                    prior=scene_prior,
                )
            )
        out = np.concatenate(pieces, axis=0)
        if out.shape != (cfg.n_particles, self.n_pixels):
            raise RuntimeError(f"unexpected posterior shape {out.shape}")
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("posterior sampler produced nan/inf")
        return out

    def sample_batch(
        self,
        building_masks: np.ndarray,
        building_values: np.ndarray,
        observations: list[SparseObservations | None] | tuple[SparseObservations | None, ...],
        config: PosteriorConfig | None = None,
        soft_constraints: list[SparseObservations | None]
        | tuple[SparseObservations | None, ...]
        | None = None,
        soft_constraint_weights: list[float] | tuple[float, ...] | None = None,
        seeds: list[int] | tuple[int, ...] | None = None,
    ) -> list[np.ndarray]:
        """Draw independent posterior banks with batched diffusion forward passes.

        All instances share the numerical sampler configuration, while their building
        masks, sparse observations, random seeds, and posterior states remain separate.
        The returned list contains one ``(M,H*W)`` bank per input instance.
        """

        cfg = config or PosteriorConfig()
        cfg.validate()
        masks = np.asarray(building_masks, dtype=bool)
        values = np.asarray(building_values, dtype=np.float32)
        if masks.ndim != 3 or tuple(masks.shape[1:]) != (self.image_size, self.image_size):
            raise ValueError("building_masks must have shape (B,H,W)")
        if values.shape != masks.shape:
            raise ValueError("building_values must match building_masks")
        if np.any(~np.isfinite(values)):
            raise ValueError("building_values must be finite")
        batch = int(masks.shape[0])
        if batch < 1 or len(observations) != batch:
            raise ValueError("one observation set is required per batch instance")
        if soft_constraints is None:
            soft_constraints = tuple(None for _ in range(batch))
        if len(soft_constraints) != batch:
            raise ValueError("one soft constraint set is required per batch instance")
        if soft_constraint_weights is None:
            soft_constraint_weights = tuple(float(cfg.soft_constraint_weight) for _ in range(batch))
        if len(soft_constraint_weights) != batch:
            raise ValueError("one soft constraint weight is required per batch instance")
        soft_weights = np.asarray(soft_constraint_weights, dtype=np.float32)
        if np.any(~np.isfinite(soft_weights)) or np.any(soft_weights < 0.0):
            raise ValueError("soft constraint weights must be finite and non-negative")
        if seeds is None:
            seeds = tuple(int(cfg.seed) + 1_000_003 * i for i in range(batch))
        if len(seeds) != batch:
            raise ValueError("one random seed is required per batch instance")
        validated_observations = tuple(
            (item or SparseObservations.empty()).validated(self.n_pixels, masks[i])
            for i, item in enumerate(observations)
        )
        validated_soft = tuple(
            (item or SparseObservations.empty()).validated(self.n_pixels, masks[i])
            for i, item in enumerate(soft_constraints)
        )

        bind = getattr(self.prior, "bind_environment", None)
        scene_prior = bind(masks) if callable(bind) else self.prior
        pieces: list[torch.Tensor] = []
        for start in range(0, cfg.n_particles, cfg.chunk_size):
            count = min(cfg.chunk_size, cfg.n_particles - start)
            pieces.append(
                self._sample_batch_chunk(
                    count=count,
                    building_masks=masks,
                    building_values=values,
                    observations=validated_observations,
                    soft_constraints=validated_soft,
                    soft_constraint_weights=tuple(float(value) for value in soft_weights),
                    config=cfg,
                    seeds=tuple(int(seed) + 1_000_003 * start for seed in seeds),
                    prior=scene_prior,
                )
            )
        banks = torch.cat(pieces, dim=1).float().cpu().numpy()
        if banks.shape != (batch, cfg.n_particles, self.n_pixels):
            raise RuntimeError(f"unexpected batched posterior shape {banks.shape}")
        if not np.all(np.isfinite(banks)):
            raise FloatingPointError("batched posterior sampler produced nan/inf")
        return [banks[i].copy() for i in range(batch)]

    def _sample_batch_chunk(
        self,
        count: int,
        building_masks: np.ndarray,
        building_values: np.ndarray,
        observations: tuple[SparseObservations, ...],
        soft_constraints: tuple[SparseObservations, ...],
        soft_constraint_weights: tuple[float, ...],
        config: PosteriorConfig,
        seeds: tuple[int, ...],
        prior: DiffusionAdapter,
    ) -> torch.Tensor:
        device = self.device
        batch, height, width = building_masks.shape
        pixels = height * width
        generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
        x = torch.cat(
            [
                torch.randn((count, 1, height, width), generator=g, device=device)
                for g in generators
            ],
            dim=0,
        )
        known_noise = torch.cat(
            [
                torch.randn((count, 1, height, width), generator=g, device=device)
                for g in generators
            ],
            dim=0,
        ).reshape(batch, count, pixels)

        hard_indices: list[np.ndarray] = []
        hard_values: list[np.ndarray] = []
        sparse_indices: list[np.ndarray] = []
        sparse_values: list[np.ndarray] = []
        sparse_variances: list[np.ndarray] = []
        peak_indices: list[np.ndarray] = []
        peak_values: list[np.ndarray] = []
        peak_variances: list[np.ndarray] = []
        for i, (mask, values, obs, soft) in enumerate(
            zip(building_masks, building_values, observations, soft_constraints)
        ):
            building_idx = (
                np.flatnonzero(mask.reshape(-1)).astype(np.int64)
                if config.enforce_building_observations
                else np.empty(0, dtype=np.int64)
            )
            exact = obs.sigma2 <= config.exact_variance_threshold
            exact_idx = obs.idx[exact]
            exact_values = obs.values[exact]
            hard_indices.append(np.concatenate((building_idx, exact_idx)))
            hard_values.append(
                np.concatenate((values.reshape(-1)[building_idx], exact_values))
            )
            noisy = ~exact
            sparse_indices.append(obs.idx[noisy])
            sparse_values.append(obs.values[noisy])
            sparse_variances.append(obs.sigma2[noisy])
            peak_indices.append(soft.idx)
            peak_values.append(soft.values)
            peak_variances.append(np.maximum(soft.sigma2, np.float32(1e-6)))

        def _pad_arrays(items: list[np.ndarray], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
            width_ = max((item.size for item in items), default=0)
            idx = np.zeros((batch, width_), dtype=np.int64)
            mask = np.zeros((batch, width_), dtype=bool)
            for row, item in enumerate(items):
                idx[row, : item.size] = item
                mask[row, : item.size] = True
            return (
                torch.as_tensor(idx, device=device, dtype=torch.long),
                torch.as_tensor(mask, device=device, dtype=torch.bool),
            )

        sparse_idx, sparse_mask = _pad_arrays(sparse_indices, torch.long)
        peak_idx, peak_mask = _pad_arrays(peak_indices, torch.long)
        sparse_y = torch.zeros_like(sparse_idx, dtype=torch.float32)
        sparse_var = torch.ones_like(sparse_y)
        peak_y = torch.zeros_like(peak_idx, dtype=torch.float32)
        peak_var = torch.ones_like(peak_y)
        peak_weight = torch.as_tensor(
            soft_constraint_weights, device=device, dtype=torch.float32
        )
        for row in range(batch):
            n_sparse = sparse_indices[row].size
            n_peak = peak_indices[row].size
            if n_sparse:
                sparse_y[row, :n_sparse] = torch.as_tensor(
                    sparse_values[row] * 2.0 - 1.0, device=device
                )
                sparse_var[row, :n_sparse] = torch.as_tensor(
                    sparse_variances[row] * 4.0, device=device
                )
            if n_peak:
                peak_y[row, :n_peak] = torch.as_tensor(
                    peak_values[row] * 2.0 - 1.0, device=device
                )
                peak_var[row, :n_peak] = torch.as_tensor(
                    peak_variances[row] * 4.0, device=device
                )

        hard_tensors = [
            (
                torch.as_tensor(indices, device=device, dtype=torch.long),
                torch.as_tensor(values * 2.0 - 1.0, device=device, dtype=torch.float32),
            )
            for indices, values in zip(hard_indices, hard_values)
        ]
        total = int(self.alphas.numel())
        times = torch.linspace(-1, total - 1, config.steps + 1, device=device)
        times = list(reversed(times.long().tolist()))
        pairs = list(zip(times[:-1], times[1:]))

        for time_now, time_next in pairs:
            use_guidance = config.guidance_scale > 0.0 and (
                bool(sparse_mask.any())
                or bool((peak_mask.any(dim=1) & (peak_weight > 0.0)).any())
            )
            x = x.detach().requires_grad_(use_guidance)
            t = torch.full((batch * count,), time_now, device=device, dtype=torch.long)
            with torch.set_grad_enabled(use_guidance):
                pred_noise, x0 = _prediction_parts(prior.model_predictions(x, t))
                del pred_noise
                x0 = x0.clamp(-1.0, 1.0)
                flat_x0 = x0.reshape(batch, count, pixels)
                nll = torch.zeros((), device=device, dtype=x.dtype)
                abar = self.alphas[time_now]
                diffusion_var = (1.0 - abar) / abar.clamp_min(1e-8)
                if sparse_idx.shape[1]:
                    gathered = torch.gather(
                        flat_x0,
                        2,
                        sparse_idx[:, None, :].expand(batch, count, -1),
                    )
                    residual = gathered - sparse_y[:, None, :]
                    denom = sparse_var[:, None, :] + diffusion_var
                    nll = nll + 0.5 * (
                        residual.square() / denom.clamp_min(1e-8)
                        * sparse_mask[:, None, :]
                    ).sum()
                if peak_idx.shape[1] and bool((peak_weight > 0.0).any()):
                    gathered = torch.gather(
                        flat_x0,
                        2,
                        peak_idx[:, None, :].expand(batch, count, -1),
                    )
                    residual = gathered - peak_y[:, None, :]
                    denom = peak_var[:, None, :] + diffusion_var
                    count_mask = peak_mask.sum(dim=1).clamp_min(1).to(x.dtype)
                    peak_loss = (
                        residual.square() / denom.clamp_min(1e-8)
                        * peak_mask[:, None, :]
                    ).sum(dim=2) / count_mask[:, None]
                    nll = nll + 0.5 * (peak_weight[:, None] * peak_loss).sum()
                grad = torch.autograd.grad(nll, x, retain_graph=False)[0] if use_guidance else None

            with torch.no_grad():
                abar_now = self.alphas[time_now]
                x0_flat = x0.reshape(batch, count, pixels).clone()
                for row, (indices, values) in enumerate(hard_tensors):
                    if indices.numel():
                        x0_flat[row, :, indices] = values[None, :]
                x0_projected = x0_flat.reshape_as(x0)
                eps_projected = (
                    x.detach() - abar_now.sqrt() * x0_projected
                ) / (1.0 - abar_now).clamp_min(1e-8).sqrt()
                if time_next < 0:
                    x_next = x0_projected
                else:
                    abar_next = self.alphas[time_next]
                    x_next = (
                        abar_next.sqrt() * x0_projected
                        + (1.0 - abar_next).sqrt() * eps_projected
                    )
                if grad is not None:
                    g = grad.reshape(batch, count, -1)
                    norm = torch.linalg.vector_norm(g, dim=2, keepdim=True).clamp_min(1e-8)
                    scale = torch.clamp(config.max_grad_norm / norm, max=1.0).reshape(
                        batch * count, 1, 1, 1
                    )
                    x_next = x_next - (
                        config.guidance_scale * (1.0 - abar_now).sqrt() * scale * grad
                    )
                flat_next = x_next.reshape(batch, count, pixels)
                for row, (indices, values) in enumerate(hard_tensors):
                    if indices.numel():
                        if time_next < 0:
                            hard_t = values[None, :].expand(count, -1)
                        else:
                            abar_next = self.alphas[time_next]
                            eps_known = known_noise[row, :, indices]
                            hard_t = (
                                abar_next.sqrt() * values[None, :]
                                + (1.0 - abar_next).sqrt() * eps_known
                            )
                        flat_next[row, :, indices] = hard_t
                x = flat_next.reshape_as(x_next).detach()

        result = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).reshape(batch, count, pixels)
        for row, (indices, values) in enumerate(hard_tensors):
            if indices.numel():
                result[row, :, indices] = (values[None, :] + 1.0) * 0.5
        return result

    def _sample_chunk(
        self,
        count: int,
        building_mask: np.ndarray,
        building_values: np.ndarray,
        observations: SparseObservations,
        soft_constraint: SparseObservations,
        config: PosteriorConfig,
        seed: int,
        prior: DiffusionAdapter,
    ) -> np.ndarray:
        device = self.device
        h = w = self.image_size
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        x = torch.randn((count, 1, h, w), generator=generator, device=device)
        known_noise = torch.randn(
            (count, 1, h, w), generator=generator, device=device
        )

        bidx_np = (
            np.flatnonzero(building_mask.reshape(-1)).astype(np.int64)
            if config.enforce_building_observations
            else np.empty(0, dtype=np.int64)
        )
        bidx = torch.as_tensor(bidx_np, device=device, dtype=torch.long)
        by = torch.as_tensor(
            building_values.reshape(-1)[bidx_np] * 2.0 - 1.0,
            device=device,
            dtype=torch.float32,
        )

        exact_obs = observations.sigma2 <= config.exact_variance_threshold
        exact_idx_np = observations.idx[exact_obs]
        exact_y_np = observations.values[exact_obs]
        if exact_idx_np.size:
            exact_idx = torch.as_tensor(exact_idx_np, device=device, dtype=torch.long)
            exact_y = torch.as_tensor(
                exact_y_np * 2.0 - 1.0, device=device, dtype=torch.float32
            )
            hard_idx = torch.cat((bidx, exact_idx), dim=0)
            hard_y = torch.cat((by, exact_y), dim=0)
        else:
            hard_idx, hard_y = bidx, by

        noisy = ~exact_obs
        sidx = torch.as_tensor(
            observations.idx[noisy], device=device, dtype=torch.long
        )
        sy = torch.as_tensor(
            observations.values[noisy] * 2.0 - 1.0,
            device=device,
            dtype=torch.float32,
        )
        # Variance scales by four under y -> 2y-1.
        svar = torch.as_tensor(
            observations.sigma2[noisy] * 4.0,
            device=device,
            dtype=torch.float32,
        )
        pidx = torch.as_tensor(
            soft_constraint.idx, device=device, dtype=torch.long
        )
        py = torch.as_tensor(
            soft_constraint.values * 2.0 - 1.0,
            device=device,
            dtype=torch.float32,
        )
        pvar = torch.as_tensor(
            np.maximum(soft_constraint.sigma2, np.float32(1e-6)) * 4.0,
            device=device,
            dtype=torch.float32,
        )

        total = int(self.alphas.numel())
        times = torch.linspace(-1, total - 1, config.steps + 1, device=device)
        times = list(reversed(times.long().tolist()))
        pairs = list(zip(times[:-1], times[1:]))

        for time_now, time_next in pairs:
            use_guidance = (
                (sidx.numel() > 0 or pidx.numel() > 0)
                and config.guidance_scale > 0
            )
            x = x.detach().requires_grad_(use_guidance)
            t = torch.full((count,), time_now, device=device, dtype=torch.long)
            with torch.set_grad_enabled(use_guidance):
                pred_noise, x0 = _prediction_parts(prior.model_predictions(x, t))
                x0 = x0.clamp(-1.0, 1.0)
                grad = None
                if use_guidance:
                    abar = self.alphas[time_now]
                    diffusion_var = (1.0 - abar) / abar.clamp_min(1e-8)
                    flat_x0 = x0.reshape(count, -1)
                    # Independent paid measurements accumulate as a sum.  The former
                    # global mean divided the evidence by both sensor count and chunk
                    # size, making guidance depend on an implementation detail and
                    # permanently weakening larger budgets.
                    nll = torch.zeros((), device=device, dtype=x.dtype)
                    if sidx.numel():
                        residual = flat_x0[:, sidx] - sy[None, :]
                        denom = svar[None, :] + diffusion_var
                        nll = nll + 0.5 * (
                            residual.square() / denom.clamp_min(1e-8)
                        ).sum()
                    if pidx.numel() and config.soft_constraint_weight > 0.0:
                        peak_residual = flat_x0[:, pidx] - py[None, :]
                        peak_denom = pvar[None, :] + diffusion_var
                        # A source hypothesis is a latent soft constraint, not a paid
                        # observation.  Normalize over its patch so it cannot dominate
                        # the true sparse likelihood merely by containing many pixels.
                        peak_loss = (
                            peak_residual.square() / peak_denom.clamp_min(1e-8)
                        ).mean(dim=1).sum()
                        nll = nll + 0.5 * config.soft_constraint_weight * peak_loss
                    grad = torch.autograd.grad(nll, x, retain_graph=False)[0]

            with torch.no_grad():
                abar_now = self.alphas[time_now]
                x0_flat = x0.reshape(count, -1).clone()
                if hard_idx.numel():
                    x0_flat[:, hard_idx] = hard_y[None, :]
                x0_projected = x0_flat.reshape_as(x0)
                # Re-derive epsilon after exact x0 projection (DDNM range projection).
                eps_projected = (
                    x.detach() - abar_now.sqrt() * x0_projected
                ) / (1.0 - abar_now).clamp_min(1e-8).sqrt()

                if time_next < 0:
                    x_next = x0_projected
                else:
                    abar_next = self.alphas[time_next]
                    x_next = (
                        abar_next.sqrt() * x0_projected
                        + (1.0 - abar_next).sqrt() * eps_projected
                    )

                if grad is not None:
                    g = grad.reshape(count, -1)
                    norm = torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-8)
                    scale = torch.clamp(
                        config.max_grad_norm / norm, max=1.0
                    ).reshape(count, 1, 1, 1)
                    step_scale = config.guidance_scale * (1.0 - abar_now).sqrt()
                    x_next = x_next - step_scale * scale * grad

                # RePaint-style exact observation in the correct q(y_t | y_0) marginal.
                if hard_idx.numel():
                    flat_next = x_next.reshape(count, -1)
                    if time_next < 0:
                        hard_t = hard_y[None, :].expand(count, -1)
                    else:
                        abar_next = self.alphas[time_next]
                        eps_known = known_noise.reshape(count, -1)[:, hard_idx]
                        hard_t = (
                            abar_next.sqrt() * hard_y[None, :]
                            + (1.0 - abar_next).sqrt() * eps_known
                        )
                    flat_next[:, hard_idx] = hard_t
                    x_next = flat_next.reshape_as(x_next)
                x = x_next.detach()

        result = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).reshape(count, -1)
        # Final exact overwrite is intentional and testable.
        if hard_idx.numel():
            result[:, hard_idx] = (hard_y[None, :] + 1.0) * 0.5
        return result.float().cpu().numpy()


def posterior_mean(particles: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return a validated weighted posterior mean."""

    x = np.asarray(particles, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("particles must have shape (M, D)")
    if weights is None:
        return x.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != x.shape[0] or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("invalid particle weights")
    total = w.sum()
    if total <= 0:
        raise ValueError("particle weights sum to zero")
    return (w / total) @ x
