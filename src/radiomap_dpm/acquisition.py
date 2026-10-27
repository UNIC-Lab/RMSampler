"""Unknown-transmitter active acquisition for RadioMapSeer DPM maps.

The routines in this module deliberately know nothing about a held-out radio map or
the true transmitter location.  They use only

* unconditional posterior particles,
* their importance weights,
* the known building mask, and
* locations that have already been measured.

Buildings are excluded from the candidate and outdoor reconstruction target sets.
They can therefore be imposed as exact (noise-free) measurements by the reconstruction
operator without spending any of the outdoor sampling budget.

All covariance calculations use the low-rank weighted ensemble factor ``F`` satisfying
``C = F.T @ F``; a dense pixel-by-pixel covariance is never formed.  Batch points are
selected serially and ``F`` is exactly square-root-downdated after every point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


_EPS = np.finfo(np.float64).eps


def _normalise_weights(weights: np.ndarray | Sequence[float] | None,
                       n_particles: int) -> np.ndarray:
    """Return finite, non-negative weights that sum to one."""
    if n_particles < 1:
        raise ValueError("at least one particle is required")
    if weights is None:
        return np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
    out = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
    if out.shape != (n_particles,):
        raise ValueError(
            f"weights must have shape ({n_particles},), got {out.shape}")
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(out.sum())
    if total <= 0.0:
        raise ValueError("weights must contain positive mass")
    out /= total
    return out


def effective_sample_size(weights: np.ndarray | Sequence[float]) -> float:
    """Effective sample size ``1 / sum(w**2)`` for normalised weights."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = _normalise_weights(w, w.size)
    return float(1.0 / max(float(np.dot(w, w)), _EPS))


def _spatial_shape(building_mask: np.ndarray) -> tuple[int, int]:
    mask = np.asarray(building_mask)
    if mask.ndim != 2:
        raise ValueError(f"building_mask must be 2-D, got shape {mask.shape}")
    if mask.size == 0:
        raise ValueError("building_mask cannot be empty")
    return int(mask.shape[0]), int(mask.shape[1])


def _flatten_particles(particles: np.ndarray,
                       shape: tuple[int, int]) -> np.ndarray:
    x = np.asarray(particles, dtype=np.float64)
    d = int(np.prod(shape))
    if x.ndim == 3 and tuple(x.shape[1:]) == shape:
        flat = x.reshape(x.shape[0], d)
    elif x.ndim == 2 and x.shape[1] == d:
        flat = x
    else:
        raise ValueError(
            "particles must have shape (P,H,W) or (P,H*W) matching "
            f"building_mask {shape}; got {x.shape}")
    if x.shape[0] < 1:
        raise ValueError("at least one particle is required")
    if not np.all(np.isfinite(flat)):
        raise ValueError("particles must be finite")
    return np.ascontiguousarray(flat, dtype=np.float64)


def _as_mask(mask: np.ndarray | None, shape: tuple[int, int],
             name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out.copy()


def _flat_indices(indices: np.ndarray | Sequence[int] | Sequence[Sequence[int]],
                  shape: tuple[int, int]) -> np.ndarray:
    """Accept flat indices or ``(row, col)`` pairs and return flat indices."""
    arr = np.asarray(indices)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        rr = arr[:, 0].astype(np.int64, copy=False)
        cc = arr[:, 1].astype(np.int64, copy=False)
        if np.any(rr < 0) or np.any(rr >= shape[0]) \
                or np.any(cc < 0) or np.any(cc >= shape[1]):
            raise IndexError("coordinate outside the map")
        return np.ravel_multi_index((rr, cc), shape).astype(np.int64)
    out = arr.astype(np.int64, copy=False).reshape(-1)
    if np.any(out < 0) or np.any(out >= int(np.prod(shape))):
        raise IndexError("flat index outside the map")
    return out


def indices_to_coordinates(indices: np.ndarray | Sequence[int],
                           shape: tuple[int, int]) -> np.ndarray:
    """Convert row-major flat indices to an integer ``(K,2)`` array."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    rr, cc = np.unravel_index(idx, shape)
    return np.column_stack((rr, cc)).astype(np.int64, copy=False)


def weighted_mean_and_factor(
    particles: np.ndarray,
    weights: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted mean and low-rank population-covariance factor.

    For ``particles`` of shape ``(P,D)``, the returned factor has the same shape and
    obeys ``factor.T @ factor == Cov_w(particles)`` (up to round-off).  Population
    rather than unbiased sample covariance is appropriate because the particles and
    weights represent a discrete posterior distribution.
    """
    x = np.asarray(particles, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("particles must have shape (P,D), with P >= 1")
    if not np.all(np.isfinite(x)):
        raise ValueError("particles must be finite")
    w = _normalise_weights(weights, x.shape[0])
    mean = w @ x
    factor = np.sqrt(w)[:, None] * (x - mean[None, :])
    return mean, factor


@dataclass(frozen=True)
class WeightUpdate:
    """Result of a Gaussian importance-weight update."""

    weights: np.ndarray
    ess: float
    log_evidence: float
    collapsed: bool

    def __iter__(self):
        # Convenient ``weights, ess = gaussian_weight_update(...)`` compatibility.
        yield self.weights
        yield self.ess


def gaussian_weight_update(
    weights: np.ndarray | Sequence[float] | None,
    predictions: np.ndarray,
    observations: np.ndarray | Sequence[float],
    noise_variance: float | np.ndarray,
    *,
    tempering: float = 1.0,
    exact_tolerance: float = 1e-8,
) -> WeightUpdate:
    """Update particle weights with independent Gaussian measurements.

    ``predictions`` has shape ``(P,K)`` and ``observations`` has shape ``(K,)``.
    A scalar or per-observation variance is accepted.  Zero variance is interpreted
    as an exact observation and implemented by a tolerance-width limiting Gaussian;
    this is numerically stable even when no finite-precision particle is bit-identical
    to the observation.  The reconstruction path should normally project the known
    building values exactly, while this update is used for newly sampled outdoor data.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.ndim != 2 or pred.shape[0] < 1:
        raise ValueError("predictions must have shape (P,K)")
    obs = np.asarray(observations, dtype=np.float64).reshape(-1)
    if obs.shape != (pred.shape[1],):
        raise ValueError(
            f"observations must have shape ({pred.shape[1]},), got {obs.shape}")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(obs)):
        raise ValueError("predictions and observations must be finite")
    if not np.isfinite(tempering) or tempering <= 0.0:
        raise ValueError("tempering must be finite and > 0")
    if not np.isfinite(exact_tolerance) or exact_tolerance <= 0.0:
        raise ValueError("exact_tolerance must be finite and > 0")

    variance = np.asarray(noise_variance, dtype=np.float64)
    if variance.ndim == 0:
        variance = np.full(obs.size, float(variance), dtype=np.float64)
    else:
        variance = variance.reshape(-1)
        if variance.shape != obs.shape:
            raise ValueError(
                f"noise_variance must be scalar or shape {obs.shape}, got "
                f"{variance.shape}")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("noise_variance must be finite and non-negative")

    prior = _normalise_weights(weights, pred.shape[0])
    residual2 = (pred - obs[None, :]) ** 2
    effective_variance = np.maximum(variance, exact_tolerance ** 2)
    log_like = -0.5 * np.sum(
        residual2 / effective_variance[None, :]
        + np.log(2.0 * np.pi * effective_variance)[None, :], axis=1)
    log_unnormalised = np.log(np.maximum(prior, np.finfo(float).tiny)) \
        + tempering * log_like
    max_log = float(np.max(log_unnormalised))
    shifted = np.exp(log_unnormalised - max_log)
    normaliser = float(shifted.sum())
    collapsed = not np.isfinite(normaliser) or normaliser <= 0.0
    if collapsed:
        posterior = prior.copy()
        log_evidence = float("-inf")
    else:
        posterior = shifted / normaliser
        log_evidence = max_log + float(np.log(normaliser))
    return WeightUpdate(
        weights=posterior,
        ess=effective_sample_size(posterior),
        log_evidence=log_evidence,
        collapsed=collapsed,
    )


def update_particle_weights(
    weights: np.ndarray | Sequence[float] | None,
    predictions: np.ndarray,
    observations: np.ndarray | Sequence[float],
    noise_variance: float | np.ndarray,
    *,
    tempering: float = 1.0,
    exact_tolerance: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Tuple-returning convenience wrapper around :func:`gaussian_weight_update`."""
    out = gaussian_weight_update(
        weights, predictions, observations, noise_variance,
        tempering=tempering, exact_tolerance=exact_tolerance)
    return out.weights, out.ess


@dataclass
class ParticleBelief:
    """Weighted radio-map particles used between closed-loop acquisition rounds."""

    particles: np.ndarray
    weights: np.ndarray | Sequence[float] | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.particles, dtype=np.float64)
        if x.ndim not in (2, 3) or x.shape[0] < 1:
            raise ValueError("particles must have shape (P,D) or (P,H,W)")
        if not np.all(np.isfinite(x)):
            raise ValueError("particles must be finite")
        self.particles = x
        self.weights = _normalise_weights(self.weights, x.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.particles.shape[0])

    @property
    def ess(self) -> float:
        return effective_sample_size(self.weights)

    def update_at(
        self,
        indices: np.ndarray | Sequence[int] | Sequence[Sequence[int]],
        observations: np.ndarray | Sequence[float],
        noise_variance: float | np.ndarray,
        *,
        tempering: float = 1.0,
        exact_tolerance: float = 1e-8,
    ) -> WeightUpdate:
        """Update weights from newly acquired sparse measurements, in place."""
        if self.particles.ndim == 3:
            shape = (int(self.particles.shape[1]), int(self.particles.shape[2]))
        else:
            shape = (1, int(self.particles.shape[1]))
        idx = _flat_indices(indices, shape)
        pred = self.particles.reshape(self.n_particles, -1)[:, idx]
        out = gaussian_weight_update(
            self.weights, pred, observations, noise_variance,
            tempering=tempering, exact_tolerance=exact_tolerance)
        self.weights = out.weights
        return out


@dataclass(frozen=True)
class TxModeSummary:
    """Outdoor peak-derived latent transmitter-mode posterior."""

    particle_peaks: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    probabilities: np.ndarray
    entropy: float
    normalized_entropy: float

    @property
    def n_modes(self) -> int:
        return int(self.probabilities.size)


def _outdoor_peak_coordinates(
    flat_particles: np.ndarray,
    outdoor_mask: np.ndarray,
    *,
    refine_radius: float,
) -> np.ndarray:
    """Find a stable local centroid around each particle's strongest outdoor pixel."""
    shape = outdoor_mask.shape
    outdoor_idx = np.flatnonzero(outdoor_mask.reshape(-1))
    if outdoor_idx.size == 0:
        raise ValueError("building_mask leaves no outdoor candidate")
    coords = indices_to_coordinates(outdoor_idx, shape).astype(np.float64)
    values = flat_particles[:, outdoor_idx]
    peaks = np.empty((flat_particles.shape[0], 2), dtype=np.float64)
    radius2 = max(float(refine_radius), 0.0) ** 2

    for p in range(flat_particles.shape[0]):
        local_values = values[p]
        maximum_pos = int(np.argmax(local_values))
        maximum_coord = coords[maximum_pos]
        if radius2 <= 0.0:
            peaks[p] = maximum_coord
            continue
        d2 = np.sum((coords - maximum_coord[None, :]) ** 2, axis=1)
        neighbourhood = d2 <= radius2
        nv = local_values[neighbourhood]
        nc = coords[neighbourhood]
        # Exponential local weights are invariant to an additive dB offset.  A robust
        # scale prevents either an unstable hard argmax or a nearly uniform centroid.
        spread = float(np.std(nv))
        temperature = max(spread, 1e-6)
        local_w = np.exp(np.clip((nv - float(np.max(nv))) / temperature, -50.0, 0.0))
        peaks[p] = (local_w[:, None] * nc).sum(axis=0) / local_w.sum()
    return peaks


def _reindex_modes(labels: np.ndarray, points: np.ndarray,
                   weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(labels)
    new_labels = np.empty_like(labels)
    centers = np.empty((unique.size, 2), dtype=np.float64)
    probs = np.empty(unique.size, dtype=np.float64)
    for new, old in enumerate(unique):
        member = labels == old
        mass = float(weights[member].sum())
        new_labels[member] = new
        probs[new] = mass
        centers[new] = (weights[member, None] * points[member]).sum(axis=0) / max(mass, _EPS)
    probs /= probs.sum()
    return new_labels, centers, probs


def _cluster_peak_coordinates(
    peaks: np.ndarray,
    weights: np.ndarray,
    merge_radius: float,
    max_modes: int,
    min_mode_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic weighted agglomeration of peak locations."""
    if merge_radius < 0.0:
        raise ValueError("merge_radius must be >= 0")
    if max_modes < 1:
        raise ValueError("max_modes must be >= 1")
    if not 0.0 <= min_mode_probability < 1.0:
        raise ValueError("min_mode_probability must be in [0,1)")

    # High-posterior particles establish modes first.  Stable index tie-breaking makes
    # the result reproducible across platforms.
    order = np.lexsort((np.arange(weights.size), -weights))
    labels = np.full(weights.size, -1, dtype=np.int64)
    centers: list[np.ndarray] = []
    masses: list[float] = []
    for particle in order:
        point = peaks[particle]
        if not centers:
            chosen = 0
            centers.append(point.copy())
            masses.append(0.0)
        else:
            distance = np.linalg.norm(np.asarray(centers) - point[None, :], axis=1)
            nearest = int(np.argmin(distance))
            if float(distance[nearest]) <= merge_radius:
                chosen = nearest
            else:
                chosen = len(centers)
                centers.append(point.copy())
                masses.append(0.0)
        old_mass = masses[chosen]
        new_mass = old_mass + float(weights[particle])
        centers[chosen] = (
            centers[chosen] * old_mass + point * float(weights[particle])) / new_mass
        masses[chosen] = new_mass
        labels[particle] = chosen

    labels, centers_arr, probs = _reindex_modes(labels, peaks, weights)

    # Repeatedly absorb implausibly small modes and enforce a finite mode vocabulary.
    while probs.size > 1 and (
            probs.size > max_modes or float(np.min(probs)) < min_mode_probability):
        small = int(np.argmin(probs))
        alternatives = np.flatnonzero(np.arange(probs.size) != small)
        distance = np.linalg.norm(
            centers_arr[alternatives] - centers_arr[small][None, :], axis=1)
        target = int(alternatives[int(np.argmin(distance))])
        labels[labels == small] = target
        labels, centers_arr, probs = _reindex_modes(labels, peaks, weights)

    # Merge pairs that became close after their weighted centroids moved.
    changed = True
    while changed and probs.size > 1:
        changed = False
        pair_distance = np.linalg.norm(
            centers_arr[:, None, :] - centers_arr[None, :, :], axis=2)
        np.fill_diagonal(pair_distance, np.inf)
        a, b = np.unravel_index(int(np.argmin(pair_distance)), pair_distance.shape)
        if float(pair_distance[a, b]) <= merge_radius:
            labels[labels == b] = a
            labels, centers_arr, probs = _reindex_modes(labels, peaks, weights)
            changed = True
    return labels, centers_arr, probs


def infer_tx_modes(
    particles: np.ndarray,
    weights: np.ndarray | Sequence[float] | None,
    outdoor_mask: np.ndarray,
    *,
    merge_radius: float = 5.0,
    peak_refine_radius: float = 2.0,
    max_modes: int = 8,
    min_mode_probability: float = 0.01,
) -> TxModeSummary:
    """Infer a discrete unknown-Tx posterior from particle outdoor peaks.

    The result is a latent summary of the *posterior particles*, not a use of the
    transmitter labels supplied with RadioMapSeer.  Consequently the function remains
    valid when Tx metadata is unavailable at deployment time.
    """
    outdoor = np.asarray(outdoor_mask, dtype=bool)
    if outdoor.ndim != 2:
        raise ValueError("outdoor_mask must be 2-D")
    flat = _flatten_particles(particles, outdoor.shape)
    w = _normalise_weights(weights, flat.shape[0])
    peaks = _outdoor_peak_coordinates(
        flat, outdoor, refine_radius=peak_refine_radius)
    labels, centers, probabilities = _cluster_peak_coordinates(
        peaks, w, float(merge_radius), int(max_modes),
        float(min_mode_probability))
    positive = probabilities > 0.0
    entropy = float(-np.sum(
        probabilities[positive] * np.log(probabilities[positive])))
    normalized = entropy / np.log(probabilities.size) if probabilities.size > 1 else 0.0
    return TxModeSummary(
        particle_peaks=peaks,
        labels=labels,
        centers=centers,
        probabilities=probabilities,
        entropy=entropy,
        normalized_entropy=float(np.clip(normalized, 0.0, 1.0)),
    )


# Descriptive alias used by experiment code.
outdoor_peak_tx_modes = infer_tx_modes


def _noise_vector(noise_variance: float | np.ndarray, d: int) -> np.ndarray:
    noise = np.asarray(noise_variance, dtype=np.float64)
    if noise.ndim == 0:
        out = np.full(d, float(noise), dtype=np.float64)
    else:
        out = noise.reshape(-1)
        if out.shape != (d,):
            raise ValueError(
                f"noise_variance must be scalar or have {d} entries; got {out.shape}")
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("noise_variance must be finite and non-negative")
    return out


def approximate_mode_mutual_information(
    particle_values: np.ndarray,
    weights: np.ndarray | Sequence[float],
    mode_labels: np.ndarray | Sequence[int],
    noise_variance: float | np.ndarray,
) -> np.ndarray:
    """Moment-matched approximation of ``I(Tx-mode; measurement_k)``.

    Each mode's predictive measurement distribution and the full predictive mixture
    are moment-matched to one-dimensional Gaussians.  The entropy difference is
    clipped to ``[0, H(mode)]``.  This vectorised approximation is cheap enough to
    evaluate at every outdoor pixel and explicitly rewards measurements whose values
    discriminate the competing, peak-derived transmitter hypotheses.
    """
    values = np.asarray(particle_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("particle_values must have shape (P,D)")
    if not np.all(np.isfinite(values)):
        raise ValueError("particle_values must be finite")
    w = _normalise_weights(weights, values.shape[0])
    labels = np.asarray(mode_labels, dtype=np.int64).reshape(-1)
    if labels.shape != (values.shape[0],):
        raise ValueError(
            f"mode_labels must have shape ({values.shape[0]},), got {labels.shape}")
    labels, _, mode_probs = _reindex_modes(
        labels, np.zeros((labels.size, 2), dtype=np.float64), w)
    if mode_probs.size <= 1:
        return np.zeros(values.shape[1], dtype=np.float64)

    noise = np.maximum(_noise_vector(noise_variance, values.shape[1]), _EPS)
    global_mean = w @ values
    total_var = w @ ((values - global_mean[None, :]) ** 2)
    conditional_log_var = np.zeros(values.shape[1], dtype=np.float64)
    for mode, probability in enumerate(mode_probs):
        member = labels == mode
        conditional_w = w[member] / probability
        conditional_values = values[member]
        conditional_mean = conditional_w @ conditional_values
        conditional_var = conditional_w @ (
            (conditional_values - conditional_mean[None, :]) ** 2)
        conditional_log_var += probability * np.log(
            np.maximum(conditional_var + noise, _EPS))
    estimate = 0.5 * (
        np.log(np.maximum(total_var + noise, _EPS)) - conditional_log_var)
    mode_entropy = float(-np.sum(mode_probs * np.log(np.maximum(mode_probs, _EPS))))
    return np.clip(estimate, 0.0, mode_entropy)


# Short alias for callers that already use ``MI`` terminology.
mode_mutual_information = approximate_mode_mutual_information


def global_a_optimal_gain(
    covariance_factor: np.ndarray,
    noise_variance: float | np.ndarray,
    *,
    candidate_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Outdoor-target global posterior-variance reduction for every candidate.

    For candidate ``k`` this is

    ``sum_{j in target} C[j,k]**2 / (C[k,k] + noise[k])``.

    The implementation uses an ensemble-space Gram matrix of size ``P x P`` and
    therefore never materialises ``C`` (which would be ``D x D``).
    """
    factor = np.asarray(covariance_factor, dtype=np.float64)
    if factor.ndim != 2 or factor.shape[0] < 1:
        raise ValueError("covariance_factor must have shape (P,D)")
    if not np.all(np.isfinite(factor)):
        raise ValueError("covariance_factor must be finite")
    d = factor.shape[1]
    candidates = np.ones(d, dtype=bool) if candidate_mask is None \
        else np.asarray(candidate_mask, dtype=bool).reshape(-1)
    targets = np.ones(d, dtype=bool) if target_mask is None \
        else np.asarray(target_mask, dtype=bool).reshape(-1)
    if candidates.shape != (d,) or targets.shape != (d,):
        raise ValueError("candidate_mask and target_mask must each have D entries")
    score = np.zeros(d, dtype=np.float64)
    candidate_idx = np.flatnonzero(candidates)
    target_idx = np.flatnonzero(targets)
    if candidate_idx.size == 0 or target_idx.size == 0:
        return score
    target_gram = factor[:, target_idx] @ factor[:, target_idx].T
    candidate_factor = factor[:, candidate_idx]
    numerator = np.einsum(
        "pc,pq,qc->c", candidate_factor, target_gram, candidate_factor,
        optimize=True)
    variance = np.sum(candidate_factor ** 2, axis=0)
    noise = _noise_vector(noise_variance, d)[candidate_idx]
    score[candidate_idx] = np.maximum(numerator, 0.0) / np.maximum(
        variance + noise, _EPS)
    return score


# Explicit name documenting the target domain.
outdoor_target_a_optimal_gain = global_a_optimal_gain


def rank_normalize(values: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Map finite valid values to tie-aware fractional ranks in ``[0,1]``."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.shape != x.shape:
            raise ValueError("valid_mask must match values")
        valid &= mask
    out = np.zeros_like(x)
    idx = np.flatnonzero(valid)
    if idx.size <= 1:
        return out
    order = idx[np.argsort(x[idx], kind="mergesort")]
    sorted_values = x[order]
    ranks = np.empty(order.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    if float(sorted_values[-1]) == float(sorted_values[0]):
        return out
    ranks /= order.size - 1
    out[order] = ranks
    return out


def min_distance_to_set(shape: tuple[int, int],
                        reference_indices: Iterable[int]) -> np.ndarray:
    """Per-pixel Euclidean distance to the closest reference point."""
    d = int(np.prod(shape))
    refs = np.asarray(list(reference_indices), dtype=np.int64).reshape(-1)
    if refs.size == 0:
        return np.ones(d, dtype=np.float64)
    refs = np.unique(refs)
    if np.any(refs < 0) or np.any(refs >= d):
        raise IndexError("reference index outside map")
    rr, cc = np.indices(shape)
    coords = np.column_stack((rr.reshape(-1), cc.reshape(-1))).astype(np.float64)
    ref_coords = coords[refs]
    distance2 = np.full(d, np.inf, dtype=np.float64)
    # Streaming over the typically small revealed set avoids a D x K temporary.
    for ref in ref_coords:
        distance2 = np.minimum(
            distance2, np.sum((coords - ref[None, :]) ** 2, axis=1))
    return np.sqrt(distance2)


def square_root_downdate(covariance_factor: np.ndarray, index: int,
                         noise_variance: float) -> np.ndarray:
    """Exact scalar-observation square-root covariance downdate.

    If ``C = F.T @ F``, the result ``F_new`` satisfies
    ``F_new.T @ F_new = C - C[:,k] C[k,:] / (C[k,k] + noise)``.
    """
    factor = np.asarray(covariance_factor, dtype=np.float64)
    if factor.ndim != 2:
        raise ValueError("covariance_factor must have shape (P,D)")
    k = int(index)
    if k < 0 or k >= factor.shape[1]:
        raise IndexError("downdate index outside map")
    noise = float(noise_variance)
    if not np.isfinite(noise) or noise < 0.0:
        raise ValueError("noise_variance must be finite and non-negative")
    f = factor[:, k]
    variance = float(np.dot(f, f))
    if variance <= _EPS:
        return factor.copy()
    shrink_along_f = 1.0 - np.sqrt(noise / max(variance + noise, _EPS))
    projection = (f @ factor) / variance
    return factor - shrink_along_f * f[:, None] * projection[None, :]


# Alias matching the terminology used by the batch selector.
sequential_batch_downdate = square_root_downdate


@dataclass(frozen=True)
class AcquisitionResult:
    """Selected locations and auditable per-pick acquisition diagnostics."""

    indices: np.ndarray
    coordinates: np.ndarray
    scores: np.ndarray
    a_optimal_gains: np.ndarray
    mode_information: np.ndarray
    mode_blend: np.ndarray
    ess: float
    mode_summary: TxModeSummary

    @property
    def selected(self) -> list[int]:
        return self.indices.tolist()

    def as_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "indices": self.indices.copy(),
            "coordinates": self.coordinates.copy(),
            "scores": self.scores.copy(),
            "a_optimal_gains": self.a_optimal_gains.copy(),
            "mode_information": self.mode_information.copy(),
            "mode_blend": self.mode_blend.copy(),
            "ess": self.ess,
            "mode_summary": self.mode_summary,
        }

    def __getitem__(self, key: str):
        # Lightweight dict compatibility for experiment scripts.
        return self.as_dict()[key]


@dataclass(frozen=True)
class AcquisitionConfig:
    """Hyperparameters for unknown-Tx closed-loop sampling."""

    noise_variance: float = 1e-4
    mode_merge_radius: float = 5.0
    peak_refine_radius: float = 2.0
    max_modes: int = 8
    min_mode_probability: float = 0.01
    min_mode_blend: float = 0.05
    max_mode_blend: float = 0.70
    anti_cluster_strength: float = 0.45

    def __post_init__(self) -> None:
        if not np.isfinite(self.noise_variance) or self.noise_variance < 0.0:
            raise ValueError("noise_variance must be finite and non-negative")
        if not 0.0 <= self.min_mode_blend <= self.max_mode_blend <= 1.0:
            raise ValueError("mode blend bounds must satisfy 0 <= min <= max <= 1")
        if not np.isfinite(self.anti_cluster_strength) \
                or self.anti_cluster_strength < 0.0:
            raise ValueError("anti_cluster_strength must be finite and >= 0")


class ClosedLoopAcquisition:
    """Unknown-Tx, multimodal active selector with serial batch downdates."""

    def __init__(self, config: AcquisitionConfig | None = None, **overrides) -> None:
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        self.config = config if config is not None else AcquisitionConfig(**overrides)

    def _dynamic_mode_blend(self, modes: TxModeSummary, weights: np.ndarray,
                            measured_fraction: float) -> float:
        if modes.n_modes <= 1:
            return 0.0
        cfg = self.config
        ess_fraction = effective_sample_size(weights) / weights.size
        # Mode discrimination is most valuable while several hypotheses retain mass.
        # It is tapered as the outdoor budget grows and when only very few particles
        # carry weight, where a moment-based MI estimate is less trustworthy.
        reliability = np.sqrt(np.clip(ess_fraction, 0.0, 1.0))
        budget_taper = np.sqrt(np.clip(1.0 - measured_fraction, 0.0, 1.0))
        dynamic = modes.normalized_entropy * reliability * budget_taper
        return float(cfg.min_mode_blend + (
            cfg.max_mode_blend - cfg.min_mode_blend) * dynamic)

    def select(
        self,
        particles: np.ndarray,
        building_mask: np.ndarray,
        batch_size: int,
        *,
        weights: np.ndarray | Sequence[float] | None = None,
        observed_mask: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
        noise_variance: float | np.ndarray | None = None,
    ) -> AcquisitionResult:
        """Select a batch using only currently available posterior information.

        ``building_mask=True`` means a known building pixel.  Such pixels are never
        candidates and never consume the budget.  ``target_mask`` can restrict the
        reconstruction objective further, but is intersected with outdoor, unobserved
        pixels.  No complete map or transmitter label is accepted by this API.
        """
        shape = _spatial_shape(building_mask)
        building = np.asarray(building_mask, dtype=bool)
        outdoor = ~building
        observed = _as_mask(observed_mask, shape, "observed_mask")
        observed &= outdoor
        flat = _flatten_particles(particles, shape)
        w = _normalise_weights(weights, flat.shape[0])
        requested = int(batch_size)
        if requested < 0:
            raise ValueError("batch_size must be >= 0")

        candidate_mask = (outdoor & ~observed).reshape(-1)
        requested = min(requested, int(candidate_mask.sum()))
        objective_mask = outdoor.copy() if target_mask is None \
            else _as_mask(target_mask, shape, "target_mask") & outdoor
        objective_mask &= ~observed
        target_flat = objective_mask.reshape(-1)

        modes = infer_tx_modes(
            flat, w, outdoor,
            merge_radius=self.config.mode_merge_radius,
            peak_refine_radius=self.config.peak_refine_radius,
            max_modes=self.config.max_modes,
            min_mode_probability=self.config.min_mode_probability,
        )
        _, factor = weighted_mean_and_factor(flat, w)
        variance_spec = self.config.noise_variance \
            if noise_variance is None else noise_variance
        noise = _noise_vector(variance_spec, flat.shape[1])

        selected: list[int] = []
        chosen_scores: list[float] = []
        chosen_a: list[float] = []
        chosen_mi: list[float] = []
        chosen_blend: list[float] = []
        reference = np.flatnonzero(observed.reshape(-1)).tolist()
        n_outdoor = max(int(outdoor.sum()), 1)

        for _ in range(requested):
            valid = candidate_mask.copy()
            if not np.any(valid):
                break
            a_gain = global_a_optimal_gain(
                factor, noise, candidate_mask=valid, target_mask=target_flat)
            # Recover particle deviations from the current factor.  Their common mean
            # is irrelevant to MI, so a zero mean is sufficient after each downdate.
            sqrt_w = np.sqrt(w)
            deviations = np.divide(
                factor, sqrt_w[:, None], out=np.zeros_like(factor),
                where=sqrt_w[:, None] > np.sqrt(_EPS))
            mode_info = approximate_mode_mutual_information(
                deviations, w, modes.labels, noise)
            a_rank = rank_normalize(a_gain, valid)
            mi_rank = rank_normalize(mode_info, valid)
            measured_fraction = (int(observed.sum()) + len(selected)) / n_outdoor
            blend = self._dynamic_mode_blend(modes, w, measured_fraction)
            combined = (1.0 - blend) * a_rank + blend * mi_rank

            if reference and self.config.anti_cluster_strength > 0.0:
                distance = min_distance_to_set(shape, reference)
                coverage_rank = rank_normalize(distance, valid)
                combined *= 1.0 + self.config.anti_cluster_strength * coverage_rank
                # Deterministic coverage tie-breaker when the ensemble has zero rank.
                combined += 1e-12 * coverage_rank
            combined[~valid] = -np.inf
            pick = int(np.argmax(combined))
            if not np.isfinite(combined[pick]):
                break

            selected.append(pick)
            chosen_scores.append(float(combined[pick]))
            chosen_a.append(float(a_gain[pick]))
            chosen_mi.append(float(mode_info[pick]))
            chosen_blend.append(blend)
            candidate_mask[pick] = False
            target_flat[pick] = False
            reference.append(pick)
            factor = square_root_downdate(factor, pick, float(noise[pick]))

        idx = np.asarray(selected, dtype=np.int64)
        return AcquisitionResult(
            indices=idx,
            coordinates=indices_to_coordinates(idx, shape),
            scores=np.asarray(chosen_scores, dtype=np.float64),
            a_optimal_gains=np.asarray(chosen_a, dtype=np.float64),
            mode_information=np.asarray(chosen_mi, dtype=np.float64),
            mode_blend=np.asarray(chosen_blend, dtype=np.float64),
            ess=effective_sample_size(w),
            mode_summary=modes,
        )

    def update_weights(
        self,
        particles: np.ndarray,
        weights: np.ndarray | Sequence[float] | None,
        indices: np.ndarray | Sequence[int] | Sequence[Sequence[int]],
        observations: np.ndarray | Sequence[float],
        map_shape: tuple[int, int],
        *,
        noise_variance: float | np.ndarray | None = None,
        tempering: float = 1.0,
    ) -> WeightUpdate:
        """Importance update after the selected sparse measurements are acquired."""
        idx = _flat_indices(indices, map_shape)
        x = np.asarray(particles, dtype=np.float64).reshape(
            np.asarray(particles).shape[0], -1)
        if x.shape[1] != int(np.prod(map_shape)):
            raise ValueError("particles do not match map_shape")
        variance_spec = self.config.noise_variance \
            if noise_variance is None else noise_variance
        full_noise = _noise_vector(variance_spec, x.shape[1])
        return gaussian_weight_update(
            weights, x[:, idx], observations, full_noise[idx],
            tempering=tempering)


def select_active_batch(
    particles: np.ndarray,
    building_mask: np.ndarray,
    batch_size: int,
    *,
    weights: np.ndarray | Sequence[float] | None = None,
    observed_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    config: AcquisitionConfig | None = None,
    noise_variance: float | np.ndarray | None = None,
) -> AcquisitionResult:
    """Functional interface to :class:`ClosedLoopAcquisition`."""
    return ClosedLoopAcquisition(config).select(
        particles, building_mask, batch_size, weights=weights,
        observed_mask=observed_mask, target_mask=target_mask,
        noise_variance=noise_variance)


__all__ = [
    "AcquisitionConfig",
    "AcquisitionResult",
    "ClosedLoopAcquisition",
    "ParticleBelief",
    "TxModeSummary",
    "WeightUpdate",
    "approximate_mode_mutual_information",
    "effective_sample_size",
    "gaussian_weight_update",
    "global_a_optimal_gain",
    "indices_to_coordinates",
    "infer_tx_modes",
    "min_distance_to_set",
    "mode_mutual_information",
    "outdoor_peak_tx_modes",
    "outdoor_target_a_optimal_gain",
    "rank_normalize",
    "select_active_batch",
    "sequential_batch_downdate",
    "square_root_downdate",
    "update_particle_weights",
    "weighted_mean_and_factor",
]
