"""Scale-free joint radio-map and source-localisation Bayes-risk acquisition.

The implementation uses a factorised moment-matched posterior.  Diffusion
particles represent the broad radio field, while propagation hypotheses represent
the source-dependent field.  A candidate is scored by the normalised one-step
trace-risk reduction for both the complete field and the source coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_EPS = np.finfo(np.float64).eps


def _normalise_weights(
    weights: np.ndarray | Sequence[float], expected: int, name: str
) -> np.ndarray:
    value = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
    if value.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},)")
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    mass = float(value.sum())
    if mass <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return value / mass


def _weighted_anomalies(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = weights @ values
    return np.sqrt(weights)[:, None] * (values - mean[None, :])


def _quadratic_columns(left: np.ndarray, gram: np.ndarray) -> np.ndarray:
    return np.einsum("ic,ij,jc->c", left, gram, left, optimize=True)


def _exact_coordinate_risk_reduction(
    coordinates: np.ndarray,
    weights: np.ndarray,
    predictions: np.ndarray,
    observation_variance: np.ndarray,
    *,
    degrees_of_freedom: float = 4.0,
    chunk_size: int = 64,
    decision: str = "map",
) -> tuple[np.ndarray, float]:
    """Finite-hypothesis Bayes risk for MAP or posterior-mean coordinates."""

    if decision not in {"map", "posterior_mean"}:
        raise ValueError("decision must be 'map' or 'posterior_mean'")

    squared_norm = np.sum(np.square(coordinates), axis=1)
    distances = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * coordinates @ coordinates.T
    )
    distances = np.maximum(0.5 * (distances + distances.T), 0.0)
    np.fill_diagonal(distances, 0.0)
    if decision == "map":
        prior_decision = int(np.argmax(weights))
        prior_risk = max(float(weights @ distances[:, prior_decision]), 0.0)
    else:
        prior_mean = weights @ coordinates
        prior_risk = max(
            float(
                np.sum(
                    weights
                    * np.sum((coordinates - prior_mean[None, :]) ** 2, axis=1)
                )
            ),
            0.0,
        )
    reduction = np.zeros(predictions.shape[1], dtype=np.float64)
    if prior_risk <= 256.0 * _EPS:
        return reduction, prior_risk

    log_prior = np.full(weights.size, -np.inf, dtype=np.float64)
    positive = weights > 0.0
    log_prior[positive] = np.log(weights[positive])
    dof = float(degrees_of_freedom)
    if not np.isfinite(dof) or dof <= 2.0:
        raise ValueError("degrees_of_freedom must exceed two")
    for start in range(0, predictions.shape[1], int(chunk_size)):
        stop = min(start + int(chunk_size), predictions.shape[1])
        values = predictions[:, start:stop].T
        residual = values[:, :, None] - values[:, None, :]
        log_posterior = -0.5 * (dof + 1.0) * np.log1p(
            np.square(residual)
            / (dof * observation_variance[start:stop, None, None])
        )
        log_posterior += log_prior[None, None, :]
        log_posterior -= np.max(log_posterior, axis=2, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= np.sum(posterior, axis=2, keepdims=True)
        if decision == "map":
            decision_indices = np.argmax(posterior, axis=2)
            decision_loss = posterior @ distances
            fantasy_risk = np.take_along_axis(
                decision_loss, decision_indices[:, :, None], axis=2
            ).reshape(values.shape[0], values.shape[1])
        else:
            posterior_mean = posterior @ coordinates
            squared_error = np.sum(
                (coordinates[None, None, :, :] - posterior_mean[:, :, None, :]) ** 2,
                axis=3,
            )
            fantasy_risk = np.sum(posterior * squared_error, axis=2)
        expected_risk = fantasy_risk @ weights
        reduction[start:stop] = np.maximum(
            (prior_risk - expected_risk) / prior_risk, 0.0
        )
    return reduction, prior_risk


@dataclass(frozen=True)
class JointRiskResult:
    """Candidate-wise joint trace-risk reductions and their components."""

    scores: np.ndarray
    field_risk_reduction: np.ndarray
    coordinate_risk_reduction: np.ndarray
    predictive_variance: np.ndarray
    prior_field_risk: float
    prior_coordinate_risk: float
    candidate_indices: np.ndarray
    target_indices: np.ndarray


def joint_normalised_trace_reduction(
    field_particles: np.ndarray,
    field_weights: np.ndarray | Sequence[float],
    source_coordinates: np.ndarray,
    source_weights: np.ndarray | Sequence[float],
    source_candidate_predictions: np.ndarray,
    candidate_indices: np.ndarray | Sequence[int],
    *,
    target_indices: np.ndarray | Sequence[int] | None = None,
    source_target_predictions: np.ndarray | None = None,
    observation_variance: float | np.ndarray = 1e-4,
    source_degrees_of_freedom: float = 4.0,
    coordinate_decision: str = "map",
) -> JointRiskResult:
    """Return normalised joint A-optimal risk reduction for each candidate.

    Under the factorised Gaussian moment surrogate, the radio field is
    ``x = b + s`` and a paid reading is ``y_k = b_k + s_k + noise``.  The broad
    field ``b`` is represented by diffusion particles and the source field ``s``
    by propagation hypotheses.  The returned score is

    ``Delta tr(Cov[x]) / tr(Cov[x]) + Delta R_source / R_source``.

    ``coordinate_decision`` selects either the exact finite-hypothesis MAP risk or
    the squared-Euclidean risk of the posterior-mean coordinate.  The expression is
    evaluated in particle space, so no pixel-by-pixel covariance matrix is formed.
    Scaling the field values and observation standard deviation by the same factor,
    or changing the coordinate unit, leaves the score unchanged.
    """

    field = np.asarray(field_particles, dtype=np.float64)
    if field.ndim == 3:
        field = field.reshape(field.shape[0], -1)
    if field.ndim != 2 or min(field.shape) < 1 or not np.all(np.isfinite(field)):
        raise ValueError("field_particles must be a finite (P,D) or (P,H,W) array")
    n_field, n_pixels = field.shape
    field_mass = _normalise_weights(field_weights, n_field, "field_weights")

    coordinates = np.asarray(source_coordinates, dtype=np.float64).reshape(-1, 2)
    if coordinates.shape[0] < 1 or not np.all(np.isfinite(coordinates)):
        raise ValueError("source_coordinates must be a finite non-empty (S,2) array")
    n_source = coordinates.shape[0]
    source_mass = _normalise_weights(source_weights, n_source, "source_weights")

    candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if candidates.size < 1 or np.any(candidates < 0) or np.any(candidates >= n_pixels):
        raise ValueError("candidate_indices must be non-empty and lie inside the field")
    if np.unique(candidates).size != candidates.size:
        raise ValueError("candidate_indices must be unique")
    source_candidate = np.asarray(source_candidate_predictions, dtype=np.float64)
    if source_candidate.shape != (n_source, candidates.size):
        raise ValueError(
            "source_candidate_predictions must have shape "
            f"({n_source},{candidates.size})"
        )
    if not np.all(np.isfinite(source_candidate)):
        raise ValueError("source_candidate_predictions must be finite")

    targets = candidates if target_indices is None else np.asarray(
        target_indices, dtype=np.int64
    ).reshape(-1)
    if targets.size < 1 or np.any(targets < 0) or np.any(targets >= n_pixels):
        raise ValueError("target_indices must be non-empty and lie inside the field")
    if source_target_predictions is None:
        if not np.array_equal(targets, candidates):
            raise ValueError(
                "source_target_predictions are required when targets differ from candidates"
            )
        source_target = source_candidate
    else:
        source_target = np.asarray(source_target_predictions, dtype=np.float64)
        if source_target.shape != (n_source, targets.size):
            raise ValueError(
                "source_target_predictions must have shape "
                f"({n_source},{targets.size})"
            )
        if not np.all(np.isfinite(source_target)):
            raise ValueError("source_target_predictions must be finite")

    variance = np.asarray(observation_variance, dtype=np.float64)
    if variance.ndim == 0:
        variance = np.full(candidates.size, float(variance), dtype=np.float64)
    else:
        variance = variance.reshape(-1)
        if variance.shape != (candidates.size,):
            raise ValueError("observation_variance must be scalar or candidate-wise")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("observation_variance must be finite and non-negative")

    field_target = _weighted_anomalies(field[:, targets], field_mass)
    field_candidate = _weighted_anomalies(field[:, candidates], field_mass)
    source_target = _weighted_anomalies(source_target, source_mass)
    source_candidate = _weighted_anomalies(source_candidate, source_mass)

    field_gram = field_target @ field_target.T
    source_gram = source_target @ source_target.T
    cross_gram = field_target @ source_target.T

    covariance_norm = _quadratic_columns(field_candidate, field_gram)
    covariance_norm += _quadratic_columns(source_candidate, source_gram)
    covariance_norm += 2.0 * np.einsum(
        "ic,ij,jc->c", field_candidate, cross_gram, source_candidate, optimize=True
    )
    covariance_norm = np.maximum(covariance_norm, 0.0)

    predictive = (
        np.sum(np.square(field_candidate), axis=0)
        + np.sum(np.square(source_candidate), axis=0)
        + variance
    )
    predictive = np.maximum(predictive, 256.0 * _EPS)
    field_risk = float(
        np.sum(np.square(field_target)) + np.sum(np.square(source_target))
    )
    coordinate_reduction, coordinate_risk = _exact_coordinate_risk_reduction(
        coordinates,
        source_mass,
        np.asarray(source_candidate_predictions, dtype=np.float64),
        np.maximum(variance, 256.0 * _EPS),
        degrees_of_freedom=source_degrees_of_freedom,
        decision=coordinate_decision,
    )

    field_reduction = np.zeros(candidates.size, dtype=np.float64)
    if field_risk > 256.0 * _EPS:
        field_reduction = covariance_norm / (predictive * field_risk)
    field_reduction = np.maximum(field_reduction, 0.0)
    coordinate_reduction = np.maximum(coordinate_reduction, 0.0)
    scores = field_reduction + coordinate_reduction
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("joint risk produced a non-finite score")

    return JointRiskResult(
        scores=scores,
        field_risk_reduction=field_reduction,
        coordinate_risk_reduction=coordinate_reduction,
        predictive_variance=predictive,
        prior_field_risk=field_risk,
        prior_coordinate_risk=coordinate_risk,
        candidate_indices=candidates.copy(),
        target_indices=targets.copy(),
    )


__all__ = ["JointRiskResult", "joint_normalised_trace_reduction"]
