"""Sequential controller around the paper's existing acquisition functions.

Only ``observe(index)`` can obtain an outdoor measurement. Neither the RM
reference nor the true transmitter location is passed to the selector.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from radiomap_dpm.acquisition import AcquisitionConfig, ClosedLoopAcquisition
from radiomap_dpm.baselines import validate_outdoor_selection
from radiomap_dpm.data import AlgorithmView
from radiomap_dpm.source_model import SourceCalibration
from radiomap_dpm.tx_posterior import TxPosteriorState
from .regularization import regularize_weights_minimum_ess


@dataclass(frozen=True)
class SamplingConfig:
    budgets: tuple[int, ...] = (8, 16)
    noise_std: float = 0.02
    model_discrepancy_std: float = 0.1
    likelihood_temperature: float = 4.0
    minimum_particle_ess_fraction: float = 0.35
    joint_source_hypotheses: int = 96
    joint_target_stride: int = 4

    @property
    def effective_observation_variance(self) -> float:
        return self.noise_std**2 + self.model_discrepancy_std**2


@dataclass
class SamplingTrace:
    indices: np.ndarray
    values: np.ndarray
    scores: np.ndarray
    field_reduction: np.ndarray
    coordinate_reduction: np.ndarray
    field_weights: np.ndarray
    entropy: np.ndarray


def select_locations(
    view: AlgorithmView,
    field_particles: np.ndarray,
    calibration: SourceCalibration,
    observe: Callable[[int], float],
    config: SamplingConfig = SamplingConfig(),
    *,
    initial_weights: np.ndarray | None = None,
) -> SamplingTrace:
    """Run the same one-point update sequence used by the RMSampler experiment."""
    budgets = tuple(int(x) for x in config.budgets)
    if not budgets or min(budgets) < 1 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budgets must be positive, distinct, and increasing")
    if config.effective_observation_variance <= 0 or config.likelihood_temperature <= 0:
        raise ValueError("positive variance and likelihood temperature are required")
    particles = np.asarray(field_particles, dtype=np.float64)
    if particles.ndim == 3:
        particles = particles.reshape(particles.shape[0], -1)
    if particles.ndim != 2 or particles.shape[1] != np.prod(view.shape):
        raise ValueError("field particles must have shape (M, H*W) or (M, H, W)")
    if not np.isfinite(particles).all():
        raise ValueError("field particles contain non-finite values")
    weights = (np.ones(len(particles)) if initial_weights is None
               else np.asarray(initial_weights, dtype=np.float64).copy())
    if weights.shape != (len(particles),) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("invalid initial particle weights")
    weights /= weights.sum()
    state = TxPosteriorState.initialize(view, calibration)
    updater = ClosedLoopAcquisition(AcquisitionConfig(noise_variance=config.effective_observation_variance))
    observation_variance = config.effective_observation_variance + calibration.residual_scale**2
    selected, values, scores, field_gains, coordinate_gains = [], [], [], [], []
    entropy = [state.entropy]
    for _ in range(max(budgets)):
        candidates, risk, _ = state.joint_risk_acquisition_scores(
            particles, weights, observed_indices=selected,
            observation_variance=observation_variance,
            max_source_hypotheses=config.joint_source_hypotheses,
            target_stride=config.joint_target_stride, coordinate_decision="map",
        )
        pick = int(np.argmax(risk.scores))
        index = int(np.ravel_multi_index(tuple(candidates[pick]), view.shape))
        assert index == int(risk.candidate_indices[pick])
        validate_outdoor_selection(np.asarray(selected + [index]), view.building_mask)
        value = float(np.float32(observe(index)))
        if not np.isfinite(value):
            raise ValueError("measurement must be finite")
        update = updater.update_weights(
            particles, weights, np.asarray([index]), np.asarray([value], dtype=np.float32),
            view.shape, noise_variance=config.effective_observation_variance,
            tempering=1.0 / config.likelihood_temperature,
        )
        weights, _, _ = regularize_weights_minimum_ess(update.weights, config.minimum_particle_ess_fraction)
        state.update(index, value, sensor_noise_std=config.noise_std)
        selected.append(index)
        values.append(value)
        scores.append(risk.scores[pick])
        field_gains.append(risk.field_risk_reduction[pick])
        coordinate_gains.append(risk.coordinate_risk_reduction[pick])
        entropy.append(state.entropy)
    return SamplingTrace(
        np.asarray(selected, dtype=np.int64), np.asarray(values, dtype=np.float32),
        np.asarray(scores), np.asarray(field_gains), np.asarray(coordinate_gains),
        weights, np.asarray(entropy),
    )
