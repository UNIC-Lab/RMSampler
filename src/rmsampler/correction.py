"""Robust correction of the joint conditional RM and Tx ensemble."""
import numpy as np

def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(1.0 / np.sum(np.square(weights)))

def _normalised_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    values = values - float(np.max(values))
    weights = np.exp(values)
    mass = float(weights.sum())
    if not np.isfinite(mass) or mass <= 0.0:
        raise FloatingPointError("joint posterior weights are not normalisable")
    return weights / mass

def _measurement_corrected_weights(
    sample_array: np.ndarray,
    mode_probabilities: np.ndarray,
    measured_indices: np.ndarray,
    measurements: np.ndarray,
    *,
    noise_std: float,
    discrepancy_std: float,
    minimum_ess_fraction: float,
    degrees_of_freedom: float = 4.0,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Return a robust generalized-Bayes posterior over joint particles."""

    n_modes, samples_per_mode, height, width = sample_array.shape
    particles = sample_array.reshape(n_modes * samples_per_mode, height * width)
    indices = np.asarray(measured_indices, dtype=np.int64).reshape(-1)
    observed = np.asarray(measurements, dtype=np.float64).reshape(1, -1)
    predictions = particles[:, indices].astype(np.float64, copy=False)
    residual = predictions - observed

    variance = max(
        float(noise_std) ** 2 + float(discrepancy_std) ** 2,
        np.finfo(np.float64).eps,
    )
    dof = float(degrees_of_freedom)
    if not np.isfinite(dof) or dof <= 2.0:
        raise ValueError("degrees_of_freedom must exceed two")
    robust_energy = 0.5 * (dof + 1.0) * np.sum(
        np.log1p(np.square(residual) / (dof * variance)), axis=1
    )

    prior_modes = np.asarray(mode_probabilities, dtype=np.float64).reshape(-1)
    prior_modes = np.maximum(prior_modes, np.finfo(np.float64).tiny)
    prior_modes /= prior_modes.sum()
    log_prior = np.repeat(
        np.log(prior_modes) - np.log(float(samples_per_mode)), samples_per_mode
    )

    n_particles = n_modes * samples_per_mode
    target_ess = min(
        float(n_particles),
        max(2.0, float(minimum_ess_fraction) * float(n_particles)),
    )

    def weights_at(temperature: float) -> np.ndarray:
        return _normalised_weights(log_prior - float(temperature) * robust_energy)

    temperature = 1.0
    weights = weights_at(temperature)
    if _effective_sample_size(weights) < target_ess:
        low, high = 0.0, 1.0
        for _ in range(40):
            middle = 0.5 * (low + high)
            candidate = weights_at(middle)
            if _effective_sample_size(candidate) >= target_ess:
                low = middle
            else:
                high = middle
        temperature = low
        weights = weights_at(temperature)

    mode_posterior = weights.reshape(n_modes, samples_per_mode).sum(axis=1)
    particle_rmse = np.sqrt(np.mean(np.square(residual), axis=1))
    return (
        weights,
        float(temperature),
        _effective_sample_size(weights),
        mode_posterior,
        particle_rmse,
    )
