"""Finite-bank ESS safeguard used by RMSampler sampling."""
import numpy as np

def regularize_weights_minimum_ess(
    weights: np.ndarray,
    minimum_ess_fraction: float,
) -> tuple[np.ndarray, float, bool]:
    """Robustly retain a requested fraction of a finite particle population.

    The likelihood posterior is mixed with the *uniform environment-conditioned
    prior bank* by the smallest amount that reaches the requested ESS.  This is a
    contamination-robust posterior safeguard: a finite, imperfect diffusion bank
    must not erase every alternative unknown-Tx hypothesis after a few noisy samples.
    It is deterministic, uses no hidden radio-map values, and returns the raw ESS for
    audit.  A requested fraction of zero is an exact no-op.
    """

    raw = np.asarray(weights, dtype=np.float64).reshape(-1)
    if raw.size < 1 or np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("weights must be a finite non-negative non-empty vector")
    mass = float(raw.sum())
    if mass <= 0.0:
        raise ValueError("weights must have positive mass")
    normalized = raw / mass
    fraction = float(minimum_ess_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("minimum_ess_fraction must lie in [0, 1]")
    raw_ess = float(1.0 / np.sum(np.square(normalized)))
    target = fraction * normalized.size
    if target <= raw_ess + 1e-12 or normalized.size == 1:
        return normalized, raw_ess, False

    uniform = np.full(normalized.size, 1.0 / normalized.size, dtype=np.float64)

    def ess_at(mix: float) -> float:
        value = (1.0 - mix) * normalized + mix * uniform
        return float(1.0 / np.sum(np.square(value)))

    lower, upper = 0.0, 1.0
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if ess_at(middle) < target:
            lower = middle
        else:
            upper = middle
    mixed = (1.0 - upper) * normalized + upper * uniform
    mixed /= mixed.sum()
    return mixed, raw_ess, True
