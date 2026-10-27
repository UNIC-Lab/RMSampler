"""Conditional diffusion generation, robust correction, and exact projection."""
from dataclasses import dataclass
import numpy as np

from .correction import _measurement_corrected_weights


@dataclass
class Reconstruction:
    mean: np.ndarray
    weights: np.ndarray
    mode_probabilities: np.ndarray
    tx_coordinate: np.ndarray
    likelihood_power: float
    ess: float


def reconstruct_from_candidates(
    samples: np.ndarray, mode_coordinates: np.ndarray, mode_probabilities: np.ndarray,
    indices: np.ndarray, values: np.ndarray, building_mask: np.ndarray,
    building_values: np.ndarray, *, noise_std: float = .02,
    discrepancy_std: float = .1, minimum_ess_fraction: float = .35,
) -> Reconstruction:
    """Apply the original robust likelihood and projection to an actual bank.

    ``samples`` has shape (Tx modes, particles per mode, H, W). This function
    does not generate samples or infer observations from an unmeasured RM.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 4 or samples.shape[-2:] != building_mask.shape:
        raise ValueError("samples must have shape (modes, particles, H, W)")
    if not np.isfinite(samples).all():
        raise ValueError("non-finite conditional candidates")
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if indices.shape != values.shape or not len(indices) or not np.isfinite(values).all():
        raise ValueError("matching finite measurements are required")
    if np.any(indices < 0) or np.any(indices >= building_mask.size) or len(np.unique(indices)) != len(indices):
        raise ValueError("measurement indices must be distinct and inside the RM")
    if np.asarray(building_mask).reshape(-1)[indices].any():
        raise ValueError("outdoor measurements cannot be building pixels")
    if np.shape(mode_coordinates) != (samples.shape[0], 2):
        raise ValueError("one transmitter coordinate is required per mode")
    if np.shape(mode_probabilities) != (samples.shape[0],):
        raise ValueError("one prior probability is required per mode")
    weights, power, ess, mode_posterior, _ = _measurement_corrected_weights(
        samples, mode_probabilities, indices, values, noise_std=noise_std,
        discrepancy_std=discrepancy_std, minimum_ess_fraction=minimum_ess_fraction,
    )
    flat = samples.reshape(-1, *samples.shape[-2:])
    mean = np.tensordot(weights, flat, axes=(0, 0)).astype(np.float32)
    mean[building_mask] = building_values[building_mask]
    mean.reshape(-1)[indices] = values
    mean = np.clip(mean, 0, 1)
    coordinate = np.asarray(mode_coordinates)[int(np.argmax(mode_posterior))].copy()
    return Reconstruction(mean, weights, mode_posterior, coordinate, power, ess)


def reconstruct(view, calibration, indices, values, model, seed: int):
    """Generate the separate conditional ensemble using measurement-derived Tx modes."""
    import torch
    from scipy.ndimage import gaussian_filter
    from radiomap_dpm.tx_posterior import TxPosteriorState
    from radiomap_dpm.radiodiff_original_adapter import make_sparse_condition

    state = TxPosteriorState.initialize(view, calibration)
    for index, value in zip(indices, values):
        state.update(int(index), float(value), sensor_noise_std=.02)
    modes = state.modes(max_modes=4, min_separation=8.)
    coordinates = np.asarray(modes.coordinates[:4], dtype=np.int64)
    probabilities = np.asarray(modes.probabilities[:4], dtype=np.float64)
    probabilities /= probabilities.sum()
    building_values = np.zeros(view.shape, dtype=np.float32)
    building_values.reshape(-1)[view.free_indices] = view.free_values
    device = next(model.parameters()).device
    banks = []
    for mode_id, coordinate in enumerate(coordinates):
        raster = np.zeros(view.shape, dtype=np.float32)
        raster[tuple(coordinate)] = 1.
        raster = gaussian_filter(raster, sigma=1., mode="nearest")
        raster /= raster.max()
        condition, _, _ = make_sparse_condition(
            view.building_mask, indices, values, building_values=building_values, third_channel=raster,
        )
        devices = [device.index] if device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            mode_seed = (int(seed) + 104729 * mode_id) % 2**32
            torch.manual_seed(mode_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(mode_seed)
            batch = condition.repeat_interleave(4, dim=0).to(device=device, dtype=torch.float32)
            sample = model.sample(batch_size=4, cond=batch, denoise=True).clamp(0, 1)
            banks.append(sample[:, 0].detach().cpu().numpy())
    samples = np.stack(banks).astype(np.float32)
    result = reconstruct_from_candidates(
        samples, coordinates, probabilities, indices, values, view.building_mask, building_values,
    )
    return result, samples, coordinates, probabilities
