"""Outdoor-only sampling baselines for RadioMapSeer DPM experiments.

Every public selector in this module depends only on map geometry (and, for the
one-shot active baseline, unconditional posterior particles).  None accepts a complete
held-out radio map or transmitter metadata.  Returned indices are row-major flat pixel
indices, contain no duplicates, and never fall inside a known building.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .acquisition import (
    AcquisitionConfig,
    AcquisitionResult,
    ClosedLoopAcquisition,
    indices_to_coordinates,
)


def _validate_geometry(building_mask: np.ndarray,
                       observed_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    building = np.asarray(building_mask, dtype=bool)
    if building.ndim != 2 or building.size == 0:
        raise ValueError("building_mask must be a non-empty 2-D array")
    if observed_mask is None:
        observed = np.zeros_like(building)
    else:
        observed = np.asarray(observed_mask, dtype=bool)
        if observed.shape != building.shape:
            raise ValueError(
                f"observed_mask must have shape {building.shape}, got {observed.shape}")
        observed = observed.copy()
    # Noise-free building observations are free side information, not sampled points.
    observed &= ~building
    return building, observed


def _requested_count(n_samples: int, available: np.ndarray) -> int:
    count = int(n_samples)
    if count < 0:
        raise ValueError("n_samples must be >= 0")
    return min(count, int(np.count_nonzero(available)))


def _format(indices: np.ndarray, shape: tuple[int, int],
            return_coordinates: bool) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return indices_to_coordinates(idx, shape) if return_coordinates else idx


def validate_outdoor_selection(indices: np.ndarray | Sequence[int],
                               building_mask: np.ndarray,
                               observed_mask: np.ndarray | None = None) -> None:
    """Raise if a design contains duplicates, buildings, or prior observations."""
    building, observed = _validate_geometry(building_mask, observed_mask)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    d = building.size
    if np.any(idx < 0) or np.any(idx >= d):
        raise ValueError("selection contains an out-of-range index")
    if np.unique(idx).size != idx.size:
        raise ValueError("selection contains duplicate points")
    if np.any(building.reshape(-1)[idx]):
        raise ValueError("selection contains a building point")
    if np.any(observed.reshape(-1)[idx]):
        raise ValueError("selection repeats an already observed point")


def random_sampling(
    building_mask: np.ndarray,
    n_samples: int,
    *,
    observed_mask: np.ndarray | None = None,
    seed: int | None = 0,
    rng: np.random.Generator | None = None,
    return_coordinates: bool = False,
) -> np.ndarray:
    """Uniform sampling without replacement over unobserved outdoor pixels."""
    building, observed = _validate_geometry(building_mask, observed_mask)
    available = (~building & ~observed).reshape(-1)
    count = _requested_count(n_samples, available)
    if rng is not None and seed not in (None, 0):
        raise ValueError("pass either rng or a non-default seed, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)
    candidates = np.flatnonzero(available)
    if count:
        selected = np.asarray(
            generator.choice(candidates, size=count, replace=False), dtype=np.int64)
    else:
        selected = np.empty(0, dtype=np.int64)
    validate_outdoor_selection(selected, building, observed)
    return _format(selected, building.shape, return_coordinates)


def _grid_dimensions(n_points: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Choose a near-square lattice with at least ``n_points`` cells."""
    if n_points <= 0:
        return 0, 0
    height, width = shape
    best: tuple[float, int, int] | None = None
    for n_rows in range(1, min(height, n_points) + 1):
        n_cols = int(np.ceil(n_points / n_rows))
        if n_cols > width:
            continue
        row_spacing = height / n_rows
        col_spacing = width / n_cols
        aspect_penalty = abs(float(np.log(row_spacing / col_spacing)))
        excess_penalty = 0.25 * (n_rows * n_cols - n_points) / n_points
        candidate = (aspect_penalty + excess_penalty, n_rows, n_cols)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        # Only possible for degenerate shapes; a one-dimensional lattice still works.
        return min(height, n_points), min(width, int(np.ceil(n_points / height)))
    return best[1], best[2]


def _farthest_order(points: np.ndarray) -> np.ndarray:
    """Order lattice targets so every prefix is spatially well spread."""
    n = points.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    first = int(np.argmin(np.sum((points - centre[None, :]) ** 2, axis=1)))
    order = np.empty(n, dtype=np.int64)
    order[0] = first
    used = np.zeros(n, dtype=bool)
    used[first] = True
    min_distance2 = np.sum((points - points[first][None, :]) ** 2, axis=1)
    for position in range(1, n):
        scores = np.where(used, -np.inf, min_distance2)
        pick = int(np.argmax(scores))
        order[position] = pick
        used[pick] = True
        min_distance2 = np.minimum(
            min_distance2,
            np.sum((points - points[pick][None, :]) ** 2, axis=1))
    return order


def _regular_targets(n_points: int, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = _grid_dimensions(n_points, shape)
    if rows == 0 or cols == 0:
        return np.empty((0, 2), dtype=np.float64)
    rr = (np.arange(rows, dtype=np.float64) + 0.5) * shape[0] / rows - 0.5
    cc = (np.arange(cols, dtype=np.float64) + 0.5) * shape[1] / cols - 0.5
    grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
    targets = np.column_stack((grid_r.reshape(-1), grid_c.reshape(-1)))
    return targets[_farthest_order(targets)]


def _nearest_unique(targets: np.ndarray, candidate_indices: np.ndarray,
                    shape: tuple[int, int], count: int) -> np.ndarray:
    if count <= 0 or candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    candidate_coords = indices_to_coordinates(
        candidate_indices, shape).astype(np.float64)
    unused = np.ones(candidate_indices.size, dtype=bool)
    selected: list[int] = []
    for target in targets:
        if len(selected) >= count or not np.any(unused):
            break
        distance2 = np.sum((candidate_coords - target[None, :]) ** 2, axis=1)
        distance2[~unused] = np.inf
        local = int(np.argmin(distance2))
        if not np.isfinite(distance2[local]):
            break
        selected.append(int(candidate_indices[local]))
        unused[local] = False
    return np.asarray(selected, dtype=np.int64)


def regular_grid_sampling(
    building_mask: np.ndarray,
    n_samples: int,
    *,
    observed_mask: np.ndarray | None = None,
    return_coordinates: bool = False,
) -> np.ndarray:
    """Deterministic regular lattice snapped to unique outdoor pixels.

    On irregular maps, a nominal lattice point that falls in a building is moved to
    its closest still-unused outdoor pixel.  If prior samples are supplied, a lattice
    for the cumulative outdoor budget is formed and only new points are returned.
    """
    building, observed = _validate_geometry(building_mask, observed_mask)
    available = (~building & ~observed).reshape(-1)
    count = _requested_count(n_samples, available)
    if count == 0:
        return _format(np.empty(0, dtype=np.int64), building.shape,
                       return_coordinates)

    cumulative = count + int(np.count_nonzero(observed))
    # Use a few extra targets because prior observations and obstacle snapping can make
    # a nominal cumulative lattice non-nested across rounds.
    target_count = min(building.size, max(cumulative, count) + max(4, count // 4))
    targets = _regular_targets(target_count, building.shape)
    selected = _nearest_unique(
        targets, np.flatnonzero(available), building.shape, count)

    if selected.size < count:
        augmented_observed = observed.copy().reshape(-1)
        augmented_observed[selected] = True
        fill = farthest_point_sampling(
            building, count - selected.size,
            observed_mask=augmented_observed.reshape(building.shape))
        selected = np.concatenate((selected, fill))
    validate_outdoor_selection(selected, building, observed)
    return _format(selected, building.shape, return_coordinates)


def farthest_point_sampling(
    building_mask: np.ndarray,
    n_samples: int,
    *,
    observed_mask: np.ndarray | None = None,
    return_coordinates: bool = False,
) -> np.ndarray:
    """Greedy max-min (FPS) design over unobserved outdoor pixels.

    Existing outdoor observations seed the distance field.  With no observations, the
    outdoor pixel closest to the geometric map centre is the deterministic first point.
    """
    building, observed = _validate_geometry(building_mask, observed_mask)
    available = (~building & ~observed).reshape(-1)
    count = _requested_count(n_samples, available)
    candidate_indices = np.flatnonzero(available)
    if count == 0:
        return _format(np.empty(0, dtype=np.int64), building.shape,
                       return_coordinates)

    candidate_coords = indices_to_coordinates(
        candidate_indices, building.shape).astype(np.float64)
    reference_indices = np.flatnonzero(observed.reshape(-1))
    min_distance2 = np.full(candidate_indices.size, np.inf, dtype=np.float64)
    if reference_indices.size:
        reference_coords = indices_to_coordinates(
            reference_indices, building.shape).astype(np.float64)
        for ref in reference_coords:
            min_distance2 = np.minimum(
                min_distance2,
                np.sum((candidate_coords - ref[None, :]) ** 2, axis=1))
    else:
        centre = 0.5 * (np.asarray(building.shape, dtype=np.float64) - 1.0)
        first_local = int(np.argmin(
            np.sum((candidate_coords - centre[None, :]) ** 2, axis=1)))
        # Inf scores make the requested centre the first deterministic maximiser.
        min_distance2[:] = -np.inf
        min_distance2[first_local] = np.inf

    used = np.zeros(candidate_indices.size, dtype=bool)
    selected = np.empty(count, dtype=np.int64)
    for position in range(count):
        scores = np.where(used, -np.inf, min_distance2)
        local = int(np.argmax(scores))
        selected[position] = candidate_indices[local]
        used[local] = True
        distance2 = np.sum(
            (candidate_coords - candidate_coords[local][None, :]) ** 2, axis=1)
        if position == 0 and reference_indices.size == 0:
            min_distance2 = distance2
        else:
            min_distance2 = np.minimum(min_distance2, distance2)

    validate_outdoor_selection(selected, building, observed)
    return _format(selected, building.shape, return_coordinates)


def one_shot_active_sampling(
    particles: np.ndarray,
    building_mask: np.ndarray,
    n_samples: int,
    *,
    weights: np.ndarray | Sequence[float] | None = None,
    observed_mask: np.ndarray | None = None,
    config: AcquisitionConfig | None = None,
    noise_variance: float | np.ndarray | None = None,
) -> AcquisitionResult:
    """Non-adaptive active baseline: choose the whole budget from one posterior.

    The batch still uses serial covariance downdates to prevent duplicate information,
    but no measured values are revealed and no particle weights are updated between
    picks.  This cleanly separates one-shot design from the closed-loop method.
    """
    result = ClosedLoopAcquisition(config).select(
        particles, building_mask, n_samples, weights=weights,
        observed_mask=observed_mask, noise_variance=noise_variance)
    validate_outdoor_selection(result.indices, building_mask, observed_mask)
    return result


# Concise aliases used in experiment configuration files.
random_select = random_sampling
regular_grid_select = regular_grid_sampling
fps_sampling = farthest_point_sampling
fps_select = farthest_point_sampling
one_shot_active_select = one_shot_active_sampling


__all__ = [
    "farthest_point_sampling",
    "fps_sampling",
    "fps_select",
    "one_shot_active_sampling",
    "one_shot_active_select",
    "random_sampling",
    "random_select",
    "regular_grid_sampling",
    "regular_grid_select",
    "validate_outdoor_selection",
]
