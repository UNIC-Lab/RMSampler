"""Radio-map metrics and the pre-registered 3A statistical acceptance gate.

The primary endpoint is outdoor-only NMSE (dB); outdoor RMSE/PSNR/SSIM are reported
alongside it.  Full-map values are secondary because the known building pixels are free,
noise-free observations.  All resampling uses fixed legacy NumPy seeds for exact replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PRIMARY_BUDGETS = (16, 32, 64)
BOOTSTRAP_SEED = 42031
BOOTSTRAP_RESAMPLES = 10_000
GATE_MIN_ADVANTAGE_DB = 2.0
GATE_MIN_WIN_RATE = 0.80
GATE_REQUIRED_BUDGETS = 2
_EPS = 1e-12


def _as_2d_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must reduce to a 2-D map, got shape {array.shape}")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _ssim_map(target: np.ndarray, prediction: np.ndarray, data_range: float) -> np.ndarray:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover - dependency error is intentionally explicit
        raise ImportError(
            "SSIM requires scikit-image in the py311 environment; no fallback is substituted"
        ) from exc
    min_side = min(target.shape)
    if min_side < 3:
        raise ValueError("SSIM requires both spatial dimensions to be at least 3")
    win_size = min(7, min_side if min_side % 2 else min_side - 1)
    _, quality_map = structural_similarity(
        target,
        prediction,
        data_range=float(data_range),
        win_size=win_size,
        full=True,
    )
    return np.asarray(quality_map, dtype=np.float64)


def _region_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    region: np.ndarray,
    quality_map: np.ndarray,
    data_range: float,
) -> dict[str, float | int]:
    n_pixels = int(region.sum())
    if n_pixels == 0:
        raise ValueError("metric region is empty")
    error = prediction[region] - target[region]
    squared_error = error * error
    mse = float(np.mean(squared_error))
    target_energy = float(np.sum(target[region] ** 2))
    nmse = float(np.sum(squared_error) / max(target_energy, _EPS))
    nmse_db = float(10.0 * np.log10(max(nmse, _EPS)))
    rmse = float(np.sqrt(mse))
    psnr_db = float(10.0 * np.log10((float(data_range) ** 2) / max(mse, _EPS)))
    ssim = float(np.mean(quality_map[region]))
    return {
        "nmse_db": nmse_db,
        "nmse": nmse,
        "rmse": rmse,
        "psnr_db": psnr_db,
        "psnr": psnr_db,
        "ssim": ssim,
        "n_pixels": n_pixels,
    }


def compute_metrics(
    prediction: Any,
    target: Any,
    building_mask: Any,
    *,
    data_range: float = 1.0,
    clip_prediction: bool = True,
) -> dict[str, Any]:
    """Compute primary outdoor and secondary full-map reconstruction metrics.

    Outdoor SSIM is the mean of the standard local SSIM map at outdoor pixels.  Thus it
    remains a spatial/structural metric instead of incorrectly treating disjoint outdoor
    pixels as a one-dimensional signal.
    """

    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError(f"data_range must be positive and finite, got {data_range}")
    pred = _as_2d_numpy(prediction, "prediction")
    truth = _as_2d_numpy(target, "target")
    mask_raw = _as_2d_numpy(building_mask, "building_mask")
    if pred.shape != truth.shape or mask_raw.shape != truth.shape:
        raise ValueError(
            f"prediction/target/building shapes differ: {pred.shape}, {truth.shape}, {mask_raw.shape}"
        )
    building = mask_raw.astype(bool)
    outdoor = ~building
    if not np.any(outdoor):
        raise ValueError("building mask leaves no outdoor evaluation pixels")
    if clip_prediction:
        pred = np.clip(pred, 0.0, float(data_range))
    quality_map = _ssim_map(truth, pred, float(data_range))
    outdoor_metrics = _region_metrics(pred, truth, outdoor, quality_map, float(data_range))
    full_metrics = _region_metrics(
        pred, truth, np.ones(truth.shape, dtype=bool), quality_map, float(data_range)
    )
    result: dict[str, Any] = {
        "primary": outdoor_metrics,
        "outdoor": outdoor_metrics,
        "secondary": full_metrics,
        "full": full_metrics,
    }
    # Flat names make serialized tables and experiment aggregation unambiguous.
    for region_name, values in (("outdoor", outdoor_metrics), ("full", full_metrics)):
        for metric_name, value in values.items():
            result[f"{region_name}_{metric_name}"] = value
    # Primary-endpoint compatibility shortcuts always mean outdoor-only.
    for metric_name in ("nmse_db", "nmse", "rmse", "psnr_db", "psnr", "ssim"):
        result[metric_name] = outdoor_metrics[metric_name]
    return result


evaluate_reconstruction = compute_metrics


def compute_peak_metrics(
    prediction: Any,
    target: Any,
    building_mask: Any,
    tx_rc: tuple[int, int],
    *,
    selected_idx: Sequence[int] | np.ndarray = (),
    measurements: Sequence[float] | np.ndarray = (),
) -> dict[str, float]:
    """Evaluator-only source-peak metrics computed after algorithm output freezes.

    ``tx_rc`` must never be passed to acquisition or reconstruction.  Keeping these
    metrics in a separate function makes that boundary explicit and testable.
    """

    pred = np.clip(_as_2d_numpy(prediction, "prediction"), 0.0, 1.0)
    truth = _as_2d_numpy(target, "target")
    building = _as_2d_numpy(building_mask, "building_mask").astype(bool)
    if pred.shape != truth.shape or building.shape != truth.shape:
        raise ValueError("peak metric arrays must share a spatial shape")
    row, col = (int(x) for x in tx_rc)
    if not (0 <= row < truth.shape[0] and 0 <= col < truth.shape[1]):
        raise ValueError("evaluator Tx coordinate lies outside the map")
    outdoor = ~building
    rr, cc = np.indices(truth.shape)
    distance = np.sqrt((rr - row) ** 2 + (cc - col) ** 2)
    r5 = outdoor & (distance <= 5.0)
    truth_r5_peak = max(float(np.max(truth[r5])), _EPS)
    predicted_r5_peak = float(np.max(pred[r5]))

    result: dict[str, float] = {
        "peak_amp_ratio_r5": predicted_r5_peak / truth_r5_peak,
        "peak_amplitude_error_r5": abs(predicted_r5_peak - truth_r5_peak),
    }
    for fraction, label in ((0.001, "top01pct"), (0.01, "top1pct")):
        truth_threshold = float(np.quantile(truth[outdoor], 1.0 - fraction))
        pred_threshold = float(np.quantile(pred[outdoor], 1.0 - fraction))
        truth_region = outdoor & (truth >= truth_threshold)
        pred_region = outdoor & (pred >= pred_threshold)
        overlap = int(np.count_nonzero(truth_region & pred_region))
        truth_count = max(int(np.count_nonzero(truth_region)), 1)
        pred_count = max(int(np.count_nonzero(pred_region)), 1)
        result[f"{label}_recall"] = overlap / truth_count
        result[f"{label}_iou"] = overlap / max(
            int(np.count_nonzero(truth_region | pred_region)), 1
        )
        result[f"{label}_dice"] = 2.0 * overlap / (truth_count + pred_count)
        result[f"{label}_rmse"] = float(
            np.sqrt(np.mean(np.square(pred[truth_region] - truth[truth_region])))
        )
        result[f"{label}_nearest_tx_distance"] = float(np.min(distance[pred_region]))

    pred_peak = np.asarray(
        np.unravel_index(np.argmax(np.where(outdoor, pred, -np.inf)), pred.shape),
        dtype=np.float64,
    )
    recovered_peak_distance = float(
        np.linalg.norm(pred_peak - np.asarray([row, col], dtype=np.float64))
    )
    result["predicted_peak_tx_distance"] = recovered_peak_distance
    for radius in (5, 10, 20):
        result[f"recovered_peak_hit_at_{radius}"] = float(
            recovered_peak_distance <= float(radius)
        )

    indices = np.asarray(selected_idx, dtype=np.int64).reshape(-1)
    if indices.size:
        coordinates = np.column_stack(np.unravel_index(indices, truth.shape))
        sample_distance = np.linalg.norm(
            coordinates - np.asarray([[row, col]], dtype=np.float64), axis=1
        )
        result["nearest_sample_tx_distance"] = float(np.min(sample_distance))
        for radius in (5, 10, 20):
            result[f"sample_hit_at_{radius}"] = float(np.any(sample_distance <= radius))
        result["max_sampled_true_gain"] = float(np.max(truth.reshape(-1)[indices]))
        observed = np.asarray(measurements, dtype=np.float64).reshape(-1)
        if observed.shape == indices.shape:
            residual = pred.reshape(-1)[indices] - observed
            result["observation_rmse"] = float(np.sqrt(np.mean(np.square(residual))))
            result["observation_max_abs_residual"] = float(np.max(np.abs(residual)))
    else:
        result.update(
            {
                "nearest_sample_tx_distance": float("inf"),
                "sample_hit_at_5": 0.0,
                "sample_hit_at_10": 0.0,
                "sample_hit_at_20": 0.0,
                "max_sampled_true_gain": float("nan"),
                "observation_rmse": 0.0,
                "observation_max_abs_residual": 0.0,
            }
        )
    return result


def _paired_advantages(
    ours: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    *,
    lower_is_better: bool,
) -> np.ndarray:
    ours_array = np.asarray(ours, dtype=np.float64).reshape(-1)
    baseline_array = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if ours_array.shape != baseline_array.shape:
        raise ValueError(
            f"paired arrays must have identical shape, got {ours_array.shape} and "
            f"{baseline_array.shape}"
        )
    if ours_array.size == 0:
        raise ValueError("paired comparison requires at least one scene")
    if not np.all(np.isfinite(ours_array)) or not np.all(np.isfinite(baseline_array)):
        raise ValueError("paired comparison contains NaN or infinite values")
    return baseline_array - ours_array if lower_is_better else ours_array - baseline_array


def paired_bootstrap_ci(
    ours: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    *,
    lower_is_better: bool = True,
    confidence: float = 0.95,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired percentile bootstrap of the mean advantage (positive favors ours)."""

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0,1), got {confidence}")
    n_resamples = int(n_resamples)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    advantages = _paired_advantages(ours, baseline, lower_is_better=lower_is_better)
    n_pairs = int(advantages.size)
    rng = np.random.RandomState(int(seed))
    boot_means = np.empty(n_resamples, dtype=np.float64)
    # Chunking avoids a large n_resamples-by-n_scenes allocation in full evaluation.
    chunk = max(1, min(n_resamples, 1_000_000 // max(1, n_pairs)))
    for start in range(0, n_resamples, chunk):
        stop = min(start + chunk, n_resamples)
        indices = rng.randint(0, n_pairs, size=(stop - start, n_pairs))
        boot_means[start:stop] = np.mean(advantages[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    wins = advantages > 0.0
    ties = advantages == 0.0
    return {
        "mean_advantage": float(np.mean(advantages)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": float(confidence),
        "win_rate": float(np.mean(wins)),
        "tie_rate": float(np.mean(ties)),
        "n_pairs": n_pairs,
        "n_resamples": n_resamples,
        "seed": int(seed),
        "lower_is_better": bool(lower_is_better),
        "paired_advantages": advantages.copy(),
    }


def _budget_values(values: Mapping[Any, Sequence[float]], budget: int, name: str) -> Any:
    if budget in values:
        return values[budget]
    if str(budget) in values:
        return values[str(budget)]
    raise KeyError(f"{name} has no results for required budget {budget}")


def evaluate_3a_gate(
    ours_by_budget: Mapping[Any, Sequence[float]],
    uniform_by_budget: Mapping[Any, Sequence[float]],
    random_by_budget: Mapping[Any, Sequence[float]],
    *,
    budgets: Sequence[int] = PRIMARY_BUDGETS,
    min_advantage_db: float = GATE_MIN_ADVANTAGE_DB,
    min_win_rate: float = GATE_MIN_WIN_RATE,
    required_passing_budgets: int = GATE_REQUIRED_BUDGETS,
    confidence: float = 0.95,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate the exact user-selected 3A gate on outdoor NMSE (dB).

    A budget passes only when *both* uniform and random comparisons have mean paired
    advantage >= 2 dB, a strictly positive 95% paired-bootstrap lower bound, and at
    least 80% strict wins.  The overall gate passes when at least two of budgets
    16/32/64 pass.  NMSE is lower-is-better, so advantage is baseline minus ours.
    """

    budgets = tuple(int(budget) for budget in budgets)
    if budgets != PRIMARY_BUDGETS:
        raise ValueError(f"the exact 3A gate must use budgets {PRIMARY_BUDGETS}, got {budgets}")
    if int(required_passing_budgets) != GATE_REQUIRED_BUDGETS:
        raise ValueError(
            f"the exact 3A gate requires {GATE_REQUIRED_BUDGETS} passing budgets"
        )
    budget_reports: dict[int, dict[str, Any]] = {}
    passing_budgets: list[int] = []
    baselines = (("uniform", uniform_by_budget), ("random", random_by_budget))
    for budget_index, budget in enumerate(budgets):
        ours = _budget_values(ours_by_budget, budget, "ours")
        comparison_reports: dict[str, dict[str, Any]] = {}
        for baseline_index, (baseline_name, baseline_values) in enumerate(baselines):
            comparison = paired_bootstrap_ci(
                ours,
                _budget_values(baseline_values, budget, baseline_name),
                lower_is_better=True,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=int(seed) + 100 * budget_index + baseline_index,
            )
            comparison["mean_advantage_db"] = comparison["mean_advantage"]
            comparison["passes_mean_advantage"] = bool(
                comparison["mean_advantage"] >= float(min_advantage_db)
            )
            comparison["passes_ci"] = bool(comparison["ci_low"] > 0.0)
            comparison["passes_win_rate"] = bool(
                comparison["win_rate"] >= float(min_win_rate)
            )
            comparison["passed"] = bool(
                comparison["passes_mean_advantage"]
                and comparison["passes_ci"]
                and comparison["passes_win_rate"]
            )
            comparison_reports[baseline_name] = comparison
        budget_passed = bool(all(item["passed"] for item in comparison_reports.values()))
        if budget_passed:
            passing_budgets.append(budget)
        budget_reports[budget] = {
            "passed": budget_passed,
            "comparisons": comparison_reports,
        }
    passed = len(passing_budgets) >= GATE_REQUIRED_BUDGETS
    return {
        "gate": "3A",
        "metric": "outdoor_nmse_db",
        "passed": bool(passed),
        "passing_budgets": tuple(passing_budgets),
        "n_passing_budgets": len(passing_budgets),
        "required_passing_budgets": GATE_REQUIRED_BUDGETS,
        "thresholds": {
            "min_mean_advantage_db": float(min_advantage_db),
            "ci_lower_strictly_above_db": 0.0,
            "confidence": float(confidence),
            "min_win_rate": float(min_win_rate),
        },
        "budgets": budget_reports,
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(n_resamples),
    }


def evaluate_low_budget_gate(
    ours_by_budget: Mapping[Any, Sequence[float]],
    uniform_by_budget: Mapping[Any, Sequence[float]],
    random_by_budget: Mapping[Any, Sequence[float]],
    *,
    budgets: Sequence[int] = (8, 16),
    min_advantage_db: float = GATE_MIN_ADVANTAGE_DB,
    min_win_rate: float = GATE_MIN_WIN_RATE,
    confidence: float = 0.95,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate the focused 8/16-point acceptance gate.

    This is deliberately separate from the pre-registered 3A (16/32/64) gate.
    Both low budgets must beat *both* uniform and random sampling by at least the
    requested paired outdoor-NMSE advantage, with a positive bootstrap lower bound
    and the requested strict-win rate.  It reports the same evidence but never calls
    an 8/16 result a 3A result.
    """

    low_budgets = tuple(int(budget) for budget in budgets)
    if low_budgets != (8, 16):
        raise ValueError(f"the focused low-budget gate requires budgets (8, 16), got {low_budgets}")
    reports: dict[int, dict[str, Any]] = {}
    passing: list[int] = []
    for budget_index, budget in enumerate(low_budgets):
        ours = _budget_values(ours_by_budget, budget, "ours")
        comparisons: dict[str, dict[str, Any]] = {}
        for baseline_index, (name, values) in enumerate(
            (("uniform", uniform_by_budget), ("random", random_by_budget))
        ):
            comparison = paired_bootstrap_ci(
                ours,
                _budget_values(values, budget, name),
                lower_is_better=True,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=int(seed) + 100 * budget_index + baseline_index,
            )
            comparison["mean_advantage_db"] = comparison["mean_advantage"]
            comparison["passes_mean_advantage"] = bool(
                comparison["mean_advantage"] >= float(min_advantage_db)
            )
            comparison["passes_ci"] = bool(comparison["ci_low"] > 0.0)
            comparison["passes_win_rate"] = bool(
                comparison["win_rate"] >= float(min_win_rate)
            )
            comparison["passed"] = bool(
                comparison["passes_mean_advantage"]
                and comparison["passes_ci"]
                and comparison["passes_win_rate"]
            )
            comparisons[name] = comparison
        budget_passed = bool(all(item["passed"] for item in comparisons.values()))
        if budget_passed:
            passing.append(budget)
        reports[budget] = {"passed": budget_passed, "comparisons": comparisons}
    return {
        "gate": "low_budget_8_16",
        "metric": "outdoor_nmse_db",
        "passed": len(passing) == len(low_budgets),
        "passing_budgets": tuple(passing),
        "n_passing_budgets": len(passing),
        "required_passing_budgets": len(low_budgets),
        "thresholds": {
            "min_mean_advantage_db": float(min_advantage_db),
            "ci_lower_strictly_above_db": 0.0,
            "confidence": float(confidence),
            "min_win_rate": float(min_win_rate),
        },
        "budgets": reports,
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(n_resamples),
    }


check_3a_gate = evaluate_3a_gate
three_a_gate = evaluate_3a_gate


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "GATE_MIN_ADVANTAGE_DB",
    "GATE_MIN_WIN_RATE",
    "GATE_REQUIRED_BUDGETS",
    "PRIMARY_BUDGETS",
    "check_3a_gate",
    "compute_metrics",
    "compute_peak_metrics",
    "evaluate_3a_gate",
    "evaluate_low_budget_gate",
    "evaluate_reconstruction",
    "paired_bootstrap_ci",
    "three_a_gate",
]
