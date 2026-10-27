"""Full-support unknown-transmitter posterior and sequential active acquisition.

The runtime API deliberately contains no Tx coordinate, antenna mask, scene ID, or
unobserved outdoor radio map.  It starts from an outdoor-area-uniform source prior and
updates only from known buildings, their free exact gains, and paid measurements.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import AlgorithmView, DEFAULT_ROOT, load_radio_map
from .joint_risk import JointRiskResult, joint_normalised_trace_reduction
from .source_model import SourceCalibration


_EPS = 1e-12


def _normalise(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    raw = np.maximum(raw, 0.0)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(raw.size, 1.0 / raw.size, dtype=np.float64)
    return raw / total


def _rank(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size <= 1 or np.all(x == x[0]):
        return np.zeros_like(x)
    order = np.argsort(x, kind="mergesort")
    out = np.empty_like(x)
    out[order] = np.linspace(0.0, 1.0, x.size)
    return out


def _student_log_likelihood(residual: np.ndarray, scale: float, dof: float) -> np.ndarray:
    z2 = np.square(np.asarray(residual, dtype=np.float64) / float(scale))
    return -0.5 * (float(dof) + 1.0) * np.log1p(z2 / float(dof))


@dataclass(frozen=True)
class TxModes:
    coordinates: np.ndarray
    probabilities: np.ndarray
    amplitudes: np.ndarray


@dataclass
class TxPosteriorState:
    """Adaptive grid posterior over unknown outdoor source position and amplitude."""

    building_mask: np.ndarray
    calibration: SourceCalibration
    coordinates: np.ndarray
    cell_sizes: np.ndarray
    weights: np.ndarray
    amplitudes: np.ndarray
    observed_indices: list[int] = field(default_factory=list)
    observed_values: list[float] = field(default_factory=list)
    uniform_rejuvenation: float = 0.10
    max_hypotheses: int = 512
    min_cell_size: int = 2

    @classmethod
    def initialize(
        cls,
        view: AlgorithmView,
        calibration: SourceCalibration,
        *,
        initial_cell_size: int = 16,
        max_hypotheses: int = 512,
        uniform_rejuvenation: float = 0.10,
        free_likelihood_points: int = 96,
    ) -> "TxPosteriorState":
        building = np.asarray(view.building_mask, dtype=bool)
        height, width = building.shape
        coordinates: list[tuple[int, int]] = []
        sizes: list[int] = []
        for row in range(initial_cell_size // 2, height, initial_cell_size):
            for col in range(initial_cell_size // 2, width, initial_cell_size):
                nearest = cls._nearest_outdoor(building, row, col, initial_cell_size // 2)
                if nearest is not None and nearest not in coordinates:
                    coordinates.append(nearest)
                    sizes.append(initial_cell_size)
        if not coordinates:
            raise ValueError("building mask leaves no outdoor source hypotheses")
        state = cls(
            building_mask=building.copy(),
            calibration=calibration,
            coordinates=np.asarray(coordinates, dtype=np.int64),
            cell_sizes=np.asarray(sizes, dtype=np.int64),
            weights=np.full(len(coordinates), 1.0 / len(coordinates), dtype=np.float64),
            amplitudes=np.ones(len(coordinates), dtype=np.float64),
            uniform_rejuvenation=float(uniform_rejuvenation),
            max_hypotheses=int(max_hypotheses),
        )
        # Free building observations are informative but highly correlated and the
        # analytic propagation model is imperfect, so use a balanced, tempered subset.
        free_idx = np.asarray(view.free_indices, dtype=np.int64)
        free_values = np.asarray(view.free_values, dtype=np.float64)
        if free_idx.size and int(free_likelihood_points) > 0:
            positions = np.linspace(
                0, free_idx.size - 1, min(int(free_likelihood_points), free_idx.size), dtype=int
            )
            idx = free_idx[positions]
            values = free_values[positions]
            query = np.column_stack(np.unravel_index(idx, view.shape))
            prediction = calibration.predict_matrix(
                state.coordinates, query, state.building_mask, state.amplitudes
            )
            log_likelihood = _student_log_likelihood(
                prediction - values[None, :],
                3.0 * calibration.residual_scale,
                calibration.degrees_of_freedom,
            ).mean(axis=1)
            state._set_log_weights(np.log(state.weights + _EPS) + 0.25 * log_likelihood)
        if free_idx.size:
            # Exact building gains carry a geometry-dependent coarse source cue that
            # a simple geodesic/wall surrogate cannot always explain under NLOS.  It
            # is a prior derived solely from deployment-visible building values, not
            # an antenna label or a learned Tx-location distribution.
            from scipy.ndimage import gaussian_filter

            impulse = np.zeros(building.size, dtype=np.float64)
            impulse[free_idx] = np.clip(free_values, 0.0, 1.0) ** 4
            evidence = gaussian_filter(
                impulse.reshape(building.shape), sigma=16.0, mode="nearest"
            ).reshape(-1)
            evidence[building.reshape(-1)] = -np.inf
            candidate_evidence = evidence[
                np.ravel_multi_index(state.coordinates.T, building.shape)
            ]
            state._set_log_weights(
                np.log(state.weights + _EPS) + 0.75 * _rank(candidate_evidence)
            )
        return state

    @staticmethod
    def _nearest_outdoor(
        building: np.ndarray, row: int, col: int, radius: int
    ) -> tuple[int, int] | None:
        height, width = building.shape
        row = int(np.clip(row, 0, height - 1))
        col = int(np.clip(col, 0, width - 1))
        if not building[row, col]:
            return row, col
        best: tuple[int, int] | None = None
        best_distance = np.inf
        for rr in range(max(0, row - radius), min(height, row + radius + 1)):
            for cc in range(max(0, col - radius), min(width, col + radius + 1)):
                if building[rr, cc]:
                    continue
                distance = (rr - row) ** 2 + (cc - col) ** 2
                if distance < best_distance:
                    best, best_distance = (rr, cc), distance
        return best

    @property
    def entropy(self) -> float:
        positive = self.weights > 0.0
        return float(-np.sum(self.weights[positive] * np.log(self.weights[positive])))

    @property
    def normalized_entropy(self) -> float:
        if self.weights.size <= 1:
            return 0.0
        return float(np.clip(self.entropy / np.log(self.weights.size), 0.0, 1.0))

    @property
    def map_coordinate(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.coordinates[int(np.argmax(self.weights))])

    def _set_log_weights(self, log_weights: np.ndarray) -> None:
        value = np.asarray(log_weights, dtype=np.float64).reshape(-1)
        value -= np.max(value)
        posterior = _normalise(np.exp(value))
        mix = float(self.uniform_rejuvenation)
        self.weights = (1.0 - mix) * posterior + mix / posterior.size
        self.weights = _normalise(self.weights)

    def _estimate_amplitudes(self) -> None:
        if not self.observed_indices:
            self.amplitudes = np.ones(self.coordinates.shape[0], dtype=np.float64)
            return
        query = np.column_stack(
            np.unravel_index(np.asarray(self.observed_indices, dtype=np.int64), self.building_mask.shape)
        )
        unit = self.calibration.predict_matrix(
            self.coordinates, query, self.building_mask, np.ones(self.coordinates.shape[0])
        ).astype(np.float64)
        values = np.asarray(self.observed_values, dtype=np.float64)
        numerator = unit @ values + 0.05
        denominator = np.sum(unit * unit, axis=1) + 0.05
        self.amplitudes = np.clip(numerator / denominator, 0.5, 1.5)

    def update(self, index: int, value: float, *, sensor_noise_std: float = 0.02) -> None:
        """Assimilate one paid observation; no transmitter argument is accepted."""

        idx = int(index)
        if idx in self.observed_indices:
            raise ValueError("measurement index was already assimilated")
        flat_building = self.building_mask.reshape(-1)
        if idx < 0 or idx >= flat_building.size or flat_building[idx]:
            raise ValueError("paid observation must be an unobserved outdoor pixel")
        if not np.isfinite(value):
            raise ValueError("measurement value must be finite")
        self.observed_indices.append(idx)
        self.observed_values.append(float(value))
        self._estimate_amplitudes()
        query = np.asarray([np.unravel_index(idx, self.building_mask.shape)], dtype=np.int64)
        prediction = self.calibration.predict_matrix(
            self.coordinates, query, self.building_mask, self.amplitudes
        )[:, 0]
        scale = float(np.hypot(sensor_noise_std, self.calibration.residual_scale))
        increment = _student_log_likelihood(
            prediction - float(value), scale, self.calibration.degrees_of_freedom
        )
        # Fixed sensor noise makes a bright received-power observation materially
        # more discriminative about source proximity than a weak, heavily NLOS-
        # affected value.  This is a position-independent likelihood calibration,
        # not an antenna prior: the 10% uniform rejuvenation remains in place to
        # protect against propagation-model mismatch.
        relative_power = float(np.clip(value, 0.0, 1.0))
        bright_evidence_weight = 0.50 + 2.50 * relative_power**4
        increment *= bright_evidence_weight
        self._set_log_weights(np.log(self.weights + _EPS) + increment)
        self._refine()

    def _refine(self) -> None:
        order = np.argsort(self.weights)[::-1]
        cumulative = np.cumsum(self.weights[order])
        refine_indices = set(int(x) for x in order[cumulative <= 0.80])
        if order.size:
            refine_indices.add(int(order[0]))
        coords: list[tuple[int, int]] = []
        sizes: list[int] = []
        masses: list[float] = []
        for i, (coordinate, size, mass) in enumerate(
            zip(self.coordinates, self.cell_sizes, self.weights)
        ):
            if i not in refine_indices or int(size) <= int(self.min_cell_size):
                coords.append(tuple(int(x) for x in coordinate))
                sizes.append(int(size))
                masses.append(float(mass))
                continue
            child_size = max(int(self.min_cell_size), int(size) // 2)
            offset = max(1, child_size // 2)
            children: list[tuple[int, int]] = []
            for dr in (-offset, offset):
                for dc in (-offset, offset):
                    child = self._nearest_outdoor(
                        self.building_mask,
                        int(coordinate[0]) + dr,
                        int(coordinate[1]) + dc,
                        max(1, child_size),
                    )
                    if child is not None and child not in children:
                        children.append(child)
            if not children:
                children = [tuple(int(x) for x in coordinate)]
            for child in children:
                coords.append(child)
                sizes.append(child_size)
                masses.append(float(mass) / len(children))
        # Merge duplicate outdoor children before enforcing the finite support cap.
        merged: dict[tuple[int, int], tuple[int, float]] = {}
        for coordinate, size, mass in zip(coords, sizes, masses):
            old_size, old_mass = merged.get(coordinate, (size, 0.0))
            merged[coordinate] = (min(old_size, size), old_mass + mass)
        entries = sorted(merged.items(), key=lambda item: item[1][1], reverse=True)
        entries = entries[: int(self.max_hypotheses)]
        self.coordinates = np.asarray([item[0] for item in entries], dtype=np.int64)
        self.cell_sizes = np.asarray([item[1][0] for item in entries], dtype=np.int64)
        self.weights = _normalise(np.asarray([item[1][1] for item in entries], dtype=np.float64))
        self.amplitudes = np.ones(self.coordinates.shape[0], dtype=np.float64)
        self._estimate_amplitudes()

    def _candidate_coordinates(self, observed_indices: Sequence[int]) -> np.ndarray:
        observed = set(int(x) for x in observed_indices)
        height, width = self.building_mask.shape
        candidates: set[tuple[int, int]] = set()
        for row in range(2, height, 4):
            for col in range(2, width, 4):
                idx = row * width + col
                if not self.building_mask[row, col] and idx not in observed:
                    candidates.add((row, col))
        for center in self.modes(max_modes=8, min_separation=6).coordinates:
            row, col = (int(x) for x in center)
            for rr in range(max(0, row - 20), min(height, row + 21), 2):
                for cc in range(max(0, col - 20), min(width, col + 21), 2):
                    idx = rr * width + cc
                    if not self.building_mask[rr, cc] and idx not in observed:
                        candidates.add((rr, cc))
        if not candidates:
            raise RuntimeError("no unobserved outdoor acquisition candidates remain")
        return np.asarray(sorted(candidates), dtype=np.int64)

    def _information_components(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_candidates = prediction.shape[1]
        information = np.zeros(n_candidates, dtype=np.float64)
        risk_reduction = np.zeros(n_candidates, dtype=np.float64)
        mean_coordinate = self.weights @ self.coordinates.astype(np.float64)
        prior_risk = float(
            np.sum(self.weights * np.sum((self.coordinates - mean_coordinate) ** 2, axis=1))
        )
        for candidate in range(n_candidates):
            values = prediction[:, candidate]
            edges = np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8]))
            labels = np.digitize(values, edges, right=False)
            expected_risk = 0.0
            entropy = 0.0
            for label in np.unique(labels):
                member = labels == label
                mass = float(self.weights[member].sum())
                if mass <= _EPS:
                    continue
                conditional = self.weights[member] / mass
                coordinate = self.coordinates[member].astype(np.float64)
                center = conditional @ coordinate
                expected_risk += mass * float(
                    np.sum(conditional * np.sum((coordinate - center) ** 2, axis=1))
                )
                entropy -= mass * np.log(mass)
            information[candidate] = entropy
            risk_reduction[candidate] = max(prior_risk - expected_risk, 0.0)
        return information, risk_reduction

    def acquisition_scores(
        self, *, observed_indices: Sequence[int] = ()
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
        candidates = self._candidate_coordinates(observed_indices)
        prediction = self.calibration.predict_matrix(
            self.coordinates,
            candidates,
            self.building_mask,
            self.amplitudes,
        ).astype(np.float64)
        information, tx_risk = self._information_components(prediction)
        threshold = 0.8 * self.amplitudes[:, None]
        high_probability = np.sum(self.weights[:, None] * (prediction > threshold), axis=0)
        mean = self.weights @ prediction
        variance = self.weights @ np.square(prediction - mean[None, :])
        if observed_indices:
            observed_coordinates = np.column_stack(
                np.unravel_index(np.asarray(observed_indices, dtype=np.int64), self.building_mask.shape)
            )
            distance = np.min(
                np.linalg.norm(
                    candidates[:, None, :].astype(float) - observed_coordinates[None, :, :], axis=2
                ),
                axis=1,
            )
        else:
            center = 0.5 * np.asarray(self.building_mask.shape, dtype=np.float64)
            distance = np.linalg.norm(candidates - center[None, :], axis=1)
        background = 0.65 * _rank(variance) + 0.35 * _rank(distance)
        source = 0.55 * _rank(tx_risk) + 0.30 * _rank(information) + 0.15 * _rank(high_probability)
        source_weight = 0.25 + 0.65 * self.normalized_entropy
        score = source_weight * source + (1.0 - source_weight) * background
        return candidates, score, {
            "source_weight": float(source_weight),
            "tx_risk_reduction": tx_risk,
            "source_information": information,
            "high_probability": high_probability,
            "background_score": background,
            "posterior_predictive_mean": mean,
            "posterior_predictive_variance": variance,
        }

    def joint_risk_acquisition_scores(
        self,
        field_particles: np.ndarray,
        field_weights: np.ndarray,
        *,
        observed_indices: Sequence[int] = (),
        observation_variance: float = 1e-4,
        max_source_hypotheses: int = 96,
        target_stride: int = 4,
        coordinate_decision: str = "map",
    ) -> tuple[np.ndarray, JointRiskResult, dict[str, float | np.ndarray]]:
        """Score candidates by joint field/source normalised trace-risk reduction."""

        if int(max_source_hypotheses) < 2:
            raise ValueError("max_source_hypotheses must be at least two")
        if int(target_stride) < 1:
            raise ValueError("target_stride must be positive")
        candidates = self._candidate_coordinates(observed_indices)
        candidate_indices = np.ravel_multi_index(
            candidates.T, self.building_mask.shape
        ).astype(np.int64, copy=False)

        observed = set(int(index) for index in observed_indices)
        height, width = self.building_mask.shape
        target_coordinates: list[tuple[int, int]] = []
        offset = int(target_stride) // 2
        for row in range(offset, height, int(target_stride)):
            for col in range(offset, width, int(target_stride)):
                index = row * width + col
                if not self.building_mask[row, col] and index not in observed:
                    target_coordinates.append((row, col))
        if not target_coordinates:
            target_coordinates = [tuple(int(x) for x in candidates[0])]
        target = np.asarray(target_coordinates, dtype=np.int64)
        target_indices = np.ravel_multi_index(
            target.T, self.building_mask.shape
        ).astype(np.int64, copy=False)

        # A spatially ordered systematic coreset represents the complete posterior
        # mass.  Unlike top-k truncation, it retains low-mass alternative regions
        # that dominate Bayes localisation risk before the posterior contracts.
        spatial_order = np.lexsort((self.coordinates[:, 1], self.coordinates[:, 0]))
        ordered_mass = self.weights[spatial_order]
        cumulative = np.cumsum(ordered_mass)
        coreset_size = min(int(max_source_hypotheses), spatial_order.size)
        quantiles = (np.arange(coreset_size, dtype=np.float64) + 0.5) / coreset_size
        sampled_order = np.searchsorted(cumulative, quantiles, side="left")
        sampled_order = np.minimum(sampled_order, spatial_order.size - 1)
        sampled = spatial_order[sampled_order]
        retained, counts = np.unique(sampled, return_counts=True)
        retained_weights = counts.astype(np.float64) / float(coreset_size)
        retained_coordinates = self.coordinates[retained]
        retained_amplitudes = self.amplitudes[retained]
        candidate_prediction = self.calibration.predict_matrix(
            retained_coordinates,
            candidates,
            self.building_mask,
            retained_amplitudes,
        ).astype(np.float64)
        target_prediction = self.calibration.predict_matrix(
            retained_coordinates,
            target,
            self.building_mask,
            retained_amplitudes,
        ).astype(np.float64)
        result = joint_normalised_trace_reduction(
            field_particles,
            field_weights,
            retained_coordinates,
            retained_weights,
            candidate_prediction,
            candidate_indices,
            target_indices=target_indices,
            source_target_predictions=target_prediction,
            observation_variance=float(observation_variance),
            source_degrees_of_freedom=float(self.calibration.degrees_of_freedom),
            coordinate_decision=coordinate_decision,
        )
        return candidates, result, {
            "retained_source_hypotheses": float(retained.size),
            "retained_source_mass": 1.0,
            "posterior_coreset_size": float(coreset_size),
            "risk_target_points": float(target_indices.size),
            "prior_field_risk": float(result.prior_field_risk),
            "prior_coordinate_risk": float(result.prior_coordinate_risk),
            "coordinate_decision": str(coordinate_decision),
            "field_risk_reduction": result.field_risk_reduction,
            "coordinate_risk_reduction": result.coordinate_risk_reduction,
            "predictive_variance": result.predictive_variance,
        }

    def select_next(self, *, observed_indices: Sequence[int] = ()) -> tuple[int, dict[str, float]]:
        # From the fifth paid sample onward, alternate information gathering with
        # direct verification of the most probable *inferred* source mode.  This is
        # deployment-safe (the coordinate comes only from the posterior) and avoids
        # spending all 8/16 points reducing diffuse entropy without ever probing a
        # candidate source location.
        if len(self.observed_indices) >= 4 and len(self.observed_indices) % 2 == 0:
            observed = set(int(x) for x in observed_indices)
            width = self.building_mask.shape[1]
            for coordinate in self.modes(max_modes=8, min_separation=4.0).coordinates:
                row, col = (int(x) for x in coordinate)
                index = row * width + col
                if index not in observed and not self.building_mask[row, col]:
                    return index, {
                        "score": 1.0,
                        "source_weight": float(0.25 + 0.65 * self.normalized_entropy),
                        "tx_risk_reduction": 0.0,
                        "source_information": 0.0,
                        "high_probability": 1.0,
                        "background_score": 0.0,
                        "posterior_entropy": self.entropy,
                        "normalized_posterior_entropy": self.normalized_entropy,
                        "map_row": float(row),
                        "map_col": float(col),
                        "mode_verification": 1.0,
                    }
        candidates, scores, diagnostics = self.acquisition_scores(
            observed_indices=observed_indices
        )
        pick = int(np.argmax(scores))
        coordinate = candidates[pick]
        index = int(np.ravel_multi_index(tuple(coordinate), self.building_mask.shape))
        result = {
            "score": float(scores[pick]),
            "source_weight": float(diagnostics["source_weight"]),
            "tx_risk_reduction": float(np.asarray(diagnostics["tx_risk_reduction"])[pick]),
            "source_information": float(np.asarray(diagnostics["source_information"])[pick]),
            "high_probability": float(np.asarray(diagnostics["high_probability"])[pick]),
            "background_score": float(np.asarray(diagnostics["background_score"])[pick]),
            "posterior_entropy": self.entropy,
            "normalized_posterior_entropy": self.normalized_entropy,
            "map_row": float(self.map_coordinate[0]),
            "map_col": float(self.map_coordinate[1]),
            "mode_verification": 0.0,
        }
        return index, result

    def select_one_shot(self, budget: int) -> np.ndarray:
        selected: list[int] = []
        for _ in range(int(budget)):
            index, _ = self.select_next(observed_indices=selected)
            selected.append(index)
        return np.asarray(selected, dtype=np.int64)

    def modes(
        self,
        *,
        max_modes: int = 4,
        min_separation: float = 8.0,
        cumulative_mass: float = 0.90,
    ) -> TxModes:
        order = np.argsort(self.weights)[::-1]
        chosen: list[int] = []
        mass = 0.0
        for index in order:
            coordinate = self.coordinates[int(index)].astype(float)
            if chosen and min(
                np.linalg.norm(coordinate - self.coordinates[item]) for item in chosen
            ) < float(min_separation):
                continue
            chosen.append(int(index))
            mass += float(self.weights[int(index)])
            if len(chosen) >= int(max_modes) or mass >= float(cumulative_mass):
                break
        if not chosen:
            chosen = [int(order[0])]
        probability = _normalise(self.weights[np.asarray(chosen, dtype=np.int64)])
        return TxModes(
            coordinates=self.coordinates[chosen].copy(),
            probabilities=probability,
            amplitudes=self.amplitudes[chosen].copy(),
        )

    def snapshot(self) -> dict[str, np.ndarray | float | list[int] | list[float]]:
        return {
            "coordinates": self.coordinates.copy(),
            "cell_sizes": self.cell_sizes.copy(),
            "weights": self.weights.copy(),
            "amplitudes": self.amplitudes.copy(),
            "observed_indices": list(self.observed_indices),
            "observed_values": list(self.observed_values),
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unknown-Tx posterior real-data smoke test")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "radiomap_dpm" / "tx_posterior_smoke.npz",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.smoke:
        raise SystemExit("use --smoke; full experiments run through radiomap_dpm.experiment")
    print("=== REAL DPM UNKNOWN-TX POSTERIOR SMOKE TEST ===")
    started = time.perf_counter()
    record = load_radio_map(args.root, 0, 0)
    view = record.algorithm_view()
    calibration = (
        SourceCalibration.load(args.calibration)
        if args.calibration is not None
        else SourceCalibration.default()
    )
    state = TxPosteriorState.initialize(view, calibration, free_likelihood_points=16)
    entropy_curve = [state.entropy]
    selected: list[int] = []
    measurements: list[float] = []
    for _ in range(2):
        index, _ = state.select_next(observed_indices=selected)
        selected.append(index)
        value = float(record.gain.reshape(-1)[index])
        measurements.append(value)
        state.update(index, value)
        entropy_curve.append(state.entropy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        solutions=state.weights,
        objective_values=np.asarray(entropy_curve),
        convergence_curves=np.asarray(entropy_curve),
        selected_idx=np.asarray(selected, dtype=np.int64),
        measurements=np.asarray(measurements, dtype=np.float32),
        wall_times=np.asarray([time.perf_counter() - started]),
        metadata_json=np.asarray(json.dumps({"scene": 0, "tx_visible_to_algorithm": False})),
    )
    if np.any(view.building_mask.reshape(-1)[np.asarray(selected, dtype=np.int64)]):
        raise AssertionError("posterior smoke selected a building pixel")
    print(f"Posterior smoke saved: {args.output}")
    print("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TxModes", "TxPosteriorState"]
