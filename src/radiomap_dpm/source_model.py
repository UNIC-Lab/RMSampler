"""Non-neural source propagation calibration for unknown-Tx active sensing.

Calibration may use transmitter coordinates on the *training* split.  The saved
artifact contains only a position-independent local source kernel and robust
propagation coefficients; it never stores an empirical transmitter-location prior.
Inference APIs accept candidate coordinates, buildings, and amplitudes only.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import DEFAULT_ROOT, RadioMapDPMDataset


@dataclass(frozen=True)
class SourceCalibration:
    kernel: np.ndarray
    coefficients: np.ndarray
    residual_scale: float
    degrees_of_freedom: float = 4.0
    line_samples: int = 12

    def __post_init__(self) -> None:
        kernel = np.asarray(self.kernel, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64).reshape(-1)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
            raise ValueError("kernel must be an odd square array")
        if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must contain intercept, distance, and wall terms")
        if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
            raise ValueError("kernel must be finite and non-negative")
        if not np.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        if self.degrees_of_freedom <= 2.0 or self.line_samples < 2:
            raise ValueError("invalid robust-likelihood configuration")
        object.__setattr__(self, "kernel", kernel.astype(np.float32))
        object.__setattr__(self, "coefficients", coefficients.astype(np.float64))

    @classmethod
    def default(cls, radius: int = 10) -> "SourceCalibration":
        """Conservative analytic fallback used only before a train artifact exists."""

        rr, cc = np.indices((2 * radius + 1, 2 * radius + 1))
        distance = np.sqrt((rr - radius) ** 2 + (cc - radius) ** 2)
        kernel = np.clip(1.0 - 0.105 * np.log1p(distance), 0.0, 1.0)
        return cls(
            kernel=kernel,
            coefficients=np.array([0.95, -0.62, -0.30], dtype=np.float64),
            residual_scale=0.10,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SourceCalibration":
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(
                kernel=payload["kernel"],
                coefficients=payload["coefficients"],
                residual_scale=float(payload["residual_scale"]),
                degrees_of_freedom=float(payload["degrees_of_freedom"]),
                line_samples=int(payload["line_samples"]),
            )

    def save(self, path: str | Path, *, metadata: dict[str, object], wall_time: float) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            kernel=self.kernel,
            coefficients=self.coefficients,
            residual_scale=np.float64(self.residual_scale),
            degrees_of_freedom=np.float64(self.degrees_of_freedom),
            line_samples=np.int64(self.line_samples),
            objective_values=np.asarray([self.residual_scale], dtype=np.float64),
            wall_times=np.asarray([wall_time], dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return output

    def _predict_from_features(
        self, distance: np.ndarray, wall_fraction: np.ndarray, amplitude: np.ndarray | float
    ) -> np.ndarray:
        normalized_distance = np.log1p(distance) / np.log1p(np.hypot(255.0, 255.0))
        base = (
            self.coefficients[0]
            + self.coefficients[1] * normalized_distance
            + self.coefficients[2] * wall_fraction
        )
        return np.clip(np.asarray(amplitude) * base, 0.0, 1.0)

    def predict_matrix(
        self,
        source_coordinates: np.ndarray,
        query_coordinates: np.ndarray,
        building_mask: np.ndarray,
        amplitudes: np.ndarray | float = 1.0,
        *,
        query_chunk: int = 512,
    ) -> np.ndarray:
        """Return robust propagation means with shape ``(n_sources, n_queries)``."""

        sources = np.asarray(source_coordinates, dtype=np.float64).reshape(-1, 2)
        queries = np.asarray(query_coordinates, dtype=np.float64).reshape(-1, 2)
        building = np.asarray(building_mask, dtype=bool)
        if building.ndim != 2 or not sources.size or not queries.size:
            raise ValueError("non-empty coordinates and a 2-D building mask are required")
        if np.any(sources < 0.0) or np.any(queries < 0.0):
            raise ValueError("coordinates must lie inside the map")
        amp = np.asarray(amplitudes, dtype=np.float64)
        if amp.ndim == 0:
            amp = np.full(sources.shape[0], float(amp), dtype=np.float64)
        amp = amp.reshape(-1)
        if amp.shape != (sources.shape[0],) or np.any(~np.isfinite(amp)) or np.any(amp <= 0.0):
            raise ValueError("amplitudes must contain one positive value per source")
        output = np.empty((sources.shape[0], queries.shape[0]), dtype=np.float64)
        fraction = np.linspace(0.0, 1.0, int(self.line_samples), dtype=np.float64)
        height, width = building.shape
        for start in range(0, queries.shape[0], int(query_chunk)):
            query = queries[start : start + int(query_chunk)]
            delta = query[None, :, :] - sources[:, None, :]
            distance = np.sqrt(np.sum(delta * delta, axis=2))
            points = sources[:, None, None, :] + delta[:, :, None, :] * fraction[None, None, :, None]
            rows = np.clip(np.rint(points[..., 0]).astype(np.int64), 0, height - 1)
            cols = np.clip(np.rint(points[..., 1]).astype(np.int64), 0, width - 1)
            wall_fraction = building[rows, cols].mean(axis=2)
            output[:, start : start + query.shape[0]] = self._predict_from_features(
                distance, wall_fraction, amp[:, None]
            )
        return output.astype(np.float32)

    def kernel_constraint(
        self,
        center: tuple[int, int] | np.ndarray,
        building_mask: np.ndarray,
        amplitude: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return visible outdoor indices and soft source-kernel values."""

        building = np.asarray(building_mask, dtype=bool)
        row, col = (int(x) for x in np.asarray(center).reshape(2))
        radius = self.kernel.shape[0] // 2
        indices: list[int] = []
        values: list[float] = []
        for kr in range(self.kernel.shape[0]):
            for kc in range(self.kernel.shape[1]):
                # The calibration patch is stored in a square array, but a source
                # is radially local.  Excluding the four corners prevents an
                # axis-aligned 21x21 support window from becoming a visible square
                # artefact in the final RM reconstruction.
                if (kr - radius) ** 2 + (kc - radius) ** 2 > radius**2:
                    continue
                rr, cc = row + kr - radius, col + kc - radius
                if not (0 <= rr < building.shape[0] and 0 <= cc < building.shape[1]):
                    continue
                if building[rr, cc]:
                    continue
                # Building pixels are excluded.  Remaining local pixels are a soft
                # near-field proposal; detailed NLOS structure is left to the DPM
                # rather than issuing hundreds of one-pixel propagation calls.
                indices.append(rr * building.shape[1] + cc)
                values.append(float(np.clip(amplitude * self.kernel[kr, kc], 0.0, 1.0)))
        return np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.float32)


def _line_wall_fraction(
    sources: np.ndarray, queries: np.ndarray, building: np.ndarray, samples: int
) -> np.ndarray:
    fraction = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
    points = sources[:, None, :] + (queries - sources)[:, None, :] * fraction[None, :, None]
    rows = np.clip(np.rint(points[..., 0]).astype(np.int64), 0, building.shape[0] - 1)
    cols = np.clip(np.rint(points[..., 1]).astype(np.int64), 0, building.shape[1] - 1)
    return building[rows, cols].mean(axis=1)


def _huber_fit(features: np.ndarray, targets: np.ndarray, iterations: int = 20) -> np.ndarray:
    coefficient, *_ = np.linalg.lstsq(features, targets, rcond=None)
    for _ in range(int(iterations)):
        residual = targets - features @ coefficient
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-3)
        ratio = np.abs(residual) / (1.5 * scale)
        weights = np.ones_like(ratio)
        mask = ratio > 1.0
        weights[mask] = 1.0 / ratio[mask]
        weighted = features * np.sqrt(weights)[:, None]
        updated, *_ = np.linalg.lstsq(weighted, targets * np.sqrt(weights), rcond=None)
        if np.linalg.norm(updated - coefficient) < 1e-8:
            coefficient = updated
            break
        coefficient = updated
    return coefficient


def calibrate_source_model(
    root: str | Path,
    *,
    max_records: int = 200,
    pixels_per_record: int = 128,
    tx_ids: Sequence[int] = (0, 20, 40, 60),
    seed: int = 42031,
) -> tuple[SourceCalibration, dict[str, object]]:
    """Fit the position-independent source model on authentic train records."""

    rng = np.random.default_rng(int(seed))
    dataset = RadioMapDPMDataset(
        root, split="train", tx_ids=tx_ids, exclude_tx_on_building=True
    )
    kernels: list[np.ndarray] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    used = 0
    radius = 10
    for record in dataset:
        if used >= int(max_records):
            break
        row, col = record.antenna_rc  # train-only calibration label
        building = np.asarray(record.building_mask, dtype=bool)
        gain = np.asarray(record.gain, dtype=np.float64)
        if (
            radius <= row < gain.shape[0] - radius
            and radius <= col < gain.shape[1] - radius
            and not building[row - radius : row + radius + 1, col - radius : col + radius + 1].any()
        ):
            patch = gain[row - radius : row + radius + 1, col - radius : col + radius + 1]
            kernels.append(patch / max(float(gain[row, col]), 1e-6))
        outdoor = np.flatnonzero(~building.reshape(-1))
        take = min(int(pixels_per_record), outdoor.size)
        chosen = rng.choice(outdoor, size=take, replace=False)
        query = np.column_stack(np.unravel_index(chosen, gain.shape)).astype(np.float64)
        source = np.repeat(np.asarray([[row, col]], dtype=np.float64), take, axis=0)
        distance = np.linalg.norm(query - source, axis=1)
        wall = _line_wall_fraction(source, query, building, samples=12)
        normalized_distance = np.log1p(distance) / np.log1p(np.hypot(255.0, 255.0))
        features.append(np.column_stack((np.ones(take), normalized_distance, wall)))
        targets.append(gain.reshape(-1)[chosen])
        used += 1
    if used < 1 or not features:
        raise RuntimeError("no authentic train records were available for calibration")
    design = np.concatenate(features, axis=0)
    response = np.concatenate(targets, axis=0)
    coefficient = _huber_fit(design, response)
    residual = response - design @ coefficient
    residual_scale = max(
        1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 0.02
    )
    if kernels:
        kernel = np.median(np.stack(kernels), axis=0)
    else:
        kernel = SourceCalibration.default(radius).kernel
    calibration = SourceCalibration(
        kernel=np.clip(kernel, 0.0, 1.0),
        coefficients=coefficient,
        residual_scale=residual_scale,
    )
    metadata: dict[str, object] = {
        "split": "train",
        "max_records": int(max_records),
        "used_records": int(used),
        "pixels_per_record": int(pixels_per_record),
        "tx_ids": [int(x) for x in tx_ids],
        "seed": int(seed),
        "learned_tx_location_prior": False,
        "neural_training": False,
        "n_regression_samples": int(response.size),
        "n_clear_kernel_patches": int(len(kernels)),
    }
    return calibration, metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate the non-neural Tx source model")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "radiomap_dpm" / "source_calibration.npz",
    )
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--pixels-per-record", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42031)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.smoke:
        print("=== REAL DPM SOURCE-CALIBRATION SMOKE TEST ===")
    started = time.perf_counter()
    calibration, metadata = calibrate_source_model(
        args.root,
        max_records=2 if args.smoke else args.max_records,
        pixels_per_record=24 if args.smoke else args.pixels_per_record,
        tx_ids=(0,),
        seed=args.seed,
    )
    output = calibration.save(args.output, metadata=metadata, wall_time=time.perf_counter() - started)
    loaded = SourceCalibration.load(output)
    if loaded.kernel.shape != calibration.kernel.shape or not np.all(np.isfinite(loaded.kernel)):
        raise RuntimeError("saved calibration failed round-trip validation")
    print(f"Calibration saved: {output}")
    if args.smoke:
        print("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SourceCalibration", "calibrate_source_model"]
