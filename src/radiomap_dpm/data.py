"""Authentic RadioMapSeer DPM data access and leakage-safe algorithm inputs.

There is deliberately no synthetic-data fallback in this module.  Missing or malformed
RadioMapSeer files fail loudly.  Scene 0 is reserved for real-data smoke tests; the
paper split is the fixed ``RandomState(42)`` permutation of scenes 1..700.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_SEED = 42
N_SCENES = 700
N_TX_PER_SCENE = 80
SMOKE_SCENE_ID = 0
ALL_TX_IDS = tuple(range(N_TX_PER_SCENE))


def _build_scene_splits() -> dict[str, tuple[int, ...]]:
    # RandomState, rather than default_rng, is part of the dataset protocol.
    order = np.random.RandomState(SPLIT_SEED).permutation(
        np.arange(1, N_SCENES + 1, dtype=np.int64)
    )
    return {
        "train": tuple(int(x) for x in order[:500]),
        "val": tuple(int(x) for x in order[500:600]),
        "test": tuple(int(x) for x in order[600:700]),
        "smoke": (SMOKE_SCENE_ID,),
    }


SCENE_SPLITS = _build_scene_splits()


def scene_splits() -> dict[str, tuple[int, ...]]:
    """Return copies of the exact 500/100/100 split (plus smoke scene 0)."""

    return {name: tuple(ids) for name, ids in SCENE_SPLITS.items()}


def get_scene_ids(split: str) -> tuple[int, ...]:
    """Return scene IDs for ``train``, ``val``, ``test``, ``smoke``, or ``all``.

    ``all`` means the 700 paper scenes and never includes scene 0.
    """

    split = str(split).lower()
    if split == "validation":
        split = "val"
    if split == "all":
        return tuple(range(1, N_SCENES + 1))
    if split not in SCENE_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected train/val/test/smoke/all")
    return tuple(SCENE_SPLITS[split])


def split_for_scene(scene_id: int) -> str:
    """Return the canonical split for a scene, with 0 reported as ``smoke``."""

    scene_id = _validate_scene_id(scene_id, allow_smoke=True)
    if scene_id == SMOKE_SCENE_ID:
        return "smoke"
    for name in ("train", "val", "test"):
        if scene_id in SCENE_SPLITS[name]:
            return name
    raise AssertionError(f"scene {scene_id} was not assigned to a split")


def _validate_scene_id(scene_id: int, *, allow_smoke: bool = True) -> int:
    scene_id = int(scene_id)
    lo = 0 if allow_smoke else 1
    if not lo <= scene_id <= N_SCENES:
        raise ValueError(f"scene_id must be in {lo}..{N_SCENES}, got {scene_id}")
    return scene_id


def _validate_tx_id(tx_id: int) -> int:
    tx_id = int(tx_id)
    if not 0 <= tx_id < N_TX_PER_SCENE:
        raise ValueError(f"tx_id must be in 0..{N_TX_PER_SCENE - 1}, got {tx_id}")
    return tx_id


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required RadioMapSeer file is missing: {path}")
    return path


def _require_case_compatible_file(root: Path, *relative_paths: str) -> Path:
    """Resolve known archive case variants without changing raw data."""

    candidates = [root / relative for relative in relative_paths]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return _require_file(candidates[0])


def _read_luma(path: Path) -> np.ndarray:
    path = _require_file(path)
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image at {path}, got {array.shape}")
    return array


def load_gain(root: str | Path, scene_id: int, tx_id: int) -> np.ndarray:
    """Load an authentic DPM gain map as ``float32`` in [0, 1]."""

    root = Path(root)
    scene_id = _validate_scene_id(scene_id)
    tx_id = _validate_tx_id(tx_id)
    raw = _read_luma(root / "gain" / "DPM" / f"{scene_id}_{tx_id}.png")
    return (raw.astype(np.float32) / np.float32(255.0)).copy()


def load_building_mask(root: str | Path, scene_id: int) -> np.ndarray:
    """Load the complete building map as bool, where ``True`` means building."""

    root = Path(root)
    scene_id = _validate_scene_id(scene_id)
    path = root / "png" / "buildings_complete" / f"{scene_id}.png"
    raw = _read_luma(path)
    values = np.unique(raw)
    if not np.all(np.isin(values, np.array([0, 255], dtype=np.uint8))):
        raise ValueError(f"building mask must be binary 0/255 at {path}; found {values.tolist()}")
    return (raw == 255).copy()


def load_antenna_mask(root: str | Path, scene_id: int, tx_id: int) -> np.ndarray:
    """Load the one-pixel antenna mask as bool."""

    root = Path(root)
    scene_id = _validate_scene_id(scene_id)
    tx_id = _validate_tx_id(tx_id)
    path = root / "png" / "antennas" / f"{scene_id}_{tx_id}.png"
    raw = _read_luma(path)
    mask = raw != 0
    if int(mask.sum()) != 1:
        raise ValueError(f"antenna mask must contain exactly one nonzero pixel at {path}")
    return mask


def load_antenna_rc(
    root: str | Path,
    scene_id: int,
    tx_id: int,
    *,
    image_shape: tuple[int, int] = (256, 256),
    verify_png: bool = False,
) -> tuple[int, int]:
    """Load antenna position and convert Cartesian JSON ``[x,y]`` to image ``(row,col)``.

    Antenna metadata is diagnostics/evaluation information.  It is intentionally omitted
    from :class:`AlgorithmView`.
    """

    root = Path(root)
    scene_id = _validate_scene_id(scene_id)
    tx_id = _validate_tx_id(tx_id)
    path = _require_case_compatible_file(
        root, f"Antenna/{scene_id}.json", f"antenna/{scene_id}.json"
    )
    with path.open("r", encoding="utf-8") as handle:
        positions = json.load(handle)
    if not isinstance(positions, list) or len(positions) != N_TX_PER_SCENE:
        raise ValueError(f"expected {N_TX_PER_SCENE} antenna positions in {path}")
    xy = positions[tx_id]
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        raise ValueError(f"invalid antenna position {xy!r} in {path}")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    col, y_cartesian = int(xy[0]), int(xy[1])
    row = height - 1 - y_cartesian
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"antenna position {xy!r} lies outside image shape {image_shape}")
    rc = (row, col)
    if verify_png:
        antenna_mask = load_antenna_mask(root, scene_id, tx_id)
        if antenna_mask.shape != (height, width):
            raise ValueError(
                f"antenna mask shape {antenna_mask.shape} differs from {(height, width)}"
            )
        png_rc = tuple(int(x) for x in np.argwhere(antenna_mask)[0])
        if png_rc != rc:
            raise ValueError(f"antenna JSON {rc} disagrees with PNG {png_rc} for {scene_id}_{tx_id}")
    return rc


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class AlgorithmView:
    """Only information an acquisition/reconstruction algorithm is allowed to see.

    There is no Tx ID, Tx coordinate, antenna mask, Tx-on-building flag, or unknown
    outdoor gain map here.  Candidate indices are flat row-major indices and are strictly
    non-building.  Building values are authentic gain pixels and have exactly zero noise
    variance; they are free observations and do not consume the outdoor sampling budget.
    """

    shape: tuple[int, int]
    building_mask: np.ndarray
    candidate_indices: np.ndarray
    free_indices: np.ndarray
    free_values: np.ndarray
    free_noise_variance: np.ndarray

    @property
    def building_indices(self) -> np.ndarray:
        return self.free_indices

    @property
    def building_values(self) -> np.ndarray:
        return self.free_values

    @property
    def outdoor_mask(self) -> np.ndarray:
        return ~self.building_mask


@dataclass(frozen=True)
class RadioMapRecord:
    """Complete evaluator-side record; pass :meth:`algorithm_view` to algorithms."""

    scene_id: int
    tx_id: int
    split: str
    gain: np.ndarray
    building_mask: np.ndarray
    antenna_rc: tuple[int, int]
    tx_on_building: bool

    def algorithm_view(self) -> AlgorithmView:
        building = np.asarray(self.building_mask, dtype=bool)
        flat_building = building.ravel()
        building_idx = np.flatnonzero(flat_building).astype(np.int64, copy=False)
        candidates = np.flatnonzero(~flat_building).astype(np.int64, copy=False)
        values = np.asarray(self.gain, dtype=np.float32).ravel()[building_idx]
        if np.intersect1d(building_idx, candidates, assume_unique=True).size:
            raise AssertionError("building observations and outdoor candidates overlap")
        if building_idx.size + candidates.size != flat_building.size:
            raise AssertionError("building/candidate masks do not partition the image")
        return AlgorithmView(
            shape=tuple(int(x) for x in building.shape),
            building_mask=_readonly(building),
            candidate_indices=_readonly(candidates),
            free_indices=_readonly(building_idx),
            free_values=_readonly(values),
            free_noise_variance=_readonly(np.zeros(values.shape, dtype=np.float32)),
        )


class TxOnBuildingError(ValueError):
    """Raised when a diagnostics-flagged record is excluded by protocol."""


def load_radio_map(
    root: str | Path,
    scene_id: int,
    tx_id: int,
    *,
    exclude_tx_on_building: bool = False,
    verify_antenna_png: bool = False,
) -> RadioMapRecord:
    """Load one real RadioMapSeer DPM record with Tx diagnostics kept evaluator-side."""

    root = Path(root)
    scene_id = _validate_scene_id(scene_id)
    tx_id = _validate_tx_id(tx_id)
    gain = load_gain(root, scene_id, tx_id)
    building = load_building_mask(root, scene_id)
    if gain.shape != building.shape:
        raise ValueError(
            f"gain/building shape mismatch for {scene_id}_{tx_id}: {gain.shape} vs {building.shape}"
        )
    antenna_rc = load_antenna_rc(
        root,
        scene_id,
        tx_id,
        image_shape=building.shape,
        verify_png=verify_antenna_png,
    )
    tx_on_building = bool(building[antenna_rc])
    if exclude_tx_on_building and tx_on_building:
        raise TxOnBuildingError(
            f"scene={scene_id}, tx={tx_id} is on a building and exclusion is enabled"
        )
    return RadioMapRecord(
        scene_id=scene_id,
        tx_id=tx_id,
        split=split_for_scene(scene_id),
        gain=gain,
        building_mask=building,
        antenna_rc=antenna_rc,
        tx_on_building=tx_on_building,
    )


class RadioMapDPMDataset(Sequence[RadioMapRecord]):
    """Lazy dataset over the fixed scene split and requested transmitter IDs."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        split: str = "train",
        tx_ids: Iterable[int] | None = None,
        exclude_tx_on_building: bool = False,
        ensure_tx_per_scene: int | None = None,
        verify_antenna_png: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = "val" if str(split).lower() == "validation" else str(split).lower()
        self.scene_ids = get_scene_ids(self.split)
        requested_tx = ALL_TX_IDS if tx_ids is None else tuple(_validate_tx_id(x) for x in tx_ids)
        if not requested_tx:
            raise ValueError("tx_ids must contain at least one transmitter ID")
        if len(set(requested_tx)) != len(requested_tx):
            raise ValueError("tx_ids must not contain duplicates")
        self.tx_ids = tuple(requested_tx)
        self.exclude_tx_on_building = bool(exclude_tx_on_building)
        self.ensure_tx_per_scene = (
            None if ensure_tx_per_scene is None else int(ensure_tx_per_scene)
        )
        if self.ensure_tx_per_scene is not None:
            if not self.exclude_tx_on_building:
                raise ValueError("ensure_tx_per_scene requires exclude_tx_on_building=True")
            if not 1 <= self.ensure_tx_per_scene <= N_TX_PER_SCENE:
                raise ValueError("ensure_tx_per_scene must lie in 1..80")
        self.verify_antenna_png = bool(verify_antenna_png)
        pairs = [(scene_id, tx_id) for scene_id in self.scene_ids for tx_id in self.tx_ids]
        self.excluded_pairs: tuple[tuple[int, int], ...] = ()
        self.supplemented_pairs: tuple[tuple[int, int], ...] = ()
        if self.exclude_tx_on_building:
            kept: list[tuple[int, int]] = []
            excluded: list[tuple[int, int]] = []
            supplemented: list[tuple[int, int]] = []
            for scene_id in self.scene_ids:
                building = load_building_mask(self.root, scene_id)
                scene_kept: list[tuple[int, int]] = []
                for tx_id in self.tx_ids:
                    rc = load_antenna_rc(self.root, scene_id, tx_id, image_shape=building.shape)
                    (excluded if bool(building[rc]) else scene_kept).append((scene_id, tx_id))
                if self.ensure_tx_per_scene is not None:
                    target = self.ensure_tx_per_scene
                    scene_kept = scene_kept[:target]
                    if len(scene_kept) < target:
                        requested = set(self.tx_ids)
                        for tx_id in ALL_TX_IDS:
                            if tx_id in requested:
                                continue
                            rc = load_antenna_rc(
                                self.root, scene_id, tx_id, image_shape=building.shape
                            )
                            if bool(building[rc]):
                                continue
                            pair = (scene_id, tx_id)
                            scene_kept.append(pair)
                            supplemented.append(pair)
                            if len(scene_kept) == target:
                                break
                    if len(scene_kept) != target:
                        raise RuntimeError(
                            f"scene {scene_id} has only {len(scene_kept)} eligible outdoor Tx"
                        )
                kept.extend(scene_kept)
            pairs = kept
            self.excluded_pairs = tuple(excluded)
            self.supplemented_pairs = tuple(supplemented)
        self._pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> RadioMapRecord:
        if isinstance(index, slice):
            raise TypeError("RadioMapDPMDataset returns one lazy record at a time, not slices")
        scene_id, tx_id = self._pairs[int(index)]
        return load_radio_map(
            self.root,
            scene_id,
            tx_id,
            exclude_tx_on_building=self.exclude_tx_on_building,
            verify_antenna_png=self.verify_antenna_png,
        )

    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Evaluator-only scene/Tx identifiers; do not pass these to the algorithm."""

        return tuple(self._pairs)

    def __iter__(self) -> Iterator[RadioMapRecord]:
        for index in range(len(self)):
            yield self[index]


# Short compatibility aliases for experiment scripts.
DPMDataset = RadioMapDPMDataset
load_dpm = load_radio_map


__all__ = [
    "ALL_TX_IDS",
    "AlgorithmView",
    "DEFAULT_ROOT",
    "DPMDataset",
    "N_SCENES",
    "N_TX_PER_SCENE",
    "RadioMapDPMDataset",
    "RadioMapRecord",
    "SCENE_SPLITS",
    "SMOKE_SCENE_ID",
    "SPLIT_SEED",
    "TxOnBuildingError",
    "get_scene_ids",
    "load_antenna_mask",
    "load_antenna_rc",
    "load_building_mask",
    "load_dpm",
    "load_gain",
    "load_radio_map",
    "scene_splits",
    "split_for_scene",
]
