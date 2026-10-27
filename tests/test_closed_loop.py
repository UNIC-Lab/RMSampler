"""Deterministic software fixtures, not simulated or measured paper evidence."""
import numpy as np
import pytest
from radiomap_dpm.data import AlgorithmView, scene_splits
from radiomap_dpm.source_model import SourceCalibration
from rmsampler import SamplingConfig, select_locations, reconstruct_from_candidates


def fixture():
    shape = (32, 32)
    building = np.zeros(shape, dtype=bool)
    building[8:13, 10:17] = True
    indices = np.flatnonzero(building)
    view = AlgorithmView(shape, building, np.flatnonzero(~building), indices,
                         np.zeros(len(indices), dtype=np.float32), np.zeros(len(indices)))
    particles = np.random.default_rng(42).uniform(.1, .9, (6, *shape)).astype(np.float32)
    particles[:, building] = 0
    return view, particles


def test_measurement_oracle_and_deterministic_prefix():
    view, particles = fixture()
    calls = []
    def observe(index):
        calls.append(index)
        return .3 + .2 * (index % 32) / 31
    cfg = SamplingConfig(budgets=(2, 4), joint_source_hypotheses=8, joint_target_stride=8)
    trace = select_locations(view, particles, SourceCalibration.default(), observe, cfg)
    assert calls == trace.indices.tolist()
    assert len(set(calls)) == 4
    assert not view.building_mask.reshape(-1)[trace.indices].any()
    repeat = select_locations(view, particles, SourceCalibration.default(), observe, cfg)
    np.testing.assert_array_equal(trace.indices, repeat.indices)
    np.testing.assert_array_equal(trace.scores, repeat.scores)
    short = select_locations(view, particles, SourceCalibration.default(), observe,
                             SamplingConfig(budgets=(2,), joint_source_hypotheses=8, joint_target_stride=8))
    np.testing.assert_array_equal(short.indices, trace.indices[:2])
    assert np.all(trace.field_reduction >= 0) and np.all(trace.coordinate_reduction >= 0)
    np.testing.assert_allclose(trace.scores, trace.field_reduction + trace.coordinate_reduction)


def test_corrected_ensemble_and_projection():
    view, particles = fixture()
    coordinates = np.array([[4, 4], [24, 24]])
    indices = np.array([0, 31, 1023])
    values = np.array([.2, .8, 1.1], dtype=np.float32)
    result = reconstruct_from_candidates(particles.reshape(2, 3, *view.shape), coordinates,
        np.array([.5, .5]), indices, values, view.building_mask, np.zeros(view.shape))
    assert result.mean.shape == view.shape and np.isfinite(result.mean).all()
    assert np.all(result.weights >= 0) and 0 <= result.likelihood_power <= 1
    np.testing.assert_allclose(result.weights.sum(), 1)
    np.testing.assert_allclose(result.mode_probabilities, result.weights.reshape(2, 3).sum(axis=1))
    np.testing.assert_array_equal(result.mean.reshape(-1)[indices], np.clip(values, 0, 1))
    np.testing.assert_array_equal(result.mean[view.building_mask], 0)
    assert result.ess >= max(2., .35 * len(result.weights)) - 1e-9


def test_scene_partition_is_disjoint():
    splits = scene_splits()
    assert [len(splits[x]) for x in ('train', 'val', 'test')] == [500, 100, 100]
    assert not (set(splits['train']) & set(splits['test']))
    assert not (set(splits['val']) & set(splits['test']))
    assert 0 not in set(splits['train'] + splits['val'] + splits['test'])


def test_rejects_observation_inside_building():
    view, particles = fixture()
    with pytest.raises(ValueError, match='building'):
        reconstruct_from_candidates(particles.reshape(2, 3, *view.shape), np.array([[4,4], [24,24]]),
            np.array([.5, .5]), np.array([view.free_indices[0]]), np.array([.5]),
            view.building_mask, np.zeros(view.shape))
