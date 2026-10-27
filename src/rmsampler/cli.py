"""Portable, weights-excluded RadioMapSeer reproduction entry point."""
import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

from radiomap_dpm.data import load_radio_map, get_scene_ids
from radiomap_dpm.source_model import SourceCalibration, calibrate_source_model
from .selection import SamplingConfig, select_locations


def paired_seed(base: int, scene: int, tx: int, stream: str) -> int:
    streams = {"spatial_noise": 11, "selection_prior": 23, "reconstruction": 53, "random_design": 71}
    return int(np.random.SeedSequence([base, scene, tx, streams[stream], 0]).generate_state(1, dtype=np.uint32)[0])


def run(args):
    args.conditional_source = args.conditional_source.expanduser().resolve()
    for path in (args.pixel_checkpoint, args.conditional_checkpoint, args.calibration):
        if not path.is_file():
            raise FileNotFoundError(f"Required local input is missing: {path}")
    if not (args.conditional_source / "denoising_diffusion_pytorch/ddm_const_sde.py").is_file():
        raise FileNotFoundError("conditional source must contain denoising_diffusion_pytorch/ddm_const_sde.py")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve earlier runs")

    import torch
    from radiomap_dpm.model import load_model50_ema
    from radiomap_dpm.posterior import MeasurementConsistentSampler, PosteriorConfig
    from radiomap_dpm.radiodiff_original_adapter import load_radio_diff_original
    from radiomap_dpm.baselines import regular_grid_sampling, random_sampling
    from radiomap_dpm.metrics import compute_metrics, compute_peak_metrics
    from .reconstruction import reconstruct
    from .rng import respect_external_rng

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    scenes = args.scene if args.scene is not None else list(get_scene_ids(args.split))[:args.limit]
    config = SamplingConfig(budgets=tuple(sorted(set(args.budgets))))
    calibration = SourceCalibration.load(args.calibration)
    prior = load_model50_ema(args.pixel_checkpoint, device=device, sampling_timesteps=16)
    prior.requires_grad_(False)
    sampler = MeasurementConsistentSampler(prior)
    conditional = load_radio_diff_original(
        args.conditional_checkpoint, source_root=args.conditional_source, device=device, sampling_timesteps=3,
    )
    respect_external_rng(conditional)
    assert all(not p.requires_grad for p in prior.parameters())
    assert all(not p.requires_grad for p in conditional.parameters())
    frozen_versions = [(parameter, parameter._version)
                       for network in (prior, conditional) for parameter in network.parameters()]
    args.output.mkdir(parents=True)
    rows = []
    for scene in scenes:
        started = time.perf_counter()
        record = load_radio_map(args.data_root, scene, args.tx)
        view = record.algorithm_view()
        building_values = np.zeros(view.shape, dtype=np.float32)
        building_values.reshape(-1)[view.free_indices] = view.free_values
        truth = record.gain.reshape(-1)
        noise_seed = paired_seed(args.seed, scene, args.tx, "spatial_noise")
        noise = np.random.default_rng(noise_seed).standard_normal(truth.size).astype(np.float32) * np.float32(.02)
        # The evaluator exposes exactly one selected value at a time.
        def observe(index):
            return np.float32(truth[index] + noise[index])
        selection_seed = paired_seed(args.seed, scene, args.tx, "selection_prior")
        particles = sampler.sample(
            building_mask=np.array(view.building_mask, copy=True), building_values=building_values,
            config=PosteriorConfig(steps=16, n_particles=12, chunk_size=3, guidance_scale=.35, seed=selection_seed),
        )
        designs = {}
        if "rmsampler" in args.methods:
            trace = select_locations(view, particles, calibration, observe, config)
            designs["rmsampler"] = {b: trace.indices[:b] for b in config.budgets}
        if "uniform" in args.methods:
            designs["uniform"] = {b: regular_grid_sampling(view.building_mask, b) for b in config.budgets}
        if "random" in args.methods:
            indices = random_sampling(view.building_mask, max(config.budgets),
                seed=paired_seed(args.seed, scene, args.tx, "random_design"))
            designs["random"] = {b: indices[:b] for b in config.budgets}
        reconstruction_seed = paired_seed(args.seed, scene, args.tx, "reconstruction")
        for method, budgets in designs.items():
            for budget, indices in budgets.items():
                values = np.asarray([observe(int(i)) for i in indices], dtype=np.float32)
                result, candidates, coordinates, probabilities = reconstruct(
                    view, calibration, indices, values, conditional, reconstruction_seed,
                )
                # Reference RM and true Tx are used only after inference finishes.
                metrics = compute_metrics(result.mean, record.gain, record.building_mask)
                metrics.update(compute_peak_metrics(result.mean, record.gain, record.building_mask,
                    record.antenna_rc, selected_idx=indices, measurements=values))
                metrics["map_tx_error_pixels"] = float(np.linalg.norm(result.tx_coordinate - record.antenna_rc))
                scalars = {k: float(v) for k, v in metrics.items() if np.isscalar(v)}
                destination = args.output / f"scene{scene:03d}_tx{args.tx:02d}_B{budget:02d}_{method}.npz"
                np.savez_compressed(destination, prediction=result.mean, selected_indices=indices,
                    measurements=values, building_mask=view.building_mask, selection_particles=particles,
                    conditional_candidates=candidates, mode_coordinates=coordinates,
                    mode_prior=probabilities, posterior_weights=result.weights,
                    mode_posterior=result.mode_probabilities, estimated_tx=result.tx_coordinate,
                    likelihood_power=result.likelihood_power, ess=result.ess,
                    seed=args.seed, selection_seed=selection_seed, noise_seed=noise_seed,
                    reconstruction_seed=reconstruction_seed, scene=scene, tx=args.tx,
                    metric_names=np.asarray(sorted(scalars)), metric_values=[scalars[k] for k in sorted(scalars)],
                    python_version=sys.version, torch_version=str(torch.__version__), numpy_version=np.__version__,
                    device=torch.cuda.get_device_name() if device.startswith("cuda") else "cpu")
                rows.append({"scene": scene, "tx": args.tx, "method": method, "budget": budget,
                             "outdoor_nmse_db": scalars["outdoor_nmse_db"],
                             "map_tx_error_pixels": scalars["map_tx_error_pixels"]})
                print(f"scene={scene} method={method} measurements={budget} NMSE={scalars['outdoor_nmse_db']:.4f} dB", flush=True)
        print(f"scene={scene} elapsed={time.perf_counter() - started:.1f} s", flush=True)
    if any(parameter._version != version for parameter, version in frozen_versions):
        raise RuntimeError("A pretrained parameter was modified during inference")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="fit only the non-neural source model on training scenes")
    calibration.add_argument("--data-root", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--seed", type=int, default=42031)
    infer = commands.add_parser("run", help="sampling, conditional reconstruction, correction, projection, evaluation")
    for name in ("data-root", "pixel-checkpoint", "conditional-source", "conditional-checkpoint", "calibration", "output"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--split", choices=("train", "val", "test", "smoke"), default="test")
    infer.add_argument("--scene", type=int, nargs="+")
    infer.add_argument("--limit", type=int, default=1)
    infer.add_argument("--tx", type=int, default=0)
    infer.add_argument("--budgets", type=int, nargs="+", default=[8, 16])
    infer.add_argument("--methods", choices=("rmsampler", "uniform", "random"), nargs="+", default=["rmsampler", "uniform", "random"])
    infer.add_argument("--seed", type=int, default=42031)
    infer.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        if args.output.exists():
            raise FileExistsError("Choose a new calibration output path")
        model, metadata = calibrate_source_model(args.data_root, max_records=200,
            pixels_per_record=128, tx_ids=(0,), seed=args.seed)
        model.save(args.output, metadata=metadata, wall_time=0.)
        print(f"Training-only numerical calibration saved to {args.output}")
    else:
        if args.limit < 1:
            parser.error("--limit must be positive")
        run(args)


if __name__ == "__main__":
    main()
