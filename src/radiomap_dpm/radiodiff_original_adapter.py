"""Original RadioDiff conditional latent-diffusion adapter for sparse RM recovery.

This module is deliberately isolated from the validated active-sampling path.  It
loads the released ``model-dpm.pt`` checkpoint with the original RadioDiff
implementation and uses the three condition channels as

``[building, signed_sparse_value, signed_observation_mask]``.

The second stage is therefore a conditional generative posterior, rather than a
hand-written source-kernel correction.  The released DPM checkpoint was trained
with ``[building, Tx-antenna raster, building]``; at inference the antenna raster
is replaced by a measurement-derived Tx posterior.  Known building pixels and
paid samples are projected back exactly after decoding.  The adapter never reads
Tx metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_RADIO_DIFF_ORIGINAL = _THIS_DIR.parents[1] / "RadioDiff-original"


def _import_radio_diff_original(source_root: str | Path | None = None):
    root = Path(source_root or _DEFAULT_RADIO_DIFF_ORIGINAL).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"original RadioDiff source directory not found: {root}")
    # The active-sampling package imports the pip ``denoising_diffusion_pytorch``
    # first.  Remove that cached package so the original RadioDiff modules,
    # which provide ``ddm_const_sde`` and ``encoder_decoder``, are authoritative.
    for name in list(sys.modules):
        if name == "denoising_diffusion_pytorch" or name.startswith(
            "denoising_diffusion_pytorch."
        ):
            del sys.modules[name]
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    from fvcore.common.config import CfgNode
    from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
    from denoising_diffusion_pytorch.mask_cond_unet import Unet

    return CfgNode, LatentDiffusion, AutoencoderKL, Unet


def _default_config() -> dict[str, Any]:
    # These values are the released original RadioDiff conditional architecture.
    # The FFT block
    # shape (8 x 5 in the checkpoint) corresponds to a 256 x 256 input.
    return {
        "model_name": "cond_unet",
        "image_size": [320, 320],
        "timesteps": 1000,
        "sampling_timesteps": 3,
        "loss_type": "l2",
        "objective": "pred_KC",
        "scale_factor": 0.3,
        "scale_by_std": True,
        "default_scale": True,
        "scale_by_softsign": False,
        "input_keys": ["image", "cond"],
        "start_dist": "normal",
        "perceptual_weight": 0,
        "use_l1": True,
        "eps": 1e-4,
        "weighting_loss": False,
        "first_stage": {
            "embed_dim": 3,
            "lossconfig": {
                "disc_start": 50001,
                "kl_weight": 1e-6,
                "disc_weight": 0.5,
                "disc_in_channels": 1,
            },
            "ddconfig": {
                "double_z": True,
                "z_channels": 3,
                "resolution": [320, 320],
                "in_channels": 1,
                "out_ch": 1,
                "ch": 128,
                "ch_mult": [1, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
        },
        "unet": {
            "dim": 128,
            "cond_net": "swin",
            "without_pretrain": True,
            # Match configs_old/BSDS_train_DPM.yaml.  This flag only controls
            # whether the pretrained Swin condition encoder is frozen.
            "fix_bb": False,
            "channels": 3,
            "out_mul": 1,
            "dim_mults": [1, 2, 4, 4],
            "cond_in_dim": 3,
            "cond_dim": 128,
            "cond_dim_mults": [2, 4],
            "window_sizes1": [[8, 8], [4, 4], [2, 2], [1, 1]],
            "window_sizes2": [[8, 8], [4, 4], [2, 2], [1, 1]],
            "fourier_scale": 16,
            "cond_pe": False,
            "num_pos_feats": 128,
            "cond_feature_size": [80, 80],
            "input_size": [80, 80],
            "DPMCARK": False,
        },
    }


def load_radio_diff_original(
    checkpoint: str | Path,
    *,
    source_root: str | Path | None = None,
    device: str | torch.device = "cuda",
    sampling_timesteps: int = 3,
):
    """Construct the original RadioDiff LDM and strictly restore its state."""

    CfgNode, LatentDiffusion, AutoencoderKL, Unet = _import_radio_diff_original(source_root)
    raw_cfg = _default_config()
    raw_cfg["sampling_timesteps"] = int(sampling_timesteps)
    cfg = CfgNode(raw_cfg)
    first_cfg = cfg.first_stage
    first_stage = AutoencoderKL(
        ddconfig=first_cfg.ddconfig,
        lossconfig=first_cfg.lossconfig,
        embed_dim=int(first_cfg.embed_dim),
    )
    unet_cfg = cfg.unet
    unet = Unet(
        dim=int(unet_cfg.dim),
        channels=int(unet_cfg.channels),
        dim_mults=tuple(int(x) for x in unet_cfg.dim_mults),
        learned_variance=False,
        out_mul=int(unet_cfg.out_mul),
        cond_in_dim=int(unet_cfg.cond_in_dim),
        cond_dim=int(unet_cfg.cond_dim),
        cond_dim_mults=tuple(int(x) for x in unet_cfg.cond_dim_mults),
        window_sizes1=[list(map(int, x)) for x in unet_cfg.window_sizes1],
        window_sizes2=[list(map(int, x)) for x in unet_cfg.window_sizes2],
        fourier_scale=float(unet_cfg.fourier_scale),
        cfg=unet_cfg,
        carsDPM=bool(unet_cfg.DPMCARK),
    )
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage,
        train_sample=-1,
        image_size=cfg.image_size,
        timesteps=int(cfg.timesteps),
        sampling_timesteps=int(cfg.sampling_timesteps),
        loss_type=str(cfg.loss_type),
        objective=str(cfg.objective),
        scale_factor=float(cfg.scale_factor),
        scale_by_std=bool(cfg.scale_by_std),
        scale_by_softsign=bool(cfg.scale_by_softsign),
        default_scale=bool(cfg.default_scale),
        input_keys=list(cfg.input_keys),
        start_dist=str(cfg.start_dist),
        perceptual_weight=float(cfg.perceptual_weight),
        use_l1=bool(cfg.use_l1),
        cfg=cfg,
    )
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Original RadioDiff checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("RadioDiff checkpoint must contain a mapping under 'model'")
    state = payload["model"]
    incompatible = ldm.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "RadioDiff checkpoint mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    ldm.eval().to(device)
    for parameter in ldm.parameters():
        parameter.requires_grad_(False)
    ldm.original_radio_diff_report = {
        "checkpoint": str(path),
        "state_keys": len(state),
        "sampling_timesteps": int(sampling_timesteps),
        "input_shape": [256, 256],
        "condition_channels": ["building", "measurement-derived Tx posterior raster", "building"],
    }
    return ldm


def make_sparse_condition(
    building_mask: np.ndarray,
    measured_indices: np.ndarray,
    measured_values: np.ndarray,
    building_values: np.ndarray | None = None,
    third_channel: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the normalized condition and exact-known projection tensors."""

    building = np.asarray(building_mask, dtype=bool)
    if building.ndim != 2:
        raise ValueError("building_mask must be [H,W]")
    h, w = building.shape
    idx = np.asarray(measured_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(measured_values, dtype=np.float32).reshape(-1)
    if idx.size != values.size or np.any(idx < 0) or np.any(idx >= h * w):
        raise ValueError("measured indices and values are inconsistent")
    value01 = np.zeros(h * w, dtype=np.float32)
    observed = np.zeros(h * w, dtype=np.float32)
    value01[idx] = np.clip(values, 0.0, 1.0)
    observed[idx] = 1.0
    # The original DPM loader uses [building, antenna, building], with all
    # channels normalized from [0,1] to [-1,1].  ``third_channel`` is retained
    # as a compatibility name for the measurement-derived Tx posterior raster.
    if third_channel is None:
        source = observed
    else:
        source = np.asarray(third_channel, dtype=np.float32)
        if source.shape != (h, w):
            raise ValueError("third_channel must have shape [H,W]")
        source = np.clip(source, 0.0, 1.0)
    cond = np.stack(
        [building.ravel().astype(np.float32), source.ravel(), building.ravel().astype(np.float32)],
        axis=0,
    ).reshape(3, h, w) * 2.0 - 1.0
    known_mask = building.ravel() | (observed > 0)
    known_values = np.zeros(h * w, dtype=np.float32)
    if building_values is not None:
        free = np.asarray(building_values, dtype=np.float32).reshape(-1)
        if free.size != h * w:
            raise ValueError("building_values must have the same flattened size as the map")
        known_values[building.ravel()] = free[building.ravel()]
    known_values[idx] = value01[idx]
    return (
        torch.from_numpy(cond[None]),
        torch.from_numpy(known_mask.reshape(1, 1, h, w)),
        torch.from_numpy(known_values.reshape(1, 1, h, w)),
    )


@torch.inference_mode()
def sample_sparse_posterior(
    ldm: Any,
    condition: torch.Tensor,
    *,
    n_samples: int = 4,
    known_mask: torch.Tensor | None = None,
    known_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a batched conditional posterior and return samples plus mean."""

    if condition.ndim != 4 or condition.shape[1] != 3:
        raise ValueError("condition must have shape [1,3,H,W]")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    device = next(ldm.parameters()).device
    cond = condition.to(device=device, dtype=torch.float32).repeat(n_samples, 1, 1, 1)
    samples = ldm.sample(batch_size=n_samples, cond=cond, denoise=True)
    if samples.ndim != 4 or samples.shape[1] != 1:
        raise RuntimeError(f"unexpected RadioDiff output shape {tuple(samples.shape)}")
    if known_mask is not None and known_values is not None:
        mask = known_mask.to(device=device, dtype=samples.dtype)
        values = known_values.to(device=device, dtype=samples.dtype)
        samples = samples * (1.0 - mask) + values * mask
    return samples, samples.mean(dim=0, keepdim=True)


def _smoke(args: argparse.Namespace) -> None:
    from PIL import Image

    def _read(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)

    gain = _read(args.root / "gain" / "DPM" / f"{args.scene}_{args.tx}.png").astype(
        np.float32
    ) / 255.0
    building = _read(args.root / "png" / "buildings_complete" / f"{args.scene}.png") == 255
    # Real-data smoke: use a deterministic sparse subset of outdoor pixels.
    outdoor = np.flatnonzero(~building.ravel())
    measured = outdoor[:: max(1, outdoor.size // 8)][:8]
    condition, known_mask, known_values = make_sparse_condition(
        building, measured, gain.ravel()[measured], building_values=gain
    )
    ldm = load_radio_diff_original(
        args.checkpoint,
        source_root=args.source_root,
        device=args.device,
        sampling_timesteps=args.steps,
    )
    samples, mean = sample_sparse_posterior(
        ldm,
        condition,
        n_samples=1,
        known_mask=known_mask,
        known_values=known_values,
    )
    if not torch.isfinite(samples).all():
        raise RuntimeError("Original RadioDiff smoke output contains NaN/Inf")
    print(
        "Smoke test PASSED",
        "method=RadioDiff-original-measurement-conditioned",
        f"scene={args.scene}",
        f"tx={args.tx}",
        f"shape={tuple(mean.shape)}",
        f"range=({float(mean.min()):.4f},{float(mean.max()):.4f})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original RadioDiff sparse posterior smoke test")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if not parsed.smoke:
        raise SystemExit("pass --smoke for the real-data smoke test")
    _smoke(parsed)
