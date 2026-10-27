"""Strict RadioMapSeer diffusion adapters for the released model-50 EMA.

The released ``model-50.pt`` network has four input channels even though the
radio map itself is single-channel.  The primary adapter binds a known binary
building environment to the network's three native condition channels.  The
posterior-facing facade then exposes no condition or transmitter argument, so
the environment cannot change during reconstruction and Tx remains unknown.
The earlier fixed-null unconditional adapter remains available for compatibility.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# The original architecture is included without its training harness.
DEFAULT_CHECKPOINT = Path("weights/pixel-prior.pt")
UNCONDITIONAL_FORMAT = "radiomap_dpm.unconditional.pred_v.v1"
from rmsampler.backbones.pixel_unet import GaussianDiffusion, Unet


class ModelPrediction(NamedTuple):
    pred_noise: Tensor
    pred_x_start: Tensor


def _torch_load(path: Path) -> Any:
    """Load tensor-only checkpoints on CPU, using mmap when supported."""
    kwargs = dict(map_location="cpu", weights_only=True)
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, **kwargs)


def _extract_source_ema(checkpoint: Mapping[str, Any]) -> OrderedDict[str, Tensor]:
    """Extract only ``ema_model`` from an ema-pytorch trainer checkpoint."""
    if "ema" not in checkpoint or not isinstance(checkpoint["ema"], Mapping):
        raise KeyError("model-50 checkpoint has no EMA state dictionary")
    ema = checkpoint["ema"]
    prefix = "ema_model."
    state = OrderedDict(
        (key[len(prefix) :], value)
        for key, value in ema.items()
        if key.startswith(prefix)
    )
    if not state:
        raise KeyError("EMA state contains no 'ema_model.*' tensors")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("EMA model state must contain tensors only")
    return state


class UnconditionalRadioDiffusion(nn.Module):
    """Fixed-null facade around the legacy conditional implementation.

    ``diffusion`` intentionally returns this facade instead of the underlying
    conditional object.  Thus posterior code can use
    ``prior.diffusion.model_predictions`` without gaining a back door through
    which scene buildings or Tx information could be passed.
    """

    CONDITION_MODE = "fixed_null"

    def __init__(
        self,
        *,
        image_size: int = 256,
        timesteps: int = 1000,
        sampling_timesteps: int = 50,
    ) -> None:
        super().__init__()
        unet = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            cond_channels=3,
            self_condition=False,
            flash_attn=False,
            image_size=image_size,
        )
        self._core = GaussianDiffusion(
            unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective="pred_v",
            beta_schedule="sigmoid",
        )
        self.checkpoint_report: dict[str, Any] = {}

    @property
    def diffusion(self) -> "UnconditionalRadioDiffusion":
        """Safe diffusion facade for posterior integrations."""
        return self

    @property
    def device(self) -> torch.device:
        return self._core.device

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(self._core.image_size)

    @property
    def alphas_cumprod(self) -> Tensor:
        return self._core.alphas_cumprod

    @property
    def betas(self) -> Tensor:
        return self._core.betas

    @property
    def num_timesteps(self) -> int:
        return self._core.num_timesteps

    @property
    def sampling_timesteps(self) -> int:
        return self._core.sampling_timesteps

    @property
    def objective(self) -> str:
        return self._core.objective

    @property
    def channels(self) -> int:
        return self._core.channels

    @property
    def condition_mode(self) -> str:
        return self.CONDITION_MODE

    def null_condition(
        self, batch: int, dtype: torch.dtype | None = None
    ) -> Tensor:
        """Return the sole legal legacy condition: an all-blank ``-1`` tensor."""
        if batch < 1:
            raise ValueError("batch must be positive")
        h, w = self.image_size
        dtype = dtype or self.betas.dtype
        return torch.full((batch, 3, h, w), -1.0, device=self.device, dtype=dtype)

    def load_model50_ema(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
        """Strictly load the EMA branch of the original trainer checkpoint."""
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        raw = _torch_load(path)
        if not isinstance(raw, Mapping):
            raise TypeError("model-50 checkpoint must be a mapping")
        state = _extract_source_ema(raw)

        first = state.get("model.init_conv.weight")
        if first is None or tuple(first.shape) != (64, 4, 7, 7):
            raise RuntimeError(
                "unexpected model-50 architecture: expected init weight (64,4,7,7)"
            )
        # With the source defaults and no min-SNR clipping, pred-v loss weights
        # equal alpha_bar.  Refuse a checkpoint trained for another objective.
        if "loss_weight" not in state or not torch.allclose(
            state["loss_weight"], state["alphas_cumprod"], rtol=1e-6, atol=1e-7
        ):
            raise RuntimeError("checkpoint is not compatible with pred_v continuation")

        # strict=True is deliberate: no missing/unexpected-key compatibility shim.
        incompatible = self._core.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:  # defensive
            raise RuntimeError(f"non-strict EMA load: {incompatible}")
        self.checkpoint_report = {
            "path": str(path),
            "source_step": int(raw.get("step", -1)),
            "source_version": str(raw.get("version", "unknown")),
            "branch": "ema_model",
            "strict": True,
            "tensor_count": len(state),
            "condition_mode": self.CONDITION_MODE,
            "objective": "pred_v",
        }
        return dict(self.checkpoint_report)

    def load_core_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Strict resume hook for checkpoints written by train_unconditional.py."""
        incompatible = self._core.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"non-strict resume load: {incompatible}")

    def load_continuation_ema(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Strictly load the EMA branch produced by ``train_unconditional.py``."""

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"continuation checkpoint not found: {path}")
        payload = _torch_load(path)
        if not isinstance(payload, Mapping) or payload.get("format") != UNCONDITIONAL_FORMAT:
            raise RuntimeError(f"not a {UNCONDITIONAL_FORMAT} checkpoint: {path}")
        metadata = payload.get("metadata", {})
        if metadata.get("condition_mode") != self.CONDITION_MODE:
            raise RuntimeError("checkpoint is not fixed-null unconditional")
        if metadata.get("objective") != "pred_v":
            raise RuntimeError("checkpoint objective is not pred_v")
        state = payload.get("ema_model")
        if not isinstance(state, Mapping) or not state:
            raise KeyError("continuation checkpoint has no EMA model state")
        self.load_core_state_dict(state)
        self.checkpoint_report = {
            "path": str(path),
            "global_step": int(payload.get("global_step", -1)),
            "continuation_steps": int(payload.get("continuation_steps", -1)),
            "branch": "ema_model",
            "strict": True,
            "tensor_count": len(state),
            "condition_mode": self.CONDITION_MODE,
            "objective": "pred_v",
        }
        return dict(self.checkpoint_report)

    def core_state_dict(self) -> OrderedDict[str, Tensor]:
        """Return the complete diffusion state for resumable checkpoints."""
        return self._core.state_dict()

    def model_predictions(
        self,
        x: Tensor,
        t: Tensor,
        *,
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
    ) -> ModelPrediction:
        """Predict noise and x0 with the immutable null condition."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("x must have shape [batch, 1, height, width]")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape [batch]")
        prediction = self._core.model_predictions(
            x,
            t,
            cond=self.null_condition(x.shape[0], x.dtype),
            clip_x_start=clip_x_start,
            rederive_pred_noise=rederive_pred_noise,
        )
        return ModelPrediction(prediction.pred_noise, prediction.pred_x_start)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        return self._core.q_sample(x_start=x_start, t=t, noise=noise)

    def training_loss(
        self,
        images_01: Tensor,
        *,
        t: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Weighted pred-v continuation loss for real DPM images in [0,1]."""
        if images_01.ndim != 4 or images_01.shape[1] != 1:
            raise ValueError("images_01 must have shape [batch,1,height,width]")
        if tuple(images_01.shape[-2:]) != self.image_size:
            raise ValueError(f"expected spatial size {self.image_size}")
        if not torch.isfinite(images_01).all():
            raise ValueError("images contain NaN or Inf")
        if images_01.detach().amin() < 0 or images_01.detach().amax() > 1:
            raise ValueError("DPM images must be normalized to [0,1]")

        batch = images_01.shape[0]
        x_start = images_01 * 2.0 - 1.0
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch,), device=self.device)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        pred_v = self._core.model(
            x_t, t, cond=self.null_condition(batch, x_t.dtype)
        )
        target_v = self._core.predict_v(x_start, t, noise)
        per_sample = F.mse_loss(pred_v, target_v, reduction="none").flatten(1).mean(1)
        weights = self._core.loss_weight.gather(0, t)
        return (per_sample * weights).mean()

    def forward(self, images_01: Tensor) -> Tensor:
        return self.training_loss(images_01)

    @torch.inference_mode()
    def sample(self, batch_size: int, *, return_all_timesteps: bool = False) -> Tensor:
        """Draw [0,1] radio-map samples; no conditional argument is accepted."""
        return self._core.sample(
            batch_size=batch_size,
            cond=self.null_condition(batch_size, self.betas.dtype),
            return_all_timesteps=return_all_timesteps,
        )


class BoundBuildingRadioDiffusion:
    """Immutable scene facade exposing no condition or transmitter argument.

    Instances are created only by
    :meth:`BuildingConditionedRadioDiffusion.bind_environment`.  The facade
    keeps a normalized three-channel building tensor and supplies it internally
    on every denoiser call.  Posterior code therefore cannot accidentally swap
    environments midway through a reverse trajectory.
    """

    __slots__ = ("_prior", "_condition")

    def __init__(
        self, prior: "BuildingConditionedRadioDiffusion", condition: Tensor
    ) -> None:
        self._prior = prior
        self._condition = condition.detach()

    @property
    def diffusion(self) -> "BoundBuildingRadioDiffusion":
        """Safe posterior-facing diffusion facade."""
        return self

    @property
    def device(self) -> torch.device:
        return self._prior.device

    @property
    def image_size(self) -> tuple[int, int]:
        return self._prior.image_size

    @property
    def alphas_cumprod(self) -> Tensor:
        return self._prior.alphas_cumprod

    @property
    def betas(self) -> Tensor:
        return self._prior.betas

    @property
    def num_timesteps(self) -> int:
        return self._prior.num_timesteps

    @property
    def sampling_timesteps(self) -> int:
        return self._prior.sampling_timesteps

    @property
    def objective(self) -> str:
        return self._prior.objective

    @property
    def channels(self) -> int:
        return self._prior.channels

    @property
    def condition_mode(self) -> str:
        return self._prior.CONDITION_MODE

    def model_predictions(
        self,
        x: Tensor,
        t: Tensor,
        *,
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
    ) -> ModelPrediction:
        """Predict with the building environment fixed at facade construction."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("x must have shape [batch, 1, height, width]")
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(f"expected spatial size {self.image_size}")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape [batch]")
        condition = self._condition.to(device=x.device, dtype=x.dtype)
        if condition.shape[0] == 1:
            condition = condition.expand(x.shape[0], -1, -1, -1)
        elif condition.shape[0] != x.shape[0]:
            if x.shape[0] % condition.shape[0] != 0:
                raise ValueError(
                    "batched building condition must divide the diffusion batch size"
                )
            condition = condition.repeat_interleave(
                x.shape[0] // condition.shape[0], dim=0
            )
        prediction = self._prior._core.model_predictions(
            x,
            t,
            cond=condition,
            clip_x_start=clip_x_start,
            rederive_pred_noise=rederive_pred_noise,
        )
        return ModelPrediction(prediction.pred_noise, prediction.pred_x_start)


class BuildingConditionedRadioDiffusion(UnconditionalRadioDiffusion):
    """Native model-50 prior that must be bound to one known building map.

    The inheritance is architecture/checkpoint reuse only.  Unbound inference,
    null conditioning, unconditional training and unconditional sampling are
    disabled so that a building-conditioned run cannot silently fall back to
    the former fixed-null route.
    """

    CONDITION_MODE = "building_conditioned"

    def bind_environment(self, building_mask: Any) -> BoundBuildingRadioDiffusion:
        """Bind a binary ``H x W`` mask as three channels mapped to ``[-1,1]``."""
        mask = torch.as_tensor(building_mask)
        if mask.ndim not in (2, 3) or tuple(mask.shape[-2:]) != self.image_size:
            raise ValueError(
                "building_mask must have shape "
                f"{self.image_size} or [batch,{self.image_size[0]},{self.image_size[1]}]"
            )
        if mask.dtype == torch.bool:
            binary = mask
        else:
            if not torch.isfinite(mask).all():
                raise ValueError("building_mask contains NaN or Inf")
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("building_mask must be binary with values 0/1")
            binary = mask.to(torch.bool)
        normalized = binary.to(device=self.device, dtype=self.betas.dtype)
        normalized = normalized.mul(2.0).sub(1.0)
        if normalized.ndim == 2:
            normalized = normalized.unsqueeze(0)
        condition = normalized.unsqueeze(1).repeat(1, 3, 1, 1)
        return BoundBuildingRadioDiffusion(self, condition.contiguous())

    def null_condition(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError(
            "building-conditioned prior has no null condition; call bind_environment(mask)"
        )

    def model_predictions(self, *args: Any, **kwargs: Any) -> ModelPrediction:
        raise RuntimeError(
            "bind the known building environment before inference: "
            "prior.bind_environment(building_mask)"
        )

    def training_loss(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("unconditional continuation training is disabled for this route")

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("use a scene-bound facade for building-conditioned inference")

    def sample(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("bind the known building environment before sampling")


def load_model50_ema(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | torch.device | None = None,
    sampling_timesteps: int = 50,
) -> UnconditionalRadioDiffusion:
    """Build the exact network, strictly load source EMA, and move it to device."""
    prior = UnconditionalRadioDiffusion(sampling_timesteps=sampling_timesteps)
    prior.load_model50_ema(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return prior.to(device)


def load_building_conditioned_ema(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | torch.device | None = None,
    sampling_timesteps: int = 50,
) -> BuildingConditionedRadioDiffusion:
    """Strictly load the native model-50 EMA for scene-bound inference."""
    prior = BuildingConditionedRadioDiffusion(
        sampling_timesteps=sampling_timesteps
    )
    prior.load_model50_ema(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return prior.to(device)


def load_unconditional_ema(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    sampling_timesteps: int = 50,
) -> UnconditionalRadioDiffusion:
    """Load the fine-tuned fixed-null EMA used for formal reconstruction."""

    prior = UnconditionalRadioDiffusion(sampling_timesteps=sampling_timesteps)
    prior.load_continuation_ema(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return prior.to(device)


__all__ = [
    "BoundBuildingRadioDiffusion",
    "BuildingConditionedRadioDiffusion",
    "DEFAULT_CHECKPOINT",
    "ModelPrediction",
    "UNCONDITIONAL_FORMAT",
    "UnconditionalRadioDiffusion",
    "load_building_conditioned_ema",
    "load_model50_ema",
    "load_unconditional_ema",
]
