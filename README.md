# RMSampler

Minimal reproducible implementation of **RMSampler: Sparse Radio Map
Reconstruction with Active Sampling and Diffusion Posterior Correction**.

[Paper preprint](https://arxiv.org/abs/2610.27464)

RMSampler optimizes the next measurement location by jointly estimating the
reduction in global RM reconstruction error and unknown-transmitter localization
error. Acquired measurements update both estimates. A separate frozen conditional
diffusion ensemble is then corrected by a robust likelihood and projected onto
the acquired measurements and known building values.

**No pretrained weights or datasets are included. No pretrained neural parameters
or architecture are fine-tuned or changed.**

## Install and test the numerical core

Use Python 3.11. The core requires NumPy, SciPy, and Pillow.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
rmsampler --help
```

The tests use explicitly labeled deterministic software fixtures. They do not
claim to reproduce paper performance without the dataset and pretrained models.

## Prepare the original model dependencies and data

Install the inference dependencies in a compatible PyTorch/CUDA environment.
Keep an existing compatible PyTorch build when one is already installed.

```bash
python -m pip install -e '.[inference]'
git clone https://github.com/UNIC-Lab/RadioDiff.git external/RadioDiff
```

The original pixel-prior inference architecture is included under
`rmsampler/backbones`, with its upstream license and attribution. The conditional
model uses the RadioDiff checkout above. Follow its model-download instructions
and keep checkpoints outside this repository. The conditional source must contain
`denoising_diffusion_pytorch/ddm_const_sde.py`. The loader
checks compatibility with the expected architecture. The required checkpoint
roles are the pixel-domain velocity-prediction EMA prior and the original
conditional latent-diffusion DPM prior, including its autoencoder.
The pixel checkpoint must contain the original EMA branch and a four-channel
input convolution of shape `(64, 4, 7, 7)`. A generic ImageNet checkpoint is not
compatible. Checkpoint distribution remains separate from this code release.
The external RadioDiff implementation can also initialize its standard perceptual
loss dependencies from the upstream model cache. These are not repository assets.

Obtain the authentic [RadioMapSeer dataset](https://github.com/RonLevie/RadioMapSeer).
The data root must contain `gain/DPM`, `png/buildings_complete`, `png/antennas`,
and `Antenna`. The Linux case variant `antenna` is also supported for metadata.
Gain PNGs, building rasters, and antenna metadata retain their original values.

## Run the complete measurement-to-reconstruction loop

First reproduce the training-only numerical propagation calibration. This
fits no neural parameters and uses no validation or test labels.

```bash
rmsampler calibrate --data-root /path/to/RadioMapSeer \
  --output outputs/calibration.npz
```

Run a real-data smoke case, using both user-supplied checkpoints.

```bash
rmsampler run --data-root /path/to/RadioMapSeer \
  --pixel-checkpoint /path/to/pixel-prior.pt \
  --conditional-source external/RadioDiff \
  --conditional-checkpoint /path/to/conditional-prior.pt \
  --calibration outputs/calibration.npz \
  --scene 0 --budgets 8 --methods rmsampler uniform random \
  --output outputs/smoke
```

Use `--split test --limit 100 --budgets 8 16` instead of `--scene 0` for the
100-scene test protocol. Use a fresh output directory for every run. All three
sampling policies share the conditional reconstructor, noise field, and latent
random streams. The benchmark saves predictions, candidate ensembles, posterior
weights, measurement sets, seeds, and numerical metrics.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact split, scientific
settings, output schema, metric definitions, and the scope of the minimal
benchmark. The bundled pixel-prior architecture retains its original license,
and the external conditional model remains under the RadioDiff license. RadioPath
is not included.
