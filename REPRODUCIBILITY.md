# Reproduction protocol

The public implementation includes the complete RMSampler-specific chain from
sequential measurement selection through conditional reconstruction, robust
posterior correction, exact projection, and evaluation. RadioPath is a separate
project and is not distributed here.

## Data and information access

- RadioMapSeer dominant-path maps have shape 256 by 256. Gain PNGs are divided
  by 255. Building pixels are 255 in the binary building raster.
- A `RandomState(42)` permutation of scenes 1 through 700 defines 500 training,
  100 validation, and 100 test scenes. Scene 0 is reserved for smoke tests.
- There are 80 transmitter configurations per scene. A case contains one
  transmitter, and the reported 100-scene comparison uses configuration 0.
- The algorithm receives building information and acquired values. True antenna
  coordinates and unmeasured outdoor gains remain evaluator-side.
- One scene-specific Gaussian noise field with standard deviation 0.02 is shared
  by the three policies and both budgets. Sampling does not reveal the noise-free
  value separately.

## Fixed settings

| Component | Setting |
|---|---|
| Sampling prior | Frozen pixel-domain EMA diffusion, 12 particles, 16 reverse steps |
| Pixel-prior observation handling | Exact known-building projection, guidance scale 0.35, batches of 3 |
| Candidate points | Outdoor stride-4 grid plus stride-2 neighborhoods extending 20 pixels around up to 8 Tx modes |
| RM acquisition target | Outdoor stride-4 grid |
| Tx posterior | Initial cell width 16, at most 512 hypotheses, finest cell width 2, uniform mass 0.1 |
| Tx acquisition representation | Spatially ordered probability-quantile coreset with at most 96 representatives |
| Sampling likelihood | Effective variance 0.02 squared plus 0.1 squared, likelihood power 0.25 |
| Sampling weight safeguard | Minimum ESS fraction 0.35, smallest uniform contamination, 64 bisections |
| Final Tx conditions | At most 4 modes separated by 8 pixels; Gaussian raster standard deviation 1 pixel |
| Final conditional prior | Frozen latent diffusion, 4 particles per mode, 3 reverse steps |
| Robust correction | Student-t with 4 degrees of freedom and discrepancy standard deviation 0.1 |
| Correction ESS | Target max(2, 0.35 times particle count), capped at particle count; 40 bisections on likelihood power [0,1] |
| Projection | Acquired values and known building values, clipped to [0,1] |

Source-model calibration is numerical, not neural training. It uses the first
200 eligible training cases with transmitter configuration 0 and 128 randomly
selected outdoor pixels per case, with seed 42031. Features are an intercept,
log-distance, and the building fraction along 12 line samples. Huber regression
fits three propagation coefficients. The residual scale is a median absolute
deviation estimate with lower bound 0.02. Calibration must precede test inference.

## Randomness and matched controls

The base seed is 42031. NumPy `SeedSequence` receives the base seed, scene,
transmitter configuration, a stream identifier, and a final zero. Stream
identifiers are 11 for measurement noise, 23 for the sampling ensemble, 53 for
conditional reconstruction, and 71 for Random sampling. Conditional modes are
generated in separate four-particle batches with a seed increment of 104729.
The RNG wrapper removes only literal seed-zero resets in the upstream sampling
loop. It does not change neural forward functions, architecture, parameters,
or buffers.

RMSampler and Random inherit the first eight measurements of their sixteen-point
sequences. Uniform constructs its deterministic geometry-specific grid separately
for each budget. All three use the same conditional reconstruction function.
Acquired coordinates and values are saved so selection and reconstruction can
also be inspected separately.

## Outputs and metrics

Each output NPZ records the selected locations, acquired values, original
sampling particles, conditional candidates, Tx modes, corrected weights,
likelihood power, ESS, prediction, seeds, and software/device information.
`metrics.csv` is a compact summary. Outputs are ignored by Git.

Outdoor NMSE is computed before conversion to dB. Per-scene dB values are
averaged for the paper, rather than converting the average linear NMSE.
Localization uses the corrected Tx-mode MAP estimate. True antenna coordinates
are used only for evaluation after inference. Full comparison statistics require
the complete 100-scene run. Numerical fixture tests check software behavior and
are not experimental performance evidence.

The measured-data protocols, their separate reconstruction and sampling cohorts,
and zero-shot inference settings are specified in the manuscript. The minimal
command-line benchmark here targets the RadioMapSeer closed loop; it does not
substitute RadioMapSeer outputs for real-world measured-data results.

## Verified release checks

The release was checked with Python 3.11.15, NumPy 2.2.6, PyTorch
2.12.1 with CUDA 13.2, and an NVIDIA RTX 3060 Ti. The five numerical tests
passed. Training-only calibration reproduced the existing experimental
calibration. An authentic RadioMapSeer scene-0 run completed eight measurements
for RMSampler, Uniform, and Random with the common conditional reconstructor.
RMSampler's selected locations, acquisition scores, and Tx posterior entropy
matched the existing experimental controller exactly. Repeating the complete
RMSampler run with the same seed reproduced the acquired values, both candidate
ensembles, corrected weights, and final RM exactly. These checks validate the
minimal release and are separate from the paper's complete evaluation cohorts.
