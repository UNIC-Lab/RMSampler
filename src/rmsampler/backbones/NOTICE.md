The pixel-prior inference definitions derive from the MIT-licensed
lucidrains/denoising-diffusion-pytorch implementation by Phil Wang, with the
RadioDiff condition-input adaptation used by the pretrained checkpoint.

Upstream: https://github.com/lucidrains/denoising-diffusion-pytorch

The original Unet and GaussianDiffusion class definitions are retained. The
unrelated training harness and image-folder dataset are omitted. No pretrained
weights are included. This inference release does not train or change the neural
network architecture, parameters, or buffers.
