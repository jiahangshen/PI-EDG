# Implementation notes

This repository keeps the numerical behavior of the trained PI-EDG voxel model while removing unused experimental branches. A few details are more specific than the compact equations in the manuscript:

- The log-density guide is supervised where the reference density exceeds `1e-10`. Its routing gradient is scaled by `routing_backbone_grad_scale` (default `1.0`), rather than always detached.
- The router stores a finite lower base value of `-35`; its first internal threshold is initialized at `-34`.
- The fused cube-root output passes through a sigmoid mask before cubing. This gate is part of the trained model and is retained here.
- Atom Gaussian inputs use voxel centers, while the nuclear-potential field uses inclusive box endpoints.

Electron-number normalization is applied after inference, not during training. The supplied 2,000 evaluation IDs are part of the provided validation split, not an independent test set.
