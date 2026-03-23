# Solving the Elastic Wave Equation with Physics-Informed Neural Networks

This repository is the code companion for the ETH Zurich Master Thesis:

- Davide Staub (2024), *Solving the Elastic Wave Equation with Physics-Informed Neural Networks: A Robust and Critical Assessment*
- Publication page: https://www.research-collection.ethz.ch/entities/publication/2f44e380-2a96-48ed-92e9-4dfbfff30e40
- DOI: https://doi.org/10.3929/ethz-b-000668359

## What this repository is for

The project investigates Physics-Informed Neural Networks (PINNs) for seismic wave simulation, with a focus on the elastic wave equation and applications in seismology.

The thesis evaluates:

- baseline PINN behavior across constant and heterogeneous parameter settings,
- architecture changes that inject wave-physics priors,
- conditioning on seismic source location to improve practical inference speed.

## Main findings reflected in this codebase

- PINNs are promising for elastic wave simulation, but standard architectures can suffer from stability and accuracy limitations in harder settings.
- Architectures that incorporate wave-physics structure (notably encoder-decoder variants with wavelet/plane-wave style components) improve robustness and accuracy compared with vanilla PINNs.
- Conditioning PINNs on source location enables one trained model to infer many source scenarios, giving large speed advantages for multi-source tasks such as seismic hazard style evaluations.

## Repository contents (high level)

- `*.py`: training, inference, model architecture, plotting, and evaluation scripts.
- `*.sh`: run/submit scripts for experiments.

Large experiment artifacts and non-code outputs are intentionally excluded from version control in this repository.

## Citation

If you use this repository, please cite the thesis:

```text
Staub, Davide. Solving the Elastic Wave Equation with Physics-Informed Neural Networks:
A Robust and Critical Assessment. Master Thesis, ETH Zurich, 2024.
DOI: 10.3929/ethz-b-000668359
```
