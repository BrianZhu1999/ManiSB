# ManiSB

Code, data, and numerical studies for manifold-aware conditional diffusion bridges.

[Experiments](#experiments) · [Getting started](#getting-started) · [Usage guide](docs/experiments.md) · [中文](README_zh.md)

## Overview

How does the geometry of paired data enter a diffusion bridge, and what allows
a model to learn that geometry? ManiSB studies these questions through a
decomposition of the bridge prediction into three components:

- **Normal contraction:** attraction from a noisy state toward the intermediate
  bridge manifold.
- **Endpoint transport:** motion from the projected bridge point toward its
  decoded endpoint.
- **Posterior correction:** the difference between the decoded endpoint and the
  conditional endpoint mean.

For smooth endpoint couplings, the paper characterizes the geometry induced
along the bridge, the component representations selected by different
supervision schemes, and the stability of projection-assisted sampling.
The numerical studies examine how learned geometric correction changes
endpoint recovery and the paths taken between paired endpoints.

## Experiments

### Smooth closed coupling

A shared latent angle pairs a three-lobed target curve with a compressed,
rotated terminal curve. The same three-head model is sampled with and without
geometric correction to examine endpoint error and distance to the evolving
bridge support.

[![Smooth closed coupling: paired curves, reverse paths, endpoint error, and support distance](figures/paper/fig_smooth_closed_coupling.png)](figures/paper/fig_smooth_closed_coupling.pdf)

[Data](data/case1_smooth_closed/) · [Code](code/case1_smooth_closed/) · [Configuration](configs/case1_swirl.json)

### Curved multi-branch coupling

A two-branch coupling introduces distinct transport paths. The experiment
compares a single-head model, a three-head model, and the three-head model with
learned geometric correction across 4,096 paired evaluations.

[![Curved multi-branch coupling: endpoint branches, reverse paths, and error distributions](figures/paper/fig3_multibranch_linear_v5.png)](figures/paper/fig3_multibranch_linear_v5.pdf)

[Data](data/case2_curved_multibranch/) · [Code](code/case2_curved_multibranch/) · [Results](results/case2_curved_multibranch/metrics_summary.json)

Model sizes, training settings, metric definitions, and result tables are in
the [experiment guide](docs/experiments.md).

## Getting started

The recorded environment uses Python 3.12.2 and PyTorch 2.6.0. Create an
environment and install the dependencies:

```bash
git clone https://github.com/BrianZhu1999/ManiSB.git
cd ManiSB
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`. For GPU training, install a PyTorch build
compatible with your CUDA driver.

Check the included data and figure files:

```bash
python code/verify_project.py --root .
python code/case1_smooth_closed/verify_exports.py --root .
python code/case2_curved_multibranch/verify_case2_package.py --root .
```

These checks run on CPU. Training commands and output locations are described
in the [usage guide](docs/experiments.md#training).

## Data and code

Both cases use synthetic endpoint couplings. The repository includes endpoint
pairs, sampled trajectories, analytic support samples, per-sample metrics,
and the figures shown above.

| Directory | Contents |
| --- | --- |
| [code/](code/) | Model, training, sampling, export, and verification source |
| [configs/](configs/) | Model and evaluation settings |
| [data/](data/) | Trajectory archives and per-sample metrics |
| [results/](results/) | Numerical summaries and verification records |
| [figures/](figures/) | Figures in PNG, SVG, PDF, and TIFF |
| [provenance/](provenance/) | Environment records and source/data hashes |

## License

The source code is released under the [MIT License](LICENSE).
