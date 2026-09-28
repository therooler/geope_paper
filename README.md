# Quantum Optimal Control with Geodesic Pulse Engineering

This repository holds the code for the figures in [*Quantum Optimal Control with Geodesic Pulse Engineering*](https://arxiv.org/pdf/2508.16029) (arXiv:2508.16029). Each figure has its own notebook, and they are built on the [`geope`](https://pypi.org/project/geope/) library and a small `common.py` with the Rydberg model, the cache, timing and plotting helpers.

## Installation

Python 3.11 or newer is required.

**With uv**:

```bash
uv sync
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

Results are cached under `cache/` and figures are written to `figures/` as PDFs.
Both folders are created on the first run. A notebook that finds its cached results
reuses them, so a later run only redraws the figure.

| Notebook | Contents |
| --- | --- |
| `fig1_toffoli.ipynb` | Toffoli gate on three Rydberg atoms at $L=12$ and $L=120$. Compares GEOPE with GRAPE using Newton, RFO, saddle-free Newton and L-BFGS, from initial pulses close to and far from the identity. Plots infidelity against both iteration count and wall-clock time. |
| `fig2_cost_per_step.ipynb` | How long one optimiser step takes as the pulse gets longer, broken into its parts: Jacobian, least squares and matrix logarithm for GEOPE, and gradient, Hessian and eigendecomposition for Newton. Includes the fitted scaling exponents. |
| `fig3_qft.ipynb` | GEOPE on the 5-qubit QFT for several pulse lengths, alongside the condition number of the flattened Jacobian. |
| `fig4_escalade.ipynb` | Adds Goodwin & Vinding's ESCALADE closed-form gate exponential (and its derivatives) to the pipeline and checks that it is exact. Benchmarks it against the auxiliary-matrix and spectral routes, and shows that the speedup is the same for GEOPE and GRAPE. |
| `fig5_large_L.ipynb` | A single CNOT run at $L=10000$ from a Gaussian-smoothed initial pulse, using Rydberg-atom time and energy scales. |
| `fig6_lbfgs_negative_curvature.ipynb` | Why L-BFGS fails on the Toffoli gate at initial spread $0.01\pi$. Tracks the Hessian spectrum along the L-BFGS trajectory: about half of the eigenvalues are negative, and a positive-definite quasi-Newton step cannot move along those directions. Saddle-free Newton is the control. |
| `fig7_hyperparameter_search.ipynb` | Hyperparameter sweeps at $L=12$ and $L=120$: `delta` for Newton, `kappa` for RFO and `memory` for L-BFGS. Also checks that GEOPE (across line searches) and saddle-free Newton (across `rcond`) need no tuning. |
| `fig8_parameter_trajectories.ipynb` | The five optimisers' paths through parameter space on the Figure 1 Toffoli problem, each projected onto its own two leading principal components. |

Rebuilding the cache from scratch can take hours.
