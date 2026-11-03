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

A notebook named `figN_…` produces the paper's Figure N (Figures 1 and 2 are drawn by
hand, so there are none for those); `fig4ab_…` and `fig4c_…` build the data and the
panels that `fig4_…` combines. The appendix figures are lettered as in the paper:
`figD1_…`, `figE_…` (Figures E1 and E2, one per pulse length) and `figF1_…`, while
Figure B1 is drawn by the `fig4_…` notebook. A notebook named `extra_figN_…` makes a
figure that is not in the paper, numbered in its own sequence.

| Notebook | Contents |
| --- | --- |
| `fig3_toffoli.ipynb` | Toffoli gate on three Rydberg atoms at $L=12$ and $L=120$. Compares GEOPE with GRAPE using Newton, RFO, saddle-free Newton and L-BFGS, from initial pulses close to and far from the identity. Plots infidelity against both iteration count and wall-clock time. |
| `fig4ab_parameter_trajectories_diffs_random.ipynb` | The three measures of `extra_fig6_parameter_trajectories_diffs.ipynb` repeated on Haar-random 3-qubit target unitaries, one seeded initial pulse per target, at the same $L$ and initial spread, with the curves and the summary averaged over the targets. |
| `fig4c_qft.ipynb` | GEOPE on the 5-qubit QFT for several pulse lengths, alongside the condition number of the flattened Jacobian. |
| `fig4_path_alignment_conditioning.ipynb` | The paper's Figure 4, drawn from the caches of the two notebooks above and of `fig4c_qft.ipynb` without re-running them: (a) the parameter path length and (b) the step alignment on the Haar-random targets, and (c) the QFT infidelity over the condition number of the Jacobian. Also draws the cumulative geodesic path length on its own as the paper's Figure B1. |
| `fig5_large_L.ipynb` | A single CNOT run at $L=10000$ from a one-period sinusoidal initial pulse, driven by two global fields (a Rabi drive and a detuning shared by both atoms) plus a static local detuning on the control atom, which breaks the exchange symmetry and makes the pair fully controllable. The model is scaled to QuEra Aquila's conventions: a $1\,\mu$s program, $g = 1.96$ rad/$\mu$s from atoms $9.4\,\mu$m apart, a local detuning of 63 rad/$\mu$s on the control atom, a Rabi frequency under the 15.8 rad/$\mu$s limit, and the two global fields ramped from zero with a raised-cosine ramp over the first and last 50 ns as the machine requires, with the local detuning held static. The gate time is held at $gT=5\pi/8$, two and a half times the $\pi/4$ a CNOT needs, so a larger $L$ subdivides the pulse instead of lengthening it. |
| `figD1_cost_per_step.ipynb` | How long one optimiser step takes as the pulse gets longer, broken into its parts: Jacobian, least squares and matrix logarithm for GEOPE, and gradient, Hessian and eigendecomposition for Newton. Includes the fitted scaling exponents. |
| `figE_hyperparameter_search.ipynb` | Hyperparameter sweeps at $L=12$ and $L=120$: `delta` for Newton, `kappa` for RFO and `memory` for L-BFGS. Also checks that GEOPE (across line searches) and saddle-free Newton (across `rcond`) need no tuning. |
| `figF1_lbfgs_negative_curvature.ipynb` | Why L-BFGS fails on the Toffoli gate at initial spread $0.01\pi$. Tracks the Hessian spectrum along the L-BFGS trajectory: about half of the eigenvalues are negative, and a positive-definite quasi-Newton step cannot move along those directions. Saddle-free Newton is the control. |
| `extra_fig1_ccz.ipynb` | The Figure 3 sweep repeated for the CCZ gate in place of the Toffoli: the same five optimisers, pulse lengths, initial spreads and seeds, written to its own cache blocks and figure. |
| `extra_fig2_fredkin.ipynb` | The Figure 3 sweep repeated for the Fredkin (controlled-SWAP) gate: the same five optimisers, pulse lengths, initial spreads and seeds, written to its own cache blocks and figure. |
| `extra_fig3_qft3.ipynb` | The Figure 3 sweep repeated for the 3-qubit QFT: the same five optimisers, pulse lengths, initial spreads and seeds, written to its own cache blocks and figure. |
| `extra_fig4_escalade.ipynb` | Adds Goodwin & Vinding's ESCALADE closed-form gate exponential (and its derivatives) to the pipeline and checks that it is exact. Benchmarks it against the auxiliary-matrix and spectral routes, and shows that the speedup is the same for GEOPE and GRAPE. |
| `extra_fig5_parameter_trajectories.ipynb` | The five optimisers' paths through parameter space on the Figure 3 Toffoli problem, each projected onto its own two leading principal components. |
| `extra_fig6_parameter_trajectories_diffs.ipynb` | The same trajectories reduced to three per-iteration measures, side by side: the cumulative path length $\sum_k \|\phi_{k+1}-\phi_k\|$ in parameter space, the cumulative geodesic distance $\sum_k d_g(U_k, U_{k+1})$ travelled on the group by the unitaries the iterates implement, and the cosine of the angle at $U_k$ between the geodesic direction to the target and the direction each step actually took. Tabulates per-method medians of both path lengths, displacements, tortuosities and alignments. |
| `extra_fig7_qft5_lbfgs.ipynb` | GEOPE against L-BFGS GRAPE on the 5-qubit QFT of Figure 4(c) at $L=120$, from initial spreads $\pi/100$ and $\pi$. L-BFGS is swept over its memory, its trial-step cap and its strong-Wolfe curvature constant, one knob at a time around the Figure 3 setting, and with its line search swapped for backtracking Armijo; GEOPE runs as in Figure 4(c). Plots each knob's sweep against iteration and wall clock in the Figure 3 layout, and every knob's median wall clock, iterations and fraction converged. |

Rebuilding the cache from scratch can take hours.
