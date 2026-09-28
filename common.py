r"""Shared setup for the referee-response figure notebooks.

One notebook per figure, one module for everything they have in common. Import it
whole:

```python
from common import *
```

**The cache key is a contract.** Every block on disk is named
``<tag>_<sha1(cfg)[:10]>``, where the digest is taken over the *config dict alone* —
never over the folder it lands in. That is why `cache_paths` takes the folder as
separate arguments instead of reading it out of ``cfg``: adding a folder key to a
config would change its digest and orphan blocks that cost hours to rebuild. Change a
``*_config`` function and you are asking for a recompute.

Sections below, in dependency order: paths, constants, slugs, the cache, the model,
timing, drawing.
"""

import dataclasses  # noqa: F401  — used by the ESCALADE manifold
import hashlib
import json
import pathlib
import time
from functools import partial  # noqa: F401

import jax  # noqa: F401
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401
import scienceplots  # noqa: F401  — registers the 'science' style
from matplotlib.lines import Line2D  # noqa: F401
from matplotlib.patches import Patch  # noqa: F401
from matplotlib.ticker import MaxNLocator, NullFormatter  # noqa: F401
from matplotlib.transforms import Bbox
from scipy.optimize import least_squares

# Re-exported so that `from common import *` is the whole of a notebook's preamble.
from geope import (  # noqa: F401
    LBFGS,
    Armijo,
    Geope,
    Grape,
    History,
    NewtonRFO,
    NewtonSaddleFree,
    NewtonTRM,
    Parameters,
)
from geope.geope import linear_comb_projected_coeffs_multigate  # noqa: F401
from geope.jax import (  # noqa: F401
    Ui,
    adj_expm,
    adj_expm_eig,
    d2expm,
    d2expm_eig,
    dexpm,
    dexpm_eig,
    expm_hvp,
    expm_hvp_eig,
    hessian_propagator,
    hessian_vjp_propagator,
    hvp_propagator,
    jacobian_propagator,
    vjp_propagator,
)
from geope.optimizers import newton_saddle_free_step  # noqa: F401
from geope.utils import (  # noqa: F401
    construct_full_pauli_basis,
    multicontrol_unitary,
    qft_unitary,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = pathlib.Path("cache")
FIGURE_DIR = pathlib.Path("figures")

# ---------------------------------------------------------------------------
# Constants shared by every figure
# ---------------------------------------------------------------------------

MAX_STEPS = 200
PRECISION = 1.0 - 1e-9
GEOPE_MAX_STEP = 1.0
SPREAD = 0.01

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

#: Column widths in inches.
ONE_COLUMN = 3.4
TWO_COLUMN = 7.1

#: One size for every "(a)"/"(b)" panel label
PANEL_LABEL_SIZE = 9
ANNOTATION_SIZE = 7
LABEL_FONT_SIZE = 7

SCIENCEPLOT_COLOURS = [
    "#0C5DA5",
    "#00B945",
    "#FF9500",
    "#FF2C00",
    "#845B97",
    "#474747",
    "#9e9e9e",
]
#: One colour per method, everywhere.
COLOURS = {
    "GEOPE": SCIENCEPLOT_COLOURS[0],
    "NewtonTRM": SCIENCEPLOT_COLOURS[1],
    "NewtonRFO": SCIENCEPLOT_COLOURS[2],
    "NewtonSaddleFree": SCIENCEPLOT_COLOURS[3],
    "LBFGS": SCIENCEPLOT_COLOURS[4],
}
LABELS = {
    "GEOPE": r"GEOPE",
    "NewtonTRM": r"Newton: TRM",
    "NewtonRFO": r"Newton: RFO",
    "NewtonSaddleFree": r"Newton: SFN",
    "LBFGS": r"L-BFGS",
}

#: The individual runs, drawn under the medians so the spread is visible.
FAINT = dict(lw=0.25, alpha=0.1, zorder=1)

plt.style.use("science")

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": LABEL_FONT_SIZE,
        "ytick.labelsize": LABEL_FONT_SIZE,
        "legend.fontsize": LABEL_FONT_SIZE,
        "lines.linewidth": 1.0,
    }
)


def save_figure(fig, name, width=None):
    """Write ``fig`` to ``figures/<name>.pdf`` at an exact width.

    Args:
        fig: The figure to write.
        name: Basename, without the extension.
        width: Target width in inches. Defaults to the figure's own declared width,
            which is `ONE_COLUMN` or `TWO_COLUMN` for every figure here.

    Returns:
        The path written.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.pdf"
    target = float(fig.get_size_inches()[0]) if width is None else float(width)
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0,
    )
    return path


def panel_label(ax, letter, x=-0.2, y=0.95):
    """The "(a)" in a panel's top-left corner, at the one shared size."""
    return ax.text(
        x,
        y,
        rf"({letter})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        zorder=5,
    )


def method_style(method):
    """Line style for one optimiser, thicker for GEOPE since it is the subject."""
    return dict(
        lw=1.4 if method == "GEOPE" else 1.0,
        ls="-",
        color=COLOURS[method],
        label=LABELS[method],
        zorder=3,
    )


def infidelities(fidelities):
    """``1 - F`` per run, floored so a log axis can draw an exact solution."""
    return [np.clip(1.0 - f, 1e-18, None) for f in fidelities]


def median_over_runs(curves):
    """Elementwise median over runs, edge-padded to the longest."""
    longest = max(c.size for c in curves)
    padded = np.array([np.pad(c, (0, longest - c.size), mode="edge") for c in curves])
    return np.median(padded, axis=0)


def draw_iterations(ax, blocks, methods):
    """Infidelity against optimisation step, medians over faint individual runs."""
    for method in methods:
        for y in infidelities(blocks[method][0]):
            ax.semilogy(np.arange(y.size), y, color=COLOURS[method], **FAINT)
    for method in methods:
        y = median_over_runs(infidelities(blocks[method][0]))
        ax.semilogy(np.arange(y.size), y, **method_style(method))
    ax.set_xlabel("Iteration number")


def draw_walltime(ax, blocks, methods):
    """The same runs against their own cumulative wall clock.

    Step 0 is dropped throughout: it sits at ``t = 0``, which a log axis cannot draw.
    """
    for method in methods:
        fids, secs = blocks[method]
        for y, t in zip(infidelities(fids), secs):
            ax.semilogy(t[1:], y[1:], color=COLOURS[method], **FAINT)
    for method in methods:
        fids, secs = blocks[method]
        y = median_over_runs(infidelities(fids))
        t = median_over_runs(secs)
        ax.semilogy(t[1:], y[1:], **method_style(method))
    ax.set_xscale("log")
    ax.set_xlabel("Wall clock time [s]")


def finish(ax, title=""):
    """The target line, the shared y range and the grid."""
    ax.axhline(1.0 - PRECISION, color="black", ls=":", lw=0.8, zorder=2)
    ax.set_ylim(1e-10, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both", lw=0.4)


def loglog_exponent(x, y, mask=None):
    r"""Least-squares power law $y = a x^b$, fitted in log-log space."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if mask is not None:
        x, y = x[mask], y[mask]
    b, log_a = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_a)), float(b)


def offset_power(x, y, fit_from):
    r"""Least-squares $t = t_0 + a L^b$ over ``x >= fit_from``, in log space.

    Returns ``(t0, a, b)``. The offset matters because every timing here has a
    dispatch floor that a pure power law would otherwise fold into the exponent.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = x >= fit_from
    x, y = x[keep], y[keep]
    log_x, log_y = np.log(x), np.log(y)
    a0, b0 = loglog_exponent(x, y)
    fit = least_squares(
        lambda p: np.logaddexp(p[0], p[1] + p[2] * log_x) - log_y,
        [np.log(y.min() / 2), np.log(a0), b0],
    )
    return float(np.exp(fit.x[0])), float(np.exp(fit.x[1])), float(fit.x[2])


def power_fit(name, x, y, fit_from):
    r"""`offset_power`, plus the printed exponent and the legend label."""
    keep = np.asarray(x, float) >= fit_from
    _amplitude, _slope = loglog_exponent(x, y, mask=keep)
    coefficients = offset_power(x, y, fit_from)
    print(rf"{name} ($\propto L^{{{coefficients[2]:.2f}}}$)")
    return coefficients, rf"{name}"


def time_batch(fn, args, rounds, *, label=""):
    """One batch of ``rounds`` timings of a warmed-up jitted callable."""
    timings = []
    announced = time.perf_counter()
    for index in range(rounds):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        timings.append(time.perf_counter() - start)
        if label and time.perf_counter() - announced >= PROGRESS_SECONDS:
            announced = time.perf_counter()
            print(
                f"      {label} round {index + 1}/{rounds}  "
                f"{timings[-1] * 1e3:.1f} ms/call",
                flush=True,
            )
    return np.asarray(timings, dtype=float)


def reduce_samples(samples):
    """The min over batches of the median within a batch, in seconds."""
    return float(np.min(np.median(np.asarray(samples), axis=1)))


def roughness(fields):
    r"""RMS step-to-step change relative to RMS amplitude, over the given columns."""
    fields = np.asarray(fields)
    scale = np.sqrt(np.mean(fields**2))
    if scale == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(fields, axis=0) ** 2)) / scale)


# ---------------------------------------------------------------------------
# Filename slugs
# ---------------------------------------------------------------------------


def spread_slug(spread):
    r"""Filename-safe name for an initial spread, in units of $\pi$."""
    return "spread{}pi".format(f"{spread:g}".replace(".", "p"))


def value_slug(value):
    """Filename-safe name for a hyperparameter value: ``1e-08`` becomes ``1em08``."""
    if isinstance(value, str):
        return value
    return f"{value:g}".replace(".", "p").replace("-", "m").replace("+", "")


def piece_slug(piece):
    """Filename-safe name for a timed component: ``GEOPE: logm`` -> ``GEOPE-logm``."""
    return piece.replace(": ", "-").replace(" ", "-")


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


def cache_digest(cfg):
    """The block's cache key: sha1 over the config dict, nothing else.

    Read the module docstring before touching this.
    """
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def cache_paths(cfg, *folder):
    """``(npz, json)`` for one block, under ``CACHE_DIR / *folder``.

    ``folder`` is passed rather than read out of ``cfg`` so that where a block lands
    is independent of what keys it into the cache. Every ``*_paths`` helper in the
    notebooks is one line of this.
    """
    stem = f"{cfg['tag']}_{cache_digest(cfg)}"
    directory = CACHE_DIR.joinpath(*(str(part) for part in folder))
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def cache_load(npz_path):
    """Every array in the block as a dict, or ``None`` if it is absent."""
    if not npz_path.exists():
        return None
    with np.load(npz_path) as handle:
        return {key: handle[key] for key in handle.files}


def cache_save(npz_path, json_path, cfg, arrays, **sidecar):
    """Write one block plus a readable sidecar of its config.

    Args:
        npz_path: Where the arrays go; parents are created.
        json_path: Where the sidecar goes.
        cfg: The config the digest was taken over.
        arrays: ``{name: array}`` to store.
        **sidecar: Extra keys to record alongside ``cfg``, e.g. ``n_runs``.
    """
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps({**cfg, **sidecar}, indent=1, sort_keys=True))


def block_paths(cfg):
    """``(npz, json)`` for a per-seed run block — one folder per method."""
    return cache_paths(cfg, cfg["method"])


def load_block(npz_path):
    """The runs already stored, or ``None`` if the block is absent.

    Deliberately returns a *partial* block: the caller extends it with the seeds it is
    missing instead of recomputing from scratch.
    """
    stored = cache_load(npz_path)
    if stored is None:
        return None
    count = int(stored["totals"].size)
    return (
        [stored[f"fid_{s}"] for s in range(count)],
        [stored[f"sec_{s}"] for s in range(count)],
        list(stored["totals"]),
    )


def save_block(npz_path, json_path, cfg, fidelities, seconds, totals):
    """Write a per-seed run block: one array per run, plus the totals."""
    cache_save(
        npz_path,
        json_path,
        cfg,
        {
            "totals": np.asarray(totals, dtype=float),
            **{f"fid_{s}": f for s, f in enumerate(fidelities)},
            **{f"sec_{s}": t for s, t in enumerate(seconds)},
        },
        n_runs=len(fidelities),
    )


def run_block(cfg, build_optimiser, n_runs, *, spread=None, force=False, verbose=True):
    """Every seed of one block, reading and extending the cache.

    Args:
        cfg: The block's config; its digest is the cache key.
        build_optimiser: ``() -> (params, optimiser, config)``, called only on a miss
            so a fully cached sweep never touches JAX.
        n_runs: How many seeds are wanted.
        spread: The initial spread the pulses are drawn at, defaulting to `SPREAD`. It
            is the caller's job to keep this in step with ``cfg`` -- the config is the
            cache key, this only draws the pulses.
        force: Recompute from seed 0, discarding what is stored.
        verbose: Print a line per run as it lands, and one per block. The per-run line
            is the heartbeat: a block is silent until its last seed otherwise, and one
            NewtonRFO block at $L=120$ has run for eleven hours.

    Returns:
        ``(fidelities, seconds)``, each a list of ``n_runs`` arrays.
    """
    spread = SPREAD if spread is None else spread
    npz_path, json_path = block_paths(cfg)
    cached = None if force else load_block(npz_path)
    fidelities, seconds, totals = cached if cached else ([], [], [])
    have = len(fidelities)

    if have >= n_runs:
        note = f"cached ({have} stored)"
    else:
        params, optimiser, config = build_optimiser()
        # Seed 10_000 is outside the run range, so the warm-up never doubles as a
        # real sample.
        warm_up(
            optimiser,
            random_pulse(params, np.random.default_rng(10_000), spread),
            config,
        )
        started = time.perf_counter()
        if verbose:
            # Before the first run, not after it: a block that computes one NewtonRFO
            # seed says nothing for an hour otherwise.
            print(
                f"  {cfg['tag']:40s} computing seed"
                f"{'s' if n_runs - have > 1 else ''} {have}-{n_runs - 1} of {n_runs}",
                flush=True,
            )
        for seed in range(have, n_runs):
            pulse = random_pulse(params, np.random.default_rng(seed), spread)
            fidelity, elapsed = timed_trajectory(
                optimiser, pulse, config, label=f"seed {seed}" if verbose else ""
            )
            fidelities.append(fidelity)
            seconds.append(elapsed)
            totals.append(float(elapsed[-1]))
            save_block(npz_path, json_path, cfg, fidelities, seconds, totals)
            if verbose:
                done = seed - have + 1
                left = (n_runs - 1 - seed) * (time.perf_counter() - started) / done
                eta = ""
                if left > 60:
                    eta = (
                        f"   ~{left / 60:.0f} min left"
                        if left < 5400
                        else f"   ~{left / 3600:.1f} h left"
                    )
                print(
                    f"    seed {seed}/{n_runs - 1}  {elapsed[-1]:7.1f}s  "
                    f"{fidelity.size - 1:3d} steps  1-F {1 - fidelity[-1]:.1e}{eta}",
                    flush=True,
                )
        note = (
            f"computed {n_runs - have} of {n_runs} in "
            f"{time.perf_counter() - started:.0f}s"
        )

    if verbose:
        reached = sum(f[-1] > PRECISION for f in fidelities[:n_runs])
        median_s = float(np.median(totals[:n_runs]))
        print(
            f"  {cfg['tag']:40s} {reached:2d}/{n_runs} converged  "
            f"median {median_s:6.2f}s   {note}",
            flush=True,
        )
    return fidelities[:n_runs], seconds[:n_runs]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

_X = np.array([[0, 1], [1, 0]], dtype=complex)
TARGETS = {
    "CNOT": multicontrol_unitary(_X, 1),
    "CCX": multicontrol_unitary(_X, 2),  # the Toffoli
}


def make_parameters(n, steps, target, *, spread=1.0, seed=0):
    r"""The ``n``-atom Rydberg array driving ``target`` over ``steps`` segments.

    Controls are $\Omega_i\sigma^x_i$ and $\Delta_i\sigma^z_i$, one pair per atom; the
    drift is $\sigma^z_i\sigma^z_j$ on every pair at equal strength.
    """
    if isinstance(target, str):
        target = TARGETS[target]
    control = {i: ["x", "z"] for i in range(1, n + 1)}
    drift = {(i, j): ["zz"] for i in range(1, n + 1) for j in range(i + 1, n + 1)}
    return Parameters(
        basis=construct_full_pauli_basis(n),
        control=control,
        drift=drift,
        target=target,
        piecewise_steps=steps,
        init_spread=spread,
        seed=seed,
    )


def make_parameters_cnot(steps, target, *, spread=1.0, seed=0, drift_values=None):
    r"""The ``n``-atom Rydberg array driving ``target`` over ``steps`` segments.

    Controls are $\Omega_i\sigma^x_i$ and $\Delta_i\sigma^z_i$, one pair per atom; the
    drift is $\sigma^z_i\sigma^z_j$ on every pair at equal strength.

    ``drift_values`` is the ZZ coefficient *per segment*, i.e. $J\,\Delta t$;
    ``None`` keeps the library default of 1.
    """
    if isinstance(target, str):
        target = TARGETS[target]
    control = {0: ["x", "z"], 1: ["x", "z"]}
    drift = {(0, 1): ["zz"]}
    return Parameters(
        basis=construct_full_pauli_basis(2),
        control=control,
        drift=drift,
        drift_values=drift_values,
        target=target,
        piecewise_steps=steps,
        init_spread=spread,
        seed=seed,
    )


def random_pulse(params, rng, spread):
    r"""A random full-basis pulse, zero outside the controllable columns.

    Uniform on $[-\pi\,\mathrm{spread}, +\pi\,\mathrm{spread}]$ — the factor of $\pi$
    lives here, so ``spread=1.0`` means an initial spread of $1.0\pi$. Full-basis
    because that is what both optimisers' ``init`` accepts; each then writes its own
    drift columns. Handing the same array to every method is what makes a per-seed
    comparison meaningful.
    """
    pulse = np.zeros((params.piecewise_steps, params.basis.lie_algebra_dim))
    pulse[:, params.projected_indices] = rng.uniform(
        -spread * np.pi,
        spread * np.pi,
        size=(params.piecewise_steps, params.projected_basis.lie_algebra_dim),
    )
    return pulse


# ---------------------------------------------------------------------------
# Running and timing one trajectory
# ---------------------------------------------------------------------------


def _optimise(optimiser, config, max_steps, callbacks=None, max_step_size=None):
    """Dispatch to whichever of the two ``optimize`` signatures applies.

    A `Geope` config is a line search, or the pair ``(line_search, max_step_size)``
    when a caller carries the bracket alongside it — both are arguments of
    `Geope.optimize` rather than fields of one object, so a sweep over line searches
    has to keep them together. A `Grape` config is the `Optimizer` instance itself.
    ``max_step_size`` defaults to `GEOPE_MAX_STEP` and is ignored by `Grape`, which
    has no bracket to cap.
    """
    if isinstance(optimiser, Geope):
        if isinstance(config, tuple):
            config, max_step_size = config
        return optimiser.optimize(
            max_steps=max_steps,
            precision=PRECISION,
            line_search=config,
            max_step_size=GEOPE_MAX_STEP if max_step_size is None else max_step_size,
            callbacks=callbacks,
        )
    return optimiser.optimize(
        max_steps=max_steps, optimizer=config, callbacks=callbacks
    )


def reinit(optimiser, init_pulse):
    """Re-initialise ``optimiser`` at ``init_pulse``, keeping the configured drift.

    Both ``init`` methods default a missing ``drift_parameters`` to ones rather
    than to ``params.drift_parameters``, so a bare ``init(init_parameters=...)``
    silently discards ``Parameters(drift_values=...)``. Passing it through is a
    no-op for every model left at the default drift of one.
    """
    optimiser.init(
        init_parameters=init_pulse,
        drift_parameters=optimiser.params.drift_parameters,
    )


def warm_up(optimiser, init_pulse, config, max_step_size=None):
    """One throwaway step, so the JIT trace lands outside every timed run."""
    reinit(optimiser, init_pulse)
    _optimise(optimiser, config, max_steps=1, max_step_size=max_step_size)


#: Seconds between step-level heartbeats. 
PROGRESS_SECONDS = 60.0


def timed_trajectory(
    optimiser, init_pulse, config, *, label="", max_step_size=None, max_steps=None
):
    """One run: ``(fidelity, cumulative seconds)`` per step, step 0 included."""
    max_steps = MAX_STEPS if max_steps is None else max_steps
    reinit(optimiser, init_pulse)
    optimiser.history.reset()
    optimiser.history.record(optimiser)  # step 0, after the re-init

    stamps = []
    started = time.perf_counter()
    announced = [started]

    def clock(step, history, opt):
        now = time.perf_counter()
        stamps.append(now - started)
        if label and now - announced[0] >= PROGRESS_SECONDS:
            announced[0] = now
            fidelities = (history or opt.history).fidelities
            print(
                f"      {label} step {step:3d}/{max_steps}  {stamps[-1]:7.1f}s  "
                f"1-F {1 - float(fidelities[-1]):.1e}",
                flush=True,
            )
        return True  # a pure logging callback must return True

    _optimise(
        optimiser,
        config,
        max_steps=max_steps,
        callbacks=clock,
        max_step_size=max_step_size,
    )

    fidelity = np.array([float(f) for f in optimiser.history.fidelities])
    seconds = np.concatenate([[0.0], np.asarray(stamps, dtype=float)])
    assert seconds.size == fidelity.size, (seconds.size, fidelity.size)
    return fidelity, seconds


def timed_trial(optimiser, init_pulse, config, *, max_step_size=None, max_steps=None):
    """``(final infidelity, iterations, seconds)`` for one seed."""
    max_steps = MAX_STEPS if max_steps is None else max_steps
    reinit(optimiser, init_pulse)
    optimiser.history.reset()
    optimiser.history.record(optimiser)  # step 0, after the re-init
    started = time.perf_counter()
    _optimise(optimiser, config, max_steps=max_steps, max_step_size=max_step_size)
    fidelities = np.asarray(optimiser.history.fidelities, dtype=float)
    elapsed = time.perf_counter() - started
    return 1.0 - float(fidelities[-1]), int(fidelities.size - 1), elapsed


METHODS = {
    "GEOPE": Armijo(),
    "LBFGS": LBFGS(memory=100),
    "Newton: TRM": NewtonTRM(delta=1e-3),
    "Newton: RFO": NewtonRFO(kappa=10**6),
    "Newton: SFN": NewtonSaddleFree(),
}

def traj_config(steps, method, spread):
    """The block's cache key: the fig1 config, under a ``traj_`` tag."""
    cfg = {
        "tag": f"traj_CCX_L{steps}_{spread_slug(spread)}_{method}",
        "kind": "fig1-toffoli-parameter-trajectories",
        "gate": "CCX",
        "steps": steps,
        "method": method,
        "config": repr(METHODS[method]),
        "spread": float(spread),
        "n_qubits": 3,
        "max_steps": MAX_STEPS,
        "precision": PRECISION,
    }
    if method == "GEOPE":
        cfg["geope_max_step"] = GEOPE_MAX_STEP
    return cfg


def _build_trajectory(steps, method, spread):
    params = make_parameters(3, steps, "CCX", spread=spread)
    free = np.asarray(params.projected_indices, dtype=bool)
    flag = {"gs": False}

    def logging_fn(opt):
        row = {
            "phi": np.real(np.array(opt.params.parameters))[:, free].ravel(),
            "fidelities": float(opt.params.fidelity),
            "gs": flag["gs"],
        }
        flag["gs"] = False
        return row

    history = History(logging_fn=logging_fn)
    if method == "GEOPE":
        optimiser = Geope(params, history=history)
        inner = optimiser.gram_schmidt

        def gram_schmidt(coeffs):
            flag["gs"] = True
            return inner(coeffs)

        optimiser.gram_schmidt = gram_schmidt
    else:
        optimiser = Grape(params, precision=PRECISION, history=history)
    return params, optimiser


def trajectory_block(steps, method, spread, n_runs, verbose=True):
    """``[{"phi": (n+1, P), "fid": (n+1,), "gs": (n+1,) bool}, ...]`` per seed."""
    cfg = traj_config(steps, method, spread)
    npz_path, json_path = cache_paths(cfg, "traj")
    stored = cache_load(npz_path) or {}
    have = int(stored.get("n_runs", np.array(0)))
    runs = [
        {k: stored[f"{k}_{s}"] for k in ("phi", "fid", "gs")} for s in range(have)
    ]
    if have < n_runs:
        params, optimiser = _build_trajectory(steps, method, spread)
        config = METHODS[method]
        warm_up(optimiser, random_pulse(params, np.random.default_rng(10_000), spread), config)
        for seed in range(have, n_runs):
            reinit(optimiser, random_pulse(params, np.random.default_rng(seed), spread))
            optimiser.history.reset()
            optimiser.history.record(optimiser)
            t0 = time.perf_counter()
            _optimise(optimiser, config, max_steps=MAX_STEPS)
            h = optimiser.history
            runs.append(
                {
                    "phi": np.stack(h.phi),
                    "fid": np.asarray(h.fidelities),
                    "gs": np.asarray(h.gs, dtype=bool),
                }
            )
            arrays = {"n_runs": np.array(len(runs))}
            for s, r in enumerate(runs):
                arrays.update({f"{k}_{s}": v for k, v in r.items()})
            cache_save(npz_path, json_path, cfg, arrays)
            if verbose:
                print(
                    f"  {cfg['tag']} seed {seed}: {len(h.fidelities) - 1} steps "
                    f"{time.perf_counter() - t0:.1f}s 1-F {1 - h.fidelities[-1]:.1e}",
                    flush=True,
                )
    return runs[:n_runs]


def metrics(run):
    """Scalar path statistics of one trajectory."""
    phi = run["phi"]
    d = np.diff(phi, axis=0)
    norms = np.linalg.norm(d, axis=1)
    path = norms.sum()
    disp = np.linalg.norm(phi[-1] - phi[0])
    excursion = np.linalg.norm(phi - phi[0], axis=1).max()
    moving = norms > 1e-14
    dm = d[moving]
    cos = np.sum(dm[1:] * dm[:-1], axis=1) / (
        np.linalg.norm(dm[1:], axis=1) * np.linalg.norm(dm[:-1], axis=1)
    )
    gs = run["gs"][1:]
    return {
        "iters": len(norms),
        "path": path,
        "path_no_gs": norms[~gs].sum(),
        "disp": disp,
        "tortuosity": path / disp,
        "excursion": excursion,
        "max_step": norms.max(),
        "rms_end": np.sqrt(np.mean(phi[-1] ** 2)),
        "rms_start": np.sqrt(np.mean(phi[0] ** 2)),
        "linf_end": np.abs(phi[-1]).max(),
        "mean_cos": cos.mean() if cos.size else np.nan,
        "frac_reversal": np.mean(cos < 0) if cos.size else np.nan,
        "n_gs": int(gs.sum()),
        "converged": bool(run["fid"][-1] > PRECISION),
    }
