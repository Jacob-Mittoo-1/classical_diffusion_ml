from __future__ import annotations

import os

# JAX otherwise preallocates nearly all available GPU memory before execution.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from classical_diffusion.jax.langevin import (
    KramersParameters,
    KramersSystem1D,
    solve_many_overdamped,
)
from classical_diffusion.simulation import TimeSpan
from scipy.constants import Boltzmann


def generate_trajectories(
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    seed: int = 234,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Langevin trajectories for one fixed set of parameters.

    Returns
    -------
    times
        One-dimensional simulation time array.
    trajectories
        Array with shape ``(n_trajectories, n_time_samples)``.
    """
    system = KramersSystem1D(params=params).as_canonical()
    initial_conditions = (jnp.zeros((1, 1)), jnp.zeros((1, 1)))
    if n_trajectories < 1:
        msg = "n_trajectories must be at least 1"
        raise ValueError(msg)

    keys = jax.random.split(jax.random.PRNGKey(seed), n_trajectories)

    def run_one(key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        times, positions, _ = solve_many_overdamped(
            system,
            time_span,
            initial_conditions,
            _key=key,
        )

        # positions has shape (dimension, initial_condition, time).
        return times, positions[0, 0]

    chunk_size = 8
    trajectory_chunks = []

    # Run only a small batch at once so JAX does not reserve a 12 GiB GPU
    # workspace for all 128 trajectories.
    first_chunk_keys = keys[:chunk_size]
    times_by_run, first_chunk = jax.vmap(run_one)(first_chunk_keys)
    times = np.asarray(times_by_run[0])
    trajectory_chunks.append(np.asarray(first_chunk))

    for start in range(chunk_size, n_trajectories, chunk_size):
        chunk_keys = keys[start : start + chunk_size]
        _times_by_run, chunk = jax.vmap(run_one)(chunk_keys)
        trajectory_chunks.append(np.asarray(chunk))

    return times, np.concatenate(trajectory_chunks, axis=0)


def fit_long_time_diffusion(
    times: np.ndarray,
    trajectories: np.ndarray,
    *,
    start_fraction: float = 0.01,
    end_fraction: float = 1.0,
) -> tuple[float, float, np.ndarray]:
    """Fit the large-time MSD slope and estimate D.

    Returns
    -------
    diffusion_constant
        Effective diffusion constant, D = slope / 2.
    msd
        Mean-square displacement for every sample time.
    """
    times = np.asarray(times, dtype=float)
    trajectories = np.asarray(trajectories, dtype=float)

    if trajectories.ndim != 2:
        msg = "trajectories must have shape (n_trajectories, n_time_samples)"
        raise ValueError(msg)
    if trajectories.shape[1] != times.size:
        msg = "The trajectory and time lengths must match"
        raise ValueError(msg)

    initial_positions = trajectories[:, 0, np.newaxis]
    displacements = trajectories - initial_positions
    msd = np.mean(displacements**2, axis=0)

    start_time = times[0] + start_fraction * (times[-1] - times[0])
    end_time = times[0] + end_fraction * (times[-1] - times[0])
    window = (times >= start_time) & (times <= end_time)

    if np.count_nonzero(window) < 2:
        msg = "The selected long-time window contains too few samples"
        raise ValueError(msg)

    coefficients, covariance = np.polyfit(
        times[window],
        msd[window],
        deg=1,
        cov=True,
    )
    slope, _intercept = coefficients
    diffusion_constant = slope / 2.0
    diffusion_uncertainty = float(np.sqrt(covariance[0, 0]) / 2.0)

    return diffusion_constant, diffusion_uncertainty, msd


def plot_msd_with_tangent(
    times: np.ndarray,
    trajectories: np.ndarray,
    *,
    start_fraction: float = 0.0,
    end_fraction: float = 1.0,
) -> tuple[float, float, np.ndarray, np.ndarray, plt.Figure, plt.Axes]:
    """Plot the ensemble MSD and its fitted large-time tangent."""
    times = np.asarray(times, dtype=float)
    trajectories = np.asarray(trajectories, dtype=float)

    if times.ndim != 1:
        msg = "times must be a one-dimensional array"
        raise ValueError(msg)
    if trajectories.ndim != 2:
        msg = "trajectories must have shape (n_trajectories, n_time_samples)"
        raise ValueError(msg)
    if trajectories.shape[1] != times.size:
        msg = "The trajectory and time lengths must match"
        raise ValueError(msg)
    if not 0.0 <= start_fraction <= end_fraction <= 1.0:
        msg = "Fitting fractions must satisfy 0 <= start <= end <= 1"
        raise ValueError(msg)
    if times.size < 2 or not np.all(np.diff(times) > 0):
        msg = "times must contain at least two strictly increasing values"
        raise ValueError(msg)

    displacements = trajectories - trajectories[:, 0, np.newaxis]
    msd = np.mean(displacements**2, axis=0)

    start_time = times[0] + start_fraction * (times[-1] - times[0])
    end_time = times[0] + end_fraction * (times[-1] - times[0])
    fit_mask = (times >= start_time) & (times <= end_time)

    if np.count_nonzero(fit_mask) < 2:
        msg = "The selected long-time window contains too few samples"
        raise ValueError(msg)

    coefficients, covariance = np.polyfit(
        times[fit_mask],
        msd[fit_mask],
        deg=1,
        cov=True,
    )
    slope, intercept = coefficients
    diffusion_constant = slope / 2.0
    diffusion_uncertainty = float(np.sqrt(covariance[0, 0]) / 2.0)

    tangent = slope * times + intercept
    tangent_uncertainty = np.sqrt(
        times**2 * covariance[0, 0] + 2.0 * times * covariance[0, 1] + covariance[1, 1]
    )

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(times, msd, linewidth=2, label="Mean square displacement")
    axes.plot(
        times[fit_mask],
        tangent[fit_mask],
        linestyle="--",
        linewidth=2,
        label=f"Large-time tangent (D = {diffusion_constant:.3e})",
    )
    axes.fill_between(
        times[fit_mask],
        tangent[fit_mask] - tangent_uncertainty[fit_mask],
        tangent[fit_mask] + tangent_uncertainty[fit_mask],
        color="tab:orange",
        alpha=0.25,
        label="Fit uncertainty (1 sigma)",
    )
    axes.axvspan(
        times[fit_mask][0],
        times[fit_mask][-1],
        color="gray",
        alpha=0.12,
        label="Fitting window",
    )
    axes.set_xlabel("Time")
    axes.set_ylabel("Mean square displacement")
    axes.set_title("Long-time diffusion from the MSD tangent")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    return diffusion_constant, diffusion_uncertainty, msd, tangent, figure, axes


def run_tangent_diffusion(
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    start_fraction: float = 0.5,
    seed: int = 234,
    output_path: Path | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run trajectories, fit the large-time MSD, plot, and save the result."""
    times, trajectories = generate_trajectories(
        params=params,
        time_span=time_span,
        n_trajectories=n_trajectories,
        seed=seed,
    )

    diffusion_constant, _diffusion_uncertainty, msd, tangent, figure, _axes = (
        plot_msd_with_tangent(
            times,
            trajectories,
            start_fraction=start_fraction,
        )
    )

    if output_path is None:
        output_path = Path(__file__).with_name("msd_tangent.png")

    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Effective diffusion constant: {diffusion_constant:.6g}")
    print(f"Saved MSD plot to {output_path.resolve()}")

    plt.show()
    return diffusion_constant, msd, tangent


if __name__ == "__main__":
    params = KramersParameters(
        omega_well=1.0,
        omega_barrier=5.0,
        barrier_energy=1.0,
        m=1.0,
        temperature=0.5 / Boltzmann,
        gamma=0.1,
    )

    run_tangent_diffusion(
        params=params,
        time_span=TimeSpan(t_start=0.0, t_end=200.0, n_steps=2_000),
        n_trajectories=16,
        start_fraction=0.01,
    )
