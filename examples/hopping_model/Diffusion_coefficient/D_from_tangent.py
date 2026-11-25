from __future__ import annotations

import os

# JAX otherwise preallocates nearly all available GPU memory before execution.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from classical_diffusion.jax.langevin import (
    FixedStep,
    KramersParameters,
    KramersSystem1D,
    solve_many_overdamped,
)
from classical_diffusion.simulation import TimeSpan
from scipy.constants import Boltzmann

if TYPE_CHECKING:
    from collections.abc import Iterator


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
    batches = list(
        iter_trajectory_batches(
            params=params,
            time_span=time_span,
            n_trajectories=n_trajectories,
            seed=seed,
        )
    )
    times = batches[0][0]
    return times, np.concatenate([batch for _times, batch in batches], axis=0)


def iter_trajectory_batches(  # noqa: PLR0913
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    batch_size: int = 32,
    seed: int = 234,
    step_size: float = 5e-4,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(times, trajectories)`` in batches of ``batch_size`` trajectories.

    The random keys are fixed up front from ``seed``, so the trajectories are
    identical to those from ``generate_trajectories`` with the same seed.
    The SDE is integrated with fixed steps of ``step_size``, which must be
    small enough to resolve the fastest motion of the system.
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
            step_config=FixedStep(step_size=step_size),
            _key=key,
        )

        # positions has shape (dimension, initial_condition, time).
        return times, positions[0, 0]

    # Run only a small batch at once so JAX does not reserve a 12 GiB GPU
    # workspace for all trajectories.
    for start in range(0, n_trajectories, batch_size):
        times_by_run, batch = jax.vmap(run_one)(keys[start : start + batch_size])
        yield np.asarray(times_by_run[0]), np.asarray(batch)


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

    diffusion_constant, diffusion_uncertainty = fit_msd_line(
        times,
        msd,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )
    return diffusion_constant, diffusion_uncertainty, msd


def fit_msd_line(
    times: np.ndarray,
    msd: np.ndarray,
    *,
    start_fraction: float,
    end_fraction: float,
) -> tuple[float, float]:
    """Fit a straight line to ``msd`` within a fractional time window.

    Returns ``(D, sigma_D)`` with D = slope / 2.
    """
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
    return float(slope / 2.0), float(np.sqrt(covariance[0, 0]) / 2.0)


def time_averaged_msd_per_trajectory(trajectories: np.ndarray) -> np.ndarray:
    """Time-averaged MSD of each trajectory, averaged over all start times.

    For lag index m, returns mean_k (x[k + m] - x[k])**2 over every start index
    k = 0 .. T - m - 1. Computed with the FFT method in O(T log T) per trajectory.
    Assumes equally spaced samples.

    Returns
    -------
    tamsd
        Array with shape ``(n_trajectories, n_time_samples)``; column m is lag m.
    """
    x = np.asarray(trajectories, dtype=float)
    if x.ndim != 2:
        msg = "trajectories must have shape (n_trajectories, n_time_samples)"
        raise ValueError(msg)
    n_times = x.shape[1]
    n_origins = n_times - np.arange(n_times)  # number of start times per lag

    # sum_k x[k+m]^2 + x[k]^2, built by removing end points as m grows
    squares = x**2
    removed = np.cumsum(squares + squares[:, ::-1], axis=1)
    sum_of_squares = np.empty_like(x)
    sum_of_squares[:, 0] = 2.0 * squares.sum(axis=1)
    sum_of_squares[:, 1:] = sum_of_squares[:, :1] - removed[:, :-1]

    # sum_k x[k+m] x[k], the autocorrelation, via zero-padded FFT
    spectrum = np.fft.rfft(x, n=2 * n_times, axis=1)
    autocorrelation = np.fft.irfft(spectrum * spectrum.conj(), axis=1)[:, :n_times]

    return (sum_of_squares - 2.0 * autocorrelation) / n_origins


def fit_time_averaged_diffusion(
    times: np.ndarray,
    trajectories: np.ndarray,
    *,
    start_fraction: float = 0.2,
    end_fraction: float = 0.5,
) -> tuple[float, float, np.ndarray]:
    """Fit D from the time-averaged MSD (all start times, then all trajectories).

    The window fractions refer to lag time. Long lags have few start times, so
    they are noisy; by default only lags up to half the run are fitted.

    Returns
    -------
    diffusion_constant, diffusion_uncertainty, tamsd
        ``tamsd`` is indexed by lag, with lag times ``times - times[0]``.
    """
    times = np.asarray(times, dtype=float)
    lag_times = times - times[0]
    tamsd = time_averaged_msd_per_trajectory(trajectories).mean(axis=0)
    diffusion_constant, diffusion_uncertainty = fit_msd_line(
        lag_times,
        tamsd,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )
    return diffusion_constant, diffusion_uncertainty, tamsd


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
    start_fraction: float = 0.01,
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
        omega_well=0.2,
        omega_barrier=5.0,
        barrier_energy=1.0,
        m=1.0,
        temperature=0.5 / Boltzmann,
        gamma=0.1,
    )

    run_tangent_diffusion(
        params=params,
        time_span=TimeSpan(t_start=0.0, t_end=100.0, n_steps=1_000),
        n_trajectories=512,
        start_fraction=0.01,
    )
