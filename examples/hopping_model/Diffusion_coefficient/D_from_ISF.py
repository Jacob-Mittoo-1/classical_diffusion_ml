from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from classical_diffusion.analysis import get_pairwise_isf
from classical_diffusion.jax.langevin import (
    KramersParameters,
    KramersSystem1D,
)
from classical_diffusion.jax.langevin import (
    solve_many_overdamped as solve_many_overdamped_jax,
)
from classical_diffusion.simulation import TimeSpan
from scipy.constants import Boltzmann

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class ISFWindowFit:
    """Linear fit of log |ISF| over one time section."""

    section_index: int
    start_time: float
    end_time: float
    n_points: int
    slope: float
    intercept: float
    r_squared: float
    is_good: bool


def generate_trajectories(
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    seed: int = 234,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate trajectories for one fixed set of Kramers parameters."""
    system = KramersSystem1D(params=params).as_canonical()
    keys = jax.random.split(jax.random.PRNGKey(seed), n_trajectories)
    initial_conditions = (jnp.zeros((1, 1)), jnp.zeros((1, 1)))

    def run_one(key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        times, positions, _ = solve_many_overdamped_jax(
            system,
            time_span,
            initial_conditions,
            _key=key,
        )
        # Keep the coordinate axis for the ISF input: (dimension, time).
        return times, positions[0]

    times_by_run, trajectories = jax.vmap(run_one)(keys)
    return np.asarray(times_by_run[0]), np.asarray(trajectories)


def get_mean_isf(
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    delta_k: tuple[float, ...],
    seed: int = 234,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a fixed system and return lag times and the mean pairwise ISF."""
    times, trajectories = generate_trajectories(
        params=params,
        time_span=time_span,
        n_trajectories=n_trajectories,
        seed=seed,
    )
    isf_by_trajectory = get_pairwise_isf(
        trajectories,
        delta_k=delta_k,
    )
    mean_isf = np.mean(isf_by_trajectory, axis=0)

    lag_times = times - times[0]
    return lag_times, mean_isf


def plot_mean_isf(
    lag_times: np.ndarray,
    mean_isf: np.ndarray,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot the mean ISF linearly and its magnitude with a logarithmic y-axis."""
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(11, 4))

    ax_linear.plot(lag_times, mean_isf.real)
    ax_linear.set_xlabel("Time")
    ax_linear.set_ylabel("Mean ISF (real part)")
    ax_linear.set_title("Mean ISF")
    ax_linear.grid(True, alpha=0.3)

    ax_log.semilogy(lag_times, np.abs(mean_isf))
    ax_log.set_xlabel("Time")
    ax_log.set_ylabel("|Mean ISF|")
    ax_log.set_title("Mean ISF, logarithmic scale")
    ax_log.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig, (ax_linear, ax_log)


def fit_isf_sections(
    lag_times: np.ndarray,
    mean_isf: np.ndarray,
    *,
    n_sections: int = 20,
    min_r_squared: float = 0.98,
) -> list[ISFWindowFit]:
    """Fit lines to equal-duration sections of log |ISF|.

    Returns every section, with ``is_good`` indicating whether its R²
    meets ``min_r_squared``. This evaluates linearity only; it does not infer D.
    """
    times = np.asarray(lag_times, dtype=float)
    isf_magnitude = np.abs(np.asarray(mean_isf))
    if (
        times.ndim != 1
        or isf_magnitude.ndim != 1
        or times.shape != isf_magnitude.shape
    ):
        msg = "lag_times and mean_isf must be one-dimensional arrays of equal length."
        raise ValueError(msg)
    if n_sections < 1:
        msg = "n_sections must be at least 1."
        raise ValueError(msg)
    if len(times) < 2 * n_sections:
        msg = "Need at least two ISF samples per section."
        raise ValueError(msg)
    if not np.all(np.diff(times) > 0):
        msg = "lag_times must be strictly increasing."
        raise ValueError(msg)

    log_isf = np.full(isf_magnitude.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(isf_magnitude)
        & (isf_magnitude > 0)
        & np.isfinite(times)
    )
    log_isf[valid] = np.log(isf_magnitude[valid])

    section_edges = np.linspace(times[0], times[-1], n_sections + 1)
    fits = []
    for section_index in range(n_sections):
        start, end = section_edges[section_index : section_index + 2]
        in_section = (times >= start) & (
            (times < end) if section_index < n_sections - 1 else (times <= end)
        )
        section_times = times[in_section]
        section_log_isf = log_isf[in_section]
        valid = np.isfinite(section_times) & np.isfinite(section_log_isf)
        section_times = section_times[valid]
        section_log_isf = section_log_isf[valid]

        if len(section_times) < 2:
            fits.append(
                ISFWindowFit(
                    section_index=section_index,
                    start_time=float(start),
                    end_time=float(end),
                    n_points=len(section_times),
                    slope=float("nan"),
                    intercept=float("nan"),
                    r_squared=float("nan"),
                    is_good=False,
                )
            )
            continue

        slope, intercept = np.polyfit(section_times, section_log_isf, deg=1)
        predicted = slope * section_times + intercept
        residual_sum_squares = np.sum((section_log_isf - predicted) ** 2)
        total_sum_squares = np.sum(
            (section_log_isf - np.mean(section_log_isf)) ** 2
        )
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else float("nan")
        )
        fits.append(
            ISFWindowFit(
                section_index=section_index,
                start_time=float(section_times[0]),
                end_time=float(section_times[-1]),
                n_points=len(section_times),
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r_squared),
                is_good=bool(r_squared >= min_r_squared),
            )
        )

    return fits


def estimate_diffusion_from_sections(
    section_fits: list[ISFWindowFit],
    *,
    q: float,
    n_consistent_sections: int = 3,
) -> tuple[float | None, tuple[ISFWindowFit, ...] | None]:
    """Estimate D from the most consistent adjacent good-fit sections.

    Uses log(|F(q,t)|) ~= intercept - D*q**2*t. Returns ``(None, None)`` if
    no adjacent group of the requested size has good, decaying fits.
    """
    if q == 0:
        msg = "q must be nonzero to estimate the diffusion constant."
        raise ValueError(msg)
    if n_consistent_sections < 1:
        msg = "n_consistent_sections must be at least 1."
        raise ValueError(msg)

    best_group = None
    best_score = float("inf")
    for start in range(len(section_fits) - n_consistent_sections + 1):
        group = tuple(section_fits[start : start + n_consistent_sections])
        slopes = np.array([fit.slope for fit in group])
        if not all(fit.is_good for fit in group) or not np.all(slopes < 0):
            continue
        mean_slope = float(np.mean(slopes))
        relative_spread = float(np.std(slopes) / abs(mean_slope))
        if relative_spread < best_score:
            best_group = group
            best_score = relative_spread

    if best_group is None:
        return None, None

    mean_slope = float(np.mean([fit.slope for fit in best_group]))
    diffusion_constant = -mean_slope / q**2
    return diffusion_constant, best_group


def main(
    *,
    params: KramersParameters | None = None,
    q: float = 0.1,
) -> float | None:
    """Generate and plot the mean ISF, returning D when a fit is found."""
    if q == 0:
        msg = "q must be nonzero to estimate the diffusion constant."
        raise ValueError(msg)
    if params is None:
        params = KramersParameters(
            omega_well=1.0,
            omega_barrier=5.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
    lag_times, mean_isf = get_mean_isf(
        params=params,
        time_span=TimeSpan(t_start=0.0, t_end=100.0, n_steps=1_000),
        n_trajectories=128,
        delta_k=(q,),
    )
    fig, _ = plot_mean_isf(lag_times, mean_isf)
    fits = fit_isf_sections(lag_times, mean_isf)
    good_fits = [fit for fit in fits if fit.is_good]
    diffusion_constant, selected_sections = estimate_diffusion_from_sections(
        fits,
        q=q,
    )
    print(f"q={q:g}; {len(good_fits)}/{len(fits)} sections have R² >= 0.98.")
    for fit in fits:
        print(
            f"section {fit.section_index + 1:02d}: "
            f"t={fit.start_time:.3g}–{fit.end_time:.3g}, "
            f"slope={fit.slope:.5g}, R²={fit.r_squared:.5f}, "
            f"good={fit.is_good}"
        )
    if selected_sections is None:
        print("No group of three adjacent good, decaying sections was found.")
    else:
        selected_range = (
            f"sections {selected_sections[0].section_index + 1}–"
            f"{selected_sections[-1].section_index + 1}"
        )
        print(f"Selected {selected_range}; estimated D={diffusion_constant:.6g}.")
    output_path = Path(__file__).with_name("mean_isf.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ISF plot to {output_path.resolve()}")
    plt.show()
    return diffusion_constant


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=0.1, help="ISF wavevector magnitude")
    main(q=parser.parse_args().q)
