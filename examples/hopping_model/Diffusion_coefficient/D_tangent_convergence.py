from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from classical_diffusion.jax.langevin import KramersParameters
from classical_diffusion.simulation import TimeSpan
from D_from_tangent import (
    fit_msd_line,
    iter_trajectory_batches,
    time_averaged_msd_per_trajectory,
)
from D_tangent_vs_LJ import lifson_jackson_diffusion_from_params
from scipy.constants import Boltzmann


def _check_points(n_trajectories: int, check_every: int) -> list[int]:
    if not 1 <= check_every <= n_trajectories:
        msg = "check_every must satisfy 1 <= check_every <= n_trajectories"
        raise ValueError(msg)
    n_values = list(range(check_every, n_trajectories + 1, check_every))
    if n_values[-1] != n_trajectories:
        n_values.append(n_trajectories)
    return n_values


def _streaming_fits(  # noqa: PLR0913
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    check_every: int,
    seed: int,
    msd_fn: Callable[[np.ndarray], np.ndarray],
    d_lj: float,
    start_fraction: float,
    end_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate in batches; fit and print D as soon as each checkpoint is reached.

    ``msd_fn`` maps a batch of trajectories to per-trajectory MSD rows.
    """
    n_values = _check_points(n_trajectories, check_every)
    msd_rows: list[np.ndarray] = []
    n_done = 0
    diffusion: list[float] = []
    uncertainty: list[float] = []

    for times, batch in iter_trajectory_batches(
        params=params,
        time_span=time_span,
        n_trajectories=n_trajectories,
        seed=seed,
    ):
        msd_rows.append(msd_fn(batch))
        n_done += batch.shape[0]

        while len(diffusion) < len(n_values) and n_values[len(diffusion)] <= n_done:
            n = n_values[len(diffusion)]
            msd = np.concatenate(msd_rows)[:n].mean(axis=0)
            d, sigma = fit_msd_line(
                times - times[0],
                msd,
                start_fraction=start_fraction,
                end_fraction=end_fraction,
            )
            diffusion.append(d)
            uncertainty.append(sigma)
            print(
                f"n={n}/{n_trajectories}: D={d:.6g} ± {sigma:.6g} "
                f"({100 * (d - d_lj) / d_lj:+.1f}% vs LJ)",
                flush=True,
            )

    return np.asarray(n_values), np.asarray(diffusion), np.asarray(uncertainty)


def _plot_convergence(
    n_values: np.ndarray,
    diffusion: np.ndarray,
    uncertainty: np.ndarray,
    d_lj: float,
    *,
    title: str,
    output_path: Path,
) -> None:
    figure, (ax_d, ax_err) = plt.subplots(1, 2, figsize=(12, 5))

    ax_d.plot(n_values, diffusion, marker="o", linewidth=2, label="MSD tangent")
    ax_d.fill_between(
        n_values,
        diffusion - uncertainty,
        diffusion + uncertainty,
        alpha=0.25,
        label="Fit uncertainty (1 sigma)",
    )
    ax_d.axhline(d_lj, color="k", linestyle="--", linewidth=2, label="Lifson–Jackson")
    ax_d.set_xlabel("Number of trajectories")
    ax_d.set_ylabel("Diffusion constant")
    ax_d.set_title(title)
    ax_d.grid(visible=True, alpha=0.3)
    ax_d.legend()

    ax_err.loglog(n_values, np.abs(diffusion - d_lj) / d_lj, marker="o", linewidth=2)
    ax_err.set_xlabel("Number of trajectories")
    ax_err.set_ylabel(r"$|D - D_{\mathrm{LJ}}| / D_{\mathrm{LJ}}$")
    ax_err.set_title("Relative deviation from Lifson–Jackson")
    ax_err.grid(visible=True, which="both", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved convergence plot to {output_path.resolve()}")
    plt.show()


def run_convergence(
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    check_every: int,
    start_fraction: float = 0.2,
    end_fraction: float = 1.0,
    seed: int = 234,
    output_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Track convergence of the MSD-tangent D with the number of trajectories.

    The MSD is measured from t = 0 only. All trajectories are simulated once;
    after every ``check_every`` trajectories the MSD is re-averaged over the
    trajectories so far and refitted, then plotted against Lifson-Jackson.

    Returns
    -------
    n_values, diffusion, uncertainty, lifson_jackson
    """
    d_lj = lifson_jackson_diffusion_from_params(params)
    print(f"Lifson–Jackson D={d_lj:.6g}", flush=True)
    n_values, diffusion, uncertainty = _streaming_fits(
        params=params,
        time_span=time_span,
        n_trajectories=n_trajectories,
        check_every=check_every,
        seed=seed,
        msd_fn=lambda batch: (batch - batch[:, :1]) ** 2,
        d_lj=d_lj,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )

    _plot_convergence(
        n_values,
        diffusion,
        uncertainty,
        d_lj,
        title="Convergence of MSD-tangent D",
        output_path=output_path
        or Path(__file__).with_name("D_tangent_convergence.png"),
    )
    return n_values, diffusion, uncertainty, d_lj


def run_time_averaged_convergence(
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    check_every: int,
    start_fraction: float = 0.2,
    end_fraction: float = 0.5,
    seed: int = 234,
    output_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Same as ``run_convergence`` but using the time-averaged MSD.

    Each trajectory's MSD is averaged over every start time before averaging
    over trajectories. The fit window fractions refer to lag time; long lags
    have few start times, so by default only lags up to half the run are used.

    Returns
    -------
    n_values, diffusion, uncertainty, lifson_jackson
    """
    d_lj = lifson_jackson_diffusion_from_params(params)
    print(f"Lifson–Jackson D={d_lj:.6g}", flush=True)
    n_values, diffusion, uncertainty = _streaming_fits(
        params=params,
        time_span=time_span,
        n_trajectories=n_trajectories,
        check_every=check_every,
        seed=seed,
        msd_fn=time_averaged_msd_per_trajectory,
        d_lj=d_lj,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )

    _plot_convergence(
        n_values,
        diffusion,
        uncertainty,
        d_lj,
        title="Convergence of time-averaged MSD-tangent D",
        output_path=output_path
        or Path(__file__).with_name("D_tangent_convergence_time_averaged.png"),
    )
    return n_values, diffusion, uncertainty, d_lj


if __name__ == "__main__":
    params = KramersParameters(
        omega_well=1.0,
        omega_barrier=5.0,
        barrier_energy=1.0,
        m=1.0,
        temperature=0.5 / Boltzmann,
        gamma=0.1,
    )
    time_span = TimeSpan(t_start=0.0, t_end=200.0, n_steps=2_000)

    run_convergence(
        params=params,
        time_span=time_span,
        n_trajectories=128,
        check_every=8,
    )
    run_time_averaged_convergence(
        params=params,
        time_span=time_span,
        n_trajectories=128,
        check_every=8,
    )
