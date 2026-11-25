"""Compare the MSD-tangent diffusion constant with Lifson-Jackson vs omega_well.

Change the settings in the ``__main__`` block at the bottom of the file.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from classical_diffusion.hopping import get_lifson_jackson_rate
from classical_diffusion.jax.langevin import KramersParameters
from classical_diffusion.langevin import KramersSystem1D as ClassicalKramersSystem1D
from classical_diffusion.simulation import TimeSpan
from D_from_tangent import (
    fit_msd_line,
    iter_trajectory_batches,
    time_averaged_msd_per_trajectory,
)
from scipy.constants import Boltzmann

type MsdKind = Literal["time-averaged", "origin"]

# Default fit window (fractions of the run) for each kind of MSD. Long lags of
# the time-averaged MSD have few start times, so only lags up to half are used.
DEFAULT_FIT_WINDOW: dict[MsdKind, tuple[float, float]] = {
    "time-averaged": (0.2, 0.5),
    "origin": (0.2, 1.0),
}

# Every parameter except omega_well, which is scanned.
BASE_PARAMS = KramersParameters(
    omega_well=1.0,
    omega_barrier=5.0,
    barrier_energy=1.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)


@dataclass(frozen=True)
class SimulationSettings:
    """How the trajectories are generated and D is fitted."""

    time_span: TimeSpan
    n_trajectories: int
    batch_size: int = 32
    step_size: float = 5e-4
    seed: int = 574
    msd: MsdKind = "time-averaged"
    start_fraction: float | None = None
    end_fraction: float | None = None

    @property
    def fit_window(self) -> tuple[float, float]:
        """The fit window, falling back to the default for this kind of MSD."""
        default_start, default_end = DEFAULT_FIT_WINDOW[self.msd]
        return (
            default_start if self.start_fraction is None else self.start_fraction,
            default_end if self.end_fraction is None else self.end_fraction,
        )


def _msd_rows(batch: np.ndarray, kind: MsdKind) -> np.ndarray:
    """Per-trajectory MSD, indexed by lag (time-averaged) or time (origin)."""
    if kind == "time-averaged":
        return time_averaged_msd_per_trajectory(batch)
    return (batch - batch[:, :1]) ** 2


def tangent_diffusion_from_params(
    params: KramersParameters, settings: SimulationSettings
) -> tuple[float, float]:
    """MSD-tangent D and its standard error for one parameter set.

    The error is the spread of D fitted separately to each batch, divided by
    sqrt(number of batches). Unlike the straight-line fit error, this accounts
    for MSD points at nearby times being correlated. With a single batch the
    fit error is returned instead.
    """
    start_fraction, end_fraction = settings.fit_window
    batch_msds: list[np.ndarray] = []
    batch_sizes: list[int] = []
    lag_times = np.empty(0)

    for times, batch in iter_trajectory_batches(
        params=params,
        time_span=settings.time_span,
        n_trajectories=settings.n_trajectories,
        batch_size=settings.batch_size,
        seed=settings.seed,
        step_size=settings.step_size,
    ):
        lag_times = times - times[0]
        batch_msds.append(_msd_rows(batch, settings.msd).mean(axis=0))
        batch_sizes.append(batch.shape[0])

    msd = np.average(batch_msds, axis=0, weights=batch_sizes)
    diffusion, fit_error = fit_msd_line(
        lag_times, msd, start_fraction=start_fraction, end_fraction=end_fraction
    )
    if len(batch_msds) < 2:  # noqa: PLR2004
        return diffusion, fit_error

    batch_diffusion = np.array(
        [
            fit_msd_line(
                lag_times,
                batch_msd,
                start_fraction=start_fraction,
                end_fraction=end_fraction,
            )[0]
            for batch_msd in batch_msds
        ]
    )
    standard_error = batch_diffusion.std(ddof=1) / np.sqrt(len(batch_diffusion))
    return diffusion, float(standard_error)


def lifson_jackson_diffusion_from_params(params: KramersParameters) -> float:
    """Lifson–Jackson diffusion constant D = k_LJ * delta_x**2 for one parameter set."""
    system = ClassicalKramersSystem1D(params=params)
    hopping_rate = get_lifson_jackson_rate(system, delta_x=params.delta_x)
    return float(hopping_rate * params.delta_x**2)


def compare_vs_omega_well(
    omega_wells: np.ndarray,
    settings: SimulationSettings,
    base_params: KramersParameters = BASE_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MSD-tangent D, its error, and Lifson–Jackson D for each omega_well."""
    tangent = np.empty(len(omega_wells))
    error = np.empty(len(omega_wells))
    lifson_jackson = np.empty(len(omega_wells))
    run_start = time.perf_counter()

    for index, omega_well in enumerate(omega_wells):
        params = dataclasses.replace(base_params, omega_well=float(omega_well))
        point_start = time.perf_counter()
        tangent[index], error[index] = tangent_diffusion_from_params(params, settings)
        lifson_jackson[index] = lifson_jackson_diffusion_from_params(params)

        now = time.perf_counter()
        deviation = (
            100 * (tangent[index] - lifson_jackson[index]) / lifson_jackson[index]
        )
        print(
            f"[{index + 1}/{len(omega_wells)}] omega_well={omega_well:.3f}: "
            f"D={tangent[index]:.6g} ± {error[index]:.2g}, "
            f"LJ={lifson_jackson[index]:.6g} ({deviation:+.1f}%) "
            f"[{now - point_start:.1f}s, total {now - run_start:.1f}s]",
            flush=True,
        )

    return tangent, error, lifson_jackson


def plot_diffusion_comparison(  # noqa: PLR0913
    *,
    omega_wells: np.ndarray,
    tangent_diffusion: np.ndarray,
    tangent_uncertainty: np.ndarray,
    lifson_jackson_diffusion: np.ndarray,
    settings: SimulationSettings,
    output_path: Path | None = None,
) -> None:
    """Plot MSD-tangent and Lifson–Jackson diffusion against omega_well."""
    figure, (ax_d, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, height_ratios=(2, 1)
    )
    ax_d.plot(
        omega_wells,
        lifson_jackson_diffusion,
        marker="o",
        linewidth=2,
        label="Lifson–Jackson",
    )
    ax_d.errorbar(
        omega_wells,
        tangent_diffusion,
        yerr=tangent_uncertainty,
        marker="x",
        linewidth=2,
        capsize=4,
        label=f"MSD tangent ({settings.msd}, 1 s.e.)",
    )
    ax_d.set_ylabel("Diffusion constant")
    ax_d.set_title(
        f"MSD tangent vs Lifson–Jackson "
        f"(N={settings.n_trajectories}, t_end={settings.time_span.t_end:g}, "
        f"dt={settings.step_size:g})"
    )
    ax_d.grid(visible=True, alpha=0.3)
    ax_d.legend()

    ax_ratio.errorbar(
        omega_wells,
        tangent_diffusion / lifson_jackson_diffusion,
        yerr=tangent_uncertainty / lifson_jackson_diffusion,
        marker="x",
        linewidth=2,
        capsize=4,
    )
    ax_ratio.axhline(1.0, color="k", linestyle="--")
    ax_ratio.set_xlabel(r"$\omega_{\mathrm{well}}$")
    ax_ratio.set_ylabel(r"$D / D_{\mathrm{LJ}}$")
    ax_ratio.grid(visible=True, alpha=0.3)
    figure.tight_layout()

    if output_path is None:
        output_path = Path(__file__).with_name("D_tangent_vs_Lifson_Jackson.png")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path.resolve()}")
    plt.show()


def main(
    omega_wells: np.ndarray,
    settings: SimulationSettings,
    output_path: Path | None = None,
) -> None:
    """Compare trajectory-derived and analytical diffusion constants."""
    tangent, error, lifson_jackson = compare_vs_omega_well(omega_wells, settings)
    plot_diffusion_comparison(
        omega_wells=omega_wells,
        tangent_diffusion=tangent,
        tangent_uncertainty=error,
        lifson_jackson_diffusion=lifson_jackson,
        settings=settings,
        output_path=output_path,
    )


if __name__ == "__main__":
    main(
        omega_wells=np.linspace(0.2, 2.0, 6),
        settings=SimulationSettings(
            # n_steps is the number of saved time points, not solver steps.
            time_span=TimeSpan(t_start=0.0, t_end=100.0, n_steps=1_000),
            n_trajectories=512,
            batch_size=32,
            step_size=1e-3,
            seed=567,
            msd="time-averaged",  # or "origin"
            # None uses the default window: 0.2-0.5 (time-averaged), 0.2-1.0 (origin)
            start_fraction=None,
            end_fraction=None,
        ),
    )
