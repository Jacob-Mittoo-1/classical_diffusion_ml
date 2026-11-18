from __future__ import annotations

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


def main() -> None:
    """Generate and plot the mean ISF for the default hopping parameters."""
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
        n_trajectories=32,
        delta_k=(0.1,),
    )
    fig, _ = plot_mean_isf(lag_times, mean_isf)
    output_path = Path(__file__).with_name("mean_isf.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ISF plot to {output_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
