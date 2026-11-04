"""Compare simulated hopping rates with the overdamped Kramers formula."""

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


def compare_kramers_rates() -> None:
    """Simulate a high-barrier, high-friction Kramers system over omega_well."""
    # These settings give a barrier of 4 kBT and friction well above the
    # characteristic frequencies. Increase trajectory count or duration to
    # reduce the sampling error in the empirical rates.
    omega_wells = np.linspace(0.5, 2.0, 9)
    omega_barrier = 1.0
    barrier_energy = 0.5
    mass = 1.0
    kbt = 0.5
    temperature = kbt
    gamma = 2.0

    n_trajectories = 128
    time_span = TimeSpan(t_start=0.0, t_end=2000.0, n_steps=4000)
    initial_positions = jnp.zeros((n_trajectories, 1))
    initial_conditions = (initial_positions, initial_positions)
    keys = jax.random.split(jax.random.PRNGKey(123), len(omega_wells))

    simulated_rates = []
    simulated_rate_errors = []
    analytical_rates = []

    for omega_well, key in zip(omega_wells, keys, strict=True):
        params = KramersParameters(
            omega_well=float(omega_well),
            omega_barrier=omega_barrier,
            barrier_energy=barrier_energy,
            m=mass,
            temperature=temperature,
            gamma=gamma,
        )
        system = KramersSystem1D(params=params).as_canonical()

        _times, positions, _ = solve_many_overdamped(
            system,
            time_span,
            initial_conditions,
            _key=key,
        )
        # solve_many_overdamped returns (trajectory, dimension, time).
        trajectories = np.asarray(positions[:, 0, :])

        # A well is centered at each integer multiple of delta_x. Counting
        # changes in well index counts crossings of the intervening barriers.
        well_indices = np.floor(trajectories / params.delta_x + 0.5).astype(int)
        crossing_counts = np.abs(np.diff(well_indices, axis=1)).sum(axis=1)

        # The Kramers formula is the escape rate over one barrier. Crossings
        # include both left and right directions, so divide the total by 2.
        per_trajectory_rates = crossing_counts / (2 * time_span.t_end)
        simulated_rates.append(np.mean(per_trajectory_rates))
        simulated_rate_errors.append(
            np.std(per_trajectory_rates, ddof=1) / np.sqrt(n_trajectories)
        )

        analytical_rates.append(
            (omega_well * omega_barrier)
            / (2 * np.pi * gamma)
            * np.exp(-barrier_energy / kbt)
        )

        print(
            f"omega_well={omega_well:.3f}: "
            f"simulation={simulated_rates[-1]:.6g} +/- "
            f"{simulated_rate_errors[-1]:.2g}, "
            f"Kramers={analytical_rates[-1]:.6g}"
        )

    output_dir = Path(__file__).parent
    fig, ax = plt.subplots()
    ax.errorbar(
        omega_wells,
        simulated_rates,
        yerr=simulated_rate_errors,
        fmt="o",
        capsize=3,
        label="Overdamped Langevin simulation",
    )
    ax.plot(omega_wells, analytical_rates, "-", label="Kramers analytical rate")
    ax.set_xlabel("Omega well")
    ax.set_ylabel("Hopping rate")
    ax.set_title("Simulation and overdamped Kramers rates")
    ax.legend()
    fig.savefig(output_dir / "kramers_rates_versus_analytical.png", dpi=300)
    plt.close(fig)

    np.savetxt(
        output_dir / "kramers_rates_versus_analytical.csv",
        np.column_stack(
            (omega_wells, simulated_rates, simulated_rate_errors, analytical_rates)
        ),
        delimiter=",",
        header="omega_well,simulation_rate,simulation_standard_error,kramers_rate",
        comments="",
    )


if __name__ == "__main__":
    compare_kramers_rates()
