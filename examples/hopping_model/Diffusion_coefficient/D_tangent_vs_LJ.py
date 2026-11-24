from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from classical_diffusion.hopping import get_lifson_jackson_rate
from classical_diffusion.jax.langevin import KramersParameters
from classical_diffusion.langevin import KramersSystem1D as ClassicalKramersSystem1D
from classical_diffusion.simulation import TimeSpan
from D_from_tangent import generate_trajectories, plot_msd_with_tangent
from scipy.constants import Boltzmann


def get_tangent_diffusion_vs_omega_well(
    *,
    omega_wells: np.ndarray,
    time_span: TimeSpan,
    n_trajectories: int,
    start_fraction: float = 0.01,
    seed: int = 234,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate MSD-tangent diffusion and uncertainty."""
    tangent_diffusion = np.empty(len(omega_wells), dtype=float)
    tangent_uncertainty = np.empty(len(omega_wells), dtype=float)

    for index, omega_well in enumerate(omega_wells):
        params = KramersParameters(
            omega_well=float(omega_well),
            omega_barrier=5.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )

        times, trajectories = generate_trajectories(
            params=params,
            time_span=time_span,
            n_trajectories=n_trajectories,
            seed=seed,
        )
        diffusion_constant, diffusion_uncertainty, _msd, _tangent, figure, _axes = (
            plot_msd_with_tangent(
                times,
                trajectories,
                start_fraction=start_fraction,
            )
        )
        tangent_diffusion[index] = diffusion_constant
        tangent_uncertainty[index] = diffusion_uncertainty
        print(
            f"omega_well={float(omega_well):.3f}: "
            f"MSD-tangent D={diffusion_constant:.6g} "
            f"± {diffusion_uncertainty:.6g}"
        )

        plt.close(figure)

    return tangent_diffusion, tangent_uncertainty


def lifson_jackson_diffusion_from_params(params: KramersParameters) -> float:
    """Lifson–Jackson diffusion constant D = k_LJ * delta_x**2 for one parameter set."""
    system = ClassicalKramersSystem1D(params=params)
    hopping_rate = get_lifson_jackson_rate(system, delta_x=params.delta_x)
    return float(hopping_rate * params.delta_x**2)


def get_lifson_jackson_diffusion(
    *,
    omega_wells: np.ndarray,
) -> np.ndarray:
    """Calculate Lifson–Jackson diffusion for several well frequencies."""
    lifson_jackson_diffusion = np.empty(len(omega_wells), dtype=float)

    for index, omega_well in enumerate(omega_wells):
        params = KramersParameters(
            omega_well=float(omega_well),
            omega_barrier=5.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
        lifson_jackson_diffusion[index] = lifson_jackson_diffusion_from_params(params)

        print(
            f"omega_well={float(omega_well):.3f}: "
            f"Lifson–Jackson D={lifson_jackson_diffusion[index]:.6g}"
        )

    return lifson_jackson_diffusion


def plot_diffusion_comparison(
    *,
    omega_wells: np.ndarray,
    tangent_diffusion: np.ndarray,
    tangent_uncertainty: np.ndarray,
    lifson_jackson_diffusion: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Plot MSD-tangent and Lifson–Jackson diffusion against omega_well."""
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(
        omega_wells,
        lifson_jackson_diffusion,
        marker="o",
        linewidth=2,
        label="Lifson–Jackson",
    )
    axes.errorbar(
        omega_wells,
        tangent_diffusion,
        yerr=tangent_uncertainty,
        marker="x",
        linewidth=2,
        capsize=4,
        label="MSD tangent (1 sigma)",
    )
    axes.set_xlabel(r"$\omega_{\mathrm{well}}$")
    axes.set_ylabel("Diffusion constant")
    axes.set_title("MSD tangent diffusion compared with Lifson–Jackson")
    axes.grid(visible=True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    if output_path is None:
        output_path = Path(__file__).with_name("D_tangent_vs_Lifson_Jackson.png")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path.resolve()}")
    plt.show()


def main(
    *,
    n_omega_well: int = 10,
    n_trajectories: int = 64,
) -> None:
    """Compare trajectory-derived and analytical diffusion constants."""
    omega_wells = np.linspace(0.2, 2.0, n_omega_well)
    time_span = TimeSpan(t_start=0.0, t_end=100.0, n_steps=1_000)

    tangent_diffusion, tangent_uncertainty = get_tangent_diffusion_vs_omega_well(
        omega_wells=omega_wells,
        time_span=time_span,
        n_trajectories=n_trajectories,
        start_fraction=0.01,
    )
    lifson_jackson_diffusion = get_lifson_jackson_diffusion(
        omega_wells=omega_wells,
    )
    plot_diffusion_comparison(
        omega_wells=omega_wells,
        tangent_diffusion=tangent_diffusion,
        tangent_uncertainty=tangent_uncertainty,
        lifson_jackson_diffusion=lifson_jackson_diffusion,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-omega-well", type=int, default=6)
    parser.add_argument("--n-trajectories", type=int, default=64)
    arguments = parser.parse_args()
    main(
        n_omega_well=arguments.n_omega_well,
        n_trajectories=arguments.n_trajectories,
    )
