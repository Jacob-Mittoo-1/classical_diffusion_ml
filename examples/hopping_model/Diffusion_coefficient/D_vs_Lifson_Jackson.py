from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from classical_diffusion.hopping import get_lifson_jackson_rate
from classical_diffusion.jax.langevin import (
    KramersParameters,
)
from classical_diffusion.langevin import KramersSystem1D as ClassicalKramersSystem1D
from classical_diffusion.simulation import TimeSpan
from D_from_ISF import (
    estimate_diffusion_from_sections,
    fit_isf_sections,
    get_mean_isf,
)
from scipy.constants import Boltzmann


def compare_diffusion_constants(
    *,
    q: float = 0.1,
    n_omega_well: int = 10,
    n_trajectories: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare ISF-derived and Lifson–Jackson D across omega_well values."""
    if q == 0:
        msg = "q must be nonzero."
        raise ValueError(msg)
    if n_omega_well < 1:
        msg = "n_omega_well must be at least 1."
        raise ValueError(msg)

    omega_wells = np.linspace(0.2, 2.0, n_omega_well)
    isf_diffusion = np.full(omega_wells.shape, np.nan)
    lifson_jackson_diffusion = np.empty(omega_wells.shape)
    time_span = TimeSpan(t_start=0.0, t_end=800.0, n_steps=8_000)

    for index, omega_well in enumerate(omega_wells):
        params = KramersParameters(
            omega_well=float(omega_well),
            omega_barrier=5.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )

        lag_times, mean_isf = get_mean_isf(
            params=params,
            time_span=time_span,
            n_trajectories=n_trajectories,
            delta_k=(q,),
            seed=234,
        )
        section_fits = fit_isf_sections(lag_times, mean_isf)
        isf_value, selected_sections = estimate_diffusion_from_sections(
            section_fits,
            q=q,
        )
        if selected_sections is not None:
            isf_diffusion[index] = isf_value

        system = ClassicalKramersSystem1D(params=params)
        lifson_jackson_rate = get_lifson_jackson_rate(
            system,
            delta_x=params.delta_x,
        )
        # The Lifson–Jackson helper returns D / delta_x**2 (a hopping rate).
        lifson_jackson_diffusion[index] = lifson_jackson_rate * params.delta_x**2

        isf_text = (
            f"{isf_diffusion[index]:.6g}"
            if np.isfinite(isf_diffusion[index])
            else "no fit"
        )
        print(
            f"omega_well={omega_well:.3f}: ISF D={isf_text}, "
            f"Lifson–Jackson D={lifson_jackson_diffusion[index]:.6g}"
        )

    return omega_wells, isf_diffusion, lifson_jackson_diffusion


def main(
    *,
    q: float = 0.01,
    n_omega_well: int = 10,
    n_trajectories: int = 128,
) -> None:
    omega_wells, isf_diffusion, lifson_jackson_diffusion = compare_diffusion_constants(
        q=q,
        n_omega_well=n_omega_well,
        n_trajectories=n_trajectories,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        omega_wells,
        lifson_jackson_diffusion,
        marker="o",
        label="Lifson–Jackson",
    )
    ax.plot(
        omega_wells,
        isf_diffusion,
        marker="x",
        label="ISF estimate",
    )
    ax.set_xlabel(r"$\omega_{\mathrm{well}}$")
    ax.set_ylabel("Diffusion constant")
    ax.set_title(f"Diffusion vs. well frequency (q={q:g})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("D_vs_Lifson_Jackson.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q", type=float, default=0.01, help="ISF wavevector magnitude"
    )
    parser.add_argument("--n-omega-well", type=int, default=10)
    parser.add_argument("--n-trajectories", type=int, default=128)
    args = parser.parse_args()
    main(
        q=args.q,
        n_omega_well=args.n_omega_well,
        n_trajectories=args.n_trajectories,
    )
