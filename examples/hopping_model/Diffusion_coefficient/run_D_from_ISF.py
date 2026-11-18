"""Generate and plot a mean ISF for a chosen system and simulation span."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from classical_diffusion.hopping import get_lifson_jackson_rate
from classical_diffusion.jax.langevin import KramersParameters
from classical_diffusion.langevin import KramersSystem1D as ClassicalKramersSystem1D
from classical_diffusion.simulation import TimeSpan
from D_from_ISF import (
    estimate_diffusion_from_sections,
    fit_isf_sections,
    get_mean_isf,
    plot_mean_isf,
)
from scipy.constants import Boltzmann

if TYPE_CHECKING:
    import numpy as np

params = KramersParameters(
    omega_well=0.2,
    omega_barrier=5.0,
    barrier_energy=1.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)
q = 0.025


def run_isf(
    *,
    params: KramersParameters,
    time_span: TimeSpan,
    n_trajectories: int,
    q: float,
    seed: int = 234,
    output_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and plot one ensemble-mean ISF and its log-scale magnitude."""
    lag_times, mean_isf = get_mean_isf(
        params=params,
        time_span=time_span,
        n_trajectories=n_trajectories,
        delta_k=(q,),
        seed=seed,
    )
    fig, _ = plot_mean_isf(lag_times, mean_isf)

    section_fits = fit_isf_sections(lag_times, mean_isf)
    isf_diffusion, selected_sections = estimate_diffusion_from_sections(
        section_fits,
        q=q,
    )
    if selected_sections is None:
        print("ISF-fit diffusion constant: no qualifying group of sections found.")
    else:
        print(f"ISF-fit diffusion constant: {isf_diffusion:.6g}")

    system = ClassicalKramersSystem1D(params=params)
    lifson_jackson_rate = get_lifson_jackson_rate(
        system,
        delta_x=params.delta_x,
    )
    lifson_jackson_diffusion = lifson_jackson_rate * params.delta_x**2
    print(f"Lifson–Jackson diffusion constant: {lifson_jackson_diffusion:.6g}")

    if output_path is None:
        output_path = Path(__file__).with_name(f"mean_isf_q_{q:g}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ISF plot to {output_path.resolve()}")
    plt.show()
    return lag_times, mean_isf


if __name__ == "__main__":
    run_isf(
        params=params,
        time_span=TimeSpan(t_start=0.0, t_end=100.0, n_steps=1_000),
        n_trajectories=128,
        q=q,
    )
