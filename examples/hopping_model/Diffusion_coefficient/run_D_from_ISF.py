"""Run ISF-based diffusion analysis with a chosen parameter set and q."""

from scipy.constants import Boltzmann
from classical_diffusion.jax.langevin import KramersParameters

from D_from_ISF import main


params = KramersParameters(
    omega_well=1.0,
    omega_barrier=5.0,
    barrier_energy=1.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)
q = 0.01


if __name__ == "__main__":
    main(params=params, q=q)
