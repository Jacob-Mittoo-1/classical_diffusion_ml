import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"


import jax.numpy as jnp
import numpy as np
from pysr import PySRRegressor
from scipy.constants import Boltzmann


def _kramers_rate(
    omega_well: jnp.ndarray,
    omega_barrier: jnp.ndarray,
    barrier_energy: jnp.ndarray,
    mass: jnp.ndarray,
    kbt: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    return ((omega_well * omega_barrier) / (2 * jnp.pi * gamma)) * jnp.exp(
        -barrier_energy / kbt
    )


def discover_clean_one_parameter_formula() -> None:
    """Ask PySR to recover the Kramers rate formula while varying omega_well."""
    omega_well = np.linspace(0.2, 2.0, 300)

    # Hold the other parameters fixed, matching the data generator.
    omega_barrier = 5.0
    barrier_energy = 1.0
    temperature = 0.5 / Boltzmann
    gamma = 0.1
    kbt = Boltzmann * temperature

    rates = ((omega_well * omega_barrier) / (2 * np.pi * gamma)) * np.exp(
        -barrier_energy / kbt
    )

    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp"],
        variable_names=["omega_well"],
    )
    model.fit(omega_well.reshape(-1, 1), rates)

    print("Discovered expression:", model.sympy())
    print("Equation candidates:")
    print(model.equations_)


if __name__ == "__main__":
    discover_clean_one_parameter_formula()
