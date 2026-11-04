import json
import runpy
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from classical_diffusion.hopping import KramersParameters, get_lifson_jackson_rate
from classical_diffusion.langevin import KramersSystem1D
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path
from generate_training_data import generate_training_data
from pysr import PySRRegressor


@dataclass
class PipelineConfig:
    """Configuration shared by the Kramers-rate learning pipeline."""

    time_span: TimeSpan = field(
        default_factory=lambda: TimeSpan(t_start=0.0, t_end=100.0, n_steps=1_000)
    )
    n_trajectories: int = 1_000
    train_fraction: float = 0.8
    random_seed: int = 1

    def __post_init__(self) -> None:
        if self.n_trajectories < 2:
            msg = "n_trajectories must be at least 2 to split the data."
            raise ValueError(msg)
        if not 0.0 < self.train_fraction < 1.0:
            msg = "train_fraction must be between 0 and 1."
            raise ValueError(msg)


def generate_data(
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate or load trajectory parameters, hop times, and hop counts."""
    with cache_base_path(Path()):
        return generate_training_data(
            time_span=config.time_span,
            n_trajectories=config.n_trajectories,
        )


def prepare_data(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter invalid trajectories and convert hop times to rates."""
    params, hop_times, num_hops = (np.asarray(values) for values in data)

    valid = (
        np.isfinite(params).all(axis=1)
        & np.isfinite(hop_times)
        & (hop_times > 0)
        & np.isfinite(num_hops)
        & (num_hops > 0)
    )

    removed = len(valid) - np.count_nonzero(valid)
    if not np.any(valid):
        msg = "No valid trajectories remain after filtering."
        raise ValueError(msg)

    print(f"Removed {removed} invalid trajectories.")

    valid_params = params[valid]
    rates = 1.0 / hop_times[valid]
    valid_num_hops = num_hops[valid]

    return valid_params, rates, valid_num_hops


def split_data(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: PipelineConfig,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Shuffle and split parameters, rates, and hop counts into train and test sets."""
    params, rates, num_hops = data
    n_samples = len(rates)

    if n_samples < 2:
        msg = "At least two valid trajectories are needed to split the data."
        raise ValueError(msg)

    rng = np.random.default_rng(config.random_seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * config.train_fraction)
    n_train = min(max(n_train, 1), n_samples - 1)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_data = (
        params[train_indices],
        rates[train_indices],
        num_hops[train_indices],
    )
    test_data = (
        params[test_indices],
        rates[test_indices],
        num_hops[test_indices],
    )

    return train_data, test_data


def train_and_plot(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> torch.nn.Module:
    """Train the hop-time model and save its training-loss plot."""
    params, rates, num_hops = train_data
    hop_times = 1.0 / rates

    trainer = runpy.run_path(str(Path(__file__).with_name("model_training")))
    train_model = trainer["train_model"]
    model = train_model((params, hop_times, num_hops))

    checkpoint_path = Path(__file__).with_name("model_checkpoint.pt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    losses = checkpoint["losses"]

    output_dir = Path(__file__).with_name("pipeline_outputs")
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss over time")
    fig.savefig(output_dir / "training_loss.png", bbox_inches="tight")
    plt.close(fig)

    return model


def _plot_rate_comparison(
    model: torch.nn.Module,
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path,
) -> None:
    """Plot measured, model-predicted, and Lifson-Jackson rates by omega_well."""
    params, measured_rates, _num_hops = test_data
    sort_indices = np.argsort(params[:, 0])
    sorted_params = params[sort_indices]
    sorted_rates = measured_rates[sort_indices]

    device = next(model.parameters()).device
    sorted_params_tensor = torch.as_tensor(
        sorted_params,
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        predicted_hop_times = model(sorted_params_tensor).reshape(-1).cpu().numpy()

    if np.any(predicted_hop_times <= 0):
        msg = "The model predicted nonpositive hop times, so rates cannot be plotted."
        raise ValueError(msg)
    predicted_rates = 1.0 / predicted_hop_times

    lifson_jackson_rates = []
    for row in sorted_params:
        kramers_params = KramersParameters(
            omega_well=float(row[0]),
            omega_barrier=float(row[1]),
            barrier_energy=float(row[2]),
            m=float(row[3]),
            temperature=float(row[4]),
            gamma=float(row[5]),
        )
        system = KramersSystem1D(params=kramers_params)
        lifson_jackson_rates.append(
            get_lifson_jackson_rate(system, delta_x=kramers_params.delta_x)
        )
    lifson_jackson_rates = np.asarray(lifson_jackson_rates)

    bin_count = max(1, len(sorted_rates) // 30)
    rate_bins = np.array_split(sorted_rates, bin_count)
    omega_bins = np.array_split(sorted_params[:, 0], bin_count)
    binned_omegas = np.array([np.mean(values) for values in omega_bins])
    binned_rates = np.array([np.mean(values) for values in rate_bins])

    fig, ax = plt.subplots()
    ax.scatter(binned_omegas, binned_rates, marker="x", label="Trajectory rates")
    ax.plot(sorted_params[:, 0], predicted_rates, label="Model rates")
    ax.plot(
        sorted_params[:, 0],
        lifson_jackson_rates,
        label="Lifson-Jackson rates",
    )
    ax.set_xlabel("Omega well")
    ax.set_ylabel("Rate")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.savefig(output_dir / "rate_comparison.png", bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model: torch.nn.Module,
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Evaluate hop-time predictions and save an actual-versus-predicted plot."""
    params, rates, _num_hops = test_data
    actual_hop_times = 1.0 / rates

    device = next(model.parameters()).device
    test_params = torch.as_tensor(params, dtype=torch.float32, device=device)

    model.eval()
    with torch.inference_mode():
        predicted_hop_times = model(test_params).reshape(-1).cpu().numpy()

    rmse = float(np.sqrt(np.mean((predicted_hop_times - actual_hop_times) ** 2)))
    mae = float(np.mean(np.abs(predicted_hop_times - actual_hop_times)))

    output_dir = Path(__file__).with_name("pipeline_outputs")
    output_dir.mkdir(exist_ok=True)

    _plot_rate_comparison(model, test_data, output_dir)

    fig, ax = plt.subplots()
    ax.scatter(actual_hop_times, predicted_hop_times, alpha=0.6)

    limits = [
        min(actual_hop_times.min(), predicted_hop_times.min()),
        max(actual_hop_times.max(), predicted_hop_times.max()),
    ]
    ax.plot(limits, limits, "k--", label="Perfect prediction")
    ax.set_xlabel("Measured hop time")
    ax.set_ylabel("Predicted hop time")
    ax.set_title(f"Test predictions (RMSE={rmse:.4g})")
    ax.legend()
    fig.savefig(output_dir / "evaluation_hop_times.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Test hop-time RMSE: {rmse:.6g}")
    print(f"Test hop-time MAE: {mae:.6g}")

    return {"rmse": rmse, "mae": mae}


def discover_expression(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    niterations: int = 40,
) -> tuple[str, float]:
    """Fit a symbolic rate expression and save it with its evaluation outputs."""
    train_params, train_rates, _ = train_data
    test_params, test_rates, _ = test_data

    # The current generator varies only omega_well.
    x_train = train_params[:, 0:1]
    x_test = test_params[:, 0:1]

    output_dir = Path(__file__).with_name("pipeline_outputs")
    output_dir.mkdir(exist_ok=True)

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp"],
        variable_names=["omega_well"],
        output_directory=str(output_dir / "pysr"),
    )
    model.fit(x_train, train_rates)

    expression = str(model.sympy())
    predicted_rates = model.predict(x_test)
    rmse = float(np.sqrt(np.mean((predicted_rates - test_rates) ** 2)))

    (output_dir / "symbolic_expression.txt").write_text(
        f"{expression}\n",
        encoding="utf-8",
    )
    model.equations_.to_csv(output_dir / "symbolic_equations.csv", index=False)
    np.savetxt(
        output_dir / "symbolic_test_predictions.csv",
        np.column_stack((x_test[:, 0], test_rates, predicted_rates)),
        delimiter=",",
        header="omega_well,measured_rate,symbolic_rate",
        comments="",
    )
    (output_dir / "symbolic_metrics.json").write_text(
        json.dumps({"held_out_rate_rmse": rmse}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Discovered expression: {expression}")
    print(f"Held-out rate RMSE: {rmse:.6g}")
    print(f"Saved symbolic outputs to {output_dir}")

    return expression, rmse


def run_pipeline(
    config: PipelineConfig | None = None,
) -> tuple[torch.nn.Module, dict[str, float], str]:
    """Run the complete data, training, evaluation, and expression workflow."""
    if config is None:
        config = PipelineConfig()

    raw_data = generate_data(config)
    prepared_data = prepare_data(raw_data)
    train_data, test_data = split_data(prepared_data, config)

    model = train_and_plot(train_data)
    metrics = evaluate_model(model, test_data)

    expression, rate_rmse = discover_expression(train_data, test_data)
    metrics["symbolic_rate_rmse"] = rate_rmse

    return model, metrics, expression


if __name__ == "__main__":
    run_pipeline()
