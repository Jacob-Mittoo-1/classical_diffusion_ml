import runpy
from pathlib import Path

import numpy as np
import torch
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path
from generate_training_data import generate_training_data

# Load train_model from the neighboring extensionless file.
trainer = runpy.run_path(str(Path(__file__).with_name("model_training")))
train_model = trainer["train_model"]


if __name__ == "__main__":
    # Generate or load the cached dataset.
    with cache_base_path(Path()):
        params, hop_times, num_hops = generate_training_data(
            time_span=TimeSpan(t_start=0.0, t_end=100.0, n_steps=1_000),
            n_trajectories=1_000,
        )

    if not (
        np.isfinite(params).all()
        and np.isfinite(hop_times).all()
        and np.isfinite(num_hops).all()
    ):
        msg = "Generated data contains NaN or infinite values."
        raise ValueError(msg)

    # Split trajectories reproducibly: 80% for training, 20% held out.
    rng = np.random.default_rng(1)
    indices = rng.permutation(len(params))
    split_index = int(0.8 * len(indices))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_params = params[train_indices]
    train_hop_times = hop_times[train_indices]
    train_num_hops = num_hops[train_indices]

    test_params = params[test_indices]
    test_hop_times = hop_times[test_indices]
    test_num_hops = num_hops[test_indices]

    print(
        f"Training on {len(train_indices)} trajectories; "
        f"evaluating on {len(test_indices)} held-out trajectories."
    )

    model = train_model((train_params, train_hop_times, train_num_hops))

    # Evaluate without computing gradients.
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        test_params_tensor = torch.as_tensor(
            test_params,
            dtype=torch.float32,
            device=device,
        )
        predictions = model(test_params_tensor).reshape(-1).cpu().numpy()

    errors = predictions - test_hop_times
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    weighted_mse = np.mean(errors**2 * test_num_hops)

    # Baseline: predict the average hop time from the training set.
    baseline_prediction = np.mean(train_hop_times)
    baseline_rmse = np.sqrt(np.mean((test_hop_times - baseline_prediction) ** 2))

    print(f"Test RMSE: {rmse:.6g}")
    print(f"Test MAE: {mae:.6g}")
    print(f"Hop-count-weighted test MSE: {weighted_mse:.6g}")
    print(f"Baseline test RMSE: {baseline_rmse:.6g}")

    print("\nFirst 10 held-out predictions:")
    for actual, predicted in zip(test_hop_times[:10], predictions[:10], strict=True):
        print(f"actual={actual:.6g}  predicted={predicted:.6g}")
