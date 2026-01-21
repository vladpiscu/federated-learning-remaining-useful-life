import argparse
import glob
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import flwr as fl
from models.lstm_model import build_lstm_model


# Strategy that saves the final aggregated model (using FedProx)
class SavingFedProx(fl.server.strategy.FedProx):
    def __init__(self, *s_args, model_builder, out_model_path: str, **s_kwargs):
        super().__init__(*s_args, **s_kwargs)
        self._model_builder = model_builder
        self._out_model_path = out_model_path
        self._latest_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None
        # Flower returns either (Parameters, metrics) or just Parameters depending on version
        if isinstance(aggregated, tuple) and len(aggregated) == 2:
            params, metrics = aggregated
            self._latest_parameters = params
            return params, metrics
        self._latest_parameters = aggregated
        return aggregated

    def save_latest(self) -> None:
        if self._latest_parameters is None:
            print("Warning: No aggregated parameters to save.")
            return
        ndarrays = fl.common.parameters_to_ndarrays(self._latest_parameters)
        model = self._model_builder()
        model.set_weights(ndarrays)
        os.makedirs(os.path.dirname(self._out_model_path) or ".", exist_ok=True)
        model.save(self._out_model_path)
        print(f"\nSaved aggregated model to: {self._out_model_path}")


def aggregate_metrics(metrics):
    """
    Aggregate client metrics using weighted average (by number of samples).
    Also computes worst-case and average statistics.
    
    Args:
        metrics: List of tuples (num_samples, metrics_dict) from each client
    
    Returns:
        Aggregated metrics dictionary with average and worst-case metrics
    """
    if not metrics:
        return {}
    
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}
    
    # Get all metric names from first client
    metric_names = list(metrics[0][1].keys())
    
    aggregated = {}
    
    # Compute weighted averages
    for metric_name in metric_names:
        weighted_sum = sum(
            num_samples * metric_dict.get(metric_name, 0.0)
            for num_samples, metric_dict in metrics
        )
        aggregated[f"{metric_name}_avg"] = weighted_sum / total_samples
    
    # Compute worst-case (maximum for error metrics, minimum for R²)
    for metric_name in metric_names:
        if metric_name == "r2":
            # For R², worst-case is minimum (worst fit)
            worst_value = min(
                metric_dict.get(metric_name, float('inf'))
                for _, metric_dict in metrics
            )
        else:
            # For error metrics (MSE, MAE, RMSE), worst-case is maximum
            worst_value = max(
                metric_dict.get(metric_name, 0.0)
                for _, metric_dict in metrics
            )
        aggregated[f"{metric_name}_worst"] = worst_value
    
    # Also store average metrics without _avg suffix for compatibility
    for metric_name in metric_names:
        aggregated[metric_name] = aggregated[f"{metric_name}_avg"]
    
    return aggregated


def create_server_strategy(
    input_shape: tuple,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    proximal_mu: float,
    fraction_fit: float,
    min_fit_clients: int,
    min_available_clients: int,
    out_model_path: str,
):
    """
    Create a FedProx server strategy with model saving capability.
    This is a reusable function that can be called from simulation scripts.
    
    Args:
        input_shape: Model input shape (timesteps, features)
        num_rounds: Number of federated rounds
        local_epochs: Local epochs per client per round
        batch_size: Local batch size
        proximal_mu: FedProx proximal term weight
        fraction_fit: Fraction of clients to sample per round
        min_fit_clients: Minimum clients to sample per round
        min_available_clients: Minimum clients that must be available
        out_model_path: Path to save the final aggregated model
    
    Returns:
        Tuple of (strategy, initial_parameters, fit_config)
    """
    # Build initial model to get parameter structure
    init_model = build_lstm_model(input_shape=input_shape)
    initial_parameters = fl.common.ndarrays_to_parameters(init_model.get_weights())

    # Configure fit/evaluate config functions (include proximal_mu for FedProx)
    fit_config = {
        "local_epochs": int(local_epochs),
        "batch_size": int(batch_size),
        "proximal_mu": float(proximal_mu),
    }

    strategy = SavingFedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: fit_config,
        on_evaluate_config_fn=lambda rnd: fit_config,
        proximal_mu=float(proximal_mu),
        model_builder=lambda: build_lstm_model(input_shape=input_shape),
        out_model_path=out_model_path,
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )
    
    return strategy, initial_parameters, fit_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flower federated learning server for CMAPSS RUL prediction."
    )
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window size (timesteps).")
    parser.add_argument(
        "--num-features",
        type=int,
        default=None,
        help=(
            "Number of features (input shape). If not provided, will be inferred from first client. "
            "Should match the number of features after preprocessing (typically 24-26 for CMAPSS)."
        ),
    )
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round.")
    parser.add_argument("--batch-size", type=int, default=256, help="Local batch size.")
    parser.add_argument(
        "--proximal-mu",
        type=float,
        default=0.01,
        help="FedProx proximal term weight (μ). Higher values increase regularization. 0.0 = FedAvg.",
    )
    parser.add_argument(
        "--fraction-fit",
        type=float,
        default=1.0,
        help="Fraction of clients to sample for training each round (0.0 to 1.0).",
    )
    parser.add_argument(
        "--min-fit-clients",
        type=int,
        default=1,
        help="Minimum number of clients to sample for training each round.",
    )
    parser.add_argument(
        "--min-available-clients",
        type=int,
        default=1,
        help="Minimum number of clients that must be available before training starts.",
    )
    parser.add_argument(
        "--out-model",
        default=os.path.join("models", "federated_lstm_rul_model.h5"),
        help="Output path for the final aggregated global model.",
    )
    parser.add_argument("--server-address", default="0.0.0.0:8080", help="Server address (host:port).")
    return parser.parse_args()




def main():
    # Reduce TF log noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()

    print("=" * 60)
    print("Federated Learning Server - CMAPSS RUL Prediction (FedProx)")
    print("=" * 60)
    print("\nNote: Each client computes its own preprocessing parameters.")
    print("Clients must use deterministic feature selection to ensure model compatibility.")
    print(f"FedProx proximal term weight (μ): {args.proximal_mu}")

    # Determine input shape
    # If num_features is provided, use it; otherwise, infer from first client
    if args.num_features is not None:
        num_features = args.num_features
        input_shape = (int(args.window_size), int(num_features))
        print(f"\nUsing provided input shape: {input_shape} (timesteps={input_shape[0]}, features={input_shape[1]})")
    else:
        # Default: assume standard CMAPSS preprocessing (3 op_settings + ~14-21 sensors after variance filtering)
        # This is approximate; clients will report actual shape
        num_features = 24  # Conservative estimate (3 op_settings + 21 sensors, but some filtered)
        input_shape = (int(args.window_size), int(num_features))
        print(f"\nUsing estimated input shape: {input_shape}")
        print("Note: If clients have different feature counts, specify --num-features explicitly.")

    # Use the reusable function to create the strategy
    print("\nBuilding initial model...")
    strategy, initial_parameters, fit_config = create_server_strategy(
        input_shape=input_shape,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        proximal_mu=args.proximal_mu,
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        out_model_path=args.out_model,
    )

    # Parse server address
    host, port = args.server_address.rsplit(":", 1)
    port = int(port)

    print("\n" + "=" * 60)
    print(f"Starting Flower server on {host}:{port}")
    print(f"Waiting for clients to connect...")
    print("=" * 60)

    # Start Flower server
    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=int(args.num_rounds)),
        strategy=strategy,
    )

    # Save final aggregated model after training completes
    strategy.save_latest()
    print("\n" + "=" * 60)
    print("Server training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
