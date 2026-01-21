import argparse
import glob
import os
import sys
from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import flwr as fl
from models.lstm_model import build_lstm_model
from preprocessing.preprocessing import COLUMNS, preprocess_data

# Import client and server functions from federated files
from federated_client import create_client_for_data, create_client_for_unit, load_unit_data
from federated_server import create_server_strategy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Federated learning simulation using Flower's run_simulation. "
            "Splits the entire dataset across N clients. "
            "Makes unit_ids unique across files (like train.py)."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="CMAPSSData",
        help="Directory containing CMAPSSData training files.",
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
    parser.add_argument("--batch-size", type=int, default=32, help="Local batch size.")
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
        "--out-model",
        default=os.path.join("models", "federated_lstm_rul_model.h5"),
        help="Output path for the final aggregated global model.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of clients to create. The entire dataset will be split across these clients. Default: 2",
    )
    parser.add_argument(
        "--cpus-per-client",
        type=float,
        default=1.0,
        help="CPU resources per client in simulation.",
    )
    parser.add_argument(
        "--test-data-dir",
        default=None,
        help="Directory containing test data files. If provided, will evaluate clients and aggregated model on test data.",
    )
    return parser.parse_args()


def compute_global_feature_selection(data_dir: str) -> dict:
    """
    Compute global feature selection (which sensors to keep) from all training data.
    This ensures all clients use the same feature set for model compatibility.
    
    Returns:
        Dictionary with 'sensor_columns_to_keep' - the sensors that should be used by all clients
    """
    import pandas as pd

    train_files = sorted(glob.glob(os.path.join(data_dir, "train_*.txt")))

    if not train_files:
        raise FileNotFoundError(f"No training files found in: {data_dir}")

    print("\nComputing global feature selection from all training files...")
    all_dataframes = []
    max_unit_id = 0

    for train_file in train_files:
        df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMNS)
        # Adjust unit_ids to be unique across all files (same as train.py)
        df["unit_id"] = df["unit_id"] + max_unit_id
        max_unit_id = int(df["unit_id"].max())
        all_dataframes.append(df)

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Compute RUL for variance calculation
    combined_df["RUL"] = combined_df.groupby("unit_id")["time_cycles"].transform(
        lambda x: x.max() - x
    )
    
    # Define sensor columns
    sensor_columns = [col for col in COLUMNS if col.startswith("sensor")]
    
    # Variance filtering on all data (global)
    variance_threshold = 0.01
    column_variances = combined_df[sensor_columns].var()
    sensor_columns_to_keep = column_variances[column_variances >= variance_threshold].index.tolist()
    
    print(f"Global feature selection: keeping {len(sensor_columns_to_keep)} out of {len(sensor_columns)} sensors")
    print(f"Selected sensors: {sensor_columns_to_keep}")
    
    return {"sensor_columns_to_keep": sensor_columns_to_keep}


def load_all_training_data(data_dir: str):
    """
    Load all training files and combine them with unit_id offsetting (same logic as train.py).
    
    Returns:
        Combined DataFrame with all training data
    """
    import pandas as pd

    train_files = sorted(glob.glob(os.path.join(data_dir, "train_*.txt")))

    if not train_files:
        raise FileNotFoundError(f"No training files found in: {data_dir}")

    print(f"\nFound {len(train_files)} training file(s):")
    for f in train_files:
        print(f"  - {f}")

    print("\nLoading all training files...")
    all_dataframes = []
    max_unit_id = 0

    for train_file in train_files:
        print(f"  Loading: {train_file}")
        df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMNS)
        # Adjust unit_ids to be unique across all files (same as train.py)
        df["unit_id"] = df["unit_id"] + max_unit_id
        max_unit_id = int(df["unit_id"].max())
        all_dataframes.append(df)
        print(f"    Loaded {len(df)} rows, unit_ids: {df['unit_id'].min()}-{df['unit_id'].max()}")

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nTotal rows loaded: {len(combined_df)}")
    print(f"Total unique unit_ids: {len(combined_df['unit_id'].unique())}")
    print(f"Unit ID range: {combined_df['unit_id'].min()} - {combined_df['unit_id'].max()}")

    return combined_df


def evaluate_federated_model_on_test(
    model_path: str,
    test_data_dir: str,
    data_partitions: List,
    global_feature_selection: dict,
    window_size: int,
    input_shape: tuple,
):
    """
    Evaluate the federated model and individual clients on test data.
    Reports metrics per client, worst-case vs average, and aggregated model performance.
    """
    import pandas as pd
    import glob
    from tensorflow import keras
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from models.lstm_model import build_lstm_model, predict_rul
    from preprocessing.preprocessing import load_preprocessing_params, preprocess_data
    
    # Load aggregated model
    print(f"\nLoading aggregated model from: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print("Aggregated model loaded successfully!")
    except Exception as e:
        print(f"Error loading aggregated model: {e}")
        return
    
    # Find test files
    test_files = sorted(glob.glob(os.path.join(test_data_dir, "test_*.txt")))
    if not test_files:
        print(f"No test files found in: {test_data_dir}")
        return
    
    print(f"\nFound {len(test_files)} test file(s)")
    
    # Load and preprocess test data
    all_test_data = []
    for test_file in test_files:
        df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMNS)
        all_test_data.append(df)
    
    combined_test_df = pd.concat(all_test_data, ignore_index=True)
    
    # Preprocess test data with global feature selection
    preprocessing_params_for_test = {
        'sensor_columns_to_keep': global_feature_selection['sensor_columns_to_keep'],
    }
    
    print("\nPreprocessing test data...")
    X_test, _ = preprocess_data(
        df=combined_test_df,
        preprocessing_params=preprocessing_params_for_test,
        compute_rul=False,
        window_size=window_size,
    )
    
    print(f"Test data shape: {X_test.shape}")
    
    # Make predictions with aggregated model
    print("\nMaking predictions with aggregated model...")
    y_pred_aggregated = predict_rul(model, X_test)
    
    # For test data, we need true RUL values
    # Since test data doesn't have RUL, we'll evaluate clients on their validation sets
    # and the aggregated model on a combined validation set
    
    print("\n" + "=" * 60)
    print("Client Performance on Validation Data")
    print("=" * 60)
    
    # Evaluate each client on its validation data
    client_metrics = []
    
    for i, partition_df in enumerate(data_partitions):
        print(f"\nEvaluating Client {i}...")
        
        # Create client to get its validation data
        from federated_client import create_client_for_data
        client = create_client_for_data(
            data_df=partition_df,
            window_size=window_size,
            val_split=0.2,
            input_shape=input_shape,
            global_feature_selection=global_feature_selection,
        )
        
        # Evaluate client's model on its validation set
        if client.X_val is not None and len(client.X_val) > 0:
            y_pred_client = client.model.predict(client.X_val, verbose=0).flatten()
            y_true_client = client.y_val.flatten()
            
            mse = mean_squared_error(y_true_client, y_pred_client)
            mae = mean_absolute_error(y_true_client, y_pred_client)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_client, y_pred_client)
            
            client_metrics.append({
                'client_id': i,
                'num_samples': len(client.X_val),
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
            })
            
            print(f"  Client {i}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # Compute average and worst-case metrics
    if client_metrics:
        print("\n" + "=" * 60)
        print("Client Performance Summary")
        print("=" * 60)
        
        total_samples = sum(m['num_samples'] for m in client_metrics)
        
        # Weighted averages
        mse_avg = sum(m['mse'] * m['num_samples'] for m in client_metrics) / total_samples
        mae_avg = sum(m['mae'] * m['num_samples'] for m in client_metrics) / total_samples
        rmse_avg = sum(m['rmse'] * m['num_samples'] for m in client_metrics) / total_samples
        r2_avg = sum(m['r2'] * m['num_samples'] for m in client_metrics) / total_samples
        
        # Worst-case (maximum for errors, minimum for R²)
        mse_worst = max(m['mse'] for m in client_metrics)
        mae_worst = max(m['mae'] for m in client_metrics)
        rmse_worst = max(m['rmse'] for m in client_metrics)
        r2_worst = min(m['r2'] for m in client_metrics)
        
        print(f"\nAverage Client Performance (weighted by samples):")
        print(f"  MSE:  {mse_avg:.4f}")
        print(f"  RMSE: {rmse_avg:.4f}")
        print(f"  MAE:  {mae_avg:.4f}")
        print(f"  R²:   {r2_avg:.4f}")
        
        print(f"\nWorst-Case Client Performance:")
        print(f"  MSE:  {mse_worst:.4f}")
        print(f"  RMSE: {rmse_worst:.4f}")
        print(f"  MAE:  {mae_worst:.4f}")
        print(f"  R²:   {r2_worst:.4f}")
        
        # Print per-client metrics table
        print(f"\n{'Client ID':<12} {'Samples':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
        print("-" * 70)
        for m in client_metrics:
            print(f"{m['client_id']:<12} {m['num_samples']:<10} {m['mse']:<12.4f} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<10.4f}")
    
    # Evaluate aggregated model on combined validation data
    print("\n" + "=" * 60)
    print("Aggregated Model Performance")
    print("=" * 60)
    
    # Combine all validation data from clients
    all_X_val = []
    all_y_val = []
    
    for partition_df in data_partitions:
        from federated_client import create_client_for_data
        client = create_client_for_data(
            data_df=partition_df,
            window_size=window_size,
            val_split=0.2,
            input_shape=input_shape,
            global_feature_selection=global_feature_selection,
        )
        if client.X_val is not None and len(client.X_val) > 0:
            all_X_val.append(client.X_val)
            all_y_val.append(client.y_val)
    
    if all_X_val:
        X_val_combined = np.concatenate(all_X_val, axis=0)
        y_val_combined = np.concatenate(all_y_val, axis=0)
        
        y_pred_aggregated_val = model.predict(X_val_combined, verbose=0).flatten()
        y_true_aggregated_val = y_val_combined.flatten()
        
        mse_agg = mean_squared_error(y_true_aggregated_val, y_pred_aggregated_val)
        mae_agg = mean_absolute_error(y_true_aggregated_val, y_pred_aggregated_val)
        rmse_agg = np.sqrt(mse_agg)
        r2_agg = r2_score(y_true_aggregated_val, y_pred_aggregated_val)
        
        print(f"\nAggregated Model Metrics (on combined validation set):")
        print(f"  MSE:  {mse_agg:.4f}")
        print(f"  RMSE: {rmse_agg:.4f}")
        print(f"  MAE:  {mae_agg:.4f}")
        print(f"  R²:   {r2_agg:.4f}")
        
        print(f"\nComparison:")
        print(f"  Average Client MSE:  {mse_avg:.4f}  |  Aggregated Model MSE:  {mse_agg:.4f}")
        print(f"  Average Client RMSE: {rmse_avg:.4f}  |  Aggregated Model RMSE: {rmse_agg:.4f}")
        print(f"  Average Client MAE:  {mae_avg:.4f}  |  Aggregated Model MAE:  {mae_agg:.4f}")
        print(f"  Average Client R²:   {r2_avg:.4f}  |  Aggregated Model R²:   {r2_agg:.4f}")


def split_dataset_into_partitions(data_df, num_partitions: int) -> List:
    """
    Split the entire dataset into N partitions for federated learning.
    Each partition will contain data from multiple units.
    
    Args:
        data_df: Combined DataFrame with all training data
        num_partitions: Number of partitions (clients) to create
    
    Returns:
        List of DataFrames, one per partition
    """
    import pandas as pd
    
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    
    if num_partitions == 1:
        return [data_df.copy()]
    
    # Get unique unit_ids
    unique_unit_ids = sorted(data_df["unit_id"].unique().tolist())
    num_units = len(unique_unit_ids)
    
    print(f"\nSplitting {num_units} units across {num_partitions} clients...")
    
    # Calculate how many units per partition
    units_per_partition = num_units // num_partitions
    remainder = num_units % num_partitions
    
    partitions = []
    unit_idx = 0
    
    for i in range(num_partitions):
        # Distribute remainder units to first partitions
        num_units_for_partition = units_per_partition + (1 if i < remainder else 0)
        
        # Get unit_ids for this partition
        partition_unit_ids = unique_unit_ids[unit_idx:unit_idx + num_units_for_partition]
        unit_idx += num_units_for_partition
        
        # Filter data for this partition
        partition_df = data_df[data_df["unit_id"].isin(partition_unit_ids)].copy()
        partitions.append(partition_df)
        
        print(f"  Client {i}: {len(partition_unit_ids)} units, {len(partition_df)} rows")
    
    return partitions






def main():
    # Reduce TF log noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()

    print("=" * 60)
    print("Federated Learning Simulation (Flower run_simulation)")
    print("=" * 60)

    # Load all training data
    try:
        all_data_df = load_all_training_data(args.data_dir)
    except Exception as e:
        print(f"\nError loading training files: {e}")
        return 1

    # Compute global feature selection to ensure all clients use the same feature set
    try:
        global_feature_selection = compute_global_feature_selection(args.data_dir)
        print(f"\nGlobal feature selection computed: {len(global_feature_selection['sensor_columns_to_keep'])} sensors")
    except Exception as e:
        print(f"\nError computing global feature selection: {e}")
        return 1

    # Split dataset into partitions for clients
    num_clients = args.num_clients
    if num_clients <= 0:
        print(f"\nError: num-clients must be > 0, got {num_clients}")
        return 1
    
    try:
        data_partitions = split_dataset_into_partitions(all_data_df, num_clients)
        print(f"\nDataset split into {len(data_partitions)} partitions")
    except Exception as e:
        print(f"\nError splitting dataset: {e}")
        return 1

    if not data_partitions or any(len(part) == 0 for part in data_partitions):
        print("\nError: Some data partitions are empty.")
        return 1

    print(f"\nWill create {num_clients} clients (dataset split across clients)")

    # Determine input shape using global feature selection
    # Calculate number of features: 3 op_settings + number of selected sensors
    num_selected_sensors = len(global_feature_selection["sensor_columns_to_keep"])
    num_features = 3 + num_selected_sensors  # 3 op_settings + selected sensors
    
    if args.num_features is not None:
        # Override with user-provided value
        num_features = args.num_features
        print(f"\nUsing provided input shape: {num_features} features")
    else:
        print(f"\nComputed input shape from global feature selection:")
        print(f"  - Operational settings: 3")
        print(f"  - Selected sensors: {num_selected_sensors}")
        print(f"  - Total features: {num_features}")
    
    input_shape = (int(args.window_size), int(num_features))
    print(f"Model input shape: {input_shape} (timesteps={input_shape[0]}, features={input_shape[1]})")

    # Use the reusable function from federated_server.py to create the strategy
    print("\nBuilding initial model and strategy...")
    # Ensure min_fit_clients doesn't exceed available clients
    min_fit_clients = min(args.min_fit_clients, num_clients)
    
    strategy, initial_parameters, fit_config = create_server_strategy(
        input_shape=input_shape,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        proximal_mu=args.proximal_mu,
        fraction_fit=args.fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=num_clients,
        out_model_path=args.out_model,
    )

    print(f"\nFedProx proximal term weight (μ): {args.proximal_mu}")
    print(f"Fraction of clients per round: {args.fraction_fit}")
    print(f"Minimum clients per round: {min_fit_clients} (requested: {args.min_fit_clients}, available: {num_clients})")

    # Use Flower's run_simulation API
    print("\n" + "=" * 60)
    print("Starting Flower simulation...")
    print("=" * 60)

    try:
        from flwr.clientapp import ClientApp
        from flwr.common import Context
        from flwr.server import ServerApp, ServerConfig
        from flwr.server.serverapp_components import ServerAppComponents
        from flwr.simulation import run_simulation

        # Store data partitions and global feature selection in a way that can be accessed by client_fn
        # Use module-level variables that can be indexed (Ray can handle this better)
        _data_partitions_for_simulation = data_partitions
        _simulation_window_size = args.window_size
        _simulation_input_shape = input_shape
        _global_feature_selection = global_feature_selection  # Pass global feature selection to clients
        
        def client_fn(context: Context):
            # Try to use node_id as index if it's a small integer
            # Otherwise, use a hash-based approach
            node_id = context.node_id
            
            # Convert to Python int
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
            else:
                node_id = int(node_id)
            
            # If node_id is a reasonable index (0 to len-1), use it directly
            # Otherwise, use modulo to map it to valid range
            if 0 <= node_id < len(_data_partitions_for_simulation):
                idx = node_id
            else:
                # Use modulo as fallback (not ideal but should work)
                idx = node_id % len(_data_partitions_for_simulation)
                print(f"Warning: node_id {node_id} out of range, using idx={idx}")
            
            partition_df = _data_partitions_for_simulation[idx]
            print(f"Creating client: node_id={node_id} -> idx={idx} -> {len(partition_df)} rows, {len(partition_df['unit_id'].unique())} units")
            
            # Import here to avoid serialization issues
            from federated_client import create_client_for_data
            
            # Use the reusable function from federated_client.py with global feature selection
            client = create_client_for_data(
                data_df=partition_df,
                window_size=_simulation_window_size,
                val_split=0.2,
                input_shape=_simulation_input_shape,
                global_feature_selection=_global_feature_selection,  # Pass global feature selection
            )
            return client.to_client()

        def server_fn(context: Context):
            return ServerAppComponents(
                strategy=strategy, config=ServerConfig(num_rounds=int(args.num_rounds))
            )

        client_app = ClientApp(client_fn=client_fn)
        server_app = ServerApp(server_fn=server_fn)

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=num_clients,
            backend_config={"client_resources": {"num_cpus": float(args.cpus_per_client)}},
        )
    except ImportError:
        # Older Flower versions
        from flwr.simulation import start_simulation
        from flwr.server import ServerConfig

        def client_fn(cid: str):
            idx = int(cid)
            
            # Ensure index is within bounds
            if idx < 0:
                idx = 0
            elif idx >= len(data_partitions):
                # If cid is out of range, try 1-based indexing
                idx = idx - 1
                if idx < 0 or idx >= len(data_partitions):
                    raise IndexError(
                        f"client_id {cid} is out of range. "
                        f"Available indices: 0-{len(data_partitions) - 1}"
                    )
            
            partition_df = data_partitions[idx]
            print(f"Creating client for cid={cid} (idx={idx}), {len(partition_df)} rows, {len(partition_df['unit_id'].unique())} units")
            
            # Use the reusable function from federated_client.py with global feature selection
            return create_client_for_data(
                data_df=partition_df,
                window_size=args.window_size,
                val_split=0.2,
                input_shape=input_shape,
                global_feature_selection=global_feature_selection,  # Pass global feature selection
            )

        start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=ServerConfig(num_rounds=int(args.num_rounds)),
            strategy=strategy,
            client_resources={"num_cpus": float(args.cpus_per_client)},
        )

    # Save final aggregated model
    strategy.save_latest()
    print("\n" + "=" * 60)
    print("Federated Learning Simulation Complete!")
    print(f"Model saved to: {args.out_model}")
    print("=" * 60)

    # Evaluate on test data if provided
    if args.test_data_dir and os.path.exists(args.test_data_dir):
        print("\n" + "=" * 60)
        print("Evaluating on Test Data")
        print("=" * 60)
        
        try:
            evaluate_federated_model_on_test(
                model_path=args.out_model,
                test_data_dir=args.test_data_dir,
                data_partitions=data_partitions,
                global_feature_selection=global_feature_selection,
                window_size=args.window_size,
                input_shape=input_shape,
            )
        except Exception as e:
            print(f"\nWarning: Error during test evaluation: {e}")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
