import argparse
import glob
import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import flwr as fl
from models.lstm_model import build_lstm_model
from preprocessing.preprocessing import COLUMNS, preprocess_data

# Import client and server functions from federated files
from federated_client import create_client_for_data, create_client_for_unit, load_unit_data
from federated_server import create_server_strategy, SavingFedProx


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
        default=3,
        help="Minimum number of clients to sample for training each round. Default: 3",
    )
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=3,
        help="Number of clients to select for training in each round. Default: 3",
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


def extract_metrics_from_result(result):
    """
    Helper function to extract metrics from Flower result objects.
    Handles different Flower versions and result formats.
    """
    metrics = {}
    client_id = None
    
    # Try to get client ID - Flower uses different attributes in different versions
    if hasattr(result, 'client'):
        client_id = result.client
    elif hasattr(result, 'client_id'):
        client_id = result.client_id
    elif hasattr(result, 'client_id') and callable(getattr(result, 'client_id', None)):
        client_id = result.client_id()
    
    # Try to get metrics - Flower wraps results in FitRes/EvaluateRes objects
    # The metrics are typically in result.metrics or result[2] if it's a tuple
    if hasattr(result, 'metrics'):
        metrics_dict = result.metrics
        if metrics_dict and isinstance(metrics_dict, dict):
            metrics = metrics_dict
    
    # Also check if result itself is a tuple (older Flower versions)
    if isinstance(result, tuple):
        # Format: (weights/params, num_examples, metrics) for fit
        # Format: (loss, num_examples, metrics) for evaluate
        if len(result) >= 3:
            if isinstance(result[2], dict):
                metrics = result[2]
            elif result[2] is not None:
                # Sometimes metrics might be in a different format
                metrics = result[2] if isinstance(result[2], dict) else {}
        elif len(result) >= 2 and isinstance(result[1], dict):
            metrics = result[1]
    
    # If still no metrics, check for FitRes/EvaluateRes attributes
    if not metrics:
        # Check for common Flower result attributes
        for attr in ['fit_res', 'evaluate_res', 'result']:
            if hasattr(result, attr):
                nested = getattr(result, attr)
                if isinstance(nested, tuple) and len(nested) >= 3:
                    if isinstance(nested[2], dict):
                        metrics = nested[2]
                        break
    
    return client_id, metrics


class MetricsTrackingStrategy(SavingFedProx):
    """
    Custom strategy that extends SavingFedProx to track metrics history
    for convergence analysis and per-client error distribution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track metrics per round
        self.metrics_history = {
            'round': [],
            'aggregated': {
                'loss': [],
                'mse': [],
                'mae': [],
                'rmse': [],
                'r2': [],
            },
            'per_client': [],  # List of dicts, one per round, with client_id -> metrics
        }
    
    def aggregate_fit(self, rnd, results, failures):
        """Override to track per-client metrics during fit."""
        # Extract per-client metrics before aggregation
        if results:
            round_client_metrics = {}
            for idx, fit_res in enumerate(results):
                # Check if result is successful
                is_ok = False
                if hasattr(fit_res, 'status'):
                    is_ok = fit_res.status.code == fl.common.Code.OK
                elif isinstance(fit_res, tuple):
                    # If it's a tuple, assume it's valid
                    is_ok = True
                else:
                    # Assume it's a FitRes object
                    is_ok = True
                
                if is_ok:
                    client_id, metrics = extract_metrics_from_result(fit_res)
                    
                    # If no metrics found, try multiple extraction methods
                    if not metrics or (isinstance(metrics, dict) and len(metrics) == 0):
                        # Method 1: Direct access to FitRes.metrics
                        if hasattr(fit_res, 'metrics'):
                            potential_metrics = fit_res.metrics
                            if potential_metrics and isinstance(potential_metrics, dict) and len(potential_metrics) > 0:
                                metrics = potential_metrics
                        
                        # Method 2: Check if FitRes has a properties dict
                        if (not metrics or (isinstance(metrics, dict) and len(metrics) == 0)) and hasattr(fit_res, 'properties'):
                            props = fit_res.properties
                            if props and isinstance(props, dict):
                                metrics = props
                        
                        # Method 3: Try to access internal _metrics or similar
                        for attr_name in ['_metrics', 'metrics_dict', 'fit_metrics']:
                            if (not metrics or (isinstance(metrics, dict) and len(metrics) == 0)) and hasattr(fit_res, attr_name):
                                potential = getattr(fit_res, attr_name)
                                if potential and isinstance(potential, dict) and len(potential) > 0:
                                    metrics = potential
                                    break
                    
                    # Store per-client metrics (use index as client_id if not available)
                    if metrics and isinstance(metrics, dict) and len(metrics) > 0:
                        key = client_id if client_id is not None else f'client_{idx}'
                        round_client_metrics[key] = metrics
                    elif isinstance(fit_res, tuple) and len(fit_res) >= 3:
                        # Try to extract metrics from tuple directly: (weights, num_examples, metrics)
                        if isinstance(fit_res[2], dict) and len(fit_res[2]) > 0:
                            key = client_id if client_id is not None else f'client_{idx}'
                            round_client_metrics[key] = fit_res[2]
            
            # Store per-client metrics for this round
            # Ensure list is long enough (rnd is 1-indexed)
            while len(self.metrics_history['per_client']) < rnd:
                self.metrics_history['per_client'].append({})
            
            # Append or update metrics for this round
            if round_client_metrics:
                if len(self.metrics_history['per_client']) >= rnd:
                    # Merge with existing metrics
                    self.metrics_history['per_client'][rnd - 1].update(round_client_metrics)
                else:
                    self.metrics_history['per_client'].append(round_client_metrics)
            else:
                # Debug: print what we got
                if rnd <= 2:  # Only print for first few rounds to avoid spam
                    print(f"DEBUG Round {rnd}: No metrics extracted from {len(results)} results")
                    if results:
                        first_res = results[0]
                        print(f"  First result type: {type(first_res)}")
                        print(f"  Has 'metrics' attr: {hasattr(first_res, 'metrics')}")
                        if hasattr(first_res, 'metrics'):
                            print(f"  metrics value: {first_res.metrics}")
                            print(f"  metrics type: {type(first_res.metrics)}")
                        # Try to see if we can access it as a tuple
                        try:
                            if isinstance(first_res, tuple):
                                print(f"  Is tuple with length: {len(first_res)}")
                                if len(first_res) >= 3:
                                    print(f"  Third element type: {type(first_res[2])}")
                                    print(f"  Third element value: {first_res[2]}")
                        except:
                            pass
                        # Check common Flower attributes
                        for attr in ['status', 'client', 'client_id', 'properties']:
                            if hasattr(first_res, attr):
                                val = getattr(first_res, attr)
                                print(f"  {attr}: {val} (type: {type(val)})")
        
        # Call parent aggregation
        aggregated = super().aggregate_fit(rnd, results, failures)
        return aggregated
    
    def aggregate_evaluate(self, rnd, results, failures):
        """Override to track metrics after evaluation."""
        # Extract per-client metrics from evaluation results
        if results:
            round_client_metrics = {}
            for eval_res in results:
                # Check if result is successful
                is_ok = False
                if hasattr(eval_res, 'status'):
                    is_ok = eval_res.status.code == fl.common.Code.OK
                elif isinstance(eval_res, tuple):
                    is_ok = True
                
                if is_ok:
                    client_id, metrics = extract_metrics_from_result(eval_res)
                    
                    # Store per-client metrics
                    if metrics:
                        key = client_id if client_id is not None else f'client_{len(round_client_metrics)}'
                        round_client_metrics[key] = metrics
            
            # Update or append per-client metrics for this round
            # Ensure list is long enough (rnd is 1-indexed)
            while len(self.metrics_history['per_client']) < rnd:
                self.metrics_history['per_client'].append({})
            # Merge with existing metrics or set new
            if round_client_metrics:
                if len(self.metrics_history['per_client']) >= rnd:
                    self.metrics_history['per_client'][rnd - 1].update(round_client_metrics)
                else:
                    self.metrics_history['per_client'].append(round_client_metrics)
        
        # Call parent aggregation
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        
        # Extract aggregated metrics
        if aggregated is not None:
            metrics = {}
            # Handle different return formats
            if isinstance(aggregated, tuple):
                if len(aggregated) >= 2:
                    # Format: (loss, num_examples, metrics) or (metrics,)
                    if isinstance(aggregated[1], dict):
                        metrics = aggregated[1]
                    elif len(aggregated) >= 3 and isinstance(aggregated[2], dict):
                        metrics = aggregated[2]
                elif len(aggregated) == 1 and isinstance(aggregated[0], dict):
                    metrics = aggregated[0]
            elif isinstance(aggregated, dict):
                metrics = aggregated
            
            # Store aggregated metrics for this round
            self.metrics_history['round'].append(rnd)
            for metric_name in ['loss', 'mse', 'mae', 'rmse', 'r2']:
                # Try different metric name variations
                value = (metrics.get(metric_name) or 
                        metrics.get(f'{metric_name}_avg') or
                        metrics.get(f'avg_{metric_name}'))
                if value is not None:
                    try:
                        self.metrics_history['aggregated'][metric_name].append(float(value))
                    except (ValueError, TypeError):
                        self.metrics_history['aggregated'][metric_name].append(None)
                else:
                    # If not found, append None to maintain list length
                    if len(self.metrics_history['aggregated'][metric_name]) < len(self.metrics_history['round']):
                        self.metrics_history['aggregated'][metric_name].append(None)
        
        return aggregated


def visualize_convergence(metrics_history: Dict, output_dir: str = "models"):
    """
    Visualize convergence behavior over communication rounds.
    
    Args:
        metrics_history: Dictionary with metrics history from MetricsTrackingStrategy
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    rounds = metrics_history['round']
    if not rounds:
        print("Warning: No metrics history to visualize.")
        return
    
    aggregated = metrics_history['aggregated']
    
    # Create figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Behavior Over Communication Rounds', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss and MSE
    ax1 = axes[0, 0]
    if aggregated['loss']:
        ax1.plot(rounds, aggregated['loss'], 'o-', label='Loss', linewidth=2, markersize=6)
    if aggregated['mse']:
        ax1.plot(rounds, aggregated['mse'], 's-', label='MSE', linewidth=2, markersize=6)
    ax1.set_xlabel('Communication Round', fontsize=11)
    ax1.set_ylabel('Error', fontsize=11)
    ax1.set_title('Loss and MSE Convergence', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMSE
    ax2 = axes[0, 1]
    if aggregated['rmse']:
        ax2.plot(rounds, aggregated['rmse'], '^-', color='green', label='RMSE', linewidth=2, markersize=6)
    ax2.set_xlabel('Communication Round', fontsize=11)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('RMSE Convergence', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MAE
    ax3 = axes[1, 0]
    if aggregated['mae']:
        ax3.plot(rounds, aggregated['mae'], 'd-', color='orange', label='MAE', linewidth=2, markersize=6)
    ax3.set_xlabel('Communication Round', fontsize=11)
    ax3.set_ylabel('MAE', fontsize=11)
    ax3.set_title('MAE Convergence', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: R² Score
    ax4 = axes[1, 1]
    if aggregated['r2']:
        ax4.plot(rounds, aggregated['r2'], '*-', color='purple', label='R²', linewidth=2, markersize=6)
    ax4.set_xlabel('Communication Round', fontsize=11)
    ax4.set_ylabel('R² Score', fontsize=11)
    ax4.set_title('R² Score Convergence', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'convergence_behavior.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConvergence plot saved to: {output_path}")
    plt.close()


def load_client_metrics_from_files(metrics_dir: str = "models/client_metrics"):
    """
    Load per-client metrics from JSON files saved during training.
    
    Args:
        metrics_dir: Directory containing client metrics JSON files
    
    Returns:
        Dictionary with per-client metrics organized by round
    """
    import json
    import glob
    
    if not os.path.exists(metrics_dir):
        print(f"Metrics directory does not exist: {metrics_dir}")
        return {}
    
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(metrics_dir, "client_*_round_*_*.json"))
    
    if not metrics_files:
        print(f"No metrics files found in {metrics_dir}")
        return {}
    
    print(f"Found {len(metrics_files)} metrics files in {metrics_dir}")
    
    per_client_metrics = {}
    
    for filepath in sorted(metrics_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            round_num = data.get('round', 0)
            client_id = data.get('client_id', 'unknown')
            phase = data.get('phase', 'fit')
            metrics = data.get('metrics', {})
            
            if not metrics:
                print(f"Warning: No metrics found in {filepath}")
                continue
            
            # Ensure round exists in dict
            if round_num not in per_client_metrics:
                per_client_metrics[round_num] = {}
            
            # Store metrics (prefer evaluate over fit if both exist)
            if client_id not in per_client_metrics[round_num] or phase == 'evaluate':
                per_client_metrics[round_num][client_id] = metrics
                print(f"  Loaded metrics for client {client_id}, round {round_num}, phase {phase}")
        except Exception as e:
            print(f"Warning: Could not load metrics from {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Loaded metrics for {len(per_client_metrics)} rounds")
    return per_client_metrics


def visualize_per_client_error_distribution(metrics_history: Dict = None, output_dir: str = "models", metrics_dir: str = "models/client_metrics"):
    """
    Visualize per-client error distribution across communication rounds.
    
    Args:
        metrics_history: Dictionary with metrics history from MetricsTrackingStrategy (optional, will load from files if None)
        output_dir: Directory to save plots
        metrics_dir: Directory containing client metrics JSON files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Always try to load from files first (they're more reliable)
    print("Loading per-client metrics from files...")
    per_client_metrics_dict = load_client_metrics_from_files(metrics_dir)
    
    # Convert to list format expected by visualization
    if per_client_metrics_dict:
        max_round = max(per_client_metrics_dict.keys())
        per_client_metrics = []
        for rnd in range(1, max_round + 1):
            per_client_metrics.append(per_client_metrics_dict.get(rnd, {}))
        print(f"Converted to list format: {len(per_client_metrics)} rounds")
    else:
        # Fallback to metrics_history if files don't exist
        print("No metrics files found, trying metrics_history...")
        if metrics_history and metrics_history.get('per_client'):
            per_client_metrics = metrics_history['per_client']
        else:
            per_client_metrics = []
    
    if not per_client_metrics:
        print("Warning: No per-client metrics to visualize.")
        return
    
    # Collect all per-client metrics across rounds
    client_errors = {
        'mse': [],
        'mae': [],
        'rmse': [],
        'r2': [],
        'round': [],
        'client_id': [],
    }
    
    print(f"\nCollecting per-client metrics from {len(per_client_metrics)} rounds...")
    total_metrics_collected = 0
    
    for round_idx, round_metrics in enumerate(per_client_metrics):
        if not isinstance(round_metrics, dict):
            print(f"  Round {round_idx + 1}: Not a dict, skipping")
            continue
        
        if len(round_metrics) == 0:
            print(f"  Round {round_idx + 1}: Empty dict, skipping")
            continue
        
        print(f"  Round {round_idx + 1}: Processing {len(round_metrics)} clients")
            
        for client_id, metrics in round_metrics.items():
            if not isinstance(metrics, dict):
                print(f"    Client {client_id}: Metrics not a dict, skipping")
                continue
                
            if not metrics:
                print(f"    Client {client_id}: Empty metrics dict, skipping")
                continue
            
            print(f"    Client {client_id}: Found metrics with keys: {list(metrics.keys())}")
            client_errors['round'].append(round_idx + 1)  # 1-indexed rounds
            client_errors['client_id'].append(str(client_id) if client_id is not None else f'client_{round_idx}')
            
            # Extract metrics (they should be directly in the dict from JSON files)
            for metric_name in ['mse', 'mae', 'rmse', 'r2']:
                value = metrics.get(metric_name)
                
                if value is not None:
                    try:
                        client_errors[metric_name].append(float(value))
                    except (ValueError, TypeError) as e:
                        print(f"      Warning: Could not convert {metric_name}={value} to float: {e}")
                        client_errors[metric_name].append(np.nan)
                else:
                    print(f"      Warning: {metric_name} not found in metrics")
                    client_errors[metric_name].append(np.nan)
            
            total_metrics_collected += 1
    
    print(f"Collected metrics from {total_metrics_collected} client-round combinations")
    
    if total_metrics_collected == 0:
        print("Warning: No valid per-client metrics found. Check if clients are returning metrics in fit/evaluate methods.")
        return
    
    # Create figure with 2 subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Per-Client Error Distribution Across Communication Rounds', fontsize=16, fontweight='bold')
    
    # Convert to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(client_errors)
    
    # Plot 1: MSE distribution (box plot)
    ax1 = axes[0]
    if 'mse' in df.columns and df['mse'].notna().any():
        # Box plot by round
        df_clean = df[df['mse'].notna()].copy()
        if not df_clean.empty and len(df_clean) > 0:
            sns.boxplot(data=df_clean, x='round', y='mse', ax=ax1, palette='viridis')
            ax1.set_xlabel('Communication Round', fontsize=11)
            ax1.set_ylabel('MSE', fontsize=11)
            ax1.set_title('MSE Distribution Across Clients (by Round)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
        else:
            ax1.text(0.5, 0.5, 'No MSE data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('MSE Distribution (No Data)', fontsize=12)
    else:
        ax1.text(0.5, 0.5, 'No MSE data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('MSE Distribution (No Data)', fontsize=12)
    
    # Plot 2: MAE distribution (box plot)
    ax2 = axes[1]
    if 'mae' in df.columns and df['mae'].notna().any():
        df_clean = df[df['mae'].notna()].copy()
        if not df_clean.empty and len(df_clean) > 0:
            sns.boxplot(data=df_clean, x='round', y='mae', ax=ax2, palette='plasma')
            ax2.set_xlabel('Communication Round', fontsize=11)
            ax2.set_ylabel('MAE', fontsize=11)
            ax2.set_title('MAE Distribution Across Clients (by Round)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'No MAE data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('MAE Distribution (No Data)', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No MAE data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('MAE Distribution (No Data)', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_client_error_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-client error distribution plot saved to: {output_path}")
    plt.close()
    
    # Also create a combined view showing MSE and MAE statistics in one plot
    fig2, ax = plt.subplots(figsize=(14, 8))
    
    # Plot MSE and MAE statistics
    metrics_to_plot = ['mse', 'mae']
    for metric in metrics_to_plot:
        if metric in df.columns and df[metric].notna().any():
            df_clean = df[df[metric].notna()].copy()
            if not df_clean.empty:
                # Group by round and compute statistics
                grouped = df_clean.groupby('round')[metric].agg(['mean', 'std', 'min', 'max'])
                rounds = grouped.index
                
                # Plot mean with error bars
                ax.errorbar(rounds, grouped['mean'], yerr=grouped['std'], 
                           label=f'{metric.upper()} (mean ± std)', 
                           marker='o', linewidth=2, markersize=6, capsize=4)
    
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Error (mean ± std)', fontsize=12)
    ax.set_title('Per-Client Error Statistics Across Rounds (MSE and MAE)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 'per_client_error_statistics.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Per-client error statistics plot saved to: {output_path2}")
    plt.close()


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

    # Use the reusable function from federated_server.py to create the base strategy
    print("\nBuilding initial model and strategy...")
    
    # Set clients per round (default 3)
    clients_per_round = args.clients_per_round
    
    # Ensure clients_per_round doesn't exceed available clients
    if clients_per_round > num_clients:
        print(f"Warning: clients-per-round ({clients_per_round}) exceeds available clients ({num_clients}). Setting to {num_clients}.")
        clients_per_round = num_clients
    
    # Calculate fraction_fit to target exactly clients_per_round clients
    # Flower selects max(min_fit_clients, fraction_fit * num_clients) clients
    # To get exactly clients_per_round, we set both to clients_per_round
    fraction_fit = clients_per_round / num_clients
    fraction_fit = min(1.0, fraction_fit)  # Cap at 1.0
    
    # Set min_fit_clients to ensure exactly clients_per_round are selected
    # This ensures that even if fraction_fit * num_clients rounds down, we still get clients_per_round
    min_fit_clients = clients_per_round
    
    print(f"\nClient Selection Configuration:")
    print(f"  Total clients available: {num_clients}")
    print(f"  Clients per round: {clients_per_round}")
    print(f"  Fraction fit: {fraction_fit:.3f}")
    print(f"  Min fit clients: {min_fit_clients}")
    
    base_strategy, initial_parameters, fit_config = create_server_strategy(
        input_shape=input_shape,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        proximal_mu=args.proximal_mu,
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=num_clients,
        out_model_path=args.out_model,
    )
    
    # Wrap with metrics tracking strategy
    strategy = MetricsTrackingStrategy(
        fraction_fit=base_strategy.fraction_fit,
        fraction_evaluate=base_strategy.fraction_evaluate,
        min_fit_clients=base_strategy.min_fit_clients,
        min_evaluate_clients=base_strategy.min_evaluate_clients,
        min_available_clients=base_strategy.min_available_clients,
        initial_parameters=base_strategy.initial_parameters,
        on_fit_config_fn=base_strategy.on_fit_config_fn,
        on_evaluate_config_fn=base_strategy.on_evaluate_config_fn,
        proximal_mu=base_strategy.proximal_mu,
        model_builder=lambda: build_lstm_model(input_shape=input_shape),
        out_model_path=args.out_model,
        fit_metrics_aggregation_fn=base_strategy.fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=base_strategy.evaluate_metrics_aggregation_fn,
    )

    print(f"\nFedProx proximal term weight (μ): {args.proximal_mu}")
    print(f"Fraction of clients per round: {fraction_fit:.3f}")
    print(f"Minimum clients per round: {min_fit_clients} (target: {clients_per_round} clients per round)")

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
            # Pass client_id so metrics can be saved to files
            client = create_client_for_data(
                data_df=partition_df,
                window_size=_simulation_window_size,
                val_split=0.2,
                input_shape=_simulation_input_shape,
                global_feature_selection=_global_feature_selection,  # Pass global feature selection
                client_id=idx,  # Pass client ID for metrics file naming
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
            # Pass client_id so metrics can be saved to files
            return create_client_for_data(
                data_df=partition_df,
                window_size=args.window_size,
                val_split=0.2,
                input_shape=input_shape,
                global_feature_selection=global_feature_selection,  # Pass global feature selection
                client_id=idx,  # Pass client ID for metrics file naming
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
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    try:
        # Visualize convergence behavior
        visualize_convergence(strategy.metrics_history, output_dir=os.path.dirname(args.out_model) or "models")
        
        # Visualize per-client error distribution (will load from files if not in history)
        metrics_dir = os.path.join(os.path.dirname(args.out_model) or "models", "client_metrics")
        visualize_per_client_error_distribution(
            metrics_history=strategy.metrics_history, 
            output_dir=os.path.dirname(args.out_model) or "models",
            metrics_dir=metrics_dir
        )
        
        print("\nAll visualizations generated successfully!")
    except Exception as e:
        print(f"\nWarning: Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

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
