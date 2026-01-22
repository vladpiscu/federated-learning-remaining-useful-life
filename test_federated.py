import numpy as np
import sys
import os
import pandas as pd
import glob
import argparse
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import preprocessing and model functions
from preprocessing.preprocessing import preprocess_data, COLUMNS
from models.lstm_model import predict_rul
from tensorflow import keras
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError


def load_true_rul(rul_file_path):
    """
    Load true RUL values from the RUL file.
    The RUL file contains one RUL value per test unit (the RUL at the last cycle).
    """
    rul_df = pd.read_csv(rul_file_path, sep=r"\s+", header=None, names=["RUL"])
    return rul_df['RUL'].values


def get_last_window_per_unit(windows_array, test_file_path, window_size=None):
    """
    For test data, we need to predict RUL for each unit at its last cycle.
    This function identifies the last window for each unit.
    
    Args:
        windows_array: All windows from test data
        test_file_path: Path to test data file (to get unit_ids)
        window_size: Window size used for preprocessing
    
    Returns:
        last_windows_indices: Indices of the last window for each unit
        unit_ids: Unit IDs corresponding to each last window
    """
    if window_size is None:
        # Derive window size from the preprocessed tensor
        if windows_array is None or len(windows_array) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        window_size = int(windows_array.shape[1])

    # Load test data to get unit structure
    df = pd.read_csv(test_file_path, sep=r"\s+", header=None, names=COLUMNS)
    
    # Get unique unit IDs
    unique_units = sorted(df['unit_id'].unique())
    
    # Find the last window index for each unit
    last_windows_indices = []
    unit_ids = []
    
    window_idx = 0
    for unit_id in unique_units:
        unit_data = df[df['unit_id'] == unit_id].sort_values('time_cycles')
        num_windows = len(unit_data) - window_size + 1
        
        if num_windows > 0:
            # Last window index for this unit
            last_window_idx = window_idx + num_windows - 1
            last_windows_indices.append(last_window_idx)
            unit_ids.append(unit_id)
            window_idx += num_windows
    
    return np.array(last_windows_indices), np.array(unit_ids)


def evaluate_predictions(y_true, y_pred):
    """Calculate and print evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(f"Mean Squared Error (MSE):     {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):    {mae:.4f}")
    print(f"R² Score:                      {r2:.4f}")
    print("=" * 60)
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def load_global_feature_selection(data_dir: str):
    """
    Load or compute global feature selection from training data.
    This should match what was used during federated training.
    """
    # Try to load from a saved file first
    feature_selection_path = os.path.join('models', 'global_feature_selection.pkl')
    if os.path.exists(feature_selection_path):
        print(f"Loading global feature selection from: {feature_selection_path}")
        with open(feature_selection_path, 'rb') as f:
            return pickle.load(f)
    
    # Otherwise, compute it from training data (same logic as run_federated.py)
    print("Computing global feature selection from training data...")
    train_files = sorted(glob.glob(os.path.join(data_dir, "train_*.txt")))
    
    if not train_files:
        raise FileNotFoundError(f"No training files found in: {data_dir}")
    
    all_dataframes = []
    max_unit_id = 0
    
    for train_file in train_files:
        df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMNS)
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
    
    global_feature_selection = {"sensor_columns_to_keep": sensor_columns_to_keep}
    
    # Save for future use
    os.makedirs('models', exist_ok=True)
    with open(feature_selection_path, 'wb') as f:
        pickle.dump(global_feature_selection, f)
    print(f"Saved global feature selection to: {feature_selection_path}")
    
    return global_feature_selection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the trained federated LSTM RUL model on CMAPSS test data."
    )
    parser.add_argument(
        "--data-dir",
        default="CMAPSSData",
        help="Directory containing CMAPSSData files (default: CMAPSSData).",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join("models", "federated_lstm_rul_model.h5"),
        help="Path to the trained federated model (default: models/federated_lstm_rul_model.h5).",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help=(
            "Test on a single test file instead of all test_*.txt. "
            "Can be a filename (e.g., test_FD001.txt) or a full/relative path."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Sliding window size used during training (default: 30).",
    )
    return parser.parse_args()


def _resolve_single_file(data_dir: str, file_arg: str) -> str:
    if os.path.exists(file_arg):
        return file_arg
    candidate = os.path.join(data_dir, file_arg)
    if os.path.exists(candidate):
        return candidate
    return file_arg


def main():
    """
    Test script for federated model that:
    1. Loads the trained federated model
    2. Loads/computes global feature selection
    3. Preprocesses test data using the same feature selection
    4. Makes predictions on test data (last window of each unit)
    5. Evaluates predictions against true RUL values
    """
    
    print("=" * 60)
    print("Testing Federated LSTM Model on Test Data")
    print("=" * 60)

    args = parse_args()
    
    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the federated model first using: python run_federated.py")
        return 1
    
    print(f"\nLoading federated model from: {args.model_path}")
    try:
        model = keras.models.load_model(
            args.model_path,
            custom_objects={
                'MeanAbsoluteError': MeanAbsoluteError,
                'MeanSquaredError': MeanSquaredError,
                'mse': MeanSquaredError,
                'mae': MeanAbsoluteError
            }
        )
    except Exception as e:
        print(f"Warning: Error loading with custom objects: {e}")
        print("Trying to load without custom objects...")
        try:
            model = keras.models.load_model(args.model_path)
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return 1
    print("Model loaded successfully!")
    
    # Load or compute global feature selection
    try:
        global_feature_selection = load_global_feature_selection(args.data_dir)
    except Exception as e:
        print(f"Error loading/computing global feature selection: {e}")
        return 1
    
    # Find all test files
    data_dir = args.data_dir
    if args.test_file:
        single_test_file = _resolve_single_file(data_dir, args.test_file)
        if not os.path.exists(single_test_file):
            print(f"Error: Test file not found: {single_test_file}")
            return 1
        test_files = [single_test_file]
    else:
        test_files = sorted(glob.glob(os.path.join(data_dir, 'test_*.txt')))
    
    if not test_files:
        print(f"Error: No test files found in {data_dir}")
        return 1
    
    print(f"\nFound {len(test_files)} test file(s):")
    for f in test_files:
        print(f"  - {f}")
    
    # Process each test file
    all_predictions = []
    all_true_rul = []
    all_unit_ids = []
    dataset_names = []
    
    for test_file in test_files:
        # Extract dataset name (e.g., FD001 from test_FD001.txt)
        dataset_name = os.path.basename(test_file).replace('test_', '').replace('.txt', '')
        dataset_names.append(dataset_name)
        
        print("\n" + "=" * 60)
        print(f"Processing Test Dataset: {dataset_name}")
        print("=" * 60)
        
        # Load test data
        print(f"\nLoading test data from: {test_file}")
        test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMNS)
        print(f"Loaded {len(test_df)} rows, {test_df['unit_id'].nunique()} units")
        
        # Preprocess test data with global feature selection
        print("\nPreprocessing test data with global feature selection...")
        preprocessing_params_for_test = {
            'sensor_columns_to_keep': global_feature_selection['sensor_columns_to_keep'],
        }
        
        X_test, _ = preprocess_data(
            df=test_df,
            preprocessing_params=preprocessing_params_for_test,
            compute_rul=False,
            window_size=args.window_size,
        )
        
        print(f"Preprocessed test data shape: {X_test.shape}")
        
        # Get last window for each unit (since we predict RUL at the last cycle)
        print("\nIdentifying last window for each test unit...")
        last_windows_indices, unit_ids = get_last_window_per_unit(X_test, test_file, args.window_size)
        last_windows = X_test[last_windows_indices]
        
        print(f"Number of test units: {len(last_windows)}")
        print(f"Last windows shape: {last_windows.shape}")
        
        # Make predictions
        print("\nMaking predictions with federated model...")
        predictions = predict_rul(model, last_windows)
        
        # Load true RUL values
        rul_file = os.path.join(data_dir, f'RUL_{dataset_name}.txt')
        if not os.path.exists(rul_file):
            print(f"Warning: RUL file not found: {rul_file}")
            print("Skipping evaluation for this dataset (no ground truth available)")
            continue
        
        print(f"Loading true RUL values from: {rul_file}")
        true_rul = load_true_rul(rul_file)
        
        if len(true_rul) != len(predictions):
            print(f"Warning: Mismatch in number of predictions ({len(predictions)}) and true RUL values ({len(true_rul)})")
            min_len = min(len(predictions), len(true_rul))
            predictions = predictions[:min_len]
            true_rul = true_rul[:min_len]
            unit_ids = unit_ids[:min_len]
        
        # Evaluate predictions for this dataset
        print(f"\n{'=' * 60}")
        print(f"Evaluation Metrics for {dataset_name}")
        print(f"{'=' * 60}")
        metrics = evaluate_predictions(true_rul, predictions)
        
        # Store for combined evaluation
        all_predictions.extend(predictions)
        all_true_rul.extend(true_rul)
        all_unit_ids.extend(unit_ids)
        
        # Print some example predictions for this dataset
        print(f"\n{'=' * 60}")
        print(f"Sample Predictions for {dataset_name} (first 10 units)")
        print(f"{'=' * 60}")
        print(f"{'Unit ID':<10} {'True RUL':<12} {'Predicted RUL':<15} {'Error':<12} {'Error %':<12}")
        print("-" * 70)
        for i in range(min(10, len(predictions))):
            error = abs(true_rul[i] - predictions[i])
            error_pct = (error / (true_rul[i] + 1e-10)) * 100
            print(f"{unit_ids[i]:<10} {true_rul[i]:<12.2f} {predictions[i]:<15.2f} {error:<12.2f} {error_pct:<12.2f}%")
    
    # Combined evaluation across all datasets
    if len(all_predictions) > 0:
        print("\n" + "=" * 60)
        print("Combined Evaluation Metrics (All Datasets)")
        print("=" * 60)
        all_predictions = np.array(all_predictions)
        all_true_rul = np.array(all_true_rul)
        combined_metrics = evaluate_predictions(all_true_rul, all_predictions)
        
        # Print summary by dataset
        print("\n" + "=" * 60)
        print("Summary by Dataset")
        print("=" * 60)
        print(f"{'Dataset':<12} {'Units':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
        print("-" * 70)
        
        # Re-evaluate each dataset for summary
        idx = 0
        for i, test_file in enumerate(test_files):
            dataset_name = dataset_names[i]
            rul_file = os.path.join(data_dir, f'RUL_{dataset_name}.txt')
            if not os.path.exists(rul_file):
                continue
            
            true_rul = load_true_rul(rul_file)
            num_units = len(true_rul)
            dataset_predictions = all_predictions[idx:idx+num_units]
            dataset_true = all_true_rul[idx:idx+num_units]
            
            mse = mean_squared_error(dataset_true, dataset_predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(dataset_true, dataset_predictions)
            r2 = r2_score(dataset_true, dataset_predictions)
            
            print(f"{dataset_name:<12} {num_units:<10} {mse:<12.2f} {rmse:<12.2f} {mae:<12.2f} {r2:<10.4f}")
            idx += num_units
    else:
        print("\nWarning: No predictions were made. Check if RUL files exist.")
        return 1
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
