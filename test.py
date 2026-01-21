import numpy as np
import sys
import os
import pandas as pd
import glob
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import preprocessing and model functions
from preprocessing.preprocessing import preprocess_data, load_preprocessing_params
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the trained LSTM RUL model on CMAPSS test data."
    )
    parser.add_argument(
        "--data-dir",
        default="CMAPSSData",
        help="Directory containing CMAPSSData files (default: CMAPSSData).",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help=(
            "Test on a single test file instead of all test_*.txt. "
            "Can be a filename (e.g., test_FD001.txt) or a full/relative path."
        ),
    )
    return parser.parse_args()


def _resolve_single_file(data_dir: str, file_arg: str) -> str:
    # If the user passed an existing path, use it directly.
    if os.path.exists(file_arg):
        return file_arg
    # Otherwise, try relative to the data directory.
    candidate = os.path.join(data_dir, file_arg)
    if os.path.exists(candidate):
        return candidate
    return file_arg  # fall back; caller will error with a clear message

def get_last_window_per_unit(windows_array, test_file_path, window_size=None):
    """
    For test data, we need to predict RUL for each unit at its last cycle.
    This function identifies the last window for each unit.
    
    Args:
        windows_array: All windows from test data
        test_file_path: Path to test data file (to get unit_ids)
    
    Returns:
        last_windows_indices: Indices of the last window for each unit
        unit_ids: Unit IDs corresponding to each last window
    """
    if window_size is None:
        # Derive window size from the preprocessed tensor to avoid mismatches.
        if windows_array is None or len(windows_array) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        window_size = int(windows_array.shape[1])

    # Load test data to get unit structure
    columns = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(test_file_path, sep=r"\s+", header=None, names=columns)
    
    # Get unique unit IDs
    unique_units = df['unit_id'].unique()
    
    # Find the last window index for each unit
    # Since windows are created sequentially per unit, we can track this
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

def main():
    """
    Test script that:
    1. Loads the trained model
    2. Loads preprocessing parameters
    3. Preprocesses test data using saved parameters
    4. Makes predictions on test data
    5. Evaluates predictions against true RUL values
    """
    
    print("=" * 60)
    print("Testing LSTM Model on Test Data")
    print("=" * 60)

    args = parse_args()
    
    # Load model
    model_path = os.path.join('models', 'lstm_rul_model.h5')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    print(f"\nLoading model from: {model_path}")
    # Load model with custom objects to handle metrics
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'MeanAbsoluteError': MeanAbsoluteError,
                'MeanSquaredError': MeanSquaredError,
                'mse': MeanSquaredError,
                'mae': MeanAbsoluteError
            }
        )
    except Exception as e:
        # If loading fails, try without custom objects (for newer model format)
        print(f"Warning: Error loading with custom objects: {e}")
        print("Trying to load without custom objects...")
        try:
            model = keras.models.load_model(model_path)
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return
    print("Model loaded successfully!")
    
    # Load preprocessing parameters
    preprocessing_params_path = os.path.join('models', 'preprocessing_params.pkl')
    if not os.path.exists(preprocessing_params_path):
        print(f"Error: Preprocessing parameters not found at {preprocessing_params_path}")
        print("Please train the model first using: python train.py")
        return
    
    print(f"\nLoading preprocessing parameters from: {preprocessing_params_path}")
    preprocessing_params = load_preprocessing_params(preprocessing_params_path)
    print("Preprocessing parameters loaded successfully!")
    
    # Find all test files
    data_dir = args.data_dir
    if args.test_file:
        single_test_file = _resolve_single_file(data_dir, args.test_file)
        if not os.path.exists(single_test_file):
            print(f"Error: Test file not found: {single_test_file}")
            return
        test_files = [single_test_file]
    else:
        test_files = sorted(glob.glob(os.path.join(data_dir, 'test_*.txt')))
    
    if not test_files:
        print(f"Error: No test files found in {data_dir}")
        return
    
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
        
        # Preprocess test data
        print(f"\nPreprocessing test data from: {test_file}")
        windows_array, _ = preprocess_data(
            test_file,
            preprocessing_params=preprocessing_params,
            compute_rul=False
        )
        
        print(f"Preprocessed test data shape: {windows_array.shape}")
        
        # Get last window for each unit (since we predict RUL at the last cycle)
        print("\nIdentifying last window for each test unit...")
        last_windows_indices, unit_ids = get_last_window_per_unit(windows_array, test_file)
        last_windows = windows_array[last_windows_indices]
        
        print(f"Number of test units: {len(last_windows)}")
        print(f"Last windows shape: {last_windows.shape}")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = predict_rul(model, last_windows)
        
        # Load true RUL values
        rul_file = os.path.join(data_dir, f'RUL_{dataset_name}.txt')
        if not os.path.exists(rul_file):
            print(f"Warning: RUL file not found: {rul_file}")
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
        print(f"Sample Predictions for {dataset_name} (first 5 units)")
        print(f"{'=' * 60}")
        print(f"{'Unit ID':<10} {'True RUL':<12} {'Predicted RUL':<15} {'Error':<12}")
        print("-" * 60)
        for i in range(min(5, len(predictions))):
            error = abs(true_rul[i] - predictions[i])
            print(f"{unit_ids[i]:<10} {true_rul[i]:<12.2f} {predictions[i]:<15.2f} {error:<12.2f}")
    
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
        print("-" * 60)
        
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
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
