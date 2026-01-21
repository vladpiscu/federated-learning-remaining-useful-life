import pandas as pd
import numpy as np
import os
import pickle

# Column definitions
COLUMNS = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

def preprocess_data(file_path=None, df=None, preprocessing_params=None, compute_rul=True, window_size=30):
    """
    Preprocess CMAPSS dataset for LSTM training/prediction.
    
    Args:
        file_path: Path to the data file (train or test). If None, df must be provided.
        df: Pre-loaded dataframe. If None, file_path must be provided.
        preprocessing_params: Dictionary containing preprocessing parameters from training.
                             If None, parameters will be computed from the data.
                             Should contain: 'sensor_columns_to_keep', 'min_max_values', 'feature_columns'
        compute_rul: Whether to compute RUL (True for training, False for test)
        window_size: Size of the sliding window (default: 30)
    
    Returns:
        windows_array: Numpy array of shape (num_windows, window_size, num_features)
        rul_array: Numpy array of shape (num_windows,) - only if compute_rul=True
        preprocessing_params: Dictionary with preprocessing parameters (for saving/reuse)
    """
    
    # Load data
    if df is None:
        if file_path is None:
            raise ValueError("Either file_path or df must be provided")
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMNS)
    else:
        # Make a copy to avoid modifying the original
        df = df.copy()
    
    # Compute RUL if needed (for training data)
    if compute_rul:
        df['RUL'] = df.groupby('unit_id')['time_cycles'].transform(lambda x: x.max() - x)
    else:
        # For test data, we don't have RUL initially
        df['RUL'] = None
    
    # Define feature columns (exclude metadata columns: unit_id, time_cycles, RUL)
    feature_columns = [col for col in COLUMNS if col not in ['unit_id', 'time_cycles']]
    
    # Define sensor columns (only columns starting with 'sensor')
    sensor_columns = [col for col in COLUMNS if col.startswith('sensor')]
    
    # Step 1: Variance filtering (only on sensor columns)
    if preprocessing_params is None:
        # Compute variance and filter columns (training mode)
        variance_threshold = 0.01
        column_variances = df[sensor_columns].var()
        sensor_columns_to_keep = column_variances[column_variances >= variance_threshold].index.tolist()
    else:
        # Use pre-computed columns (test mode or federated mode with global feature selection)
        if 'sensor_columns_to_keep' in preprocessing_params:
            sensor_columns_to_keep = preprocessing_params['sensor_columns_to_keep']
        else:
            # Fallback: compute from data if not provided
            variance_threshold = 0.01
            column_variances = df[sensor_columns].var()
            sensor_columns_to_keep = column_variances[column_variances >= variance_threshold].index.tolist()
    
    # Keep all feature columns (sensor + op_setting), but only filtered sensor columns
    non_sensor_features = [col for col in feature_columns if not col.startswith('sensor')]
    columns_to_keep = ['unit_id', 'time_cycles'] + non_sensor_features + sensor_columns_to_keep
    if compute_rul:
        columns_to_keep += ['RUL']
    df = df[columns_to_keep]
    
    # Update sensor_columns to only include the ones we kept
    sensor_columns = [col for col in sensor_columns_to_keep if col in df.columns]
    
    # Update feature_columns to reflect what's actually in the dataframe
    feature_columns = [col for col in df.columns if col not in ['unit_id', 'time_cycles', 'RUL']]
    
    # Step 2: Apply moving average to smooth sensor features (BEFORE normalization)
    df = df.sort_values(['unit_id', 'time_cycles']).reset_index(drop=True)
    window_size_ma = 5  # Moving average window size
    for col in sensor_columns:
        # Apply moving average per unit_id to maintain temporal sequence
        df[col] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=window_size_ma, min_periods=1, center=True).mean()
        )
    
    # Step 3: Normalize sensor columns to scale 0 -> 1 (Min-Max scaling)
    # If preprocessing_params has min_max_values, use them; otherwise compute locally
    if preprocessing_params is None or 'min_max_values' not in preprocessing_params:
        # Compute min/max values locally (for federated learning, each client computes its own)
        min_max_values = {}
        for col in sensor_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            min_max_values[col] = {'min': min_val, 'max': max_val}
            if max_val != min_val:  # Avoid division by zero
                df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        # Use pre-computed min/max values (test mode or centralized training)
        min_max_values = preprocessing_params['min_max_values']
        for col in sensor_columns:
            if col in min_max_values:
                min_val = min_max_values[col]['min']
                max_val = min_max_values[col]['max']
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
    
    # Step 4: Create windows
    windows = []
    rul_windows = []
    
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id].sort_values('time_cycles').reset_index(drop=True)
        
        # Create sliding windows of size window_size
        for i in range(len(unit_data) - window_size + 1):
            window = unit_data.iloc[i:i + window_size]
            # Extract feature columns for the window (exclude unit_id, time_cycles, RUL)
            window_features = window[feature_columns]
            windows.append(window_features)
            
            # Store RUL from the last row of each window (if available)
            if compute_rul and 'RUL' in window.columns:
                rul_windows.append(window['RUL'].iloc[-1])
    
    # Convert windows to numpy array
    windows_array = np.array([window.values for window in windows])
    
    # Prepare return values
    preprocessing_params_dict = {
        'sensor_columns_to_keep': sensor_columns_to_keep,
        'min_max_values': min_max_values,
        'feature_columns': feature_columns
    }
    
    if compute_rul:
        rul_array = np.array(rul_windows)
        return windows_array, rul_array, preprocessing_params_dict
    else:
        return windows_array, preprocessing_params_dict

def save_preprocessing_params(preprocessing_params, file_path):
    """Save preprocessing parameters to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(preprocessing_params, f)

def load_preprocessing_params(file_path):
    """Load preprocessing parameters from a file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# For backward compatibility - run preprocessing if script is executed directly
if __name__ == "__main__":
    # Default: process training data
    train_file = 'CMAPSSData/train_FD001.txt'
    windows_array, rul_array, preprocessing_params = preprocess_data(train_file, compute_rul=True)
    
    print(f"Number of windows created: {len(windows_array)}")
    print(f"Window shape: {windows_array.shape}")
    print(f"RUL array shape: {rul_array.shape}")
    print(f"First window shape: {windows_array[0].shape}")
    print(windows_array[0])
