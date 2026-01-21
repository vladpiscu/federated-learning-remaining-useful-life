import numpy as np
import sys
import os
import glob

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import preprocessing and model functions
from preprocessing.preprocessing import preprocess_data, save_preprocessing_params
from models.lstm_model import train_model, predict_rul

def main():
    """
    Main training script that:
    1. Loads and preprocesses all training data files
    2. Saves preprocessing parameters for use with test data
    3. Trains the LSTM model
    4. Saves the trained model
    """
    
    print("=" * 60)
    print("Training LSTM Model for Remaining Useful Life Prediction")
    print("=" * 60)
    
    # Find all training files
    data_dir = 'CMAPSSData'
    train_files = sorted(glob.glob(os.path.join(data_dir, 'train_*.txt')))
    
    if not train_files:
        print(f"Error: No training files found in {data_dir}")
        return
    
    print(f"\nFound {len(train_files)} training file(s):")
    for f in train_files:
        print(f"  - {f}")
    
    # Load all training files into a single dataframe
    print("\nLoading all training files into a single dataframe...")
    import pandas as pd
    from preprocessing.preprocessing import COLUMNS
    
    all_dataframes = []
    max_unit_id = 0
    
    for train_file in train_files:
        print(f"  Loading: {train_file}")
        df = pd.read_csv(train_file, sep='\s+', header=None, names=COLUMNS)
        # Adjust unit_ids to be unique across all files
        df['unit_id'] = df['unit_id'] + max_unit_id
        max_unit_id = df['unit_id'].max()
        all_dataframes.append(df)
        print(f"    Loaded {len(df)} rows, unit_ids: {df['unit_id'].min()}-{df['unit_id'].max()}")
    
    # Combine all dataframes
    print("\nCombining all dataframes...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"  Combined dataframe: {len(combined_df)} rows, {combined_df['unit_id'].nunique()} unique units")
    
    # Preprocess the combined dataframe
    print("\nPreprocessing combined training data...")
    windows_array, rul_array, preprocessing_params = preprocess_data(
        df=combined_df,
        preprocessing_params=None,
        compute_rul=True
    )
    
    print(f"\nCombined preprocessed data shapes:")
    print(f"  Windows array: {windows_array.shape}")
    print(f"  RUL array: {rul_array.shape}")
    
    # Save preprocessing parameters for use with test data
    preprocessing_params_path = 'models/preprocessing_params.pkl'
    os.makedirs('models', exist_ok=True)
    save_preprocessing_params(preprocessing_params, preprocessing_params_path)
    print(f"\nPreprocessing parameters saved to: {preprocessing_params_path}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Training LSTM Model")
    print("=" * 60)
    
    model, history = train_model(
        windows_array,
        rul_array,
        validation_split=0.2,
        epochs=30,
        batch_size=128,
        verbose=1
    )
    
    # Save the trained model
    model_path = os.path.join('models', 'lstm_rul_model.h5')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
