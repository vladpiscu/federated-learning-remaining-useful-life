import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
import flwr as fl
from models.lstm_model import build_lstm_model
from preprocessing.preprocessing import (
    COLUMNS,
    preprocess_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flower federated learning client for CMAPSS RUL prediction."
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        required=True,
        help="Unit ID (device ID) to train on. This client will only use data from this unit.",
    )
    parser.add_argument(
        "--data-dir",
        default="CMAPSSData",
        help="Directory containing CMAPSSData training files.",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help=(
            "Use a single training file instead of all train_*.txt. "
            "Can be a filename (e.g., train_FD001.txt) or a full/relative path."
        ),
    )
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window size (timesteps).")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split for local data.")
    parser.add_argument(
        "--server-address",
        default="localhost:8080",
        help="Server address to connect to (host:port).",
    )
    return parser.parse_args()


def _resolve_single_file(data_dir: str, file_arg: str) -> str:
    if os.path.exists(file_arg):
        return file_arg
    candidate = os.path.join(data_dir, file_arg)
    if os.path.exists(candidate):
        return candidate
    return file_arg


def load_unit_data(data_dir: str, train_file: str | None, unit_id: int):
    """Load data for a specific unit_id from training files."""
    import pandas as pd
    import glob

    if train_file:
        single_train_file = _resolve_single_file(data_dir, train_file)
        if not os.path.exists(single_train_file):
            raise FileNotFoundError(f"Training file not found: {single_train_file}")
        train_files = [single_train_file]
    else:
        train_files = sorted(glob.glob(os.path.join(data_dir, "train_*.txt")))

    if not train_files:
        raise FileNotFoundError(f"No training files found in: {data_dir}")

    all_dataframes = []
    max_unit_id = 0

    for fpath in train_files:
        df = pd.read_csv(fpath, sep=r"\s+", header=None, names=COLUMNS)
        # Make unit_ids unique across files (same as train.py and server)
        df["unit_id"] = df["unit_id"] + max_unit_id
        max_unit_id = int(df["unit_id"].max())
        all_dataframes.append(df)

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Filter to only this unit's data
    unit_df = combined_df[combined_df["unit_id"] == unit_id].copy()
    if len(unit_df) == 0:
        raise ValueError(f"No data found for unit_id={unit_id}")

    return unit_df


class RULClient(fl.client.NumPyClient):
    """Flower client that trains on a specific unit's data with FedProx support."""

    def __init__(self, X_train, y_train, X_val, y_val, input_shape):
        self.model = build_lstm_model(input_shape=input_shape)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.global_weights = None  # Store global weights for FedProx proximal term

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set global model weights and store them for FedProx
        self.model.set_weights(parameters)
        self.global_weights = [np.array(w) for w in parameters]

        # Get FedProx proximal term weight (μ)
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Train locally with FedProx proximal term if μ > 0
        epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 256))

        if proximal_mu > 0.0:
            # Custom training with proximal term
            self._train_with_proximal_term(epochs, batch_size, proximal_mu)
        else:
            # Standard training (FedAvg)
            self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

        # Return updated weights and number of training examples
        return self.model.get_weights(), len(self.X_train), {}

    def _train_with_proximal_term(self, epochs, batch_size, proximal_mu):
        """Train model with FedProx proximal term: loss + (μ/2) * ||w - w_global||²"""
        import tensorflow as tf
        from tensorflow import keras

        optimizer = self.model.optimizer
        
        # Get the actual loss function (model.loss might be a string name)
        loss_fn = self.model.loss
        if isinstance(loss_fn, str):
            # If it's a string, get the actual function from Keras
            loss_fn = keras.losses.get(loss_fn)
        # If it's already callable, use it directly

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        dataset = dataset.shuffle(buffer_size=len(self.X_train)).batch(batch_size)

        for epoch in range(epochs):
            for batch_x, batch_y in dataset:
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = self.model(batch_x, training=True)
                    # Standard loss
                    loss = loss_fn(batch_y, predictions)

                    # Add proximal term: (μ/2) * ||w - w_global||²
                    if self.global_weights is not None:
                        proximal_term = 0.0
                        for local_w, global_w in zip(self.model.trainable_variables, self.global_weights):
                            global_w_tensor = tf.constant(global_w, dtype=local_w.dtype)
                            diff = local_w - global_w_tensor
                            proximal_term += tf.reduce_sum(tf.square(diff))
                        loss = loss + (proximal_mu / 2.0) * proximal_term

                # Compute gradients and update weights
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def evaluate(self, parameters, config):
        # Set global model weights
        self.model.set_weights(parameters)

        # Evaluate on validation set (or training set if no validation)
        if self.X_val is not None and len(self.X_val) > 0:
            loss, mae, mse = self.model.evaluate(self.X_val, self.y_val, verbose=0)
            num_examples = len(self.X_val)
        else:
            # Fallback to training set if no validation data
            loss, mae, mse = self.model.evaluate(self.X_train, self.y_train, verbose=0)
            num_examples = len(self.X_train)

        return float(loss), num_examples, {"mae": float(mae), "mse": float(mse)}


def create_client_for_data(
    data_df,
    window_size: int,
    val_split: float,
    input_shape: tuple,
    global_feature_selection: dict | None = None,
):
    """
    Create a Flower client from a DataFrame partition.
    This is a reusable function that can be called from simulation scripts.
    
    Args:
        data_df: DataFrame containing the data partition for this client
        window_size: Sliding window size
        val_split: Validation split ratio
        input_shape: Model input shape (timesteps, features)
        global_feature_selection: Optional dict with 'sensor_columns_to_keep' to ensure
                                 all clients use the same feature set
    
    Returns:
        RULClient instance
    """
    if len(data_df) == 0:
        raise ValueError("Data partition is empty.")

    # Preprocess this partition's data
    # If global_feature_selection is provided, use it to ensure consistent feature set
    # Each client will still compute its own normalization (min/max) for those features
    preprocessing_params_for_client = None
    if global_feature_selection is not None:
        # Use global feature selection, but let client compute its own normalization
        preprocessing_params_for_client = {
            'sensor_columns_to_keep': global_feature_selection['sensor_columns_to_keep'],
            # min_max_values will be computed by preprocess_data
        }
    
    X, y, preprocessing_params = preprocess_data(
        df=data_df,
        preprocessing_params=preprocessing_params_for_client,  # Use global feature selection if provided
        compute_rul=True,
        window_size=window_size,
    )

    if X is None or len(X) == 0:
        raise ValueError("Data partition has no valid windows (too short).")

    # Split into train/validation
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx].astype(np.float32, copy=False)
    y = y[idx].astype(np.float32, copy=False)

    n_val = max(1, int(round(len(X) * float(val_split)))) if len(X) > 1 else 0
    if n_val > 0:
        X_val, y_val = X[:n_val], y[:n_val]
        X_train, y_train = X[n_val:], y[n_val:]
    else:
        X_train, y_train = X, y
        X_val, y_val = X[:0], y[:0]

    # Create Flower client
    return RULClient(X_train, y_train, X_val, y_val, input_shape)


def create_client_for_unit(
    unit_id: int,
    data_dir: str,
    train_file: str | None,
    window_size: int,
    val_split: float,
    input_shape: tuple,
    global_feature_selection: dict | None = None,
):
    """
    Create a Flower client for a specific unit_id.
    This is a reusable function that can be called from simulation scripts.
    
    Args:
        unit_id: The unit_id to create a client for
        data_dir: Directory containing training files
        train_file: Optional single training file (None = use all files)
        window_size: Sliding window size
        val_split: Validation split ratio
        input_shape: Model input shape (timesteps, features)
        global_feature_selection: Optional dict with 'sensor_columns_to_keep' to ensure
                                 all clients use the same feature set
    
    Returns:
        RULClient instance
    """
    # Load data for this specific unit
    unit_df = load_unit_data(data_dir, train_file, unit_id)
    
    # Use the generic create_client_for_data function
    return create_client_for_data(
        data_df=unit_df,
        window_size=window_size,
        val_split=val_split,
        input_shape=input_shape,
        global_feature_selection=global_feature_selection,
    )


def main():
    # Reduce TF log noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    args = parse_args()

    print("=" * 60)
    print(f"Federated Learning Client - Unit ID {args.unit_id}")
    print("=" * 60)

    # Use the reusable function to create the client
    print(f"\nLoading data for unit_id={args.unit_id}...")
    
    # First, determine input shape by preprocessing a sample
    unit_df = load_unit_data(args.data_dir, args.train_file, args.unit_id)
    print(f"Loaded {len(unit_df)} rows for unit {args.unit_id}")
    
    print("\nPreprocessing unit data (computing local preprocessing parameters)...")
    X, y, preprocessing_params = preprocess_data(
        df=unit_df,
        preprocessing_params=None,
        compute_rul=True,
        window_size=args.window_size,
    )
    print(f"Selected {len(preprocessing_params['sensor_columns_to_keep'])} sensor features")
    print(f"Total features: {len(preprocessing_params['feature_columns'])}")
    
    if X is None or len(X) == 0:
        raise ValueError(f"Unit {args.unit_id} has no valid windows (too short).")
    
    print(f"Created {len(X)} windows from unit {args.unit_id}")
    
    # Derive model input shape from this client's preprocessing
    num_features = len(preprocessing_params["feature_columns"])
    input_shape = (int(args.window_size), int(num_features))
    print(f"\nModel input shape: {input_shape} (timesteps={input_shape[0]}, features={input_shape[1]})")
    
    # Create client using the reusable function (no global feature selection for standalone client)
    client = create_client_for_unit(
        unit_id=args.unit_id,
        data_dir=args.data_dir,
        train_file=args.train_file,
        window_size=args.window_size,
        val_split=args.val_split,
        input_shape=input_shape,
        global_feature_selection=None,  # Standalone client computes its own features
    )
    
    print(f"Training samples: {len(client.X_train)}, Validation samples: {len(client.X_val)}")
    print("Note: All clients must use the same feature selection logic to ensure model compatibility.")

    print(f"\nConnecting to server at {args.server_address}...")
    print("=" * 60)

    # Connect to server and participate in federated learning
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

    print("\n" + "=" * 60)
    print(f"Client (unit_id={args.unit_id}) training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
