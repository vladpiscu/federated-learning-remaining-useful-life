import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

@keras.utils.register_keras_serializable(package="losses")
class QuantileLoss(keras.losses.Loss):
    """
    Quantile (pinball) loss.

    For error e = y_true - y_pred:
      - under-prediction (y_pred < y_true, e > 0) is weighted by q
      - over-prediction  (y_pred > y_true, e < 0) is weighted by (1 - q)

    So to penalize predictions ABOVE target more, choose q < 0.5 (e.g. q=0.3).
    """

    def __init__(self, q: float = 0.3, name: str = "quantile_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1). Got {q}.")
        self.q = float(q)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        e = y_true - y_pred
        # Pinball loss (keeps sign of error; asymmetric by quantile q)
        # If you want to penalize OVER-prediction more, pick q < 0.5 (e.g., 0.3).
        loss = tf.maximum(self.q * e, (self.q - 1.0) * e)
        # Reduce last axis so the loss is per-sample (shape: [batch]).
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"q": self.q})
        return config

@keras.utils.register_keras_serializable(package="losses")
class CustomLoss(keras.losses.Loss):
    """
    Quantile (pinball) loss.

    For error e = y_true - y_pred:
      - under-prediction (y_pred < y_true, e > 0) is weighted by q
      - over-prediction  (y_pred > y_true, e < 0) is weighted by (1 - q)

    So to penalize predictions ABOVE target more, choose q < 0.5 (e.g. q=0.3).
    """

    def __init__(self, q: float = 0.3, name: str = "quantile_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1). Got {q}.")
        self.q = float(q)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        e = y_true - y_pred
        # Pinball loss (keeps sign of error; asymmetric by quantile q)
        # If you want to penalize OVER-prediction more, pick q < 0.5 (e.g., 0.3).

        #The loss should be computed this way: if e < 0, we use mean squared error, otherwise we use mean absolute error.
        loss = tf.where(e < 0, 0.6 * tf.abs(e) ** 2, 0.4 * tf.abs(e) ** 2)
        # Reduce last axis so the loss is per-sample (shape: [batch]).
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"q": self.q})
        return config


def build_lstm_model(input_shape, lstm_units=256, dropout_rate=0.6):
    """
    Build an LSTM model for Remaining Useful Life (RUL) prediction.
    
    Args:
        input_shape: Tuple (timesteps, features) - shape of input sequences
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    # First LSTM layer with return_sequences=True to pass sequences to next layer
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    #model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(lstm_units // 2, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Dense layers for regression
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation='relu'))
    
    # Output layer (single value for RUL prediction)
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"],
    )
    
    return model

def train_model(X, y, validation_split=0.2, epochs=30, batch_size=128, verbose=1):
    """
    Train the LSTM model on the preprocessed data.
    
    Args:
        X: Input sequences (num_windows, timesteps, features)
        y: Target RUL values (num_windows,)
        validation_split: Fraction of data to use for validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity mode
    
    Returns:
        Trained model and training history
    """
    # Ensure data is float32 and check for NaN/Inf
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    # Check for NaN or Inf values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: X contains NaN or Inf values. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("Warning: y contains NaN or Inf values. Replacing with 0.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, shuffle=True
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Training RUL range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"Validation RUL range: [{y_val.min():.2f}, {y_val.max():.2f}]")
    print(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_lstm_model(input_shape)
    
    # Display model architecture
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history

def predict_rul(model, X):
    """
    Predict RUL for given input sequences.
    
    Args:
        model: Trained Keras model
        X: Input sequences (num_windows, timesteps, features)
    
    Returns:
        Predicted RUL values
    """
    predictions = model.predict(X, verbose=0)
    return predictions.flatten()

if __name__ == "__main__":
    # This will be used when the model is run directly
    # The preprocessing script should save the processed data first
    print("LSTM Model for RUL Prediction")
    print("Load preprocessed data and train the model")
