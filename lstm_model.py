"""
LSTM Model for Stock Direction Prediction.

Architecture:
- Input: (batch_size, seq_length, n_features) — sequence of past N days
- LSTM layers → Fully connected → Sigmoid (probability of UP)

The model looks at a window of past days to predict if the stock
will go UP or DOWN in the next 10 trading days.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class StockLSTM(nn.Module):
    """
    LSTM classifier for stock direction prediction.
    
    Args:
        input_size:  Number of features per day
        hidden_size: LSTM hidden units (default 64)
        num_layers:  Number of LSTM layers (default 2)
        dropout:     Dropout rate to prevent overfitting (default 0.2)
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # Input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()           # Output: probability between 0 and 1
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            predictions: (batch_size, 1) — probability of UP
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step's output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        prediction = self.fc(last_output)
        return prediction


def create_sequences(features, target, seq_length=30):
    """
    Create sequences for LSTM training.
    
    For each day, we take the past `seq_length` days of features
    as input, and the corresponding day's target as label.
    
    Args:
        features:   numpy array of shape (n_days, n_features)
        target:     numpy array of shape (n_days,)
        seq_length: number of past days to use as input
    
    Returns:
        X: (n_samples, seq_length, n_features)
        y: (n_samples,)
    """
    X, y = [], []
    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length:i])  # Past N days
        y.append(target[i])                      # Current day's target
    return np.array(X), np.array(y)


def train_lstm(model, X_train, y_train, epochs=50, lr=0.001, batch_size=64):
    """
    Train the LSTM model.
    
    Args:
        model:    StockLSTM instance
        X_train:  (n_samples, seq_length, n_features)
        y_train:  (n_samples,)
        epochs:   training epochs
        lr:       learning rate
        batch_size: mini-batch size
    
    Returns:
        model: trained model
        losses: list of training losses
    """
    device = torch.device('cpu')
    model = model.to(device)
    model.train()
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device).unsqueeze(1)
    
    # Binary cross-entropy loss + Adam optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    losses = []
    n_samples = len(X_tensor)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            X_batch = X_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item() * (end - start)
        
        avg_loss = epoch_loss / n_samples
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
    
    return model, losses


def predict_lstm(model, X, batch_size=256):
    """
    Make predictions with the trained LSTM model.
    
    Args:
        model: trained StockLSTM
        X:     (n_samples, seq_length, n_features)
    
    Returns:
        probabilities: numpy array of shape (n_samples,)
    """
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    predictions = []
    
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            end = min(start + batch_size, len(X_tensor))
            batch = X_tensor[start:end]
            output = model(batch)
            predictions.append(output.cpu().numpy())
    
    return np.concatenate(predictions, axis=0).flatten()


def prepare_lstm_data(full_data, target_col, date_idx, retrain_years=5, seq_length=30):
    """
    Prepare scaled sequences for LSTM training and prediction.
    
    Args:
        full_data:     DataFrame with all features
        target_col:    name of target column
        date_idx:      current date index position
        retrain_years: years of data to use
        seq_length:    sequence length for LSTM
    
    Returns:
        X_train:  (n_samples, seq_length, n_features)
        y_train:  (n_samples,)
        X_today:  (1, seq_length, n_features) — for prediction
        scaler:   fitted StandardScaler
        feature_cols: list of feature column names
    """
    start_idx = max(0, date_idx - retrain_years * 252)
    window_data = full_data.iloc[start_idx:date_idx + 1]
    
    # Separate features and target
    feature_cols = [c for c in window_data.columns if c != target_col]
    X_raw = window_data[feature_cols].values
    y_raw = window_data[target_col].values
    
    # Scale features (fit on training data only)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw[:-1])  # Fit on everything except today
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    # Create sequences for training
    if len(X_scaled) < seq_length + 10:
        return None, None, None, None, None
    
    X_train, y_train = create_sequences(X_scaled, y_raw[:-1], seq_length)
    
    # Prepare today's input (past seq_length days including today)
    today_scaled = scaler.transform(X_raw[-seq_length:])
    today_scaled = np.nan_to_num(today_scaled, nan=0.0)
    X_today = today_scaled.reshape(1, seq_length, -1)
    
    return X_train, y_train, X_today, scaler, feature_cols
