# AUTHOR: Cihan Aygün
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ---------------------------------------------------------
# SETTINGS & DEVICE SELECTION
# ---------------------------------------------------------
INPUT_FILE = 'data/processed/N-CMAPSS_DS02_ClimbCruise.parquet'
MODEL_SAVE_PATH = 'models/model_reference.pth'
SCALER_W_PATH = 'data/scalers/scaler_W.pkl'
SCALER_X_PATH = 'data/scalers/scaler_Xs.pkl'

# Hyperparameters
HEALTHY_CYCLE_LIMIT = 15
EPOCHS = 100            # Can use more epochs in PyTorch, early stopping will handle it
BATCH_SIZE = 512
LEARNING_RATE = 0.001
PATIENCE = 5            # Early Stopping patience

# GPU Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------
# 1. DATA LOADING AND PREPARATION
# ---------------------------------------------------------
def prepare_data():
    print(f"Reading data: {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)
    
    # Separate only healthy data
    print(f"Extracting first {HEALTHY_CYCLE_LIMIT} cycles (Healthy Data)...")
    df_healthy = df[df['cycle'] <= HEALTHY_CYCLE_LIMIT].copy()
    
    W_cols = ['alt', 'Mach', 'TRA', 'T2']
    Xs_cols = ['Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50', 
               'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50']
    
    X = df_healthy[W_cols].values
    y = df_healthy[Xs_cols].values
    
    # Train - Validation Split (10% Validation)
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    # Scaling (Critical: Fit scaler only on Train set)
    print("Normalizing...")
    scaler_W = StandardScaler()
    scaler_Xs = StandardScaler()
    
    X_train_scaled = scaler_W.fit_transform(X_train_raw)
    y_train_scaled = scaler_Xs.fit_transform(y_train_raw)
    
    X_val_scaled = scaler_W.transform(X_val_raw)
    y_val_scaled = scaler_Xs.transform(y_val_raw)
    
    # Save scalers
    joblib.dump(scaler_W, SCALER_W_PATH)
    joblib.dump(scaler_Xs, SCALER_X_PATH)
    print("Scaler objects saved.")
    
    # Tensor Conversion and preparation for GPU
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_scaled, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, len(W_cols), len(Xs_cols)

# ---------------------------------------------------------
# 2. MODEL DEFINITION (PyTorch)
# ---------------------------------------------------------
class ReferenceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReferenceModel, self).__init__()
        
        # Literature Architecture: 4 Hidden Layers, 200 Neurons
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim) # Output layer (Linear activation)
        )
        
        # Weight Initialization (He Normal) - Standard for ReLU
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------
def train_model():
    train_loader, val_loader, input_dim, output_dim = prepare_data()
    
    model = ReferenceModel(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nModel Architecture:")
    print(model)
    print("\nTraining Started...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # --- Early Stopping & Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly Stopping triggered! Best Val Loss: {best_val_loss:.5f}")
                break
    
    print(f"Training completed. Model saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()