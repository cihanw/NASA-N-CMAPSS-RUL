# AUTHOR: Cihan Aygün
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
INPUT_FILE = "data/processed/N-CMAPSS_DS02_ClimbCruise.parquet"
MODEL_PATH = "models/model_reference.pth"
SCALER_W_PATH = "data/scalers/scaler_W.pkl"
SCALER_X_PATH = "data/scalers/scaler_Xs.pkl"
OUTPUT_FILE = "data/processed/N-CMAPSS_DS02_WithResiduals.parquet"
BATCH_SIZE = 4096  # High batch size for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 1. DEFINE MODEL CLASS (Required for loading)
# ---------------------------------------------------------
class ReferenceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReferenceModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 2. LOAD DATA AND SCALERS
# ---------------------------------------------------------
print(f"Loading data: {INPUT_FILE} ...")
df = pd.read_parquet(INPUT_FILE)

print("\n--- FIRST 5 ROWS (INPUT) ---")
print(df.head())

# Load Scalers
print("Loading scalers...")
scaler_W = joblib.load(SCALER_W_PATH)
scaler_Xs = joblib.load(SCALER_X_PATH)

# Column Names
W_cols = ["alt", "Mach", "TRA", "T2"]
Xs_cols = [
    "Wf",
    "Nf",
    "Nc",
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
]

# ---------------------------------------------------------
# 3. DATA PREPARATION (SCALING)
# ---------------------------------------------------------
print("Normalizing input data (W)...")
X_w = df[W_cols].values
X_w_scaled = scaler_W.transform(X_w)

# Normalize actual sensor data as well (For residual calculation)
print("Normalizing actual sensor data (Xs)...")
Y_xs = df[Xs_cols].values
Y_xs_scaled = scaler_Xs.transform(Y_xs)

# ---------------------------------------------------------
# 4. LOAD MODEL AND PREDICT (INFERENCE)
# ---------------------------------------------------------
input_dim = len(W_cols)
output_dim = len(Xs_cols)

model = ReferenceModel(input_dim, output_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Evaluation mode (disables Dropout etc.)

print(f"Model predicting on {DEVICE}...")

# Convert Data to Tensor
dataset = TensorDataset(torch.tensor(X_w_scaled, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

predictions = []

with torch.no_grad():  # No gradient calculation, only forward pass
    for batch in dataloader:
        inputs = batch[0].to(DEVICE)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

# Convert list to a single numpy array
Y_pred_scaled = np.concatenate(predictions, axis=0)

print("Prediction completed.")

# ---------------------------------------------------------
# 5. RESIDUAL CALCULATION
# ---------------------------------------------------------
# Residual = Real (Scaled) - Predicted (Scaled)
print("Calculating residuals...")
Residuals = Y_xs_scaled - Y_pred_scaled

# ---------------------------------------------------------
# 6. SAVING
# ---------------------------------------------------------
# Name residual columns (e.g., R_T24, R_Nf ...)
residual_cols = [f"R_{col}" for col in Xs_cols]

# Add to DataFrame
df_res = pd.DataFrame(Residuals, columns=residual_cols)

# Add to Original Data (Concat)
# We can keep only necessary parts from original data (Unit, Cycle, RUL, Flight_Phase and Residuals)
# But keeping all is good for analysis for now.
df_final = pd.concat([df, df_res], axis=1)

# Memory optimization (Float32)
df_final = df_final.astype({col: "float32" for col in residual_cols})

print(f"Saving file: {OUTPUT_FILE}")

print("\n--- FIRST 5 ROWS (OUTPUT) ---")
print(df_final.head())
df_final.to_parquet(OUTPUT_FILE, index=False)

print("\n✅ PROCESS COMPLETED.")
print(f"New dataset dimensions: {df_final.shape}")
print("Sample Residual Columns:", residual_cols[:3])
