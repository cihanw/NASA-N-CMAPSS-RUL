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
FILES_TO_PROCESS = [
    # 1. Development (Train) Seti
    {
        'input': 'data/processed/N-CMAPSS_DS02_Dev_ClimbCruise.parquet',
        'output': 'data/processed/N-CMAPSS_DS02_Dev_WithResiduals.parquet'
    },
    # 2. Test Seti
    {
        'input': 'data/processed/N-CMAPSS_DS02_Test_ClimbCruise.parquet',
        'output': 'data/processed/N-CMAPSS_DS02_Test_WithResiduals.parquet'
    }
]

MODEL_PATH = 'models/model_reference.pth'
SCALER_W_PATH = 'data/scalers/scaler_W.pkl'
SCALER_X_PATH = 'data/scalers/scaler_Xs.pkl'

BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. DEFINE MODEL CLASS
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
            nn.Linear(200, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. PROCESSING FUNCTION
# ---------------------------------------------------------
def generate_residuals_for_file(file_info, model, scaler_W, scaler_Xs):
    input_file = file_info['input']
    output_file = file_info['output']
    
    if not os.path.exists(input_file):
        print(f"⚠️ SKIPPING: Input file not found: {input_file}")
        return

    print(f"\n" + "="*50)
    print(f"🔄 Processing: {input_file}")
    print("="*50)
    
    # 1. Load Data
    df = pd.read_parquet(input_file)
    print(f"Data Loaded. Shape: {df.shape}")

    # Column Names
    W_cols = ['alt', 'Mach', 'TRA', 'T2']
    Xs_cols = ['Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50', 
               'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50']

    # 2. Data Prep (Scaling)
    print("Normalizing input (W) and target (Xs)...")
    
    X_w = df[W_cols].values
    X_w_scaled = scaler_W.transform(X_w)
    
    Y_xs = df[Xs_cols].values
    Y_xs_scaled = scaler_Xs.transform(Y_xs)

    # 3. Inference
    print("Running Inference...")
    dataset = TensorDataset(torch.tensor(X_w_scaled, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(DEVICE)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            
    Y_pred_scaled = np.concatenate(predictions, axis=0)

    # 4. Calculate Residuals
    print("Calculating Residuals (Real - Predicted)...")
    Residuals = Y_xs_scaled - Y_pred_scaled
    
    # 5. Save and Organize Columns
    residual_cols = [f"R_{col}" for col in Xs_cols]
    df_res = pd.DataFrame(Residuals, columns=residual_cols)
    
    # Orijinal veriye ekle
    df_final = pd.concat([df, df_res], axis=1)

    # Drop original sensor columns and additional columns
    # hs KORUNUYOR, T2 SİLİNİYOR.
    additional_drop_cols = ['alt', 'Mach', 'TRA', 'T2'] 
    
    cols_to_drop = Xs_cols + additional_drop_cols
    
    # RUL ve hs silinmesin diye kontrol
    cols_to_drop = [c for c in cols_to_drop if c not in ['RUL', 'hs']]
    cols_to_drop = [c for c in cols_to_drop if c in df_final.columns]
    
    print(f"Dropping columns: {cols_to_drop}...")
    df_final.drop(columns=cols_to_drop, inplace=True)

    # --- SÜTUN SIRALAMA (REORDERING) ---
    # RUL sütununu en sona atıyoruz
    if 'RUL' in df_final.columns:
        print("Reordering columns: Moving RUL to the end...")
        # RUL dışındaki tüm sütunları al
        cols = [c for c in df_final.columns if c != 'RUL']
        # RUL'u listenin en sonuna ekle
        cols.append('RUL')
        # DataFrame'i yeni sıraya göre düzenle
        df_final = df_final[cols]
    else:
        print("❌ CRITICAL ERROR: RUL column missing, cannot reorder.")

    # Float32 optimizasyonu
    df_final = df_final.astype({col: 'float32' for col in residual_cols})
    
    print(f"Final columns: {df_final.columns.tolist()}")
    print(f"Saving to: {output_file}")
    df_final.to_parquet(output_file, index=False)
    print("✅ Done.")

# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- A. Load Scalers (ONCE) ---
    print("Loading scalers...")
    scaler_W = joblib.load(SCALER_W_PATH)
    scaler_Xs = joblib.load(SCALER_X_PATH)
    
    # --- B. Load Model (ONCE) ---
    print(f"Loading model from {MODEL_PATH}...")
    W_cols_len = 4
    Xs_cols_len = 14
    
    model = ReferenceModel(W_cols_len, Xs_cols_len).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 
    
    # --- C. Process Files Loop ---
    for file_info in FILES_TO_PROCESS:
        generate_residuals_for_file(file_info, model, scaler_W, scaler_Xs)
        
    print("\n🎉 ALL FILES PROCESSED SUCCESSFULLY.")