# AUTHOR: Cihan Aygün
import h5py
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
FILENAME = 'data/N-CMAPSS_DS02-006.h5'
OUTPUT_FILE = 'data/processed/N-CMAPSS_DS02_ClimbCruise.parquet'

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def load_data(filepath, subset_name='Development'):
    """
    Reads data from N-CMAPSS h5 file.
    Supports both Hierarchical (Grouped) and Flat structures.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        print(f"--- Reading {subset_name} data ---")
        
        suffix = 'dev' if subset_name == 'Development' else 'test'
        
        # --- STRUCTURAL CHECK ---
        # Checking if 'Development' or 'Test' group exists.
        if subset_name in f:
            # Standard Structure: f['Development']['W_dev']
            base_group = f[subset_name]
            print(f"Structure: Hierarchical (Grouped) - '{subset_name}' group found.")
        else:
            # Flat Structure: f['W_dev']
            base_group = f
            print(f"Structure: Flat - Reading directly from root.")

        # Reading Datasets
        try:
            # 1. W (Auxiliary / Scenario descriptors) - 4 Columns
            W = np.array(base_group[f'W_{suffix}'])
            W_cols = ['alt', 'Mach', 'TRA', 'T2']
            
            # 2. X_s (Sensor measurements) - 14 Columns
            X_s = np.array(base_group[f'X_s_{suffix}'])
            X_s_cols = ['Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50', 
                        'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50']
            
            # 3. Y (RUL) - Target Variable
            Y = np.array(base_group[f'Y_{suffix}'])
            Y_cols = ['RUL']
            
            # 4. A (Auxiliary data) - Unit ID, Cycle etc.
            A = np.array(base_group[f'A_{suffix}'])
            A_cols = ['unit', 'cycle', 'Fc', 'hs']
            
        except KeyError as e:
            print(f"\nERROR: Expected dataset not found ({e}).")
            print("File content (Keys):", list(f.keys()))
            raise

        # Concatenation
        # Dimension check: Row counts must vary
        print(f"Concatenating data parts... (Row count: {W.shape[0]})")
        data_all = np.concatenate([A, W, X_s, Y], axis=1)
        
        # Column names
        all_cols = A_cols + W_cols + X_s_cols + Y_cols
        
        # DataFrame creation (float32 for memory optimization)
        df = pd.DataFrame(data_all, columns=all_cols)
        df = df.astype(np.float32)
        
        return df

def label_flight_phases(df):
    """
    Labels flight modes based on physical parameters.
    Separates based on TRA (Throttle), Altitude, and Mach values.
    """
    print("Labeling flight modes...")
    
    # Conditions
    conditions = [
        # Descent: Low TRA (Idle/Descent)
        (df['TRA'] < 40),
        
        # Cruise: High Altitude (>20k ft) AND High Speed (>0.5 Mach) AND Active Throttle
        (df['alt'] > 20000) & (df['Mach'] > 0.5) & (df['TRA'] >= 40),
        
        # Climb: Remaining (Throttle active but altitude/speed not yet cruise)
        (df['TRA'] >= 40) & ((df['alt'] <= 20000) | (df['Mach'] <= 0.5))
    ]
    
    choices = ['Descent', 'Cruise', 'Climb']
    
    # Labeling
    df['flight_phase'] = np.select(conditions, choices, default='Unknown')
    return df

# ---------------------------------------------------------
# MAIN BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. Load Data
        df = load_data(FILENAME, subset_name='Development')
        
        print("\n--- FIRST 5 ROWS (RAW) ---")
        print(df.head())
        
        # 2. Label
        df = label_flight_phases(df)
        
        # Show statistics
        print("\n--- Labeling Results ---")
        print(df['flight_phase'].value_counts())
        
        # 3. Filter out Descent
        print("\nCleaning 'Descent' data...")
        df_filtered = df[df['flight_phase'] != 'Descent'].copy()
        
        # Clean Unknown if exists
        df_filtered = df_filtered[df_filtered['flight_phase'] != 'Unknown']
        df_filtered.reset_index(drop=True, inplace=True)
        
        print(f"Original Data Shape: {df.shape}")
        print(f"Filtered Data Shape: {df_filtered.shape}")
        
        # 4. Save (Parquet)
        print(f"\nSaving file: {OUTPUT_FILE}")
        
        print("\n--- FIRST 5 ROWS (PROCESSED) ---")
        print(df_filtered.head())
        
        df_filtered.to_parquet(OUTPUT_FILE, index=False)
        
        print("\n✅ PROCESS COMPLETED SUCCESSFULLY.")
        
    except Exception as e:
        print(f"\n❌ AN ERROR OCCURRED: {e}")