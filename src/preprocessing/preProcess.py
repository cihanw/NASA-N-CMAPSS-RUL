# AUTHOR: Cihan Aygün
import h5py
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
FILENAME = 'data/N-CMAPSS_DS02-006.h5'

# Çıktı dosyalarını ayırıyoruz
OUTPUT_FILE_DEV = 'data/processed/N-CMAPSS_DS02_Dev_ClimbCruise.parquet'
OUTPUT_FILE_TEST = 'data/processed/N-CMAPSS_DS02_Test_ClimbCruise.parquet'

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
        print(f"\n--- Reading {subset_name} data ---")
        
        suffix = 'dev' if subset_name == 'Development' else 'test'
        
        # --- STRUCTURAL CHECK ---
        if subset_name in f:
            base_group = f[subset_name]
            print(f"Structure: Hierarchical (Grouped) - '{subset_name}' group found.")
        else:
            base_group = f
            print(f"Structure: Flat - Reading directly from root.")

        try:
            # 1. W (Auxiliary)
            W = np.array(base_group[f'W_{suffix}'])
            W_cols = ['alt', 'Mach', 'TRA', 'T2']
            
            # 2. X_s (Sensors)
            X_s = np.array(base_group[f'X_s_{suffix}'])
            X_s_cols = ['Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50', 
                        'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50']
            
            # 3. Y (RUL)
            Y = np.array(base_group[f'Y_{suffix}'])
            Y_cols = ['RUL']
            
            # 4. A (Auxiliary data)
            A = np.array(base_group[f'A_{suffix}'])
            A_cols = ['unit', 'cycle', 'Fc', 'hs']
            
        except KeyError as e:
            print(f"\nERROR: Expected dataset not found ({e}).")
            print("File content (Keys):", list(f.keys()))
            raise

        print(f"Concatenating data parts... (Row count: {W.shape[0]})")
        data_all = np.concatenate([A, W, X_s, Y], axis=1)
        
        all_cols = A_cols + W_cols + X_s_cols + Y_cols
        
        df = pd.DataFrame(data_all, columns=all_cols)
        df = df.astype(np.float32)
        
        return df

def label_flight_phases(df):
    """
    Labels flight modes based on physical parameters.
    """
    print("Labeling flight modes...")
    conditions = [
        (df['TRA'] < 40), # Descent
        (df['alt'] > 20000) & (df['Mach'] > 0.5) & (df['TRA'] >= 40), # Cruise
        (df['TRA'] >= 40) & ((df['alt'] <= 20000) | (df['Mach'] <= 0.5)) # Climb
    ]
    choices = ['Descent', 'Cruise', 'Climb']
    df['flight_phase'] = np.select(conditions, choices, default='Unknown')
    return df

def process_pipeline(subset_name, output_file):
    """
    Tekrarlanan işlemleri yapan fonksiyon.
    """
    # 1. Load
    df = load_data(FILENAME, subset_name=subset_name)
    print(f"Loaded {subset_name} shape: {df.shape}")

    # 2. Label
    df = label_flight_phases(df)
    
    # 3. Filter Descent
    print("Filtering out 'Descent'...")
    df_filtered = df[df['flight_phase'] != 'Descent'].copy()
    df_filtered = df_filtered[df_filtered['flight_phase'] != 'Unknown']
    df_filtered.reset_index(drop=True, inplace=True)
    
    print(f"Filtered {subset_name} shape: {df_filtered.shape}")

    # 4. Save
    print(f"Saving to: {output_file}")
    # Drop 'flight_phase' columns before saving
    if 'flight_phase' in df_filtered.columns:
        df_filtered.drop(columns=['flight_phase'], inplace=True)
    
    df_filtered.to_parquet(output_file, index=False)
    print("✅ Done.")

# ---------------------------------------------------------
# MAIN BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        # Development Setini İşle
        process_pipeline('Development', OUTPUT_FILE_DEV)
        
        print("\n" + "="*50 + "\n")
        
        # Test Setini İşle
        process_pipeline('Test', OUTPUT_FILE_TEST)
        
        print("\n🎉 ALL DATASETS PROCESSED SUCCESSFULLY.")
        
    except Exception as e:
        print(f"\n❌ AN ERROR OCCURRED: {e}")