# AUTHOR: Cihan Aygün
import pandas as pd
import numpy as np
import os
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
FILES_TO_PROCESS = [
    {
        'name': 'Development',
        'input': 'data/processed/N-CMAPSS_DS02_Dev_WithResiduals.parquet',
        'out_dl':  'data/processed/DL_train.parquet',
        'out_svr': 'data/processed/SVR_train.parquet',
        'out_rf':  'data/processed/RF_train.parquet'
    },
    {
        'name': 'Test',
        'input': 'data/processed/N-CMAPSS_DS02_Test_WithResiduals.parquet',
        'out_dl':  'data/processed/DL_test.parquet',
        'out_svr': 'data/processed/SVR_test.parquet',
        'out_rf':  'data/processed/RF_test.parquet'
    }
]

# Ortak Ayarlar
RUL_CLIP_LIMIT = 125

# SVR Ayarları
SVR_WINDOW = 15     # Smoothing Window
SVR_STRIDE = 3      # Her 3 satırdan 1'ini al

# RF Ayarları
RF_WINDOW = 50      # Pencere Genişliği
RF_STRIDE = 10      # Kaydırma Adımı
RF_FEATURES = ['mean', 'std', 'ptp', 'trend'] 

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def prepare_common_data(df):
    """
    RUL Clipping ve RUL_true oluşturma işlemleri.
    """
    print("   -> Creating 'RUL_true' and Clipping RUL...")
    
    if 'RUL_true' not in df.columns:
        df['RUL_true'] = df['RUL'].copy()
    
    df['RUL'] = df['RUL_true'].clip(upper=RUL_CLIP_LIMIT)
    
    if 'flight_phase' in df.columns:
        df.drop(columns=['flight_phase'], inplace=True)
        
    return df

def process_svr_branch(df, output_path):
    """
    SVR: Moving Average + Subsampling.
    DÜZELTME: 'unit' sütununun kaybolması engellendi.
    """
    print(f"   -> Generating SVR Data (Window={SVR_WINDOW}, Stride={SVR_STRIDE})...")
    df_svr = df.copy()
    
    residual_cols = [c for c in df_svr.columns if c.startswith('R_')]
    
    # 1. Smoothing (Moving Average)
    df_svr[residual_cols] = df_svr.groupby('unit')[residual_cols].transform(
        lambda x: x.rolling(window=SVR_WINDOW, min_periods=SVR_WINDOW).mean()
    )
    df_svr.dropna(inplace=True)
    
    # 2. Subsampling (Seyreltme)
    # DÜZELTME: groupby(...).nth(...) kullanarak sütunların (özellikle 'unit') korunmasını sağlıyoruz.
    # as_index=False diyerek unit'in index olmasını engelliyoruz.
    df_svr = df_svr.groupby('unit', as_index=False).nth(slice(None, None, SVR_STRIDE))
    
    print(f"      SVR Shape: {df_svr.shape}")
    df_svr.to_parquet(output_path, index=False)

def process_rf_branch(df, output_path):
    """
    RF: Feature Extraction + Windowing.
    DÜZELTMELER: 
    1. 'fc' ve 'hs' sütunları eklendi.
    2. 'unit' ve 'cycle' korundu.
    3. RUL sütunları en sona atıldı.
    4. Axis hatası giderildi.
    """
    print(f"   -> Generating RF Data (Window={RF_WINDOW}, Stride={RF_STRIDE})...")
    
    residual_cols = [c for c in df.columns if c.startswith('R_')]
    
    # Meta (Bilgi) sütunlarını belirle: Varsa fc ve hs'yi de ekle
    # RUL sütunlarını buradan ÇIKARIYORUZ çünkü en sona ekleyeceğiz.
    possible_meta_cols = ['unit', 'cycle', 'Fc', 'hs']
    meta_cols = [c for c in possible_meta_cols if c in df.columns]
    
    target_cols = ['RUL', 'RUL_true']
    
    feature_list = []
    
    for unit_id, group in df.groupby('unit'):
        
        # Values Extraction
        vals_res = group[residual_cols].values 
        vals_meta = group[meta_cols].values
        vals_target = group[target_cols].values
        
        if len(vals_res) < RF_WINDOW:
            continue
            
        # --- WINDOWING ---
        # Shape: (num_windows, n_features, window_size) -> (N, 14, 50)
        # Window boyutu EN SON eksene gelir.
        
        # 1. Residuals (Feature Extraction için)
        windows_res = sliding_window_view(vals_res, window_shape=RF_WINDOW, axis=0)[::RF_STRIDE]
        
        # 2. Meta Data (Sadece son adım)
        # Shape: (N, n_meta, 50)
        windows_meta = sliding_window_view(vals_meta, window_shape=RF_WINDOW, axis=0)[::RF_STRIDE]
        # Son zaman adımını al: (N, n_meta)
        meta_final = windows_meta[:, :, -1] 
        
        # 3. Target Data (RUL - Sadece son adım)
        # Shape: (N, n_target, 50)
        windows_target = sliding_window_view(vals_target, window_shape=RF_WINDOW, axis=0)[::RF_STRIDE]
        # Son zaman adımını al: (N, n_target)
        target_final = windows_target[:, :, -1]
        
        # --- FEATURE EXTRACTION (Residuals Üzerinde) ---
        # İstatistikleri ZAMAN ekseni (axis=-1) üzerinde alıyoruz.
        
        f_mean = np.mean(windows_res, axis=-1)
        f_std = np.std(windows_res, axis=-1)
        f_ptp = np.ptp(windows_res, axis=-1)
        f_trend = windows_res[:, :, -1] - windows_res[:, :, 0]
        
        # Özellikleri Birleştir: (N, 14*4)
        features_combined = np.hstack([f_mean, f_std, f_ptp, f_trend])
        
        # --- FINAL STACKING (İstenilen Sırada) ---
        # Sıra: [META] + [FEATURES] + [TARGETS]
        # Meta: unit, cycle, fc, hs
        # Features: R_... istatistikleri
        # Targets: RUL, RUL_true (EN SONDA)
        batch_data = np.hstack([meta_final, features_combined, target_final])
        feature_list.append(batch_data)
        
    # Listeyi birleştir
    if feature_list:
        final_data = np.vstack(feature_list)
        
        # Sütun İsimlerini Oluşturma
        new_cols = []
        new_cols.extend(meta_cols) # Önce meta (unit, cycle, fc, hs)
        
        # Feature isimleri
        for stat in ['mean', 'std', 'ptp', 'trend']:
            for col in residual_cols:
                new_cols.append(f"{col}_{stat}")
        
        new_cols.extend(target_cols) # En sona RUL ve RUL_true
                
        # DataFrame'e çevir
        df_rf = pd.DataFrame(final_data, columns=new_cols)
        
        # Veri tiplerini düzelt (Meta ve RUL genelde int veya float olabilir ama unit int olmalı)
        # Hepsini float32 yapıp, unit ve cycle'ı int'e çevirelim
        df_rf = df_rf.astype('float32')
        if 'unit' in df_rf.columns: df_rf['unit'] = df_rf['unit'].astype(int)
        if 'cycle' in df_rf.columns: df_rf['cycle'] = df_rf['cycle'].astype(int)
        
        print(f"      RF Shape: {df_rf.shape} (Columns: {df_rf.shape[1]})")
        print(f"      RF Columns Check (Last 5): {df_rf.columns[-5:].tolist()}")
        df_rf.to_parquet(output_path, index=False)
    else:
        print("      ⚠️ WARNING: No data generated for RF (dataset too small?)")

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    for file_info in FILES_TO_PROCESS:
        subset = file_info['name']
        input_file = file_info['input']
        
        if not os.path.exists(input_file):
            print(f"⚠️ Missing file: {input_file}")
            continue
            
        print(f"\n" + "="*60)
        print(f"🚀 PROCESSING: {subset}")
        print("="*60)
        
        # 1. Yükle
        df = pd.read_parquet(input_file)
        
        # 2. Ortak İşlemler
        df = prepare_common_data(df)
        
        # --- DL BRANCH ---
        print(f"   -> Saving DL Data...")
        df.to_parquet(file_info['out_dl'], index=False)
        
        # --- SVR BRANCH ---
        process_svr_branch(df, file_info['out_svr'])
        
        # --- RF BRANCH ---
        process_rf_branch(df, file_info['out_rf'])
        
    print("\n🎉 ALL BRANCHES GENERATED SUCCESSFULLY.")

if __name__ == "__main__":
    main()