# AUTHOR: Cihan Aygün
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
INPUT_FILE = 'data/processed/N-CMAPSS_DS02_WithResiduals.parquet'
OUTPUT_FILE = 'data/processed/N-CMAPSS_DS02_WithResiduals-piecewiseLinear.parquet' 

# Literature Standard: 125 or 130 Cycles
RUL_CLIP_LIMIT = 125 

# Smoothing Parameter (Konuştuğumuz 15 adımlık pencere)
SMOOTHING_WINDOW = 15

# ---------------------------------------------------------
# PROCESS
# ---------------------------------------------------------
print(f"Loading data: {INPUT_FILE} ...")
df = pd.read_parquet(INPUT_FILE)

# --- CHECK 1: BEFORE PROCESSING ---
print("\n" + "="*60)
print("🛑 DATA BEFORE PROCESSING (First 10 Rows)")
print("="*60)
# Show relevant columns including a residual example (e.g., R_T24) to see the change
cols_to_show = ['unit', 'cycle', 'RUL', 'flight_phase']
# R_T24 varsa listeye ekle, değişimini gözlemleyelim
if 'R_T24' in df.columns:
    cols_to_show.append('R_T24')

if 'RUL_true' in df.columns:
    cols_to_show.append('RUL_true')

print(df[cols_to_show].head(10))
print("-" * 60)


# =========================================================
# STEP 1: RESIDUAL SMOOTHING (MOVING AVERAGE) - YENİ EKLENDİ
# =========================================================
# Sadece 'R_' ile başlayan residual sütunlarını seç
residual_cols = [col for col in df.columns if col.startswith('R_')]

if residual_cols:
    print(f"\nApplying Moving Average (Window={SMOOTHING_WINDOW}) to {len(residual_cols)} residual columns...")
    print(f"Columns: {residual_cols}")
    
    # Her unit (motor) birbirinden bağımsızdır. 
    # Unit 1'in son verisi Unit 2'nin başını etkilemesin diye 'groupby' kullanıyoruz.
    # min_periods=1: İlk 15 satırda NaN oluşmasını engeller, elindeki kadar veriyle ortalama alır.
    df[residual_cols] = df.groupby('unit')[residual_cols].transform(
        lambda x: x.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    )
    print("✅ Smoothing completed.")
else:
    print("\n⚠️ WARNING: No columns starting with 'R_' found. Smoothing skipped.")


# =========================================================
# STEP 2: RUL BACKUP AND CLIPPING
# =========================================================
if 'RUL_true' not in df.columns:
    print("\nBacking up original RUL to 'RUL_true' column...")
    df['RUL_true'] = df['RUL'].copy()
else:
    print("\n'RUL_true' column already exists, backup skipped.")

print(f"Clipping RUL value to upper limit {RUL_CLIP_LIMIT} (Clipping)...")
df['RUL'] = df['RUL_true'].clip(upper=RUL_CLIP_LIMIT)


# =========================================================
# STEP 3: DROP FLIGHT PHASE
# =========================================================
if 'flight_phase' in df.columns:
    print("\nDropping 'flight_phase' column...")
    df.drop(columns=['flight_phase'], inplace=True)
    print("✅ 'flight_phase' column successfully removed.")
else:
    print("\n⚠️ 'flight_phase' column not found (maybe already removed).")


# --- CHECK 2: AFTER PROCESSING ---
print("\n" + "="*60)
print("✅ DATA AFTER PROCESSING (First 10 Rows)")
print("="*60)
# Update list for display
cols_to_show_after = ['unit', 'cycle', 'RUL', 'RUL_true']
if 'R_T24' in df.columns:
    cols_to_show_after.append('R_T24')

print(df[cols_to_show_after].head(10))
print("-" * 60)


# =========================================================
# STEP 4: VISUAL CHECKS & SAVE
# =========================================================
unit_id = df['unit'].unique()[0]
sample_data = df[df['unit'] == unit_id]

# Grafik 1: RUL Değişimi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(sample_data['cycle'], sample_data['RUL_true'], label='True RUL', linestyle='--')
plt.plot(sample_data['cycle'], sample_data['RUL'], label=f'Target (Clipped)', linewidth=2)
plt.xlabel('Cycle')
plt.ylabel('RUL')
plt.title('RUL Clipping Effect')
plt.legend()
plt.grid(True)

# Grafik 2: Smoothing Etkisi (Bir Residual örneği üzerinden)
if 'R_T24' in df.columns:
    plt.subplot(1, 2, 2)
    # Smoothing sonrası veri zaten df içinde güncellendiği için burada sadece onu çizebiliriz.
    # Ancak etkinin çalıştığını görmek için düz çizgi mi yoksa hala çok mu gürültülü ona bakabilirsin.
    plt.plot(sample_data['cycle'], sample_data['R_T24'], label='Smoothed R_T24', color='orange')
    plt.xlabel('Cycle')
    plt.title(f'Smoothed Residual Example (Window={SMOOTHING_WINDOW})')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('Process_Check_Plot.png')
print("\nPlot saved as 'Process_Check_Plot.png'.")

# Save
print(f"Updating file: {OUTPUT_FILE}")
df.to_parquet(OUTPUT_FILE, index=False)

print("\n✅ All processes completed successfully.")