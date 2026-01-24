# AUTHOR: Cihan Aygün
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the prepared Parquet file
filename = 'data/processed/N-CMAPSS_DS02_ClimbCruise.parquet'
df = pd.read_parquet(filename)

print("Data loaded, analysis starting...")

# 'hs' column indicates Health State.
# In N-CMAPS typically:
# hs = 1.0  -> Fully Healthy
# hs < 1.0  -> Degradation Started
# Or vice versa (0 healthy, 1 degraded). Let's check this first.

# Let's look at hs values for Unit 2 (Engine) as an example
unit_id = 2.0 
sample_unit = df[df['unit'] == unit_id]

# Plot the change in hs value over time
plt.figure(figsize=(10, 4))
plt.plot(sample_unit['cycle'], sample_unit['hs'], label='Health State (hs)')
plt.xlabel('Cycle (Flight Count)')
plt.ylabel('Health State')
plt.title(f'Unit {int(unit_id)} Health State Change')
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------------------------------------
# STATISTICAL ANALYSIS: WHEN DOES EACH ENGINE FAIL?
# ---------------------------------------------------------

degradation_start_cycles = []

# Iterate through all units
for u in df['unit'].unique():
    unit_data = df[df['unit'] == u]
    
    # Health state value in healthy condition (Value at first cycle)
    initial_hs = unit_data.iloc[0]['hs']
    
    # Find the first moment degradation starts (where hs deviates from initial value)
    # Tolerance comparison (for float precision)
    degradation_point = unit_data[abs(unit_data['hs'] - initial_hs) > 0.001]
    
    if not degradation_point.empty:
        start_cycle = degradation_point.iloc[0]['cycle']
        degradation_start_cycles.append(start_cycle)
    else:
        # If never degraded (could be in Test data)
        pass

# Print Results
min_start = min(degradation_start_cycles)
avg_start = np.mean(degradation_start_cycles)

print(f"\n--- ANALYSIS RESULT ---")
print(f"Total Engines Analyzed: {len(df['unit'].unique())}")
print(f"Earliest Degradation Start: {min_start}. Cycle")
print(f"Average Degradation Start: {avg_start:.1f}. Cycle")
print(f"Latest Degradation Start: {max(degradation_start_cycles)}. Cycle")

print(f"\nDECISION:")
if min_start > 15:
    print(f"✅ 'First 15 Cycle' assumption is SAFE. (Even first {int(min_start)-1} cycles can be used)")
else:
    print(f"⚠️ WARNING: Some engines start degrading at cycle {min_start}. 15 cycle assumption might be WRONG.")