# NASA N-CMAPSS RUL Prediction

This project focuses on Remaining Useful Life (RUL) prediction using the NASA N-CMAPSS dataset.

## Preprocessing Pipeline

The raw data undergoes a comprehensive 7-step preprocessing pipeline to prepare it for model training.

### Step 1: Raw Data Analysis and Configuration (Raw Data Acquisition)
*   **Process:** The hierarchical data structure in HDF5 format is converted into an analyzable planar (tabular) format. The dataset is decomposed into operational conditions ($W$), sensor measurements ($X_s$), and auxiliary data ($A$) to create a matrix structure suitable for modeling.
*   **Reason:** Unlike legacy C-MAPSS datasets, N-CMAPSS data includes real flight data which is non-stationary.

### Step 2: Flight Phase Labeling
*   **Process:** Flight modes not explicitly stated in the raw data are labeled as **Climb**, **Cruise**, and **Descent** using physical threshold values on Altitude (alt), Mach Number (Mach), and Throttle Resolver Angle (TRA) parameters.
*   **Reason:** The engine's thermodynamic responses vary across different flight modes. Separating data into these modes is a critical prerequisite for subsequent filtering and analysis stages.

### Step 3: Data Filtration (Descent Removal)
*   **Process:** Data blocks labeled as "Descent" are removed from the training set.
*   **Reason:** During descent, the engine is typically near idle and produces low power. Sensor data collected in this phase has low information gain regarding the engine's health status and contains a high signal-to-noise ratio. Only **Climb** and **Cruise** data, where the engine is under load, are retained to improve model stability.

### Step 4: Healthy State Reference Modeling
*   **Process:** The first 15 cycles of each engine unit's life are isolated and assumed to represent the "fully healthy" state. A deep learning (MLP) model is trained using this healthy data to predict sensor values ($X_s$) using operational conditions ($W$) as input.
*   **Reason:** Classical normalization methods (e.g., Min-Max) cannot filter out large variations caused by operational conditions (e.g., temperature drop as altitude increases). This reference model mathematically models the effect of physical conditions by learning the "ideal" values the engine should have under current conditions.

### Step 5: Residual Generation and Feature Extraction
*   **Process:** The entire dataset is passed through the reference model, and residual matrices are generated using the formula `Residual = Actual Value - Predicted Value`. Subsequently, "Flight Phase" labels (categorical data), which are not used as model inputs, are discarded.
*   **Reason:** This is the most critical step (Domain Adaptation). The resulting residuals are pure signals representing only internal engine degradation, cleansed of flight condition effects (e.g., pilot throttling, climbing). This enables the model to distinguish between a climb maneuver and a fault.

### Step 6: Signal Smoothing (Residual Smoothing via Moving Average)
*   **Process:** A **Moving Average** filter with a window size of 15 steps ($W=15$) is applied to the calculated raw residual values.
*   **Reason:** Residuals obtained from the reference model contain stochastic sensor noise and momentary model prediction errors in addition to the true engine health status. High-frequency noise can hamper the RUL prediction model's learning process and lead to overfitting. Smoothing "irons out" the signal, minimizing noise and making the degradation trend more distinct.

### Step 7: Target Variable Adjustment (Piecewise Linear RUL Target)
*   **Process:** The Remaining Useful Life (RUL) target values are clipped at an upper limit of **125 flight cycles**. For all data points where the actual life exceeds 125 cycles, the RUL value is fixed at 125. Once it drops below 125, it is allowed to decrease linearly.
*   **Reason:** In the initial "healthy stage" where parts are new and wear has not yet begun, no significant changes are observed in sensor data. Attempting to model a scenario where inputs (sensors/residuals) remain constant but the output (RUL) continuously decreases makes convergence difficult for neural networks. The Piecewise Linear approach aligns the correlation between sensor degradation signatures and the RUL target with physical reality.

## Current Status

*   **Train Set:** Done.
*   **Test Set:** Done.
*   **Next Steps:** Training ML and AI models using the processed data.
