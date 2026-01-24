# NASA N-CMAPSS RUL Prediction

This project focuses on Remaining Useful Life (RUL) prediction using the NASA N-CMAPSS dataset.

## Preprocessing Pipeline

The raw data undergoes a specific preprocessing pipeline before being used for model training. The steps are as follows:

1.  **Residual Smoothing (Moving Average)**
    *   A moving average filter is applied to all residual columns (columns starting with `R_`).
    *   **Window Size:** 15
    *   **Scope:** The smoothing is applied individually for each unit (engine) to ensure no data leakage occurs between different engines.
    *   **Handling Start of Series:** A minimum period of 1 is used to handle the initial data points where a full window is not available.

2.  **RUL Backup and Piecewise Linear Transformation**
    *   The original RUL values are preserved in a new column named `RUL_true`.
    *   The target variable `RUL` is transformed into a piecewise linear function by clipping the values.
    *   **Clip Limit:** 125 cycles. (RUL values greater than 125 are set to 125).

3.  **Feature Selection**
    *   The `flight_phase` column is removed from the dataset as it is not used in the training process.

## Current Status

*   **Test Set:** Preprocessing is complete.
*   **Train Set:** Currently being processed through the same pipeline.
*   **Next Steps:** Training ML and AI models using the processed data.
