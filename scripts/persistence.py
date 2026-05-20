"""
Electricity Consumption Forecasting - Persistence Baseline

This script evaluates the naive Persistence baseline (ŷ(t+1) = y(t))
for daily electricity consumption forecasting using RTE data.

It loads the dataset, applies a temporal train/test split (2022+),
computes forecasts and evaluates performance with MAE and MAPE.

Used as a reference benchmark for more advanced models.

Author: Éric Duhamel
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# -----------------------------------
# Persistence baseline
# -----------------------------------


def persistence_forecast(series):
    """
    y(t+1) = y(t)
    """
    return series[:-1], series[1:]


# -----------------------------------
# Data loading
# -----------------------------------

script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
file_path = base_dir / "data" / "RTE" / "rte_daily_consumption_2012_2024.csv"


df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"])

# Temporal split
mask_test = df["date"] >= "2022-01-01"

# The series is complete, we'll not check for missing values here
test_series = df.loc[mask_test, "consumption_MWh"].values

y_true_persist, y_pred_persist = persistence_forecast(test_series)
mae_persist = mean_absolute_error(y_true_persist, y_pred_persist)


print("\n=== PERSISTENCE BASELINE ===")
print(f"Persistence MAE: {mae_persist:,.2f} MWh")
print(f"Number of observations: {len(y_true_persist):,}")
print(
    f"Period: {df.loc[mask_test, 'date'].min():%Y-%m-%d} → {df.loc[mask_test, 'date'].max():%Y-%m-%d}"
)
mape_persist = mean_absolute_percentage_error(y_true_persist, y_pred_persist) * 100
print(f"Persistence MAPE: {mape_persist:.2f}%")
