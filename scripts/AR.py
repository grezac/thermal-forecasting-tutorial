"""
Autoregressive (AR) baseline for day-ahead electricity consumption forecasting.

This script evaluates several forecasting approaches on French electricity
consumption data (RTE, 2012-2024), using a temporal train/test split at
2022-01-01.

Three approaches are compared:
    1. Persistence baseline : y(t+1) = y(t)
    2. RTE official forecast : provided by the French TSO (not trained here)
    3. AR model             : linear regression on a sliding window of past
                              consumption values, evaluated for window sizes
                              of 7, 14, 30, 60, 90 and 120 days.

The AR model uses a realistic rolling forecast strategy: the model is
initialized with the last window of the training set, then advances day
by day appending true observations (not predictions) to the history buffer.

Performance is measured by Mean Absolute Error (MAE) in MWh, with delta
reported against both the persistence baseline and the RTE official forecast.

Author: Éric Duhamel
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# -----------------------------------
# Utils
# -----------------------------------


def create_windowed_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def train_model(train_series, window_size):
    X_train, y_train = create_windowed_data(train_series, window_size)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# -----------------------------------
# Rolling forecast (IMPORTANT)
# -----------------------------------


def rolling_forecast(model, train_series, test_series, window_size):
    """
    Realistic forecasting:
    - initialized with the end of the training series
    - then advances day by day
    """
    history = list(train_series[-window_size:])  # last window from training set
    preds = []

    for t in range(len(test_series)):
        X_input = np.array(history[-window_size:]).reshape(1, -1)
        y_pred = model.predict(X_input)[0]
        preds.append(y_pred)

        # append the true value (not the prediction!)
        history.append(test_series[t])

    return np.array(preds)


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

train_series = df.loc[~mask_test, "consumption_MWh"].values
test_series = df.loc[mask_test, "consumption_MWh"].values

# RTE forecast
rte_test = df.loc[mask_test, "RTEprediction_MWh"].values

# -----------------------------------
# 1. Persistence baseline
# -----------------------------------

y_true_persist, y_pred_persist = persistence_forecast(test_series)
mae_persist = mean_absolute_error(y_true_persist, y_pred_persist)

print("\n=== PERSISTENCE BASELINE ===")
print(f"Persistence MAE: {mae_persist:,.2f} MWh")


# -----------------------------------
# 2. RTE (official forecast: not used in this part of the tutorial)
# -----------------------------------

# alignment (1 point dropped to match persistence)
rte_aligned = rte_test[1:]
y_true_rte = test_series[1:]

mae_rte = mean_absolute_error(y_true_rte, rte_aligned)

# -----------------------------------
# 3. AR model (linear regression)
# -----------------------------------

print("\n=== AUTOREGRESSIVE MODEL ===")

for window in [7, 14, 30, 60, 90, 120]:
    model = train_model(train_series, window)
    preds = rolling_forecast(model, train_series, test_series, window)
    # alignment with persistence
    y_true = test_series
    mae_model = mean_absolute_error(y_true, preds)
    delta_vs_persist = mae_model - mae_persist
    delta_vs_rte = mae_model - mae_rte
    print(f"\nWindow: {window} days")
    print(f"Model MAE: {mae_model:,.2f} MWh")
    print(f"Δ vs persistence: {delta_vs_persist:,.2f}")
