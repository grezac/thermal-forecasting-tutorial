"""
Electricity Consumption Forecasting - Autoregressive Model with Calendar Features

This script implements an autoregressive (AR) model enhanced with calendar features
to forecast daily electricity consumption in France (RTE data).

Main features:
    - Advanced calendar feature engineering:
        * One-hot encoding of the day of the week
        * Weekend, French public holidays, and bridge day indicators
    - Linear autoregressive model with sliding window
    - Rolling forecast evaluation (realistic out-of-sample simulation)
    - Comparison against the naive Persistence baseline
    - Performance evaluation using Mean Absolute Error (MAE)

Data:
    - Daily electricity consumption (MWh) from 2012 to 2024

The script evaluates multiple window sizes (7, 14, 30, 60, 90, 120 days)
and displays the performance of each configuration.

Author: Éric Duhamel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import holidays

# -----------------------------------
# PATH HANDLING
# -----------------------------------

script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent.parent
file_path = (
    base_dir / "tutorial" / "data" / "RTE" / "rte_daily_consumption_2012_2024.csv"
)


# -----------------------------------
# CALENDAR FEATURES
# -----------------------------------


def create_calendar_features(dates):
    # Convert to pandas datetime (critical fix)
    dates = pd.to_datetime(dates)
    years = list(set(d.year for d in dates))
    french_holidays = holidays.France(years=years)
    holiday_dates = set(french_holidays.keys())
    features = []

    for d in dates:
        dow = d.weekday()
        # One-hot encoding
        dow_oh = [int(dow == i) for i in range(7)]
        is_weekend = int(dow >= 5)
        is_holiday = int(d.date() in holiday_dates)
        # Bridge day
        is_bridge = 0
        if not is_weekend and not is_holiday:
            prev_day = (d - pd.Timedelta(days=1)).date()
            next_day = (d + pd.Timedelta(days=1)).date()
            if prev_day in holiday_dates or next_day in holiday_dates:
                is_bridge = 1
        features.append(dow_oh + [is_weekend, is_holiday, is_bridge])

    return np.array(features)


# -----------------------------------
# DATA PREPARATION
# -----------------------------------


def create_windowed_data_with_calendar(series, dates, window_size):
    X, y = [], []
    calendar_features = create_calendar_features(dates)

    for i in range(len(series) - window_size):
        # Autoregressive part
        ar_part = series[i : i + window_size]
        # Calendar part (target day t+1)
        cal_part = calendar_features[i + window_size]
        X.append(np.concatenate([ar_part, cal_part]))
        y.append(series[i + window_size])

    return np.array(X), np.array(y)


def train_model_with_calendar(series, dates, window_size):
    X, y = create_windowed_data_with_calendar(series, dates, window_size)
    model = LinearRegression()
    model.fit(X, y)
    return model


# -----------------------------------
# ROLLING FORECAST
# -----------------------------------


def rolling_forecast_with_calendar(
    model, train_series, train_dates, test_series, test_dates, window_size
):

    history = list(train_series[-window_size:])
    all_dates = list(train_dates[-window_size:]) + list(test_dates)
    preds = []

    for t in range(len(test_series)):
        idx = window_size + t
        ar_part = history[-window_size:]
        cal_part = create_calendar_features([all_dates[idx]])[0]
        X_input = np.concatenate([ar_part, cal_part]).reshape(1, -1)
        y_pred = model.predict(X_input)[0]
        preds.append(y_pred)
        # Update with true observation
        history.append(test_series[t])

    return np.array(preds)


# -----------------------------------
# BASELINES
# -----------------------------------


def persistence_forecast(series):
    return series[:-1], series[1:]


# -----------------------------------
# RESIDUALS DISPLAY
# The following function generates a residual plot for a specified period.
# This can help identify patterns in the errors, such as seasonality or
# specific periods where the model underperforms.
# Note that this function is not called in the main script,
# but you can use it to analyze model performance in more detail.
# Feel free to delete it.
# -----------------------------------


def display_residuals(start_date, end_date, window_size=14):

    import matplotlib.pyplot as plt

    # Recompute predictions with chosen window
    model = train_model_with_calendar(
        train_series,
        train_dates,
        window_size=window_size,
    )

    preds = rolling_forecast_with_calendar(
        model,
        train_series,
        train_dates,
        test_series,
        test_dates,
        window_size=window_size,
    )

    # Residuals
    residuals = test_series - preds

    # Build dataframe for easy filtering
    residuals_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "residual": residuals,
        }
    )

    # Period filtering
    mask = (residuals_df["date"] >= pd.to_datetime(start_date)) & (
        residuals_df["date"] <= pd.to_datetime(end_date)
    )

    residuals_df = residuals_df.loc[mask]

    if residuals_df.empty:
        print("No data available for this period.")
        return

        # Dynamic figure width
    n_days = len(residuals_df)

    # Width adapts to period but remains reasonable
    fig_width = min(max(10, n_days / 25), 22)

    fig, ax = plt.subplots(
        figsize=(fig_width, 5),
        constrained_layout=True,
    )

    ax.plot(
        residuals_df["date"],
        residuals_df["residual"],
        linewidth=1,
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(f"Forecast residuals from {start_date} to {end_date}")

    ax.set_ylabel("Residual (MWh)")

    ax.grid(True, alpha=0.3)

    # Prevent left tick labels from being truncated
    fig.subplots_adjust(left=0.10)

    plt.show()


# -----------------------------------
# MAIN SCRIPT
# -----------------------------------

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"])

# Train / Test split
mask_test = df["date"] >= "2022-01-01"

train_series = df.loc[~mask_test, "consumption_MWh"].values
test_series = df.loc[mask_test, "consumption_MWh"].values

train_dates = df.loc[~mask_test, "date"].values
test_dates = df.loc[mask_test, "date"].values


# -----------------------------------
# PERSISTENCE
# -----------------------------------

y_true_persist, y_pred_persist = persistence_forecast(test_series)
mae_persist = mean_absolute_error(y_true_persist, y_pred_persist)

print("\n=== PERSISTENCE BASELINE ===")
print(f"MAE persistence: {mae_persist:,.2f} MWh")


# -----------------------------------
# AUTOREGRESSIVE + CALENDAR MODEL
# -----------------------------------

print("\n=== AR + CALENDAR MODEL ===")

for window in [7, 14, 30, 60, 90, 120]:
    model = train_model_with_calendar(train_series, train_dates, window_size=window)
    preds = rolling_forecast_with_calendar(
        model, train_series, train_dates, test_series, test_dates, window_size=window
    )
    y_true = test_series
    mae_model = mean_absolute_error(y_true, preds)
    delta_vs_persist = mae_model - mae_persist
    print(f"\nWindow: {window} days")
    print(f"Model MAE: {mae_model:,.2f} MWh")
    print(f"Δ vs persistence: {delta_vs_persist:,.2f}")

    # if window == 14:
    #     display_residuals("2022-01-01", "2024-12-31")
