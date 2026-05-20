"""
Electricity Consumption Forecasting - AR + Calendar + Meteorological Features

This script implements a two-stage forecasting model for daily electricity
consumption in France (RTE data):

1. First stage: Autoregressive model + Calendar features
   - Sliding window autoregression
   - One-hot day of week, weekend, French holidays and bridge days

2. Second stage: Residual modeling with meteorological data
   - Loads temperature, HDD (Heating Degree Days) and CDD (Cooling Degree Days)
   - Uses spatial aggregates (mean, min, max) and quadratic terms
   - Applies lagged features (window of 14 days) to predict residuals

Main capabilities:
    - Temporal train/test split (test from 2022 onward)
    - Rolling-style evaluation while preserving temporal order
    - Linear regression on meteorological features
    - Feature importance analysis (coefficients by lag)
    - Optional visualization of predictions and error distribution

Data sources:
    - RTE daily consumption (CSV)
    - Meteorological aggregates (SQLite database)

Author: Éric Duhamel
"""

from pathlib import Path
import sqlite3
import pickle
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays

script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent

rte_path = base_dir / "data" / "RTE" / "rte_daily_consumption_2012_2024.csv"
db_path = base_dir / "data" / "thermal.db"

WINDOW_SIZE = 14


# ================================
# CALENDAR
# ================================


def create_calendar_features(dates):

    dates = pd.to_datetime(dates)

    years = list(set(d.year for d in dates))

    french_holidays = holidays.France(years=years)

    holiday_dates = set(french_holidays.keys())

    features = []

    for d in dates:

        dow = d.weekday()

        # One-hot encoding of day of week
        dow_oh = [int(dow == i) for i in range(7)]

        is_weekend = int(dow >= 5)

        is_holiday = int(d.date() in holiday_dates)

        # Bridge day
        is_bridge = 0

        if not is_weekend and not is_holiday:

            if ((d - pd.Timedelta(days=1)).date() in holiday_dates) or (
                (d + pd.Timedelta(days=1)).date() in holiday_dates
            ):

                is_bridge = 1

        features.append(
            dow_oh
            + [
                is_weekend,
                is_holiday,
                is_bridge,
            ]
        )

    return np.array(features)


# ================================
# AR + CALENDAR DATASET
# ================================


def build_ar_cal(series, dates, window):

    cal = create_calendar_features(dates)

    X = []
    y = []

    for i in range(len(series) - window):

        ar = series[i : i + window]

        cal_part = cal[i + window]

        X.append(np.concatenate([ar, cal_part]))

        y.append(series[i + window])

    return np.array(X), np.array(y)


def get_ar_calendar_residuals(
    full_series,
    full_dates,
    n_train,
    window,
):
    """
    Compute AR+calendar residuals without breaking
    temporal context.

    Important:
    AR windows are built on the full series first,
    then split according to target dates.

    Result:
        len(test_pred) == len(raw_test)

    Example:
        raw train = 3653
        raw test  = 1096
        window    = 14

        train residuals = 3639
        test residuals  = 1096
    """

    # ================================
    # BUILD WINDOWED DATASET
    # ================================

    X_full, y_full = build_ar_cal(
        full_series,
        full_dates,
        window,
    )

    # ================================
    # SPLIT
    # ================================

    split_index = n_train - window

    X_train = X_full[:split_index]
    y_train = y_full[:split_index]

    X_test = X_full[split_index:]
    y_test = y_full[split_index:]

    # print()
    # print("AR+calendar split:")
    # print(f"split index: {split_index}")
    # print(f"train samples: {len(y_train)}")
    # print(f"test samples: {len(y_test)}")

    # ================================
    # FIT
    # ================================

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    # ================================
    # PREDICT
    # ================================

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    # ================================
    # RESIDUALS
    # ================================

    train_residuals = y_train - train_pred

    test_residuals = y_test - test_pred

    # ================================
    # CHECKS
    # ================================

    expected_train_len = n_train - window

    expected_test_len = len(full_series) - n_train

    # print()
    # print("Alignment check:")
    # print(f"expected train length: {expected_train_len}")
    # print(f"train residuals: {len(train_residuals)}")

    # print(f"expected test length: {expected_test_len}")
    # print(f"test target: {len(y_test)}")
    # print(f"test prediction: {len(test_pred)}")
    # print(f"test residuals: {len(test_residuals)}")

    assert len(train_residuals) == expected_train_len
    assert len(test_residuals) == expected_test_len
    assert len(test_pred) == expected_test_len

    return (
        train_residuals,
        test_residuals,
        train_pred,
        test_pred,
    )


def add_meteo_features(
    train_residuals, test_residuals, base_pred_test, test_dates, real_test_series
):
    """
    Fit a linear regression model on meteorological features to predict
    AR+calendar residuals, and return final electricity consumption forecasts.

    Meteorological features are loaded from a SQLite database and include
    spatial aggregates (mean, min, max) of daily temperature, heating degree
    days (HDD), cooling degree days (CDD), and their squared versions.
    A lagged dataset is built using a sliding window of size WINDOW_SIZE.

    Parameters
    ----------
    train_residuals : np.ndarray
        Residuals from the AR+calendar model on the training set.
    test_residuals : np.ndarray
        Residuals from the AR+calendar model on the test set.
    base_pred_test : np.ndarray
        AR+calendar predictions on the test set (used to reconstruct
        final consumption forecasts).
    test_dates : np.ndarray
        Dates corresponding to the test set.

    Returns
    -------
    dict with keys:
        - "dates"          : aligned test dates (after windowing)
        - "real"           : actual electricity consumption on the test set
        - "base_pred"      : AR+calendar predictions on the test set
        - "residual_real"  : actual residuals on the test set
        - "residual_pred"  : predicted residuals on the test set
        - "final_pred"     : final consumption forecasts (base + residual)
    """

    # ==========================================
    # BASELINE: thermal aggregates + squares
    # ==========================================

    # print()
    # print("================================")
    # print("BASELINE: thermal + quadratic")
    # print("================================")

    # ------------------------------------------
    # Residual target
    # ------------------------------------------

    residual_series = np.concatenate(
        [
            train_residuals,
            test_residuals,
        ]
    )

    # print(f"residual_series shape: {residual_series.shape}")

    # ------------------------------------------
    # Helper
    # ------------------------------------------

    def load_variable(var_name):

        conn = sqlite3.connect(db_path)

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT series
            FROM feature_series
            WHERE var_name = ?
            ORDER BY id
            """,
            (var_name,),
        )

        rows = cursor.fetchall()

        cursor.close()

        conn.close()

        all_series = []

        for (series_blob,) in rows:

            s = pickle.loads(series_blob)

            all_series.append(s)

        return np.array(
            all_series,
            dtype=np.float32,
        )

    # ------------------------------------------
    # Load variables
    # ------------------------------------------

    T = load_variable("2m_temperature_daily_mean")

    HDD = load_variable("heating_degree_days")

    CDD = load_variable("cooling_degree_days")

    HDD2 = load_variable("heating_degree_days_squared")

    CDD2 = load_variable("cooling_degree_days_squared")

    # print(f"Tmean series: {len(T)}")
    # print(f"HDD series: {len(HDD)}")
    # print(f"CDD series: {len(CDD)}")

    # ------------------------------------------
    # Spatial aggregates
    # ------------------------------------------

    T_mean = np.mean(
        T,
        axis=0,
    )

    T_min = np.min(
        T,
        axis=0,
    )

    T_max = np.max(
        T,
        axis=0,
    )

    HDD_mean = np.mean(
        HDD,
        axis=0,
    )

    CDD_mean = np.mean(
        CDD,
        axis=0,
    )

    HDD2_mean = np.mean(
        HDD2,
        axis=0,
    )

    CDD2_mean = np.mean(
        CDD2,
        axis=0,
    )

    # ------------------------------------------
    # Shape = (n_days, 7)
    # ------------------------------------------

    X_raw = np.column_stack(
        [
            T_mean,
            T_min,
            T_max,
            HDD_mean,
            CDD_mean,
            HDD2_mean,
            CDD2_mean,
        ]
    )

    # ------------------------------------------
    # Align with residuals
    # ------------------------------------------

    X_raw = X_raw[WINDOW_SIZE:]  # Supprimé !!!!!!!!!!! NON !!!

    # print(f"X_raw shape: {X_raw.shape}")

    assert len(X_raw) == len(residual_series)

    # ------------------------------------------
    # Build lagged dataset
    # ------------------------------------------

    X = []

    y = []

    for i in range(len(residual_series) - WINDOW_SIZE):

        block = X_raw[i : i + WINDOW_SIZE]

        X.append(block.flatten())

        y.append(residual_series[i + WINDOW_SIZE])

    X = np.array(
        X,
        dtype=np.float32,
    )

    y = np.array(
        y,
        dtype=np.float32,
    )

    # print(f"X shape: {X.shape}")
    # print(f"y shape: {y.shape}")

    # ------------------------------------------
    # Train / test split
    # ------------------------------------------

    split_idx = len(train_residuals) - WINDOW_SIZE

    X_train = X[:split_idx]

    y_train = y[:split_idx]

    X_test = X[split_idx:]

    y_test = y[split_idx:]

    # print(f"train samples: {len(X_train)}")
    # print(f"test samples : {len(X_test)}")

    # ==========================================
    # MODEL
    # In this tutorial, we use LinearRegression
    # You can also test other regression models :
    # "Ridge": Ridge(alpha=10.0),
    # "Lasso": Lasso(
    #     alpha=0.1,
    #     max_iter=10000,
    #     random_state=42,
    # ),
    # "ElasticNet": ElasticNet(
    #     alpha=1.0,
    #     l1_ratio=0.01,
    #     max_iter=50000,
    #     random_state=42,
    # ),
    # or you can try boosting models, like XGBoost or LightGBM but these
    # models underperform compared to the previous regression models
    # ==========================================

    model = LinearRegression()

    # ==========================================
    # TRAIN
    # ==========================================

    print()
    print("----------------------------------------------------")
    print("LinearRegression (Residuals Modeling)")
    print("----------------------------------------------------")

    model.fit(
        X_train,
        y_train,
    )

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    mae_train = mean_absolute_error(
        y_train,
        pred_train,
    )

    mae_test = mean_absolute_error(
        y_test,
        pred_test,
    )

    print(f"train MAE : {mae_train:.2f}")
    print(f"test MAE  : {mae_test:.2f}")

    # ==========================================
    # FEATURE ANALYSIS
    # ==========================================

    feature_names = [
        "T_mean",
        "T_min",
        "T_max",
        "HDD_mean",
        "CDD_mean",
        "HDD2_mean",
        "CDD2_mean",
    ]

    coef_matrix = model.coef_.reshape(
        WINDOW_SIZE,
        len(feature_names),
    )

    # print()
    # print("================================")
    # print("FEATURE ANALYSIS")
    # print("================================")

    for lag in range(WINDOW_SIZE):

        # print()
        # print(f"Lag J-{WINDOW_SIZE - lag}")

        for j, feature_name in enumerate(feature_names):

            coef = coef_matrix[
                lag,
                j,
            ]

            # print(f"{feature_name:15s} : " f"{coef:12.4f}")

    print()
    print("================================")
    print("GLOBAL FEATURE IMPORTANCE")
    print("================================")

    importance = np.mean(
        np.abs(coef_matrix),
        axis=0,
    )

    for feature_name, imp in sorted(
        zip(
            feature_names,
            importance,
        ),
        key=lambda x: -x[1],
    ):

        print(f"{feature_name:15s} : " f"{imp:12.4f}")

    # ==========================================
    # FINAL PREDICTIONS
    # ==========================================

    # IMPORTANT:
    # pred_test and y_test already have the correct
    # temporal alignment with base_pred_test.
    # No additional slicing must be applied here.

    final_pred_test = base_pred_test + pred_test

    # Use the TRUE real consumption values,
    # not a reconstruction from residuals.
    real_test = real_test_series

    # print()
    # print("DEBUG")
    # print("======")

    # print(f"mean abs(pred_test)                 : {np.mean(np.abs(pred_test)):.2f}")
    # print(f"mean abs(y_test)                    : {np.mean(np.abs(y_test)):.2f}")
    # print(
    #     f"mean abs(base_pred_test)            : {np.mean(np.abs(base_pred_test)):.2f}"
    # )

    # print(
    #     f"mean abs(final_pred - base_pred)    : "
    #     f"{np.mean(np.abs(final_pred_test - base_pred_test)):.2f}"
    # )

    # print(
    #     f"mean abs(real - base_pred)          : "
    #     f"{np.mean(np.abs(real_test - base_pred_test)):.2f}"
    # )

    # print(
    #     f"mean abs(real - final_pred)         : "
    #     f"{np.mean(np.abs(real_test - final_pred_test)):.2f}"
    # )

    # ------------------------------------------
    # Align dates
    # ------------------------------------------

    aligned_dates = test_dates

    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return {
        "dates": aligned_dates,
        "real": real_test,
        "base_pred": base_pred_test,
        "residual_real": y_test,
        "residual_pred": pred_test,
        "final_pred": final_pred_test,
    }


# Optional function to plot predictions and errors for a specific period
def plot_predictions_and_errors(
    dates_test,
    real,
    base_pred,
    final_pred,
    # start_idx=180,
    start_idx=17,
    n_days=28,
):

    # ==========================================
    # SUBSET
    # ==========================================

    s = start_idx
    e = start_idx + n_days

    dates = dates_test[s:e]

    start_str = pd.to_datetime(dates[0]).strftime("%Y-%m-%d")

    end_str = pd.to_datetime(dates[-1]).strftime("%Y-%m-%d")

    real = real[s:e]

    for i in range(10):
        print(f"{dates[i]}, {real[i]}")

    base_pred = base_pred[s:e]
    final_pred = final_pred[s:e]

    # ==========================================
    # MAE
    # ==========================================

    mae_base = np.mean(np.abs(real - base_pred))

    mae_final = np.mean(np.abs(real - final_pred))

    improvement = 100 * (mae_base - mae_final) / mae_base

    # ==========================================
    # FIGURE 1
    # ==========================================

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        dates,
        real,
        linestyle="--",
        linewidth=3,
        label="Real",
    )

    ax.plot(
        dates,
        base_pred,
        label=f"AR + calendar (MAE={mae_base:,.0f} MWh)",
    )

    ax.plot(
        dates,
        final_pred,
        label=f"Prediction (MAE={mae_final:,.0f} MWh)",
    )

    # ax.set_title(f"Predictions vs Real " f"({dates[0]} -> {dates[-1]})")
    ax.set_title(f"Predictions vs Real " f"({start_str} -> {end_str})")

    ax.grid(True)

    ax.legend()

    plt.tight_layout()

    plt.show()

    # ==========================================
    # ERRORS
    # ==========================================

    err_base = np.abs(real - base_pred)

    err_final = np.abs(real - final_pred)

    p95_base = np.percentile(
        err_base,
        95,
    )

    p95_final = np.percentile(
        err_final,
        95,
    )

    # ==========================================
    # FIGURE 2
    # ==========================================

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(
        [
            err_base,
            err_final,
        ],
        tick_labels=[
            ("AR + calendar\n" f"MAE={mae_base:,.0f}\n" f"P95={p95_base:,.0f}"),
            ("Prediction\n" f"MAE={mae_final:,.0f}\n" f"P95={p95_final:,.0f}"),
        ],
    )

    ax.set_ylabel("Absolute error (MWh)")

    ax.set_title(
        "Error distribution\n"
        f"({start_str} -> {end_str})\n"
        f"Improvement: {improvement:.2f}%"
    )

    ax.grid(True)

    plt.tight_layout()

    plt.show()


def main():

    # Load REAL electricity consumption data from CSV and split into train/test sets.
    # Data before 2022-01-01 is used for training, from 2022-01-01 onward for testing.
    # The full series is also kept for autoregressive windowing.
    print("Loading consumption data...")

    df = pd.read_csv(rte_path)

    df["date"] = pd.to_datetime(df["date"])

    mask_test = df["date"] >= "2022-01-01"

    train_series = df.loc[~mask_test, "consumption_MWh"].values
    test_series = df.loc[mask_test, "consumption_MWh"].values

    # Full series for AR windowing
    full_series = df["consumption_MWh"].values
    full_dates = df["date"].values

    n_train = len(train_series)

    print(f"Training days: {n_train}")
    print(f"Test days: {len(test_series)}")
    print(f"Window size: {WINDOW_SIZE}")

    # Fit an AR+calendar model on the full series and compute residuals.
    # AR windows are built before the train/test split to preserve temporal
    # context, ensuring len(test_residuals) == len(raw_test).
    (
        train_residuals,
        test_residuals,
        ar_train_pred,
        ar_test_pred,
    ) = get_ar_calendar_residuals(
        full_series,
        full_dates,
        n_train,
        WINDOW_SIZE,
    )

    # ==========================================
    # Align test dates with AR predictions
    # ==========================================

    dates_ar = full_dates[WINDOW_SIZE:]
    split_idx = n_train - WINDOW_SIZE
    test_dates = dates_ar[split_idx:]

    # real_test_series = test_series[WINDOW_SIZE:]
    real_test_series = test_series

    # ==========================================
    # Add meteorological correction
    # ==========================================

    results = add_meteo_features(
        train_residuals=train_residuals,
        test_residuals=test_residuals,
        base_pred_test=ar_test_pred,
        test_dates=test_dates,
        real_test_series=real_test_series,
    )

    # ==========================================
    # FINAL EVALUATION ON REAL CONSUMPTION
    # ==========================================
    #
    # As a final consistency check, we compare the
    # reconstructed forecasts of daily electricity
    # consumption against the true observed RTE values.
    #
    # This evaluates the real forecasting performance
    # of the complete AR + meteorological pipeline.
    # ==========================================

    mae_ar = mean_absolute_error(
        results["real"],
        results["base_pred"],
    )

    mae_final = mean_absolute_error(
        results["real"],
        results["final_pred"],
    )

    improvement = 100 * (mae_ar - mae_final) / mae_ar

    print()
    print("=======================================================")
    print("FINAL CONSUMPTION FORECAST PERFORMANCE ON TEST DATASET")
    print("=======================================================")

    print(f"AR + calendar MAE : {mae_ar:,.2f} MWh")
    print(f"Final model MAE   : {mae_final:,.2f} MWh")
    print(f"Improvement       : {improvement:.2f}%")

    # Optionally, plot predictions and errors for a specific period
    # plot_predictions_and_errors(
    #     dates_test=results["dates"],
    #     real=results["real"],
    #     base_pred=results["base_pred"],
    #     final_pred=results["final_pred"],
    # )


if __name__ == "__main__":
    main()
