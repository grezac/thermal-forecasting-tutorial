"""
RTE Data Processing - Daily Consumption and Predictions

This script processes raw RTE eCO2mix files (.xls) from 2012 to 2024.

It performs the following steps:
    - Reads annual files containing 15-minute granularity data
    - Extracts actual electricity consumption and RTE day-ahead predictions (J-1)
    - Aggregates data to daily level (MWh and average MW)
    - Handles data quality (counts number of points per day)
    - Saves the cleaned dataset as rte_daily_consumption_2012_2024.csv

Output columns:
    - date
    - consumption_MWh
    - RTEprediction_MWh
    - n_points, consumption_mean_MW, RTEprediction_mean_MW, is_incomplete

Author: Éric Duhamel
"""

import pandas as pd
from pathlib import Path

script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
INPUT_DIR = base_dir / "data" / "RTE"

FILES = [
    INPUT_DIR / f"eCO2mix_RTE_Annuel-Definitif_{year}.xls" for year in range(2012, 2025)
]

OUTPUT_CSV = INPUT_DIR / "rte_daily_consumption_2012_2024.csv"


# ============================================================
# PARSING
# ============================================================


def load_file(path):

    print("Loading", path)

    rows = []

    with open(path, encoding="latin1") as f:

        # header
        next(f)

        for line in f:
            tokens = line.split()
            if len(tokens) < 7:
                continue

            try:

                date_str = tokens[3]
                time_str = tokens[4]

                consumption = float(tokens[5])

                # J-1
                prediction = float(tokens[6])

                rows.append(
                    {
                        "datetime": pd.Timestamp(f"{date_str} {time_str}"),
                        "consumption": consumption,
                        "prediction": prediction,
                    }
                )

            except ValueError:

                continue

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

dfs = []

for path in FILES:
    dfs.append(load_file(path))

df = pd.concat(dfs, ignore_index=True)

print()
print("Raw rows:", len(df))


# ============================================================
# DAILY
# ============================================================

df["date"] = df["datetime"].dt.date


# quarter of an hour (96 data points per day)
FACTOR = 0.25


daily = (
    df.groupby("date")
    .agg(
        consumption_MWh=("consumption", lambda x: x.sum() * FACTOR),
        RTEprediction_MWh=("prediction", lambda x: x.sum() * FACTOR),
        n_points=("consumption", "count"),
        consumption_mean_MW=("consumption", "mean"),
        RTEprediction_mean_MW=("prediction", "mean"),
    )
    .reset_index()
)

daily["is_incomplete"] = daily["n_points"] != 96


daily.to_csv(OUTPUT_CSV, index=False)


print()
print(daily.head())
