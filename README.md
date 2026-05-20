# Thermal Forecasting Tutorial Scripts

Python scripts associated with the Sentinel Forecasting Lab tutorial series on time series forecasting, daily electricity consumption prediction, and meteorological data analysis.

This repository contains the scripts used throughout the tutorial to build and evaluate progressively more sophisticated forecasting models, ranging from simple persistence baselines to autoregressive models using meteorological variables and calendar features.

---

# Repository Structure

```text
[root project directory]

├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
└── scripts
    ├── AR.py
    ├── AR_with_calendar.py
    ├── persistence.py
    ├── rte_with_preds.py
    └── with_meteo.py
```

---

# Dataset

The datasets required to run the scripts are available on Zenodo:

https://doi.org/10.5281/zenodo.20306136

The dataset repository contains:

- aggregated SQLite database
- ERA5-derived meteorological variables
- French electricity consumption data
- database reconstruction scripts

---

# Installation

Clone the repository:

```bash
git clone https://github.com/grezac/thermal-forecasting-tutorial
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

---

# Dependencies

Main Python dependencies:

- numpy
- pandas
- matplotlib
- scikit-learn
- holidays

---

# Running the Scripts

From the project root directory, scripts can be executed with:

```bash
python scripts/[script_name].py
```

Example:

```bash
python scripts/AR.py
```

---

# Script Overview

## persistence.py

Baseline persistence forecasting model.

---

## AR.py

Simple autoregressive forecasting model.

---

## AR_with_calendar.py

Autoregressive model including calendar-based features.

---

## with_meteo.py

Forecasting model using meteorological variables derived from ERA5 datasets.

---

## rte_with_preds.py

Visualization and comparison utilities for predictions and observed values.

---

# Educational Purpose

This repository is intended for:

- educational tutorials
- reproducible forecasting experiments
- introductory machine learning workflows
- time series analysis demonstrations

The code prioritizes readability and pedagogical clarity over optimization.

---

# License

This repository is distributed under the MIT License.

See the `LICENSE` file for details.

---

# Author

Éric Duhamel  
Sentinel Forecasting Lab  
France

Tutorial:  
https://sentinel-forecasting.com/RTE_tutorial/
