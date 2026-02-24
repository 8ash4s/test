"""
Macro Forecasting — FRED Data + ARIMA & VAR Models
Author: Matthew Bowers

Setup:
    1. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
    2. Set it as an environment variable:
           export FRED_API_KEY="your_key_here"
    3. Install dependencies:
           pip install -r requirements.txt

Usage:
    python macro_forecasting.py                  # ARIMA forecasts
    python macro_forecasting.py --model var      # VAR forecast (monthly series)
    python macro_forecasting.py --model both     # Run both
    python macro_forecasting.py --periods 24     # Forecast horizon (months)
"""

import os
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fredapi import Fred
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# FRED series to pull
#   Key   : label used in code / plots
#   Value : FRED series ID
# ---------------------------------------------------------------------------
MONTHLY_SERIES = {
    "CPI":          "CPIAUCSL",   # CPI All Urban Consumers (index)
    "Unemployment": "UNRATE",     # Unemployment Rate (%)
    "Fed Funds":    "FEDFUNDS",   # Effective Federal Funds Rate (%)
}

QUARTERLY_SERIES = {
    "GDP":          "GDPC1",      # Real GDP (billions of chained 2017 dollars)
}

# Default ARIMA orders — (p, d, q)
ARIMA_ORDERS = {
    "CPI":          (2, 1, 1),
    "Unemployment": (2, 1, 1),
    "Fed Funds":    (1, 1, 1),
    "GDP":          (1, 1, 0),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_fred_client():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY environment variable not set.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then run:  export FRED_API_KEY='your_key_here'"
        )
    return Fred(api_key=key)


def fetch_data(fred, start="2000-01-01"):
    """Pull all series from FRED and return (monthly_df, quarterly_df)."""
    print("Fetching data from FRED...")

    monthly = pd.DataFrame({
        name: fred.get_series(sid, observation_start=start)
        for name, sid in MONTHLY_SERIES.items()
    }).dropna()
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M")

    quarterly = pd.DataFrame({
        name: fred.get_series(sid, observation_start=start)
        for name, sid in QUARTERLY_SERIES.items()
    }).dropna()
    quarterly.index = pd.DatetimeIndex(quarterly.index).to_period("Q")

    print(f"  Monthly  : {monthly.index[0]} → {monthly.index[-1]}  ({len(monthly)} obs)")
    print(f"  Quarterly: {quarterly.index[0]} → {quarterly.index[-1]}  ({len(quarterly)} obs)")
    return monthly, quarterly


# ---------------------------------------------------------------------------
# Stationarity helper
# ---------------------------------------------------------------------------

def is_stationary(series, sig=0.05):
    """Return True if ADF test rejects unit root at given significance level."""
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < sig


def make_stationary(series):
    """
    Difference the series until stationary (max 2 differences).
    Returns (stationary_series, n_diffs).
    """
    s, d = series.copy(), 0
    while not is_stationary(s) and d < 2:
        s = s.diff().dropna()
        d += 1
    return s, d


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def run_arima(series, name, order, periods=12):
    """
    Fit ARIMA model and return a forecast DataFrame with confidence intervals.
    """
    model = ARIMA(series, order=order)
    fit = model.fit()
    forecast = fit.get_forecast(steps=periods)
    mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)

    # Build a clean period index for the forecast
    last = series.index[-1]
    freq = series.index.freq
    future_idx = pd.period_range(start=last + 1, periods=periods, freq=freq)

    result = pd.DataFrame({
        "forecast": mean.values,
        "lower_95": ci.iloc[:, 0].values,
        "upper_95": ci.iloc[:, 1].values,
    }, index=future_idx)

    print(f"  {name:15s} ARIMA{order}  AIC={fit.aic:.1f}  "
          f"forecast[+1]={result['forecast'].iloc[0]:.3f}")
    return fit, result


def plot_arima_forecasts(history_dict, forecast_dict, periods, output_dir):
    """Plot each series with its ARIMA forecast on a 2×2 grid."""
    names = list(history_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"ARIMA Forecasts — {periods}-Period Horizon", fontsize=14, fontweight="bold")

    for ax, name in zip(axes.flat, names):
        hist = history_dict[name]
        fc = forecast_dict[name]

        hist_dt = hist.index.to_timestamp()
        fc_dt = fc.index.to_timestamp()

        ax.plot(hist_dt, hist.values, color="steelblue", linewidth=1.2, label="Historical")
        ax.plot(fc_dt, fc["forecast"], color="tomato", linewidth=1.5, linestyle="--", label="Forecast")
        ax.fill_between(fc_dt, fc["lower_95"], fc["upper_95"],
                        color="tomato", alpha=0.15, label="95% CI")
        ax.axvline(hist_dt[-1], color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(name, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "arima_forecasts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------

def run_var(monthly_df, periods=12):
    """
    Fit a VAR model on the monthly series (CPI, Unemployment, Fed Funds).
    Differences non-stationary series, fits VAR, then undoes differencing
    to return level forecasts.
    """
    print("\nFitting VAR model...")

    # Difference as needed and track how many diffs per column
    stationary_parts = {}
    n_diffs = {}
    for col in monthly_df.columns:
        s, d = make_stationary(monthly_df[col])
        stationary_parts[col] = s
        n_diffs[col] = d
        print(f"  {col:15s} differenced {d}×  (now stationary)")

    # Align on common index after differencing
    stat_df = pd.DataFrame(stationary_parts).dropna()

    # Select lag order (up to 12, BIC criterion)
    var_model = VAR(stat_df)
    lag_order = var_model.select_order(maxlags=12)
    best_lag = lag_order.selected_orders.get("bic", 2)
    print(f"  VAR lag order (BIC): {best_lag}")

    fit = var_model.fit(best_lag)
    raw_fc = fit.forecast(stat_df.values[-best_lag:], steps=periods)
    fc_df = pd.DataFrame(raw_fc, columns=monthly_df.columns)

    # Undo differencing to recover level forecasts
    level_fc = {}
    for col in monthly_df.columns:
        last_vals = monthly_df[col].dropna()
        fc_vals = fc_df[col].values
        for _ in range(n_diffs[col]):
            fc_vals = np.cumsum(np.insert(fc_vals, 0, last_vals.iloc[-1]))[1:]
        level_fc[col] = fc_vals

    last = monthly_df.index[-1]
    future_idx = pd.period_range(start=last + 1, periods=periods, freq="M")
    result = pd.DataFrame(level_fc, index=future_idx)
    return fit, result


def plot_var_forecast(monthly_df, var_fc, output_dir):
    """Plot VAR level forecasts against history for each monthly series."""
    cols = monthly_df.columns.tolist()
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 4))
    fig.suptitle("VAR Model Forecasts (Monthly Series)", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, cols):
        hist_dt = monthly_df.index.to_timestamp()
        fc_dt = var_fc.index.to_timestamp()
        ax.plot(hist_dt[-60:], monthly_df[col].values[-60:],
                color="steelblue", linewidth=1.2, label="Historical (5 yr)")
        ax.plot(fc_dt, var_fc[col].values,
                color="darkorange", linewidth=1.5, linestyle="--", label="VAR Forecast")
        ax.axvline(hist_dt[-1], color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(col, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "var_forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Macro forecasting with FRED data.")
    p.add_argument("--model", choices=["arima", "var", "both"], default="arima",
                   help="Which model(s) to run (default: arima)")
    p.add_argument("--periods", type=int, default=12,
                   help="Forecast horizon in periods (default: 12)")
    p.add_argument("--start", default="2000-01-01",
                   help="Data start date (default: 2000-01-01)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fred = get_fred_client()
    monthly_df, quarterly_df = fetch_data(fred, start=args.start)

    # --- ARIMA ---
    if args.model in ("arima", "both"):
        print(f"\nFitting ARIMA models ({args.periods}-period horizon)...")
        all_series = {**{c: monthly_df[c] for c in monthly_df.columns},
                      **{c: quarterly_df[c] for c in quarterly_df.columns}}

        arima_forecasts = {}
        for name, series in all_series.items():
            _, fc = run_arima(series, name, ARIMA_ORDERS[name], periods=args.periods)
            arima_forecasts[name] = fc

        # Save ARIMA forecasts to CSV
        rows = []
        for name, fc in arima_forecasts.items():
            for period, row in fc.iterrows():
                rows.append({"series": name, "period": str(period),
                              "forecast": round(row["forecast"], 4),
                              "lower_95": round(row["lower_95"], 4),
                              "upper_95": round(row["upper_95"], 4)})
        pd.DataFrame(rows).to_csv(output_dir / "arima_forecasts.csv", index=False)

        plot_arima_forecasts(all_series, arima_forecasts, args.periods, output_dir)

    # --- VAR ---
    if args.model in ("var", "both"):
        _, var_fc = run_var(monthly_df, periods=args.periods)
        plot_var_forecast(monthly_df, var_fc, output_dir)
        var_fc.index = var_fc.index.astype(str)
        var_fc.to_csv(output_dir / "var_forecast.csv")
        print(f"  Saved → {output_dir / 'var_forecast.csv'}")

    print("\nDone.")
