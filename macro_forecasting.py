"""
Macro Forecasting — FRED Data + ARIMA & VAR Models
Author: Matthew Bowers

Setup:
    1. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
    2. Set it as an environment variable:
           export FRED_API_KEY="your_key_here"
       OR the key below will be used as fallback.

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
# FRED API key — env var takes priority, hardcoded key is fallback
# ---------------------------------------------------------------------------
_FALLBACK_FRED_KEY = "12174540861647b9e4dde323304478d6"

# ---------------------------------------------------------------------------
# FRED series
# ---------------------------------------------------------------------------
MONTHLY_SERIES = {
    "CPI":          "CPIAUCSL",
    "Unemployment": "UNRATE",
    "Fed Funds":    "FEDFUNDS",
}

QUARTERLY_SERIES = {
    "GDP": "GDPC1",
}

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
    key = os.environ.get("FRED_API_KEY") or _FALLBACK_FRED_KEY
    if not key:
        raise EnvironmentError(
            "No FRED API key found. Set FRED_API_KEY env var or add your key to "
            "_FALLBACK_FRED_KEY in the script.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
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
# Stationarity
# ---------------------------------------------------------------------------

def is_stationary(series, sig=0.05):
    """ADF test — returns True if unit root is rejected."""
    result = adfuller(np.asarray(series.dropna()), autolag="AIC")
    return result[1] < sig


def make_stationary(series, max_diffs=2):
    """
    Difference the series conservatively.
    CPI (CPIAUCSL) is known I(1), so force exactly one diff. All other
    series are differenced adaptively up to `max_diffs`.
    Returns (stationary_series, n_diffs).
    """
    if series.name == "CPI":
        # CPI is textbook I(1) — avoid over-differencing.
        return series.diff().dropna(), 1

    s, d = series.copy(), 0
    while not is_stationary(s) and d < max_diffs:
        s = s.diff().dropna()
        d += 1
    return s, d


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def _future_period_index(last_period, periods):
    """
    Build a future PeriodIndex that works for both monthly ('M') and
    quarterly ('Q') series — avoids relying on PeriodIndex.freq.freqstr.
    """
    freq = last_period.freqstr[0]   # 'M' or 'Q'
    return pd.period_range(start=last_period + 1, periods=periods, freq=freq)


def run_arima(series, name, order, periods=12):
    """Fit ARIMA and return (fit, forecast_df) with 95 % CI."""
    model = ARIMA(series, order=order)
    fit   = model.fit()
    fc    = fit.get_forecast(steps=periods)
    mean  = fc.predicted_mean
    ci    = fc.conf_int(alpha=0.05)

    future_idx = _future_period_index(series.index[-1], periods)

    result = pd.DataFrame({
        "forecast":  mean.values,
        "lower_95":  ci.iloc[:, 0].values,
        "upper_95":  ci.iloc[:, 1].values,
    }, index=future_idx)

    print(f"  {name:15s} ARIMA{order}  AIC={fit.aic:.1f}  "
          f"forecast[+1]={result['forecast'].iloc[0]:.3f}")
    return fit, result


def _plot_single_arima_grid(history_dict, forecast_dict, periods, title, axes):
    """Shared plotting logic for any dict of series + forecasts."""
    for ax, name in zip(axes, history_dict.keys()):
        hist = history_dict[name]
        fc   = forecast_dict[name]
        hist_dt = hist.index.to_timestamp()
        fc_dt   = fc.index.to_timestamp()

        ax.plot(hist_dt, hist.values, color="steelblue", lw=1.2, label="Historical")
        ax.plot(fc_dt, fc["forecast"], color="tomato", lw=1.5,
                linestyle="--", label="Forecast")
        ax.fill_between(fc_dt, fc["lower_95"], fc["upper_95"],
                        color="tomato", alpha=0.15, label="95% CI")
        ax.axvline(hist_dt[-1], color="gray", linestyle=":", lw=0.8)
        ax.set_title(name, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)


def plot_arima_forecasts(monthly_hist, monthly_fc, quarterly_hist, quarterly_fc,
                         periods, output_dir):
    """2×2 grid for monthly + 1 panel for quarterly."""
    # --- Monthly 2×2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"ARIMA Forecasts — {periods}-Period Horizon (Monthly)",
                 fontsize=14, fontweight="bold")
    _plot_single_arima_grid(monthly_hist, monthly_fc, periods, "", axes.flat)
    # hide unused 4th panel if only 3 monthly series
    for ax in list(axes.flat)[len(monthly_hist):]:
        ax.set_visible(False)
    plt.tight_layout()
    out = output_dir / "arima_forecasts_monthly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # --- Quarterly (GDP) ---
    fig, axes = plt.subplots(1, len(quarterly_hist), figsize=(7, 4))
    if len(quarterly_hist) == 1:
        axes = [axes]
    fig.suptitle(f"ARIMA Forecasts — {periods}-Period Horizon (Quarterly)",
                 fontsize=13, fontweight="bold")
    _plot_single_arima_grid(quarterly_hist, quarterly_fc, periods, "", axes)
    plt.tight_layout()
    out = output_dir / "arima_forecasts_quarterly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------

def run_var(monthly_df, periods=12):
    """
    VAR on monthly series. Differences non-stationary columns, fits VAR,
    then reconstructs level forecasts via cumulative sum.
    """
    print("\nFitting VAR model...")

    stationary_parts, n_diffs = {}, {}
    for col in monthly_df.columns:
        s, d = make_stationary(monthly_df[col])
        stationary_parts[col] = s
        n_diffs[col] = d
        print(f"  {col:15s} differenced {d}×")

    stat_df = pd.DataFrame(stationary_parts).dropna()

    var_model = VAR(stat_df)
    lag_res   = var_model.select_order(maxlags=12)
    # Fix: use .bic attribute directly, not .selected_orders dict
    best_lag  = int(lag_res.bic) if lag_res.bic >= 1 else 2
    print(f"  VAR lag order (BIC): {best_lag}")

    fit     = var_model.fit(best_lag)
    raw_fc  = fit.forecast(stat_df.values[-best_lag:], steps=periods)
    fc_df   = pd.DataFrame(raw_fc, columns=monthly_df.columns)

    # Reconstruct levels
    level_fc = {}
    for col in monthly_df.columns:
        last_vals = monthly_df[col].dropna()
        fc_vals   = fc_df[col].values
        for _ in range(n_diffs[col]):
            fc_vals = np.cumsum(np.insert(fc_vals, 0, last_vals.iloc[-1]))[1:]
        level_fc[col] = fc_vals

    future_idx = pd.period_range(start=monthly_df.index[-1] + 1,
                                 periods=periods, freq="M")
    return fit, pd.DataFrame(level_fc, index=future_idx)


def plot_var_forecast(monthly_df, var_fc, output_dir):
    """VAR level forecasts vs. last 5 years of history."""
    cols = monthly_df.columns.tolist()
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 4))
    fig.suptitle("VAR Model Forecasts (Monthly Series)",
                 fontsize=13, fontweight="bold")

    for ax, col in zip(axes, cols):
        hist_dt = monthly_df.index.to_timestamp()
        fc_dt   = var_fc.index.to_timestamp()
        ax.plot(hist_dt[-60:], monthly_df[col].values[-60:],
                color="steelblue", lw=1.2, label="Historical (5 yr)")
        ax.plot(fc_dt, var_fc[col].values, color="darkorange",
                lw=1.5, linestyle="--", label="VAR Forecast")
        ax.axvline(hist_dt[-1], color="gray", linestyle=":", lw=0.8)
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
    p.add_argument("--model",   choices=["arima", "var", "both"], default="arima")
    p.add_argument("--periods", type=int, default=12,
                   help="Forecast horizon in periods (default: 12)")
    p.add_argument("--start",   default="2000-01-01",
                   help="Data start date (default: 2000-01-01)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Init client first — fail fast before creating directories
    fred = get_fred_client()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    monthly_df, quarterly_df = fetch_data(fred, start=args.start)

    # --- ARIMA ---
    if args.model in ("arima", "both"):
        print(f"\nFitting ARIMA models ({args.periods}-period horizon)...")

        monthly_hist = {c: monthly_df[c] for c in monthly_df.columns}
        quarterly_hist = {c: quarterly_df[c] for c in quarterly_df.columns}

        monthly_fc   = {}
        quarterly_fc = {}

        for name, series in monthly_hist.items():
            _, fc = run_arima(series, name, ARIMA_ORDERS[name], periods=args.periods)
            monthly_fc[name] = fc

        for name, series in quarterly_hist.items():
            _, fc = run_arima(series, name, ARIMA_ORDERS[name], periods=args.periods)
            quarterly_fc[name] = fc

        # Save CSV
        rows = []
        for fc_dict in (monthly_fc, quarterly_fc):
            for name, fc in fc_dict.items():
                for period, row in fc.iterrows():
                    rows.append({
                        "series":    name,
                        "period":    str(period),
                        "forecast":  round(row["forecast"], 4),
                        "lower_95":  round(row["lower_95"], 4),
                        "upper_95":  round(row["upper_95"], 4),
                    })
        csv_out = output_dir / "arima_forecasts.csv"
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"  Saved → {csv_out}")

        plot_arima_forecasts(monthly_hist, monthly_fc,
                             quarterly_hist, quarterly_fc,
                             args.periods, output_dir)

    # --- VAR ---
    if args.model in ("var", "both"):
        _, var_fc = run_var(monthly_df, periods=args.periods)
        plot_var_forecast(monthly_df, var_fc, output_dir)

        var_out = output_dir / "var_forecast.csv"
        var_fc_export = var_fc.copy()
        var_fc_export.index = var_fc_export.index.strftime("%Y-%m")  # Fix: proper period formatting
        var_fc_export.to_csv(var_out)
        print(f"  Saved → {var_out}")

    print("\nDone.")
