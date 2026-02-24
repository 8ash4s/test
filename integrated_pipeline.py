"""
Integrated Macro-Finance Pipeline
==================================
Runs all four stages end-to-end:

  Stage 1 — FRED pull + ARIMA forecasts      (macro_forecasting.py)
  Stage 2 — Extract MacroInputs              (macro_bridge.py)
  Stage 3 — Macro-aware options pricing      (options_pricing.py)
  Stage 4 — Macro-driven DCF scenarios       (financial_modeling.py)

Usage:
  python integrated_pipeline.py
  python integrated_pipeline.py --S0 470 --K 480 --sigma 0.22 --n-sims 200000
  python integrated_pipeline.py --credit-spread 0.025 --periods 4
"""

import argparse
from pathlib import Path

import pandas as pd

from macro_forecasting import get_fred_client, fetch_data, run_arima, ARIMA_ORDERS
from macro_bridge import extract_macro_inputs
from options_pricing import MonteCarloEngine
from financial_modeling import build_scenarios, export_scenarios


def parse_args():
    p = argparse.ArgumentParser(description="Integrated Macro-Finance Pipeline")
    p.add_argument("--S0",            type=float, default=450,   help="Current stock price")
    p.add_argument("--K",             type=float, default=460,   help="Strike price")
    p.add_argument("--T",             type=float, default=0.25,  help="Time to maturity (years)")
    p.add_argument("--sigma",         type=float, default=0.20,  help="Base volatility (pre-regime)")
    p.add_argument("--barrier",       type=float, default=470,   help="Barrier level")
    p.add_argument("--n-sims",        type=int,   default=100_000)
    p.add_argument("--credit-spread", type=float, default=0.02,  help="Spread over risk-free rate")
    p.add_argument("--periods",       type=int,   default=4,     help="ARIMA forecast horizon")
    p.add_argument("--start",         default="2000-01-01",      help="FRED data start date")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # Stage 1: FRED → ARIMA forecasts
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STAGE 1 — MACRO FORECASTING  (FRED + ARIMA)")
    print("=" * 60)
    fred = get_fred_client()
    monthly_df, quarterly_df = fetch_data(fred, start=args.start)

    all_series = {
        **{c: monthly_df[c]   for c in monthly_df.columns},
        **{c: quarterly_df[c] for c in quarterly_df.columns},
    }
    arima_forecasts = {}
    for name, series in all_series.items():
        _, fc = run_arima(series, name, ARIMA_ORDERS[name], periods=args.periods)
        arima_forecasts[name] = fc

    # ---------------------------------------------------------------
    # Stage 2: Extract MacroInputs
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2 — MACRO BRIDGE")
    print("=" * 60)
    macro = extract_macro_inputs(
        arima_forecasts, monthly_df, quarterly_df,
        credit_spread=args.credit_spread,
    )
    print(macro.summary())
    macro.to_csv(outdir / "macro_inputs.csv")
    print(f"  Saved → {outdir / 'macro_inputs.csv'}")

    # ---------------------------------------------------------------
    # Stage 3: Options pricing — FRED r + regime-adjusted sigma
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3 — OPTIONS PRICING  (MACRO-AWARE)")
    print("=" * 60)
    eff_sigma = args.sigma * macro.sigma_multiplier
    print(f"  sigma  {args.sigma:.2%} × {macro.sigma_multiplier:.2f} "
          f"({macro.regime}) = {eff_sigma:.2%}")
    print(f"  r      {macro.forward_rate:.2%}  (ARIMA[+1] FEDFUNDS)")

    engine = MonteCarloEngine(
        S0            = args.S0,
        K             = args.K,
        T             = args.T,
        r             = macro.forward_rate,   # FRED-derived
        sigma         = eff_sigma,            # regime-adjusted
        n_simulations = args.n_sims,
        n_steps       = max(int(args.T * 252), 1),
    )

    bs            = engine.black_scholes_call()
    asian, a_se   = engine.price_asian_call()
    barrier, b_se = engine.price_barrier_call(barrier=args.barrier)

    print(f"\n  Black-Scholes Call          ${bs:.4f}")
    print(f"  Asian Call (arithmetic)     ${asian:.4f} +/- ${a_se:.4f}")
    print(f"  Barrier Call (up-and-out)   ${barrier:.4f} +/- ${b_se:.4f}")

    options_out = outdir / "options_results_macro.csv"
    pd.DataFrame({
        "Option Type": ["Black-Scholes", "Asian Call", "Barrier (Up-and-Out)"],
        "Price":       [bs,   asian,  barrier],
        "Std Error":   [0.0,  a_se,   b_se],
        "r (FRED)":    [macro.forward_rate] * 3,
        "sigma (adj)": [eff_sigma] * 3,
        "Regime":      [macro.regime] * 3,
    }).to_csv(options_out, index=False)
    print(f"  Saved → {options_out}")

    # ---------------------------------------------------------------
    # Stage 4: DCF scenarios — FRED-driven WACC + growth
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4 — DCF SCENARIO MODEL  (MACRO-DRIVEN)")
    print("=" * 60)
    scenarios = build_scenarios(macro)
    print(scenarios.to_string(index=False))
    scen_out = export_scenarios(macro, outdir / "scenario_assumptions.csv")
    print(f"  Saved → {scen_out}")

    print("\n" + "=" * 60)
    print("Pipeline complete — all outputs in ./outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
