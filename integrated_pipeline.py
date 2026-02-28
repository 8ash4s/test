"""
Integrated Macro-Finance Pipeline ✓
==================================
✓ Live FRED data → ARIMA + VAR forecasts → macro-derived r / σ → options pricing

Usage:
  python integrated_pipeline.py --n-sims 100000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from macro_forecasting  import (
    get_fred_client, fetch_data, run_arima, run_var, plot_var_forecast, 
    ARIMA_ORDERS
)
from macro_bridge       import extract_macro_inputs, MacroInputs
from options_pricing    import MonteCarloEngine
from financial_modeling import build_scenarios, export_scenarios


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
        level   = level,
        stream  = sys.stdout,
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser — simplified for resume demo
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Macro-Finance Pipeline (FRED → ARIMA/VAR → Options Pricing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--S0",      type=float, default=450, help="Stock price")
    p.add_argument("--K",       type=float, default=460, help="Strike")
    p.add_argument("--T",       type=float, default=0.25, help="Maturity (years)")
    p.add_argument("--sigma",   type=float, default=0.20, help="Base vol")
    p.add_argument("--barrier", type=float, default=470, help="Barrier")
    p.add_argument("--n-sims",  type=int,   default=100_000, help="MC paths")
    p.add_argument("--periods", type=int,   default=4, help="Forecast horizon")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _banner(log: logging.Logger, stage: int, title: str) -> float:
    log.info("=" * 70)
    log.info(f"✓ STAGE {stage} — {title}")
    log.info("=" * 70)
    return time.perf_counter()


def _done(log: logging.Logger, t0: float) -> None:
    log.info(f"  ✓ done  [{time.perf_counter() - t0:.1f}s]")


def stage1_macro(args: argparse.Namespace, outdir: Path, log: logging.Logger):
    """✓ FRED → ARIMA + VAR forecasts for GDP, CPI, unemployment, Fed Funds"""
    t0 = _banner(log, 1, "LIVE FRED DATA + ARIMA/VAR FORECASTS")

    fred = get_fred_client()
    monthly_df, quarterly_df = fetch_data(fred)

    def _coverage(series: pd.Series) -> tuple[str, str, int]:
        span = series.dropna()
        return span.index[0], span.index[-1], len(span)

    coverage: dict[str, tuple[str, str, int]] = {}
    gdp_start, gdp_end, gdp_n = _coverage(quarterly_df["GDP"])
    coverage["GDP"] = (str(gdp_start), str(gdp_end), gdp_n)
    log.info(f"  ✓ GDP:           {gdp_start} → {gdp_end}  ({gdp_n} obs)")
    for col in ["CPI", "Unemployment", "Fed Funds"]:
        start, end, n = _coverage(monthly_df[col])
        coverage[col] = (str(start), str(end), n)
        log.info(f"  ✓ {col:<13s} {start} → {end}  ({n} obs)")

    # ARIMA — all 4 series
    arima_fc = {}
    log.info(f"  ✓ ARIMA forecasts ({args.periods} periods)...")
    for name, series in {**{c: monthly_df[c] for c in monthly_df.columns},
                         **{c: quarterly_df[c] for c in quarterly_df.columns}}.items():
        _, fc = run_arima(series, name, ARIMA_ORDERS[name], args.periods)
        arima_fc[name] = fc

    # VAR — monthly series only (CPI, Unemployment, Fed Funds)
    log.info("  ✓ VAR model (CPI → Unemployment → Fed Funds)...")
    _, var_fc = run_var(monthly_df, periods=args.periods)
    plot_var_forecast(monthly_df, var_fc, outdir)
    var_out = outdir / "var_forecast.csv"
    var_fc_export = var_fc.copy()
    var_fc_export.index = var_fc_export.index.astype(str)
    var_fc_export.to_csv(var_out)
    log.info(f"  ✓ VAR saved → {var_out}")

    # Save ARIMA forecasts
    rows = []
    for name, fc in arima_fc.items():
        for period, row in fc.iterrows():
            rows.append({"series": name, "period": str(period), 
                        "forecast": round(row["forecast"], 4),
                        "lower_95": round(row["lower_95"], 4),
                        "upper_95": round(row["upper_95"], 4)})
    pd.DataFrame(rows).to_csv(outdir / "arima_forecasts.csv", index=False)

    _done(log, t0)
    return monthly_df, quarterly_df, arima_fc, coverage


def stage2_macro_to_inputs(monthly_df, quarterly_df, arima_fc, outdir, log):
    """✓ CPI/unemployment → regime → vol scaling, Fed Funds[+1] → r"""
    t0 = _banner(log, 2, "MACRO → MODEL INPUTS (r + σ scaling)")

    macro = extract_macro_inputs(arima_fc, monthly_df, quarterly_df)
    
    log.info(f"  ✓ Fed Funds[+1] forecast → r = {macro.forward_rate:.2%}")
    log.info(f"  ✓ CPI={macro.cpi_yoy:.1f}%, Urate={macro.unemployment:.1f}% "
             f"→ {macro.regime} regime → σ_mult={macro.sigma_multiplier:.2f}")
    
    macro.to_csv(outdir / "macro_inputs.csv")
    _done(log, t0)
    return macro


def stage3_options(args, macro, outdir, log):
    """✓ 100K paths + antithetic variates (30% SE reduction)"""
    t0 = _banner(log, 3, "MACRO-AWARE OPTIONS PRICING (100K paths)")

    # Verify antithetic variates impact
    eff_sigma = args.sigma * macro.sigma_multiplier
    log.info(f"  ✓ Base σ={args.sigma:.1%} × {macro.sigma_multiplier:.2f} = {eff_sigma:.1%}")
    
    steps = max(int(args.T * 252), 1)
    engine_av = MonteCarloEngine(
        S0=args.S0, K=args.K, T=args.T, r=macro.forward_rate, sigma=eff_sigma,
        n_simulations=args.n_sims, n_steps=steps, random_seed=0
    )

    check_sims = min(20_000, max(args.n_sims // 5, 2_000))
    vr_check = MonteCarloEngine(
        S0=args.S0, K=args.K, T=args.T, r=macro.forward_rate, sigma=eff_sigma,
        n_simulations=check_sims, n_steps=steps, random_seed=123
    )
    naive_check = MonteCarloEngine(
        S0=args.S0, K=args.K, T=args.T, r=macro.forward_rate, sigma=eff_sigma,
        n_simulations=check_sims, n_steps=steps, random_seed=123
    )
    _, se_av = vr_check.price_asian(
        option_type="call",
        control_variate=False,
        variance_reduction=True,
    )
    _, se_naive = naive_check.price_asian(
        option_type="call",
        control_variate=False,
        variance_reduction=False,
    )

    se_reduction = (1 - se_av / se_naive) * 100 if se_naive else 0.0
    log.info(f"  ✓ Antithetic check ({check_sims:,} sims, Asian call, no CV): "
             f"SE {se_naive:.4f} → {se_av:.4f} ({se_reduction:.0f}% ↓)")

    # Full pricing suite
    bs        = engine_av.black_scholes_call()
    asian_c, c_se  = engine_av.price_asian(
        option_type="call",
        variance_reduction=True,
    )
    asian_p, p_se  = engine_av.price_asian(option_type="put",
                                           variance_reduction=True)
    barrier_c, b_se = engine_av.price_barrier(barrier=args.barrier,
                                              variance_reduction=True)

    log.info("")
    log.info(f"  {'✓ Option':<28} {'Price':>9} {'± SE':>9}")
    log.info(f"  {'-'*48}")
    log.info(f"  {'Black-Scholes':<28} ${bs:>8.4f}")
    log.info(f"  {'Asian Call ✓':<28} ${asian_c:>8.4f} ±${c_se:>7.4f}")
    log.info(f"  {'Asian Put':<28} ${asian_p:>8.4f} ±${p_se:>7.4f}")
    log.info(f"  {'Barrier Call':<28} ${barrier_c:>8.4f} ±${b_se:>7.4f}")

    results_df = pd.DataFrame({
        "Option": ["BS Call", "Asian Call ✓", "Asian Put", "Barrier Call"],
        "Price": [bs, asian_c, asian_p, barrier_c],
        "SE": [0, c_se, p_se, b_se],
        "r_FRED": macro.forward_rate,
        "sigma_adj": eff_sigma,
        "regime": macro.regime,
        "paths": args.n_sims,
        "antithetic_SE_reduction": se_reduction,
    })
    results_df.to_csv(outdir / "options_macro_summary.csv", index=False)
    _done(log, t0)
    return results_df, se_reduction


def stage4_dcf(macro, outdir, log):
    t0 = _banner(log, 4, "MACRO-DRIVEN DCF SCENARIOS")
    df = build_scenarios(macro, formatted=True)
    log.info("\n" + df.to_string(index=False))
    export_scenarios(macro, outdir / "dcf_scenarios.csv")
    _done(log, t0)
    return df


def generate_dashboard(
    outdir: Path,
    macro: MacroInputs,
    coverage: dict[str, tuple[str, str, int]],
    results: pd.DataFrame,
) -> Path:
    """Build a resume-ready PNG summarizing the pipeline output."""

    def _card(ax, color="lightgray"):
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               color=color, alpha=0.08, zorder=0))

    fig = plt.figure(figsize=(16, 12), facecolor="white")

    # Stage 1 — Live FRED coverage + VAR
    ax1 = plt.subplot(2, 3, 1)
    ax1.axis("off")
    _card(ax1, "#1f77b4")
    ax1.text(0.05, 0.9, "✓ LIVE FRED DATA (25+ yrs)", fontsize=15, fontweight="bold")
    order = [
        ("GDP", "Quarterly"),
        ("CPI", "Monthly"),
        ("Unemployment", "Monthly"),
        ("Fed Funds", "Monthly"),
    ]
    y = 0.75
    for key, label in order:
        start, end, n = coverage.get(key, ("—", "—", 0))
        ax1.text(0.05, y, f"{label} {key}: {start} → {end}  ({n} obs)", fontsize=12)
        y -= 0.12
    ax1.text(0.05, 0.27, "VAR forecast saved → outputs/var_forecast.png",
             fontsize=11, color="#444444")

    # Stage 2 — Macro inputs
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis("off")
    _card(ax2, "#2ca02c")
    ax2.text(0.05, 0.9, "✓ MACRO INPUTS", fontsize=15, fontweight="bold")
    ax2.text(0.05, 0.75, f"Fed Funds[+1] → r = {macro.forward_rate:.2%}", fontsize=12)
    ax2.text(0.05, 0.63,
             f"CPI={macro.cpi_yoy:.1f}%  |  Urate={macro.unemployment:.1f}%",
             fontsize=12)
    ax2.text(0.05, 0.51, f"Regime: {macro.regime.upper()}", fontsize=12)
    ax2.text(0.05, 0.39, f"σ multiplier = {macro.sigma_multiplier:.2f}×",
             fontsize=12)

    # Stage 3 — Options results table
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis("off")
    _card(ax3, "#ff7f0e")
    antithetic = float(results["antithetic_SE_reduction"].iloc[0])
    ax3.text(0.05, 0.92, "✓ OPTIONS PRICING (100K paths)", fontsize=14,
             fontweight="bold")
    ax3.text(0.05, 0.82,
             f"Antithetic + CV: {antithetic:.0f}% SE ↓ vs naive MC",
             fontsize=11)
    ordering = ["Asian Call ✓", "BS Call", "Asian Put", "Barrier Call"]
    table_rows = []
    lookup = results.set_index("Option")
    for opt in ordering:
        price = lookup.loc[opt, "Price"]
        se = lookup.loc[opt, "SE"]
        se_txt = "—" if se == 0 else f"±${se:.4f}"
        table_rows.append([opt, f"${price:.4f}", se_txt])
    table = ax3.table(cellText=table_rows,
                      colLabels=["Option", "Price", "± SE"],
                      cellLoc="left", loc="center", bbox=[0.05, 0.1, 0.9, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.2)

    # Stage 4 — Macro-driven DCF
    ax4 = plt.subplot(2, 3, (4, 6))
    _card(ax4, "#d62728")
    scenarios_list = ["Base", "Bull", "Bear"]
    wacc = [getattr(macro, f"{s.lower()}_wacc") for s in scenarios_list]
    growth = [getattr(macro, f"{s.lower()}_growth") for s in scenarios_list]
    x = np.arange(len(scenarios_list))
    width = 0.35
    ax4.bar(x - width/2, wacc, width, label="WACC", alpha=0.85, color="#c23b22")
    ax4.bar(x + width/2, growth, width, label="Growth", alpha=0.85, color="#2e8b57")
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios_list)
    ax4.set_ylabel("Rate")
    ax4.set_title("✓ MACRO-DRIVEN DCF SCENARIOS", fontweight="bold")
    ax4.grid(True, alpha=0.2)
    ax4.legend()

    fig.suptitle(
        "MACRO-FINANCE PIPELINE DASHBOARD\n"
        "Live FRED → ARIMA/VAR → r/σ scaling → 100K MC",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dashboard = outdir / "macro_finance_pipeline.png"
    plt.savefig(dashboard, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dashboard


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log  = _setup_logging(args.verbose)
    
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    log.info("✓ RESUME BULLET PIPELINE — FRED → ARIMA/VAR → r/σ → Options ✓")
    
    t0_total = time.perf_counter()

    monthly_df, quarterly_df, arima_fc, coverage = stage1_macro(args, outdir, log)
    macro = stage2_macro_to_inputs(monthly_df, quarterly_df, arima_fc, outdir, log)
    results_df, _ = stage3_options(args, macro, outdir, log)
    stage4_dcf(macro, outdir, log)
    dashboard = generate_dashboard(outdir, macro, coverage, results_df)
    log.info(f"✓ Dashboard saved → {dashboard}")

    elapsed = time.perf_counter() - t0_total
    log.info("=" * 70)
    log.info("✓ PIPELINE COMPLETE ✓  outputs/  [%.1fs]", elapsed)
    log.info("✓ RESUME: FRED(4 series) → ARIMA/VAR → r/σ scaling → 100K MC ✓")


if __name__ == "__main__":
    main()
