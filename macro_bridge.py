"""
Macro Bridge
============
Extracts structured MacroInputs from FRED ARIMA forecasts and feeds them into:
  - MonteCarloEngine  (risk-free rate + regime-adjusted vol)
  - build_scenarios() (WACC + revenue growth + margins)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
# Thresholds calibrated to post-2000 U.S. macro history.
# Each regime maps to a volatility multiplier applied to base sigma.

# Rows are evaluated in order; the first match wins.
# Covers all CPI / unemployment combinations without gaps.
_REGIME_TABLE = [
    # (cpi_thresh, cpi_op, urate_thresh, urate_op, label,           sigma_mult)
    (3.0, "<",  5.0, "<",  "expansion",   0.85),   # low inflation, low unemployment
    (4.0, ">=", 5.0, "<",  "overheating", 1.15),   # high inflation, low unemployment
    (4.0, ">=", 5.5, ">=", "stagflation", 1.35),   # high inflation, high unemployment
    (3.0, "<",  5.5, ">=", "contraction", 1.25),   # low inflation, high unemployment
    # Mid-range CPI [3 %, 4 %) — these were previously unmatched "neutral" gaps
    (4.0, "<",  5.0, "<",  "mild_growth",  0.90),  # CPI 3-4 %, urate < 5 %
    (4.0, "<",  5.5, "<",  "mild_growth",  0.95),  # CPI 3-4 %, urate 5-5.5 %
    (4.0, "<",  5.5, ">=", "slowdown",     1.20),  # CPI 3-4 %, urate >= 5.5 %
]


def classify_regime(cpi_yoy: float, unemployment: float) -> tuple[str, float]:
    """Return (regime_label, sigma_multiplier) for given macro readings."""
    for cpi_thresh, cpi_op, u_thresh, u_op, label, mult in _REGIME_TABLE:
        cpi_ok = (cpi_yoy < cpi_thresh) if cpi_op == "<" else (cpi_yoy >= cpi_thresh)
        u_ok   = (unemployment < u_thresh) if u_op == "<" else (unemployment >= u_thresh)
        if cpi_ok and u_ok:
            return label, mult
    return "neutral", 1.0


# ---------------------------------------------------------------------------
# MacroInputs dataclass
# ---------------------------------------------------------------------------

@dataclass
class MacroInputs:
    """
    Structured container of macro-derived model inputs.

    Options layer
    -------------
    forward_rate     : ARIMA[+1] Fed Funds forecast (decimal, e.g. 0.043)
    sigma_multiplier : vol scaling factor from macro regime
    regime           : regime label (expansion / overheating / stagflation /
                       contraction / neutral)

    DCF / scenario layer
    --------------------
    base/bull/bear_wacc   : scenario WACCs (decimal)
    base/bull/bear_growth : scenario revenue growth rates (decimal)
    base/bull/bear_margin : scenario EBITDA margins (decimal)
    """
    # Options inputs
    forward_rate:     float
    sigma_multiplier: float
    regime:           str
    cpi_yoy:          float
    unemployment:     float

    # DCF inputs
    base_wacc:    float;  bull_wacc:    float;  bear_wacc:    float
    base_growth:  float;  bull_growth:  float;  bear_growth:  float
    base_margin:  float;  bull_margin:  float;  bear_margin:  float

    def summary(self) -> str:
        return "\n".join([
            f"  Regime           : {self.regime.upper()}",
            f"  CPI YoY          : {self.cpi_yoy:.2f}%",
            f"  Unemployment     : {self.unemployment:.1f}%",
            f"  Forward Rate     : {self.forward_rate:.2%}  (ARIMA[+1] Fed Funds)",
            f"  Vol Multiplier   : {self.sigma_multiplier:.2f}×",
            f"  WACC  B/Bu/Be    : "
            f"{self.base_wacc:.2%} / {self.bull_wacc:.2%} / {self.bear_wacc:.2%}",
            f"  Growth B/Bu/Be   : "
            f"{self.base_growth:.2%} / {self.bull_growth:.2%} / {self.bear_growth:.2%}",
        ])

    def to_csv(self, path: str | Path = "outputs/macro_inputs.csv") -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(self)]).to_csv(out, index=False)
        return out


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_macro_inputs(
    arima_forecasts: dict,
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    credit_spread: float = 0.02,
) -> MacroInputs:
    """
    Build MacroInputs from ARIMA forecast outputs and raw FRED series.

    Parameters
    ----------
    arima_forecasts : dict[str, pd.DataFrame]
        Keyed by series name; each value is the forecast DataFrame from
        run_arima() with columns ['forecast', 'lower_95', 'upper_95'].
    monthly_df      : Historical monthly FRED data (CPI, Unemployment, Fed Funds).
    quarterly_df    : Historical quarterly FRED data (GDP).
    credit_spread   : Spread over risk-free rate for WACC (default: 200 bps).
    """

    # 1. Forward risk-free rate  (FRED reports Fed Funds in %, convert to decimal)
    ff_fc        = arima_forecasts["Fed Funds"]
    forward_rate = float(ff_fc["forecast"].iloc[0]) / 100.0
    forward_rate = max(forward_rate, 0.001)          # floor at 10 bps
    ff_lower     = float(ff_fc["lower_95"].iloc[0]) / 100.0
    ff_upper     = float(ff_fc["upper_95"].iloc[0]) / 100.0

    # 2. Macro regime  →  vol multiplier
    cpi_series   = monthly_df["CPI"]
    cpi_yoy      = float((cpi_series.iloc[-1] / cpi_series.iloc[-13] - 1) * 100)
    unemployment = float(monthly_df["Unemployment"].iloc[-1])
    regime, sigma_mult = classify_regime(cpi_yoy, unemployment)

    # 3. WACC scenarios
    #    Base = forward_rate + credit_spread
    #    Bull = lower CI of Fed Funds (accommodative) + tighter spread
    #    Bear = upper CI of Fed Funds (tightening)    + wider spread
    base_wacc = forward_rate + credit_spread
    bull_wacc = max(ff_lower, 0.001) + credit_spread * 0.9
    bear_wacc = ff_upper + credit_spread * 1.1

    # 4. Revenue growth scenarios (annualized 1-quarter-ahead GDP change)
    gdp_fc      = arima_forecasts["GDP"]
    gdp_last    = float(quarterly_df["GDP"].iloc[-1])
    gdp_next    = float(gdp_fc["forecast"].iloc[0])
    base_growth = float(max((gdp_next / gdp_last - 1) * 4, -0.10))
    bull_growth = float(min(base_growth + 0.04,  0.15))
    bear_growth = float(max(base_growth - 0.04, -0.15))  # floor at -15 %, below base floor

    # 5. EBITDA margin scenarios (CPI-driven cost pressure)
    if cpi_yoy > 5.0:
        base_m, bull_m, bear_m = 0.20, 0.23, 0.17
    elif cpi_yoy > 3.0:
        base_m, bull_m, bear_m = 0.22, 0.25, 0.19
    else:
        base_m, bull_m, bear_m = 0.24, 0.27, 0.21

    return MacroInputs(
        forward_rate     = round(forward_rate, 5),
        sigma_multiplier = round(sigma_mult,   3),
        regime           = regime,
        cpi_yoy          = round(cpi_yoy,      2),
        unemployment     = round(unemployment,  1),
        base_wacc        = round(base_wacc,     4),
        bull_wacc        = round(bull_wacc,     4),
        bear_wacc        = round(bear_wacc,     4),
        base_growth      = round(base_growth,   4),
        bull_growth      = round(bull_growth,   4),
        bear_growth      = round(bear_growth,   4),
        base_margin      = base_m,
        bull_margin      = bull_m,
        bear_margin      = bear_m,
    )
