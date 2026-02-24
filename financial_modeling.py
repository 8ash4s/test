"""
Financial Model Automation
Author: Matthew Bowers
"""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from macro_bridge import MacroInputs


# ---------------------------------------------------------------------------
# Regime-aware exit multiples
# Higher-growth / lower-risk regimes command premium multiples; stagflation
# and contraction compress them.  Tuples are (base, bull, bear).
# ---------------------------------------------------------------------------
_REGIME_EXIT_MULTIPLES: dict[str, tuple[float, float, float]] = {
    "expansion":   (11.0, 12.0, 10.0),
    "mild_growth": (10.5, 11.5,  9.5),
    "overheating": (10.0, 11.0,  9.0),
    "neutral":     (10.0, 11.0,  9.0),
    "slowdown":    ( 9.0, 10.0,  8.0),
    "contraction": ( 8.5,  9.5,  7.5),
    "stagflation": ( 7.5,  8.5,  6.5),
}


def _exit_multiples(regime: str) -> tuple[float, float, float]:
    """Return (base, bull, bear) exit multiples for a given macro regime."""
    return _REGIME_EXIT_MULTIPLES.get(regime, _REGIME_EXIT_MULTIPLES["neutral"])


def build_scenarios(macro: "MacroInputs | None" = None) -> pd.DataFrame:
    """
    Return Base / Bull / Bear scenario assumptions.
    If MacroInputs is supplied, WACC, growth, and exit multiples are
    FRED/regime-derived.  Falls back to static defaults otherwise.
    """
    if macro is not None:
        em_base, em_bull, em_bear = _exit_multiples(macro.regime)
        df = pd.DataFrame({
            "Scenario":       ["Base",             "Bull",             "Bear"],
            "Revenue Growth": [macro.base_growth,   macro.bull_growth,  macro.bear_growth],
            "EBITDA Margin":  [macro.base_margin,   macro.bull_margin,  macro.bear_margin],
            "WACC":           [macro.base_wacc,     macro.bull_wacc,    macro.bear_wacc],
            "Exit Multiple":  [em_base,             em_bull,            em_bear],
            "Macro Regime":   [macro.regime] * 3,
        })
    else:
        # Static fallback — no FRED connection required
        df = pd.DataFrame({
            "Scenario":       ["Base",   "Bull",  "Bear"],
            "Revenue Growth": [0.08,     0.12,    0.04],
            "EBITDA Margin":  [0.22,     0.25,    0.19],
            "WACC":           [0.095,    0.09,    0.105],
            "Exit Multiple":  [10.0,     11.0,    9.0],
            "Macro Regime":   ["static"] * 3,
        })

    # Round numeric columns to clean, consistent precision
    pct_cols = ["Revenue Growth", "EBITDA Margin", "WACC"]
    df[pct_cols] = df[pct_cols].round(4)
    df["Exit Multiple"] = df["Exit Multiple"].round(1)
    return df


def export_scenarios(
    macro: "MacroInputs | None" = None,
    output_path: str = "outputs/scenario_assumptions.csv",
) -> Path:
    """Write scenario assumptions to CSV and return the output path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_scenarios(macro).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = export_scenarios()
    print(f"Exported scenario assumptions to {out} on {date.today().isoformat()}")
    print(build_scenarios().to_string(index=False))
