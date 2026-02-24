"""
Financial Model Automation Helpers
Author: Matthew Bowers
"""

from datetime import date
from pathlib import Path

import pandas as pd


def build_scenarios():
    data = {
        "Scenario": ["Base", "Bull", "Bear"],
        "Revenue Growth": [0.08, 0.12, 0.04],
        "EBITDA Margin": [0.22, 0.25, 0.19],
        "WACC": [0.095, 0.09, 0.105],
        "Exit Multiple": [10.0, 11.0, 9.0],
    }
    return pd.DataFrame(data)


def export_scenarios(output_path="scenario_assumptions.csv"):
    df = build_scenarios()
    out_file = Path(output_path)
    df.to_csv(out_file, index=False)
    return out_file


if __name__ == "__main__":
    output_file = export_scenarios()
    print(f"Exported scenario assumptions to {output_file} on {date.today().isoformat()}")
