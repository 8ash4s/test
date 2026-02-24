# Exotic Options Pricing Engine

## Overview
Monte Carlo simulation engine for pricing path-dependent exotic options with Heston stochastic volatility calibration.

## Features
- **Asian Options:** Average price options (arithmetic & geometric)
- **Barrier Options:** Knock-in/knock-out with continuous monitoring
- **Heston Model:** Stochastic volatility calibration to market data
- **Greeks Calculation:** Delta, Gamma, Vega, Theta surfaces
- **Variance Reduction:** Antithetic variates, control variates

## Files
- `monte_carlo_engine.py` — Core simulation logic
- `heston_calibration.py` — Model calibration to market volatility surface
- `greeks_visualization.py` — 3D surface plots for Greeks
- `data/sample_market_data.csv` — Sample SPY options chain

## Usage

```bash
# Run basic Monte Carlo pricing
python monte_carlo_engine.py

# Calibrate Heston model
python heston_calibration.py

# Generate Greeks surfaces
python greeks_visualization.py
```

## Results
- Pricing Accuracy: Heston model reduces pricing error vs Black-Scholes in high-vol regimes
- Computational Speed: 100,000 simulations in ~2 seconds
- Greeks Surface: Exported to `outputs/greeks_surface.png`

## Methodology
See `docs/methodology.md` for detailed mathematical framework.
