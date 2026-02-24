# Methodology

## Exotic Options Pricing
- Simulate asset paths under Geometric Brownian Motion using Monte Carlo.
- Apply antithetic variates to reduce estimator variance.
- Price Asian and Barrier options from discounted expected payoffs.
- Compare baseline European call value using Black-Scholes.

## Heston Calibration
- Represent stochastic variance dynamics with mean reversion.
- Use Fourier inversion for European call pricing under Heston.
- Calibrate parameters by minimizing RMSE against observed market option prices.
- Enforce positivity and the Feller condition during optimization.

## Greeks Estimation
- Compute Delta and Gamma via central finite differences.
- Build 3D surfaces across stock price and maturity grids.

## Financial Modeling
- Build a linked 3-statement forecast model.
- Perform DCF valuation with explicit forecast and terminal value.
- Evaluate LBO returns using debt schedule and exit assumptions.
