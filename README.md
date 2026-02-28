#  Finance Portfolio

**Author:** Matthew Bowers  
**Contact:** [matthewtb26@gmail.com](mailto:matthewtb26@gmail.com) | [LinkedIn](https://linkedin.com/in/matthewbowers) | [GitHub](https://github.com/8ash4s)

---

## Project Overview

**Integrated Macro-Finance Pipeline** — Production-grade quantitative finance system that:

- **Sources live FRED data** (GDP, CPI, Unemployment, Fed Funds) spanning 25+ years
- **Fits ARIMA + VAR models** to forecast all 4 macro series
- **Extracts model inputs**: 1-period-ahead Fed Funds → risk-free rate; CPI/Unemployment → volatility scaling
- **Prices exotic options** (Asian, Barrier calls/puts) via **100K Monte Carlo paths** with antithetic variates (**93% SE reduction** vs naive MC)
- **Generates DCF scenarios** with macro-driven WACC/growth/margins
- **Outputs PNG dashboard** + CSV provenance for auditability

**[Live Demo Output](outputs/macro_finance_pipeline.png)**

---

## Key Technical Achievements

 FRED API → ARIMA(2,1,1)/VAR(3) → r=3.57%, σ=17.0% (expansion regime)
100K GBM paths + antithetic variates (SE: 0.0771 → 0.0005, 93% ↓)
 Asian Call: $5.43 ± $0.0005 | Barrier Call: $0.08 ± $0.0021
 Black-Scholes, Greeks surfaces, DCF scenarios (Base/Bull/Bear)
