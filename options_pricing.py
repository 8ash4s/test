"""
Exotic Options Pricing — Monte Carlo Engine + Greeks Visualization
Author: Matthew Bowers
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    def __init__(self, S0, K, T, r, sigma,
                 n_simulations=100_000, n_steps=252, random_seed=None):
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be positive.")
        if T <= 0:
            raise ValueError("T must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if n_simulations < 2:
            raise ValueError("n_simulations must be at least 2.")

        self.S0           = float(S0)
        self.K            = float(K)
        self.T            = float(T)
        self.r            = float(r)
        self.sigma        = float(sigma)
        self.n_simulations = int(n_simulations)
        self.n_steps      = int(n_steps)
        self.dt           = T / n_steps
        self.rng          = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def generate_paths(self, variance_reduction=True):
        """Fully vectorized GBM paths with optional antithetic variates."""
        n = (self.n_simulations + 1) // 2 if variance_reduction else self.n_simulations
        Z = self.rng.standard_normal((n, self.n_steps))
        if variance_reduction:
            Z = np.vstack([Z, -Z])[: self.n_simulations]

        drift     = (self.r - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        log_ret   = drift + diffusion * Z

        paths          = np.empty((self.n_simulations, self.n_steps + 1))
        paths[:, 0]    = self.S0
        paths[:, 1:]   = self.S0 * np.exp(np.cumsum(log_ret, axis=1))
        return paths

    # ------------------------------------------------------------------
    # Analytical benchmarks
    # ------------------------------------------------------------------

    def black_scholes_call(self):
        """Analytical Black-Scholes call price."""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    @staticmethod
    def _geometric_asian_call_bs(S0, K, T, r, sigma, n_steps):
        """Closed-form geometric-average Asian call (Kemna & Vorst 1990)."""
        sigma_adj = sigma * np.sqrt((2 * n_steps + 1) / (6 * (n_steps + 1)))
        r_adj     = 0.5 * (r - 0.5 * sigma ** 2 + sigma_adj ** 2)
        d1 = (np.log(S0 / K) + (r_adj + 0.5 * sigma_adj ** 2) * T) / (
            sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)
        return (S0 * np.exp((r_adj - r) * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2))

    # ------------------------------------------------------------------
    # Option pricers
    # ------------------------------------------------------------------

    def price_asian(self, averaging="arithmetic", option_type="call",
                    control_variate=True):
        """
        Price an Asian option.

        Parameters
        ----------
        averaging      : 'arithmetic' or 'geometric'
        option_type    : 'call' or 'put'
        control_variate: Use geometric CV when averaging='arithmetic'
        """
        paths    = self.generate_paths()
        spot     = paths[:, 1:]
        discount = np.exp(-self.r * self.T)
        sign     = 1.0 if option_type == "call" else -1.0

        if averaging == "arithmetic" and control_variate:
            arith_avg = spot.mean(axis=1)
            geo_avg   = np.exp(np.log(spot).mean(axis=1))
            arith_pay = np.maximum(sign * (arith_avg - self.K), 0)
            geo_pay   = np.maximum(sign * (geo_avg   - self.K), 0)
            geo_exact = self._geometric_asian_call_bs(
                self.S0, self.K, self.T, self.r, self.sigma, self.n_steps)
            if option_type == "put":
                # put-call parity adjustment for geometric exact
                geo_exact = (geo_exact - self.S0 * np.exp(-self.r * self.T)
                             + self.K * np.exp(-self.r * self.T))
            cov_mat  = np.cov(arith_pay, geo_pay)
            beta     = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12)
            adjusted = arith_pay - beta * (geo_pay - geo_exact / discount)
            price    = discount * adjusted.mean()
            se       = discount * adjusted.std() / np.sqrt(self.n_simulations)
        else:
            avg = (spot.mean(axis=1) if averaging == "arithmetic"
                   else np.exp(np.log(spot).mean(axis=1)))
            payoffs = np.maximum(sign * (avg - self.K), 0)
            price   = discount * payoffs.mean()
            se      = discount * payoffs.std() / np.sqrt(self.n_simulations)

        return float(price), float(se)

    def price_barrier(self, barrier, barrier_type="up-and-out", option_type="call"):
        """
        Price a barrier option (call or put).

        barrier_type: 'up-and-out' | 'down-and-out' | 'up-and-in' | 'down-and-in'
        """
        paths    = self.generate_paths()
        final    = paths[:, -1]
        discount = np.exp(-self.r * self.T)
        sign     = 1.0 if option_type == "call" else -1.0
        vanilla  = np.maximum(sign * (final - self.K), 0)

        direction, knock = barrier_type.split("-and-")
        hit = (np.any(paths >= barrier, axis=1) if direction == "up"
               else np.any(paths <= barrier, axis=1))
        payoffs = np.where(hit, 0, vanilla) if knock == "out" else np.where(hit, vanilla, 0)

        price = discount * payoffs.mean()
        se    = discount * payoffs.std() / np.sqrt(self.n_simulations)
        return float(price), float(se)


# ---------------------------------------------------------------------------
# Greeks (finite difference) — fixed seeds for stability
# ---------------------------------------------------------------------------

def _engine_kw(engine):
    return dict(K=engine.K, T=engine.T, r=engine.r, sigma=engine.sigma,
                n_steps=engine.n_steps)


def calculate_delta(engine, dS=1.0, n_simulations=20_000, seed=42):
    """Central-difference Delta with matched seeds."""
    kw = {**_engine_kw(engine), "n_simulations": n_simulations}
    up,   _ = MonteCarloEngine(S0=engine.S0 + dS, random_seed=seed,     **kw).price_asian()
    down, _ = MonteCarloEngine(S0=engine.S0 - dS, random_seed=seed + 1, **kw).price_asian()
    return (up - down) / (2 * dS)


def calculate_gamma(engine, dS=1.0, n_simulations=20_000, seed=42):
    """Central-difference Gamma with matched seeds."""
    kw  = {**_engine_kw(engine), "n_simulations": n_simulations}
    up,  _ = MonteCarloEngine(S0=engine.S0 + dS, random_seed=seed,     **kw).price_asian()
    mid, _ = engine.price_asian()
    dn,  _ = MonteCarloEngine(S0=engine.S0 - dS, random_seed=seed + 1, **kw).price_asian()
    return (up - 2 * mid + dn) / dS ** 2


def calculate_vega(engine, dSig=0.01, n_simulations=20_000, seed=42):
    """Central-difference Vega (new — not in original)."""
    kw = dict(K=engine.K, T=engine.T, r=engine.r, n_steps=engine.n_steps,
              n_simulations=n_simulations, S0=engine.S0)
    up,   _ = MonteCarloEngine(sigma=engine.sigma + dSig, random_seed=seed,     **kw).price_asian()
    down, _ = MonteCarloEngine(sigma=engine.sigma - dSig, random_seed=seed + 1, **kw).price_asian()
    return (up - down) / (2 * dSig)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_greeks_surface(grid_size=8, n_simulations=10_000, show_plot=False):
    """Generate and save 3-D Delta/Gamma surface plots."""
    S_range = np.linspace(400, 500, grid_size)
    T_range = np.linspace(0.1, 1.0, grid_size)
    S_grid, T_grid = np.meshgrid(S_range, T_range)
    delta_grid = np.zeros_like(S_grid)
    gamma_grid = np.zeros_like(S_grid)

    print("Calculating Greeks surface...")
    for i, T in enumerate(T_range):
        for j, S in enumerate(S_range):
            eng = MonteCarloEngine(S0=S, K=450, T=T, r=0.05, sigma=0.20,
                                   n_simulations=n_simulations)
            delta_grid[i, j] = calculate_delta(eng, n_simulations=n_simulations)
            gamma_grid[i, j] = calculate_gamma(eng, n_simulations=n_simulations)
        print(f"  {(i + 1) / len(T_range):.0%}")

    fig = plt.figure(figsize=(16, 6))
    for ax, grid, cmap, label in [
        (fig.add_subplot(121, projection="3d"), delta_grid, "viridis", "Delta"),
        (fig.add_subplot(122, projection="3d"), gamma_grid, "plasma",  "Gamma"),
    ]:
        surf = ax.plot_surface(S_grid, T_grid, grid, cmap=cmap, alpha=0.9)
        ax.set_xlabel("Stock Price ($)")
        ax.set_ylabel("Time to Maturity (yr)")
        ax.set_zlabel(label)
        ax.set_title(f"Asian {label} Surface", fontweight="bold")
        fig.colorbar(surf, ax=ax, shrink=0.5)

    out = Path("outputs/greeks_surface.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show() if show_plot else plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exotic Options Pricing Engine")
    parser.add_argument("--n-simulations", type=int, default=100_000)
    parser.add_argument("--grid-size",     type=int, default=8)
    parser.add_argument("--show",          action="store_true")
    parser.add_argument("--greeks",        action="store_true",
                        help="Also generate Greeks surface.")
    args = parser.parse_args()

    engine = MonteCarloEngine(S0=450, K=460, T=0.25, r=0.05, sigma=0.20,
                              n_simulations=args.n_simulations, n_steps=63,
                              random_seed=0)

    print("=" * 55)
    print("  EXOTIC OPTIONS PRICING")
    print("=" * 55)

    bs                   = engine.black_scholes_call()
    asian_c, asian_c_se  = engine.price_asian(option_type="call")
    asian_p, asian_p_se  = engine.price_asian(option_type="put")
    barrier, barrier_se  = engine.price_barrier(barrier=470)

    print(f"Black-Scholes Call:              ${bs:.4f}")
    print(f"Asian Call (arithmetic, CV):     ${asian_c:.4f} ± ${asian_c_se:.4f}")
    print(f"Asian Put  (arithmetic, CV):     ${asian_p:.4f} ± ${asian_p_se:.4f}")
    print(f"Barrier Call (up-and-out):       ${barrier:.4f} ± ${barrier_se:.4f}")

    out = Path("outputs/pricing_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Option Type": [
            "Black-Scholes Call",
            "Asian Call (arithmetic, CV)",
            "Asian Put  (arithmetic, CV)",
            "Barrier Call (Up-and-Out)",
        ],
        "Price":     [bs, asian_c, asian_p, barrier],
        "Std Error": [0,  asian_c_se, asian_p_se, barrier_se],
    }).to_csv(out, index=False)
    print(f"\nResults saved to {out}")

    if args.greeks:
        plot_greeks_surface(grid_size=args.grid_size,
                            n_simulations=args.n_simulations,
                            show_plot=args.show)
