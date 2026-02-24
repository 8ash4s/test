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
    def __init__(self, S0, K, T, r, sigma, n_simulations=100000, n_steps=252, random_seed=None):
        """
        Parameters
        ----------
        S0            : Initial stock price
        K             : Strike price
        T             : Time to maturity (years)
        r             : Risk-free rate
        sigma         : Volatility
        n_simulations : Number of Monte Carlo paths
        n_steps       : Number of time steps
        """
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be positive.")
        if T <= 0:
            raise ValueError("T must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if n_simulations < 2:
            raise ValueError("n_simulations must be at least 2.")

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.n_simulations = int(n_simulations)
        self.n_steps = int(n_steps)
        self.dt = T / n_steps
        self.rng = np.random.default_rng(random_seed)

    def generate_paths(self, variance_reduction=True):
        """Generate GBM stock price paths (antithetic variates optional)."""
        n = (self.n_simulations + 1) // 2 if variance_reduction else self.n_simulations
        Z = self.rng.standard_normal((n, self.n_steps))
        if variance_reduction:
            Z = np.concatenate([Z, -Z], axis=0)[: self.n_simulations]

        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S0
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])
        return paths

    def price_asian_call(self, averaging="arithmetic"):
        """Price an Asian call option. Returns (price, std_error)."""
        paths = self.generate_paths()
         # Exclude the fixed t=0 column (S0) — average over future prices only
        avg = np.mean(paths[:, 1:], axis=1) if averaging == "arithmetic" \
            else np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        payoffs = np.maximum(avg - self.K, 0)
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(self.n_simulations)
        return price, se

    def price_barrier_call(self, barrier, barrier_type="up-and-out"):
        """Price a barrier call option. Returns (price, std_error)."""
        paths = self.generate_paths()
        final = paths[:, -1]
        if barrier_type == "up-and-out":
            hit = np.any(paths >= barrier, axis=1)
            payoffs = np.where(hit, 0, np.maximum(final - self.K, 0))
        elif barrier_type == "down-and-out":
            hit = np.any(paths <= barrier, axis=1)
            payoffs = np.where(hit, 0, np.maximum(final - self.K, 0))
        elif barrier_type == "up-and-in":
            hit = np.any(paths >= barrier, axis=1)
            payoffs = np.where(hit, np.maximum(final - self.K, 0), 0)
        else:  # down-and-in
            hit = np.any(paths <= barrier, axis=1)
            payoffs = np.where(hit, np.maximum(final - self.K, 0), 0)
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(self.n_simulations)
        return price, se

    def black_scholes_call(self):
        """Analytical Black-Scholes call price."""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Greeks (finite difference)
# ---------------------------------------------------------------------------

def calculate_delta(engine, dS=1.0, n_simulations=10000):
    """Central-difference Delta."""
    kw = dict(K=engine.K, T=engine.T, r=engine.r, sigma=engine.sigma, n_simulations=n_simulations)
    up, _ = MonteCarloEngine(S0=engine.S0 + dS, **kw).price_asian_call()
    down, _ = MonteCarloEngine(S0=engine.S0 - dS, **kw).price_asian_call()
    return (up - down) / (2 * dS)


def calculate_gamma(engine, dS=1.0, n_simulations=10000):
    """Central-difference Gamma."""
    kw = dict(K=engine.K, T=engine.T, r=engine.r, sigma=engine.sigma, n_simulations=n_simulations)
    up, _ = MonteCarloEngine(S0=engine.S0 + dS, **kw).price_asian_call()
    mid, _ = engine.price_asian_call()
    down, _ = MonteCarloEngine(S0=engine.S0 - dS, **kw).price_asian_call()
    return (up - 2 * mid + down) / (dS**2)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_greeks_surface(grid_size=8, n_simulations=10000, show_plot=False):
    """Generate and save 3-D Delta/Gamma surface plots."""
    S_range = np.linspace(400, 500, grid_size)
    T_range = np.linspace(0.1, 1.0, grid_size)
    S_grid, T_grid = np.meshgrid(S_range, T_range)
    delta_grid, gamma_grid = np.zeros_like(S_grid), np.zeros_like(S_grid)

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
        ax.set_ylabel("Time to Maturity (years)")
        ax.set_zlabel(label)
        ax.set_title(f"{label} Surface", fontweight="bold")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-simulations", type=int, default=100000)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--greeks", action="store_true", help="Also generate Greeks surface.")
    args = parser.parse_args()

    engine = MonteCarloEngine(S0=450, K=460, T=0.25, r=0.05, sigma=0.20,
                              n_simulations=args.n_simulations, n_steps=63)

    print("=" * 50)
    print("EXOTIC OPTIONS PRICING")
    print("=" * 50)

    bs = engine.black_scholes_call()
    asian, asian_se = engine.price_asian_call()
    barrier, barrier_se = engine.price_barrier_call(barrier=470)

    print(f"Black-Scholes Call:          ${bs:.4f}")
    print(f"Asian Call (arithmetic):     ${asian:.4f} ± ${asian_se:.4f}")
    print(f"Barrier Call (up-and-out):   ${barrier:.4f} ± ${barrier_se:.4f}")

    out = Path("outputs/pricing_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Option Type": ["Black-Scholes Call", "Asian Call", "Barrier Call (Up-and-Out)"],
        "Price": [bs, asian, barrier],
        "Std Error": [0, asian_se, barrier_se],
    }).to_csv(out, index=False)
    print(f"\nResults saved to {out}")

    if args.greeks:
        plot_greeks_surface(grid_size=args.grid_size,
                            n_simulations=args.n_simulations,
                            show_plot=args.show)
