"""
Monte Carlo Simulation Engine for Exotic Options Pricing
Author: Matthew Bowers
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm


class MonteCarloEngine:
    def __init__(
        self,
        S0,
        K,
        T,
        r,
        sigma,
        n_simulations=100000,
        n_steps=252,
        random_seed=None,
    ):
        """
        Initialize Monte Carlo engine.

        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        n_simulations : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps
        """
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be positive.")
        if T <= 0:
            raise ValueError("T must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if n_simulations < 2:
            raise ValueError("n_simulations must be at least 2.")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

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
        """
        Generate stock price paths using Geometric Brownian Motion.

        Parameters:
        -----------
        variance_reduction : bool
            Use antithetic variates for variance reduction

        Returns:
        --------
        paths : ndarray
            Array of shape (n_simulations, n_steps+1) with price paths
        """
        n_draws = self.n_simulations
        if variance_reduction:
            n_draws = (self.n_simulations + 1) // 2

        Z = self.rng.standard_normal(
            (
                n_draws,
                self.n_steps,
            )
        )

        if variance_reduction:
            Z = np.concatenate([Z, -Z], axis=0)
            Z = Z[: self.n_simulations]

        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S0

        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt
                + self.sigma * np.sqrt(self.dt) * Z[:, t - 1]
            )

        return paths

    def price_asian_call(self, averaging="arithmetic"):
        """
        Price Asian call option.

        Parameters:
        -----------
        averaging : str
            'arithmetic' or 'geometric' averaging

        Returns:
        --------
        price : float
            Option price
        std_error : float
            Standard error of estimate
        """
        paths = self.generate_paths()

        if averaging == "arithmetic":
            avg_prices = np.mean(paths, axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))

        payoffs = np.maximum(avg_prices - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(self.n_simulations)

        return price, std_error

    def price_barrier_call(self, barrier, barrier_type="up-and-out"):
        """
        Price barrier call option.

        Parameters:
        -----------
        barrier : float
            Barrier level
        barrier_type : str
            'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'

        Returns:
        --------
        price : float
            Option price
        std_error : float
            Standard error of estimate
        """
        paths = self.generate_paths()
        final_prices = paths[:, -1]

        if barrier_type == "up-and-out":
            barrier_hit = np.any(paths >= barrier, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(final_prices - self.K, 0))
        elif barrier_type == "down-and-out":
            barrier_hit = np.any(paths <= barrier, axis=1)
            payoffs = np.where(barrier_hit, 0, np.maximum(final_prices - self.K, 0))
        elif barrier_type == "up-and-in":
            barrier_hit = np.any(paths >= barrier, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(final_prices - self.K, 0), 0)
        else:
            barrier_hit = np.any(paths <= barrier, axis=1)
            payoffs = np.where(barrier_hit, np.maximum(final_prices - self.K, 0), 0)

        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(self.n_simulations)

        return price, std_error

    def black_scholes_call(self):
        """
        Calculate Black-Scholes call option price for comparison.

        Returns:
        --------
        price : float
            Black-Scholes call price
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)

        price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return price


if __name__ == "__main__":
    engine = MonteCarloEngine(
        S0=450,
        K=460,
        T=0.25,
        r=0.05,
        sigma=0.20,
        n_simulations=100000,
        n_steps=63,
    )

    print("=" * 60)
    print("EXOTIC OPTIONS PRICING ENGINE")
    print("=" * 60)

    bs_price = engine.black_scholes_call()
    print(f"\nBlack-Scholes Call Price: ${bs_price:.4f}")

    asian_price, asian_se = engine.price_asian_call(averaging="arithmetic")
    print("\nAsian Call (Arithmetic Average):")
    print(f"  Price: ${asian_price:.4f} ± ${asian_se:.4f}")

    barrier_price, barrier_se = engine.price_barrier_call(
        barrier=470, barrier_type="up-and-out"
    )
    print("\nUp-and-Out Barrier Call (Barrier=$470):")
    print(f"  Price: ${barrier_price:.4f} ± ${barrier_se:.4f}")

    print("\n" + "=" * 60)
    print("Results exported to outputs/pricing_comparison.csv")

    results = pd.DataFrame(
        {
            "Option Type": [
                "Black-Scholes Call",
                "Asian Call",
                "Barrier Call (Up-and-Out)",
            ],
            "Price": [bs_price, asian_price, barrier_price],
            "Std Error": [0, asian_se, barrier_se],
        }
    )
    output_path = Path("outputs/pricing_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
