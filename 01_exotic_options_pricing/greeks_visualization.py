"""
Greeks Surface Visualization
Author: Matthew Bowers
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monte_carlo_engine import MonteCarloEngine


def calculate_delta(engine, dS=1.0, n_simulations=10000):
    """Calculate Delta using finite difference."""
    engine_up = MonteCarloEngine(
        engine.S0 + dS,
        engine.K,
        engine.T,
        engine.r,
        engine.sigma,
        n_simulations=n_simulations,
    )
    engine_down = MonteCarloEngine(
        engine.S0 - dS,
        engine.K,
        engine.T,
        engine.r,
        engine.sigma,
        n_simulations=n_simulations,
    )

    price_up, _ = engine_up.price_asian_call()
    price_down, _ = engine_down.price_asian_call()

    return (price_up - price_down) / (2 * dS)


def calculate_gamma(engine, dS=1.0, n_simulations=10000):
    """Calculate Gamma using finite difference."""
    engine_up = MonteCarloEngine(
        engine.S0 + dS,
        engine.K,
        engine.T,
        engine.r,
        engine.sigma,
        n_simulations=n_simulations,
    )
    engine_down = MonteCarloEngine(
        engine.S0 - dS,
        engine.K,
        engine.T,
        engine.r,
        engine.sigma,
        n_simulations=n_simulations,
    )
    engine_mid = engine

    price_up, _ = engine_up.price_asian_call()
    price_down, _ = engine_down.price_asian_call()
    price_mid, _ = engine_mid.price_asian_call()

    return (price_up - 2 * price_mid + price_down) / (dS**2)


def plot_greeks_surface(grid_size=8, n_simulations=10000, show_plot=False):
    """Generate 3D surface plots for Delta and Gamma."""
    S_range = np.linspace(400, 500, grid_size)
    T_range = np.linspace(0.1, 1.0, grid_size)

    S_grid, T_grid = np.meshgrid(S_range, T_range)
    delta_grid = np.zeros_like(S_grid)
    gamma_grid = np.zeros_like(S_grid)

    print("Calculating Greeks surface...")

    for i in range(len(T_range)):
        for j in range(len(S_range)):
            engine = MonteCarloEngine(
                S0=S_range[j],
                K=450,
                T=T_range[i],
                r=0.05,
                sigma=0.20,
                n_simulations=n_simulations,
            )
            delta_grid[i, j] = calculate_delta(engine, n_simulations=n_simulations)
            gamma_grid[i, j] = calculate_gamma(engine, n_simulations=n_simulations)

        print(f"Progress: {((i + 1) / len(T_range) * 100):.0f}%")

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(S_grid, T_grid, delta_grid, cmap="viridis", alpha=0.9)
    ax1.set_xlabel("Stock Price ($)", fontsize=10)
    ax1.set_ylabel("Time to Maturity (years)", fontsize=10)
    ax1.set_zlabel("Delta", fontsize=10)
    ax1.set_title("Delta Surface", fontsize=12, fontweight="bold")
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(S_grid, T_grid, gamma_grid, cmap="plasma", alpha=0.9)
    ax2.set_xlabel("Stock Price ($)", fontsize=10)
    ax2.set_ylabel("Time to Maturity (years)", fontsize=10)
    ax2.set_zlabel("Gamma", fontsize=10)
    ax2.set_title("Gamma Surface", fontsize=12, fontweight="bold")
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    output_path = Path("outputs/greeks_surface.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("\nGreeks surface plot saved to outputs/greeks_surface.png")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Delta/Gamma Greeks surfaces.")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid size per axis.")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10000,
        help="Monte Carlo paths used per pricing call.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot window after saving figure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_greeks_surface(
        grid_size=args.grid_size,
        n_simulations=args.n_simulations,
        show_plot=args.show,
    )
