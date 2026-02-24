"""
Heston Stochastic Volatility Model Calibration
Author: Matthew Bowers
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize


class HestonModel:
    def __init__(self, S0, v0, kappa, theta, sigma_v, rho, r):
        """
        Initialize Heston model parameters.

        Parameters:
        -----------
        S0 : float
            Initial stock price
        v0 : float
            Initial variance
        kappa : float
            Mean reversion speed
        theta : float
            Long-term variance
        sigma_v : float
            Volatility of volatility
        rho : float
            Correlation between stock and volatility
        r : float
            Risk-free rate
        """
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.r = r

    def characteristic_function(self, u, T):
        """
        Heston characteristic function for option pricing.
        """
        u = np.asarray(u, dtype=np.complex128)
        sigma_sq = self.sigma_v**2
        i_u = 1j * u
        b = self.kappa - self.rho * self.sigma_v * i_u
        d = np.sqrt(b**2 + sigma_sq * (u**2 + i_u))

        eps = 1e-12
        denominator = b + d
        denominator = np.where(np.abs(denominator) < eps, denominator + eps, denominator)
        g = (b - d) / denominator

        exp_term = np.exp(-d * T)
        one_minus_g_exp = 1 - g * exp_term
        one_minus_g = 1 - g

        one_minus_g_exp = np.where(
            np.abs(one_minus_g_exp) < eps, one_minus_g_exp + eps, one_minus_g_exp
        )
        one_minus_g = np.where(np.abs(one_minus_g) < eps, one_minus_g + eps, one_minus_g)

        C = i_u * self.r * T + (self.kappa * self.theta / sigma_sq) * (
            (b - d) * T - 2 * np.log(one_minus_g_exp / one_minus_g)
        )
        D = ((b - d) / sigma_sq) * ((1 - exp_term) / one_minus_g_exp)

        return np.exp(C + D * self.v0 + i_u * np.log(self.S0))

    def price_call(self, K, T, integration_limit=100.0):
        """
        Price European call using the Lewis (2001) Fourier representation.
        """
        if K <= 0 or T <= 0:
            return np.nan

        log_moneyness = np.log(self.S0 / K)

        def integrand(u):
            shifted_cf = self.characteristic_function(u - 0.5j, T)
            numerator = np.exp(1j * u * log_moneyness) * shifted_cf
            return np.real(numerator / (u**2 + 0.25))

        integral, _ = quad(integrand, 0.0, integration_limit, limit=250)
        price = self.S0 - (np.sqrt(self.S0 * K) * np.exp(-self.r * T) / np.pi) * integral

        return float(max(price, 0.0))

    def calibrate_to_market(self, market_data):
        """
        Calibrate Heston model parameters to market option prices.

        Parameters:
        -----------
        market_data : DataFrame
            Columns: ['Strike', 'Maturity', 'MarketPrice']

        Returns:
        --------
        params : dict
            Calibrated parameters
        rmse : float
            Root mean squared error
        """

        def objective(params):
            v0, kappa, theta, sigma_v, rho = params

            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1:
                return 1e10

            if 2 * kappa * theta < sigma_v**2:
                return 1e10

            model = HestonModel(self.S0, v0, kappa, theta, sigma_v, rho, self.r)

            errors = []
            for _, row in market_data.iterrows():
                model_price = model.price_call(row["Strike"], row["Maturity"])
                if not np.isfinite(model_price):
                    return 1e10
                errors.append((model_price - row["MarketPrice"]) ** 2)

            return np.sqrt(np.mean(errors))

        x0 = [self.v0, 2.0, 0.04, 0.3, -0.7]
        bounds = [
            (0.001, 1.0),
            (0.1, 10.0),
            (0.001, 1.0),
            (0.01, 2.0),
            (-0.99, 0.99),
        ]

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        if not result.success:
            raise RuntimeError(f"Heston calibration failed: {result.message}")

        v0_cal, kappa_cal, theta_cal, sigma_v_cal, rho_cal = result.x

        return {
            "v0": v0_cal,
            "kappa": kappa_cal,
            "theta": theta_cal,
            "sigma_v": sigma_v_cal,
            "rho": rho_cal,
        }, result.fun


if __name__ == "__main__":
    market_data = pd.read_csv("data/sample_market_data.csv")

    model = HestonModel(
        S0=450,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7,
        r=0.05,
    )

    print("=" * 60)
    print("HESTON MODEL CALIBRATION")
    print("=" * 60)

    params, rmse = model.calibrate_to_market(market_data)

    print("\nCalibrated Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nRMSE: ${rmse:.4f}")
    print("\n" + "=" * 60)
