"""
XP3 PRO FOREX - Quantitative Framework Diagnostic
Verifies the implementation of Hurst, MMI, Kalman, Kelly, CVaR and Monte Carlo.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xp3_forex.indicators.regime import calculate_hurst_rs, calculate_mmi_numba, RegimeConfig
from xp3_forex.indicators.filters import KalmanFilter, KalmanConfig
from xp3_forex.risk.institutional import calculate_cvar_cornish_fisher, calculate_fractional_kelly, RiskConfig
from xp3_forex.optimization.monte_carlo import MonteCarloSimulator

def run_diagnostic():
    print("=== XP3 Quantitative Framework Diagnostic ===\n")
    
    # 1. Test Regime Detection (Synthetic Trend)
    print("1. Testing Regime Detection...")
    trend_data = np.linspace(100, 200, 1000) + np.random.normal(0, 1, 1000)
    random_data = 100 + np.random.normal(0, 5, 1000)
    
    h_trend = calculate_hurst_rs(trend_data)
    h_random = calculate_hurst_rs(random_data)
    mmi_trend = calculate_mmi_numba(trend_data)
    
    print(f"   Hurst (Trend): {h_trend:.4f} (Expected > 0.5)")
    print(f"   Hurst (Random): {h_random:.4f} (Expected ~ 0.5)")
    print(f"   MMI (Trend): {mmi_trend:.2f}")
    
    # 2. Test Kalman Filter Adaptation
    print("\n2. Testing Adaptive Kalman Filter...")
    kf = KalmanFilter(KalmanConfig())
    hurst_series = pd.Series([h_trend] * 1000) # Simulating constant trend state
    price_series = pd.Series(trend_data)
    filtered = kf.apply(price_series, hurst_series)
    print(f"   Filtered Price Head: {filtered.head(3).values}")
    print(f"   Last Filtered Price: {filtered.iloc[-1]:.4f} vs Real: {trend_data[-1]:.4f}")

    # 3. Test Institutional Risk
    print("\n3. Testing Risk Management...")
    # Generate some periodic returns for CVaR
    rets = np.random.normal(0.0001, 0.01, 500)
    rets[10:15] = -0.05 # Add some outliers
    cvar = calculate_cvar_cornish_fisher(rets)
    print(f"   Cornish-Fisher CVaR (99%): {cvar*100:.2f}%")
    
    k_size = calculate_fractional_kelly(0.55, 1.5, h_trend, RiskConfig())
    print(f"   Fractional Kelly Size (H={h_trend:.2f}): {k_size*100:.2f}%")
    
    k_size_noise = calculate_fractional_kelly(0.55, 1.5, 0.5, RiskConfig())
    print(f"   Fractional Kelly Size (H=0.50): {k_size_noise*100:.2f}% (Expected reduction)")

    # 4. Test Monte Carlo
    print("\n4. Testing Monte Carlo Validation...")
    simulator = MonteCarloSimulator(iterations=1000)
    # Generate a "profitable but lucky" trade set
    trades = np.random.normal(0.005, 0.02, 50) 
    mc_results = simulator.run_simulation(trades)
    print(f"   Original Sortino: {mc_results['original_sortino']:.4f}")
    print(f"   Simulation p-value: {mc_results['p_value']:.4f}")
    print(f"   Passes Validation: {mc_results['pass_validation']}")

    print("\n=== Diagnostic Complete ===")

if __name__ == "__main__":
    run_diagnostic()
