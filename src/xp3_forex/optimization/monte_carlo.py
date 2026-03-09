"""
XP3 PRO FOREX - Monte Carlo Validation
Implements Reshuffle/Resample simulations and p-value validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """Calculates the Sortino Ratio (Downside deviation only)"""
    if len(returns) < 2:
        return 0.0
    
    expected_return = np.mean(returns)
    downside_rets = returns[returns < target_return]
    
    if len(downside_rets) == 0:
        return float('inf') if expected_return > target_return else 0.0
        
    downside_std = np.std(downside_rets)
    if downside_std == 0:
        return 0.0
        
    return (expected_return - target_return) / downside_std

class MonteCarloSimulator:
    def __init__(self, iterations: int = 5000):
        self.iterations = iterations

    def run_simulation(self, trade_returns: np.ndarray) -> Dict[str, Any]:
        """
        Runs Monte Carlo simulations on trade returns.
        Types:
        1. Reshuffle: Randomize order of trades.
        2. Resample: Bootstrap sampling with replacement.
        """
        n = len(trade_returns)
        if n < 10:
            return {"error": "Insufficient trades for Monte Carlo"}

        original_sortino = calculate_sortino_ratio(trade_returns)
        original_equity = np.cumsum(trade_returns)
        
        sim_sortinos = []
        sim_max_drawdowns = []
        
        for _ in range(self.iterations):
            # Bootstrap Resampling (with replacement)
            sim_rets = np.random.choice(trade_returns, size=n, replace=True)
            
            sim_sortinos.append(calculate_sortino_ratio(sim_rets))
            
            # Max Drawdown for simulation
            equity = np.cumsum(sim_rets) + 10000 # Dummy initial capital
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            sim_max_drawdowns.append(np.max(dd))
            
        # Calculate p-value: probability that a random reshuffle/resample yields better results
        # A low p-value (< 0.05) indicates the strategy is robust.
        sim_sortinos = np.sort(np.array(sim_sortinos))
        p_value = 1.0 - (np.searchsorted(sim_sortinos, original_sortino) / self.iterations)
        
        return {
            "original_sortino": original_sortino,
            "p_value": p_value,
            "mean_sim_sortino": np.mean(sim_sortinos),
            "median_sim_sortino": np.median(sim_sortinos),
            "confidence_interval_sortino": (np.percentile(sim_sortinos, 5), np.percentile(sim_sortinos, 95)),
            "max_dd_95_percentile": np.percentile(sim_max_drawdowns, 95),
            "pass_validation": p_value < 0.05
        }

def validate_model_robustness(trade_history: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to validate model using Monte Carlo.
    Expects DataFrame with a 'profit' column.
    """
    if 'profit' not in trade_history.columns:
        return {"error": "DataFrame must contain 'profit' column"}
        
    returns = trade_history['profit'].values
    simulator = MonteCarloSimulator()
    return simulator.run_simulation(returns)
