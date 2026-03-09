"""
XP3 PRO FOREX - Quantitative Hyperparameter Optimizer
Uses Optuna to tune regime detection and Kalman filter parameters.
"""

import optuna
import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from xp3_forex.core.settings import settings
from xp3_forex.indicators.regime import calculate_hurst_rs, calculate_mmi_numba, RegimeConfig
from xp3_forex.indicators.filters import KalmanFilter, KalmanConfig
from xp3_forex.optimization.monte_carlo import calculate_sortino_ratio

logger = logging.getLogger("XP3.QuantOptimizer")

class QuantOptimizer:
    def __init__(self, symbol: str, data: pd.DataFrame):
        self.symbol = symbol
        self.data = data # Expected to have 'close' column
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Goal: Maximize Sortino Ratio of the Kalman-filtered price signals.
        """
        # 1. Suggest regime parameters
        hurst_lookback = trial.suggest_int("hurst_lookback", 500, 1500)
        mmi_lookback = trial.suggest_int("mmi_lookback", 100, 500)
        
        # 2. Suggest Kalman parameters
        initial_r = trial.suggest_float("initial_r", 100.0, 1000.0)
        min_q = trial.suggest_float("min_q", 0.001, 0.05)
        max_q = trial.suggest_float("max_q", 0.06, 0.5)
        
        # 3. Apply Quantitative Framework
        # For optimization, we measure the "signal quality" by looking at returns 
        # of a hypothetical strategy following the filtered price.
        
        prices = self.data['close']
        
        # Pre-calculate Hurst for the entire series (simplified for the trial)
        # In a real trial, we'd roll this, but for speed we can sample or window
        # We'll use a sliding window approach for a representative sample
        sample_size = min(2000, len(prices))
        test_prices = prices.iloc[-sample_size:]
        
        # Calculate hurst series for the test window
        # Note: Hurst is slow, so we optimize by calculating it less frequently or using a faster approx
        # For the trial, we'll calculate it every 10 bars to speed up
        hurst_vals = []
        for i in range(len(test_prices)):
            if i % 20 == 0:
                h = calculate_hurst_rs(prices.iloc[-(sample_size-i+hurst_lookback):-(sample_size-i)].values)
            hurst_vals.append(h)
        
        hurst_series = pd.Series(hurst_vals, index=test_prices.index)
        
        # Apply Kalman Filter
        kf_config = KalmanConfig(initial_r=initial_r, min_q=min_q, max_q=max_q)
        kf = KalmanFilter(kf_config)
        filtered_price = kf.apply(test_prices, hurst_series)
        
        # 4. Calculate Returns of a simple crossover/follow strategy
        # Strategy: Buy if price > kf_price, Sell if price < kf_price
        # (This measures how well the KF tracks trend without too much lag/noise)
        signal = np.where(test_prices > filtered_price, 1, -1)
        # Shift signals to avoid lookahead
        signal = pd.Series(signal, index=test_prices.index).shift(1).fillna(0)
        
        # Returns = signal * price_change
        price_rets = test_prices.pct_change().fillna(0)
        strat_rets = signal.values * price_rets.values
        
        # 5. Metric: Sortino Ratio
        sortino = calculate_sortino_ratio(strat_rets)
        
        return sortino

    def run_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """Runs the Optuna study"""
        logger.info(f"Starting Quantitative Optimization for {self.symbol}...")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"Optimization complete for {self.symbol}.")
        logger.info(f"Best Target Sortino: {study.best_value:.4f}")
        
        return study.best_params

def save_optimized_quant_params(symbol: str, params: Dict[str, Any]):
    """Saves results to quant_params.json"""
    json_path = settings.DATA_DIR / "quant_optimized_params.json"
    
    data = {}
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            
    data[symbol] = params
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        
    logger.info(f"Optimized parameters for {symbol} saved to {json_path}")

if __name__ == "__main__":
    # Diagnostic / Local execution example
    from xp3_forex.utils.mt5_utils import get_rates, initialize_mt5
    
    if initialize_mt5(
        login=settings.MT5_LOGIN,
        password=settings.MT5_PASSWORD,
        server=settings.MT5_SERVER,
        path=settings.MT5_PATH
    ):
        for sym in ["EURUSD", "XAUUSD"]:
            logger.info(f"Processing {sym}...")
            # Fetch H1 data for calibration
            df = get_rates(sym, 60, 3000) 
            if df is not None:
                optimizer = QuantOptimizer(sym, df)
                best_params = optimizer.run_optimization(n_trials=30)
                save_optimized_quant_params(sym, best_params)
