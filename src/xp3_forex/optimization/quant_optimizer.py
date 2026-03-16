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
        
    def objective(self, trial: optuna.Trial, regime_filter: str = "ALL") -> float:
        """
        Optuna objective function v5.1 with Regime Filtering.
        """
        # Suggest parameters
        hurst_lookback = trial.suggest_int("hurst_lookback", 300, 1000)
        initial_r = trial.suggest_float("initial_r", 100.0, 800.0)
        min_q = trial.suggest_float("min_q", 0.001, 0.04)
        max_q = trial.suggest_float("max_q", 0.05, 0.4)
        
        prices = self.data['close']
        sample_size = min(3000, len(prices))
        test_prices = prices.iloc[-sample_size:]
        
        # Pre-calculate Hurst for filtering
        # Optimized for speed: sample Hurst every 20 bars
        hurst_vals = []
        h = 0.5
        for i in range(len(test_prices)):
            if i % 20 == 0:
                h = calculate_hurst_rs(prices.iloc[-(sample_size-i+hurst_lookback):-(sample_size-i)].values)
            hurst_vals.append(h)
        
        hurst_series = pd.Series(hurst_vals, index=test_prices.index)
        
        # Apply Kalman
        kf_config = KalmanConfig(initial_r=initial_r, min_q=min_q, max_q=max_q)
        kf = KalmanFilter(kf_config)
        filtered_price = kf.apply(test_prices, hurst_series)
        
        # Strategy Signal
        signal = np.where(test_prices > filtered_price, 1, -1)
        signal = pd.Series(signal, index=test_prices.index).shift(1).fillna(0)
        
        price_rets = test_prices.pct_change().fillna(0)
        strat_rets = signal.values * price_rets.values
        
        # --- REGIME FILTERING ---
        if regime_filter == "TREND":
            # Only consider returns where Hurst > 0.55
            mask = hurst_series > 0.55
            if mask.sum() < 50: return -10.0
            strat_rets = strat_rets[mask.values]
        elif regime_filter == "SIDEWAYS":
            # Only consider returns where 0.40 <= Hurst <= 0.55
            mask = (hurst_series >= 0.40) & (hurst_series <= 0.55)
            if mask.sum() < 50: return -10.0
            strat_rets = strat_rets[mask.values]
            
        sortino = calculate_sortino_ratio(strat_rets)
        return sortino

    def calculate_hard_dollar_stop(self) -> float:
        """Calculates ideal Hard Dollar Stop based on weekly ATR"""
        if len(self.data) < 100: return 50.0
        
        # Use H1 ATR
        high = self.data.get('high', self.data['close'])
        low = self.data.get('low', self.data['close'])
        close = self.data['close']
        
        tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
        atr = tr.rolling(100).mean().iloc[-1]
        
        # Suggest stop based on 3x ATR converted to dollars (assuming 0.01 lot)
        # Simplified: $50 min or 3 * ATR_points * lot_const
        return max(50.0, atr * 1000) # Placeholder scaling

    def run_optimization(self, n_trials: int = 40) -> Dict[str, Any]:
        """Runs separate optimizations for Trend, Sideways and Protection"""
        logger.info(f"🚀 XP3 PRO v5.1 - Triple-Regime Optimization for {self.symbol}...")
        
        # 1. Trend Study
        study_trend = optuna.create_study(direction="maximize")
        study_trend.optimize(lambda t: self.objective(t, "TREND"), n_trials=n_trials)
        
        # 2. Sideways Study
        study_sideways = optuna.create_study(direction="maximize")
        study_sideways.optimize(lambda t: self.objective(t, "SIDEWAYS"), n_trials=n_trials)
        
        # 3. Dynamic Thresholds & Asset-level params
        # We take the best lookback from the trend study as the asset-level default
        best_hurst_lookback = study_trend.best_params.get("hurst_lookback", 500)
        
        results = {
            "hurst_lookback": best_hurst_lookback,
            "mmi_lookback": 200,
            "threshold_trend": 0.55,
            "threshold_range_min": 0.40,
            "hard_dollar_stop": self.calculate_hard_dollar_stop(),
            "last_calibrated": datetime.now().isoformat(),
            "Trend_Config": study_trend.best_params,
            "Sideways_Config": study_sideways.best_params,
            "Protection_Config": {
                "initial_r": 500.0,
                "min_q": 0.001,
                "max_q": 0.01 # Very slow adaptation for protection
            }
        }
        
        return results

def save_optimized_quant_params(symbol: str, results: Dict[str, Any]):
    """Saves results to quant_params.json with dual-regime support"""
    json_path = settings.DATA_DIR / "quant_optimized_params.json"
    
    data = {}
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            
    # Structure: symbol -> {Trend_Config: {}, Sideways_Config: {}, ...}
    data[symbol] = results
    
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        
    logger.info(f"✅ XP3 PRO v5.1 Params for {symbol} saved to {json_path}")

if __name__ == "__main__":
    from xp3_forex.utils.mt5_utils import get_rates, initialize_mt5
    from datetime import datetime
    
    logging.basicConfig(level=logging.INFO)
    
    if initialize_mt5(
        login=settings.MT5_LOGIN,
        password=settings.MT5_PASSWORD,
        server=settings.MT5_SERVER,
        path=settings.MT5_PATH
    ):
        for sym in ["EURUSD", "XAUUSD", "GBPUSD"]:
            logger.info(f"Processing {sym}...")
            df = get_rates(sym, 60, 4000) 
            if df is not None:
                optimizer = QuantOptimizer(sym, df)
                results = optimizer.run_optimization(n_trials=30)
                save_optimized_quant_params(sym, results)
