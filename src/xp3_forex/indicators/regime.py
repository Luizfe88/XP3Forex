"""
XP3 PRO FOREX - Market Regime Detection
Implements Hurst Exponent and Market Meanness Index (MMI).
"""

import numpy as np
import pandas as pd
from numba import njit
from pydantic import BaseModel, Field
from typing import Tuple, Dict

class RegimeConfig(BaseModel):
    """Hyperparameters for Regime Detection"""
    hurst_lookback: int = Field(default=500, ge=100, le=2000)
    mmi_lookback: int = Field(default=200, ge=50, le=500)
    trend_threshold: float = Field(default=0.65, description="Hurst > trend_threshold => Trending")
    mean_reversion_threshold: float = Field(default=0.45, description="Hurst < mean_reversion_threshold => Mean Reverting")

@njit
def _hurst_rs_values(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper for Hurst R/S calculation to avoid np.polyfit in njit"""
    n = len(returns)
    max_k = int(np.floor(np.log2(n)))
    rs_values = []
    n_values = []
    
    for k in range(4, max_k + 1):
        window_size = 2**k
        num_windows = n // window_size
        
        rs_list = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            segment = returns[start:end]
            
            mean_adj = segment - np.mean(segment)
            cum_sum = np.cumsum(mean_adj)
            
            r = np.max(cum_sum) - np.min(cum_sum)
            s = np.std(segment)
            
            if s > 0:
                rs_list.append(r / s)
        
        if rs_list:
            rs_values.append(np.mean(np.array(rs_list)))
            n_values.append(float(window_size))
            
    return np.array(rs_values), np.array(n_values)

def calculate_hurst_rs(series: np.ndarray) -> float:
    """
    Calculates Hurst Exponent using Rescaled Range (R/S) Analysis.
    """
    if len(series) < 100:
        return 0.5
    
    returns = np.diff(np.log(series))
    rs_values, n_values = _hurst_rs_values(returns)
    
    if len(rs_values) < 2:
        return 0.5
        
    m, _ = np.polyfit(np.log(n_values), np.log(rs_values), 1)
    return m

@njit
def calculate_mmi_numba(series: np.ndarray) -> float:
    """
    Calculates Market Meanness Index (MMI).
    MMI measures how many price changes are in the same direction.
    A value < 75 suggests trend, > 75 suggests noise/reversion (depending on normalization).
    Common use: MMI > 75 => False signals / High Noise.
    """
    n = len(series)
    if n < 2:
        return 50.0
        
    median_val = np.median(series)
    m = 0
    for i in range(1, n):
        if (series[i] > median_val and series[i] > series[i-1]) or \
           (series[i] < median_val and series[i] < series[i-1]):
            m += 1
            
    return 100.0 * (1.0 - (m / (n - 1)))

def get_market_regime(data: pd.Series, config: RegimeConfig) -> Dict[str, float]:
    """
    Helper to get all regime metrics.
    """
    vals = data.values
    hurst = calculate_hurst_rs(vals[-config.hurst_lookback:])
    mmi = calculate_mmi_numba(vals[-config.mmi_lookback:])
    
    regime = "Random"
    if hurst > config.trend_threshold:
        regime = "Trend"
    elif hurst < config.mean_reversion_threshold:
        regime = "Mean Reversion"
        
    return {
        "hurst": hurst,
        "mmi": mmi,
        "regime": regime
    }
