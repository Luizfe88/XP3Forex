"""
XP3 PRO FOREX - Adaptive Kalman Filter
Implements a 2D Kalman Filter (Price, Velocity) with dynamic H-based calibration.
"""

import numpy as np
import pandas as pd
from numba import njit
from pydantic import BaseModel, Field

class KalmanConfig(BaseModel):
    """Configuration for Kalman Filter"""
    initial_r: float = Field(default=500.0, description="Measurement Variance (R)")
    min_q: float = Field(default=0.01, description="Min Process Variance (Q) for mean reversion")
    max_q: float = Field(default=0.1, description="Max Process Variance (Q) for strong trends")

@njit
def adaptive_kalman_filter_numba(prices: np.ndarray, hurst_values: np.ndarray, r_val: float, q_min: float, q_max: float) -> np.ndarray:
    """
    2D Adaptive Kalman Filter.
    State vector x = [price, velocity]
    Adjusts Q based on Hurst Exponent.
    """
    n = len(prices)
    filtered_prices = np.zeros(n)
    
    if n == 0:
        return filtered_prices

    # Initial state (2D column vector)
    x = np.zeros((2, 1))
    x[0, 0] = prices[0]
    
    # Initial covariance matrix
    p = np.eye(2) * 1000.0
    
    # State transition matrix
    f = np.zeros((2, 2))
    f[0, 0] = 1.0
    f[0, 1] = 1.0
    f[1, 1] = 1.0
    
    # Measurement matrix
    h = np.zeros((1, 2))
    h[0, 0] = 1.0
    
    # Measurement noise covariance
    r = np.zeros((1, 1))
    r[0, 0] = r_val
    
    for i in range(n):
        h_val = hurst_values[i]
        
        if h_val > 0.65:
            q_scalar = q_max
        elif h_val < 0.45:
            q_scalar = q_min
        else:
            ratio = (h_val - 0.45) / (0.65 - 0.45)
            ratio = max(0.0, min(1.0, ratio))
            q_scalar = q_min + ratio * (q_max - q_min)
            
        # Process noise covariance
        q = np.eye(2) * q_scalar
        
        # 2. Predict
        x = f @ x
        p = f @ p @ f.T + q
        
        # 3. Update
        z = np.zeros((1, 1))
        z[0, 0] = prices[i]
        
        y = z - h @ x 
        s = h @ p @ h.T + r 
        k = p @ h.T @ np.linalg.inv(s) 
        
        x = x + k @ y
        p = (np.eye(2) - k @ h) @ p
        
        filtered_prices[i] = x[0, 0]
        
    return filtered_prices

class KalmanFilter:
    """Wrapper for the Kalman Filter calculation"""
    def __init__(self, config: KalmanConfig = KalmanConfig()):
        self.config = config
        
    def apply(self, prices: pd.Series, hurst_series: pd.Series) -> pd.Series:
        """
        Applies the adaptive filter to a series of prices.
        Expects a series of Hurst values corresponding to each price.
        """
        # Ensure we have aligned arrays
        p_vals = prices.values
        h_vals = hurst_series.values
        
        filtered = adaptive_kalman_filter_numba(
            p_vals, 
            h_vals, 
            self.config.initial_r,
            self.config.min_q,
            self.config.max_q
        )
        
        return pd.Series(filtered, index=prices.index)
