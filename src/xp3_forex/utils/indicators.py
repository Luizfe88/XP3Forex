"""Technical indicators for XP3 PRO FOREX"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Optional

@njit
def ema_numba(x, period):
    """Calcula EMA usando Numba para performance"""
    alpha = 2.0 / (period + 1.0)
    result = np.empty_like(x)
    if len(x) == 0:
        return result
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1.0 - alpha) * result[i - 1]
    return result

@njit
def calculate_rsi_numba(close, period=14):
    """Calcula RSI usando Numba para performance"""
    rsi = np.zeros_like(close)
    gains = np.zeros_like(close)
    losses = np.zeros_like(close)
    
    for i in range(1, len(close)):
        change = close[i] - close[i - 1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = abs(change)
    
    avg_gain = 0.0
    avg_loss = 0.0
    
    for i in range(period):
        avg_gain += gains[i]
        avg_loss += losses[i]
    
    avg_gain /= period
    avg_loss /= period
    
    for i in range(period, len(close)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calcula Exponential Moving Average"""
    return pd.Series(ema_numba(data.values, period), index=data.index)

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calcula Relative Strength Index"""
    return pd.Series(calculate_rsi_numba(data.values, period), index=data.index)

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calcula Average Directional Index"""
    if len(high) < period + 1:
        return pd.Series([np.nan] * len(high), index=high.index)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=high.index)
    
    # Calculate +DM and -DM
    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    
    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0), index=high.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0), index=high.index)
    
    # Calculate smoothed TR, +DM, -DM
    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calcula Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=high.index)
    
    return tr.rolling(window=period).mean()

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Calcula Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    
    return k_percent, d_percent

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula MACD"""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram