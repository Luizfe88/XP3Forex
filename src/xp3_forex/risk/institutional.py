"""
XP3 PRO FOREX - Institutional Risk Management
Implements Fractional Kelly, Covariance-adjusted exposure, and Cornish-Fisher CVaR.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple

class RiskConfig(BaseModel):
    """Configuration for Risk Management"""
    kelly_fraction: float = Field(default=0.25, description="Fractional Kelly (e.g., 0.25 for Quarter Kelly)")
    confidence_level: float = Field(default=0.99, description="Confidence level for CVaR (99%)")
    max_portfolio_cvar_pct: float = Field(default=0.015, description="MPE: Max CVaR as % of portfolio (1.5%)")
    hurst_noise_range: Tuple[float, float] = Field(default=(0.45, 0.55), description="Hurst range considered random noise")

def calculate_cornish_fisher_z(alpha: float, skewness: float, kurtosis: float) -> float:
    """
    Adjusts the standard normal Z-score using Cornish-Fisher expansion for skewness and kurtosis.
    """
    z = norm.ppf(alpha)
    
    # Cornish-Fisher Expansion
    # z_adj = z + (z^2 - 1)S/6 + (z^3 - 3z)K/24 - (2z^3 - 5z)S^2/36
    s = skewness
    k = kurtosis - 3 # Excess kurtosis
    
    z_adj = (z + 
             (z**2 - 1) * s / 6 + 
             (z**3 - 3*z) * k / 24 - 
             (2*z**3 - 5*z) * s**2 / 36)
    
    return z_adj

def calculate_cvar_cornish_fisher(returns: np.ndarray, alpha: float = 0.99) -> float:
    """
    Calculates Conditional Value at Risk (CVaR) using Cornish-Fisher expansion.
    """
    if len(returns) < 30:
        # Fallback to simple historical CVaR if not enough data
        var = np.percentile(returns, (1 - alpha) * 100)
        return abs(np.mean(returns[returns <= var])) if any(returns <= var) else 0.0

    mu = np.mean(returns)
    sigma = np.std(returns)
    
    # Calculate Skewness and Kurtosis
    n = len(returns)
    skew = np.sum((returns - mu)**3) / (n * sigma**3)
    kurt = np.sum((returns - mu)**4) / (n * sigma**4)
    
    # Modified Z-score for VaR
    z_cf = calculate_cornish_fisher_z(1 - alpha, skew, kurt)
    
    # Cornish-Fisher VaR (as a return value)
    var_cf = mu + z_cf * sigma
    
    # CVaR is the expectation of losses exceeding VaR
    # Approximate CVaR for CF: Since we don't have a closed-form CDF for CF, 
    # we use the historical returns that exceed the CF VaR.
    losses_beyond_var = returns[returns <= var_cf]
    
    if len(losses_beyond_var) > 0:
        cvar = abs(np.mean(losses_beyond_var))
    else:
        # Theoretical approximation if no data points exceed (rare)
        cvar = abs(var_cf) * 1.1 
        
    return cvar

def calculate_fractional_kelly(win_rate: float, win_loss_ratio: float, hurst: float, config: RiskConfig) -> float:
    """
    Calculates Fractional Kelly sizing with a noise penalty.
    """
    if win_loss_ratio <= 0:
        return 0.0
        
    # Standard Kelly Criterion: f* = p - (1-p)/b = (p*(b+1) - 1) / b
    p = win_rate
    b = win_loss_ratio
    kelly_f = (p * (b + 1) - 1) / b
    
    if kelly_f <= 0:
        return 0.0
        
    # Apply primary fraction (Half, Quarter, etc.)
    base_f = kelly_f * config.kelly_fraction
    
    # Noise Penalty: if 0.45 < H < 0.55, reduce size significantly
    h_min, h_max = config.hurst_noise_range
    if h_min < hurst < h_max:
        # Linear penalty: deepest at H=0.5
        dist = abs(hurst - 0.5) / 0.05 # 0 to 1
        penalty = 0.2 + 0.8 * dist # Max 80% reduction at H=0.5
        base_f *= penalty
        
    return max(0.0, base_f)

class InstitutionalRiskManager:
    def __init__(self, config: RiskConfig = RiskConfig()):
        self.config = config
        
    def validate_exposure(self, portfolio_returns: Dict[str, np.ndarray], current_equity: float) -> Tuple[bool, float]:
        """
        Validates MPE (Maximum Permissible Exposure) using Portfolio CVaR.
        portfolio_returns: Dict of {symbol: return_array}
        Returns (is_passed, multiplier). multiplier < 1.0 means downsizing required.
        """
        if not portfolio_returns:
            return True, 1.0
            
        # 1. Combine returns for portfolio CVaR (assuming equal weighting or provided)
        # For simplicity, we aggregate and check total volatility/cvar
        all_rets = []
        for sym, rets in portfolio_returns.items():
            all_rets.append(rets)
            
        # Minimum length check
        min_len = min(len(r) for r in all_rets)
        combined_returns = np.zeros(min_len)
        for r in all_rets:
            combined_returns += r[:min_len] / len(all_rets)
            
        p_cvar = calculate_cvar_cornish_fisher(combined_returns, self.config.confidence_level)
        
        # MPE Kill Switch check
        limit = self.config.max_portfolio_cvar_pct
        if p_cvar > limit:
            # Calculate downsizing factor
            multiplier = limit / p_cvar
            return False, multiplier
            
        return True, 1.0

    def get_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates correlation to avoid directional bias"""
        return returns_df.corr()
