import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from xp3_forex.indicators.regime import get_market_regime, RegimeConfig
from xp3_forex.strategies.adaptive_ema_rsi import MarketRegime
from xp3_forex.core.models import TradeSignal, Position

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XP3.Verify")

def test_regime_classification():
    logger.info("Testing Regime Classification v5.1...")
    
    # 1. Trend Case (Hurst > 0.5) - Brownian Motion with Drift
    n = 1000
    steps = np.random.normal(0.0005, 0.01, n) # Drift 0.0005
    data_trend = pd.Series(100 * np.exp(np.cumsum(steps)))
    
    config = RegimeConfig(threshold_trend=0.55, threshold_range_min=0.40)
    res_trend = get_market_regime(data_trend, config)
    logger.info(f"Trend Data -> Hurst: {res_trend['hurst']:.4f}, Regime: {res_trend['regime']}")
    # Assert trend or sideways (Hurst > 0.45 usually for drift)
    assert res_trend['hurst'] > 0.45
    
    # 2. Sideways Case (Mean Reversion)
    steps_mr = np.random.normal(0, 0.01, n)
    # Simple mean reversion: x_t = 0.9 * x_{t-1} + noise
    x = [100.0]
    for s in steps_mr:
        x.append(100.0 + 0.5 * (x[-1] - 100.0) + s)
    data_side = pd.Series(x)
    res_side = get_market_regime(data_side, config)
    logger.info(f"Sideways Data -> Hurst: {res_side['hurst']:.4f}, Regime: {res_side['regime']}")

    # 3. Protection Case (Hurst < 0.40 - Anti-persistent)
    # Alternating steps
    steps_noise = []
    last = 1
    for _ in range(n):
        last = -last + np.random.normal(0, 0.1)
        steps_noise.append(last)
    data_noise = pd.Series(100 + np.cumsum(steps_noise))
    res_noise = get_market_regime(data_noise, config)
    logger.info(f"Noise Data -> Hurst: {res_noise['hurst']:.4f}, Regime: {res_noise['regime']}")
    assert res_noise['hurst'] < 0.45

def test_model_updates():
    logger.info("Testing Model Updates...")
    sig = TradeSignal(
        symbol="EURUSD",
        order_type="BUY",
        entry_price=1.1000,
        stop_loss=1.0900,
        take_profit=1.1200,
        volume=0.1,
        confidence=0.8,
        reason="Test",
        regime=MarketRegime.TREND
    )
    assert sig.regime == MarketRegime.TREND
    
    pos = Position(
        symbol="EURUSD",
        order_type="BUY",
        volume=0.1,
        entry_price=1.1000,
        current_price=1.1050,
        stop_loss=1.0900,
        take_profit=1.1200,
        profit=50.0,
        pips=50,
        open_time=datetime.now(),
        magic_number=123,
        regime=MarketRegime.TREND
    )
    assert pos.regime == MarketRegime.TREND

if __name__ == "__main__":
    try:
        test_regime_classification()
        test_model_updates()
        logger.info("✅ All verifications passed!")
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        sys.exit(1)
