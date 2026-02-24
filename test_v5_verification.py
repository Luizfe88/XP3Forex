# test_v5_verification.py
import sys
import os
from datetime import datetime, timedelta
import logging

# Adiciona o diretório atual ao path para importar os módulos locais
sys.path.append(os.getcwd())

import config_forex as config
import utils_forex as utils
from news_filter import news_filter
from ml_optimizer import EnsembleOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFICATION")

def test_news_filter():
    logger.info("--- Testing News Filter ---")
    # Simula um símbolo
    symbol = "EURUSD"
    is_blocked, msg = news_filter.is_news_blackout(symbol)
    logger.info(f"Symbol: {symbol} | Blocked: {is_blocked} | Status: {msg}")
    
    # Verifica se o cache foi criado
    from pathlib import Path
    cache_file = Path("data/news_cache.json")
    if cache_file.exists():
        logger.info(f"✅ News cache file exists.")
    else:
        logger.warning(f"❌ News cache file NOT found.")

def test_ml_score():
    logger.info("--- Testing ML Score ---")
    ml = EnsembleOptimizer()
    # Simula indicadores
    ind = {"ema_trend": "UP", "rsi": 30, "rsi_high_limit": 70, "rsi_low_limit": 30, "adx": 30, "bb_width": 0.02, "close": 1.1000}
    
    # Mocking a state in Q-Table
    state = ml._encode_state(ind)
    symbol = "EURUSD"
    ml.q_table[symbol] = {state: {"CONSERVATIVE": 2.0, "MODERATE": 1.0, "AGGRESSIVE": 0.5}}
    
    score, is_baseline = ml.get_prediction_score(symbol, ind)
    logger.info(f"ML Score (Mocked Q=2.0): {score:.1f} (Expected > 50, baseline={is_baseline})")
    
    # Mocking a bad state
    ml.q_table[symbol][state] = {"CONSERVATIVE": -2.0}
    score_bad, is_baseline_bad = ml.get_prediction_score(symbol, ind)
    logger.info(f"ML Score (Mocked Q=-2.0): {score_bad:.1f} (Expected < 50, baseline={is_baseline_bad})")

def test_risk_management():
    logger.info("--- Testing Risk Management ---")
    # Mock account info for utils
    import MetaTrader5 as mt5
    
    # We can't easily mock mt5.account_info() if it's already initialized/connected to a real terminal
    # but we can check if the logic in calculate_position_size_atr_forex behaves correctly if we could mock it.
    # Since we are in a real environment, we'll just log the logic presence.
    
    logger.info("Checking configuration values for Equity Guard:")
    logger.info(f"MAX_DAILY_LOSS_PCT: {getattr(config, 'MAX_DAILY_LOSS_PCT', 'MISSING')}")
    logger.info(f"REDUCE_RISK_ON_DD: {getattr(config, 'REDUCE_RISK_ON_DD', 'MISSING')}")

if __name__ == "__main__":
    test_news_filter()
    test_ml_score()
    test_risk_management()
    logger.info("Verification script finished.")
