import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xp3_forex.core.settings import settings
from xp3_forex.core.bot import XP3Bot
from xp3_forex.optimization.monte_carlo import MonteCarloSimulator

def fetch_data(symbol, timeframe, n_candles=2000):
    """Fetch candles from MT5"""
    if timeframe == "M5":
        tf = mt5.TIMEFRAME_M5
    elif timeframe == "M15":
        tf = mt5.TIMEFRAME_M15
    elif timeframe == "M30":
        tf = mt5.TIMEFRAME_M30
    else:
        return None

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_candles)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def run_study(symbols=["EURUSD", "PETR4", "VALE3", "WIN$N"]):
    """Run comparison study"""
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    results = []
    simulator = MonteCarloSimulator(iterations=2000)

    for symbol in symbols:
        # Check if symbol exists
        info = mt5.symbol_info(symbol)
        if not info:
            print(f"Skipping {symbol}: Symbol not found")
            continue

        for tf_str in ["M5", "M15", "M30"]:
            print(f"Analyzing {symbol} @ {tf_str}...")
            df = fetch_data(symbol, tf_str)
            if df is None:
                continue

            # Simulate simple volatility-based strategy for study
            # (Close-Open returns as proxy for 'trades' in a persistent market)
            # In real study we'd use the actual strategy signals, 
            # but we can simulate the 'persistence' check.
            
            # Using basic return series to see which timeframe is 'less random'
            rets = df['close'].pct_change().dropna().values
            
            # Run Monte Carlo
            mc_results = simulator.run_simulation(rets)
            
            results.append({
                "Symbol": symbol,
                "Timeframe": tf_str,
                "Sortino": mc_results['original_sortino'],
                "p-value": mc_results['p_value'],
                "Pass": mc_results['pass_validation']
            })

    mt5.shutdown()
    
    if results:
        res_df = pd.DataFrame(results)
        print("\n=== TIMEFRAME STUDY RESULTS ===")
        print(res_df.sort_values(by=["Symbol", "Sortino"], ascending=[True, False]))
        return res_df
    return None

if __name__ == "__main__":
    run_study()
