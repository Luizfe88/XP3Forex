
import sys
import os
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(r"C:\Users\luizf\Documents\xp3forex\src")))

from xp3_forex.mt5.symbol_manager import symbol_manager
from xp3_forex.core.settings import settings

def test_categorization():
    symbols_to_test = ["EURUSD", "USDTHB", "BNBUSD", "XAUUSD", "NAS100", "USDJPY", "USDBRL"]
    
    print(f"{'Symbol':<10} | {'Category':<10} | {'Max Spread':<10}")
    print("-" * 35)
    
    for sym in symbols_to_test:
        category = symbol_manager._categorize_symbol(sym)
        max_spread = symbol_manager._get_max_spread_for_category(category)
        print(f"{sym:<10} | {category:<10} | {max_spread:<10}")

if __name__ == "__main__":
    test_categorization()
