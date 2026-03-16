
import MetaTrader5 as mt5
import os
import sys
from pathlib import Path
import time

# Add src to sys.path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from xp3_forex.utils.mt5_utils import initialize_mt5
from xp3_forex.core.settings import settings

def diagnose():
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
        
    print(f"--- XP3 ACCOUNT DIAGNOSIS ---")
    print(f"Time (Local): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    account = mt5.account_info()
    if not account:
        print("❌ Failed to get account info")
    else:
        print(f"Account: {account.login}")
        print(f"Server: {account.server}")
        print(f"Ccy: {account.currency}")
        print(f"Balance: {account.balance:.2f}")
        print(f"Equity: {account.equity:.2f}")
        print(f"Profit: {account.profit:.2f}")
        print(f"Margin: {account.margin:.2f}")
        print(f"Margin Free: {account.margin_free:.2f}")
        print(f"Margin Level: {account.margin_level:.1f}%")
        
        # Health Checks similar to TradeExecutor
        if account.margin_free <= 0:
            print("❌ NO FREE MARGIN")
        elif account.margin > 0 and account.margin_level < 100:
            print("❌ MARGIN CALL RISK")
        elif account.equity <= 0:
            print("❌ NEGATIVE EQUITY")
            
    term = mt5.terminal_info()
    if term:
        print(f"Connected: {term.connected}")
        print(f"Algo Trading: {term.trade_allowed}")
        
    print(f"--- SYMBOLS STATUS ---")
    symbols = settings.symbols_list
    if not symbols:
        print("Whitelisting ALL symbols from Market Watch")
    else:
        print(f"Checking {len(symbols)} configured symbols...")
        for s in symbols:
            info = mt5.symbol_info(s)
            if not info:
                # Try common suffixes
                resolved = None
                for suffix in ["", ".a", ".pro", ".r", ".c", ".m", ".b", "+", "_i", "_op"]:
                    if mt5.symbol_info(s + suffix):
                        resolved = s + suffix
                        break
                if resolved:
                    info = mt5.symbol_info(resolved)
                    print(f"[{resolved}] OK")
                else:
                    print(f"[{s}] NOT FOUND")
            else:
                print(f"[{s}] OK (Trade Mode: {info.trade_mode})")

    mt5.shutdown()

if __name__ == "__main__":
    diagnose()
