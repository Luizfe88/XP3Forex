
import MetaTrader5 as mt5
import os
import sys
import json
import sqlite3
from pathlib import Path

# Add src to path to import settings if needed
sys.path.append(str(Path.cwd() / "src"))

try:
    from xp3_forex.core.settings import settings
except ImportError:
    settings = None

def get_contract_size(symbol):
    # Standard defaults
    if "XAU" in symbol: return 100
    if "XAG" in symbol: return 5000
    if "UK100" in symbol: return 1
    if "US30" in symbol: return 1
    if "NAS100" in symbol: return 1
    # Forex
    return 100000

def get_notional_value_usd(symbol, volume, entry_price):
    contract_size = get_contract_size(symbol)
    
    # If symbol starts with USD (e.g. USDCHF, USDJPY, USDCNH)
    if symbol.startswith("USD"):
        # Base is USD, so Notional is Volume * ContractSize in USD
        return volume * contract_size
    
    # If symbol ends with USD (e.g. EURUSD, XAUUSD, XAGUSD)
    if symbol.endswith("USD"):
        # Notional is Volume * ContractSize * EntryPrice in USD
        return volume * contract_size * entry_price
    
    # Crosses or non-USD base/quote
    # For XAUAUD, XAGAUD, etc.
    # Value = Volume * ContractSize * EntryPrice (in AUD)
    # We'd need AUDUSD to convert to USD.
    # Let's assume some common rates if we can't get live ones.
    if symbol.endswith("AUD"):
        audusd = 0.65 # Approximate
        return volume * contract_size * entry_price * audusd
    
    if symbol.endswith("CHF"):
        # USDCHF = 0.9, so 1 CHF = 1.1 USD
        return volume * contract_size * entry_price * 1.1
    
    # Default to just Volume * ContractSize * EntryPrice and hope it's close or USD-like
    return volume * contract_size * entry_price

def main():
    db_path = "data/trades.db"
    output_path = "trades_notional_report.txt"

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticket, symbol, volume, entry_price, exit_price, profit, exit_time 
        FROM trades 
        WHERE status = 'closed' 
        ORDER BY exit_time DESC 
        LIMIT 10
    """)
    rows = cursor.fetchall()
    
    print(f"{'Ticket':<12} | {'Ativo':<8} | {'Volume':<6} | {'Entrada':<10} | {'Valor $':<12} | {'Retorno':<10}")
    print("-" * 75)
    
    with open(output_path, "w") as f:
        f.write(f"{'Ticket':<12} | {'Ativo':<8} | {'Volume':<6} | {'Entrada':<10} | {'Valor $':<12} | {'Retorno':<10}\n")
        f.write("-" * 75 + "\n")
        
        for row in rows:
            notional = get_notional_value_usd(row['symbol'], row['volume'], row['entry_price'])
            line = f"{row['ticket']:<12} | {row['symbol']:<8} | {row['volume']:<6} | {row['entry_price']:<10.5f} | {notional:<12.2f} | {row['profit']:<10.2f}\n"
            print(line, end="")
            f.write(line)
            
    conn.close()

if __name__ == "__main__":
    main()
