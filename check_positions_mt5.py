
import MetaTrader5 as mt5
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from xp3_forex.utils.mt5_utils import initialize_mt5
from xp3_forex.core.settings import settings

def check_positions():
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
        
    print(f"Bot Magic Number: {settings.MAGIC_NUMBER}")
    
    positions = mt5.positions_get()
    if positions is None:
        print(f"Error getting positions: {mt5.last_error()}")
    elif len(positions) == 0:
        print("No positions found in MT5.")
    else:
        print(f"Found {len(positions)} positions in MT5:")
        for p in positions:
            print(f"Ticket: {p.ticket}, Symbol: {p.symbol}, Magic: {p.magic}, Type: {p.type}, Volume: {p.volume}")
            
    mt5.shutdown()

if __name__ == "__main__":
    check_positions()
