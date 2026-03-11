
import MetaTrader5 as mt5
import time
from datetime import datetime

def check_time_sync():
    if not mt5.initialize():
        print("Falha ao inicializar MT5")
        return

    print(f"Machine Time (UTC): {datetime.utcnow().timestamp()}")
    print(f"Machine Time (Local): {time.time()}")
    
    # Get any symbol tick to get server time
    symbol = "EURUSD"
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"Server Time (from tick): {tick.time}")
        print(f"Difference (Server - Local): {tick.time - time.time()}")

    # Check open positions
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            print(f"Position {pos.ticket} ({pos.symbol}):")
            print(f"  Open Time: {pos.time}")
            print(f"  Calculated Age (Local): {time.time() - pos.time}")
            if tick:
                print(f"  Calculated Age (Server): {tick.time - pos.time}")
    else:
        print("No open positions found to check.")

    mt5.shutdown()

if __name__ == "__main__":
    check_time_sync()
