import MetaTrader5 as mt5
import sys

def main():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        print("No positions found.")
        mt5.shutdown()
        return

    print(f"Found {len(positions)} positions.")
    for pos in positions:
        print(f"Position {pos.ticket} - {pos.symbol} volume: {pos.volume}")
        symbol = pos.symbol
        volume = pos.volume
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print("Failed to get tick for", symbol)
            continue
            
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        sym_info = mt5.symbol_info(symbol)
        digits = sym_info.digits if sym_info else 5
        
        filling_mode = mt5.ORDER_FILLING_IOC
        if sym_info:
            fm = sym_info.filling_mode
            if fm & 2: filling_mode = mt5.ORDER_FILLING_IOC
            elif fm & 4: filling_mode = mt5.ORDER_FILLING_RETURN
            elif fm & 1: filling_mode = mt5.ORDER_FILLING_FOK

        request_price = float(round(price, digits))
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": pos.ticket,
            "price": request_price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "XP3: CLOSE POS",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode, 
        }

        print("Testing order_check...")
        check = mt5.order_check(request)
        if check:
            print(f"Check Result: {check.retcode} - {check.comment}")
            if check.retcode != 0:
                 print("Attempting with price = 0.0")
                 request["price"] = 0.0
                 check = mt5.order_check(request)
                 print(f"Fallback check Result: {check.retcode} - {check.comment}")
                 
                 print("Attempting with SL/TP = 0.0")
                 request["sl"] = 0.0
                 request["tp"] = 0.0
                 check = mt5.order_check(request)
                 print(f"Fallback 2 check Result: {check.retcode} - {check.comment}")

            result = mt5.order_send(request)
            print(f"Send result: {result}")
        else:
            print("order_check returned None")

    mt5.shutdown()

if __name__ == "__main__":
    main()
