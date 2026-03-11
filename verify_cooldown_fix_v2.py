
import sys
import time
from unittest.mock import MagicMock
from datetime import datetime

# Mock MetaTrader5
mock_mt5 = MagicMock()
sys.modules['MetaTrader5'] = mock_mt5
sys.modules['xp3_forex.utils.telegram_utils'] = MagicMock()
sys.modules['xp3_forex.utils.data_utils'] = MagicMock()

# Import components
from xp3_forex.core.trade_executor import TradeExecutor
from xp3_forex.mt5.symbol_manager import symbol_manager

# Ensure symbols_get is mocked to avoid long runs
mock_mt5.symbols_get.return_value = []

def test_cooldown_bypass_and_time_sync():
    executor = TradeExecutor()
    executor.mode = "live"
    
    # Define a custom class for mock position to avoid MagicMock formatting issues
    class MockPos:
        def __init__(self, ticket, symbol, volume, open_time, profit, swap, commission, type):
            self.ticket = ticket
            self.symbol = symbol
            self.volume = float(volume)
            self.time = int(open_time)
            self.profit = float(profit)
            self.swap = float(swap)
            self.commission = float(commission)
            self.type = type

    # 1. Test Negative Age Sync
    # pos.time is in the future relative to machine/server time
    future_time = int(time.time() + 3600)
    mock_pos = MockPos(ticket=1, symbol="XAGUSD", volume=0.1, open_time=future_time, 
                      profit=10.0, swap=0.0, commission=0.0, type=0) # mt5.ORDER_TYPE_BUY
    
    mock_mt5.terminal_info.return_value.connected = True
    mock_mt5.positions_get.return_value = [mock_pos]
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    
    mock_sym_info = MagicMock()
    mock_sym_info.digits = 5
    mock_sym_info.filling_mode = 2
    mock_sym_info.trade_exemode = 2
    mock_mt5.symbol_info.return_value = mock_sym_info
    
    mock_result = MagicMock()
    mock_result.retcode = 10009 # DONE
    mock_result.price = 1.1000
    mock_result.volume = 0.1
    mock_result.comment = "Request executed"
    mock_mt5.order_send.return_value = mock_result
    mock_mt5.TRADE_RETCODE_DONE = 10009
    
    # Mock tick to be machine time (so age is negative)
    mock_tick = MagicMock()
    mock_tick.time = int(time.time())
    mock_tick.bid = 1.1
    mock_tick.ask = 1.1001
    symbol_manager.get_tick = MagicMock(return_value=mock_tick)
    
    print("Testing negative age with ignore_cooldown=False (should block)...")
    result = executor.close_position(ticket=1, symbol="XAGUSD", ignore_cooldown=False)
    if not result:
        print("✅ Correct: Blocked because age (normalized to 0) < 60s.")
    else:
        print("❌ Failure: Should have blocked.")

    print("\nTesting negative age with ignore_cooldown=True (should bypass)...")
    # Reset mock_mt5 to track calls
    mock_mt5.order_send.reset_mock()
    result = executor.close_position(ticket=1, symbol="XAGUSD", ignore_cooldown=True)
    if result:
        print("✅ Correct: Bypassed cooldown despite negative/short age.")
        if mock_mt5.order_send.called:
             print("✅ Success: order_send was actually called.")
    else:
        print("❌ Failure: Should have bypassed.")

    # 2. Test Server Time Sync (Should use server time from tick)
    server_time = int(time.time() + 5000)
    mock_pos.time = server_time - 120 # Age = 120s in server time
    mock_tick.time = server_time
    
    print("\nTesting holding time using Server Time (age 120s, should allow)...")
    mock_mt5.order_send.reset_mock()
    result = executor.close_position(ticket=1, symbol="XAGUSD", ignore_cooldown=False)
    if result:
        print("✅ Correct: Allowed because server-side age is 120s > 60s.")
    else:
        print("❌ Failure: Should have allowed.")

if __name__ == "__main__":
    test_cooldown_bypass_and_time_sync()
