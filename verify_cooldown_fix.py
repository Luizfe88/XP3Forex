
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

def test_cooldown_bypass_and_time_sync():
    executor = TradeExecutor()
    executor.mode = "live"
    
    # 1. Test Negative Age Sync (Should be detected as 0 and NOT bypass if ignore_cooldown=False)
    mock_pos = MagicMock()
    mock_pos.time = time.time() + 3600 # 1 hour into the FUTURE (simulating server clock behind machine clock)
    mock_pos.symbol = "XAGUSD"
    mock_pos.profit = 10.0
    mock_pos.swap = 0.0
    mock_pos.commission = 0.0
    
    mock_mt5.terminal_info.return_value.connected = True
    mock_mt5.positions_get.return_value = [mock_pos]
    mock_mt5.symbol_info.return_value.digits = 5
    mock_mt5.symbol_info.return_value.filling_mode = 2
    mock_mt5.symbol_info.return_value.trade_exemode = 2
    mock_mt5.order_send.return_value.retcode = 10009
    
    # Mock tick to be machine time (so age is negative)
    symbol_manager.get_tick = MagicMock(return_value=MagicMock(time=time.time(), bid=1.1, ask=1.1001))
    
    print("Testing negative age with ignore_cooldown=False (should block)...")
    result = executor.close_position(ticket=1, symbol="XAGUSD", ignore_cooldown=False)
    if not result:
        print("✅ Correct: Blocked because age (normalized to 0) < 60s.")
    else:
        print("❌ Failure: Should have blocked.")

    print("\nTesting negative age with ignore_cooldown=True (should bypass)...")
    result = executor.close_position(ticket=1, symbol="XAGUSD", ignore_cooldown=True)
    if result:
        print("✅ Correct: Bypassed cooldown despite negative/short age.")
    else:
        print("❌ Failure: Should have bypassed.")

    # 2. Test Server Time Sync (Should use server time from tick)
    server_time = time.time() + 5000 
    mock_pos.time = server_time - 120 # Age = 120s in server time
    symbol_manager.get_tick.return_value.time = server_time
    
    print("\nTesting holding time using Server Time (age 120s, should allow)...")
    result = executor.close_position(ticket=1, symbol="XAGUSD", ignore_cooldown=False)
    if result:
        print("✅ Correct: Allowed because server-side age is 120s > 60s.")
    else:
        print("❌ Failure: Should have allowed.")

if __name__ == "__main__":
    test_cooldown_bypass_and_time_sync()
