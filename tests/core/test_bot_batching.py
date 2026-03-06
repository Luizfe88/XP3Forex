import unittest
from unittest.mock import MagicMock, patch
import queue
from datetime import datetime
from xp3_forex.core.bot import XP3Bot
from xp3_forex.core.models import TradeSignal

class TestBotBatching(unittest.TestCase):
    @patch('xp3_forex.core.bot.HealthMonitor')
    @patch('xp3_forex.core.bot.DataFeeder')
    @patch('xp3_forex.core.bot.AdaptiveEmaRsiStrategy')
    @patch('xp3_forex.core.bot.SymbolManager')
    @patch('xp3_forex.core.bot.RateCache')
    def setUp(self, mock_cache, mock_sm, mock_strat, mock_feeder, mock_health):
        self.bot = XP3Bot()
        self.bot.is_trading_active = True

    def test_signal_batching_and_ranking(self):
        # 1. Create mock signals with different confidence levels
        sig1 = TradeSignal(symbol="EURUSD", order_type="BUY", entry_price=1.1, stop_loss=1.09, take_profit=1.12, volume=0.1, confidence=0.7, reason="test")
        sig2 = TradeSignal(symbol="GBPUSD", order_type="SELL", entry_price=1.3, stop_loss=1.31, take_profit=1.28, volume=0.1, confidence=0.9, reason="test")
        sig3 = TradeSignal(symbol="USDJPY", order_type="BUY", entry_price=110.0, stop_loss=109.0, take_profit=112.0, volume=0.1, confidence=0.8, reason="test")

        # 2. Simulate signals arriving in the queue
        # We don't use the queue directly in this test, but we test the logic of accumulation
        self.bot.analyze_symbol = MagicMock()
        
        # Manually add to pending_signals to simulate what consumer_loop does
        self.bot.pending_signals.append(sig1)
        self.bot.pending_signals.append(sig2)
        self.bot.pending_signals.append(sig3)

        # 3. Mock execute_trade to track order of execution
        execution_order = []
        def mock_execute(signal):
            execution_order.append(signal.symbol)
            return True
        self.bot.execute_trade = MagicMock(side_effect=mock_execute)

        # 4. Trigger batch processing
        self.bot.process_batch_signals()

        # 5. Assertions
        # Sorted by confidence: sig2 (0.9), sig3 (0.8), sig1 (0.7)
        expected_order = ["GBPUSD", "USDJPY", "EURUSD"]
        self.assertEqual(execution_order, expected_order)
        self.assertEqual(len(self.bot.pending_signals), 0)

    def test_max_positions_respected_in_batch(self):
        # 1. Mock settings.MAX_POSITIONS to 2
        with patch('xp3_forex.core.bot.settings') as mock_settings:
            mock_settings.MAX_POSITIONS = 2
            
            # 2. Create 3 signals
            sig1 = TradeSignal(symbol="EURUSD", order_type="BUY", entry_price=1.1, stop_loss=1.09, take_profit=1.12, volume=0.1, confidence=0.9, reason="test")
            sig2 = TradeSignal(symbol="GBPUSD", order_type="SELL", entry_price=1.3, stop_loss=1.31, take_profit=1.28, volume=0.1, confidence=0.8, reason="test")
            sig3 = TradeSignal(symbol="USDJPY", order_type="BUY", entry_price=110.0, stop_loss=109.0, take_profit=112.0, volume=0.1, confidence=0.7, reason="test")

            self.bot.pending_signals = [sig1, sig2, sig3]

            # 3. Mock execute_trade to update positions count
            def mock_execute_side_effect(signal):
                self.bot.positions[signal.symbol] = MagicMock()
                return True
            self.bot.execute_trade = MagicMock(side_effect=mock_execute_side_effect)
            self.bot.positions = {"EXISTING": MagicMock()} # Start with 1 position
            
            # 4. Process
            self.bot.process_batch_signals()

            # 5. Assertions
            # Only 1 more trade should be executed because limit is 2 and we already have 1.
            self.assertEqual(self.bot.execute_trade.call_count, 1)
            called_signal = self.bot.execute_trade.call_args[0][0]
            self.assertEqual(called_signal.symbol, "EURUSD") # Highest confidence

if __name__ == '__main__':
    unittest.main()
