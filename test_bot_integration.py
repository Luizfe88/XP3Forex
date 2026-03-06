import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import threading
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from xp3_forex.core.bot import XP3Bot, SHUTDOWN_EVENT
from xp3_forex.core.settings import settings

class TestBotIntegration(unittest.TestCase):
    
    @patch('xp3_forex.core.bot.ProcessWatcher.is_mt5_running')
    @patch('xp3_forex.core.bot.initialize_mt5')
    def test_initialize_mt5_blocks_when_running(self, mock_init, mock_is_running):
        # Setup: MT5 is running
        mock_is_running.return_value = True
        
        bot = XP3Bot()
        result = bot.initialize_mt5()
        
        self.assertFalse(result)
        mock_init.assert_not_called()

    @patch('xp3_forex.core.bot.ProcessWatcher.is_mt5_running')
    @patch('xp3_forex.core.bot.XP3Bot.update_positions')
    def test_consumer_loop_shutdown_when_mt5_opens(self, mock_update, mock_is_running):
        # Setup: Bot is running, MT5 is NOT running initially
        mock_is_running.side_effect = [False, True] # First call False, second True
        
        bot = XP3Bot()
        SHUTDOWN_EVENT.clear()
        
        # We need to run consumer_loop in a way that we can break it or it breaks itself
        # Since SHUTDOWN_EVENT is global and used in a while loop, we can test it.
        # However, consumer_loop is blocking. We'll mock the internal checks to speed up.
        
        def run_loop():
            bot.consumer_loop()
        
        thread = threading.Thread(target=run_loop)
        thread.start()
        
        # Wait for thread to finish (it should break loop when mock returns True)
        thread.join(timeout=5)
        
        self.assertTrue(SHUTDOWN_EVENT.is_set())
        self.assertFalse(thread.is_alive())

if __name__ == '__main__':
    unittest.main()
