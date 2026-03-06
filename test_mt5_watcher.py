import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from xp3_forex.utils.process_watcher import ProcessWatcher
from xp3_forex.core.settings import settings

class TestProcessWatcher(unittest.TestCase):
    
    @patch('psutil.process_iter')
    def test_is_mt5_running_false(self, mock_process_iter):
        # Setup mock to return no relevant processes
        mock_process_iter.return_value = [
            MagicMock(info={'name': 'python.exe', 'exe': 'path/to/python.exe'}),
            MagicMock(info={'name': 'chrome.exe', 'exe': 'path/to/chrome.exe'})
        ]
        
        self.assertFalse(ProcessWatcher.is_mt5_running())

    @patch('psutil.process_iter')
    def test_is_mt5_running_true(self, mock_process_iter):
        # Setup mock to return MT5 process
        mt5_path = Path(settings.MT5_PATH).resolve()
        
        mock_proc = MagicMock()
        mock_proc.info = {'name': 'terminal64.exe', 'exe': str(mt5_path)}
        
        mock_process_iter.return_value = [mock_proc]
        
        self.assertTrue(ProcessWatcher.is_mt5_running())

    @patch('psutil.process_iter')
    def test_is_mt5_running_wrong_path(self, mock_process_iter):
        # Setup mock to return MT5 process but with different path
        mock_proc = MagicMock()
        mock_proc.info = {'name': 'terminal64.exe', 'exe': 'C:/Wrong/Path/terminal64.exe'}
        
        mock_process_iter.return_value = [mock_proc]
        
        self.assertFalse(ProcessWatcher.is_mt5_running())

if __name__ == '__main__':
    unittest.main()
