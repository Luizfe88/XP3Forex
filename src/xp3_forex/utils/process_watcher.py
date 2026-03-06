"""
Utility to watch for specific processes (MetaTrader 5)
"""
import psutil
import os
import logging
from pathlib import Path
from xp3_forex.core.settings import settings

logger = logging.getLogger("XP3_PROCESS_WATCHER")

class ProcessWatcher:
    """Monitoring utility for MT5 processes"""
    
    @staticmethod
    def is_mt5_running() -> bool:
        """
        Check if MetaTrader 5 (specifically the one in settings) is running.
        Returns:
            bool: True if MT5 is running, False otherwise.
        """
        try:
            mt5_path = Path(settings.MT5_PATH).resolve()
            
            for proc in psutil.process_iter(['name', 'exe']):
                try:
                    # Check by name first for efficiency
                    if proc.info['name'] == 'terminal64.exe':
                        # Check if the path matches to avoid closing the wrong MT5
                        if proc.info['exe'] and Path(proc.info['exe']).resolve() == mt5_path:
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
            return False
        except Exception as e:
            logger.error(f"Error checking for MT5 process: {e}")
            return False
