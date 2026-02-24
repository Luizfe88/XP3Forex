"""
XP3 PRO FOREX - Legacy Utils Wrapper

This module provides backward compatibility for code that imports utils_forex
while using the new src-layout architecture.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import from new structure
try:
    from xp3_forex.utils import *
    from xp3_forex.utils.mt5_utils import *
    from xp3_forex.utils.indicators import *
    from xp3_forex.utils.calculations import *
    from xp3_forex.utils.data_utils import *
    from xp3_forex.core.config import *
except ImportError as e:
    print(f"❌ Erro ao importar módulos do novo sistema: {e}")
    # Fallback to legacy imports
    try:
        import config_forex as config
        from decimal import Decimal, ROUND_HALF_UP
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta, timezone
        import pytz
        import time
        import logging
        import threading
        import queue
        from concurrent.futures import Future
        import json
        import csv
        import sqlite3
        from typing import Optional, Tuple, Dict, Any, List
        from numba import njit
        
        logger = logging.getLogger("XP3_UTILS")
        print("⚠️ Usando fallback para imports legados")
    except ImportError as fallback_error:
        print(f"❌ Erro no fallback: {fallback_error}")
        sys.exit(1)

# Re-export all commonly used functions and variables
__all__ = [
    # MT5 utilities
    'get_rates', 'mt5_exec', 'initialize_mt5', 'mt5_lock',
    # Indicators
    'calculate_ema', 'calculate_rsi', 'calculate_adx', 'calculate_atr',
    # Calculations
    'calculate_lot_size', 'calculate_sl_tp', 'get_pip_size',
    # Data utilities
    'save_trade', 'get_trade_history', 'setup_database',
    # Config (legacy compatibility)
    'config'
]

# Create config alias for legacy compatibility
if 'config' not in locals():
    try:
        from xp3_forex.core import config as _new_config
        config = _new_config
    except ImportError:
        import config_forex as config

print("✅ Utils wrapper carregado - usando nova arquitetura src-layout")