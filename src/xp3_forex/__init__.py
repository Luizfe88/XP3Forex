"""
XP3 PRO FOREX - Módulo principal

Trading bot profissional para Forex com ML e otimização avançada.
"""

__version__ = "4.2.0"
__author__ = "Luizfe88"
__description__ = "XP3 PRO FOREX - Trading Bot com Machine Learning"

from .core import *
from .core.bot import XP3Bot
from .utils import *
from .utils.mt5_utils import get_rates, mt5_exec, initialize_mt5
from .utils.indicators import calculate_ema, calculate_rsi, calculate_adx, calculate_atr
from .utils.calculations import calculate_lot_size, calculate_sl_tp, get_pip_size
from .utils.data_utils import save_trade, get_trade_history