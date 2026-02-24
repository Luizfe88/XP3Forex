"""
XP3 PRO FOREX - Validation Module (Migrated to src-layout)

This module has been migrated to the new src-layout architecture.
Original: validation_forex.py
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from xp3_forex.core import config
from xp3_forex.utils import mt5_utils
from xp3_forex.utils import calculations
from xp3_forex.utils.data_utils import daily_logger

logger = logging.getLogger("XP3_VALIDATION")

# ===========================
# ENUMS
# ===========================
class OrderSide(Enum):
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL

# ===========================
# DATACLASSES
# ===========================
@dataclass
class OrderParams:
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    sl: float
    tp: float
    comment: str
    magic: int

# ===========================
# ORDER VALIDATION & EXECUTION
# ===========================
def validate_and_create_order_forex(params: OrderParams) -> Tuple[bool, str, int]:
    """
    Valida e cria ordem no MT5 com gestão de risco avançada.
    
    Returns:
        Tuple[bool, str, int]: (success, message, ticket)
    """
    try:
        # Validate symbol
        symbol_info = mt5.symbol_info(params.symbol)
        if symbol_info is None:
            return False, f"Símbolo não encontrado: {params.symbol}", 0
        
        # Check if symbol is visible
        if not symbol_info.visible:
            if not mt5.symbol_select(params.symbol, True):
                return False, f"Falha ao selecionar símbolo: {params.symbol}", 0
        
        # Get current price
        tick = mt5.symbol_info_tick(params.symbol)
        if tick is None:
            return False, f"Falha ao obter tick para: {params.symbol}", 0
        
        # Calculate current price based on order side
        current_price = tick.ask if params.side == OrderSide.BUY else tick.bid
        
        # Risk validation
        max_risk_per_trade = getattr(config, 'MAX_RISK_PER_TRADE', 0.02)
        max_volume = getattr(config, 'MAX_VOLUME_PER_TRADE', 10.0)
        
        if params.volume > max_volume:
            return False, f"Volume excede limite máximo: {params.volume} > {max_volume}", 0
        
        # Calculate stop loss distance
        sl_distance = abs(params.entry_price - params.sl)
        if sl_distance <= 0:
            return False, "Stop loss inválido", 0
        
        # Validate take profit
        if params.tp <= 0:
            return False, "Take profit inválido", 0
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": params.symbol,
            "volume": params.volume,
            "type": params.side.value,
            "price": params.entry_price,
            "sl": params.sl,
            "tp": params.tp,
            "deviation": getattr(config, 'MAX_DEVIATION', 20),
            "magic": params.magic,
            "comment": params.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Ordem rejeitada: {result.comment}", 0
        
        # Log successful order
        logger.info(f"✅ Ordem executada: {params.symbol} {params.side.name} {params.volume}@{params.entry_price}")
        daily_logger.log_trade(params.symbol, params.side.name, params.volume, params.entry_price, params.sl, params.tp, result.order)
        
        return True, f"Ordem executada com sucesso: {result.order}", result.order
        
    except Exception as e:
        logger.error(f"❌ Erro na validação/criação da ordem: {e}")
        return False, f"Erro na ordem: {str(e)}", 0

# ===========================
# BACKWARD COMPATIBILITY
# ===========================
# Re-export for legacy compatibility
OrderSide = OrderSide
OrderParams = OrderParams
validate_and_create_order_forex = validate_and_create_order_forex

print("✅ Validation module migrated to src-layout")