from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class TradeSignal:
    """Represents a trading signal"""
    symbol: str
    order_type: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    order_type: str
    volume: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    profit: float
    pips: float
    open_time: datetime
    magic_number: int
    partial_taken: bool = False
    initial_sl_dist: float = 0.0
