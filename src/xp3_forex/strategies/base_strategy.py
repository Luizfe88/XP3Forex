from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies in XP3 PRO FOREX.
    Enforces a standard interface for strategy implementation.
    """

    def __init__(self, name: str, bot):
        self.name = name
        self.bot = bot  # Reference to the main XP3Bot instance
        self.logger = logging.getLogger(f"XP3.Strategy.{name}")
        self.is_active = True

    @abstractmethod
    def startup(self):
        """
        Called when the bot starts.
        Use this for initial analysis, data loading, or regime detection.
        """
        pass

    @abstractmethod
    def on_tick(self, symbol: str, tick: Any):
        """
        Called on every tick for a specific symbol.
        """
        pass

    @abstractmethod
    def on_bar(self, symbol: str, timeframe: str):
        """
        Called when a new bar is closed/opened.
        """
        pass
