import MetaTrader5 as mt5
import time
import logging
from threading import Lock
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from ..utils.mt5_utils import mt5_exec, resolve_symbol_name

logger = logging.getLogger(__name__)

class SymbolManager:
    """
    Singleton para gerenciamento de símbolos com cache e Circuit Breaker.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SymbolManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._symbol_cache: Dict[str, Any] = {}
        self._tick_value_cache: Dict[str, float] = {}
        self._point_cache: Dict[str, float] = {}
        
        # Circuit Breaker State
        self._failures: Dict[str, int] = {}
        self._cooldowns: Dict[str, float] = {}  # timestamp
        
        # Configurações do Circuit Breaker
        self.MAX_FAILURES = 5
        self.COOLDOWN_SECONDS = 300  # 5 minutos
        
        logger.info("SymbolManager Singleton inicializado")

    def get_info(self, symbol: str) -> Optional[Any]:
        """Retorna informações estáticas do símbolo com cache (Point, Digits, etc)"""
        if not self.is_available(symbol):
            return None
            
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        
        # Tenta obter do MT5
        info = mt5_exec(mt5.symbol_info, symbol)
        if info:
            self._symbol_cache[symbol] = info
            self._tick_value_cache[symbol] = info.trade_tick_value
            self._point_cache[symbol] = info.point
            self._reset_failure(symbol)
            return info
        else:
            self._record_failure(symbol)
            return None

    def get_tick(self, symbol: str) -> Optional[Any]:
        """Retorna tick atual (preço fresco)"""
        if not self.is_available(symbol):
            return None
            
        tick = mt5_exec(mt5.symbol_info_tick, symbol)
        if tick:
            self._reset_failure(symbol)
            return tick
        else:
            self._record_failure(symbol)
            return None

    def get_tick_value(self, symbol: str) -> float:
        """Retorna tick value cacheado"""
        if symbol in self._tick_value_cache:
            return self._tick_value_cache[symbol]
        
        info = self.get_info(symbol)
        return info.trade_tick_value if info else 0.0

    def get_point(self, symbol: str) -> float:
        """Retorna point cacheado"""
        if symbol in self._point_cache:
            return self._point_cache[symbol]
        
        info = self.get_info(symbol)
        return info.point if info else 0.00001

    def is_available(self, symbol: str) -> bool:
        """Verifica se o símbolo está operante (fora do cooldown)"""
        if symbol in self._cooldowns:
            if time.time() < self._cooldowns[symbol]:
                # Ainda em cooldown
                return False
            else:
                # Cooldown expirou
                del self._cooldowns[symbol]
                self._failures[symbol] = 0
                logger.info(f"Símbolo {symbol} saiu do Cool Down")
                return True
        return True

    def report_success(self, symbol: str):
        """Reporta sucesso na obtenção de dados"""
        self._reset_failure(symbol)

    def report_failure(self, symbol: str):
        """Reporta falha na obtenção de dados"""
        self._record_failure(symbol)

    def _record_failure(self, symbol: str):
        """Registra falha e ativa Circuit Breaker se necessário"""
        self._failures[symbol] = self._failures.get(symbol, 0) + 1
        logger.warning(f"Falha registrada para {symbol}. Total: {self._failures[symbol]}")
        
        if self._failures[symbol] >= self.MAX_FAILURES:
            self._cooldowns[symbol] = time.time() + self.COOLDOWN_SECONDS
            logger.error(f"CIRCUIT BREAKER ATIVADO: {symbol} em Cool Down por {self.COOLDOWN_SECONDS}s")

    def _reset_failure(self, symbol: str):
        """Reseta contador de falhas após sucesso"""
        if symbol in self._failures and self._failures[symbol] > 0:
            self._failures[symbol] = 0

    def resolve_name(self, base_symbol: str) -> str:
        """Wrapper para resolve_symbol_name com cache local se possível"""
        # Aqui poderia ter cache de resolução de nomes também
        # Por enquanto usa o utilitário, mas armazena o resultado no cache de info
        # se for chamado subsequentemente
        
        # Se já temos no cache, retorna o próprio (assumindo que base_symbol já é o resolvido ou chave)
        if base_symbol in self._symbol_cache:
            return base_symbol
            
        resolved = resolve_symbol_name(base_symbol)
        return resolved if resolved else base_symbol
