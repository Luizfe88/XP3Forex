
import MetaTrader5 as mt5
import time
import logging
from threading import Lock
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta
from xp3_forex.utils.mt5_utils import mt5_exec

logger = logging.getLogger(__name__)

class SymbolManager:
    """
    Singleton para gerenciamento de símbolos com cache inteligente,
    detecção automática de sufixos e Circuit Breaker.
    """
    _instance = None
    _lock = Lock()
    
    # Common suffixes to try automatically
    COMMON_SUFFIXES = [
        "", ".a", ".pro", ".r", ".c", ".m", ".b", "+", "_i", "_op", "m", "c"
    ]

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
        
        # Caches
        self._symbol_info_cache: Dict[str, Any] = {}
        self._tick_cache: Dict[str, Any] = {}
        self._symbol_name_map: Dict[str, str] = {}  # Map input name (EURUSD) -> real name (EURUSD.a)
        self._all_symbols_names: Optional[Set[str]] = None
        self._last_all_symbols_update: float = 0
        
        # Circuit Breaker State
        self._failures: Dict[str, int] = {}
        self._cooldowns: Dict[str, float] = {}
        
        # Configs
        self.MAX_FAILURES = 5
        self.COOLDOWN_SECONDS = 300
        self.CACHE_TTL = 3600  # 1 hour for static info
        self.ALL_SYMBOLS_REFRESH = 3600 * 24 # 24 hours
        
        logger.info("SymbolManager Singleton inicializado")

    def resolve_name(self, base_symbol: str) -> Optional[str]:
        """
        Resolve o nome do símbolo (ex: 'EURUSD' -> 'EURUSD.a').
        Usa cache para resoluções anteriores.
        """
        base_symbol = base_symbol.strip().upper()
        
        # 1. Check cache
        if base_symbol in self._symbol_name_map:
            return self._symbol_name_map[base_symbol]
        
        # 2. Try direct match first (most common)
        if self._check_symbol_exists(base_symbol):
            self._symbol_name_map[base_symbol] = base_symbol
            return base_symbol
            
        # 3. Try common suffixes
        for suffix in self.COMMON_SUFFIXES:
            if not suffix: continue
            candidate = f"{base_symbol}{suffix}"
            if self._check_symbol_exists(candidate):
                logger.info(f"Símbolo resolvido: {base_symbol} -> {candidate}")
                self._symbol_name_map[base_symbol] = candidate
                return candidate
                
        # 4. Fallback: Search in all symbols (expensive, do lazily)
        real_name = self._find_in_all_symbols(base_symbol)
        if real_name:
            self._symbol_name_map[base_symbol] = real_name
            return real_name
            
        logger.error(f"Símbolo não encontrado na corretora: {base_symbol}")
        return None

    def _check_symbol_exists(self, symbol: str) -> bool:
        """Verifica se símbolo existe e seleciona no Market Watch"""
        info = mt5_exec(mt5.symbol_info, symbol)
        if info is not None:
            if not info.select:
                mt5_exec(mt5.symbol_select, symbol, True)
            return True
        return False

    def _find_in_all_symbols(self, base: str) -> Optional[str]:
        """Procura na lista completa de símbolos"""
        now = time.time()
        # Update cache if needed
        if self._all_symbols_names is None or (now - self._last_all_symbols_update > self.ALL_SYMBOLS_REFRESH):
            logger.info("Atualizando cache de todos os símbolos (pode demorar)...")
            symbols = mt5_exec(mt5.symbols_get)
            if symbols:
                self._all_symbols_names = {s.name for s in symbols}
                self._last_all_symbols_update = now
            else:
                return None
        
        # Search in cache
        for name in self._all_symbols_names:
            if name.startswith(base) and (len(name) <= len(base) + 4):
                 # Heuristic: starts with base and is not too long (avoid EURUSD -> EURUSDCAD)
                 return name
        return None

    def get_info(self, symbol: str) -> Optional[Any]:
        """Retorna info do símbolo com cache e circuit breaker"""
        real_name = self.resolve_name(symbol)
        if not real_name:
            return None
            
        if not self.is_available(real_name):
            return None
            
        # Return cached info if fresh
        # Note: symbol_info is mostly static, but spread changes. 
        # For spread, use get_tick or specific call. Here we cache "structure"
        if real_name in self._symbol_info_cache:
            return self._symbol_info_cache[real_name]
            
        info = mt5_exec(mt5.symbol_info, real_name)
        if info:
            self._symbol_info_cache[real_name] = info
            self._reset_failure(real_name)
            return info
        else:
            self._record_failure(real_name)
            return None

    def get_price(self, symbol: str) -> float:
        """Helper para pegar preço atual (Bid)"""
        tick = self.get_tick(symbol)
        return tick.bid if tick else 0.0

    def get_tick(self, symbol: str) -> Optional[Any]:
        """Retorna tick atual"""
        real_name = self.resolve_name(symbol)
        if not real_name or not self.is_available(real_name):
            return None
            
        tick = mt5_exec(mt5.symbol_info_tick, real_name)
        if tick:
            self._reset_failure(real_name)
            return tick
        else:
            self._record_failure(real_name)
            return None

    def is_available(self, symbol: str) -> bool:
        """Verifica Circuit Breaker"""
        if symbol in self._cooldowns:
            if time.time() < self._cooldowns[symbol]:
                return False
            else:
                del self._cooldowns[symbol]
                self._failures[symbol] = 0
                logger.info(f"Símbolo {symbol} recuperado do Circuit Breaker")
                return True
        return True

    def _record_failure(self, symbol: str):
        self._failures[symbol] = self._failures.get(symbol, 0) + 1
        if self._failures[symbol] >= self.MAX_FAILURES:
            self._cooldowns[symbol] = time.time() + self.COOLDOWN_SECONDS
            logger.error(f"CIRCUIT BREAKER: {symbol} pausado por {self.COOLDOWN_SECONDS}s")

    def _reset_failure(self, symbol: str):
        if self._failures.get(symbol, 0) > 0:
            self._failures[symbol] = 0

    def report_success(self, symbol: str):
        """Public wrapper for resetting failure count"""
        self._reset_failure(symbol)

    def report_failure(self, symbol: str):
        """Public wrapper for recording failure"""
        self._record_failure(symbol)

# Global Instance
symbol_manager = SymbolManager()
