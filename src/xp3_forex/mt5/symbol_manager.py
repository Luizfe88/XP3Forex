
import MetaTrader5 as mt5
import time
import logging
from threading import Lock
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime, timedelta, timezone
from xp3_forex.utils.mt5_utils import mt5_exec
from xp3_forex.core.settings import settings

logger = logging.getLogger(__name__)

class SymbolManager:
    """
    Singleton para gerenciamento de símbolos com cache inteligente,
    detecção automática de sufixos, Circuit Breaker e Filtros de Spread.
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
        
        # Spread Check State
        self._last_spread_log: Dict[str, float] = {}
        self._spread_ignore_count: int = 0
        self._last_summary_log: float = 0
        self._ignored_symbols_buffer: Set[str] = set()
        
        # Circuit Breaker State
        self._failures: Dict[str, int] = {}
        self._cooldowns: Dict[str, float] = {}
        
        # Configs
        self.MAX_FAILURES = 5
        self.COOLDOWN_SECONDS = 300
        self.CACHE_TTL = 3600  # 1 hour for static info
        self.ALL_SYMBOLS_REFRESH = 3600 * 24 # 24 hours
        
        # Trade Mode status for logging
        self._last_trademode_log: Dict[str, float] = {}
        
        logger.info("SymbolManager Singleton inicializado")

    def get_tradable_symbols(self, ignore_spread: bool = False) -> List[str]:
        """
        Retorna lista de símbolos negociáveis baseada na whitelist e filtro de spread.
        Loga resumo de símbolos ignorados periodicamente.
        """
        tradable_symbols = []
        
        # 1. Obter lista candidata (Whitelist + Market Watch se configurado)
        candidate_symbols = settings.symbols_list
        
        if not candidate_symbols or "ALL" in candidate_symbols or "MARKET_WATCH" in candidate_symbols:
            raw_symbols = self._get_market_watch_symbols()
            candidate_symbols = [s for s in raw_symbols if self._is_safe_category(s)]
        
        # 2. Filtrar
        for symbol in candidate_symbols:
            resolved_name = self.resolve_name(symbol)
            if not resolved_name:
                continue
            
            # --- Validação de Trade Mode (Full Only) ---
            if not self._check_trade_mode(resolved_name):
                continue

            if ignore_spread:
                tradable_symbols.append(resolved_name)
                # Mantém o buffer para o log de resumo mesmo se ignorarmos o filtro agora
                if not self._check_spread_and_log(resolved_name):
                    self._ignored_symbols_buffer.add(resolved_name)
            elif self._check_spread_and_log(resolved_name):
                tradable_symbols.append(resolved_name)
            else:
                self._ignored_symbols_buffer.add(resolved_name)

        # 3. Log Summary a cada 5 minutos
        now = time.time()
        if now - self._last_summary_log > 300 and self._ignored_symbols_buffer:
            logger.info(
                f"🚫 Spread Filter Summary: Ignored {len(self._ignored_symbols_buffer)} symbols due to high spread "
                f"(Ex: {list(self._ignored_symbols_buffer)[:5]}...)"
            )
            self._ignored_symbols_buffer.clear()
            self._last_summary_log = now
            
        return tradable_symbols

    def _is_safe_category(self, symbol: str) -> bool:
        """Verifica se a categoria é segura para trading automático"""
        category = self._categorize_symbol(symbol)
        
        # Get allowed categories from settings
        allowed = [c.strip().lower() for c in settings.ALLOWED_ASSET_CATEGORIES.split(",") if c.strip()]
        
        return category in allowed

    def check_spread(self, symbol: str) -> bool:
        """
        Verifica spread silenciosamente.
        Retorna True se aprovado, False se reprovado.
        Wrapper público para uso no Bot.
        """
        return self._check_spread_and_log(symbol)

    def _get_market_watch_symbols(self) -> List[str]:
        """Retorna todos os símbolos visíveis no Market Watch"""
        # A API oficial do MetaTrader5 Python não tem a função symbol_name.
        # A maneira correta é obter todos os símbolos e filtrar por 'select' (visible=True).
        try:
            # Obtém todos os símbolos disponíveis no terminal
            all_symbols = mt5.symbols_get()
            
            if all_symbols:
                # Filtra apenas os que estão selecionados no Market Watch
                visible_symbols = [s.name for s in all_symbols if s.select]
                logger.info(f"Market Watch scan: {len(visible_symbols)} symbols found.")
                return visible_symbols
            else:
                logger.warning("No symbols found in Market Watch (symbols_get returned None)")
                return []
                
        except Exception as e:
            logger.error(f"Error scanning Market Watch symbols: {e}")
            return []

    def _check_spread_and_log(self, symbol: str) -> bool:
        """
        Verifica spread silenciosamente.
        Retorna True se aprovado, False se reprovado.
        """
        info = self.get_symbol_info(symbol)
        if not info:
            return False
            
        # Calcular spread em pontos
        # Spread no MT5 info já vem em pontos (spread) ou flutuante (ask-bid)
        # info.spread é inteiro em pontos para a maioria, mas calcularemos manual para garantir
        # spread_points = (ask - bid) / point
        
        tick = self.get_tick(symbol)
        if not tick:
            return False
            
        spread_points = int((tick.ask - tick.bid) / info.point) if info.point > 0 else info.spread
        
        # 0. Verificar Trade Mode Primeiro
        if not self._check_trade_mode(symbol):
            return False

        # Categorizar e pegar threshold
        category = self._categorize_symbol(symbol)
        current_hour_utc = datetime.now(timezone.utc).hour
        max_spread = self._get_max_spread_for_category(category, current_hour_utc)
        
        if spread_points > max_spread:
            # Log apenas 1x por dia ou debug
            now = time.time()
            if now - self._last_spread_log.get(symbol, 0) > 86400: # 24h
                logger.info(f"🚫 Spread alto para {symbol} ({category}): {spread_points} > {max_spread}. Ignorando (Log diário).")
                self._last_spread_log[symbol] = now
            return False
            
        return True

    def _check_trade_mode(self, symbol: str) -> bool:
        """
        Verifica se o símbolo tem permissão total de negociação.
        Ignora se estiver em Close-Only ou Disabled.
        """
        real_name = self.resolve_name(symbol)
        if not real_name:
            return False
            
        # Busca sempre informação fresca para trade_mode pois pode mudar dinamicamente
        info = mt5_exec(mt5.symbol_info, real_name)
        if not info:
            return False
            
        if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            # Traduzir modo para string legível
            modes = {
                mt5.SYMBOL_TRADE_MODE_DISABLED: "DISABLED (Desativado)",
                mt5.SYMBOL_TRADE_MODE_LONGONLY: "LONG-ONLY (Apenas Compra)",
                mt5.SYMBOL_TRADE_MODE_SHORTONLY: "SHORT-ONLY (Apenas Venda)",
                mt5.SYMBOL_TRADE_MODE_CLOSEONLY: "CLOSE-ONLY (Apenas Fechamento)"
            }
            mode_str = modes.get(info.trade_mode, f"UNKNOWN ({info.trade_mode})")
            
            # Log de aviso (1x por hora para evitar spam)
            now = time.time()
            if now - self._last_trademode_log.get(symbol, 0) > 3600:
                logger.warning(f"⚠️ Ativo {symbol} temporariamente ignorado: Modo de negociação restrito pela corretora: {mode_str}")
                self._last_trademode_log[symbol] = now
            return False
            
        return True

    def _categorize_symbol(self, symbol: str) -> str:
        """Categoriza o símbolo automaticamente usando configurações institucionais"""
        s = symbol.upper()
        
        # 1. Crypto (Priority)
        if any(token in s for token in settings.TOKENS_CRYPTO):
            return "crypto"
            
        # 2. Metals
        if any(token in s for token in settings.TOKENS_METALS):
            return "metal"
            
        # 3. Indices
        if any(token in s for token in settings.TOKENS_INDICES):
            return "index"
            
        # 4. Exotics (Check BEFORE Majors to avoid overlapping like USDTHB)
        if any(token in s for token in settings.TOKENS_EXOTICS):
            return "exotic"
            
        # 5. Forex Majors/Minors
        # Check standard 6-char pairs (EURUSD, GBPJPY, etc.)
        majors_base = ["EUR", "GBP", "AUD", "NZD", "USD"]
        majors_quote = ["USD", "JPY", "CHF", "CAD"]
        
        # Símbolo limpo (remove sufixos)
        clean_symbol = s[:6]
        
        # Só é major se tiver base major AND quote major
        # E se não tiver sido pego pela lista de exóticos acima
        if any(m in clean_symbol for m in majors_base) and any(q in clean_symbol for q in majors_quote):
            return "major"
            
        return "exotic" # Fallback perigoso mas filtrado pelo spread alto

    def _get_max_spread_for_category(self, category: str, current_hour_utc: int = None) -> int:
        base_limits = {"major": 30, "minor": 60, "exotic": 120}
        limit = base_limits.get(category, 60)
        
        # Relaxar fora do horário de maior liquidez (08h–17h UTC)
        if current_hour_utc is not None:
            if not (8 <= current_hour_utc <= 17):
                limit = int(limit * 1.8)  # 30 → 54 para majors fora do horário
        
        return limit

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
                # logger.info(f"Símbolo resolvido: {base_symbol} -> {candidate}")
                self._symbol_name_map[base_symbol] = candidate
                return candidate
                
        # 4. Fallback: Search in all symbols (expensive, do lazily)
        # (Disabled to avoid massive lag on startup if not needed)
        # real_name = self._find_in_all_symbols(base_symbol)
        
        logger.warning(f"Símbolo não encontrado na corretora: {base_symbol}")
        return None

    def _check_symbol_exists(self, symbol: str) -> bool:
        """Verifica se símbolo existe e seleciona no Market Watch"""
        info = mt5_exec(mt5.symbol_info, symbol)
        if info is not None:
            if not info.select:
                mt5_exec(mt5.symbol_select, symbol, True)
            return True
        return False

    def initialize_symbol(self, symbol: str) -> bool:
        """
        Inicializa um símbolo:
        1. Resolve o nome correto.
        2. Seleciona no Market Watch.
        3. Verifica disponibilidade de dados.
        """
        resolved_name = self.resolve_name(symbol)
        if not resolved_name:
            return False
            
        # Tenta selecionar
        if not mt5_exec(mt5.symbol_select, resolved_name, True):
            logger.error(f"Falha ao selecionar {resolved_name} no Market Watch.")
            return False
            
        # Verifica se temos info básica
        info = mt5_exec(mt5.symbol_info, resolved_name)
        if not info:
            logger.error(f"Sem info para {resolved_name}")
            return False
            
        return True

    def get_tick(self, symbol: str) -> Optional[Any]:
        """Obtém o tick mais recente para o símbolo (com cache curto)"""
        real_name = self.resolve_name(symbol)
        if not real_name:
            return None
            
        # Circuit Breaker check
        if self.is_circuit_open(real_name):
            return None
            
        try:
            tick = mt5_exec(mt5.symbol_info_tick, real_name)
            if tick:
                self.report_success(real_name)
                return tick
            else:
                self.report_failure(real_name)
                return None
        except Exception:
            self.report_failure(real_name)
            return None

    def get_point(self, symbol: str) -> float:
        info = self.get_symbol_info(symbol)
        return info.point if info else 0.00001

    def get_tick_value(self, symbol: str) -> float:
        info = self.get_symbol_info(symbol)
        # Default fallback se não tiver info ou trade_tick_value
        if info and hasattr(info, 'trade_tick_value'):
             val = info.trade_tick_value
             if val > 0: return val
        return 1.0 # Fallback seguro

    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        real_name = self.resolve_name(symbol)
        if not real_name: return None
        
        # Cache info structure (static mostly)
        if real_name in self._symbol_info_cache:
            return self._symbol_info_cache[real_name]

        info = mt5_exec(mt5.symbol_info, real_name)
        if info:
            self._symbol_info_cache[real_name] = info
            return info
        return None

    def is_available(self, symbol: str) -> bool:
        """Verifica se o símbolo está disponível (sem Circuit Breaker)"""
        real_name = self.resolve_name(symbol)
        if not real_name: return False
        return not self.is_circuit_open(real_name)

    def is_circuit_open(self, symbol: str) -> bool:
        """Retorna True se o Circuit Breaker estiver aberto (bloqueado)"""
        if symbol in self._cooldowns:
            if time.time() < self._cooldowns[symbol]:
                return True
            else:
                del self._cooldowns[symbol]
                self._failures[symbol] = 0 # Reset após cooldown
                logger.info(f"Símbolo {symbol} recuperado do Circuit Breaker")
        return False

    def report_success(self, symbol: str):
        """Reporta sucesso na operação (reseta falhas)"""
        if symbol in self._failures:
            self._failures[symbol] = 0

    def report_failure(self, symbol: str):
        """Reporta falha na operação (incrementa contador)"""
        self._failures[symbol] = self._failures.get(symbol, 0) + 1
        if self._failures[symbol] >= self.MAX_FAILURES:
            logger.warning(f"Circuit Breaker ATIVADO para {symbol} por {self.COOLDOWN_SECONDS}s")
            self._cooldowns[symbol] = time.time() + self.COOLDOWN_SECONDS

# Global Instance
symbol_manager = SymbolManager()
