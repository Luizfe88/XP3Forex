<<<<<<< HEAD
# bot_forex.py - XP3 PRO FOREX BOT v4.2 INSTITUCIONAL
"""
🚀 XP3 PRO FOREX BOT - VERSÃO INSTITUCIONAL v4.2
✅ Integração PERFEITA com otimizador v6.0
✅ Usa parâmetros otimizados em tempo real
✅ Bollinger Squeeze integrado
✅ Auditoria completa de sinais
✅ Watchdog com auto-recuperação
✅ Gestão de risco avançada
✅ CORREÇÕES: Tratamento de erros, uso de getattr e get para config/dicts
✅ CORREÇÕES: Uso seguro de ELITE_CONFIG e SYMBOL_MAP
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import sys
import os
import time
import threading
import logging
import json
import signal
import traceback
import subprocess  # ✅ Added for dashboard launching
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock, RLock, Event
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# ===========================
# FIX WINDOWS ENCODING
# ===========================
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    os.environ['PYTHONIOENCODING'] = 'utf-8'

# ===========================
# IMPORTS
# ===========================
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    from xp3_forex.core import config as config
    from xp3_forex.utils import mt5_utils, indicators, calculations, data_utils as utils
    from xp3_forex.risk.validation import validate_and_create_order_forex, OrderParams, OrderSide
    from xp3_forex.analysis.news_filter from xp3_forex.analysis import news_filter
    from daily_analysis_logger import daily_logger # ✅ Land Trading Standard
    try:
        from risk_engine import block_manager, profit_optimizer, adaptive_manager
    except Exception:
        block_manager = None
        profit_optimizer = None
        adaptive_manager = None
    # ✅ ADAPTIVE ENGINE INTEGRATION - Sistema Adaptativo 4 Camadas
=======
# bot_forex.py - XP3 PRO FOREX BOT v4.2 INSTITUCIONAL
"""
🚀 XP3 PRO FOREX BOT - VERSÃO INSTITUCIONAL v4.2
✅ Integração PERFEITA com otimizador v6.0
✅ Usa parâmetros otimizados em tempo real
✅ Bollinger Squeeze integrado
✅ Auditoria completa de sinais
✅ Watchdog com auto-recuperação
✅ Gestão de risco avançada
✅ CORREÇÕES: Tratamento de erros, uso de getattr e get para config/dicts
✅ CORREÇÕES: Uso seguro de ELITE_CONFIG e SYMBOL_MAP
"""

import sys
import os
import time
import threading
import logging
import json
import signal
import traceback
import subprocess  # ✅ Added for dashboard launching
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock, RLock, Event
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# ===========================
# FIX WINDOWS ENCODING
# ===========================
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    os.environ['PYTHONIOENCODING'] = 'utf-8'

# ===========================
# IMPORTS
# ===========================
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    import config_forex as config
    import utils_forex as utils
    from validation_forex import validate_and_create_order_forex, OrderParams, OrderSide
    from news_filter import news_filter
    from daily_analysis_logger import daily_logger # ✅ Land Trading Standard
    try:
        from risk_engine import block_manager, profit_optimizer, adaptive_manager
    except Exception:
        block_manager = None
        profit_optimizer = None
        adaptive_manager = None
    # ✅ ADAPTIVE ENGINE INTEGRATION - Sistema Adaptativo 4 Camadas
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
    try:
        from adaptive_engine import AdaptiveEngine, SensorLayer, BrainLayer, MechanicLayer, EvolutionLayer, PanicMode
        adaptive_engine = AdaptiveEngine()
        print("✅ Adaptive Engine (4-Layer System) inicializado com sucesso!")
    except Exception as e:
        print(f"⚠️ Adaptive Engine não disponível: {e}")
        adaptive_engine = None
<<<<<<< HEAD
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False
    print("⚠️ Rich não disponível - usando logs simples")


def update_heartbeat():
    """Atualiza o arquivo de heartbeat para o watchdog"""
    try:
        Path("bot_heartbeat.timestamp").touch()
    except Exception as e:
        logger.debug(f"Erro ao atualizar heartbeat: {e}")
def heartbeat_writer():
    try:
        interval = max(1, int(getattr(config, 'HEARTBEAT_WRITE_INTERVAL', 5)))
    except Exception:
        interval = 5
    while not shutdown_event.is_set():
        try:
            update_heartbeat()
        except Exception:
            pass
        time.sleep(interval)
# ===========================
# LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOGS_DIR / "xp3_forex.log", encoding="utf-8"), # ✅ CORREÇÃO: Usa LOGS_DIR
    ],
)
logger = logging.getLogger("XP3_BOT")
try:
    from pythonjsonlogger import jsonlogger
    _json_handler = logging.FileHandler(config.LOGS_DIR / "xp3_forex.json.log", encoding="utf-8")
    _json_formatter = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s %(symbol)s %(score)s %(strategy)s")
    _json_handler.setFormatter(_json_formatter)
    logging.getLogger().addHandler(_json_handler)
except Exception:
    pass

# ✅ Land Trading: Garante diretório de logs
Path("analysis_logs").mkdir(exist_ok=True)

def safe_log(level, message, *args, **kwargs):
    """Loga apenas se dashboard desabilitado ou em arquivo"""
    if not getattr(config, 'ENABLE_DASHBOARD', False) or not RICH_AVAILABLE: # ✅ CORREÇÃO: getattr para ENABLE_DASHBOARD
        logger.log(level, message, *args, **kwargs)
    else:
        # Só loga em arquivo
        file_handler = logging.getLogger().handlers[1] if len(logging.getLogger().handlers) > 1 else None
        if file_handler:
            record = logging.LogRecord(
                name=logger.name, level=level, pathname="", lineno=0,
                msg=message, args=args, exc_info=None
            )
            file_handler.emit(record)

# ===========================
# VARIÁVEIS GLOBAIS
# ===========================
shutdown_event = Event()
daily_trades_per_symbol = defaultdict(int)
signal_history = deque(maxlen=getattr(config, 'SIGNAL_HISTORY_MAXLEN', 100)) # ✅ CORREÇÃO: SIGNAL_HISTORY_MAXLEN
signal_history_lock = Lock()
mt5_lock = utils.mt5_lock
mt5_exec = utils.mt5_exec
GLOBAL_MONITOR_CACHE = {} # ✅ Land Trading Transparency v5.0
GLOBAL_ACTIVE_SYMBOLS = [] # ✅ LISTA ATIVA DE SESSÃO (FILTRADA)
RECENT_ORDERS_CACHE = {} # ✅ NOVO: Cache para evitar ordens duplicadas
ORDER_COOLDOWN_TRACKER = {} # ✅ NOVO: {symbol: timestamp} para cooldown após fechar posição
ATTEMPT_COOLDOWN_TRACKER = {} # ✅ NOVO: Limita tentativas de análise/execução por candle
GLOBAL_SESSION_BLACKLIST = set() # ✅ NOVO: Blacklist de sessão (dinâmica)
ENTRY_IDEMPOTENCY = {}
LAST_DAILY_EXPORT_DATE = None
LAST_FRIDAY_SNAPSHOT_DATE = None
LAST_FRIDAY_AUTOCLOSE_DATE = None

# ===========================
# DAILY DRAWDOWN TRACKER v5.0
# ===========================
DAILY_PNL_TRACKER = {
    "start_equity": 0.0,
    "current_pnl": 0.0,
    "last_reset": None,
    "is_circuit_breaker_active": False,
    "current_pnl": 0.0,
    "last_reset": None,
    "is_circuit_breaker_active": False,
    "daily_loss_pct": 0.0
}

# ===========================
# KILL SWITCH / PAUSE TRACKER v5.2
# ===========================
PAUSED_SYMBOLS = {} # {symbol: {"until": datetime, "reason": str}}
KILL_SWITCH_TRACKER = {} # {symbol: {"win_rate": float, "last_check": datetime}}


# ===========================
# SYMBOL FILTERING (v5.0.6)
# ===========================
def filter_and_validate_symbols() -> List[str]:
    """
    ✅ v5.0.6: Filtra a lista de símbolos, aplica aliases e remove cabeçalhos.
    """
    sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
    sector_map = getattr(config, "SECTOR_MAP", None)

    symbols_to_test = None
    if isinstance(sector_map, dict) and sector in sector_map:
        try:
            symbols_to_test = list(sector_map.get(sector) or [])
        except Exception:
            symbols_to_test = None

    if not symbols_to_test:
        symbols_to_test = getattr(config, 'ALL_AVAILABLE_SYMBOLS', [])
    if not symbols_to_test:
        symbols_to_test = list(getattr(config, 'SYMBOL_MAP', []))
    
    if not symbols_to_test:
        logger.error("❌ Nenhuma lista de símbolos encontrada no config_forex.py")
        return []

    logger.info(f"🔍 Iniciando auditoria de {len(symbols_to_test)} símbolos no config...")
    valid_symbols = []
    
    for sym in symbols_to_test:
        # Se for um cabeçalho (ex: FOREX_MAJORS), ele falhará no normalize_symbol
        # mas normalize_symbol retorna o próprio nome se não encontrar.
        # Precisamos testar se o resultado existe no MT5.
        
        sym = str(sym).strip()
        if not sym:
            continue
        real_symbol = utils.normalize_symbol(sym)
        
        exists = mt5_exec(mt5.symbol_select, real_symbol, True)
            
        if exists:
            if real_symbol != sym:
                logger.info(f"✅ Símbolo '{sym}' mapeado para '{real_symbol}'")
            valid_symbols.append(real_symbol)
        else:
            logger.info(f"🔹 Símbolo '{sym}' ignorado (Não existe no MT5 ou é um cabeçalho de categoria)")

    return list(set(valid_symbols)) # Remove duplicatas

# ===========================
# ELITE CONFIG LOADER (v4.2)
# ===========================
def load_elite_config() -> Dict[str, Dict]:
    """
    ✅ v4.2: Carrega parâmetros otimizados do arquivo elite_settings_YYYYMMDD.txt

    Retorna:
        Dict: {symbol: {ema_short, ema_long, rsi_low, rsi_high, adx_threshold,
                        sl_atr, tp_atr, bb_squeeze_threshold, min_score}}
    """
    import re

    output_dir = Path(getattr(config, 'OPTIMIZER_OUTPUT', 'optimizer_output')) # ✅ CORREÇÃO: getattr para OPTIMIZER_OUTPUT
    files = list(output_dir.glob("elite_settings_*.txt"))

    if not files:
        logger.warning("⚠️ Nenhum arquivo elite_settings encontrado - usando config padrão")
        return {}

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"📂 Carregando parâmetros otimizados de: {latest_file.name}")

    elite_config = {}

    try:
        import ast
        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.read()

            # Extrai o bloco ELITE_CONFIG = {...}
            match = re.search(r"ELITE_CONFIG = (\{.*?\})", content, re.DOTALL)
            if match:
                config_str = match.group(1)
                # ✅ v5.0.1: Usar ast.literal_eval para maior robustez com dicts Python
                elite_config = ast.literal_eval(config_str)
            else:
                logger.error("❌ Não foi possível extrair ELITE_CONFIG do arquivo.")


        logger.info(f"✅ Carregados parâmetros de {len(elite_config)} símbolos")

        # Log dos símbolos carregados
        for sym, params in elite_config.items():
            logger.debug(f"  {sym}: EMA({params.get('ema_short', '?')}/{params.get('ema_long', '?')}) "
                        f"RSI({params.get('rsi_low', '?')}/{params.get('rsi_high', '?')}) "
                        f"ADX>{params.get('adx_threshold', '?')}) "
                        f"MinScore:{params.get('min_score', '?')}") # ✅ CORREÇÃO: Adiciona min_score ao log

    except Exception as e:
        logger.error(f"❌ Erro ao carregar elite_config: {e}", exc_info=True) # ✅ CORREÇÃO: Adiciona exc_info
        elite_config = {} # Garante que elite_config seja um dict vazio em caso de erro
    
    try:
        if not elite_config:
            from config_forex import FOREX_PAIRS
            if isinstance(FOREX_PAIRS, dict) and FOREX_PAIRS:
                logger.warning("⚠️ Elite vazio. Usando fallback de FOREX_PAIRS do config.")
                elite_config = dict(FOREX_PAIRS)
                logger.info(f"✅ Fallback aplicado: {len(elite_config)} símbolos")
    except Exception:
        pass
    return elite_config

# Carrega na inicialização
ELITE_CONFIG = load_elite_config()

# ===========================
# DATACLASSES
# ===========================
@dataclass
class SignalAnalysis:
    """Registro de análise de sinal"""
    timestamp: datetime
    symbol: str
    signal: str
    strategy: str
    score: float
    rejected: bool
    rejection_reason: str
    indicators: dict = field(default_factory=dict)

@dataclass
class Position:
    ticket: int
    symbol: str
    side: str
    volume: float
    entry_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    pips: float
    open_time: datetime
    breakeven_moved: bool = False
    trailing_active: bool = False

@dataclass
class TradeMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0

    @property
    def win_rate(self) -> float:
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return abs(self.total_profit / self.total_loss) if self.total_loss != 0 else 0.0

# ===========================
# BOT STATE
# ===========================
class BotState:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BotState, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self._internal_lock = Lock()
        self._indicators: Dict[str, dict] = {}
        self._top_pairs: List[str] = []
        self._positions: Dict[int, Position] = {}
        self._trading_paused: bool = False
        self._pause_reason: str = ""
        self._last_daily_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._monitoring_status: Dict[str, dict] = {} # ✅ v5.0.2: {symbol: {status, reason, ml_score, timestamp}}

    def update_indicators(self, indicators: Dict[str, dict], top_pairs: List[str]):
        with self._internal_lock:
            self._indicators = indicators.copy()
            self._top_pairs = list(top_pairs)

    def update_monitoring(self, symbol: str, status: str, reason: str, ml_score: float):
        with self._internal_lock:
            # 1. Atualiza estado interno
            self._monitoring_status[symbol] = {
                "status": status,
                "reason": reason,
                "ml_score": ml_score,
                "timestamp": datetime.now()
            }
            
            # 2. ✅ CORREÇÃO: Sincroniza com o Cache Global do Painel/Log
            # Isso garante que o status visual mude instantaneamente
            global GLOBAL_MONITOR_CACHE
            if symbol not in GLOBAL_MONITOR_CACHE:
                GLOBAL_MONITOR_CACHE[symbol] = {}
            
            GLOBAL_MONITOR_CACHE[symbol].update({
                "status": status,
                "reason": reason,
                "ml_score": ml_score,
                "timestamp": datetime.now()
            })

    def get_monitoring_status(self) -> Dict[str, dict]:
        with self._internal_lock:
            return self._monitoring_status.copy()

    def get_indicators(self, symbol: str) -> dict:
        with self._internal_lock:
            return self._indicators.get(symbol, {}).copy()

    def get_top_pairs(self) -> List[str]:
        with self._internal_lock:
            return self._top_pairs.copy()

    def update_position(self, ticket: int, position: Position):
        with self._internal_lock:
            self._positions[ticket] = position

    def remove_position(self, ticket: int):
        with self._internal_lock:
            self._positions.pop(ticket, None)

    def get_positions(self) -> Dict[int, Position]:
        with self._internal_lock:
            return self._positions.copy()

    def pause_trading(self, reason: str):
        with self._internal_lock:
            self._trading_paused = True
            self._pause_reason = reason
            logger.warning(f"⏸️ Trading pausado: {reason}")

    def resume_trading(self):
        with self._internal_lock:
            self._trading_paused = False
            self._pause_reason = ""
            safe_log(logging.INFO, "▶️ Trading retomado")

    def is_paused(self) -> Tuple[bool, str]:
        with self._internal_lock:
            return self._trading_paused, self._pause_reason

    def check_and_reset_daily_limits(self):
        """Verifica e reseta contadores diários."""
        now = datetime.now()
        if now.day != self._last_daily_reset.day or now.month != self._last_daily_reset.month or now.year != self._last_daily_reset.year:
            with self._internal_lock:
                global daily_trades_per_symbol
                daily_trades_per_symbol.clear()
                self._last_daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
                safe_log(logging.INFO, "🔄 Limites diários de trades resetados.")

bot_state = BotState()
metrics = TradeMetrics()

# ===========================
# WATCHDOG
# ===========================
class ThreadWatchdog:
    """Monitora threads críticas e reinicia se travarem"""
    def __init__(self):
        self.threads = {}
        self.last_heartbeat = {}
        self.max_silence = 300 # Default
        self.lock = threading.Lock()
        self.custom_timeouts = {
            "SlowLoop": max(getattr(config, 'SLOW_LOOP_INTERVAL', 300) * 3, 450),
            "FastLoop": max(
                getattr(config, 'FAST_LOOP_INTERVAL', 15) * 5,
                getattr(config, 'FAST_LOOP_STALE_SECONDS', 120) + 60,
                getattr(config, 'FAST_LOOP_WATCHDOG_TIMEOUT', 0),
                300
            ),
            "Panel": max(getattr(config, 'PANEL_UPDATE_INTERVAL', 5) * 8, 40)
        }

    def register_thread(self, name: str, target_func, args=()):
        with self.lock:
            self.threads[name] = {
                "target": target_func,
                "args": args,
                "thread": None,
                "restarts": 0
            }
            self.last_heartbeat[name] = time.time()

    def heartbeat(self, name: str):
        with self.lock:
            self.last_heartbeat[name] = time.time()

    def check_and_restart(self):
        with self.lock:
            current_time = time.time()

            for name, info in self.threads.items():
                thread = info["thread"]
                last_beat = self.last_heartbeat.get(name, 0)
                silence_time = current_time - last_beat
                max_allowed = self.custom_timeouts.get(name, self.max_silence)

                is_dead = (thread is None or not thread.is_alive())
                is_silent = silence_time > max_allowed

                if is_silent and not is_dead:
                    safe_log(logging.WARNING, f"⏳ Thread {name} silenciosa há {int(silence_time)}s (timeout {int(max_allowed)}s).")
                    continue

                if is_dead:
                    info["restarts"] += 1
                    safe_log(logging.ERROR, f"💀 Thread {name} MORTA! Reiniciando... (#{info['restarts']})")

                    try:
                        new_thread = threading.Thread(
                            target=info["target"],
                            args=info["args"],
                            daemon=True,
                            name=name
                        )
                        new_thread.start()
                        info["thread"] = new_thread
                        self.last_heartbeat[name] = current_time
                        safe_log(logging.INFO, f"✅ Thread {name} reiniciada!")
                        # ✅ REQUISITO: Controle de Reinício (2s Delay)
                        time.sleep(2)
                    except Exception as e:
                        logger.critical(f"❌ FALHA ao reiniciar {name}: {e}")
                        if info["restarts"] >= getattr(config, 'MAX_THREAD_RESTARTS', 3): # ✅ CORREÇÃO: MAX_THREAD_RESTARTS
                            shutdown_event.set()

watchdog = ThreadWatchdog()

# ===========================
# SIGNAL ANALYSIS LOGGER
# ===========================
def log_signal_analysis(
    symbol: str,
    signal: str,
    strategy: str,
    score: float,
    rejected: bool,
    reason: str,
    indicators: dict,
    ml_score: float = 0.0,
    is_baseline: bool = False,
    session_name: Optional[str] = None
):
    """Registra análise de sinal"""
    with signal_history_lock:
        signal_history.append(SignalAnalysis(
            timestamp=datetime.now(),
            symbol=symbol,
            signal=signal or "NONE",
            strategy=strategy or "N/A",
            score=score,
            rejected=rejected,
            rejection_reason=reason,
            indicators={
                "rsi": indicators.get("rsi", 0),
                "adx": indicators.get("adx", 0),
                "spread_pips": indicators.get("spread_pips", 0),
                "volume_ratio": indicators.get("volume_ratio", 0),
                "ema_trend": indicators.get("ema_trend", "N/A"),
                "bb_width": indicators.get("bb_width", 0),
                "close": indicators.get("close", 0) # ✅ NOVO: Adiciona preço de fechamento
            }
        ))
    
    # ✅ Land Trading: Log em arquivo instantâneo
    try:
        user = os.getenv("XP3_OPERATOR", "XP3_BOT")
        context = {
            "session": (session_name or "UNKNOWN"),
            "action": ("ACEITE" if not rejected else "BLOQUEIO"),
            "ml_score": f"{ml_score:.1f}",
            "source": "FastLoop"
        }
        daily_logger.log_analysis(
            symbol=symbol,
            signal=signal or "NONE",
            strategy=strategy or "XP3_PRO_V4",
            score=score,
            rejected=rejected,
            reason=reason,
            indicators=indicators,
            ml_score=ml_score,
            is_baseline=is_baseline,
            user=user,
            context=context
        )
    except Exception as e:
        logger.error(f"⚠️ Erro ao registrar log diário: {e}")

    try:
        sess = session_name
        if not sess:
            sess = (utils.get_current_trading_session() or {}).get("name", "UNKNOWN")
        executed = not rejected
        utils.update_session_metrics(sess, executed=executed, rejected=rejected, reason=reason)
    except Exception:
        pass
    
    # ✅ Land Trading: Atualiza GLOBAL_MONITOR_CACHE apenas se não for monitoramento básico
    # (Evita sobrescrever dados mais completos do loop principal)
        GLOBAL_MONITOR_CACHE[symbol] = {
            "status": "🟢 SINAL" if not rejected else "🔵 MONITORANDO",
            "reason": reason,
            "ml_score": ml_score,
            "is_baseline": is_baseline,
            "timestamp": datetime.now()
        }

# ===========================
# PAUSE BOT HELPER v5.2
# ===========================
def pause_bot(symbol: str, minutes: int, reason: str):
    """Pausa o bot para um símbolo específico por X minutos."""
    until = datetime.now() + timedelta(minutes=minutes)
    PAUSED_SYMBOLS[symbol] = {"until": until, "reason": reason}
    logger.warning(f"⏸️ BOT PAUSED for {symbol} until {until.strftime('%H:%M')} | Reason: {reason}")
    bot_state.update_monitoring(symbol, "⏸️ PAUSADO", reason, 0.0)
    try:
        utils.send_telegram_message(f"🚨 Mudança de Regime de Mercado\n{symbol}: {reason}\n⏸️ Pausado por {minutes} min (até {until.strftime('%H:%M')}).")
    except Exception:
        pass
def _apply_pause_requests():
    try:
        import json, os, time
        path = os.path.join("data", "pause_requests.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        changed = False
        for sym, data in (payload or {}).items():
            try:
                until_ts = int(data.get("until", 0))
                minutes = int(data.get("minutes", 10))
                reason = str(data.get("reason", "EXECUTION_QUALITY"))
                if until_ts > int(time.time()):
                    pause_bot(sym, minutes, reason)
                    changed = True
            except Exception:
                pass
        if changed:
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception:
        pass
def perform_system_checks(current_session: dict, iteration_count: int) -> tuple:
    try:
        total_positions = mt5_exec(mt5.positions_total)
        try:
            total_positions = int(total_positions or 0)
        except Exception:
            total_positions = 0
        try:
            gm = getattr(config, 'MAX_GLOBAL_ALGO_ORDERS', 3)
            global_max = int(gm if gm is not None else 3)
        except Exception:
            global_max = 3
        if total_positions >= global_max:
            return False, {"sleep": 1, "reason": f"Limite Global de Ordens ({total_positions}/{global_max}) atingido. Entradas bloqueadas."}
        is_rollover, rollover_reason = utils.is_rollover_period()
        if is_rollover:
            return False, {"sleep": 10, "reason": rollover_reason}
        global DAILY_PNL_TRACKER
        account_info = mt5_exec(mt5.account_info)
        if account_info:
            current_equity = account_info.equity
            current_balance = account_info.balance
            today = datetime.now().date()
            if DAILY_PNL_TRACKER["last_reset"] != today:
                DAILY_PNL_TRACKER["start_equity"] = current_balance
                DAILY_PNL_TRACKER["last_reset"] = today
                DAILY_PNL_TRACKER["is_circuit_breaker_active"] = False
            if DAILY_PNL_TRACKER["start_equity"] > 0:
                daily_pnl = current_equity - DAILY_PNL_TRACKER["start_equity"]
                daily_loss_pct = abs(min(0, daily_pnl)) / DAILY_PNL_TRACKER["start_equity"]
                DAILY_PNL_TRACKER["current_pnl"] = daily_pnl
                DAILY_PNL_TRACKER["daily_loss_pct"] = daily_loss_pct
                max_dd = getattr(config, 'MAX_DAILY_DRAWDOWN_PCT', 0.02)
                if daily_loss_pct >= max_dd:
                    if not DAILY_PNL_TRACKER["is_circuit_breaker_active"]:
                        DAILY_PNL_TRACKER["is_circuit_breaker_active"] = True
                        logger.critical(f"CIRCUIT BREAKER ATIVO! Perda diária ({daily_loss_pct:.2%}) atingiu limite ({max_dd:.2%}). Bot pausado até amanhã.")
                    return False, {"sleep": 60, "reason": "Circuit breaker ativo"}
        if not GLOBAL_ACTIVE_SYMBOLS:
            return False, {"sleep": 5, "reason": "GLOBAL_ACTIVE_SYMBOLS vazia"}
        return True, {}
    except Exception as e:
        logger.error(f"perform_system_checks erro: {e}")
        return True, {}
def _compute_symbols_to_analyze(current_session: dict) -> list:
    symbols_to_analyze = GLOBAL_ACTIVE_SYMBOLS.copy()
    if current_session.get("name") == "ASIAN":
        priority_pairs = getattr(config, 'ASIAN_PRIORITY_PAIRS', [])
        symbols_to_analyze.sort(key=lambda s: s not in priority_pairs)
    return symbols_to_analyze
def manage_open_positions(active_tickets_map: dict) -> dict:
    try:
        current_positions = mt5_exec(mt5.positions_get)
        my_positions = [p for p in current_positions if p.magic == getattr(config, 'MAGIC_NUMBER', 123456)] if current_positions else []
        current_tickets_map = {p.ticket: p.symbol for p in my_positions}
        closed_tickets = set(active_tickets_map.keys()) - set(current_tickets_map.keys())
        for ticket in closed_tickets:
            symbol = active_tickets_map[ticket]
            ORDER_COOLDOWN_TRACKER[symbol] = time.time()
            try:
                from datetime import timedelta
                deals = mt5_exec(mt5.history_deals_get, datetime.now() - timedelta(days=2), datetime.now())
                magic = int(getattr(config, 'MAGIC_NUMBER', 123456))
                pos_deals = [d for d in (deals or []) if int(getattr(d, "position_id", 0) or 0) == int(ticket) and int(getattr(d, "magic", magic) or magic) == magic]
                if pos_deals:
                    total_profit = sum(float(getattr(d, "profit", 0.0) or 0.0) for d in pos_deals)
                    open_deals = [d for d in pos_deals if getattr(d, "entry", None) == mt5.DEAL_ENTRY_IN]
                    close_deals = [d for d in pos_deals if getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT]
                    side = "BUY" if (open_deals and getattr(open_deals[0], "type", None) == mt5.DEAL_TYPE_BUY) else "SELL"
                    open_price = float(getattr(open_deals[0], "price", 0.0) or 0.0) if open_deals else 0.0
                    close_price = float(getattr(close_deals[-1], "price", 0.0) or 0.0) if close_deals else 0.0
                    try:
                        if block_manager:
                            block_manager.on_trade_close(symbol, open_price, close_price, float(getattr(open_deals[0], "volume", 0.0) or 0.0) if open_deals else 0.0, total_profit, side)
                    except Exception:
                        pass
                    pip_size = utils.get_pip_size(symbol)
                    pips = 0.0
                    if pip_size > 0 and open_price > 0 and close_price > 0:
                        pips = (close_price - open_price) / pip_size if side == "BUY" else (open_price - close_price) / pip_size
                    msg = (
                        f"🏁 Saída de posição\n"
                        f"Par: {symbol}\n"
                        f"Ticket: {ticket}\n"
                        f"Lado: {side}\n"
                        f"PnL: ${total_profit:+.2f}\n"
                        f"Pips: {pips:+.1f}\n"
                        f"Entrada: {open_price:.5f}\n"
                        f"Saída: {close_price:.5f}\n"
                        f"Hora: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    utils.send_telegram_alert(msg, "SUCCESS" if total_profit >= 0 else "WARNING")
                    try:
                        if adaptive_manager:
                            strat = ""
                            try:
                                if open_deals and getattr(open_deals[0], "comment", None):
                                    c = str(getattr(open_deals[0], "comment", ""))
                                    if "XP3_" in c:
                                        strat = c.split("XP3_")[1].split()[0]
                            except Exception:
                                strat = ""
                            adaptive_manager.on_trade_close(symbol, strat, float(total_profit or 0.0))
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"❌ Erro ao enviar alerta de saída para {symbol}/{ticket}: {e}")
        return current_tickets_map
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar rastreio de posições: {e}")
        return active_tickets_map
def attempt_entry(symbol: str, signal: str, strategy: str, ind: dict, params: dict, current_session: dict, iteration_count: int, final_score: float, ml_score: float, ml_override_risk_mult: float, ml_override_used: bool, rejection_stats: dict) -> bool:
    try:
        base_sl = float(params.get("sl_atr", getattr(config, 'DEFAULT_STOP_LOSS_ATR_MULTIPLIER', 1.5)))
        base_tp = float(params.get("tp_atr", getattr(config, 'DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER', 3.0)))
        if adaptive_manager:
            sl_atr_mult, tp_atr_mult = adaptive_manager.get_current_params(symbol, strategy or "", base_sl, base_tp)
        else:
            sl_atr_mult, tp_atr_mult = base_sl, base_tp
        try:
            regime = utils.get_volatility_regime(symbol, ind.get("df") if isinstance(ind, dict) else None)
            if regime == "HIGH":
                tp_atr_mult = min(getattr(config, "MAX_TP_ATR_MULT", 6.0), tp_atr_mult + 0.5)
                sl_atr_mult = min(getattr(config, "MAX_SL_ATR_MULT", 3.5), sl_atr_mult + 0.3)
            elif regime == "LOW":
                tp_atr_mult = max(getattr(config, "MIN_TP_ATR_MULT", 2.0), tp_atr_mult - 0.5)
        except Exception:
            pass
        tick = mt5_exec(mt5.symbol_info_tick, symbol)
        if not tick:
            return False
        entry_price = tick.ask if signal == "BUY" else tick.bid
        volume = utils.calculate_position_size_atr_forex(
            symbol, entry_price, ind.get("atr_pips", 0), sl_atr_mult=sl_atr_mult, risk_multiplier=ml_override_risk_mult
        )
        if volume <= 0:
            return False
        sl, tp = utils.calculate_dynamic_levels(symbol, entry_price, ind, sl_atr_mult, tp_atr_mult, signal=signal)
        if sl <= 0 or tp <= 0:
            return False
        current_positions = mt5_exec(mt5.positions_get)
        exp_ok, exp_msg = utils.check_currency_exposure(current_positions)
        if not exp_ok:
            bot_state.update_monitoring(symbol, "🟠 EXPOSIÇÃO", exp_msg, ml_score)
            return False
        tot_ok, tot_msg = utils.check_total_exposure_limit(
            pending_symbol=symbol,
            pending_volume=volume,
            pending_side=signal
        )
        if not tot_ok:
            bot_state.update_monitoring(symbol, "🟠 EXPOSIÇÃO TOTAL", tot_msg, ml_score)
            return False
        hard_global_cap = int(getattr(config, 'MAX_GLOBAL_ALGO_ORDERS', 6))
        bot_positions_count = 0
        if current_positions:
            for p in current_positions:
                if p.magic == getattr(config, 'MAGIC_NUMBER', 123456):
                    bot_positions_count += 1
        if bot_positions_count >= hard_global_cap:
            reason = f"Limite global duro atingido ({bot_positions_count}/{hard_global_cap})"
            bot_state.update_monitoring(symbol, "🛑 LIMITE GLOBAL DURO", reason, ml_score)
            return False
        max_global_positions = int(getattr(config, 'MAX_SYMBOLS', hard_global_cap))
        try:
            session_name = (current_session or {}).get("name")
            session_map = getattr(config, "SESSION_MAX_POSITIONS", {})
            if isinstance(session_map, dict) and session_name in session_map:
                max_global_positions = int(session_map.get(session_name, max_global_positions))
        except Exception:
            pass
        if bot_positions_count >= max_global_positions:
            reason = f"Limite de posições simultâneas ({bot_positions_count}/{max_global_positions})"
            bot_state.update_monitoring(symbol, "🛑 LIMITE GLOBAL", reason, ml_score)
            return False
        candle_time = ind.get("time")
        candle_key = f"{symbol}|{signal}|{candle_time}"
        last_key_time = ENTRY_IDEMPOTENCY.get(candle_key)
        if last_key_time and (datetime.now() - last_key_time).total_seconds() < 3600:
            bot_state.update_monitoring(symbol, "⏳ COOLDOWN", "Idempotência (candle)", ml_score)
            return False
        ENTRY_IDEMPOTENCY[candle_key] = datetime.now()
        if len(ENTRY_IDEMPOTENCY) > 5000:
            cutoff = datetime.now() - timedelta(hours=6)
            ENTRY_IDEMPOTENCY_KEYS = list(ENTRY_IDEMPOTENCY.keys())
            for k in ENTRY_IDEMPOTENCY_KEYS:
                if ENTRY_IDEMPOTENCY.get(k) and ENTRY_IDEMPOTENCY[k] < cutoff:
                    del ENTRY_IDEMPOTENCY[k]
        order_params = OrderParams(
            symbol=symbol, side=OrderSide.BUY if signal == "BUY" else OrderSide.SELL,
            volume=volume, entry_price=entry_price, sl=sl, tp=tp,
            comment=f"XP3_{strategy}", magic=getattr(config, 'MAGIC_NUMBER', 123456)
        )
        RECENT_ORDERS_CACHE[symbol] = datetime.now()
        success = False
        try:
            success, message, ticket = validate_and_create_order_forex(order_params)
            if success:
                daily_trades_per_symbol[symbol] += 1
                side_str = "COMPRA 🟢" if signal == "BUY" else "VENDA 🔴"
                msg_tele = (
                    f"🚀 <b>XP3 PRO: Ordem Executada</b>\n\n"
                    f"🆔 Ativo: <b>{symbol}</b>\n"
                    f"📡 Sinal: {side_str}\n"
                    f"💵 Preço: {entry_price:.5f}\n"
                    f"🛑 SL: {sl:.5f} | 🎯 TP: {tp:.5f}\n"
                    f"📊 Score: {final_score:.1f} (ML: {ml_score:.1f}){' | 🟣 OVERRIDE' if ml_override_used else ''}\n"
                    f"🧪 Estratégia: {strategy}"
                )
                utils.send_telegram_alert(msg_tele)
                try:
                    utils.record_order_open(
                        symbol=symbol,
                        side=("BUY" if signal == "BUY" else "SELL"),
                        volume=volume,
                        entry_price=entry_price,
                        sl=sl,
                        tp=tp,
                        order_id=int(ticket or 0),
                        comment=f"XP3_{strategy}"
                    )
                except Exception:
                    pass
                exec_reason = "EXECUTADA (OVERRIDE)" if ml_override_used else "EXECUTADA"
                bot_state.update_monitoring(symbol, "🔴 OPERANDO", exec_reason, ml_score)
                log_signal_analysis(symbol, signal, strategy, final_score, False, exec_reason, ind, session_name=current_session.get("name"))
                return True
            else:
                if symbol in RECENT_ORDERS_CACHE:
                    del RECENT_ORDERS_CACHE[symbol]
                logger.error(f"❌ Falha para {symbol}: {message}")
                bot_state.update_monitoring(symbol, "⚠️ ERRO", message[:20], ml_score)
                time.sleep(10)
                return False
        except Exception as exc:
            if symbol in RECENT_ORDERS_CACHE:
                del RECENT_ORDERS_CACHE[symbol]
            logger.critical(f"💀 Erro na execução para {symbol}: {exc}", exc_info=True)
            time.sleep(30)
            return False
    except Exception as e:
        logger.error(f"attempt_entry erro: {e}")
        return False

# ===========================
# CHECK FOR SIGNALS (v4.2)
# ===========================
def check_for_signals(symbol: str, current_session: dict = None) -> Tuple[Optional[str], dict, Optional[str], str]:
    """
    ✅ v5.0.2: Estratégia Híbrida com Negative Edge Reporting.
    ✅ Horário de Ouro: Regras dinâmicas aplicadas.
    ✅ v6.0: Adaptive Engine Integration - Sistema Adaptativo 4 Camadas
    Retorna: (signal, indicators, strategy, reason)
    """

    # ✅ ADAPTIVE ENGINE: Processa dados de mercado em tempo real
    if adaptive_engine and getattr(config, 'ENABLE_ADAPTIVE_ENGINE', True):
        try:
            # Coleta dados de mercado para o símbolo
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'session': current_session.get('name', 'NORMAL') if current_session else 'NORMAL',
                'price_data': utils.get_price_data(symbol),  # Função auxiliar para obter dados de preço
                'volatility': utils.get_volatility(symbol),   # Função auxiliar para obter volatilidade
                'volume': utils.get_volume_data(symbol)       # Função auxiliar para obter volume
            }
            
            # Processa através do Adaptive Engine
            adaptive_result = adaptive_engine.process_market_data(market_data)
            
            # Verifica Panic Mode
            if adaptive_result.get('panic_mode_active'):
                logger.warning(f"🚨 PANIC MODE ATIVO para {symbol}: {adaptive_result.get('panic_reason')}")
                return None, None, None, "Panic Mode Ativo - Operações Suspensas"
            
            # Aplica ajustes de parâmetros sugeridos pelo sistema adaptativo
            if adaptive_result.get('parameter_adjustments'):
                logger.info(f"🧠 Adaptive Engine ajustando parâmetros para {symbol}")
                # Os ajustes serão aplicados nos parâmetros abaixo
                
        except Exception as e:
            logger.error(f"❌ Erro no Adaptive Engine para {symbol}: {e}")
            # Continua com parâmetros padrão em caso de erro

    # 1. Busca parâmetros otimizados
    params = ELITE_CONFIG.get(symbol, {})

    ema_short = params.get("ema_short", getattr(config, 'EMA_SHORT_PERIOD', 20))
    ema_long = params.get("ema_long", getattr(config, 'EMA_LONG_PERIOD', 50))
    rsi_period = params.get("rsi_period", getattr(config, 'RSI_PERIOD', 14))
    adx_period = params.get("adx_period", getattr(config, 'ADX_PERIOD', 14))
    # ADX threshold por símbolo (tendência baixa pode operar com corte menor)
    base_adx_threshold = params.get("adx_threshold", getattr(config, 'ADX_THRESHOLD', 25))
    try:
        low_trend_map = getattr(config, 'LOW_TREND_ADX_THRESHOLDS', {})
        adx_threshold = low_trend_map.get(symbol, base_adx_threshold)
    except Exception:
        adx_threshold = base_adx_threshold
    
    # --- AJUSTE DE SENSIBILIDADE RSI (SESSÃO ASIÁTICA) ---
    session_name = current_session.get("name", "NORMAL") if current_session else "NORMAL"
    rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
    rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
    bb_period = params.get("bb_period", getattr(config, 'BB_PERIOD', 20))
    bb_dev = params.get("bb_dev", getattr(config, 'BB_DEVIATION', 2.0))
    bb_squeeze_threshold = params.get("bb_squeeze_threshold", getattr(config, 'BB_SQUEEZE_THRESHOLD', 0.015))

    # 2. Obtém indicadores
    ind = utils.get_indicators_forex(
        symbol,
        ema_short=ema_short,
        ema_long=ema_long,
        rsi_period=rsi_period,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        adx_period=adx_period,
        bb_period=bb_period,
        bb_dev=bb_dev
    )

    # ✅ REQUISITO: Verificação nula + erro em dicionário
    if ind is None or (isinstance(ind, dict) and ind.get("error")):
        error_msg = ind.get("message", "Erro desconhecido") if isinstance(ind, dict) else "Retorno nulo de indicadores"
        return "NONE", None, "NONE", f"Erro nos indicadores para {symbol}: {error_msg}"

    # ===========================
    # ✅ v5.0: EMA 200 MACRO TREND FILTER
    # ===========================
    if getattr(config, 'ENABLE_EMA_200_FILTER', True):
        ema_200_data = utils.get_ema_200(symbol)
        if not ema_200_data.get("error"):
            ind["ema_200"] = ema_200_data.get("ema_200")
            ind["ema_200_trend"] = ema_200_data.get("trend_direction")
            ind["is_above_ema_200"] = ema_200_data.get("is_above_ema")
        else:
            # Se erro, não bloqueia mas loga
            logger.warning(f"⚠️ EMA 200 indisponível para {symbol}: {ema_200_data.get('message')}")
            ind["ema_200"] = None
            ind["ema_200_trend"] = "UNKNOWN"
            ind["is_above_ema_200"] = None

    # ===========================
    # ✅ v5.0: ADX MINIMUM STRENGTH FILTER (Ajustado para Land Trading)
    # ===========================
    adx_now = ind.get("adx", 0)
    # Apenas loga aviso, não retorna None aqui para permitir Reversão
    adx_min_strength = getattr(config, 'ADX_MIN_STRENGTH', 20)
    is_low_adx = adx_now < adx_min_strength
    ind["adx_low"] = is_low_adx

    # 3. Filtros Básicos (Spread e Volume)
    spread_points = ind.get("spread_points", ind.get("spread_pips", 0))
    
    # --- REGRAS DINÂMICAS POR SESSÃO ---
    session_name = current_session.get("name", "NORMAL") if current_session else "NORMAL"
    
    symbol_upper = symbol.upper()
    crypto_list = getattr(config, 'TOKENS_CRYPTO', ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"])
    indices_list = getattr(config, 'TOKENS_INDICES', ["US30", "NAS100", "USTEC", "DE40", "GER40", "GER30", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"])
    metals_list = getattr(config, 'TOKENS_METALS', ["XAU", "XAG", "GOLD", "SILVER"])
    exotics_list = getattr(config, 'TOKENS_EXOTICS', ["TRY", "ZAR", "MXN", "RUB", "CNH", "PLN", "HUF", "CZK", "DKK", "NOK", "SEK"])

    is_crypto = any(c in symbol_upper for c in crypto_list)
    is_indices = any(i in symbol_upper for i in indices_list)
    is_metals = any(m in symbol_upper for m in metals_list)
    is_exotic = any(x in symbol_upper for x in exotics_list)

    if is_crypto:
        max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
        spread_check = spread_points > max_spread
    elif is_indices:
        max_spread = getattr(config, 'MAX_SPREAD_INDICES', 600)
        spread_check = spread_points > max_spread
    elif is_metals:
        max_spread = getattr(config, 'MAX_SPREAD_METALS', 80)
        spread_check = spread_points > max_spread
    elif is_exotic:
        max_spread = getattr(config, 'MAX_SPREAD_EXOTICS', 8000)
        spread_check = spread_points > max_spread
    else:
        max_spread = getattr(config, 'MAX_SPREAD_ACCEPTABLE', 25)
        spread_check = spread_points > max_spread

    if not (is_crypto or is_indices or is_metals or is_exotic):
        if session_name == "GOLDEN":
            max_spread *= (1 + getattr(config, 'GOLDEN_SPREAD_ALLOWANCE_PCT', 0.20))
        elif session_name == "PROTECTION":
            max_spread = getattr(config, 'PROTECTION_MAX_SPREAD_FOREX', 20)
        elif session_name == "ASIAN":
            max_spread *= 1.3
        spread_check = spread_points > max_spread

    if spread_check:
        return None, ind, None, f"Spread Alto ({spread_points} > {max_spread})"

    # Ajuste de Volume
    vol_ratio = ind.get("volume_ratio", 0)
    min_vol_ratio = getattr(config, 'MIN_VOLUME_COEFFICIENT', 0.4)
    if session_name == "GOLDEN":
        min_vol_ratio *= (1 - getattr(config, 'GOLDEN_VOLUME_REDUCTION_PCT', 0.30))
    elif session_name == "ASIAN":
        min_vol_ratio = 0.5 # Fixo em 0.5x da média
    
    vol_ok = vol_ratio >= min_vol_ratio
    # Não bloqueamos imediatamente para permitir bypass via ML (v6.1)

    # 4. Extrai indicadores para estratégia
    rsi_now = ind.get("rsi", 50)
    adx_now = ind.get("adx", 0)
    close_price = ind.get("close", 0)
    bb_upper = ind.get("bb_upper")
    bb_lower = ind.get("bb_lower")
    bb_width = ind.get("bb_width", 0)
    ema_trend = ind.get("ema_trend")

    signal = None
    strategy = None
    reason = "Monitorando - Sem sinal claro"

    # 5. Bollinger Squeeze Check (Marcar mas não vetar se ML for usado)
    has_squeeze = bb_width < bb_squeeze_threshold

    # 6. Estratégia baseada no ADX
    if adx_now >= adx_threshold:
        if ema_trend in ("UP", "NEUTRAL"):
            signal = "BUY"
            strategy = "TREND"
            rsi_ok = True
        elif ema_trend == "DOWN":
            signal = "SELL"
            strategy = "TREND"
            rsi_ok = True
        else:
            reason = "Trend Indefinida (EMA Flat)"
            rsi_ok = True # Ignora se não há tendência
    
    else:
        adx_min_rev = getattr(config, 'ADX_MIN_FOR_REVERSION', 10)
        if adx_now < adx_min_rev:
            return None, ind, strategy, f"ADX crítico ({adx_now:.1f})"
        if close_price <= bb_lower:
            if rsi_now < 30:
                signal = "BUY"
                strategy = "REVERSION"
            else:
                reason = f"RSI em Neutro ({rsi_now:.1f})" # ✅ Land Trading Standard
        elif close_price >= bb_upper:
            if rsi_now > 70:
                signal = "SELL"
                strategy = "REVERSION"
            else:
                reason = f"RSI em Neutro ({rsi_now:.1f})" # ✅ Land Trading Standard
        else:
            reason = f"ADX baixo ({adx_now:.1f}) e preço dentro das bandas"

    # 8. Confirmação "Sniper" v5.0 (Price Action)
    if signal:
        # ===========================
        # ✅ v5.0: EMA 200 GOLDEN RULE
        # ===========================
        # BUY só válido se Preço > EMA 200 (tendência de alta)
        # SELL só válido se Preço < EMA 200 (tendência de baixa)
        if getattr(config, 'ENABLE_EMA_200_FILTER', True) and ind.get("ema_200") is not None:
            ema_200_val = ind.get("ema_200")
            price = ind.get("price") or ind.get("close")
            atr = ind.get("atr")

            # Fallback de segurança
            if price is None or atr is None:
                logger.warning(f"{symbol}: EMA200 ativo mas price/ATR indisponível")
            else:
                distance = abs(price - ema_200_val)
                
                # Hybrid Rule EMA 200 (✅ v5.3)
                is_against = (signal == "BUY" and price < ema_200_val) or (signal == "SELL" and price > ema_200_val)
                
                if is_against:
                    ind["ema_against"] = True
                    dist_atr = distance / atr if atr > 0 else 999.99
                    if dist_atr > 3.0:
                        ind["ema_penalty"] = max(ind.get("ema_penalty", 0), 15)
                    elif dist_atr > 1.5:
                        ind["ema_penalty"] = max(ind.get("ema_penalty", 0), 8)

        # ===========================
        # ✅ v5.3: MULTI-TIMEFRAME SCORE ADJUSTMENT (não mais veto)
        # ===========================
        # Correção 3: Se EMA200 já validou tendência, ignora H4 (evita redundância)
        macro_trend_ok = (
            signal and 
            getattr(config, 'ENABLE_EMA_200_FILTER', True) and 
            ind.get("ema_200") is not None and
            not ind.get("ema_penalty") # Se tem penalidade, não está "ok" o suficiente para ignorar H4
        )
        
        if getattr(config, 'ENABLE_MULTI_TIMEFRAME', True) and not macro_trend_ok:
            mtf_penalty, mtf_reason, mtf_trend = utils.get_multi_timeframe_trend(symbol, signal)
            if mtf_penalty < 0:
                ind["mtf_penalty"] = abs(mtf_penalty)
                logger.info(f"📉 {symbol}: {mtf_reason}")
        elif macro_trend_ok and getattr(config, 'ENABLE_MULTI_TIMEFRAME', True):
            logger.debug(f"🔄 {symbol}: H4 check skipped (EMA200 já validou macro trend)")
        
        close_now = ind.get("close", 0)
        open_now = ind.get("open", 0)
        candle_ok = (signal == "BUY" and close_now > open_now) or (signal == "SELL" and close_now < open_now)
        
        if not candle_ok:
            ind["penalty_candle"] = max(ind.get("penalty_candle", 0), 10)
            reason = f"Candle não confirmado (penalidade)"
        
        # Verificação de Filtros
        if not vol_ok:
            reason = f"Veto Volume: {reason} | Vol {vol_ratio:.2f}"
            signal = "BUY_VETO" if signal == "BUY" else "SELL_VETO"
        else:
            reason = f"Sinal {signal} confirmado por {strategy}"
            if has_squeeze:
                reason += " (Squeeze Detectado)"

    return signal, ind, strategy, reason

    return signal, ind, strategy
# ===========================
# EMA 200 MACRO FILTER (WITH PULLBACK TOLERANCE)
# ===========================

def is_macro_trend_allowed(
    signal: str,
    price: float,
    ema_200: float,
    atr: float
) -> bool:
    """
    Valida tendência macro com tolerância de pullback baseada em ATR.
    """

    # BUY em macro bullish
    if signal == "BUY":
        if price >= ema_200:
            return True

        if EMA_200_ALLOW_PULLBACK:
            distance = abs(price - ema_200)
            if distance <= atr * EMA_200_PULLBACK_ATR_TOLERANCE:
                return True

        return False

    # SELL em macro bearish
    if signal == "SELL":
        if price <= ema_200:
            return True

        if EMA_200_ALLOW_PULLBACK:
            distance = abs(price - ema_200)
            if distance <= atr * EMA_200_PULLBACK_ATR_TOLERANCE:
                return True

        return False

    return False

# Variáveis Globais de Controle de Thread
FAST_LOOP_LOCK = threading.Lock()
FAST_LOOP_ACTIVE = False # ✅ Impede múltiplas instâncias da thread rápida
FAST_LOOP_LAST_BEAT = 0.0
FAST_LOOP_DUP_REJECT_TS = 0.0

# ===========================
# FAST LOOP
# ===========================
def fast_loop():
    """Loop rápido de análise e execução"""
    global FAST_LOOP_ACTIVE, FAST_LOOP_LAST_BEAT, FAST_LOOP_DUP_REJECT_TS
    
    # 🛡️ SINGLE INSTANCE CHECK
    FAST_LOOP_LAST_BEAT = time.time()

    # try:  <-- REMOVIDO para evitar erro de indentação massiva
    safe_log(logging.INFO, "🚀 Fast Loop iniciado (Single Instance)")

    last_cache_update = 0
    iteration_count = 0
    rejection_stats = {"ML": 0, "TECHNICAL": 0, "NEWS": 0, "RISK": 0}
    last_status_report = {} # {symbol: timestamp} para throttling de negative edge logs
    last_heartbeat_update = 0 
    active_tickets_map = {} # ✅ NOVO: Rastreio de tickets para Cooldown {ticket: symbol}
    last_session_update = 0
    current_session = {"name": "NORMAL", "display": "NORMAL", "emoji": "⚖️", "color": "blue"}
    
    # Inicia cache de log de ML (anti-spam)
    ml_log_cache = {}
    analysis_log_state = {}
    last_weekend_action_ts = 0.0

    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            FAST_LOOP_LAST_BEAT = current_time
            watchdog.heartbeat("FastLoop")
            _apply_pause_requests()
            if current_time - last_heartbeat_update > 30:  # Atualiza a cada 30s
                update_heartbeat()
                last_heartbeat_update = current_time

            # ✅ SESSION DETECTION (Brasil Time)
            if current_time - last_session_update > 60: # Atualiza a cada minuto
                current_session = utils.get_current_trading_session()
                last_session_update = current_time
                safe_log(logging.INFO, f"🌐 SESSÃO ATUAL: {current_session['display']} {current_session['emoji']}")

            ok, info = perform_system_checks(current_session, iteration_count)
            if not ok:
                if iteration_count % 60 == 0:
                    logger.warning(info.get("reason", ""))
                time.sleep(int(info.get("sleep", 1)))
                continue

            # --- PRIORALIZAÇÃO DE PARES (SESSÃO ASIÁTICA) ---
            symbols_to_analyze = _compute_symbols_to_analyze(current_session)

            iteration_count += 1
            watchdog.heartbeat("FastLoop")
            current_time = time.time()
            logger.info(f"🔎 FastLoop ativo | it={iteration_count} | símbolos={len(symbols_to_analyze)}")

            # ========================================
            # ✅ ATUALIZA RASTREIO DE POSIÇÕES (COOLDOWN)
            # ========================================
            active_tickets_map = manage_open_positions(active_tickets_map)


            if iteration_count % 60 == 0:
                logger.info(f"🔄 Fast Loop - Iteração #{iteration_count}")
                logger.info(
                    "📊 Rejeições (últimas iterações) | ML=%d | Técnico=%d | News=%d | Risco=%d",
                    rejection_stats["ML"],
                    rejection_stats["TECHNICAL"],
                    rejection_stats["NEWS"],
                    rejection_stats["RISK"],
                )

            # ========================================
            # ATUALIZA CACHE DE INDICADORES E TOP PAIRS
            # ========================================
            # ✅ CORREÇÃO: Usa config.SLOW_LOOP_INTERVAL para cache update
            if current_time - last_cache_update > getattr(config, 'SLOW_LOOP_INTERVAL', 300):
                last_cache_update = current_time
                panel_cache = {}
                try:
                    if profit_optimizer:
                        profit_optimizer.scan_and_optimize()
                except Exception:
                    pass

                # ✅ v5.0.6: Usa a lista global filtrada
                symbols_to_process = GLOBAL_ACTIVE_SYMBOLS[:getattr(config, 'MAX_SYMBOLS_CACHE', 15)]

                for symbol in symbols_to_process:
                    try:
                        # ✅ DASHBOARD FEED: Garante que o painel mostre "Analisando" imediatamente
                        GLOBAL_MONITOR_CACHE[symbol] = {
                            "status": "🔍 ANALISANDO",
                            "reason": "Buscando oportunidades...",
                            #"ml_score": 50.0,
                            "is_baseline": True,
                            "timestamp": datetime.now()
                        }

                        # Remove da blacklist se já passou o tempo
                        if symbol in GLOBAL_SESSION_BLACKLIST:
                            # Se o símbolo está na blacklist da sessão, verifica se já passou o tempo de "castigo"
                            # Por exemplo, 1 hora. Se sim, remove da blacklist.
                            # Isso é um placeholder, a lógica real de remoção pode ser mais complexa.
                            # Por enquanto, apenas ignora se estiver na blacklist.
                            continue

                        # ✅ REQUISITO: Normalização de Símbolo Sugerida
                        real_symbol = utils.normalize_symbol(symbol)
                        if real_symbol != symbol:
                            logger.info(f"🔄 Símbolo {symbol} normalizado para {real_symbol}")
                        
                        # ✅ CORREÇÃO: Passa parâmetros otimizados para get_indicators_forex
                        params = ELITE_CONFIG.get(symbol, {})
                        ind = utils.get_indicators_forex(
                            symbol,
                            ema_short=params.get("ema_short"),
                            ema_long=params.get("ema_long"),
                            rsi_period=params.get("rsi_period"),
                            rsi_low=params.get("rsi_low"),
                            rsi_high=params.get("rsi_high"),
                            adx_period=params.get("adx_period"),
                            bb_period=params.get("bb_period"),
                            bb_dev=params.get("bb_dev")
                        )
                        
                        # ✅ REQUISITO: Verificação robusta para evitar morte de thread
                        if ind is None or (isinstance(ind, dict) and ind.get("error")):
                            continue

                        # Limpa o DF para economizar memória no cache
                        if 'df' in ind:
                            del ind['df']

                        panel_cache[symbol] = ind
                    except Exception as e:
                        logger.error(f"❌ Erro ao atualizar cache de indicadores para {symbol}: {e}", exc_info=True)
                        continue

                if panel_cache:
                    # ✅ CORREÇÃO: calculate_signal_score precisa de todos os parâmetros otimizados
                    # Para simplificar, vamos usar uma versão básica do score aqui para o top_pairs
                    # Uma implementação mais robusta passaria os params otimizados para calculate_signal_score
                    top_pairs = sorted(
                        panel_cache.keys(),
                        key=lambda s: utils.calculate_signal_score(panel_cache[s])[0], # Score básico
                        reverse=True
                    )
                    bot_state.update_indicators(panel_cache, top_pairs)

            # ========================================
            # VERIFICAÇÕES DE ESTADO DO BOT E MERCADO
            # ========================================
            if not utils.is_market_open():
                if iteration_count % 60 == 0:
                    market = utils.get_market_status()
                    logger.warning(f"💤 Mercado fechado: {market['message']}")
                time.sleep(60)
                continue

            ks_path = Path(getattr(config, 'KILLSWITCH_FILE', 'killswitch.txt'))
            if ks_path.exists():
                try:
                    ttl_seconds = int(getattr(config, 'KILLSWITCH_TTL_SECONDS', 900))
                    mtime = datetime.fromtimestamp(ks_path.stat().st_mtime)
                    age = (datetime.now() - mtime).total_seconds()
                    if age <= ttl_seconds:
                        logger.critical("🚨 KILL SWITCH ATIVADO! Fechando todas as posições e encerrando.")
                        utils.close_all_positions()
                        try:
                            ks_path.unlink()
                        except Exception:
                            pass
                        shutdown_event.set()
                        break
                    else:
                        try:
                            ks_path.unlink()
                        except Exception:
                            pass
                        logger.warning("⚠️ Ignorando KILL SWITCH antigo. Arquivo removido.")
                except Exception as e:
                    logger.error(f"❌ Erro ao processar KILL SWITCH: {e}")

            is_paused, pause_reason = bot_state.is_paused()
            if is_paused:
                if iteration_count % 60 == 0:
                    logger.warning(f"⏸️ Bot pausado: {pause_reason}")
                time.sleep(10)
                continue

            # ✅ NOVO: Reset diário de limites
            bot_state.check_and_reset_daily_limits()

            # ========================================
            # ANÁLISE DE SÍMBOLOS E EXECUÇÃO DE ORDENS
            # ========================================
            # ✅ v5.0.6: Usa a lista global filtrada para análise
            symbols_to_analyze = GLOBAL_ACTIVE_SYMBOLS

            weekend_state = utils.get_weekend_protection_state() if hasattr(utils, "get_weekend_protection_state") else {"block_entries": False, "force_close": False, "reason": ""}
            allow_entries = not weekend_state.get("block_entries", False)
            if weekend_state.get("force_close", False):
                now_ts = time.time()
                if (now_ts - last_weekend_action_ts) > 60:
                    last_weekend_action_ts = now_ts
                    utils.send_telegram_alert(weekend_state.get("reason", "Fim de pregão"), "WARNING")
                    utils.close_all_positions()
            
            # Conjunto de símbolos para excluir se não existirem no MT5
            static_blacklist = getattr(config, 'BLACKLISTED_SYMBOLS', set())
            
            # ✅ REQUISITO: Lista Negra Dinâmica de Sessão
            # GLOBAL_SESSION_BLACKLIST já inicializado globalmente

            # GLOBAL_SESSION_BLACKLIST já inicializado globalmente

            for symbol in symbols_to_analyze:
                try:
                    watchdog.heartbeat("FastLoop")
                    now_ts_local = time.time()
                    if (now_ts_local - last_heartbeat_update) > 20:
                        update_heartbeat()
                        last_heartbeat_update = now_ts_local
                except Exception:
                    pass
                symbol_start_ts = time.time()
                if symbol in static_blacklist or symbol in GLOBAL_SESSION_BLACKLIST:
                    continue

                if not allow_entries:
                    reason_weekend = weekend_state.get("reason", "Entradas bloqueadas")
                    bot_state.update_monitoring(symbol, "🏁 FIM DE SEMANA", reason_weekend, 0.0)
                    now_ts = time.time()
                    throttle_s = getattr(config, 'ANALYSIS_LOG_THROTTLE_SECONDS', 120)
                    prev = analysis_log_state.get(symbol)
                    prev_reason = prev.get("reason") if prev else None
                    prev_ts = prev.get("ts") if prev else 0
                    if (prev_reason != reason_weekend) or ((now_ts - prev_ts) > throttle_s):
                        analysis_log_state[symbol] = {"reason": reason_weekend, "ts": now_ts}
                        log_signal_analysis(symbol, "NONE", "WEEKEND", 0, True, reason_weekend, {})
                    continue
                
                # ✅ v6.1: Proteção contra ordens duplicadas (Cache Recente)
                if symbol in RECENT_ORDERS_CACHE:
                    last_order_time = RECENT_ORDERS_CACHE[symbol]
                    if (datetime.now() - last_order_time).total_seconds() < 15: # 15s de segurança
                        if iteration_count % 5 == 0:
                            logger.debug(f"🛡️ {symbol}: Ignorando análise (Ordem enviada há <15s)")
                        bot_state.update_monitoring(symbol, "⏳ WAITING", "Post-Order Safety", 0.0)
                        continue
                    else:
                        del RECENT_ORDERS_CACHE[symbol] # Limpa cache antigo

                # ===========================
                # ✅ v5.2: CHECK PAUSED SYMBOLS (Kill Switch)
                # ===========================
                if symbol in PAUSED_SYMBOLS:
                    pause_data = PAUSED_SYMBOLS[symbol]
                    if datetime.now() < pause_data["until"]:
                        # Ainda está pausado
                        if iteration_count % 60 == 0:
                            logger.info(f"⏸️ {symbol} pausado até {pause_data['until'].strftime('%H:%M')} ({pause_data['reason']})")
                        bot_state.update_monitoring(symbol, "⏸️ PAUSADO", pause_data['reason'], 0.0)
                        continue
                    else:
                        # Tempo expirou, remove da pausa
                        del PAUSED_SYMBOLS[symbol]
                        logger.info(f"▶️ {symbol}: Pausa expirada. Retomando operações.")
                
                if block_manager:
                    is_blocked, reason = block_manager.is_blocked(symbol)
                    if is_blocked:
                        bot_state.update_monitoring(symbol, "⛔ BLOQUEIO", reason, 0.0)
                        now_ts = time.time()
                        throttle_s = getattr(config, 'ANALYSIS_LOG_THROTTLE_SECONDS', 120)
                        prev = analysis_log_state.get(symbol)
                        prev_reason = prev.get("reason") if prev else None
                        prev_ts = prev.get("ts") if prev else 0
                        if (prev_reason != reason) or ((now_ts - prev_ts) > throttle_s):
                            analysis_log_state[symbol] = {"reason": reason, "ts": now_ts}
                            log_signal_analysis(symbol, "NONE", "RISK_BLOCK", 0, True, reason, {})
                        continue

                try:
                    # ✅ REQUISITO: Validação Instantânea de Existência
                    if not mt5_exec(mt5.symbol_select, symbol, True):
                            logger.error(f"❌ Símbolo {symbol} removido por inexistência no MT5 nesta sessão.")
                            GLOBAL_SESSION_BLACKLIST.add(symbol)
                            continue
                    
                    # ✅ FIX: Inicializa params no início do loop para evitar UnboundLocalError
                    params = ELITE_CONFIG.get(symbol, {})

                    # =====================================================
                    # ✅ UNIFIED SAFETY LOCKS (CRITICAL FIX)
                    # =====================================================
                    # 1. Verifica limite de ordens por símbolo (PRIMEIRA CHECAGEM)
                    # Garante que nada aconteça se já houver posição aberta
                    existing_positions = mt5_exec(mt5.positions_get, symbol=symbol)
                    max_orders = getattr(config, 'MAX_ORDERS_PER_SYMBOL', 1)
                    
                    if existing_positions and len(existing_positions) >= max_orders:
                        # ✅ SNIPER MODE: Log Limpo
                        if iteration_count % 60 == 0: 
                            logger.info(f"🔭 {symbol}: Sniper Mode - Monitorando posição existente (Ticket {existing_positions[0].ticket}).")
                        bot_state.update_monitoring(symbol, "🔴 OPERANDO", f"Gestão Ativa ({len(existing_positions)}/{max_orders})", 50.0) # Score neutro
                        continue # ⛔ SAI IMEDIATAMENTE

                    # 2. Verifica cooldown após fechamento de ordem
                    cooldown_seconds = getattr(config, 'ORDER_COOLDOWN_SECONDS', 300)
                    last_close_time = ORDER_COOLDOWN_TRACKER.get(symbol, 0)
                    time_since_close = time.time() - last_close_time
                    if last_close_time > 0 and time_since_close < cooldown_seconds:
                        remaining = int(cooldown_seconds - time_since_close)
                        if iteration_count % 30 == 0: # Throttling log
                           logger.info(f"⏳ {symbol}: Cooldown ativo ({remaining}s restantes)")
                        bot_state.update_monitoring(symbol, "⏳ COOLDOWN", f"Aguardando {remaining}s", 0.0)
                        continue # ⛔ SAI IMEDIATAMENTE

                     # 3. Verificação de Spread em Tempo Real (Re-fetch)
                    current_spread = 0 # ✅ INIT Seguro
                    symbol_info_realtime = mt5_exec(mt5.symbol_info, symbol)
                    if symbol_info_realtime:
                        current_spread = symbol_info_realtime.spread
                        symbol_upper = str(symbol).upper().strip()

                        crypto_list = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]
                        indices_list = ["US30", "NAS100", "USTEC", "DE40", "GER40", "GER30", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"]
                        metals_list = ["XAU", "XAG", "GOLD", "SILVER"]

                        is_crypto = any(c in symbol_upper for c in crypto_list)
                        is_indices = any(i in symbol_upper for i in indices_list)
                        is_metals = any(m in symbol_upper for m in metals_list)

                        if is_crypto:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
                        elif is_indices:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_INDICES', 800)
                        elif is_metals:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_METALS', 80)
                        else:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_FOREX', 25)
                        # Exóticos (TRY/ZAR/MXN etc.)
                        exotics_list = getattr(config, 'TOKENS_EXOTICS', ["TRY","ZAR","MXN","RUB","CNH","PLN","HUF","CZK","DKK","NOK","SEK"])
                        is_exotics = any(x in symbol_upper for x in exotics_list)
                        if is_exotics:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_EXOTICS', 10000)

                        sess_name = (current_session or {}).get("name")
                        if not (is_crypto or is_indices or is_metals):
                            if sess_name == "ASIAN":
                                max_spread_limit *= 1.3
                            elif sess_name == "GOLDEN":
                                max_spread_limit *= 1.2

                        if current_spread > max_spread_limit:
                            if iteration_count % 30 == 0:
                                logger.warning(f"🚫 {symbol}: Spread Alto ({current_spread} > {max_spread_limit}).")
                            bot_state.update_monitoring(symbol, "⚠️ BLOQUEADO", f"Spread Alto ({current_spread} > {max_spread_limit})", 0.0)
                            continue # ⛔ SAI IMEDIATAMENTE
                    # =====================================================

                    # =====================================================
                    # ✅ v5.3: INSTITUTIONAL RISK - KILL SWITCH & DRAWDOWN
                    # =====================================================
                    risk_ok, risk_msg = utils.check_institutional_risk()
                    if not risk_ok:
                        if iteration_count % 60 == 0:
                            logger.critical(f"🛑 BLOQUEIO INSTITUCIONAL: {risk_msg}")
                        bot_state.pause_trading(risk_msg)
                        time.sleep(300) # Pausa longa para preservação de capital
                        continue
                    # =====================================================

                    # =====================================================
                    # 4. Attempt Cooldown (Um Sinal por Candle/Minuto)
                    # =====================================================
                    last_attempt = ATTEMPT_COOLDOWN_TRACKER.get(symbol, 0)
                    if time.time() - last_attempt < 60:
                        # Silenciosamente pula para não spammar log
                        remaining = int(60 - (time.time() - last_attempt))
                        bot_state.update_monitoring(symbol, "⏳ COOLDOWN", f"Veto Cooldown ({remaining}s)", 0.0)
                        continue
                    # =====================================================

                    # =====================================================
                    # ✅ v5.1: CORRELATION FILTER (Land Trading)
                    # =====================================================
                    is_corr_blocked, corr_reason, corr_symbol = utils.check_correlation(symbol)
                    if is_corr_blocked:
                        if iteration_count % 30 == 0:
                            logger.info(f"🔗 {symbol}: {corr_reason}")
                            log_signal_analysis(symbol, "NONE", "CORRELATION", 0, True, corr_reason, {}, session_name=current_session.get("name"))
                        bot_state.update_monitoring(symbol, "🔗 CORRELAÇÃO", corr_reason, 0.0)
                        continue

                    # =====================================================
                    # ✅ v5.1: VOLATILITY FILTER (Land Trading)
                    # =====================================================
                    vol_ok, vol_reason, atr_val = utils.is_volatility_ok(symbol)
                    if not vol_ok:
                        if iteration_count % 60 == 0:
                            logger.info(f"📉 {symbol}: {vol_reason}")
                            log_signal_analysis(symbol, "NONE", "VOLATILITY", 0, True, vol_reason, {}, session_name=current_session.get("name"))
                        bot_state.update_monitoring(symbol, "📉 BAIXA VOL", vol_reason, 0.0)
                        continue

                    # ✅ v5.3: Coleta de Motivos de Rejeição
                    reject_reasons = []
                    
                    # 1. Busca sinal e motivo detalhado
                    signal, ind, strategy, reason = check_for_signals(symbol, current_session)
                    try:
                        watchdog.heartbeat("FastLoop")
                    except Exception:
                        pass
                    
                    # ✅ CORREÇÃO: Sempre registrar no analysis log mesmo sem sinal técnico
                    if not signal or signal == "NONE":
                        display_reason = reason if reason else "Aguardando setup..."
                        bot_state.update_monitoring(symbol, "🔵 MONITORANDO", display_reason, 0.0)
                        log_signal_analysis(symbol, "NONE", strategy or "N/A", 0, True, display_reason, ind or {}, session_name=current_session.get("name"))
                        continue

                    # ✅ MODO PROTEÇÃO: Bloqueia novas entradas
                    if current_session.get("name") == "PROTECTION":
                        reject_reasons.append("Modo Proteção Ativo")
                        if iteration_count % 60 == 0:
                            logger.info(f"🛡️ {symbol}: Modo Proteção Ativo. Novas entradas bloqueadas.")
                        bot_state.update_monitoring(symbol, "🛡️ PROTEÇÃO", "Apenas Gerenciamento", 0.0)
                        log_signal_analysis(symbol, signal, strategy, 0, True, "Modo Proteção Ativo", ind, session_name=current_session.get("name"))
                        continue
                    
                    # ✅ REQUISITO: Verificação de indicadores nulos para pular ML
                    if ind is None:
                        reject_reasons.append("RISK: Indicadores Nulos")
                        logger.warning(f"⚠️ Pulando análise de ML para {symbol} devido a indicadores ausentes.")
                        bot_state.update_monitoring(symbol, "⚠️ IND. ERROR", "Indicadores Nulos", 0.0)
                        continue

                    # Regime Filters
                    try:
                        adx_req = float(getattr(config, "ADX_MIN_STRENGTH", 28))
                        adx_now = float(ind.get("adx", 0))
                        if adx_now < adx_req:
                            reason_reg = f"Regime: ADX baixo ({adx_now:.1f} < {adx_req:.1f})"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        close_price = ind.get("close", ind.get("current_price"))
                        bb_upper = ind.get("bb_upper")
                        bb_lower = ind.get("bb_lower")
                        if signal == "BUY" and (bb_upper is None or close_price is None or close_price < bb_upper):
                            reason_reg = "Regime: BB sem breakout (BUY)"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        if signal == "SELL" and (bb_lower is None or close_price is None or close_price > bb_lower):
                            reason_reg = "Regime: BB sem breakout (SELL)"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        atr_now = float(ind.get("atr", 0) or 0)
                        atr20 = float(ind.get("atr20", 0) or 0)
                        if atr20 > 0 and atr_now < (1.3 * atr20):
                            reason_reg = f"Regime: ATR baixo ({atr_now:.5f} < 1.3×ATR20)"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        ema200_h4 = ind.get("ema200_h4")
                        if ema200_h4 is not None and close_price is not None:
                            if signal == "BUY" and not (close_price > ema200_h4):
                                reason_reg = "Regime: H4 EMA200 contra BUY"
                                bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                                log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                                continue
                            if signal == "SELL" and not (close_price < ema200_h4):
                                reason_reg = "Regime: H4 EMA200 contra SELL"
                                bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                                log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                                continue
                    except Exception:
                        pass

                    # =====================================================
                    # ✅ SOVEREIGN VETO (FIM DO ML PRIORITY)
                    # =====================================================
                    veto_signal = signal in ["BUY_VETO", "SELL_VETO"]
                    if veto_signal:
                        reject_reasons.append(f"TECHNICAL: Veto Técnico: {reason}")
                        ATTEMPT_COOLDOWN_TRACKER[symbol] = time.time() 
                        
                        reason_veto = f"❌ VETO {reason}"
                        if iteration_count % 10 == 0:
                            logger.info(f"🛑 {symbol}: {reason_veto}")
                        
                        bot_state.update_monitoring(symbol, "🔵 MONITORANDO", reason_veto, 0.0)
                        now_ts = time.time()
                        throttle_s = getattr(config, 'ANALYSIS_LOG_THROTTLE_SECONDS', 120)
                        prev = analysis_log_state.get(symbol)
                        prev_reason = prev.get("reason") if prev else None
                        prev_ts = prev.get("ts") if prev else 0
                        if (prev_reason != reason_veto) or ((now_ts - prev_ts) > throttle_s):
                            analysis_log_state[symbol] = {"reason": reason_veto, "ts": now_ts}
                            log_signal_analysis(symbol, signal, strategy or "N/A", 0, True, reason_veto, ind or {}, session_name=current_session.get("name"))
                        continue

                    ml_confidence_real = ind.get('ml_confidence')
                    if (time.time() - symbol_start_ts) > float(getattr(config, 'MAX_SYMBOL_PROCESS_SECONDS', 3.0)):
                        logger.warning(f"⏳ {symbol}: Tempo de análise excedido ({time.time() - symbol_start_ts:.1f}s). Pulando para próximo símbolo.")
                        bot_state.update_monitoring(symbol, "⏳ TIMEOUT", "Análise excedeu tempo máximo", 0.0)
                        try:
                            log_signal_analysis(symbol, "NONE", strategy or "N/A", 0, True, "Timeout de análise", ind or {}, session_name=current_session.get("name"))
                        except Exception:
                            pass
                        continue
                    ml_is_baseline = False

                    if hasattr(utils, "ml_optimizer_instance") and utils.ml_optimizer_instance:
                        try:
                            df_ml = ind.get("df")
                            ml_score_raw, ml_is_baseline = utils.ml_optimizer_instance.get_prediction_score(
                                symbol, ind, df_ml, signal=signal
                            )
                            ml_confidence_real = ml_score_raw / 100.0
                        except Exception as e_ml:
                            logger.error(f"Erro na predição ML em fast_loop para {symbol}: {e_ml}")
                            if ml_confidence_real is None:
                                ml_confidence_real = 0.5
                            ml_is_baseline = True
                    else:
                        if ml_confidence_real is None:
                            ml_confidence_real = 0.5

                    if isinstance(ind, dict) and "df" in ind:
                        del ind["df"]

                    ml_score = ml_confidence_real * 100
                    if isinstance(ind, dict) and ind.get("penalty_candle"):
                        ml_score = max(0.0, ml_score - float(ind.get("penalty_candle", 0)))

                    confidence_threshold = params.get("ml_threshold", getattr(config, 'ML_CONFIDENCE_THRESHOLD', 0.60))
                    min_ml_score = confidence_threshold * 100
                    try:
                        sess_name = (current_session or {}).get("name")
                        # ✅ Verificação de spread por sessão (Forex)
                        try:
                            sp_points = float(ind.get("spread_points", 0.0) or 0.0)
                            base_ok = bool(ind.get("spread_ok", True))
                            effective_spread_ok = base_ok
                            cat = "FOREX"
                            su = symbol.upper()
                            if any(idx in su for idx in ["US30", "NAS100", "USTEC", "DE40", "GER40", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"]):
                                cat = "INDICES"
                            elif any(met in su for met in ["XAU", "XAG", "GOLD", "SILVER"]):
                                cat = "METALS"
                            elif any(c in su for c in ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]):
                                cat = "CRYPTO"
                            sess_limits = getattr(config, "SESSION_SPREAD_LIMITS", {})
                            lim = sess_limits.get(sess_name or "", {}).get(cat, {})
                            if sess_name == "GOLDEN":
                                allowance = float(lim.get("allowance_pct", getattr(config, "GOLDEN_SPREAD_ALLOWANCE_PCT", 0.0)))
                                if not base_ok and allowance > 0:
                                    effective_spread_ok = True
                            elif sess_name == "ASIAN":
                                max_pts = float(lim.get("max_points", getattr(config, "ASIAN_MAX_SPREAD_POINTS", 50)))
                                effective_spread_ok = sp_points <= max_pts
                            elif sess_name == "PROTECTION":
                                max_pips = float(lim.get("max_pips", getattr(config, "PROTECTION_MAX_SPREAD_FOREX", 20)))
                                pip_size = utils.get_pip_size(symbol)
                                point = getattr(utils.get_symbol_info(symbol), "point", 0.0) if utils.get_symbol_info(symbol) else 0.0
                                sp_pips = (sp_points * point) / pip_size if pip_size > 0 else 999.0
                                effective_spread_ok = sp_pips <= max_pips
                            if not effective_spread_ok:
                                reason_spread = f"Spread alto ({sp_points:.0f} pts) sessão {sess_name or 'N/A'}"
                                reject_reasons.append(reason_spread)
                                bot_state.update_monitoring(symbol, "🟡 SPREAD", reason_spread, ml_score)
                                log_signal_analysis(symbol, signal, strategy, ml_score, True, reason_spread, ind, ml_score=ml_score)
                                continue
                        except Exception:
                            pass
                    except Exception:
                        pass
                    if isinstance(ind, dict) and ind.get("ema_against"):
                        min_ml_score = max(min_ml_score, 50.0)
                    try:
                        eff_rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
                        eff_rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
                        if params:
                            allowed = {"ema_short", "ema_long", "adx_threshold", "bb_squeeze_threshold"}
                            safe_params = {k: v for k, v in params.items() if k in allowed}
                            ts_bypass, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high, **safe_params)
                        else:
                            ts_bypass, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high)
                        if ts_bypass >= 65:
                            ml_score = 70.0 if ml_score < 70.0 else ml_score
                    except Exception:
                        pass
                    try:
                        wr, total, _wins = utils.calculate_rolling_win_rate(symbol)
                        if total >= 20 and wr > 0.65:
                            min_ml_score = float(min_ml_score) * 0.95
                    except Exception:
                        pass

                    cache_entry = GLOBAL_MONITOR_CACHE.get(symbol, {})
                    GLOBAL_MONITOR_CACHE[symbol] = {
                        "status": cache_entry.get("status", "🔍 ANALISANDO"),
                        "reason": cache_entry.get("reason", "ML Avaliação"),
                        "ml_score": ml_score,
                        "is_baseline": ml_is_baseline,
                        "timestamp": datetime.now(),
                    }

                    # Log detalhado para debug
                    logger.info(f"[{symbol}] ML confidence real: {ml_confidence_real:.3f} → Score: {ml_score:.1f}/100 | Threshold: {min_ml_score:.0f}")

                    ml_override_used = False
                    ml_override_risk_mult = 1.0
                    if ml_score < min_ml_score:
                        allow_override = getattr(config, "ENABLE_ML_VETO_OVERRIDE", True)
                        allowed_sessions = getattr(config, "ML_VETO_OVERRIDE_ALLOWED_SESSIONS", ["GOLDEN", "NORMAL"])
                        try:
                            sess_name = (current_session or {}).get("name")
                            if isinstance(allowed_sessions, (list, tuple, set)) and sess_name and sess_name not in allowed_sessions:
                                allow_override = False
                        except Exception:
                            pass
                        try:
                            min_conf = float(getattr(config, "ML_OVERRIDE_MIN_CONFIDENCE", 0.90))
                            if ml_confidence_real < min_conf:
                                allow_override = False
                        except Exception:
                            allow_override = False
                        override_min_score = getattr(config, "ML_VETO_OVERRIDE_MIN_TECH_SCORE", 80)
                        override_max_spread = getattr(config, "ML_VETO_OVERRIDE_MAX_SPREAD_PIPS", 2.0)
                        spread_pips = float(ind.get("spread_pips", 999.0) or 999.0)
                        try:
                            if ind.get("spread_ok") is False:
                                spread_pips = 999.0
                        except Exception:
                            pass
                        tech_score_for_override = 0.0
                        if allow_override:
                            try:
                                eff_rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
                                eff_rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
                                if params:
                                    allowed = {"ema_short", "ema_long", "adx_threshold", "bb_squeeze_threshold"}
                                    safe_params = {k: v for k, v in params.items() if k in allowed}
                                    tech_score_for_override, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high, **safe_params)
                                else:
                                    tech_score_for_override, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high)
                            except Exception:
                                tech_score_for_override = 0.0

                        if allow_override and tech_score_for_override >= override_min_score and spread_pips <= override_max_spread:
                            ml_override_used = True
                            ml_override_risk_mult = float(getattr(config, "ML_VETO_OVERRIDE_RISK_MULTIPLIER", 0.5))
                            reason_override = f"ML: Override ({ml_score:.0f} < {min_ml_score:.0f}) | Tech:{tech_score_for_override:.1f} | Spread:{spread_pips:.2f}p"
                            bot_state.update_monitoring(symbol, "🟣 OVERRIDE", reason_override, ml_score)
                        else:
                            reason_ml = f"ML: Veto ML ({ml_score:.0f} < {min_ml_score:.0f})"
                            reject_reasons.append(reason_ml)
                            bot_state.update_monitoring(symbol, "🔵 MONITORANDO", reason_ml, ml_score)
                            log_signal_analysis(symbol, signal, strategy, ml_score, True, reason_ml, ind, ml_score=ml_score)
                            continue
                    else:
                        logger.info(f"[{symbol}] ML aprovado: {ml_score:.1f} >= {min_ml_score:.0f}")

                    # ✅ Gate de força de tendência via ADX mínimo
                    try:
                        adx_min = float(getattr(config, "ADX_MIN_STRENGTH", 25))
                        sym_up = symbol.upper()
                        adx_map = getattr(config, "ADX_MIN_STRENGTH_BY_SYMBOL", {})
                        if isinstance(adx_map, dict) and sym_up in adx_map:
                            adx_min = float(adx_map[sym_up])
                        current_adx = float(ind.get("adx", 0.0) or 0.0)
                        if current_adx < adx_min:
                            reason_adx = f"ADX baixo ({current_adx:.1f} < {adx_min:.0f})"
                            reject_reasons.append(reason_adx)
                            bot_state.update_monitoring(symbol, "🔹 ADX", reason_adx, ml_score)
                            log_signal_analysis(symbol, signal, strategy, ml_score, True, reason_adx, ind, ml_score=ml_score)
                            continue
                    except Exception:
                        pass
                    # Verifica News Filter
                    is_blackout, news_reason = news_filter.is_news_blackout(symbol)
                    if is_blackout:
                        reject_reasons.append(f"NEWS: {news_reason}")
                        bot_state.update_monitoring(symbol, "⚠️ BLOQUEADO", news_reason, ml_score)
                        log_signal_analysis(symbol, signal, strategy, ml_score, True, f"Notícia: {news_reason}", ind, ml_score=ml_score)
                        continue

                    # Cálculo de Score Final com prioridade ao ML
                    eff_rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
                    eff_rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
                    if params:
                        allowed = {"ema_short", "ema_long", "adx_threshold", "bb_squeeze_threshold"}
                        safe_params = {k: v for k, v in params.items() if k in allowed}
                        tech_score, details = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high, **safe_params)
                    else:
                        tech_score, details = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high)
                    logger.info(f"[{symbol}] Score Técnico (referência): {tech_score:.1f} | Breakdown: {details}")
                    penalties = 0.0
                    if ind.get("ema_against"):
                        penalties += 10.0
                    if ind.get("penalty_candle"):
                        penalties += float(ind.get("penalty_candle", 10.0))
                    if ind.get("adx_low"):
                        penalties += 5.0
                    final_score = max(0.0, ml_score - penalties)
                    boost_min_ml = float(getattr(config, 'ML_BOOST_MIN_ML_SCORE', 60))
                    boost_min_tech = float(getattr(config, 'ML_BOOST_MIN_TECH_SCORE', 28))
                    boost_cap = float(getattr(config, 'ML_BOOST_MAX_POINTS', 25))
                    if ml_score >= boost_min_ml and tech_score >= boost_min_tech:
                        final_score += min(boost_cap, tech_score * 0.1)

                    current_threshold = float(getattr(config, 'ML_MIN_SCORE', 70.0))
                    logger.info(f"[{symbol}] Threshold ML: {current_threshold:.1f} | Score Final: {final_score:.1f}")
                    
                    try:
                        regime = utils.get_volatility_regime(symbol, ind.get("df") if isinstance(ind, dict) else None)
                        if regime == "HIGH":
                            current_threshold -= 10
                        elif regime == "LOW":
                            current_threshold += 5
                    except Exception:
                        pass
                    min_threshold = float(getattr(config, 'ML_MIN_SCORE', 70.0))
                    if current_threshold < min_threshold:
                        current_threshold = min_threshold

                    if final_score < current_threshold:
                        reject_reasons.append(f"ML: Score Insuficiente ({final_score:.1f} < {current_threshold:.1f})")
                        logger.info(f"[{symbol}] Score insuficiente ({final_score:.1f} < {current_threshold:.1f})")
                        log_signal_analysis(symbol, signal, strategy, final_score, True, f"Score Baixo: {final_score:.1f}", ind, ml_score=ml_score)
                        continue

                    # Candle não confirmado já tratado como penalidade; não vete aqui

                    ok_port, msg_port = utils.check_portfolio_exposure(
                        pending_symbol=symbol,
                        pending_volume=0.0,  # volume ainda não definido; validaremos no attempt_entry
                        pending_side=signal
                    )
                    if not ok_port and iteration_count % 30 == 0:
                        logger.warning(f"🟠 {symbol}: {msg_port}")
                        bot_state.update_monitoring(symbol, "🟠 PORTFÓLIO", msg_port, ml_score)
                        continue
                    executed = attempt_entry(symbol, signal, strategy, ind, params, current_session, iteration_count, final_score, ml_score, ml_override_risk_mult, ml_override_used, rejection_stats)
                    if not executed:
                        continue

                except Exception as e:
                    logger.error(f"❌ Erro crítico ao analisar/executar para {symbol}: {e}", exc_info=True)
                    bot_state.update_monitoring(symbol, "⚠️ ERRO", "Falha na Análise", 0.0)
                    continue

            # ========================================
            # GESTÃO DE POSIÇÕES (Breakeven, Trailing Stop)
            # ========================================
            positions = mt5_exec(mt5.positions_get)

            if positions:
                for pos in positions:
                    if pos.magic != getattr(config, 'MAGIC_NUMBER', 123456): # ✅ CORREÇÃO: getattr para MAGIC_NUMBER
                        continue

                    try:
                        symbol_info = utils.get_symbol_info(pos.symbol)
                        if not symbol_info:
                            logger.warning(f"⚠️ Não foi possível obter info para {pos.symbol} para gerenciar posição {pos.ticket}.")
                            continue

                        pip_size = utils.get_pip_size(pos.symbol)
                        if pip_size == 0:
                            logger.warning(f"⚠️ Pip size é zero para {pos.symbol}. Não é possível gerenciar posição {pos.ticket}.")
                            continue

                        current_tick = mt5_exec(mt5.symbol_info_tick, pos.symbol)
                        if current_tick is None:
                            logger.warning(f"⚠️ Não foi possível obter tick atual para {pos.symbol}. Não é possível gerenciar posição {pos.ticket}.")
                            continue

                        current_price = current_tick.bid if pos.type == mt5.POSITION_TYPE_BUY else current_tick.ask

                        if pos.type == mt5.POSITION_TYPE_BUY:
                            pips_profit = (current_price - pos.price_open) / pip_size
                        else:
                            pips_profit = (pos.price_open - current_price) / pip_size

                        # ========================================
                        # ✅ v5.0: NEWS-BASED POSITION CLOSING
                        # ========================================
                        # Fecha posições lucrativas 15min antes de notícias alto impacto
                        # ========================================
                        try:
                            # ========================================
                            # ✅ TRIPLE SWAP GUARD (Quarta-Feira)
                            # ========================================
                            # Evita pagar swap triplo se a posição estiver pagando swap
                            # Executa entre 18:00 e 18:55 BRT (Antes do Rollover das 19:00)
                            now_guard = datetime.now()
                            triple_swap_close = False
                            
                            if now_guard.weekday() == 2 and now_guard.hour == 18 and now_guard.minute >= 30:
                                # Verifica Swap da posição
                                if pos.swap < 0:
                                    logger.warning(f"📅 TRIPLE SWAP ALERT: Fechando {pos.symbol} para evitar custo triplo (Swap: {pos.swap}).")
                                    triple_swap_close = True
                            
                            should_close_news, news_reason, minutes_to = news_filter.should_close_for_news(
                                pos.symbol, pos.profit
                            )
                            
                            if (should_close_news and pos.profit > 0) or triple_swap_close:
                                close_reason = f"Triple Swap Avoidance (Swap: {pos.swap})" if triple_swap_close else news_reason
                                logger.info(f"🛡️ Fechando posição {pos.ticket} ({pos.symbol}): {close_reason}")
                                
                                # Fecha a posição
                                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                close_price = current_tick.bid if pos.type == mt5.POSITION_TYPE_BUY else current_tick.ask
                                
                                close_request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": pos.symbol,
                                    "volume": pos.volume,
                                    "type": order_type,
                                    "position": pos.ticket,
                                    "price": close_price,
                                    "deviation": getattr(config, 'DEVIATION', 20),
                                    "magic": getattr(config, 'MAGIC_NUMBER', 123456),
                                    "comment": f"Close: {close_reason}",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                                
                                result = mt5_exec(mt5.order_send, close_request)
                                
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(f"✅ Posição {pos.ticket} fechada com sucesso.")
                                    # ✅ TELEGRAM: Notifica Fechamento
                                    msg_close = (
                                        f"🛡️ <b>XP3 PRO: Fechamento de Proteção</b>\n\n"
                                        f"🆔 Ativo: <b>{pos.symbol}</b>\n"
                                        f"💰 Lucro/Perda: <b>${pos.profit:.2f}</b>\n"
                                        f"⚠️ Motivo: {close_reason}"
                                    )
                                    utils.send_telegram_alert(msg_close)
                                    try:
                                        utils.record_trade_close(
                                            ticket=int(getattr(pos, "ticket", 0)),
                                            symbol=str(getattr(pos, "symbol", "")),
                                            side=("BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"),
                                            volume=float(getattr(pos, "volume", 0.0)),
                                            open_time=datetime.fromtimestamp(getattr(pos, "time", time.time())),
                                            close_time=datetime.now(),
                                            open_price=float(getattr(pos, "price_open", 0.0)),
                                            close_price=float(close_price or 0.0),
                                            sl=float(getattr(pos, "sl", 0.0)),
                                            tp=float(getattr(pos, "tp", 0.0)),
                                            profit=float(getattr(pos, "profit", 0.0)),
                                            commission=float(getattr(pos, "commission", 0.0)) if hasattr(pos, "commission") else 0.0,
                                            swap=float(getattr(pos, "swap", 0.0)),
                                            magic=int(getattr(pos, "magic", getattr(config, 'MAGIC_NUMBER', 123456))),
                                            comment=str(close_reason)
                                        )
                                    except Exception:
                                        pass
                                else:
                                    error_msg = result.comment if result else "result is None"
                                    logger.error(f"❌ Falha ao fechar posição {pos.ticket}: {error_msg}")
                                continue  # Não processa mais esta posição neste ciclo
                        except Exception as e:
                            logger.error(f"❌ Erro ao verificar news/swap close para {pos.symbol}: {e}")

                        # ========================================
                        # ✅ BREAKEVEN PLUS (SNIPER MODE v5.1)
                        # ========================================
                        # Gatilho: 1.5x Risco (1.5R) OU Config Manual
                        # Ação: Move SL para Entrada + 2 pips (Cobre taxas)
                        # ========================================
                        
                        # Cálculo do Risco Inicial (R)
                        if pos.type == mt5.POSITION_TYPE_BUY:
                            initial_risk_pips = (pos.price_open - pos.sl) / pip_size if pos.sl > 0 else 0
                        else:
                            initial_risk_pips = (pos.sl - pos.price_open) / pip_size if pos.sl > 0 else 0
                            
                        # Cálculo do Retorno Atual em R (R-Multiple)
                        current_r = pips_profit / initial_risk_pips if initial_risk_pips > 0 else 0
                        
                        be_trigger_r = getattr(config, 'BREAKEVEN_TRIGGER_R', 1.5)
                        be_buffer_pips = getattr(config, 'BREAKEVEN_BUFFER_PIPS', 2.0)
                        
                        # Verifica se deve acionar o BE
                        should_trigger_be = False
                        if getattr(config, 'ENABLE_BREAKEVEN', True) and pos.sl != 0:
                            # Lógica R-Multiple (Prioridade)
                            if current_r >= be_trigger_r:
                                should_trigger_be = True
                            # Fallback para Pips Fixos (Legacy)
                            elif pips_profit >= getattr(config, 'BREAKEVEN_TRIGGER_PIPS', 15):
                                should_trigger_be = True
                                
                        if should_trigger_be:
                            # Verifica se já está no Breakeven (ou melhor)
                            is_in_profit_zone = False
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                if pos.sl >= pos.price_open: is_in_profit_zone = True
                            else:
                                if pos.sl <= pos.price_open: is_in_profit_zone = True

                            # Se não estiver no BE, move
                            if not is_in_profit_zone:
                                if pos.type == mt5.POSITION_TYPE_BUY:
                                    new_sl = pos.price_open + (be_buffer_pips * pip_size)
                                    if new_sl > pos.sl: # Garante que só sobe
                                        if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                            logger.info(f"🛡️ BREAKEVEN PLUS ({current_r:.1f}R) para {symbol}: SL movido para Entrada + {be_buffer_pips} pips")
                                else:
                                    new_sl = pos.price_open - (be_buffer_pips * pip_size)
                                    if new_sl < pos.sl: # Garante que só desce
                                        if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                            logger.info(f"🛡️ BREAKEVEN PLUS ({current_r:.1f}R) para {symbol}: SL movido para Entrada - {be_buffer_pips} pips")

                        # Trailing Stop
                        if getattr(config, 'ENABLE_TRAILING_STOP', False) and pips_profit >= getattr(config, 'TRAILING_START_PIPS', 100):
                            
                            ts_step = getattr(config, 'TRAILING_STEP_PIPS', 50)
                            ts_dist = getattr(config, 'TRAILING_DISTANCE_PIPS', 30)

                            if pos.type == mt5.POSITION_TYPE_BUY:
                                # Preço Alvo do SL = Preço Atual - Distancia
                                target_sl = current_price - (ts_dist * pip_size)
                                # Lógica de Degrau (Step):
                                # Só move se o Target SL for maior que o SL Atual + Step
                                if target_sl >= (pos.sl + (ts_step * pip_size)):
                                    new_sl = round(target_sl, symbol_info.digits)
                                    if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                        logger.info(f"📈 TRAILING STOP (Step {ts_step}) para {pos.symbol}: SL subiu para {new_sl} (Travando {ts_dist} pips)")

                            else: # SELL
                                target_sl = current_price + (ts_dist * pip_size)
                                # Só move se o Target SL for menor que o SL Atual - Step
                                if target_sl <= (pos.sl - (ts_step * pip_size)):
                                    new_sl = round(target_sl, symbol_info.digits)
                                    if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                        logger.info(f"📉 TRAILING STOP (Step {ts_step}) para {pos.symbol}: SL desceu para {new_sl} (Travando {ts_dist} pips)")

                    except Exception as e:
                        logger.error(f"❌ Erro ao gerenciar posição {pos.ticket} para {pos.symbol}: {e}", exc_info=True)
                        continue
            
            # ✅ PERF: Libera mt5_lock para o painel atualizar antes do sleep longo
            time.sleep(1)

            # ✅ NOVO: Log Periódico de Status (Heartbeat Visual) - A cada ~1 min
            if iteration_count % 4 == 0:
                status_summary = []
                for sym, data in GLOBAL_MONITOR_CACHE.items():
                    # Pega status curto
                    status = data.get('status', 'N/A')
                    if "MONITORANDO" in status:
                        # Se monitorando, mostra o motivo/score
                        reason = data.get('reason', '')
                        ml_val = data.get('ml_score', 0)
                        # 🔄 AJUSTE: Truncamento aumentado para 35 chars para ver valor do Spread
                        status_summary.append(f"{sym}: {reason[:35]} (ML:{ml_val:.0f})")
                    elif "OPERANDO" in status:
                        status_summary.append(f"🔴 {sym}: EM POSIÇÃO")
                    else:
                        status_summary.append(f"{sym}: {status}")
                
                if status_summary:
                    # ✅ Exibe todos os ativos monitorados (sem limite)
                    summary_str = " | ".join(status_summary)
                    logger.info(f"📊 Status Geral: {summary_str}")

            # ===========================
            # BUY DIAGNOSTIC (FOREX_PAIRS)
            # ===========================
            try:
                last_diag_ts = globals().get("_LAST_BUY_DIAG_TS", 0.0)
                do_diag = getattr(config, "ENABLE_BUY_DIAGNOSTIC", True)
                diag_interval = float(getattr(config, "BUY_DIAGNOSTIC_INTERVAL_SECONDS", 90))
                if do_diag and (time.time() - last_diag_ts) >= diag_interval:
                    globals()["_LAST_BUY_DIAG_TS"] = time.time()
                    all_fx = list(getattr(config, "FOREX_PAIRS", {}).keys())
                    max_syms = int(getattr(config, "BUY_DIAGNOSTIC_MAX_SYMBOLS", 30))
                    if not all_fx:
                        pass
                    else:
                        syms_diag = GLOBAL_ACTIVE_SYMBOLS.copy()
                        lines = []
                        for s in syms_diag:
                            try:
                                mt5_exec(mt5.symbol_select, s, True)
                                sig, ind, strat, reason = check_for_signals(s, current_session)
                                if not isinstance(ind, dict) or ind is None:
                                    lines.append(f"{s}: Indicadores indisponíveis ({reason})")
                                    continue
                                params = ELITE_CONFIG.get(s, {})
                                ema_trend = ind.get("ema_trend")
                                rsi_now = ind.get("rsi", 50)
                                adx_now = ind.get("adx", 0)
                                bb_lower = ind.get("bb_lower")
                                close_price = ind.get("close", 0)
                                open_price = ind.get("open", 0)
                                spread_pts = ind.get("spread_points", ind.get("spread_pips", 0))
                                vol_ratio = ind.get("volume_ratio", 0.0)
                                ema_200 = ind.get("ema_200")
                                atr = ind.get("atr", 0.0)
                                price = ind.get("price") or ind.get("close")
                                
                                # Limites
                                base_adx_thr = params.get("adx_threshold", getattr(config, 'ADX_THRESHOLD', 20))
                                try:
                                    low_trend_map = getattr(config, 'LOW_TREND_ADX_THRESHOLDS', {})
                                    adx_thr = low_trend_map.get(s, base_adx_thr)
                                except Exception:
                                    adx_thr = base_adx_thr
                                rsi_req = 50
                                min_vol = getattr(config, 'MIN_VOLUME_COEFFICIENT', 0.4)
                                # Spread max por classe
                                sym_up = s.upper()
                                is_crypto = any(c in sym_up for c in ["BTC","ETH","SOL","ADA","BNB","XRP","LTC","DOGE"])
                                is_indices = any(i in sym_up for i in ["US30","NAS100","USTEC","DE40","GER40","GER30","UK100","US500","USA500","SPX500","HK50","JP225","FRA40"])
                                is_metals = any(m in sym_up for m in ["XAU","XAG","GOLD","SILVER"])
                                is_exotic = any(x in sym_up for x in ["TRY","ZAR","MXN","RUB","CNH","PLN","HUF","CZK","DKK","NOK","SEK"])
                                if is_crypto:
                                    max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
                                elif is_indices:
                                    max_spread = getattr(config, 'MAX_SPREAD_INDICES', 600)
                                elif is_metals:
                                    max_spread = getattr(config, 'MAX_SPREAD_METALS', 80)
                                elif is_exotic:
                                    max_spread = getattr(config, 'MAX_SPREAD_EXOTICS', 8000)
                                else:
                                    max_spread = getattr(config, 'MAX_SPREAD_ACCEPTABLE', 25)
                                # Sessão ajusta spread
                                sess_name = current_session.get("name")
                                if sess_name == "GOLDEN" and not (is_crypto or is_indices or is_metals or is_exotic):
                                    max_spread = max_spread * (1 + getattr(config, 'GOLDEN_SPREAD_ALLOWANCE_PCT', 0.20))
                                elif sess_name == "PROTECTION" and not (is_crypto or is_indices or is_metals or is_exotic):
                                    max_spread = getattr(config, 'PROTECTION_MAX_SPREAD_FOREX', 20)
                                elif sess_name == "ASIAN" and not (is_crypto or is_indices or is_metals or is_exotic):
                                    max_spread = getattr(config, 'ASIAN_MAX_SPREAD_POINTS', 50)
                                
                                missing = []
                                # Trend path readiness (permite NEUTRAL além de UP)
                                if ema_trend not in ("UP", "NEUTRAL"):
                                    missing.append(f"EMA Trend=UP/NEUTRAL (atual {ema_trend or 'N/A'})")
                                if adx_now < adx_thr:
                                    missing.append(f"ADX≥{adx_thr} (atual {adx_now:.1f})")
                                if rsi_now <= rsi_req:
                                    missing.append(f"RSI>{rsi_req} (atual {rsi_now:.1f})")
                                if close_price <= open_price:
                                    missing.append("Candle de Alta (C>O)")
                                # Macro EMA200
                                if getattr(config, 'ENABLE_EMA_200_FILTER', True) and ema_200 is not None and price is not None:
                                    if not (price >= ema_200):
                                        missing.append(f"Preço≥EMA200 (P:{price:.5f} EMA200:{ema_200:.5f})")
                                # Vol/Spread
                                if vol_ratio < min_vol:
                                    missing.append(f"Volume≥{min_vol:.2f} (atual {vol_ratio:.2f})")
                                if spread_pts > max_spread:
                                    missing.append(f"Spread≤{max_spread:.0f} (atual {spread_pts})")
                                # ML requirement (usa ML_MIN_SCORE explícito)
                                try:
                                    ml_conf = ind.get('ml_confidence')
                                    if hasattr(utils, "ml_optimizer_instance") and utils.ml_optimizer_instance and isinstance(ind, dict):
                                        df_ml = ind.get("df")
                                        ml_score_raw, _ = utils.ml_optimizer_instance.get_prediction_score(s, ind, df_ml, signal="BUY")
                                        ml_conf = ml_score_raw / 100.0
                                    if ml_conf is None:
                                        ml_conf = 0.5
                                    min_ml_score = float(getattr(config, "ML_MIN_SCORE", 45))
                                    if (ml_conf * 100) < min_ml_score:
                                        missing.append(f"ML≥{min_ml_score:.0f} (atual {ml_conf*100:.0f})")
                                except Exception:
                                    pass
                                
                                ml_score_int = int(((ind.get('ml_confidence') or 0.5) * 100))
                                if missing:
                                    lines.append(f"{s}: faltam " + "; ".join(missing))
                                    bot_state.update_monitoring(s, "🔵 MONITORANDO", "Faltam: " + " | ".join(missing), float(ml_score_int))
                                    log_signal_analysis(s, "NONE", strat or "N/A", 0, True, " | ".join(missing), ind or {}, session_name=current_session.get("name"))
                                else:
                                    lines.append(f"{s}: pronto para BUY (condições atendidas)")
                                    bot_state.update_monitoring(s, "🟢 PRONTO", "Condições atendidas", float(ml_score_int))
                                    log_signal_analysis(s, sig or "BUY", strat or "TREND", 0, False, "Condições atendidas", ind or {}, session_name=current_session.get("name"))
                            except Exception as e_diag:
                                lines.append(f"{s}: erro diagnóstico ({e_diag})")
                                log_signal_analysis(s, "NONE", strat or "N/A", 0, True, f"Erro diagnóstico: {e_diag}", ind or {}, session_name=current_session.get("name"))
                        if lines:
                            logger.info("🧪 WHY-BUY:\n" + "\n".join(lines))
            except Exception:
                pass

            watchdog.heartbeat("FastLoop")
            time.sleep(getattr(config, 'FAST_LOOP_INTERVAL', 15)) # ✅ CORREÇÃO: getattr para FAST_LOOP_INTERVAL

        except Exception as e:
            # ✅ Land Trading: Log de Erro Sem Morte
            logger.critical(f"💀 ERRO FATAL NA THREAD FastLoop: {e}", exc_info=True)
            time.sleep(5)
            continue
            
    # finally: removido para evitar erro de sintaxe.
    # O Lock Check no início já impede duplicação.
    # Se a thread morrer, o restart do bot limpa a flag.

# ===========================
# SLOW LOOP
# ===========================
def slow_loop():
    """Loop lento: manutenção, re-otimização ML, news filter"""
    safe_log(logging.INFO, "🧠 Slow Loop iniciado")

    last_health_check = 0
    last_ml_reoptimization = datetime.now() # ✅ NOVO: Para controlar re-otimização ML
    last_heartbeat_update = 0
    while not shutdown_event.is_set():
        try:
            current_time_ts = time.time()
            if current_time_ts - last_heartbeat_update > 60:  # Atualiza a cada 60s
                update_heartbeat()
                last_heartbeat_update = current_time_ts
            # ✅ REQUISITO: Verificação de Iteráveis
            if not GLOBAL_ACTIVE_SYMBOLS:
                logger.warning("⚠️ SlowLoop: GLOBAL_ACTIVE_SYMBOLS está vazia! Aguardando 5s...")
                time.sleep(5)
                continue

            watchdog.heartbeat("SlowLoop")
            current_time = time.time()

            if not utils.is_market_open():
                time.sleep(300)
                continue

            # Health check MT5
            if current_time - last_health_check > 60:
                last_health_check = current_time
                if not utils.check_mt5_connection():
                    logger.warning("⚠️ Conexão MT5 perdida. Tentando reconectar...")
                    if not utils.ensure_mt5_connection():
                        bot_state.pause_trading("MT5 desconectado")
                    else:
                        bot_state.resume_trading()

            # ✅ NOVO: Re-otimização ML
            if getattr(config, 'ENABLE_ML_OPTIMIZER', False) and hasattr(utils, 'ml_optimizer_instance'):
                reoptimize_interval = timedelta(hours=getattr(config, 'ML_RETRAIN_INTERVAL_HOURS', 24))
                if (datetime.now() - last_ml_reoptimization) > reoptimize_interval:
                    logger.info("⏳ Acionando re-otimização do ML Optimizer...")
                    utils.ml_optimizer_instance.check_and_reoptimize()
                    global ELITE_CONFIG # Recarrega a elite config após otimização
                    ELITE_CONFIG = load_elite_config()
                    last_ml_reoptimization = datetime.now()

            # ✅ NOVO: News Filter (placeholder)
            if getattr(config, 'ENABLE_NEWS_FILTER', False):
                # Implementar lógica de filtro de notícias aqui
                pass

            # ===========================
            # ✅ v5.2: KILL SWITCH MONITOR (Land Trading)
            # ===========================
            if getattr(config, 'ENABLE_KILL_SWITCH', True):
                min_wr = getattr(config, 'KILL_SWITCH_WIN_RATE', 0.50)
                pause_min = getattr(config, 'KILL_SWITCH_PAUSE_MINUTES', 60)
                
                for symbol in GLOBAL_ACTIVE_SYMBOLS:
                    # Ignora se já estiver pausado
                    if symbol in PAUSED_SYMBOLS:
                        continue
                        
                    wr, total, wins = utils.calculate_rolling_win_rate(symbol)
                    
                    # Atualiza tracker
                    KILL_SWITCH_TRACKER[symbol] = {
                        "win_rate": wr, 
                        "last_check": datetime.now()
                    }
                    
                    # Se WR estiver abaixo do limite (e tiver histórico suficiente)
                    # Nota: utils retorna 50% e 0 trades se não tiver histórico, então só pausa se > 0 trades reais
                    if total >= getattr(config, 'KILL_SWITCH_TRADES', 10) and wr < min_wr:
                        reason = f"Kill Switch: WR {wr:.1%} < {min_wr:.0%} ({wins}/{total} trades)"
                        pause_bot(symbol, pause_min, reason)

            watchdog.heartbeat("SlowLoop")
            
            # ✅ v5.3: METRICS LOGGING (a cada 1 hora)
            current_time_minutes = datetime.now().hour * 60 + datetime.now().minute
            if current_time_minutes % 60 == 0:  # A cada hora cheia
                utils.log_metrics_summary()
            try:
                utils.sync_mt5_trades_to_db()
            except Exception:
                pass
            
            try:
                br_now = utils.get_brasilia_time()
                wd = br_now.weekday()
                hm = br_now.strftime("%H:%M")
                br_date_str = br_now.strftime("%Y-%m-%d")
                try:
                    if getattr(config, "FRIDAY_AUTO_CLOSE_ENABLED", True):
                        cutoff = str(getattr(config, "FRIDAY_AUTO_CLOSE_BRT", "16:30"))
                        buffer_min = int(getattr(config, "FRIDAY_AUTO_CLOSE_BUFFER_MINUTES", 0))
                        if wd == 4 and hm >= cutoff:
                            if LAST_FRIDAY_AUTOCLOSE_DATE != br_date_str:
                                utils.close_all_positions()
                                try:
                                    utils.send_telegram_message(f"🛡️ Fechamento automático de sexta ({cutoff} BRT) executado.")
                                except Exception:
                                    pass
                                LAST_FRIDAY_AUTOCLOSE_DATE = br_date_str
                except Exception as _e:
                    logger.error(f"erro ao fechar automaticamente na sexta: {_e}")
                if wd == 4 and hm >= "19:00":
                    if LAST_FRIDAY_SNAPSHOT_DATE != br_date_str:
                        utils.run_weekly_snapshot()
                        LAST_FRIDAY_SNAPSHOT_DATE = br_date_str
            except Exception:
                logger.error("erro ao executar snapshot semanal")
            
            # ✅ AUTO-EXPORT DIÁRIO (CSV + TXT) via Telegram — Horário de Brasília
            try:
                br_now = utils.get_brasilia_time()
                br_date_str = br_now.strftime("%Y-%m-%d")
                hm = br_now.strftime("%H:%M")
                if LAST_DAILY_EXPORT_DATE != br_date_str and hm >= "23:55":
                    path_csv, summary_csv = utils.export_bot_trades_csv(br_date_str)
                    path_txt, summary_txt = utils.export_bot_trades_txt(br_date_str)
                    if path_csv:
                        caption_csv = (
                            f"<b>Trades (CSV)</b> ({summary_csv.get('date')})\n"
                            f"Total: {summary_csv.get('trades', 0)} | WR: {summary_csv.get('win_rate', 0):.1f}% | "
                            f"PF: {summary_csv.get('profit_factor', 0):.2f} | PnL: ${summary_csv.get('pnl', 0):+.2f}"
                        )
                        if not utils.send_telegram_document(path_csv, caption=caption_csv):
                            utils.send_telegram_message(caption_csv + f"\n\nArquivo: {path_csv}")
                    if path_txt:
                        caption_txt = (
                            f"<b>Trades (TXT)</b> ({summary_txt.get('date')})\n"
                            f"Total: {summary_txt.get('trades', 0)} | WR: {summary_txt.get('win_rate', 0):.1f}% | "
                            f"PF: {summary_txt.get('profit_factor', 0):.2f} | PnL: ${summary_txt.get('pnl', 0):+.2f}"
                        )
                        if not utils.send_telegram_document(path_txt, caption=caption_txt):
                            utils.send_telegram_message(caption_txt + f"\n\nArquivo: {path_txt}")
                    LAST_DAILY_EXPORT_DATE = br_date_str
            except Exception:
                pass
            
            time.sleep(getattr(config, 'SLOW_LOOP_INTERVAL', 300)) # ✅ CORREÇÃO: getattr para SLOW_LOOP_INTERVAL

        except Exception as e:
            logger.critical(f"💀 ERRO FATAL NA THREAD SlowLoop: {e}", exc_info=True)
            time.sleep(5)
            continue

def _telegram_api_request(method: str, payload: dict = None, timeout_s: int = 35) -> Optional[dict]:
    bot_token, _ = utils.get_telegram_credentials()
    if not bot_token:
        return None
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    try:
        import requests
        response = requests.post(url, json=payload or {}, timeout=timeout_s)
        if response.status_code != 200:
            return None
        data = response.json()
        if not isinstance(data, dict) or not data.get("ok"):
            return None
        return data
    except Exception:
        return None

def _format_positions_for_telegram(positions) -> str:
    if not positions:
        return "Sem posições abertas."
    lines = []
    for pos in positions:
        side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        lines.append(f"{pos.symbol} {side} {pos.volume} | PnL: {pos.profit:+.2f} | Ticket: {pos.ticket}")
    text = "\n".join(lines)
    if len(text) > 3500:
        text = text[:3500] + "\n..."
    return text

def _format_status_for_telegram(max_items: int = 25) -> str:
    items = []
    for sym, data in GLOBAL_MONITOR_CACHE.items():
        status = data.get("status", "N/A")
        reason = data.get("reason", "")
        ml = data.get("ml_score", 0)
        items.append((sym, f"{status}: {reason} (ML:{ml:.0f})"))
    items.sort(key=lambda x: x[0])
    if not items:
        return "Sem status disponível ainda (aguardando loop)."
    lines = [f"{sym}: {msg}" for sym, msg in items[:max_items]]
    if len(items) > max_items:
        lines.append(f"... (+{len(items) - max_items} ativos)")
    return "\n".join(lines)

def telegram_command_loop():
    _, chat_id_allowed = utils.get_telegram_credentials()
    chat_id_allowed = str(chat_id_allowed or "").strip()
    # Token é obrigatório; chat_id pode ser aprendido dinamicamente
    if not utils.get_telegram_credentials()[0]:
        return
    try:
        _ = _telegram_api_request("deleteWebhook", {"drop_pending_updates": True}, timeout_s=10)
    except Exception:
        pass

    def _send_telegram_long(text: str, chunk_size: int = 3500):
        remaining = text or ""
        while remaining:
            if len(remaining) <= chunk_size:
                utils.send_telegram_message(remaining)
                break
            cut = remaining.rfind("\n", 0, chunk_size)
            if cut < 500:
                cut = chunk_size
            part = remaining[:cut].rstrip()
            utils.send_telegram_message(part)
            remaining = remaining[cut:].lstrip()

    offset = 0
    try:
        bootstrap = _telegram_api_request("getUpdates", {"timeout": 0, "offset": 0}, timeout_s=10)
        if bootstrap and isinstance(bootstrap.get("result"), list) and bootstrap["result"]:
            offset = max(u.get("update_id", 0) for u in bootstrap["result"]) + 1
    except Exception:
        offset = 0

    utils.send_telegram_message(
        "🤖 <b>Comandos Telegram</b> ativos.\n"
        "Use /help para ver os comandos disponíveis."
    )

    while not shutdown_event.is_set():
        try:
            watchdog.heartbeat("Telegram")
            data = _telegram_api_request("getUpdates", {"timeout": 30, "offset": offset, "allowed_updates": ["message"]}, timeout_s=35)
            if not data:
                continue
            updates = data.get("result") or []
            if not isinstance(updates, list) or not updates:
                continue

            for upd in updates:
                offset = max(offset, upd.get("update_id", 0) + 1)
                msg = (upd.get("message") or {})
                text = (msg.get("text") or "").strip()
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", "")).strip()
                # Se não houver chat_id salvo ainda, vincula dinamicamente ao primeiro chat
                if not chat_id_allowed:
                    chat_id_allowed = chat_id
                    try:
                        utils.set_telegram_chat_id(chat_id_allowed)
                        utils.send_telegram_message_to(chat_id_allowed, "🤝 Bot vinculado a este chat. Use /help para os comandos.")
                    except Exception:
                        pass
                elif chat_id != chat_id_allowed:
                    if text.startswith("/start") or text.startswith("/help") or text.startswith("/ajuda"):
                        chat_id_allowed = chat_id
                        try:
                            utils.set_telegram_chat_id(chat_id_allowed)
                            utils.send_telegram_message_to(chat_id_allowed, "🤝 Bot vinculado a este chat. Use /help para os comandos.")
                        except Exception:
                            pass
                    else:
                        try:
                            utils.send_telegram_message_to(chat_id, "⚠️ Este chat não está vinculado ao bot.\nEnvie /start ou /help para vincular.")
                        except Exception:
                            pass
                        continue
                if not text.startswith("/"):
                    continue

                parts = text.split()
                cmd = parts[0].split("@")[0].lower()
                args = parts[1:]

                logger.info(f"📩 Telegram comando recebido: {text}")
                utils.send_telegram_message(f"✅ Comando recebido: {text}")

                if cmd in ["/help", "/ajuda"]:
                    utils.send_telegram_message(
                        "<b>Comandos</b>\n"
                        "/status - Status dos ativos\n"
                        "/metrics - Métricas do dia\n"
                        "/sessionmetrics - Métricas por sessão\n"
                        "/dayreport [YYYY-MM-DD] - Relatório do dia\n"
                        "/trades [YYYY-MM-DD] - Exporta trades do dia\n"
                        "/trades_txt [YYYY-MM-DD] - Exporta trades em TXT\n"
                        "/snapshot - Snapshot imediato da carteira\n"
                        "/positions - Posições abertas\n"
                        "/pause SYMBOL MIN [motivo] - Pausa símbolo\n"
                        "/resume SYMBOL - Retoma símbolo\n"
                        "/pauseall MIN [motivo] - Pausa bot\n"
                        "/resumeall - Retoma bot\n"
                        "/closeall - Fecha posições do bot\n"
                        "/killswitch - Fecha tudo e encerra"
                    )
                    continue

                if cmd == "/status":
                    utils.send_telegram_message(_format_status_for_telegram())
                    continue

                if cmd == "/metrics":
                    m = utils.calculate_current_metrics()
                    if not m:
                        utils.send_telegram_message("Sem métricas disponíveis agora.")
                        continue
                    utils.send_telegram_message(
                        "<b>Métricas (Hoje)</b>\n"
                        f"Trades: {m.get('trades_today', 0)}\n"
                        f"WR: {m.get('win_rate', 0):.1f}%\n"
                        f"PF: {m.get('profit_factor_today', 0):.2f}\n"
                        f"PnL: ${m.get('pnl_today', 0):+.2f}\n"
                        f"PnL/trade: ${m.get('avg_trade', 0):+.2f}\n"
                        f"DD intraday: {m.get('max_dd_intraday', 0):.2f}%"
                    )
                    continue

                if cmd == "/sessionmetrics":
                    try:
                        utils.send_telegram_message(utils.get_session_metrics_summary())
                    except Exception:
                        utils.send_telegram_message("Falha ao gerar métricas por sessão.")
                    continue

                if cmd == "/dayreport":
                    date_arg = args[0] if args else None
                    try:
                        report = utils.generate_day_report(date_arg)
                        _send_telegram_long(report)
                    except Exception:
                        utils.send_telegram_message("Falha ao gerar dayreport.")
                    continue

                if cmd in ["/trades", "/trade"]:
                    date_arg = args[0] if args else None
                    path, summary = utils.export_bot_trades_csv(date_arg)
                    if not path:
                        utils.send_telegram_message(f"Falha ao exportar trades: {summary.get('error', 'erro desconhecido')}")
                        continue
                    caption = (
                        f"<b>Trades</b> ({summary.get('date')})\n"
                        f"Total: {summary.get('trades', 0)} | WR: {summary.get('win_rate', 0):.1f}% | "
                        f"PF: {summary.get('profit_factor', 0):.2f} | PnL: ${summary.get('pnl', 0):+.2f}"
                    )
                    if not utils.send_telegram_document(path, caption=caption):
                        utils.send_telegram_message(caption + f"\n\nArquivo: {path}")
                    continue

                if cmd in ["/trades_txt", "/tradestxt"]:
                    date_arg = args[0] if args else None
                    path, summary = utils.export_bot_trades_txt(date_arg)
                    if not path:
                        utils.send_telegram_message(f"Falha ao exportar trades: {summary.get('error', 'erro desconhecido')}")
                        continue
                    caption = (
                        f"<b>Trades (TXT)</b> ({summary.get('date')})\n"
                        f"Total: {summary.get('trades', 0)} | WR: {summary.get('win_rate', 0):.1f}% | "
                        f"PF: {summary.get('profit_factor', 0):.2f} | PnL: ${summary.get('pnl', 0):+.2f}"
                    )
                    if not utils.send_telegram_document(path, caption=caption):
                        utils.send_telegram_message(caption + f"\n\nArquivo: {path}")
                    continue

                if cmd == "/positions":
                    positions = mt5_exec(mt5.positions_get)
                    utils.send_telegram_message("<b>Posições</b>\n" + _format_positions_for_telegram(positions))
                    continue
                
                if cmd == "/balance":
                    acc = mt5_exec(mt5.account_info)
                    if not acc:
                        utils.send_telegram_message("Conta indisponível no momento.")
                        continue
                    utils.send_telegram_message(
                        "<b>Finanças</b>\n"
                        f"Saldo: ${acc.balance:,.2f}\n"
                        f"Equity: ${acc.equity:,.2f}\n"
                        f"Margem Livre: ${acc.margin_free:,.2f}\n"
                        f"PnL Atual: ${acc.profit:+,.2f}"
                    )
                    continue
                
                if cmd in ["/history", "/hist"]:
                    date_arg = args[0] if args else None
                    path, summary = utils.export_bot_trades_txt(date_arg)
                    if not path:
                        utils.send_telegram_message(f"Falha ao obter histórico: {summary.get('error', 'erro desconhecido')}")
                        continue
                    caption = (
                        f"<b>Histórico</b> ({summary.get('date')})\n"
                        f"Total: {summary.get('trades', 0)} | WR: {summary.get('win_rate', 0):.1f}% | "
                        f"PF: {summary.get('profit_factor', 0):.2f} | PnL: ${summary.get('pnl', 0):+.2f}"
                    )
                    if not utils.send_telegram_document(path, caption=caption):
                        utils.send_telegram_message(caption + f"\n\nArquivo: {path}")
                    continue

                if cmd in ["/snapshot", "/snap"]:
                    try:
                        result = utils.run_weekly_snapshot()
                        utils.send_telegram_message(f"Snapshot executado.\nJSON: {result.get('paths', {}).get('json')}")
                    except Exception:
                        utils.send_telegram_message("Falha ao executar snapshot.")
                    continue
                
                if cmd in ["/adaptive_backtest", "/abtest"]:
                    try:
                        from risk_engine import adaptive_manager
                        window = int(args[0]) if args else getattr(config, "ADAPTIVE_BACKTEST_DEFAULT_WINDOW", 100)
                        res = adaptive_manager.backtest(window=window)
                        utils.send_telegram_message(
                            "<b>Adaptive TP/SL Backtest</b>\n"
                            f"Janela: {int(window)} trades\n"
                            f"PnL estático: ${res.get('pnl_static', 0.0):+.2f}\n"
                            f"PnL dinâmico (escala TP): ${res.get('pnl_dynamic_scaled', 0.0):+.2f}"
                        )
                    except Exception as e:
                        utils.send_telegram_message(f"Falha no backtest adaptativo: {e}")
                    continue
                
                if cmd == "/riskstatus":
                    try:
                        import sqlite3, time
                        from risk_engine import DB_PATH as _DB
                        if not os.path.exists(_DB):
                            utils.send_telegram_message("Sem registros de bloqueio.")
                        else:
                            conn = sqlite3.connect(_DB)
                            cur = conn.cursor()
                            now_ts = int(time.time())
                            cur.execute("SELECT symbol, reason, end_ts FROM blocks WHERE end_ts > ? ORDER BY end_ts ASC LIMIT 50", (now_ts,))
                            rows = cur.fetchall()
                            conn.close()
                            if not rows:
                                utils.send_telegram_message("Sem bloqueios ativos.")
                            else:
                                lines = ["<b>Bloqueios Ativos</b>"]
                                for s, r, e in rows:
                                    until = datetime.fromtimestamp(e).strftime("%H:%M")
                                    lines.append(f"{s}: {r} (até {until})")
                                utils.send_telegram_message("\n".join(lines))
                    except Exception:
                        utils.send_telegram_message("Falha ao consultar bloqueios.")
                    continue
                
                if cmd == "/riskunblock" and len(args) >= 1:
                    try:
                        import sqlite3
                        from risk_engine import DB_PATH as _DB
                        sym = args[0].upper()
                        if os.path.exists(_DB):
                            conn = sqlite3.connect(_DB)
                            cur = conn.cursor()
                            cur.execute("DELETE FROM blocks WHERE symbol = ?", (sym,))
                            conn.commit()
                            conn.close()
                        if sym in PAUSED_SYMBOLS:
                            del PAUSED_SYMBOLS[sym]
                        utils.send_telegram_message(f"Desbloqueado {sym}.")
                    except Exception:
                        utils.send_telegram_message("Falha ao desbloquear símbolo.")
                    continue

                if cmd == "/pause" and len(args) >= 2:
                    symbol = args[0].upper()
                    minutes = int(float(args[1]))
                    reason = " ".join(args[2:]).strip() or "Telegram"
                    pause_bot(symbol, minutes, reason)
                    utils.send_telegram_message(f"⏸️ Pausado {symbol} por {minutes} min. Motivo: {reason}")
                    continue

                if cmd == "/resume" and len(args) >= 1:
                    symbol = args[0].upper()
                    if symbol in PAUSED_SYMBOLS:
                        del PAUSED_SYMBOLS[symbol]
                    bot_state.update_monitoring(symbol, "🔵 MONITORANDO", "Retomado via Telegram", 0.0)
                    utils.send_telegram_message(f"▶️ Retomado {symbol}.")
                    continue

                if cmd == "/pauseall" and len(args) >= 1:
                    minutes = int(float(args[0]))
                    reason = " ".join(args[1:]).strip() or "Telegram"
                    bot_state.pause_trading(f"{reason} ({minutes} min)")
                    utils.send_telegram_message(f"⏸️ Bot pausado por {minutes} min. Motivo: {reason}")
                    seconds = max(1, min(minutes * 60, 12 * 3600))
                    def _auto_resume():
                        time.sleep(seconds)
                        bot_state.resume_trading()
                        utils.send_telegram_message("▶️ Bot retomado automaticamente.")
                    threading.Thread(target=_auto_resume, daemon=True).start()
                    continue

                if cmd == "/resumeall":
                    bot_state.resume_trading()
                    utils.send_telegram_message("▶️ Bot retomado.")
                    continue

                if cmd == "/closeall":
                    utils.close_all_positions()
                    utils.send_telegram_message("✅ Fechamento solicitado para todas as posições do bot.")
                    continue

                if cmd in ["/killswitch", "/shutdown"]:
                    ks = Path(getattr(config, 'KILLSWITCH_FILE', 'killswitch.txt'))
                    try:
                        ks.touch()
                    except Exception:
                        pass
                    utils.close_all_positions()
                    shutdown_event.set()
                    utils.send_telegram_message("🚨 Kill switch acionado. Fechando posições e encerrando.")
                    break

                utils.send_telegram_message("Comando não reconhecido. Use /help.")

        except Exception:
            time.sleep(2)

# ===========================
# PANEL
# ===========================
def render_panel_enhanced():
    """Painel visual com Rich"""
    if not RICH_AVAILABLE or not getattr(config, 'ENABLE_DASHBOARD', False): # ✅ CORREÇÃO: getattr para ENABLE_DASHBOARD
        return

    def generate_display() -> Layout:
        layout = Layout()

        # Header
        market = utils.get_market_status()
        acc = mt5_exec(mt5.account_info)

        # ✅ REQUISITO: Proteção do Painel (Loading State)
        monitoring_data = bot_state.get_monitoring_status()
        if not acc or not monitoring_data:
            return Layout(Panel("⏳ CARREGANDO DADOS... (Aguardando indicadores)", title="XP3 PRO", border_style="yellow"))

        header_text = Text()
        header_text.append("🚀 XP3 PRO FOREX v4.2\n", style="bold cyan") # ✅ CORREÇÃO: Versão
        
        # ✅ NOVO: Exibição de PnL no Topo
        profit_color = "green" if acc.profit >= 0 else "red"
        header_text.append("💰 PnL Atual: ", style="bold white")
        header_text.append(f"${acc.profit:+,.2f}\n", style=f"bold {profit_color}")
        
        header_text.append(f"{market['emoji']} {market['message']}\n", style=market.get('color', 'green'))
        header_text.append(f"� Equity: ${acc.equity:,.2f} | Balance: ${acc.balance:,.2f}\n", style="white")
        header_text.append(f"📊 Elite Config: {len(ELITE_CONFIG)} símbolos otimizados\n", style="yellow")

        is_paused, pause_reason = bot_state.is_paused()
        if is_paused:
            header_text.append(f"⏸️ BOT PAUSADO: {pause_reason}\n", style="bold red")

        # --- NEWS & RISK STATUS v5.0 ---
        # News Status (Usando USD como proxy para o status geral ou o primeiro do top_pairs)
        top_pairs = bot_state.get_top_pairs()
        news_sym = top_pairs[0] if top_pairs else "EURUSD"
        is_blackout, news_msg = news_filter.is_news_blackout(news_sym)
        news_color = "red" if is_blackout else "green"
        header_text.append(f"📰 Status Notícias ({news_sym}): ", style="white")
        header_text.append(f"{news_msg}\n", style=news_color)

        # Risk Status
        daily_loss = acc.equity - acc.balance
        # ===========================
        # HEADER (Real-Time PnL & Finances)
        # ===========================
        acc_info = mt5_exec(mt5.account_info)
        current_profit = acc_info.profit if acc_info else 0.0
        
        # Cores Financeiras (Solicitação do Usuário)
        profit_color = "green" if current_profit >= 0 else "red"
        
        # Equity Color logic: Cyan if >= Balance, else Yellow/Red logic
        balance = acc_info.balance if acc_info else 0.0
        equity = acc_info.equity if acc_info else 0.0
        margin_free = acc_info.margin_free if acc_info else 0.0
        
        equity_style = "bold cyan" if equity >= balance else "bold yellow" if equity >= (balance * 0.95) else "bold red"
        
        header_text = Text() # Reinicia texto
        header_text.append("💰 PnL: ", style="bold white")
        header_text.append(f"${current_profit:+,.2f}", style=f"bold {profit_color}")
        header_text.append(" | 🏦 Saldo: ", style="bold white")
        header_text.append(f"${balance:,.2f}", style="bold green") 
        header_text.append(" | 📈 Equity: ", style="bold white")
        header_text.append(f"${equity:,.2f}", style=equity_style)
        header_text.append(" | 💸 Margem Livre: ", style="bold white")
        header_text.append(f"${margin_free:,.2f}", style="bold white")
        
        try:
            basis = str(getattr(config, 'MAX_TOTAL_EXPOSURE_BASIS', 'balance')).lower()
            limit_mult = float(getattr(config, 'MAX_TOTAL_EXPOSURE_MULTIPLIER', 2.0))
            base_val = balance if basis == "balance" else equity
            authorized = base_val * limit_mult
            positions = mt5_exec(mt5.positions_get)
            def _estimate(symbol: str, volume: float) -> float:
                info = utils.get_symbol_info(symbol)
                if not info or volume <= 0:
                    return 0.0
                contract = float(getattr(info, 'trade_contract_size', 100000) or 100000)
                tick = mt5_exec(mt5.symbol_info_tick, symbol)
                price = float(getattr(tick, 'bid', 0.0) or 0.0) if tick else 0.0
                s = symbol.upper()
                if len(s) >= 6 and s[3:6] == "USD":
                    return contract * volume * (price if price > 0 else 1.0)
                if len(s) >= 6 and s[0:3] == "USD":
                    return contract * volume
                if ("XAUUSD" in s) or ("XAGUSD" in s) or ("US30" in s) or ("US500" in s) or ("NAS100" in s) or ("USTEC" in s) or ("USA500" in s):
                    return contract * volume * (price if price > 0 else 1.0)
                return contract * volume
            exposure = 0.0
            for p in (positions or []):
                exposure += _estimate(p.symbol, float(getattr(p, 'volume', 0.0) or 0.0))
            usage = (exposure / authorized) if authorized > 0 else 0.0
            warn_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_WARNING_PCT', 0.80))
            alert_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_ALERT_PCT', 0.95))
            usage_style = "bold green"
            usage_label = f"{usage*100:.0f}%"
            if usage >= alert_pct:
                usage_style = "bold red"
                usage_label = f"🚨 {usage*100:.0f}%"
            elif usage >= warn_pct:
                usage_style = "bold yellow"
                usage_label = f"⚠️ {usage*100:.0f}%"
            header_text.append("\n🛡️ Margem Autorizada: ", style="bold white")
            header_text.append(f"${authorized:,.0f}", style="bold cyan")
            header_text.append(" | Exposição: ", style="bold white")
            header_text.append(f"${exposure:,.0f}", style=usage_style)
            header_text.append(" | Uso: ", style="bold white")
            header_text.append(usage_label, style=usage_style)
        except Exception:
            pass
        
        # Verifica status global
        total_open_positions = len(mt5_exec(mt5.positions_get) or [])
        global_max = getattr(config, 'MAX_GLOBAL_ALGO_ORDERS', 3)
        
        if total_open_positions >= global_max:
            system_status = "[bold red]🚨 LIMIT REACHED[/bold red]"
        elif bot_state.is_paused()[0]:
            system_status = f"[yellow]⏸️ PAUSADO: {bot_state.is_paused()[1]}[/yellow]"
        else:
            system_status = "[bold green]⚡ ATIVO[/bold green]"

        # ✅ SESSION STATUS
        session = utils.get_current_trading_session()
        s_display = session['display']
        s_emoji = session['emoji']
        s_color = "bold yellow" if session['name'] == "GOLDEN" else "bold red" if session['name'] == "PROTECTION" else f"bold {session['color']}"

        header_text.append(f"\n📊 Ordens: {total_open_positions}/{global_max} | {system_status}")
        header_text.append(f" | 🕒 SESSÃO: ", style="bold white")
        header_text.append(f"{s_display} {s_emoji}", style=s_color)
        
        header_panel = Panel(header_text, style="bold white", border_style="blue")

        # Posições
        # ===========================
        # POSITIONS (Limit 10)
        # ===========================
        pos_table = Table(show_header=True, header_style="bold green")
        pos_table.add_column("Symbol")
        pos_table.add_column("Side")
        pos_table.add_column("Profit")
        pos_table.add_column("Pips")
        pos_table.add_column("SL") # ✅ NOVO
        pos_table.add_column("TP") # ✅ NOVO

        positions = mt5_exec(mt5.positions_get)
            
        if positions:
            # ✅ MEMORY CLEAN: Limita a 10 posições visuais
            for pos in list(positions)[-10:]:
                if pos.magic != getattr(config, 'MAGIC_NUMBER', 123456): # ✅ CORREÇÃO: getattr para MAGIC_NUMBER
                    continue

                symbol_info = utils.get_symbol_info(pos.symbol)
                symbol_info_pos = mt5_exec(mt5.symbol_info, pos.symbol)
                digits = symbol_info_pos.digits if symbol_info_pos else 5
                
                # ✅ CORREÇÃO: Pip Size Robusto
                if symbol_info_pos:
                    if digits == 3 or digits == 5:
                        pip_size = symbol_info_pos.point * 10
                    else:
                        pip_size = symbol_info_pos.point
                else:
                    pip_size = 0.0001 # Fallback inseguro mas evita crash

                if pip_size == 0: # Evita divisão por zero
                    pips = 0.0
                elif pos.type == mt5.POSITION_TYPE_BUY:
                    pips = (pos.price_current - pos.price_open) / pip_size
                    side = "🟢 BUY"
                else:
                    pips = (pos.price_open - pos.price_current) / pip_size
                    side = "🔴 SELL"

                profit_color = "green" if pos.profit >= 0 else "red"
                pips_color = "green" if pips >= 0 else "red"

                pos_table.add_row(
                    pos.symbol,
                    side,
                    f"[{profit_color}]${pos.profit:+.2f}[/{profit_color}]",
                    f"[{pips_color}]{pips:+.1f}[/{pips_color}]",
                    f"{pos.sl:.{digits}f}", # ✅ NOVO: Mostra SL/TP
                    f"{pos.tp:.{digits}f}"
                )
        else:
            pos_table.add_row("-", "-", "-", "[dim](Nenhuma)[/dim]", "-", "-")

        positions_panel = Panel(pos_table, title="💼 POSIÇÕES (Last 10)", border_style="green")

        # Análises
        analysis_table = Table(show_header=True, header_style="bold yellow")
        analysis_table.add_column("Hora")
        analysis_table.add_column("Par")
        analysis_table.add_column("Sinal")
        analysis_table.add_column("Score")
        analysis_table.add_column("Status")
        analysis_table.add_column("Preço") # ✅ NOVO: Adiciona preço

        with signal_history_lock:
            recent = list(signal_history)[-10:]

            for analysis in reversed(recent):
                # Land Trading: Filtro de histórico
                if not getattr(config, 'SHOW_REJECTED_SIGNALS_HISTORY', True) and analysis.rejected:
                    pass

                time_str = analysis.timestamp.strftime("%H:%M:%S")

                if analysis.signal == "BUY":
                    signal_display = "[green]🟢BUY[/green]"
                elif analysis.signal == "SELL":
                    signal_display = "[red]🔴SELL[/red]"
                else:
                    signal_display = "[dim]--[/dim]"

                score_color = "green" if analysis.score >= 60 else "white" if analysis.score >= 40 else "dim"
                score_display = f"[{score_color}]{analysis.score:.0f}[/{score_color}]"

                reason = analysis.rejection_reason
                status_color = "bold green" if "EXECUTADA" in reason else "yellow" if "Spread Alto" in reason else "dim"

                # ✅ NOVO: Preço de fechamento da análise
                close_price_display = f"{analysis.indicators.get('close', 0):.5f}" if analysis.indicators.get('close', 0) > 0 else "[dim]N/A[/dim]"

                analysis_table.add_row(
                    f"[dim]{time_str}[/dim]",
                    analysis.symbol,
                    signal_display,
                    score_display,
                    f"[{status_color}]{reason[:50]}[/{status_color}]",
                    close_price_display
                )

        analysis_panel = Panel(analysis_table, title="🔍 ÚLTIMOS SINAIS", border_style="yellow")

        # MONITORAMENTO GLOBAL v5.1 (Land Trading)
        monitor_table = Table(box=box.MINIMAL, expand=True, border_style="bright_black")
        monitor_table.add_column("SYMBOL", style="cyan", width=12)
        monitor_table.add_column("STATUS", width=18)
        monitor_table.add_column("MOTIVO / TÉCNICA", style="white") # Renomeado para refletir conteúdo
        monitor_table.add_column("SPR", justify="right", width=5)   # ✅ NOVO COLUNA SPREAD
        monitor_table.add_column("ML", justify="right", width=6)

        # Usa o GLOBAL_MONITOR_CACHE solicitado pelo usuário
        for sym in sorted(GLOBAL_MONITOR_CACHE.keys()):
            m = GLOBAL_MONITOR_CACHE[sym]
            status = m['status']
            ml = m['ml_score']
            reason_text = m['reason']
            
            # 1. Cor pelo Status
            s_style = "white"
            if "SINAL" in status: s_style = "green"
            elif "OPERANDO" in status: s_style = "bright_blue"
            elif "BLOQUEADO" in status: s_style = "yellow"
            elif "MONITORANDO" in status: s_style = "blue"
            elif "ERRO" in status: s_style = "bold red"

            # 2. Cor pelo ML e Suffix de Baseline
            ml_style = "green" if ml > 60 else "bright_black" if ml < 40 else "white"
            ml_text = f"{ml:.0f}"
            if m.get("is_baseline", False):
                ml_text += " [dim](B)[/dim]" # (B) para Confiança Estatística (Backtest)
                ml_style = "yellow" if ml > 60 else ml_style # Amarelo indica "Estatística"
            
            # 3. Cor pelo Motivo
            reason_style = "yellow" if "Spread Alto" in reason_text else "white"
            if "❌" in reason_text: reason_style = "red" # Destaque para Vetos

            # 4. Spread (Busca do Cache ou Recalcula)
            spread_val = m.get('spread', 0)
            
            # Recalcula limite para validar cor (Lógica duplicada do fast_loop para display)
            symbol_upper = sym.upper()
            if any(idx in symbol_upper for idx in ["US30", "NAS100", "USTEC", "DE40", "GER40", "UK100", "US500", "USA500"]):
                max_spread = getattr(config, 'MAX_SPREAD_INDICES', 150)
            elif any(met in symbol_upper for met in ["XAU", "XAG", "GOLD", "SILVER"]):
                max_spread = getattr(config, 'MAX_SPREAD_METALS', 60)
            elif any(crypto in symbol_upper for crypto in ["BTC", "ETH", "SOL", "ADA", "BNB"]):
                max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
            else:
                max_spread = getattr(config, 'MAX_SPREAD_FOREX', 25)

            spread_style = "bold red" if spread_val > max_spread else "dim" # ✅ RED if above limit
            spread_display = str(spread_val)

            monitor_table.add_row(
                sym,
                Text(status, style=s_style),
                Text(reason_text[:75], style=reason_style), 
                Text(spread_display, style=spread_style), # ✅ Applied Style
                Text.from_markup(f"[{ml_style}]{ml_text}[/{ml_style}]")
            )
        
        monitor_panel = Panel(monitor_table, title="🛰️ LAND TRADING - MONITORAMENTO EM TEMPO REAL", border_style="blue")

        # Layout
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body", ratio=3)
        )

        layout["header"].update(header_panel)

        layout["body"].split_column(
            Layout(name="monitoring", ratio=2),
            Layout(name="analysis", ratio=1),
            Layout(name="positions", ratio=1)
        )

        layout["body"]["monitoring"].update(monitor_panel)
        layout["body"]["analysis"].update(analysis_panel)
        layout["body"]["positions"].update(positions_panel)

        return layout

    try:
        with Live(
            generate_display(),
            console=console,
            screen=True,
            refresh_per_second=1,
            auto_refresh=False
        ) as live:
            while not shutdown_event.is_set():
                try:
                    watchdog.heartbeat("Panel")
                    live.update(generate_display(), refresh=True)
                    panel_interval = max(getattr(config, 'DASHBOARD_REFRESH_RATE', 1), 3)
                    time.sleep(panel_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.critical(f"💀 ERRO FATAL NA THREAD Panel: {e}", exc_info=True)
                    time.sleep(5)
                
    except Exception as e:
        logger.error(f"❌ Painel falhou: {e}", exc_info=True)

# ===========================
# SIGNAL HANDLING (v5.0.7)
# ===========================
def handle_exit(sig, frame):
    """Garante fechamento limpo solicitado pelo sistema ou usuário."""
    logger.info(f"🛑 Fechamento limpo solicitado pelo sistema (Sinal {sig}).")
    shutdown_event.set()

# Configura sinais de encerramento
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ===========================
# MAIN
# ===========================
def main():
    print("="*80)
    print("🚀 XP3 PRO FOREX BOT v4.2") # ✅ CORREÇÃO: Versão
    print("="*80)

    try:
        Path("bot_heartbeat.timestamp").touch()
    except:
        pass

    if not mt5_exec(mt5.initialize, path=getattr(config, 'MT5_TERMINAL_PATH', None)):
        logger.critical(f"❌ Falha ao inicializar MT5 em: {getattr(config, 'MT5_TERMINAL_PATH', 'Caminho não especificado')}")
        return
    
    sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
    sector_map = getattr(config, "SECTOR_MAP", None)
    allowed = None
    if isinstance(sector_map, dict) and sector in sector_map:
        allowed = list(sector_map.get(sector) or [])
    else:
        allowed = list(getattr(config, "SYMBOL_MAP", []) or [])

    sync_result = utils.sync_market_watch(allowed)
    
    # ✅ REQUISITO: Log de Versão MT5 para Diagnóstico
    try:
        ver = mt5_exec(mt5.version)
        print(f"✅ MT5 Terminal Version: {ver}")
        logger.info(f"📊 MT5 Terminal Version: {ver}")
    except:
        pass

    account_info = mt5_exec(mt5.account_info)
    if not account_info:
        logger.critical("❌ Não foi possível obter informações da conta")
        mt5_exec(mt5.shutdown)
        return

    print(f"✅ Conta: {account_info.login} | Balance: ${account_info.balance:,.2f}")

    # ✅ NOVO: Inicializa ML Optimizer e o anexa ao utils para acesso global
    if getattr(config, 'ENABLE_ML_OPTIMIZER', False):
        try:
            from ml_optimizer import EnsembleOptimizer
            utils.ml_optimizer_instance = EnsembleOptimizer()
            logger.info("✅ ML Optimizer inicializado e anexado ao utils.")
        except ImportError as e:
            logger.error(f"❌ Não foi possível importar ML Optimizer: {e}. Desabilitando ML Optimizer.")
            config.ENABLE_ML_OPTIMIZER = False # Desabilita se não puder importar
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar ML Optimizer: {e}. Desabilitando ML Optimizer.", exc_info=True)
            config.ENABLE_ML_OPTIMIZER = False

    # ✅ REQUISITO: Sincronização e Filtragem de Ativos
    
    global GLOBAL_ACTIVE_SYMBOLS
    kept = (sync_result or {}).get("kept") if isinstance(sync_result, dict) else None
    GLOBAL_ACTIVE_SYMBOLS = kept if isinstance(kept, list) and kept else filter_and_validate_symbols()

    if getattr(config, "ENABLE_DYNAMIC_ASSET_SELECTION", True):
        try:
            sel_path = Path(getattr(config, "DATA_DIR", "data")) / "selected_assets.json"
            if sel_path.exists():
                payload = json.loads(sel_path.read_text(encoding="utf-8"))
                desired = list(payload.get("symbols", []))
                if desired:
                    desired_set = set(desired)
                    GLOBAL_ACTIVE_SYMBOLS = [s for s in GLOBAL_ACTIVE_SYMBOLS if s in desired_set]
        except Exception as e:
            logger.warning(f"⚠️ Falha ao aplicar seleção dinâmica de ativos: {e}")
    
    if not GLOBAL_ACTIVE_SYMBOLS:
        logger.critical("🚨 [CRITICAL] Nenhum símbolo da configuração foi encontrado no seu Terminal MT5. Verifique os nomes no Market Watch.")
        mt5_exec(mt5.shutdown)
        return

    logger.info(f"📊 Sessão iniciada com {len(GLOBAL_ACTIVE_SYMBOLS)} ativos operacionais.")

    # ✅ NOVO: Teste de Telegram no Boot
    if utils.get_telegram_credentials()[0]:
        utils.send_telegram_message("🚀 <b>XP3 PRO FOREX</b> Iniciado com Sucesso!\n\nPronto para operar sob regras da <i>Land Trading</i>.")

    # Registra threads
    watchdog.register_thread("FastLoop", fast_loop)
    watchdog.register_thread("SlowLoop", slow_loop)
    if utils.get_telegram_credentials()[0] and getattr(config, 'ENABLE_TELEGRAM_COMMANDS', True):
        watchdog.register_thread("Telegram", telegram_command_loop)

    # ✅ CORREÇÃO: Verifica ENABLE_DASHBOARD e lança via Streamlit (Headless Bot)
    if getattr(config, 'ENABLE_DASHBOARD', False):
        # watchdog.register_thread("Panel", render_panel_enhanced) # ❌ Disabled Render Panel
        try:
             # Lança o dashboard em processo separado na porta 8502
             cmd = ["streamlit", "run", "dashboard.py", "--server.port", "8502", "--server.headless", "true"]
             # subprocess.Popen(cmd, cwd=os.getcwd(), shell=True) # shell=True pode ser problemático com sinais, mas ok para Windows
             # Para evitar bloquear, usamos Popen.
             # No Windows, shell=True ajuda a resolver caminhos.
             subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), shell=True)
             logger.info("✅ Dashboard iniciado no navegador (Porta 8502)")
        except Exception as e:
             logger.error(f"❌ Falha ao iniciar dashboard: {e}")

    # ✅ REQUISITO: Sincronização de Nomes

    # Inicia threads
    threads = []
    for name, info in watchdog.threads.items():
        thread = threading.Thread(
            target=info["target"],
            args=info["args"],
            daemon=True,
            name=name
        )
        thread.start()
        threads.append(thread)
        print(f"✅ Thread '{name}' iniciada")
    hb = threading.Thread(target=heartbeat_writer, daemon=True, name="HeartbeatWriter")
    hb.start()
    threads.append(hb)

    print("="*80)
    print("📊 Sistema ativo")

    try:
        while not shutdown_event.is_set():
            watchdog.check_and_restart()
            time.sleep(getattr(config, 'WATCHDOG_CHECK_INTERVAL', 60))
    except KeyboardInterrupt:
        print("\n⏹️ Encerrando...")
    finally:
        shutdown_event.set()
        # Espera as threads terminarem
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=5) # Dá um tempo para a thread terminar
        mt5_exec(mt5.shutdown)
        print("👋 Shutdown completo")

def start():
    print("="*80)
    print("🚀 XP3 PRO FOREX BOT v4.2 (Linear Mode)")
    print("="*80)
    try:
        Path("bot_heartbeat.timestamp").touch()
    except:
        pass
    if not mt5_exec(mt5.initialize, path=getattr(config, 'MT5_TERMINAL_PATH', None)):
        logger.critical(f"❌ Falha ao inicializar MT5 em: {getattr(config, 'MT5_TERMINAL_PATH', 'Caminho não especificado')}")
        return
    try:
        ver = mt5_exec(mt5.version)
        logger.info(f"📊 MT5 Terminal Version: {ver}")
    except:
        pass
    account_info = mt5_exec(mt5.account_info)
    if not account_info:
        logger.critical("❌ Não foi possível obter informações da conta")
        mt5_exec(mt5.shutdown)
        return
    print(f"✅ Conta: {account_info.login} | Balance: ${account_info.balance:,.2f}")
    if getattr(config, 'ENABLE_ML_OPTIMIZER', False):
        try:
            from ml_optimizer import EnsembleOptimizer
            utils.ml_optimizer_instance = EnsembleOptimizer()
            logger.info("✅ ML Optimizer inicializado e anexado ao utils.")
        except Exception as e:
            logger.error(f"❌ Falha ao inicializar ML Optimizer: {e}")
            config.ENABLE_ML_OPTIMIZER = False
    sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
    sector_map = getattr(config, "SECTOR_MAP", None)
    allowed = list(sector_map.get(sector) or []) if isinstance(sector_map, dict) and sector in sector_map else list(getattr(config, "SYMBOL_MAP", []) or [])
    sync_result = utils.sync_market_watch(allowed)
    global GLOBAL_ACTIVE_SYMBOLS
    kept = (sync_result or {}).get("kept") if isinstance(sync_result, dict) else None
    GLOBAL_ACTIVE_SYMBOLS = kept if isinstance(kept, list) and kept else filter_and_validate_symbols()
    if not GLOBAL_ACTIVE_SYMBOLS:
        logger.critical("🚨 [CRITICAL] Nenhum símbolo válido encontrado no MT5.")
        mt5_exec(mt5.shutdown)
        return
    logger.info(f"📊 Sessão iniciada com {len(GLOBAL_ACTIVE_SYMBOLS)} ativos operacionais (Linear).")
    try:
        fast_loop()  # Executa loop único e linear
    except KeyboardInterrupt:
        logger.info("🛑 Encerrando Linear Mode...")
    finally:
        shutdown_event.set()
        try:
            mt5_exec(mt5.shutdown)
        except:
            pass

if __name__ == "__main__":
    try:
        start()
    except Exception as e:
        logger.critical(f"💀 Erro fatal na inicialização (Linear): {e}", exc_info=True)
        try:
            mt5_exec(mt5.shutdown)
        except:
            pass
        time.sleep(5)
        sys.exit(1)
=======
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False
    print("⚠️ Rich não disponível - usando logs simples")


def update_heartbeat():
    """Atualiza o arquivo de heartbeat para o watchdog"""
    try:
        Path("bot_heartbeat.timestamp").touch()
    except Exception as e:
        logger.debug(f"Erro ao atualizar heartbeat: {e}")
def heartbeat_writer():
    try:
        interval = max(1, int(getattr(config, 'HEARTBEAT_WRITE_INTERVAL', 5)))
    except Exception:
        interval = 5
    while not shutdown_event.is_set():
        try:
            update_heartbeat()
        except Exception:
            pass
        time.sleep(interval)
# ===========================
# LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOGS_DIR / "xp3_forex.log", encoding="utf-8"), # ✅ CORREÇÃO: Usa LOGS_DIR
    ],
)
logger = logging.getLogger("XP3_BOT")
try:
    from pythonjsonlogger import jsonlogger
    _json_handler = logging.FileHandler(config.LOGS_DIR / "xp3_forex.json.log", encoding="utf-8")
    _json_formatter = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s %(symbol)s %(score)s %(strategy)s")
    _json_handler.setFormatter(_json_formatter)
    logging.getLogger().addHandler(_json_handler)
except Exception:
    pass

# ✅ Land Trading: Garante diretório de logs
Path("analysis_logs").mkdir(exist_ok=True)

def safe_log(level, message, *args, **kwargs):
    """Loga apenas se dashboard desabilitado ou em arquivo"""
    if not getattr(config, 'ENABLE_DASHBOARD', False) or not RICH_AVAILABLE: # ✅ CORREÇÃO: getattr para ENABLE_DASHBOARD
        logger.log(level, message, *args, **kwargs)
    else:
        # Só loga em arquivo
        file_handler = logging.getLogger().handlers[1] if len(logging.getLogger().handlers) > 1 else None
        if file_handler:
            record = logging.LogRecord(
                name=logger.name, level=level, pathname="", lineno=0,
                msg=message, args=args, exc_info=None
            )
            file_handler.emit(record)

# ===========================
# VARIÁVEIS GLOBAIS
# ===========================
shutdown_event = Event()
daily_trades_per_symbol = defaultdict(int)
signal_history = deque(maxlen=getattr(config, 'SIGNAL_HISTORY_MAXLEN', 100)) # ✅ CORREÇÃO: SIGNAL_HISTORY_MAXLEN
signal_history_lock = Lock()
mt5_lock = utils.mt5_lock
mt5_exec = utils.mt5_exec
GLOBAL_MONITOR_CACHE = {} # ✅ Land Trading Transparency v5.0
GLOBAL_ACTIVE_SYMBOLS = [] # ✅ LISTA ATIVA DE SESSÃO (FILTRADA)
RECENT_ORDERS_CACHE = {} # ✅ NOVO: Cache para evitar ordens duplicadas
ORDER_COOLDOWN_TRACKER = {} # ✅ NOVO: {symbol: timestamp} para cooldown após fechar posição
ATTEMPT_COOLDOWN_TRACKER = {} # ✅ NOVO: Limita tentativas de análise/execução por candle
GLOBAL_SESSION_BLACKLIST = set() # ✅ NOVO: Blacklist de sessão (dinâmica)
ENTRY_IDEMPOTENCY = {}
LAST_DAILY_EXPORT_DATE = None
LAST_FRIDAY_SNAPSHOT_DATE = None
LAST_FRIDAY_AUTOCLOSE_DATE = None

# ===========================
# DAILY DRAWDOWN TRACKER v5.0
# ===========================
DAILY_PNL_TRACKER = {
    "start_equity": 0.0,
    "current_pnl": 0.0,
    "last_reset": None,
    "is_circuit_breaker_active": False,
    "current_pnl": 0.0,
    "last_reset": None,
    "is_circuit_breaker_active": False,
    "daily_loss_pct": 0.0
}

# ===========================
# KILL SWITCH / PAUSE TRACKER v5.2
# ===========================
PAUSED_SYMBOLS = {} # {symbol: {"until": datetime, "reason": str}}
KILL_SWITCH_TRACKER = {} # {symbol: {"win_rate": float, "last_check": datetime}}


# ===========================
# SYMBOL FILTERING (v5.0.6)
# ===========================
def filter_and_validate_symbols() -> List[str]:
    """
    ✅ v5.0.6: Filtra a lista de símbolos, aplica aliases e remove cabeçalhos.
    """
    sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
    sector_map = getattr(config, "SECTOR_MAP", None)

    symbols_to_test = None
    if isinstance(sector_map, dict) and sector in sector_map:
        try:
            symbols_to_test = list(sector_map.get(sector) or [])
        except Exception:
            symbols_to_test = None

    if not symbols_to_test:
        symbols_to_test = getattr(config, 'ALL_AVAILABLE_SYMBOLS', [])
    if not symbols_to_test:
        symbols_to_test = list(getattr(config, 'SYMBOL_MAP', []))
    
    if not symbols_to_test:
        logger.error("❌ Nenhuma lista de símbolos encontrada no config_forex.py")
        return []

    logger.info(f"🔍 Iniciando auditoria de {len(symbols_to_test)} símbolos no config...")
    valid_symbols = []
    
    for sym in symbols_to_test:
        # Se for um cabeçalho (ex: FOREX_MAJORS), ele falhará no normalize_symbol
        # mas normalize_symbol retorna o próprio nome se não encontrar.
        # Precisamos testar se o resultado existe no MT5.
        
        sym = str(sym).strip()
        if not sym:
            continue
        real_symbol = utils.normalize_symbol(sym)
        
        exists = mt5_exec(mt5.symbol_select, real_symbol, True)
            
        if exists:
            if real_symbol != sym:
                logger.info(f"✅ Símbolo '{sym}' mapeado para '{real_symbol}'")
            valid_symbols.append(real_symbol)
        else:
            logger.info(f"🔹 Símbolo '{sym}' ignorado (Não existe no MT5 ou é um cabeçalho de categoria)")

    return list(set(valid_symbols)) # Remove duplicatas

# ===========================
# ELITE CONFIG LOADER (v4.2)
# ===========================
def load_elite_config() -> Dict[str, Dict]:
    """
    ✅ v4.2: Carrega parâmetros otimizados do arquivo elite_settings_YYYYMMDD.txt

    Retorna:
        Dict: {symbol: {ema_short, ema_long, rsi_low, rsi_high, adx_threshold,
                        sl_atr, tp_atr, bb_squeeze_threshold, min_score}}
    """
    import re

    output_dir = Path(getattr(config, 'OPTIMIZER_OUTPUT', 'optimizer_output')) # ✅ CORREÇÃO: getattr para OPTIMIZER_OUTPUT
    files = list(output_dir.glob("elite_settings_*.txt"))

    if not files:
        logger.warning("⚠️ Nenhum arquivo elite_settings encontrado - usando config padrão")
        return {}

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"📂 Carregando parâmetros otimizados de: {latest_file.name}")

    elite_config = {}

    try:
        import ast
        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.read()

            # Extrai o bloco ELITE_CONFIG = {...}
            match = re.search(r"ELITE_CONFIG = (\{.*?\})", content, re.DOTALL)
            if match:
                config_str = match.group(1)
                # ✅ v5.0.1: Usar ast.literal_eval para maior robustez com dicts Python
                elite_config = ast.literal_eval(config_str)
            else:
                logger.error("❌ Não foi possível extrair ELITE_CONFIG do arquivo.")


        logger.info(f"✅ Carregados parâmetros de {len(elite_config)} símbolos")

        # Log dos símbolos carregados
        for sym, params in elite_config.items():
            logger.debug(f"  {sym}: EMA({params.get('ema_short', '?')}/{params.get('ema_long', '?')}) "
                        f"RSI({params.get('rsi_low', '?')}/{params.get('rsi_high', '?')}) "
                        f"ADX>{params.get('adx_threshold', '?')}) "
                        f"MinScore:{params.get('min_score', '?')}") # ✅ CORREÇÃO: Adiciona min_score ao log

    except Exception as e:
        logger.error(f"❌ Erro ao carregar elite_config: {e}", exc_info=True) # ✅ CORREÇÃO: Adiciona exc_info
        elite_config = {} # Garante que elite_config seja um dict vazio em caso de erro
    
    try:
        if not elite_config:
            from config_forex import FOREX_PAIRS
            if isinstance(FOREX_PAIRS, dict) and FOREX_PAIRS:
                logger.warning("⚠️ Elite vazio. Usando fallback de FOREX_PAIRS do config.")
                elite_config = dict(FOREX_PAIRS)
                logger.info(f"✅ Fallback aplicado: {len(elite_config)} símbolos")
    except Exception:
        pass
    return elite_config

# Carrega na inicialização
ELITE_CONFIG = load_elite_config()

# ===========================
# DATACLASSES
# ===========================
@dataclass
class SignalAnalysis:
    """Registro de análise de sinal"""
    timestamp: datetime
    symbol: str
    signal: str
    strategy: str
    score: float
    rejected: bool
    rejection_reason: str
    indicators: dict = field(default_factory=dict)

@dataclass
class Position:
    ticket: int
    symbol: str
    side: str
    volume: float
    entry_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    pips: float
    open_time: datetime
    breakeven_moved: bool = False
    trailing_active: bool = False

@dataclass
class TradeMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0

    @property
    def win_rate(self) -> float:
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return abs(self.total_profit / self.total_loss) if self.total_loss != 0 else 0.0

# ===========================
# BOT STATE
# ===========================
class BotState:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BotState, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self._internal_lock = Lock()
        self._indicators: Dict[str, dict] = {}
        self._top_pairs: List[str] = []
        self._positions: Dict[int, Position] = {}
        self._trading_paused: bool = False
        self._pause_reason: str = ""
        self._last_daily_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._monitoring_status: Dict[str, dict] = {} # ✅ v5.0.2: {symbol: {status, reason, ml_score, timestamp}}

    def update_indicators(self, indicators: Dict[str, dict], top_pairs: List[str]):
        with self._internal_lock:
            self._indicators = indicators.copy()
            self._top_pairs = list(top_pairs)

    def update_monitoring(self, symbol: str, status: str, reason: str, ml_score: float):
        with self._internal_lock:
            # 1. Atualiza estado interno
            self._monitoring_status[symbol] = {
                "status": status,
                "reason": reason,
                "ml_score": ml_score,
                "timestamp": datetime.now()
            }
            
            # 2. ✅ CORREÇÃO: Sincroniza com o Cache Global do Painel/Log
            # Isso garante que o status visual mude instantaneamente
            global GLOBAL_MONITOR_CACHE
            if symbol not in GLOBAL_MONITOR_CACHE:
                GLOBAL_MONITOR_CACHE[symbol] = {}
            
            GLOBAL_MONITOR_CACHE[symbol].update({
                "status": status,
                "reason": reason,
                "ml_score": ml_score,
                "timestamp": datetime.now()
            })

    def get_monitoring_status(self) -> Dict[str, dict]:
        with self._internal_lock:
            return self._monitoring_status.copy()

    def get_indicators(self, symbol: str) -> dict:
        with self._internal_lock:
            return self._indicators.get(symbol, {}).copy()

    def get_top_pairs(self) -> List[str]:
        with self._internal_lock:
            return self._top_pairs.copy()

    def update_position(self, ticket: int, position: Position):
        with self._internal_lock:
            self._positions[ticket] = position

    def remove_position(self, ticket: int):
        with self._internal_lock:
            self._positions.pop(ticket, None)

    def get_positions(self) -> Dict[int, Position]:
        with self._internal_lock:
            return self._positions.copy()

    def pause_trading(self, reason: str):
        with self._internal_lock:
            self._trading_paused = True
            self._pause_reason = reason
            logger.warning(f"⏸️ Trading pausado: {reason}")

    def resume_trading(self):
        with self._internal_lock:
            self._trading_paused = False
            self._pause_reason = ""
            safe_log(logging.INFO, "▶️ Trading retomado")

    def is_paused(self) -> Tuple[bool, str]:
        with self._internal_lock:
            return self._trading_paused, self._pause_reason

    def check_and_reset_daily_limits(self):
        """Verifica e reseta contadores diários."""
        now = datetime.now()
        if now.day != self._last_daily_reset.day or now.month != self._last_daily_reset.month or now.year != self._last_daily_reset.year:
            with self._internal_lock:
                global daily_trades_per_symbol
                daily_trades_per_symbol.clear()
                self._last_daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
                safe_log(logging.INFO, "🔄 Limites diários de trades resetados.")

bot_state = BotState()
metrics = TradeMetrics()

# ===========================
# WATCHDOG
# ===========================
class ThreadWatchdog:
    """Monitora threads críticas e reinicia se travarem"""
    def __init__(self):
        self.threads = {}
        self.last_heartbeat = {}
        self.max_silence = 300 # Default
        self.lock = threading.Lock()
        self.custom_timeouts = {
            "SlowLoop": max(getattr(config, 'SLOW_LOOP_INTERVAL', 300) * 3, 450),
            "FastLoop": max(
                getattr(config, 'FAST_LOOP_INTERVAL', 15) * 5,
                getattr(config, 'FAST_LOOP_STALE_SECONDS', 120) + 60,
                getattr(config, 'FAST_LOOP_WATCHDOG_TIMEOUT', 0),
                300
            ),
            "Panel": max(getattr(config, 'PANEL_UPDATE_INTERVAL', 5) * 8, 40)
        }

    def register_thread(self, name: str, target_func, args=()):
        with self.lock:
            self.threads[name] = {
                "target": target_func,
                "args": args,
                "thread": None,
                "restarts": 0
            }
            self.last_heartbeat[name] = time.time()

    def heartbeat(self, name: str):
        with self.lock:
            self.last_heartbeat[name] = time.time()

    def check_and_restart(self):
        with self.lock:
            current_time = time.time()

            for name, info in self.threads.items():
                thread = info["thread"]
                last_beat = self.last_heartbeat.get(name, 0)
                silence_time = current_time - last_beat
                max_allowed = self.custom_timeouts.get(name, self.max_silence)

                is_dead = (thread is None or not thread.is_alive())
                is_silent = silence_time > max_allowed

                if is_silent and not is_dead:
                    safe_log(logging.WARNING, f"⏳ Thread {name} silenciosa há {int(silence_time)}s (timeout {int(max_allowed)}s).")
                    continue

                if is_dead:
                    info["restarts"] += 1
                    safe_log(logging.ERROR, f"💀 Thread {name} MORTA! Reiniciando... (#{info['restarts']})")

                    try:
                        new_thread = threading.Thread(
                            target=info["target"],
                            args=info["args"],
                            daemon=True,
                            name=name
                        )
                        new_thread.start()
                        info["thread"] = new_thread
                        self.last_heartbeat[name] = current_time
                        safe_log(logging.INFO, f"✅ Thread {name} reiniciada!")
                        # ✅ REQUISITO: Controle de Reinício (2s Delay)
                        time.sleep(2)
                    except Exception as e:
                        logger.critical(f"❌ FALHA ao reiniciar {name}: {e}")
                        if info["restarts"] >= getattr(config, 'MAX_THREAD_RESTARTS', 3): # ✅ CORREÇÃO: MAX_THREAD_RESTARTS
                            shutdown_event.set()

watchdog = ThreadWatchdog()

# ===========================
# SIGNAL ANALYSIS LOGGER
# ===========================
def log_signal_analysis(
    symbol: str,
    signal: str,
    strategy: str,
    score: float,
    rejected: bool,
    reason: str,
    indicators: dict,
    ml_score: float = 0.0,
    is_baseline: bool = False,
    session_name: Optional[str] = None
):
    """Registra análise de sinal"""
    with signal_history_lock:
        signal_history.append(SignalAnalysis(
            timestamp=datetime.now(),
            symbol=symbol,
            signal=signal or "NONE",
            strategy=strategy or "N/A",
            score=score,
            rejected=rejected,
            rejection_reason=reason,
            indicators={
                "rsi": indicators.get("rsi", 0),
                "adx": indicators.get("adx", 0),
                "spread_pips": indicators.get("spread_pips", 0),
                "volume_ratio": indicators.get("volume_ratio", 0),
                "ema_trend": indicators.get("ema_trend", "N/A"),
                "bb_width": indicators.get("bb_width", 0),
                "close": indicators.get("close", 0) # ✅ NOVO: Adiciona preço de fechamento
            }
        ))
    
    # ✅ Land Trading: Log em arquivo instantâneo
    try:
        user = os.getenv("XP3_OPERATOR", "XP3_BOT")
        context = {
            "session": (session_name or "UNKNOWN"),
            "action": ("ACEITE" if not rejected else "BLOQUEIO"),
            "ml_score": f"{ml_score:.1f}",
            "source": "FastLoop"
        }
        daily_logger.log_analysis(
            symbol=symbol,
            signal=signal or "NONE",
            strategy=strategy or "XP3_PRO_V4",
            score=score,
            rejected=rejected,
            reason=reason,
            indicators=indicators,
            ml_score=ml_score,
            is_baseline=is_baseline,
            user=user,
            context=context
        )
    except Exception as e:
        logger.error(f"⚠️ Erro ao registrar log diário: {e}")

    try:
        sess = session_name
        if not sess:
            sess = (utils.get_current_trading_session() or {}).get("name", "UNKNOWN")
        executed = not rejected
        utils.update_session_metrics(sess, executed=executed, rejected=rejected, reason=reason)
    except Exception:
        pass
    
    # ✅ Land Trading: Atualiza GLOBAL_MONITOR_CACHE apenas se não for monitoramento básico
    # (Evita sobrescrever dados mais completos do loop principal)
        GLOBAL_MONITOR_CACHE[symbol] = {
            "status": "🟢 SINAL" if not rejected else "🔵 MONITORANDO",
            "reason": reason,
            "ml_score": ml_score,
            "is_baseline": is_baseline,
            "timestamp": datetime.now()
        }

# ===========================
# PAUSE BOT HELPER v5.2
# ===========================
def pause_bot(symbol: str, minutes: int, reason: str):
    """Pausa o bot para um símbolo específico por X minutos."""
    until = datetime.now() + timedelta(minutes=minutes)
    PAUSED_SYMBOLS[symbol] = {"until": until, "reason": reason}
    logger.warning(f"⏸️ BOT PAUSED for {symbol} until {until.strftime('%H:%M')} | Reason: {reason}")
    bot_state.update_monitoring(symbol, "⏸️ PAUSADO", reason, 0.0)
    try:
        utils.send_telegram_message(f"🚨 Mudança de Regime de Mercado\n{symbol}: {reason}\n⏸️ Pausado por {minutes} min (até {until.strftime('%H:%M')}).")
    except Exception:
        pass
def _apply_pause_requests():
    try:
        import json, os, time
        path = os.path.join("data", "pause_requests.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        changed = False
        for sym, data in (payload or {}).items():
            try:
                until_ts = int(data.get("until", 0))
                minutes = int(data.get("minutes", 10))
                reason = str(data.get("reason", "EXECUTION_QUALITY"))
                if until_ts > int(time.time()):
                    pause_bot(sym, minutes, reason)
                    changed = True
            except Exception:
                pass
        if changed:
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception:
        pass
def perform_system_checks(current_session: dict, iteration_count: int) -> tuple:
    try:
        total_positions = mt5_exec(mt5.positions_total)
        try:
            total_positions = int(total_positions or 0)
        except Exception:
            total_positions = 0
        try:
            gm = getattr(config, 'MAX_GLOBAL_ALGO_ORDERS', 3)
            global_max = int(gm if gm is not None else 3)
        except Exception:
            global_max = 3
        if total_positions >= global_max:
            return False, {"sleep": 1, "reason": f"Limite Global de Ordens ({total_positions}/{global_max}) atingido. Entradas bloqueadas."}
        is_rollover, rollover_reason = utils.is_rollover_period()
        if is_rollover:
            return False, {"sleep": 10, "reason": rollover_reason}
        global DAILY_PNL_TRACKER
        account_info = mt5_exec(mt5.account_info)
        if account_info:
            current_equity = account_info.equity
            current_balance = account_info.balance
            today = datetime.now().date()
            if DAILY_PNL_TRACKER["last_reset"] != today:
                DAILY_PNL_TRACKER["start_equity"] = current_balance
                DAILY_PNL_TRACKER["last_reset"] = today
                DAILY_PNL_TRACKER["is_circuit_breaker_active"] = False
            if DAILY_PNL_TRACKER["start_equity"] > 0:
                daily_pnl = current_equity - DAILY_PNL_TRACKER["start_equity"]
                daily_loss_pct = abs(min(0, daily_pnl)) / DAILY_PNL_TRACKER["start_equity"]
                DAILY_PNL_TRACKER["current_pnl"] = daily_pnl
                DAILY_PNL_TRACKER["daily_loss_pct"] = daily_loss_pct
                max_dd = getattr(config, 'MAX_DAILY_DRAWDOWN_PCT', 0.02)
                if daily_loss_pct >= max_dd:
                    if not DAILY_PNL_TRACKER["is_circuit_breaker_active"]:
                        DAILY_PNL_TRACKER["is_circuit_breaker_active"] = True
                        logger.critical(f"CIRCUIT BREAKER ATIVO! Perda diária ({daily_loss_pct:.2%}) atingiu limite ({max_dd:.2%}). Bot pausado até amanhã.")
                    return False, {"sleep": 60, "reason": "Circuit breaker ativo"}
        if not GLOBAL_ACTIVE_SYMBOLS:
            return False, {"sleep": 5, "reason": "GLOBAL_ACTIVE_SYMBOLS vazia"}
        return True, {}
    except Exception as e:
        logger.error(f"perform_system_checks erro: {e}")
        return True, {}
def _compute_symbols_to_analyze(current_session: dict) -> list:
    symbols_to_analyze = GLOBAL_ACTIVE_SYMBOLS.copy()
    if current_session.get("name") == "ASIAN":
        priority_pairs = getattr(config, 'ASIAN_PRIORITY_PAIRS', [])
        symbols_to_analyze.sort(key=lambda s: s not in priority_pairs)
    return symbols_to_analyze
def manage_open_positions(active_tickets_map: dict) -> dict:
    try:
        current_positions = mt5_exec(mt5.positions_get)
        my_positions = [p for p in current_positions if p.magic == getattr(config, 'MAGIC_NUMBER', 123456)] if current_positions else []
        current_tickets_map = {p.ticket: p.symbol for p in my_positions}
        closed_tickets = set(active_tickets_map.keys()) - set(current_tickets_map.keys())
        for ticket in closed_tickets:
            symbol = active_tickets_map[ticket]
            ORDER_COOLDOWN_TRACKER[symbol] = time.time()
            try:
                from datetime import timedelta
                deals = mt5_exec(mt5.history_deals_get, datetime.now() - timedelta(days=2), datetime.now())
                magic = int(getattr(config, 'MAGIC_NUMBER', 123456))
                pos_deals = [d for d in (deals or []) if int(getattr(d, "position_id", 0) or 0) == int(ticket) and int(getattr(d, "magic", magic) or magic) == magic]
                if pos_deals:
                    total_profit = sum(float(getattr(d, "profit", 0.0) or 0.0) for d in pos_deals)
                    open_deals = [d for d in pos_deals if getattr(d, "entry", None) == mt5.DEAL_ENTRY_IN]
                    close_deals = [d for d in pos_deals if getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT]
                    side = "BUY" if (open_deals and getattr(open_deals[0], "type", None) == mt5.DEAL_TYPE_BUY) else "SELL"
                    open_price = float(getattr(open_deals[0], "price", 0.0) or 0.0) if open_deals else 0.0
                    close_price = float(getattr(close_deals[-1], "price", 0.0) or 0.0) if close_deals else 0.0
                    try:
                        if block_manager:
                            block_manager.on_trade_close(symbol, open_price, close_price, float(getattr(open_deals[0], "volume", 0.0) or 0.0) if open_deals else 0.0, total_profit, side)
                    except Exception:
                        pass
                    pip_size = utils.get_pip_size(symbol)
                    pips = 0.0
                    if pip_size > 0 and open_price > 0 and close_price > 0:
                        pips = (close_price - open_price) / pip_size if side == "BUY" else (open_price - close_price) / pip_size
                    msg = (
                        f"🏁 Saída de posição\n"
                        f"Par: {symbol}\n"
                        f"Ticket: {ticket}\n"
                        f"Lado: {side}\n"
                        f"PnL: ${total_profit:+.2f}\n"
                        f"Pips: {pips:+.1f}\n"
                        f"Entrada: {open_price:.5f}\n"
                        f"Saída: {close_price:.5f}\n"
                        f"Hora: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    utils.send_telegram_alert(msg, "SUCCESS" if total_profit >= 0 else "WARNING")
                    try:
                        if adaptive_manager:
                            strat = ""
                            try:
                                if open_deals and getattr(open_deals[0], "comment", None):
                                    c = str(getattr(open_deals[0], "comment", ""))
                                    if "XP3_" in c:
                                        strat = c.split("XP3_")[1].split()[0]
                            except Exception:
                                strat = ""
                            adaptive_manager.on_trade_close(symbol, strat, float(total_profit or 0.0))
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"❌ Erro ao enviar alerta de saída para {symbol}/{ticket}: {e}")
        return current_tickets_map
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar rastreio de posições: {e}")
        return active_tickets_map
def attempt_entry(symbol: str, signal: str, strategy: str, ind: dict, params: dict, current_session: dict, iteration_count: int, final_score: float, ml_score: float, ml_override_risk_mult: float, ml_override_used: bool, rejection_stats: dict) -> bool:
    try:
        base_sl = float(params.get("sl_atr", getattr(config, 'DEFAULT_STOP_LOSS_ATR_MULTIPLIER', 1.5)))
        base_tp = float(params.get("tp_atr", getattr(config, 'DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER', 3.0)))
        if adaptive_manager:
            sl_atr_mult, tp_atr_mult = adaptive_manager.get_current_params(symbol, strategy or "", base_sl, base_tp)
        else:
            sl_atr_mult, tp_atr_mult = base_sl, base_tp
        try:
            regime = utils.get_volatility_regime(symbol, ind.get("df") if isinstance(ind, dict) else None)
            if regime == "HIGH":
                tp_atr_mult = min(getattr(config, "MAX_TP_ATR_MULT", 6.0), tp_atr_mult + 0.5)
                sl_atr_mult = min(getattr(config, "MAX_SL_ATR_MULT", 3.5), sl_atr_mult + 0.3)
            elif regime == "LOW":
                tp_atr_mult = max(getattr(config, "MIN_TP_ATR_MULT", 2.0), tp_atr_mult - 0.5)
        except Exception:
            pass
        tick = mt5_exec(mt5.symbol_info_tick, symbol)
        if not tick:
            return False
        entry_price = tick.ask if signal == "BUY" else tick.bid
        volume = utils.calculate_position_size_atr_forex(
            symbol, entry_price, ind.get("atr_pips", 0), sl_atr_mult=sl_atr_mult, risk_multiplier=ml_override_risk_mult
        )
        if volume <= 0:
            return False
        sl, tp = utils.calculate_dynamic_levels(symbol, entry_price, ind, sl_atr_mult, tp_atr_mult, signal=signal)
        if sl <= 0 or tp <= 0:
            return False
        current_positions = mt5_exec(mt5.positions_get)
        exp_ok, exp_msg = utils.check_currency_exposure(current_positions)
        if not exp_ok:
            bot_state.update_monitoring(symbol, "🟠 EXPOSIÇÃO", exp_msg, ml_score)
            return False
        tot_ok, tot_msg = utils.check_total_exposure_limit(
            pending_symbol=symbol,
            pending_volume=volume,
            pending_side=signal
        )
        if not tot_ok:
            bot_state.update_monitoring(symbol, "🟠 EXPOSIÇÃO TOTAL", tot_msg, ml_score)
            return False
        hard_global_cap = int(getattr(config, 'MAX_GLOBAL_ALGO_ORDERS', 6))
        bot_positions_count = 0
        if current_positions:
            for p in current_positions:
                if p.magic == getattr(config, 'MAGIC_NUMBER', 123456):
                    bot_positions_count += 1
        if bot_positions_count >= hard_global_cap:
            reason = f"Limite global duro atingido ({bot_positions_count}/{hard_global_cap})"
            bot_state.update_monitoring(symbol, "🛑 LIMITE GLOBAL DURO", reason, ml_score)
            return False
        max_global_positions = int(getattr(config, 'MAX_SYMBOLS', hard_global_cap))
        try:
            session_name = (current_session or {}).get("name")
            session_map = getattr(config, "SESSION_MAX_POSITIONS", {})
            if isinstance(session_map, dict) and session_name in session_map:
                max_global_positions = int(session_map.get(session_name, max_global_positions))
        except Exception:
            pass
        if bot_positions_count >= max_global_positions:
            reason = f"Limite de posições simultâneas ({bot_positions_count}/{max_global_positions})"
            bot_state.update_monitoring(symbol, "🛑 LIMITE GLOBAL", reason, ml_score)
            return False
        candle_time = ind.get("time")
        candle_key = f"{symbol}|{signal}|{candle_time}"
        last_key_time = ENTRY_IDEMPOTENCY.get(candle_key)
        if last_key_time and (datetime.now() - last_key_time).total_seconds() < 3600:
            bot_state.update_monitoring(symbol, "⏳ COOLDOWN", "Idempotência (candle)", ml_score)
            return False
        ENTRY_IDEMPOTENCY[candle_key] = datetime.now()
        if len(ENTRY_IDEMPOTENCY) > 5000:
            cutoff = datetime.now() - timedelta(hours=6)
            ENTRY_IDEMPOTENCY_KEYS = list(ENTRY_IDEMPOTENCY.keys())
            for k in ENTRY_IDEMPOTENCY_KEYS:
                if ENTRY_IDEMPOTENCY.get(k) and ENTRY_IDEMPOTENCY[k] < cutoff:
                    del ENTRY_IDEMPOTENCY[k]
        order_params = OrderParams(
            symbol=symbol, side=OrderSide.BUY if signal == "BUY" else OrderSide.SELL,
            volume=volume, entry_price=entry_price, sl=sl, tp=tp,
            comment=f"XP3_{strategy}", magic=getattr(config, 'MAGIC_NUMBER', 123456)
        )
        RECENT_ORDERS_CACHE[symbol] = datetime.now()
        success = False
        try:
            success, message, ticket = validate_and_create_order_forex(order_params)
            if success:
                daily_trades_per_symbol[symbol] += 1
                side_str = "COMPRA 🟢" if signal == "BUY" else "VENDA 🔴"
                msg_tele = (
                    f"🚀 <b>XP3 PRO: Ordem Executada</b>\n\n"
                    f"🆔 Ativo: <b>{symbol}</b>\n"
                    f"📡 Sinal: {side_str}\n"
                    f"💵 Preço: {entry_price:.5f}\n"
                    f"🛑 SL: {sl:.5f} | 🎯 TP: {tp:.5f}\n"
                    f"📊 Score: {final_score:.1f} (ML: {ml_score:.1f}){' | 🟣 OVERRIDE' if ml_override_used else ''}\n"
                    f"🧪 Estratégia: {strategy}"
                )
                utils.send_telegram_alert(msg_tele)
                try:
                    utils.record_order_open(
                        symbol=symbol,
                        side=("BUY" if signal == "BUY" else "SELL"),
                        volume=volume,
                        entry_price=entry_price,
                        sl=sl,
                        tp=tp,
                        order_id=int(ticket or 0),
                        comment=f"XP3_{strategy}"
                    )
                except Exception:
                    pass
                exec_reason = "EXECUTADA (OVERRIDE)" if ml_override_used else "EXECUTADA"
                bot_state.update_monitoring(symbol, "🔴 OPERANDO", exec_reason, ml_score)
                log_signal_analysis(symbol, signal, strategy, final_score, False, exec_reason, ind, session_name=current_session.get("name"))
                return True
            else:
                if symbol in RECENT_ORDERS_CACHE:
                    del RECENT_ORDERS_CACHE[symbol]
                logger.error(f"❌ Falha para {symbol}: {message}")
                bot_state.update_monitoring(symbol, "⚠️ ERRO", message[:20], ml_score)
                time.sleep(10)
                return False
        except Exception as exc:
            if symbol in RECENT_ORDERS_CACHE:
                del RECENT_ORDERS_CACHE[symbol]
            logger.critical(f"💀 Erro na execução para {symbol}: {exc}", exc_info=True)
            time.sleep(30)
            return False
    except Exception as e:
        logger.error(f"attempt_entry erro: {e}")
        return False

# ===========================
# CHECK FOR SIGNALS (v4.2)
# ===========================
def check_for_signals(symbol: str, current_session: dict = None) -> Tuple[Optional[str], dict, Optional[str], str]:
    """
    ✅ v5.0.2: Estratégia Híbrida com Negative Edge Reporting.
    ✅ Horário de Ouro: Regras dinâmicas aplicadas.
    ✅ v6.0: Adaptive Engine Integration - Sistema Adaptativo 4 Camadas
    Retorna: (signal, indicators, strategy, reason)
    """

    # ✅ ADAPTIVE ENGINE: Processa dados de mercado em tempo real
    if adaptive_engine and getattr(config, 'ENABLE_ADAPTIVE_ENGINE', True):
        try:
            # Coleta dados de mercado para o símbolo
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'session': current_session.get('name', 'NORMAL') if current_session else 'NORMAL',
                'price_data': utils.get_price_data(symbol),  # Função auxiliar para obter dados de preço
                'volatility': utils.get_volatility(symbol),   # Função auxiliar para obter volatilidade
                'volume': utils.get_volume_data(symbol)       # Função auxiliar para obter volume
            }
            
            # Processa através do Adaptive Engine
            adaptive_result = adaptive_engine.process_market_data(market_data)
            
            # Verifica Panic Mode
            if adaptive_result.get('panic_mode_active'):
                logger.warning(f"🚨 PANIC MODE ATIVO para {symbol}: {adaptive_result.get('panic_reason')}")
                return None, None, None, "Panic Mode Ativo - Operações Suspensas"
            
            # Aplica ajustes de parâmetros sugeridos pelo sistema adaptativo
            if adaptive_result.get('parameter_adjustments'):
                logger.info(f"🧠 Adaptive Engine ajustando parâmetros para {symbol}")
                # Os ajustes serão aplicados nos parâmetros abaixo
                
        except Exception as e:
            logger.error(f"❌ Erro no Adaptive Engine para {symbol}: {e}")
            # Continua com parâmetros padrão em caso de erro

    # 1. Busca parâmetros otimizados
    params = ELITE_CONFIG.get(symbol, {})

    ema_short = params.get("ema_short", getattr(config, 'EMA_SHORT_PERIOD', 20))
    ema_long = params.get("ema_long", getattr(config, 'EMA_LONG_PERIOD', 50))
    rsi_period = params.get("rsi_period", getattr(config, 'RSI_PERIOD', 14))
    adx_period = params.get("adx_period", getattr(config, 'ADX_PERIOD', 14))
    # ADX threshold por símbolo (tendência baixa pode operar com corte menor)
    base_adx_threshold = params.get("adx_threshold", getattr(config, 'ADX_THRESHOLD', 25))
    try:
        low_trend_map = getattr(config, 'LOW_TREND_ADX_THRESHOLDS', {})
        adx_threshold = low_trend_map.get(symbol, base_adx_threshold)
    except Exception:
        adx_threshold = base_adx_threshold
    
    # --- AJUSTE DE SENSIBILIDADE RSI (SESSÃO ASIÁTICA) ---
    session_name = current_session.get("name", "NORMAL") if current_session else "NORMAL"
    rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
    rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
    bb_period = params.get("bb_period", getattr(config, 'BB_PERIOD', 20))
    bb_dev = params.get("bb_dev", getattr(config, 'BB_DEVIATION', 2.0))
    bb_squeeze_threshold = params.get("bb_squeeze_threshold", getattr(config, 'BB_SQUEEZE_THRESHOLD', 0.015))

    # 2. Obtém indicadores
    ind = utils.get_indicators_forex(
        symbol,
        ema_short=ema_short,
        ema_long=ema_long,
        rsi_period=rsi_period,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        adx_period=adx_period,
        bb_period=bb_period,
        bb_dev=bb_dev
    )

    # ✅ REQUISITO: Verificação nula + erro em dicionário
    if ind is None or (isinstance(ind, dict) and ind.get("error")):
        error_msg = ind.get("message", "Erro desconhecido") if isinstance(ind, dict) else "Retorno nulo de indicadores"
        return "NONE", None, "NONE", f"Erro nos indicadores para {symbol}: {error_msg}"

    # ===========================
    # ✅ v5.0: EMA 200 MACRO TREND FILTER
    # ===========================
    if getattr(config, 'ENABLE_EMA_200_FILTER', True):
        ema_200_data = utils.get_ema_200(symbol)
        if not ema_200_data.get("error"):
            ind["ema_200"] = ema_200_data.get("ema_200")
            ind["ema_200_trend"] = ema_200_data.get("trend_direction")
            ind["is_above_ema_200"] = ema_200_data.get("is_above_ema")
        else:
            # Se erro, não bloqueia mas loga
            logger.warning(f"⚠️ EMA 200 indisponível para {symbol}: {ema_200_data.get('message')}")
            ind["ema_200"] = None
            ind["ema_200_trend"] = "UNKNOWN"
            ind["is_above_ema_200"] = None

    # ===========================
    # ✅ v5.0: ADX MINIMUM STRENGTH FILTER (Ajustado para Land Trading)
    # ===========================
    adx_now = ind.get("adx", 0)
    # Apenas loga aviso, não retorna None aqui para permitir Reversão
    adx_min_strength = getattr(config, 'ADX_MIN_STRENGTH', 20)
    is_low_adx = adx_now < adx_min_strength
    ind["adx_low"] = is_low_adx

    # 3. Filtros Básicos (Spread e Volume)
    spread_points = ind.get("spread_points", ind.get("spread_pips", 0))
    
    # --- REGRAS DINÂMICAS POR SESSÃO ---
    session_name = current_session.get("name", "NORMAL") if current_session else "NORMAL"
    
    symbol_upper = symbol.upper()
    crypto_list = getattr(config, 'TOKENS_CRYPTO', ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"])
    indices_list = getattr(config, 'TOKENS_INDICES', ["US30", "NAS100", "USTEC", "DE40", "GER40", "GER30", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"])
    metals_list = getattr(config, 'TOKENS_METALS', ["XAU", "XAG", "GOLD", "SILVER"])
    exotics_list = getattr(config, 'TOKENS_EXOTICS', ["TRY", "ZAR", "MXN", "RUB", "CNH", "PLN", "HUF", "CZK", "DKK", "NOK", "SEK"])

    is_crypto = any(c in symbol_upper for c in crypto_list)
    is_indices = any(i in symbol_upper for i in indices_list)
    is_metals = any(m in symbol_upper for m in metals_list)
    is_exotic = any(x in symbol_upper for x in exotics_list)

    if is_crypto:
        max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
        spread_check = spread_points > max_spread
    elif is_indices:
        max_spread = getattr(config, 'MAX_SPREAD_INDICES', 600)
        spread_check = spread_points > max_spread
    elif is_metals:
        max_spread = getattr(config, 'MAX_SPREAD_METALS', 80)
        spread_check = spread_points > max_spread
    elif is_exotic:
        max_spread = getattr(config, 'MAX_SPREAD_EXOTICS', 8000)
        spread_check = spread_points > max_spread
    else:
        max_spread = getattr(config, 'MAX_SPREAD_ACCEPTABLE', 25)
        spread_check = spread_points > max_spread

    if not (is_crypto or is_indices or is_metals or is_exotic):
        if session_name == "GOLDEN":
            max_spread *= (1 + getattr(config, 'GOLDEN_SPREAD_ALLOWANCE_PCT', 0.20))
        elif session_name == "PROTECTION":
            max_spread = getattr(config, 'PROTECTION_MAX_SPREAD_FOREX', 20)
        elif session_name == "ASIAN":
            max_spread *= 1.3
        spread_check = spread_points > max_spread

    if spread_check:
        return None, ind, None, f"Spread Alto ({spread_points} > {max_spread})"

    # Ajuste de Volume
    vol_ratio = ind.get("volume_ratio", 0)
    min_vol_ratio = getattr(config, 'MIN_VOLUME_COEFFICIENT', 0.4)
    if session_name == "GOLDEN":
        min_vol_ratio *= (1 - getattr(config, 'GOLDEN_VOLUME_REDUCTION_PCT', 0.30))
    elif session_name == "ASIAN":
        min_vol_ratio = 0.5 # Fixo em 0.5x da média
    
    vol_ok = vol_ratio >= min_vol_ratio
    # Não bloqueamos imediatamente para permitir bypass via ML (v6.1)

    # 4. Extrai indicadores para estratégia
    rsi_now = ind.get("rsi", 50)
    adx_now = ind.get("adx", 0)
    close_price = ind.get("close", 0)
    bb_upper = ind.get("bb_upper")
    bb_lower = ind.get("bb_lower")
    bb_width = ind.get("bb_width", 0)
    ema_trend = ind.get("ema_trend")

    signal = None
    strategy = None
    reason = "Monitorando - Sem sinal claro"

    # 5. Bollinger Squeeze Check (Marcar mas não vetar se ML for usado)
    has_squeeze = bb_width < bb_squeeze_threshold

    # 6. Estratégia baseada no ADX
    if adx_now >= adx_threshold:
        if ema_trend in ("UP", "NEUTRAL"):
            signal = "BUY"
            strategy = "TREND"
            rsi_ok = True
        elif ema_trend == "DOWN":
            signal = "SELL"
            strategy = "TREND"
            rsi_ok = True
        else:
            reason = "Trend Indefinida (EMA Flat)"
            rsi_ok = True # Ignora se não há tendência
    
    else:
        adx_min_rev = getattr(config, 'ADX_MIN_FOR_REVERSION', 10)
        if adx_now < adx_min_rev:
            return None, ind, strategy, f"ADX crítico ({adx_now:.1f})"
        if close_price <= bb_lower:
            if rsi_now < 30:
                signal = "BUY"
                strategy = "REVERSION"
            else:
                reason = f"RSI em Neutro ({rsi_now:.1f})" # ✅ Land Trading Standard
        elif close_price >= bb_upper:
            if rsi_now > 70:
                signal = "SELL"
                strategy = "REVERSION"
            else:
                reason = f"RSI em Neutro ({rsi_now:.1f})" # ✅ Land Trading Standard
        else:
            reason = f"ADX baixo ({adx_now:.1f}) e preço dentro das bandas"

    # 8. Confirmação "Sniper" v5.0 (Price Action)
    if signal:
        # ===========================
        # ✅ v5.0: EMA 200 GOLDEN RULE
        # ===========================
        # BUY só válido se Preço > EMA 200 (tendência de alta)
        # SELL só válido se Preço < EMA 200 (tendência de baixa)
        if getattr(config, 'ENABLE_EMA_200_FILTER', True) and ind.get("ema_200") is not None:
            ema_200_val = ind.get("ema_200")
            price = ind.get("price") or ind.get("close")
            atr = ind.get("atr")

            # Fallback de segurança
            if price is None or atr is None:
                logger.warning(f"{symbol}: EMA200 ativo mas price/ATR indisponível")
            else:
                distance = abs(price - ema_200_val)
                
                # Hybrid Rule EMA 200 (✅ v5.3)
                is_against = (signal == "BUY" and price < ema_200_val) or (signal == "SELL" and price > ema_200_val)
                
                if is_against:
                    ind["ema_against"] = True
                    dist_atr = distance / atr if atr > 0 else 999.99
                    if dist_atr > 3.0:
                        ind["ema_penalty"] = max(ind.get("ema_penalty", 0), 15)
                    elif dist_atr > 1.5:
                        ind["ema_penalty"] = max(ind.get("ema_penalty", 0), 8)

        # ===========================
        # ✅ v5.3: MULTI-TIMEFRAME SCORE ADJUSTMENT (não mais veto)
        # ===========================
        # Correção 3: Se EMA200 já validou tendência, ignora H4 (evita redundância)
        macro_trend_ok = (
            signal and 
            getattr(config, 'ENABLE_EMA_200_FILTER', True) and 
            ind.get("ema_200") is not None and
            not ind.get("ema_penalty") # Se tem penalidade, não está "ok" o suficiente para ignorar H4
        )
        
        if getattr(config, 'ENABLE_MULTI_TIMEFRAME', True) and not macro_trend_ok:
            mtf_penalty, mtf_reason, mtf_trend = utils.get_multi_timeframe_trend(symbol, signal)
            if mtf_penalty < 0:
                ind["mtf_penalty"] = abs(mtf_penalty)
                logger.info(f"📉 {symbol}: {mtf_reason}")
        elif macro_trend_ok and getattr(config, 'ENABLE_MULTI_TIMEFRAME', True):
            logger.debug(f"🔄 {symbol}: H4 check skipped (EMA200 já validou macro trend)")
        
        close_now = ind.get("close", 0)
        open_now = ind.get("open", 0)
        candle_ok = (signal == "BUY" and close_now > open_now) or (signal == "SELL" and close_now < open_now)
        
        if not candle_ok:
            ind["penalty_candle"] = max(ind.get("penalty_candle", 0), 10)
            reason = f"Candle não confirmado (penalidade)"
        
        # Verificação de Filtros
        if not vol_ok:
            reason = f"Veto Volume: {reason} | Vol {vol_ratio:.2f}"
            signal = "BUY_VETO" if signal == "BUY" else "SELL_VETO"
        else:
            reason = f"Sinal {signal} confirmado por {strategy}"
            if has_squeeze:
                reason += " (Squeeze Detectado)"

    return signal, ind, strategy, reason

    return signal, ind, strategy
# ===========================
# EMA 200 MACRO FILTER (WITH PULLBACK TOLERANCE)
# ===========================

def is_macro_trend_allowed(
    signal: str,
    price: float,
    ema_200: float,
    atr: float
) -> bool:
    """
    Valida tendência macro com tolerância de pullback baseada em ATR.
    """

    # BUY em macro bullish
    if signal == "BUY":
        if price >= ema_200:
            return True

        if EMA_200_ALLOW_PULLBACK:
            distance = abs(price - ema_200)
            if distance <= atr * EMA_200_PULLBACK_ATR_TOLERANCE:
                return True

        return False

    # SELL em macro bearish
    if signal == "SELL":
        if price <= ema_200:
            return True

        if EMA_200_ALLOW_PULLBACK:
            distance = abs(price - ema_200)
            if distance <= atr * EMA_200_PULLBACK_ATR_TOLERANCE:
                return True

        return False

    return False

# Variáveis Globais de Controle de Thread
FAST_LOOP_LOCK = threading.Lock()
FAST_LOOP_ACTIVE = False # ✅ Impede múltiplas instâncias da thread rápida
FAST_LOOP_LAST_BEAT = 0.0
FAST_LOOP_DUP_REJECT_TS = 0.0

# ===========================
# FAST LOOP
# ===========================
def fast_loop():
    """Loop rápido de análise e execução"""
    global FAST_LOOP_ACTIVE, FAST_LOOP_LAST_BEAT, FAST_LOOP_DUP_REJECT_TS
    
    # 🛡️ SINGLE INSTANCE CHECK
    FAST_LOOP_LAST_BEAT = time.time()

    # try:  <-- REMOVIDO para evitar erro de indentação massiva
    safe_log(logging.INFO, "🚀 Fast Loop iniciado (Single Instance)")

    last_cache_update = 0
    iteration_count = 0
    rejection_stats = {"ML": 0, "TECHNICAL": 0, "NEWS": 0, "RISK": 0}
    last_status_report = {} # {symbol: timestamp} para throttling de negative edge logs
    last_heartbeat_update = 0 
    active_tickets_map = {} # ✅ NOVO: Rastreio de tickets para Cooldown {ticket: symbol}
    last_session_update = 0
    current_session = {"name": "NORMAL", "display": "NORMAL", "emoji": "⚖️", "color": "blue"}
    
    # Inicia cache de log de ML (anti-spam)
    ml_log_cache = {}
    analysis_log_state = {}
    last_weekend_action_ts = 0.0

    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            FAST_LOOP_LAST_BEAT = current_time
            watchdog.heartbeat("FastLoop")
            _apply_pause_requests()
            if current_time - last_heartbeat_update > 30:  # Atualiza a cada 30s
                update_heartbeat()
                last_heartbeat_update = current_time

            # ✅ SESSION DETECTION (Brasil Time)
            if current_time - last_session_update > 60: # Atualiza a cada minuto
                current_session = utils.get_current_trading_session()
                last_session_update = current_time
                safe_log(logging.INFO, f"🌐 SESSÃO ATUAL: {current_session['display']} {current_session['emoji']}")

            ok, info = perform_system_checks(current_session, iteration_count)
            if not ok:
                if iteration_count % 60 == 0:
                    logger.warning(info.get("reason", ""))
                time.sleep(int(info.get("sleep", 1)))
                continue

            # --- PRIORALIZAÇÃO DE PARES (SESSÃO ASIÁTICA) ---
            symbols_to_analyze = _compute_symbols_to_analyze(current_session)

            iteration_count += 1
            watchdog.heartbeat("FastLoop")
            current_time = time.time()
            logger.info(f"🔎 FastLoop ativo | it={iteration_count} | símbolos={len(symbols_to_analyze)}")

            # ========================================
            # ✅ ATUALIZA RASTREIO DE POSIÇÕES (COOLDOWN)
            # ========================================
            active_tickets_map = manage_open_positions(active_tickets_map)


            if iteration_count % 60 == 0:
                logger.info(f"🔄 Fast Loop - Iteração #{iteration_count}")
                logger.info(
                    "📊 Rejeições (últimas iterações) | ML=%d | Técnico=%d | News=%d | Risco=%d",
                    rejection_stats["ML"],
                    rejection_stats["TECHNICAL"],
                    rejection_stats["NEWS"],
                    rejection_stats["RISK"],
                )

            # ========================================
            # ATUALIZA CACHE DE INDICADORES E TOP PAIRS
            # ========================================
            # ✅ CORREÇÃO: Usa config.SLOW_LOOP_INTERVAL para cache update
            if current_time - last_cache_update > getattr(config, 'SLOW_LOOP_INTERVAL', 300):
                last_cache_update = current_time
                panel_cache = {}
                try:
                    if profit_optimizer:
                        profit_optimizer.scan_and_optimize()
                except Exception:
                    pass

                # ✅ v5.0.6: Usa a lista global filtrada
                symbols_to_process = GLOBAL_ACTIVE_SYMBOLS[:getattr(config, 'MAX_SYMBOLS_CACHE', 15)]

                for symbol in symbols_to_process:
                    try:
                        # ✅ DASHBOARD FEED: Garante que o painel mostre "Analisando" imediatamente
                        GLOBAL_MONITOR_CACHE[symbol] = {
                            "status": "🔍 ANALISANDO",
                            "reason": "Buscando oportunidades...",
                            #"ml_score": 50.0,
                            "is_baseline": True,
                            "timestamp": datetime.now()
                        }

                        # Remove da blacklist se já passou o tempo
                        if symbol in GLOBAL_SESSION_BLACKLIST:
                            # Se o símbolo está na blacklist da sessão, verifica se já passou o tempo de "castigo"
                            # Por exemplo, 1 hora. Se sim, remove da blacklist.
                            # Isso é um placeholder, a lógica real de remoção pode ser mais complexa.
                            # Por enquanto, apenas ignora se estiver na blacklist.
                            continue

                        # ✅ REQUISITO: Normalização de Símbolo Sugerida
                        real_symbol = utils.normalize_symbol(symbol)
                        if real_symbol != symbol:
                            logger.info(f"🔄 Símbolo {symbol} normalizado para {real_symbol}")
                        
                        # ✅ CORREÇÃO: Passa parâmetros otimizados para get_indicators_forex
                        params = ELITE_CONFIG.get(symbol, {})
                        ind = utils.get_indicators_forex(
                            symbol,
                            ema_short=params.get("ema_short"),
                            ema_long=params.get("ema_long"),
                            rsi_period=params.get("rsi_period"),
                            rsi_low=params.get("rsi_low"),
                            rsi_high=params.get("rsi_high"),
                            adx_period=params.get("adx_period"),
                            bb_period=params.get("bb_period"),
                            bb_dev=params.get("bb_dev")
                        )
                        
                        # ✅ REQUISITO: Verificação robusta para evitar morte de thread
                        if ind is None or (isinstance(ind, dict) and ind.get("error")):
                            continue

                        # Limpa o DF para economizar memória no cache
                        if 'df' in ind:
                            del ind['df']

                        panel_cache[symbol] = ind
                    except Exception as e:
                        logger.error(f"❌ Erro ao atualizar cache de indicadores para {symbol}: {e}", exc_info=True)
                        continue

                if panel_cache:
                    # ✅ CORREÇÃO: calculate_signal_score precisa de todos os parâmetros otimizados
                    # Para simplificar, vamos usar uma versão básica do score aqui para o top_pairs
                    # Uma implementação mais robusta passaria os params otimizados para calculate_signal_score
                    top_pairs = sorted(
                        panel_cache.keys(),
                        key=lambda s: utils.calculate_signal_score(panel_cache[s])[0], # Score básico
                        reverse=True
                    )
                    bot_state.update_indicators(panel_cache, top_pairs)

            # ========================================
            # VERIFICAÇÕES DE ESTADO DO BOT E MERCADO
            # ========================================
            if not utils.is_market_open():
                if iteration_count % 60 == 0:
                    market = utils.get_market_status()
                    logger.warning(f"💤 Mercado fechado: {market['message']}")
                time.sleep(60)
                continue

            ks_path = Path(getattr(config, 'KILLSWITCH_FILE', 'killswitch.txt'))
            if ks_path.exists():
                try:
                    ttl_seconds = int(getattr(config, 'KILLSWITCH_TTL_SECONDS', 900))
                    mtime = datetime.fromtimestamp(ks_path.stat().st_mtime)
                    age = (datetime.now() - mtime).total_seconds()
                    if age <= ttl_seconds:
                        logger.critical("🚨 KILL SWITCH ATIVADO! Fechando todas as posições e encerrando.")
                        utils.close_all_positions()
                        try:
                            ks_path.unlink()
                        except Exception:
                            pass
                        shutdown_event.set()
                        break
                    else:
                        try:
                            ks_path.unlink()
                        except Exception:
                            pass
                        logger.warning("⚠️ Ignorando KILL SWITCH antigo. Arquivo removido.")
                except Exception as e:
                    logger.error(f"❌ Erro ao processar KILL SWITCH: {e}")

            is_paused, pause_reason = bot_state.is_paused()
            if is_paused:
                if iteration_count % 60 == 0:
                    logger.warning(f"⏸️ Bot pausado: {pause_reason}")
                time.sleep(10)
                continue

            # ✅ NOVO: Reset diário de limites
            bot_state.check_and_reset_daily_limits()

            # ========================================
            # ANÁLISE DE SÍMBOLOS E EXECUÇÃO DE ORDENS
            # ========================================
            # ✅ v5.0.6: Usa a lista global filtrada para análise
            symbols_to_analyze = GLOBAL_ACTIVE_SYMBOLS

            weekend_state = utils.get_weekend_protection_state() if hasattr(utils, "get_weekend_protection_state") else {"block_entries": False, "force_close": False, "reason": ""}
            allow_entries = not weekend_state.get("block_entries", False)
            if weekend_state.get("force_close", False):
                now_ts = time.time()
                if (now_ts - last_weekend_action_ts) > 60:
                    last_weekend_action_ts = now_ts
                    utils.send_telegram_alert(weekend_state.get("reason", "Fim de pregão"), "WARNING")
                    utils.close_all_positions()
            
            # Conjunto de símbolos para excluir se não existirem no MT5
            static_blacklist = getattr(config, 'BLACKLISTED_SYMBOLS', set())
            
            # ✅ REQUISITO: Lista Negra Dinâmica de Sessão
            # GLOBAL_SESSION_BLACKLIST já inicializado globalmente

            # GLOBAL_SESSION_BLACKLIST já inicializado globalmente

            for symbol in symbols_to_analyze:
                try:
                    watchdog.heartbeat("FastLoop")
                    now_ts_local = time.time()
                    if (now_ts_local - last_heartbeat_update) > 20:
                        update_heartbeat()
                        last_heartbeat_update = now_ts_local
                except Exception:
                    pass
                symbol_start_ts = time.time()
                if symbol in static_blacklist or symbol in GLOBAL_SESSION_BLACKLIST:
                    continue

                if not allow_entries:
                    reason_weekend = weekend_state.get("reason", "Entradas bloqueadas")
                    bot_state.update_monitoring(symbol, "🏁 FIM DE SEMANA", reason_weekend, 0.0)
                    now_ts = time.time()
                    throttle_s = getattr(config, 'ANALYSIS_LOG_THROTTLE_SECONDS', 120)
                    prev = analysis_log_state.get(symbol)
                    prev_reason = prev.get("reason") if prev else None
                    prev_ts = prev.get("ts") if prev else 0
                    if (prev_reason != reason_weekend) or ((now_ts - prev_ts) > throttle_s):
                        analysis_log_state[symbol] = {"reason": reason_weekend, "ts": now_ts}
                        log_signal_analysis(symbol, "NONE", "WEEKEND", 0, True, reason_weekend, {})
                    continue
                
                # ✅ v6.1: Proteção contra ordens duplicadas (Cache Recente)
                if symbol in RECENT_ORDERS_CACHE:
                    last_order_time = RECENT_ORDERS_CACHE[symbol]
                    if (datetime.now() - last_order_time).total_seconds() < 15: # 15s de segurança
                        if iteration_count % 5 == 0:
                            logger.debug(f"🛡️ {symbol}: Ignorando análise (Ordem enviada há <15s)")
                        bot_state.update_monitoring(symbol, "⏳ WAITING", "Post-Order Safety", 0.0)
                        continue
                    else:
                        del RECENT_ORDERS_CACHE[symbol] # Limpa cache antigo

                # ===========================
                # ✅ v5.2: CHECK PAUSED SYMBOLS (Kill Switch)
                # ===========================
                if symbol in PAUSED_SYMBOLS:
                    pause_data = PAUSED_SYMBOLS[symbol]
                    if datetime.now() < pause_data["until"]:
                        # Ainda está pausado
                        if iteration_count % 60 == 0:
                            logger.info(f"⏸️ {symbol} pausado até {pause_data['until'].strftime('%H:%M')} ({pause_data['reason']})")
                        bot_state.update_monitoring(symbol, "⏸️ PAUSADO", pause_data['reason'], 0.0)
                        continue
                    else:
                        # Tempo expirou, remove da pausa
                        del PAUSED_SYMBOLS[symbol]
                        logger.info(f"▶️ {symbol}: Pausa expirada. Retomando operações.")
                
                if block_manager:
                    is_blocked, reason = block_manager.is_blocked(symbol)
                    if is_blocked:
                        bot_state.update_monitoring(symbol, "⛔ BLOQUEIO", reason, 0.0)
                        now_ts = time.time()
                        throttle_s = getattr(config, 'ANALYSIS_LOG_THROTTLE_SECONDS', 120)
                        prev = analysis_log_state.get(symbol)
                        prev_reason = prev.get("reason") if prev else None
                        prev_ts = prev.get("ts") if prev else 0
                        if (prev_reason != reason) or ((now_ts - prev_ts) > throttle_s):
                            analysis_log_state[symbol] = {"reason": reason, "ts": now_ts}
                            log_signal_analysis(symbol, "NONE", "RISK_BLOCK", 0, True, reason, {})
                        continue

                try:
                    # ✅ REQUISITO: Validação Instantânea de Existência
                    if not mt5_exec(mt5.symbol_select, symbol, True):
                            logger.error(f"❌ Símbolo {symbol} removido por inexistência no MT5 nesta sessão.")
                            GLOBAL_SESSION_BLACKLIST.add(symbol)
                            continue
                    
                    # ✅ FIX: Inicializa params no início do loop para evitar UnboundLocalError
                    params = ELITE_CONFIG.get(symbol, {})

                    # =====================================================
                    # ✅ UNIFIED SAFETY LOCKS (CRITICAL FIX)
                    # =====================================================
                    # 1. Verifica limite de ordens por símbolo (PRIMEIRA CHECAGEM)
                    # Garante que nada aconteça se já houver posição aberta
                    existing_positions = mt5_exec(mt5.positions_get, symbol=symbol)
                    max_orders = getattr(config, 'MAX_ORDERS_PER_SYMBOL', 1)
                    
                    if existing_positions and len(existing_positions) >= max_orders:
                        # ✅ SNIPER MODE: Log Limpo
                        if iteration_count % 60 == 0: 
                            logger.info(f"🔭 {symbol}: Sniper Mode - Monitorando posição existente (Ticket {existing_positions[0].ticket}).")
                        bot_state.update_monitoring(symbol, "🔴 OPERANDO", f"Gestão Ativa ({len(existing_positions)}/{max_orders})", 50.0) # Score neutro
                        continue # ⛔ SAI IMEDIATAMENTE

                    # 2. Verifica cooldown após fechamento de ordem
                    cooldown_seconds = getattr(config, 'ORDER_COOLDOWN_SECONDS', 300)
                    last_close_time = ORDER_COOLDOWN_TRACKER.get(symbol, 0)
                    time_since_close = time.time() - last_close_time
                    if last_close_time > 0 and time_since_close < cooldown_seconds:
                        remaining = int(cooldown_seconds - time_since_close)
                        if iteration_count % 30 == 0: # Throttling log
                           logger.info(f"⏳ {symbol}: Cooldown ativo ({remaining}s restantes)")
                        bot_state.update_monitoring(symbol, "⏳ COOLDOWN", f"Aguardando {remaining}s", 0.0)
                        continue # ⛔ SAI IMEDIATAMENTE

                     # 3. Verificação de Spread em Tempo Real (Re-fetch)
                    current_spread = 0 # ✅ INIT Seguro
                    symbol_info_realtime = mt5_exec(mt5.symbol_info, symbol)
                    if symbol_info_realtime:
                        current_spread = symbol_info_realtime.spread
                        symbol_upper = str(symbol).upper().strip()

                        crypto_list = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]
                        indices_list = ["US30", "NAS100", "USTEC", "DE40", "GER40", "GER30", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"]
                        metals_list = ["XAU", "XAG", "GOLD", "SILVER"]

                        is_crypto = any(c in symbol_upper for c in crypto_list)
                        is_indices = any(i in symbol_upper for i in indices_list)
                        is_metals = any(m in symbol_upper for m in metals_list)

                        if is_crypto:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
                        elif is_indices:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_INDICES', 800)
                        elif is_metals:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_METALS', 80)
                        else:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_FOREX', 25)
                        # Exóticos (TRY/ZAR/MXN etc.)
                        exotics_list = getattr(config, 'TOKENS_EXOTICS', ["TRY","ZAR","MXN","RUB","CNH","PLN","HUF","CZK","DKK","NOK","SEK"])
                        is_exotics = any(x in symbol_upper for x in exotics_list)
                        if is_exotics:
                            max_spread_limit = getattr(config, 'MAX_SPREAD_EXOTICS', 10000)

                        sess_name = (current_session or {}).get("name")
                        if not (is_crypto or is_indices or is_metals):
                            if sess_name == "ASIAN":
                                max_spread_limit *= 1.3
                            elif sess_name == "GOLDEN":
                                max_spread_limit *= 1.2

                        if current_spread > max_spread_limit:
                            if iteration_count % 30 == 0:
                                logger.warning(f"🚫 {symbol}: Spread Alto ({current_spread} > {max_spread_limit}).")
                            bot_state.update_monitoring(symbol, "⚠️ BLOQUEADO", f"Spread Alto ({current_spread} > {max_spread_limit})", 0.0)
                            continue # ⛔ SAI IMEDIATAMENTE
                    # =====================================================

                    # =====================================================
                    # ✅ v5.3: INSTITUTIONAL RISK - KILL SWITCH & DRAWDOWN
                    # =====================================================
                    risk_ok, risk_msg = utils.check_institutional_risk()
                    if not risk_ok:
                        if iteration_count % 60 == 0:
                            logger.critical(f"🛑 BLOQUEIO INSTITUCIONAL: {risk_msg}")
                        bot_state.pause_trading(risk_msg)
                        time.sleep(300) # Pausa longa para preservação de capital
                        continue
                    # =====================================================

                    # =====================================================
                    # 4. Attempt Cooldown (Um Sinal por Candle/Minuto)
                    # =====================================================
                    last_attempt = ATTEMPT_COOLDOWN_TRACKER.get(symbol, 0)
                    if time.time() - last_attempt < 60:
                        # Silenciosamente pula para não spammar log
                        remaining = int(60 - (time.time() - last_attempt))
                        bot_state.update_monitoring(symbol, "⏳ COOLDOWN", f"Veto Cooldown ({remaining}s)", 0.0)
                        continue
                    # =====================================================

                    # =====================================================
                    # ✅ v5.1: CORRELATION FILTER (Land Trading)
                    # =====================================================
                    is_corr_blocked, corr_reason, corr_symbol = utils.check_correlation(symbol)
                    if is_corr_blocked:
                        if iteration_count % 30 == 0:
                            logger.info(f"🔗 {symbol}: {corr_reason}")
                            log_signal_analysis(symbol, "NONE", "CORRELATION", 0, True, corr_reason, {}, session_name=current_session.get("name"))
                        bot_state.update_monitoring(symbol, "🔗 CORRELAÇÃO", corr_reason, 0.0)
                        continue

                    # =====================================================
                    # ✅ v5.1: VOLATILITY FILTER (Land Trading)
                    # =====================================================
                    vol_ok, vol_reason, atr_val = utils.is_volatility_ok(symbol)
                    if not vol_ok:
                        if iteration_count % 60 == 0:
                            logger.info(f"📉 {symbol}: {vol_reason}")
                            log_signal_analysis(symbol, "NONE", "VOLATILITY", 0, True, vol_reason, {}, session_name=current_session.get("name"))
                        bot_state.update_monitoring(symbol, "📉 BAIXA VOL", vol_reason, 0.0)
                        continue

                    # ✅ v5.3: Coleta de Motivos de Rejeição
                    reject_reasons = []
                    
                    # 1. Busca sinal e motivo detalhado
                    signal, ind, strategy, reason = check_for_signals(symbol, current_session)
                    try:
                        watchdog.heartbeat("FastLoop")
                    except Exception:
                        pass
                    
                    # ✅ CORREÇÃO: Sempre registrar no analysis log mesmo sem sinal técnico
                    if not signal or signal == "NONE":
                        display_reason = reason if reason else "Aguardando setup..."
                        bot_state.update_monitoring(symbol, "🔵 MONITORANDO", display_reason, 0.0)
                        log_signal_analysis(symbol, "NONE", strategy or "N/A", 0, True, display_reason, ind or {}, session_name=current_session.get("name"))
                        continue

                    # ✅ MODO PROTEÇÃO: Bloqueia novas entradas
                    if current_session.get("name") == "PROTECTION":
                        reject_reasons.append("Modo Proteção Ativo")
                        if iteration_count % 60 == 0:
                            logger.info(f"🛡️ {symbol}: Modo Proteção Ativo. Novas entradas bloqueadas.")
                        bot_state.update_monitoring(symbol, "🛡️ PROTEÇÃO", "Apenas Gerenciamento", 0.0)
                        log_signal_analysis(symbol, signal, strategy, 0, True, "Modo Proteção Ativo", ind, session_name=current_session.get("name"))
                        continue
                    
                    # ✅ REQUISITO: Verificação de indicadores nulos para pular ML
                    if ind is None:
                        reject_reasons.append("RISK: Indicadores Nulos")
                        logger.warning(f"⚠️ Pulando análise de ML para {symbol} devido a indicadores ausentes.")
                        bot_state.update_monitoring(symbol, "⚠️ IND. ERROR", "Indicadores Nulos", 0.0)
                        continue

                    # Regime Filters
                    try:
                        adx_req = float(getattr(config, "ADX_MIN_STRENGTH", 28))
                        adx_now = float(ind.get("adx", 0))
                        if adx_now < adx_req:
                            reason_reg = f"Regime: ADX baixo ({adx_now:.1f} < {adx_req:.1f})"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        close_price = ind.get("close", ind.get("current_price"))
                        bb_upper = ind.get("bb_upper")
                        bb_lower = ind.get("bb_lower")
                        if signal == "BUY" and (bb_upper is None or close_price is None or close_price < bb_upper):
                            reason_reg = "Regime: BB sem breakout (BUY)"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        if signal == "SELL" and (bb_lower is None or close_price is None or close_price > bb_lower):
                            reason_reg = "Regime: BB sem breakout (SELL)"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        atr_now = float(ind.get("atr", 0) or 0)
                        atr20 = float(ind.get("atr20", 0) or 0)
                        if atr20 > 0 and atr_now < (1.3 * atr20):
                            reason_reg = f"Regime: ATR baixo ({atr_now:.5f} < 1.3×ATR20)"
                            bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                            log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                            continue
                        ema200_h4 = ind.get("ema200_h4")
                        if ema200_h4 is not None and close_price is not None:
                            if signal == "BUY" and not (close_price > ema200_h4):
                                reason_reg = "Regime: H4 EMA200 contra BUY"
                                bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                                log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                                continue
                            if signal == "SELL" and not (close_price < ema200_h4):
                                reason_reg = "Regime: H4 EMA200 contra SELL"
                                bot_state.update_monitoring(symbol, "⚖️ REGIME", reason_reg, 0.0)
                                log_signal_analysis(symbol, signal, strategy, 0, True, reason_reg, ind, session_name=current_session.get("name"))
                                continue
                    except Exception:
                        pass

                    # =====================================================
                    # ✅ SOVEREIGN VETO (FIM DO ML PRIORITY)
                    # =====================================================
                    veto_signal = signal in ["BUY_VETO", "SELL_VETO"]
                    if veto_signal:
                        reject_reasons.append(f"TECHNICAL: Veto Técnico: {reason}")
                        ATTEMPT_COOLDOWN_TRACKER[symbol] = time.time() 
                        
                        reason_veto = f"❌ VETO {reason}"
                        if iteration_count % 10 == 0:
                            logger.info(f"🛑 {symbol}: {reason_veto}")
                        
                        bot_state.update_monitoring(symbol, "🔵 MONITORANDO", reason_veto, 0.0)
                        now_ts = time.time()
                        throttle_s = getattr(config, 'ANALYSIS_LOG_THROTTLE_SECONDS', 120)
                        prev = analysis_log_state.get(symbol)
                        prev_reason = prev.get("reason") if prev else None
                        prev_ts = prev.get("ts") if prev else 0
                        if (prev_reason != reason_veto) or ((now_ts - prev_ts) > throttle_s):
                            analysis_log_state[symbol] = {"reason": reason_veto, "ts": now_ts}
                            log_signal_analysis(symbol, signal, strategy or "N/A", 0, True, reason_veto, ind or {}, session_name=current_session.get("name"))
                        continue

                    ml_confidence_real = ind.get('ml_confidence')
                    if (time.time() - symbol_start_ts) > float(getattr(config, 'MAX_SYMBOL_PROCESS_SECONDS', 3.0)):
                        logger.warning(f"⏳ {symbol}: Tempo de análise excedido ({time.time() - symbol_start_ts:.1f}s). Pulando para próximo símbolo.")
                        bot_state.update_monitoring(symbol, "⏳ TIMEOUT", "Análise excedeu tempo máximo", 0.0)
                        try:
                            log_signal_analysis(symbol, "NONE", strategy or "N/A", 0, True, "Timeout de análise", ind or {}, session_name=current_session.get("name"))
                        except Exception:
                            pass
                        continue
                    ml_is_baseline = False

                    if hasattr(utils, "ml_optimizer_instance") and utils.ml_optimizer_instance:
                        try:
                            df_ml = ind.get("df")
                            ml_score_raw, ml_is_baseline = utils.ml_optimizer_instance.get_prediction_score(
                                symbol, ind, df_ml, signal=signal
                            )
                            ml_confidence_real = ml_score_raw / 100.0
                        except Exception as e_ml:
                            logger.error(f"Erro na predição ML em fast_loop para {symbol}: {e_ml}")
                            if ml_confidence_real is None:
                                ml_confidence_real = 0.5
                            ml_is_baseline = True
                    else:
                        if ml_confidence_real is None:
                            ml_confidence_real = 0.5

                    if isinstance(ind, dict) and "df" in ind:
                        del ind["df"]

                    ml_score = ml_confidence_real * 100
                    if isinstance(ind, dict) and ind.get("penalty_candle"):
                        ml_score = max(0.0, ml_score - float(ind.get("penalty_candle", 0)))

                    confidence_threshold = params.get("ml_threshold", getattr(config, 'ML_CONFIDENCE_THRESHOLD', 0.60))
                    min_ml_score = confidence_threshold * 100
                    try:
                        sess_name = (current_session or {}).get("name")
                        # ✅ Verificação de spread por sessão (Forex)
                        try:
                            sp_points = float(ind.get("spread_points", 0.0) or 0.0)
                            base_ok = bool(ind.get("spread_ok", True))
                            effective_spread_ok = base_ok
                            cat = "FOREX"
                            su = symbol.upper()
                            if any(idx in su for idx in ["US30", "NAS100", "USTEC", "DE40", "GER40", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"]):
                                cat = "INDICES"
                            elif any(met in su for met in ["XAU", "XAG", "GOLD", "SILVER"]):
                                cat = "METALS"
                            elif any(c in su for c in ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]):
                                cat = "CRYPTO"
                            sess_limits = getattr(config, "SESSION_SPREAD_LIMITS", {})
                            lim = sess_limits.get(sess_name or "", {}).get(cat, {})
                            if sess_name == "GOLDEN":
                                allowance = float(lim.get("allowance_pct", getattr(config, "GOLDEN_SPREAD_ALLOWANCE_PCT", 0.0)))
                                if not base_ok and allowance > 0:
                                    effective_spread_ok = True
                            elif sess_name == "ASIAN":
                                max_pts = float(lim.get("max_points", getattr(config, "ASIAN_MAX_SPREAD_POINTS", 50)))
                                effective_spread_ok = sp_points <= max_pts
                            elif sess_name == "PROTECTION":
                                max_pips = float(lim.get("max_pips", getattr(config, "PROTECTION_MAX_SPREAD_FOREX", 20)))
                                pip_size = utils.get_pip_size(symbol)
                                point = getattr(utils.get_symbol_info(symbol), "point", 0.0) if utils.get_symbol_info(symbol) else 0.0
                                sp_pips = (sp_points * point) / pip_size if pip_size > 0 else 999.0
                                effective_spread_ok = sp_pips <= max_pips
                            if not effective_spread_ok:
                                reason_spread = f"Spread alto ({sp_points:.0f} pts) sessão {sess_name or 'N/A'}"
                                reject_reasons.append(reason_spread)
                                bot_state.update_monitoring(symbol, "🟡 SPREAD", reason_spread, ml_score)
                                log_signal_analysis(symbol, signal, strategy, ml_score, True, reason_spread, ind, ml_score=ml_score)
                                continue
                        except Exception:
                            pass
                    except Exception:
                        pass
                    if isinstance(ind, dict) and ind.get("ema_against"):
                        min_ml_score = max(min_ml_score, 50.0)
                    try:
                        eff_rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
                        eff_rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
                        if params:
                            allowed = {"ema_short", "ema_long", "adx_threshold", "bb_squeeze_threshold"}
                            safe_params = {k: v for k, v in params.items() if k in allowed}
                            ts_bypass, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high, **safe_params)
                        else:
                            ts_bypass, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high)
                        if ts_bypass >= 65:
                            ml_score = 70.0 if ml_score < 70.0 else ml_score
                    except Exception:
                        pass
                    try:
                        wr, total, _wins = utils.calculate_rolling_win_rate(symbol)
                        if total >= 20 and wr > 0.65:
                            min_ml_score = float(min_ml_score) * 0.95
                    except Exception:
                        pass

                    cache_entry = GLOBAL_MONITOR_CACHE.get(symbol, {})
                    GLOBAL_MONITOR_CACHE[symbol] = {
                        "status": cache_entry.get("status", "🔍 ANALISANDO"),
                        "reason": cache_entry.get("reason", "ML Avaliação"),
                        "ml_score": ml_score,
                        "is_baseline": ml_is_baseline,
                        "timestamp": datetime.now(),
                    }

                    # Log detalhado para debug
                    logger.info(f"[{symbol}] ML confidence real: {ml_confidence_real:.3f} → Score: {ml_score:.1f}/100 | Threshold: {min_ml_score:.0f}")

                    ml_override_used = False
                    ml_override_risk_mult = 1.0
                    if ml_score < min_ml_score:
                        allow_override = getattr(config, "ENABLE_ML_VETO_OVERRIDE", True)
                        allowed_sessions = getattr(config, "ML_VETO_OVERRIDE_ALLOWED_SESSIONS", ["GOLDEN", "NORMAL"])
                        try:
                            sess_name = (current_session or {}).get("name")
                            if isinstance(allowed_sessions, (list, tuple, set)) and sess_name and sess_name not in allowed_sessions:
                                allow_override = False
                        except Exception:
                            pass
                        try:
                            min_conf = float(getattr(config, "ML_OVERRIDE_MIN_CONFIDENCE", 0.90))
                            if ml_confidence_real < min_conf:
                                allow_override = False
                        except Exception:
                            allow_override = False
                        override_min_score = getattr(config, "ML_VETO_OVERRIDE_MIN_TECH_SCORE", 80)
                        override_max_spread = getattr(config, "ML_VETO_OVERRIDE_MAX_SPREAD_PIPS", 2.0)
                        spread_pips = float(ind.get("spread_pips", 999.0) or 999.0)
                        try:
                            if ind.get("spread_ok") is False:
                                spread_pips = 999.0
                        except Exception:
                            pass
                        tech_score_for_override = 0.0
                        if allow_override:
                            try:
                                eff_rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
                                eff_rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
                                if params:
                                    allowed = {"ema_short", "ema_long", "adx_threshold", "bb_squeeze_threshold"}
                                    safe_params = {k: v for k, v in params.items() if k in allowed}
                                    tech_score_for_override, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high, **safe_params)
                                else:
                                    tech_score_for_override, _ = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high)
                            except Exception:
                                tech_score_for_override = 0.0

                        if allow_override and tech_score_for_override >= override_min_score and spread_pips <= override_max_spread:
                            ml_override_used = True
                            ml_override_risk_mult = float(getattr(config, "ML_VETO_OVERRIDE_RISK_MULTIPLIER", 0.5))
                            reason_override = f"ML: Override ({ml_score:.0f} < {min_ml_score:.0f}) | Tech:{tech_score_for_override:.1f} | Spread:{spread_pips:.2f}p"
                            bot_state.update_monitoring(symbol, "🟣 OVERRIDE", reason_override, ml_score)
                        else:
                            reason_ml = f"ML: Veto ML ({ml_score:.0f} < {min_ml_score:.0f})"
                            reject_reasons.append(reason_ml)
                            bot_state.update_monitoring(symbol, "🔵 MONITORANDO", reason_ml, ml_score)
                            log_signal_analysis(symbol, signal, strategy, ml_score, True, reason_ml, ind, ml_score=ml_score)
                            continue
                    else:
                        logger.info(f"[{symbol}] ML aprovado: {ml_score:.1f} >= {min_ml_score:.0f}")

                    # ✅ Gate de força de tendência via ADX mínimo
                    try:
                        adx_min = float(getattr(config, "ADX_MIN_STRENGTH", 25))
                        sym_up = symbol.upper()
                        adx_map = getattr(config, "ADX_MIN_STRENGTH_BY_SYMBOL", {})
                        if isinstance(adx_map, dict) and sym_up in adx_map:
                            adx_min = float(adx_map[sym_up])
                        current_adx = float(ind.get("adx", 0.0) or 0.0)
                        if current_adx < adx_min:
                            reason_adx = f"ADX baixo ({current_adx:.1f} < {adx_min:.0f})"
                            reject_reasons.append(reason_adx)
                            bot_state.update_monitoring(symbol, "🔹 ADX", reason_adx, ml_score)
                            log_signal_analysis(symbol, signal, strategy, ml_score, True, reason_adx, ind, ml_score=ml_score)
                            continue
                    except Exception:
                        pass
                    # Verifica News Filter
                    is_blackout, news_reason = news_filter.is_news_blackout(symbol)
                    if is_blackout:
                        reject_reasons.append(f"NEWS: {news_reason}")
                        bot_state.update_monitoring(symbol, "⚠️ BLOQUEADO", news_reason, ml_score)
                        log_signal_analysis(symbol, signal, strategy, ml_score, True, f"Notícia: {news_reason}", ind, ml_score=ml_score)
                        continue

                    # Cálculo de Score Final com prioridade ao ML
                    eff_rsi_low = params.get("rsi_low", getattr(config, 'RSI_LOW_LIMIT', 30))
                    eff_rsi_high = params.get("rsi_high", getattr(config, 'RSI_HIGH_LIMIT', 70))
                    if params:
                        allowed = {"ema_short", "ema_long", "adx_threshold", "bb_squeeze_threshold"}
                        safe_params = {k: v for k, v in params.items() if k in allowed}
                        tech_score, details = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high, **safe_params)
                    else:
                        tech_score, details = utils.calculate_signal_score(ind, rsi_low=eff_rsi_low, rsi_high=eff_rsi_high)
                    logger.info(f"[{symbol}] Score Técnico (referência): {tech_score:.1f} | Breakdown: {details}")
                    penalties = 0.0
                    if ind.get("ema_against"):
                        penalties += 10.0
                    if ind.get("penalty_candle"):
                        penalties += float(ind.get("penalty_candle", 10.0))
                    if ind.get("adx_low"):
                        penalties += 5.0
                    final_score = max(0.0, ml_score - penalties)
                    boost_min_ml = float(getattr(config, 'ML_BOOST_MIN_ML_SCORE', 60))
                    boost_min_tech = float(getattr(config, 'ML_BOOST_MIN_TECH_SCORE', 28))
                    boost_cap = float(getattr(config, 'ML_BOOST_MAX_POINTS', 25))
                    if ml_score >= boost_min_ml and tech_score >= boost_min_tech:
                        final_score += min(boost_cap, tech_score * 0.1)

                    current_threshold = float(getattr(config, 'ML_MIN_SCORE', 70.0))
                    logger.info(f"[{symbol}] Threshold ML: {current_threshold:.1f} | Score Final: {final_score:.1f}")
                    
                    try:
                        regime = utils.get_volatility_regime(symbol, ind.get("df") if isinstance(ind, dict) else None)
                        if regime == "HIGH":
                            current_threshold -= 10
                        elif regime == "LOW":
                            current_threshold += 5
                    except Exception:
                        pass
                    min_threshold = float(getattr(config, 'ML_MIN_SCORE', 70.0))
                    if current_threshold < min_threshold:
                        current_threshold = min_threshold

                    if final_score < current_threshold:
                        reject_reasons.append(f"ML: Score Insuficiente ({final_score:.1f} < {current_threshold:.1f})")
                        logger.info(f"[{symbol}] Score insuficiente ({final_score:.1f} < {current_threshold:.1f})")
                        log_signal_analysis(symbol, signal, strategy, final_score, True, f"Score Baixo: {final_score:.1f}", ind, ml_score=ml_score)
                        continue

                    # Candle não confirmado já tratado como penalidade; não vete aqui

                    ok_port, msg_port = utils.check_portfolio_exposure(
                        pending_symbol=symbol,
                        pending_volume=0.0,  # volume ainda não definido; validaremos no attempt_entry
                        pending_side=signal
                    )
                    if not ok_port and iteration_count % 30 == 0:
                        logger.warning(f"🟠 {symbol}: {msg_port}")
                        bot_state.update_monitoring(symbol, "🟠 PORTFÓLIO", msg_port, ml_score)
                        continue
                    executed = attempt_entry(symbol, signal, strategy, ind, params, current_session, iteration_count, final_score, ml_score, ml_override_risk_mult, ml_override_used, rejection_stats)
                    if not executed:
                        continue

                except Exception as e:
                    logger.error(f"❌ Erro crítico ao analisar/executar para {symbol}: {e}", exc_info=True)
                    bot_state.update_monitoring(symbol, "⚠️ ERRO", "Falha na Análise", 0.0)
                    continue

            # ========================================
            # GESTÃO DE POSIÇÕES (Breakeven, Trailing Stop)
            # ========================================
            positions = mt5_exec(mt5.positions_get)

            if positions:
                for pos in positions:
                    if pos.magic != getattr(config, 'MAGIC_NUMBER', 123456): # ✅ CORREÇÃO: getattr para MAGIC_NUMBER
                        continue

                    try:
                        symbol_info = utils.get_symbol_info(pos.symbol)
                        if not symbol_info:
                            logger.warning(f"⚠️ Não foi possível obter info para {pos.symbol} para gerenciar posição {pos.ticket}.")
                            continue

                        pip_size = utils.get_pip_size(pos.symbol)
                        if pip_size == 0:
                            logger.warning(f"⚠️ Pip size é zero para {pos.symbol}. Não é possível gerenciar posição {pos.ticket}.")
                            continue

                        current_tick = mt5_exec(mt5.symbol_info_tick, pos.symbol)
                        if current_tick is None:
                            logger.warning(f"⚠️ Não foi possível obter tick atual para {pos.symbol}. Não é possível gerenciar posição {pos.ticket}.")
                            continue

                        current_price = current_tick.bid if pos.type == mt5.POSITION_TYPE_BUY else current_tick.ask

                        if pos.type == mt5.POSITION_TYPE_BUY:
                            pips_profit = (current_price - pos.price_open) / pip_size
                        else:
                            pips_profit = (pos.price_open - current_price) / pip_size

                        # ========================================
                        # ✅ v5.0: NEWS-BASED POSITION CLOSING
                        # ========================================
                        # Fecha posições lucrativas 15min antes de notícias alto impacto
                        # ========================================
                        try:
                            # ========================================
                            # ✅ TRIPLE SWAP GUARD (Quarta-Feira)
                            # ========================================
                            # Evita pagar swap triplo se a posição estiver pagando swap
                            # Executa entre 18:00 e 18:55 BRT (Antes do Rollover das 19:00)
                            now_guard = datetime.now()
                            triple_swap_close = False
                            
                            if now_guard.weekday() == 2 and now_guard.hour == 18 and now_guard.minute >= 30:
                                # Verifica Swap da posição
                                if pos.swap < 0:
                                    logger.warning(f"📅 TRIPLE SWAP ALERT: Fechando {pos.symbol} para evitar custo triplo (Swap: {pos.swap}).")
                                    triple_swap_close = True
                            
                            should_close_news, news_reason, minutes_to = news_filter.should_close_for_news(
                                pos.symbol, pos.profit
                            )
                            
                            if (should_close_news and pos.profit > 0) or triple_swap_close:
                                close_reason = f"Triple Swap Avoidance (Swap: {pos.swap})" if triple_swap_close else news_reason
                                logger.info(f"🛡️ Fechando posição {pos.ticket} ({pos.symbol}): {close_reason}")
                                
                                # Fecha a posição
                                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                close_price = current_tick.bid if pos.type == mt5.POSITION_TYPE_BUY else current_tick.ask
                                
                                close_request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": pos.symbol,
                                    "volume": pos.volume,
                                    "type": order_type,
                                    "position": pos.ticket,
                                    "price": close_price,
                                    "deviation": getattr(config, 'DEVIATION', 20),
                                    "magic": getattr(config, 'MAGIC_NUMBER', 123456),
                                    "comment": f"Close: {close_reason}",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                                
                                result = mt5_exec(mt5.order_send, close_request)
                                
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(f"✅ Posição {pos.ticket} fechada com sucesso.")
                                    # ✅ TELEGRAM: Notifica Fechamento
                                    msg_close = (
                                        f"🛡️ <b>XP3 PRO: Fechamento de Proteção</b>\n\n"
                                        f"🆔 Ativo: <b>{pos.symbol}</b>\n"
                                        f"💰 Lucro/Perda: <b>${pos.profit:.2f}</b>\n"
                                        f"⚠️ Motivo: {close_reason}"
                                    )
                                    utils.send_telegram_alert(msg_close)
                                    try:
                                        utils.record_trade_close(
                                            ticket=int(getattr(pos, "ticket", 0)),
                                            symbol=str(getattr(pos, "symbol", "")),
                                            side=("BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"),
                                            volume=float(getattr(pos, "volume", 0.0)),
                                            open_time=datetime.fromtimestamp(getattr(pos, "time", time.time())),
                                            close_time=datetime.now(),
                                            open_price=float(getattr(pos, "price_open", 0.0)),
                                            close_price=float(close_price or 0.0),
                                            sl=float(getattr(pos, "sl", 0.0)),
                                            tp=float(getattr(pos, "tp", 0.0)),
                                            profit=float(getattr(pos, "profit", 0.0)),
                                            commission=float(getattr(pos, "commission", 0.0)) if hasattr(pos, "commission") else 0.0,
                                            swap=float(getattr(pos, "swap", 0.0)),
                                            magic=int(getattr(pos, "magic", getattr(config, 'MAGIC_NUMBER', 123456))),
                                            comment=str(close_reason)
                                        )
                                    except Exception:
                                        pass
                                else:
                                    error_msg = result.comment if result else "result is None"
                                    logger.error(f"❌ Falha ao fechar posição {pos.ticket}: {error_msg}")
                                continue  # Não processa mais esta posição neste ciclo
                        except Exception as e:
                            logger.error(f"❌ Erro ao verificar news/swap close para {pos.symbol}: {e}")

                        # ========================================
                        # ✅ BREAKEVEN PLUS (SNIPER MODE v5.1)
                        # ========================================
                        # Gatilho: 1.5x Risco (1.5R) OU Config Manual
                        # Ação: Move SL para Entrada + 2 pips (Cobre taxas)
                        # ========================================
                        
                        # Cálculo do Risco Inicial (R)
                        if pos.type == mt5.POSITION_TYPE_BUY:
                            initial_risk_pips = (pos.price_open - pos.sl) / pip_size if pos.sl > 0 else 0
                        else:
                            initial_risk_pips = (pos.sl - pos.price_open) / pip_size if pos.sl > 0 else 0
                            
                        # Cálculo do Retorno Atual em R (R-Multiple)
                        current_r = pips_profit / initial_risk_pips if initial_risk_pips > 0 else 0
                        
                        be_trigger_r = getattr(config, 'BREAKEVEN_TRIGGER_R', 1.5)
                        be_buffer_pips = getattr(config, 'BREAKEVEN_BUFFER_PIPS', 2.0)
                        
                        # Verifica se deve acionar o BE
                        should_trigger_be = False
                        if getattr(config, 'ENABLE_BREAKEVEN', True) and pos.sl != 0:
                            # Lógica R-Multiple (Prioridade)
                            if current_r >= be_trigger_r:
                                should_trigger_be = True
                            # Fallback para Pips Fixos (Legacy)
                            elif pips_profit >= getattr(config, 'BREAKEVEN_TRIGGER_PIPS', 15):
                                should_trigger_be = True
                                
                        if should_trigger_be:
                            # Verifica se já está no Breakeven (ou melhor)
                            is_in_profit_zone = False
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                if pos.sl >= pos.price_open: is_in_profit_zone = True
                            else:
                                if pos.sl <= pos.price_open: is_in_profit_zone = True

                            # Se não estiver no BE, move
                            if not is_in_profit_zone:
                                if pos.type == mt5.POSITION_TYPE_BUY:
                                    new_sl = pos.price_open + (be_buffer_pips * pip_size)
                                    if new_sl > pos.sl: # Garante que só sobe
                                        if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                            logger.info(f"🛡️ BREAKEVEN PLUS ({current_r:.1f}R) para {symbol}: SL movido para Entrada + {be_buffer_pips} pips")
                                else:
                                    new_sl = pos.price_open - (be_buffer_pips * pip_size)
                                    if new_sl < pos.sl: # Garante que só desce
                                        if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                            logger.info(f"🛡️ BREAKEVEN PLUS ({current_r:.1f}R) para {symbol}: SL movido para Entrada - {be_buffer_pips} pips")

                        # Trailing Stop
                        if getattr(config, 'ENABLE_TRAILING_STOP', False) and pips_profit >= getattr(config, 'TRAILING_START_PIPS', 100):
                            
                            ts_step = getattr(config, 'TRAILING_STEP_PIPS', 50)
                            ts_dist = getattr(config, 'TRAILING_DISTANCE_PIPS', 30)

                            if pos.type == mt5.POSITION_TYPE_BUY:
                                # Preço Alvo do SL = Preço Atual - Distancia
                                target_sl = current_price - (ts_dist * pip_size)
                                # Lógica de Degrau (Step):
                                # Só move se o Target SL for maior que o SL Atual + Step
                                if target_sl >= (pos.sl + (ts_step * pip_size)):
                                    new_sl = round(target_sl, symbol_info.digits)
                                    if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                        logger.info(f"📈 TRAILING STOP (Step {ts_step}) para {pos.symbol}: SL subiu para {new_sl} (Travando {ts_dist} pips)")

                            else: # SELL
                                target_sl = current_price + (ts_dist * pip_size)
                                # Só move se o Target SL for menor que o SL Atual - Step
                                if target_sl <= (pos.sl - (ts_step * pip_size)):
                                    new_sl = round(target_sl, symbol_info.digits)
                                    if utils.modify_position_sl_tp(pos.ticket, new_sl, pos.tp):
                                        logger.info(f"📉 TRAILING STOP (Step {ts_step}) para {pos.symbol}: SL desceu para {new_sl} (Travando {ts_dist} pips)")

                    except Exception as e:
                        logger.error(f"❌ Erro ao gerenciar posição {pos.ticket} para {pos.symbol}: {e}", exc_info=True)
                        continue
            
            # ✅ PERF: Libera mt5_lock para o painel atualizar antes do sleep longo
            time.sleep(1)

            # ✅ NOVO: Log Periódico de Status (Heartbeat Visual) - A cada ~1 min
            if iteration_count % 4 == 0:
                status_summary = []
                for sym, data in GLOBAL_MONITOR_CACHE.items():
                    # Pega status curto
                    status = data.get('status', 'N/A')
                    if "MONITORANDO" in status:
                        # Se monitorando, mostra o motivo/score
                        reason = data.get('reason', '')
                        ml_val = data.get('ml_score', 0)
                        # 🔄 AJUSTE: Truncamento aumentado para 35 chars para ver valor do Spread
                        status_summary.append(f"{sym}: {reason[:35]} (ML:{ml_val:.0f})")
                    elif "OPERANDO" in status:
                        status_summary.append(f"🔴 {sym}: EM POSIÇÃO")
                    else:
                        status_summary.append(f"{sym}: {status}")
                
                if status_summary:
                    # ✅ Exibe todos os ativos monitorados (sem limite)
                    summary_str = " | ".join(status_summary)
                    logger.info(f"📊 Status Geral: {summary_str}")

            # ===========================
            # BUY DIAGNOSTIC (FOREX_PAIRS)
            # ===========================
            try:
                last_diag_ts = globals().get("_LAST_BUY_DIAG_TS", 0.0)
                do_diag = getattr(config, "ENABLE_BUY_DIAGNOSTIC", True)
                diag_interval = float(getattr(config, "BUY_DIAGNOSTIC_INTERVAL_SECONDS", 90))
                if do_diag and (time.time() - last_diag_ts) >= diag_interval:
                    globals()["_LAST_BUY_DIAG_TS"] = time.time()
                    all_fx = list(getattr(config, "FOREX_PAIRS", {}).keys())
                    max_syms = int(getattr(config, "BUY_DIAGNOSTIC_MAX_SYMBOLS", 30))
                    if not all_fx:
                        pass
                    else:
                        syms_diag = GLOBAL_ACTIVE_SYMBOLS.copy()
                        lines = []
                        for s in syms_diag:
                            try:
                                mt5_exec(mt5.symbol_select, s, True)
                                sig, ind, strat, reason = check_for_signals(s, current_session)
                                if not isinstance(ind, dict) or ind is None:
                                    lines.append(f"{s}: Indicadores indisponíveis ({reason})")
                                    continue
                                params = ELITE_CONFIG.get(s, {})
                                ema_trend = ind.get("ema_trend")
                                rsi_now = ind.get("rsi", 50)
                                adx_now = ind.get("adx", 0)
                                bb_lower = ind.get("bb_lower")
                                close_price = ind.get("close", 0)
                                open_price = ind.get("open", 0)
                                spread_pts = ind.get("spread_points", ind.get("spread_pips", 0))
                                vol_ratio = ind.get("volume_ratio", 0.0)
                                ema_200 = ind.get("ema_200")
                                atr = ind.get("atr", 0.0)
                                price = ind.get("price") or ind.get("close")
                                
                                # Limites
                                base_adx_thr = params.get("adx_threshold", getattr(config, 'ADX_THRESHOLD', 20))
                                try:
                                    low_trend_map = getattr(config, 'LOW_TREND_ADX_THRESHOLDS', {})
                                    adx_thr = low_trend_map.get(s, base_adx_thr)
                                except Exception:
                                    adx_thr = base_adx_thr
                                rsi_req = 50
                                min_vol = getattr(config, 'MIN_VOLUME_COEFFICIENT', 0.4)
                                # Spread max por classe
                                sym_up = s.upper()
                                is_crypto = any(c in sym_up for c in ["BTC","ETH","SOL","ADA","BNB","XRP","LTC","DOGE"])
                                is_indices = any(i in sym_up for i in ["US30","NAS100","USTEC","DE40","GER40","GER30","UK100","US500","USA500","SPX500","HK50","JP225","FRA40"])
                                is_metals = any(m in sym_up for m in ["XAU","XAG","GOLD","SILVER"])
                                is_exotic = any(x in sym_up for x in ["TRY","ZAR","MXN","RUB","CNH","PLN","HUF","CZK","DKK","NOK","SEK"])
                                if is_crypto:
                                    max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
                                elif is_indices:
                                    max_spread = getattr(config, 'MAX_SPREAD_INDICES', 600)
                                elif is_metals:
                                    max_spread = getattr(config, 'MAX_SPREAD_METALS', 80)
                                elif is_exotic:
                                    max_spread = getattr(config, 'MAX_SPREAD_EXOTICS', 8000)
                                else:
                                    max_spread = getattr(config, 'MAX_SPREAD_ACCEPTABLE', 25)
                                # Sessão ajusta spread
                                sess_name = current_session.get("name")
                                if sess_name == "GOLDEN" and not (is_crypto or is_indices or is_metals or is_exotic):
                                    max_spread = max_spread * (1 + getattr(config, 'GOLDEN_SPREAD_ALLOWANCE_PCT', 0.20))
                                elif sess_name == "PROTECTION" and not (is_crypto or is_indices or is_metals or is_exotic):
                                    max_spread = getattr(config, 'PROTECTION_MAX_SPREAD_FOREX', 20)
                                elif sess_name == "ASIAN" and not (is_crypto or is_indices or is_metals or is_exotic):
                                    max_spread = getattr(config, 'ASIAN_MAX_SPREAD_POINTS', 50)
                                
                                missing = []
                                # Trend path readiness (permite NEUTRAL além de UP)
                                if ema_trend not in ("UP", "NEUTRAL"):
                                    missing.append(f"EMA Trend=UP/NEUTRAL (atual {ema_trend or 'N/A'})")
                                if adx_now < adx_thr:
                                    missing.append(f"ADX≥{adx_thr} (atual {adx_now:.1f})")
                                if rsi_now <= rsi_req:
                                    missing.append(f"RSI>{rsi_req} (atual {rsi_now:.1f})")
                                if close_price <= open_price:
                                    missing.append("Candle de Alta (C>O)")
                                # Macro EMA200
                                if getattr(config, 'ENABLE_EMA_200_FILTER', True) and ema_200 is not None and price is not None:
                                    if not (price >= ema_200):
                                        missing.append(f"Preço≥EMA200 (P:{price:.5f} EMA200:{ema_200:.5f})")
                                # Vol/Spread
                                if vol_ratio < min_vol:
                                    missing.append(f"Volume≥{min_vol:.2f} (atual {vol_ratio:.2f})")
                                if spread_pts > max_spread:
                                    missing.append(f"Spread≤{max_spread:.0f} (atual {spread_pts})")
                                # ML requirement (usa ML_MIN_SCORE explícito)
                                try:
                                    ml_conf = ind.get('ml_confidence')
                                    if hasattr(utils, "ml_optimizer_instance") and utils.ml_optimizer_instance and isinstance(ind, dict):
                                        df_ml = ind.get("df")
                                        ml_score_raw, _ = utils.ml_optimizer_instance.get_prediction_score(s, ind, df_ml, signal="BUY")
                                        ml_conf = ml_score_raw / 100.0
                                    if ml_conf is None:
                                        ml_conf = 0.5
                                    min_ml_score = float(getattr(config, "ML_MIN_SCORE", 45))
                                    if (ml_conf * 100) < min_ml_score:
                                        missing.append(f"ML≥{min_ml_score:.0f} (atual {ml_conf*100:.0f})")
                                except Exception:
                                    pass
                                
                                ml_score_int = int(((ind.get('ml_confidence') or 0.5) * 100))
                                if missing:
                                    lines.append(f"{s}: faltam " + "; ".join(missing))
                                    bot_state.update_monitoring(s, "🔵 MONITORANDO", "Faltam: " + " | ".join(missing), float(ml_score_int))
                                    log_signal_analysis(s, "NONE", strat or "N/A", 0, True, " | ".join(missing), ind or {}, session_name=current_session.get("name"))
                                else:
                                    lines.append(f"{s}: pronto para BUY (condições atendidas)")
                                    bot_state.update_monitoring(s, "🟢 PRONTO", "Condições atendidas", float(ml_score_int))
                                    log_signal_analysis(s, sig or "BUY", strat or "TREND", 0, False, "Condições atendidas", ind or {}, session_name=current_session.get("name"))
                            except Exception as e_diag:
                                lines.append(f"{s}: erro diagnóstico ({e_diag})")
                                log_signal_analysis(s, "NONE", strat or "N/A", 0, True, f"Erro diagnóstico: {e_diag}", ind or {}, session_name=current_session.get("name"))
                        if lines:
                            logger.info("🧪 WHY-BUY:\n" + "\n".join(lines))
            except Exception:
                pass

            watchdog.heartbeat("FastLoop")
            time.sleep(getattr(config, 'FAST_LOOP_INTERVAL', 15)) # ✅ CORREÇÃO: getattr para FAST_LOOP_INTERVAL

        except Exception as e:
            # ✅ Land Trading: Log de Erro Sem Morte
            logger.critical(f"💀 ERRO FATAL NA THREAD FastLoop: {e}", exc_info=True)
            time.sleep(5)
            continue
            
    # finally: removido para evitar erro de sintaxe.
    # O Lock Check no início já impede duplicação.
    # Se a thread morrer, o restart do bot limpa a flag.

# ===========================
# SLOW LOOP
# ===========================
def slow_loop():
    """Loop lento: manutenção, re-otimização ML, news filter"""
    safe_log(logging.INFO, "🧠 Slow Loop iniciado")

    last_health_check = 0
    last_ml_reoptimization = datetime.now() # ✅ NOVO: Para controlar re-otimização ML
    last_heartbeat_update = 0
    while not shutdown_event.is_set():
        try:
            current_time_ts = time.time()
            if current_time_ts - last_heartbeat_update > 60:  # Atualiza a cada 60s
                update_heartbeat()
                last_heartbeat_update = current_time_ts
            # ✅ REQUISITO: Verificação de Iteráveis
            if not GLOBAL_ACTIVE_SYMBOLS:
                logger.warning("⚠️ SlowLoop: GLOBAL_ACTIVE_SYMBOLS está vazia! Aguardando 5s...")
                time.sleep(5)
                continue

            watchdog.heartbeat("SlowLoop")
            current_time = time.time()

            if not utils.is_market_open():
                time.sleep(300)
                continue

            # Health check MT5
            if current_time - last_health_check > 60:
                last_health_check = current_time
                if not utils.check_mt5_connection():
                    logger.warning("⚠️ Conexão MT5 perdida. Tentando reconectar...")
                    if not utils.ensure_mt5_connection():
                        bot_state.pause_trading("MT5 desconectado")
                    else:
                        bot_state.resume_trading()

            # ✅ NOVO: Re-otimização ML
            if getattr(config, 'ENABLE_ML_OPTIMIZER', False) and hasattr(utils, 'ml_optimizer_instance'):
                reoptimize_interval = timedelta(hours=getattr(config, 'ML_RETRAIN_INTERVAL_HOURS', 24))
                if (datetime.now() - last_ml_reoptimization) > reoptimize_interval:
                    logger.info("⏳ Acionando re-otimização do ML Optimizer...")
                    utils.ml_optimizer_instance.check_and_reoptimize()
                    global ELITE_CONFIG # Recarrega a elite config após otimização
                    ELITE_CONFIG = load_elite_config()
                    last_ml_reoptimization = datetime.now()

            # ✅ NOVO: News Filter (placeholder)
            if getattr(config, 'ENABLE_NEWS_FILTER', False):
                # Implementar lógica de filtro de notícias aqui
                pass

            # ===========================
            # ✅ v5.2: KILL SWITCH MONITOR (Land Trading)
            # ===========================
            if getattr(config, 'ENABLE_KILL_SWITCH', True):
                min_wr = getattr(config, 'KILL_SWITCH_WIN_RATE', 0.50)
                pause_min = getattr(config, 'KILL_SWITCH_PAUSE_MINUTES', 60)
                
                for symbol in GLOBAL_ACTIVE_SYMBOLS:
                    # Ignora se já estiver pausado
                    if symbol in PAUSED_SYMBOLS:
                        continue
                        
                    wr, total, wins = utils.calculate_rolling_win_rate(symbol)
                    
                    # Atualiza tracker
                    KILL_SWITCH_TRACKER[symbol] = {
                        "win_rate": wr, 
                        "last_check": datetime.now()
                    }
                    
                    # Se WR estiver abaixo do limite (e tiver histórico suficiente)
                    # Nota: utils retorna 50% e 0 trades se não tiver histórico, então só pausa se > 0 trades reais
                    if total >= getattr(config, 'KILL_SWITCH_TRADES', 10) and wr < min_wr:
                        reason = f"Kill Switch: WR {wr:.1%} < {min_wr:.0%} ({wins}/{total} trades)"
                        pause_bot(symbol, pause_min, reason)

            watchdog.heartbeat("SlowLoop")
            
            # ✅ v5.3: METRICS LOGGING (a cada 1 hora)
            current_time_minutes = datetime.now().hour * 60 + datetime.now().minute
            if current_time_minutes % 60 == 0:  # A cada hora cheia
                utils.log_metrics_summary()
            try:
                utils.sync_mt5_trades_to_db()
            except Exception:
                pass
            
            try:
                br_now = utils.get_brasilia_time()
                wd = br_now.weekday()
                hm = br_now.strftime("%H:%M")
                br_date_str = br_now.strftime("%Y-%m-%d")
                try:
                    if getattr(config, "FRIDAY_AUTO_CLOSE_ENABLED", True):
                        cutoff = str(getattr(config, "FRIDAY_AUTO_CLOSE_BRT", "16:30"))
                        buffer_min = int(getattr(config, "FRIDAY_AUTO_CLOSE_BUFFER_MINUTES", 0))
                        if wd == 4 and hm >= cutoff:
                            if LAST_FRIDAY_AUTOCLOSE_DATE != br_date_str:
                                utils.close_all_positions()
                                try:
                                    utils.send_telegram_message(f"🛡️ Fechamento automático de sexta ({cutoff} BRT) executado.")
                                except Exception:
                                    pass
                                LAST_FRIDAY_AUTOCLOSE_DATE = br_date_str
                except Exception as _e:
                    logger.error(f"erro ao fechar automaticamente na sexta: {_e}")
                if wd == 4 and hm >= "19:00":
                    if LAST_FRIDAY_SNAPSHOT_DATE != br_date_str:
                        utils.run_weekly_snapshot()
                        LAST_FRIDAY_SNAPSHOT_DATE = br_date_str
            except Exception:
                logger.error("erro ao executar snapshot semanal")
            
            # ✅ AUTO-EXPORT DIÁRIO (CSV + TXT) via Telegram — Horário de Brasília
            try:
                br_now = utils.get_brasilia_time()
                br_date_str = br_now.strftime("%Y-%m-%d")
                hm = br_now.strftime("%H:%M")
                if LAST_DAILY_EXPORT_DATE != br_date_str and hm >= "23:55":
                    path_csv, summary_csv = utils.export_bot_trades_csv(br_date_str)
                    path_txt, summary_txt = utils.export_bot_trades_txt(br_date_str)
                    if path_csv:
                        caption_csv = (
                            f"<b>Trades (CSV)</b> ({summary_csv.get('date')})\n"
                            f"Total: {summary_csv.get('trades', 0)} | WR: {summary_csv.get('win_rate', 0):.1f}% | "
                            f"PF: {summary_csv.get('profit_factor', 0):.2f} | PnL: ${summary_csv.get('pnl', 0):+.2f}"
                        )
                        if not utils.send_telegram_document(path_csv, caption=caption_csv):
                            utils.send_telegram_message(caption_csv + f"\n\nArquivo: {path_csv}")
                    if path_txt:
                        caption_txt = (
                            f"<b>Trades (TXT)</b> ({summary_txt.get('date')})\n"
                            f"Total: {summary_txt.get('trades', 0)} | WR: {summary_txt.get('win_rate', 0):.1f}% | "
                            f"PF: {summary_txt.get('profit_factor', 0):.2f} | PnL: ${summary_txt.get('pnl', 0):+.2f}"
                        )
                        if not utils.send_telegram_document(path_txt, caption=caption_txt):
                            utils.send_telegram_message(caption_txt + f"\n\nArquivo: {path_txt}")
                    LAST_DAILY_EXPORT_DATE = br_date_str
            except Exception:
                pass
            
            time.sleep(getattr(config, 'SLOW_LOOP_INTERVAL', 300)) # ✅ CORREÇÃO: getattr para SLOW_LOOP_INTERVAL

        except Exception as e:
            logger.critical(f"💀 ERRO FATAL NA THREAD SlowLoop: {e}", exc_info=True)
            time.sleep(5)
            continue

def _telegram_api_request(method: str, payload: dict = None, timeout_s: int = 35) -> Optional[dict]:
    bot_token, _ = utils.get_telegram_credentials()
    if not bot_token:
        return None
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    try:
        import requests
        response = requests.post(url, json=payload or {}, timeout=timeout_s)
        if response.status_code != 200:
            return None
        data = response.json()
        if not isinstance(data, dict) or not data.get("ok"):
            return None
        return data
    except Exception:
        return None

def _format_positions_for_telegram(positions) -> str:
    if not positions:
        return "Sem posições abertas."
    lines = []
    for pos in positions:
        side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        lines.append(f"{pos.symbol} {side} {pos.volume} | PnL: {pos.profit:+.2f} | Ticket: {pos.ticket}")
    text = "\n".join(lines)
    if len(text) > 3500:
        text = text[:3500] + "\n..."
    return text

def _format_status_for_telegram(max_items: int = 25) -> str:
    items = []
    for sym, data in GLOBAL_MONITOR_CACHE.items():
        status = data.get("status", "N/A")
        reason = data.get("reason", "")
        ml = data.get("ml_score", 0)
        items.append((sym, f"{status}: {reason} (ML:{ml:.0f})"))
    items.sort(key=lambda x: x[0])
    if not items:
        return "Sem status disponível ainda (aguardando loop)."
    lines = [f"{sym}: {msg}" for sym, msg in items[:max_items]]
    if len(items) > max_items:
        lines.append(f"... (+{len(items) - max_items} ativos)")
    return "\n".join(lines)

def telegram_command_loop():
    _, chat_id_allowed = utils.get_telegram_credentials()
    chat_id_allowed = str(chat_id_allowed or "").strip()
    # Token é obrigatório; chat_id pode ser aprendido dinamicamente
    if not utils.get_telegram_credentials()[0]:
        return
    try:
        _ = _telegram_api_request("deleteWebhook", {"drop_pending_updates": True}, timeout_s=10)
    except Exception:
        pass

    def _send_telegram_long(text: str, chunk_size: int = 3500):
        remaining = text or ""
        while remaining:
            if len(remaining) <= chunk_size:
                utils.send_telegram_message(remaining)
                break
            cut = remaining.rfind("\n", 0, chunk_size)
            if cut < 500:
                cut = chunk_size
            part = remaining[:cut].rstrip()
            utils.send_telegram_message(part)
            remaining = remaining[cut:].lstrip()

    offset = 0
    try:
        bootstrap = _telegram_api_request("getUpdates", {"timeout": 0, "offset": 0}, timeout_s=10)
        if bootstrap and isinstance(bootstrap.get("result"), list) and bootstrap["result"]:
            offset = max(u.get("update_id", 0) for u in bootstrap["result"]) + 1
    except Exception:
        offset = 0

    utils.send_telegram_message(
        "🤖 <b>Comandos Telegram</b> ativos.\n"
        "Use /help para ver os comandos disponíveis."
    )

    while not shutdown_event.is_set():
        try:
            watchdog.heartbeat("Telegram")
            data = _telegram_api_request("getUpdates", {"timeout": 30, "offset": offset, "allowed_updates": ["message"]}, timeout_s=35)
            if not data:
                continue
            updates = data.get("result") or []
            if not isinstance(updates, list) or not updates:
                continue

            for upd in updates:
                offset = max(offset, upd.get("update_id", 0) + 1)
                msg = (upd.get("message") or {})
                text = (msg.get("text") or "").strip()
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", "")).strip()
                # Se não houver chat_id salvo ainda, vincula dinamicamente ao primeiro chat
                if not chat_id_allowed:
                    chat_id_allowed = chat_id
                    try:
                        utils.set_telegram_chat_id(chat_id_allowed)
                        utils.send_telegram_message_to(chat_id_allowed, "🤝 Bot vinculado a este chat. Use /help para os comandos.")
                    except Exception:
                        pass
                elif chat_id != chat_id_allowed:
                    if text.startswith("/start") or text.startswith("/help") or text.startswith("/ajuda"):
                        chat_id_allowed = chat_id
                        try:
                            utils.set_telegram_chat_id(chat_id_allowed)
                            utils.send_telegram_message_to(chat_id_allowed, "🤝 Bot vinculado a este chat. Use /help para os comandos.")
                        except Exception:
                            pass
                    else:
                        try:
                            utils.send_telegram_message_to(chat_id, "⚠️ Este chat não está vinculado ao bot.\nEnvie /start ou /help para vincular.")
                        except Exception:
                            pass
                        continue
                if not text.startswith("/"):
                    continue

                parts = text.split()
                cmd = parts[0].split("@")[0].lower()
                args = parts[1:]

                logger.info(f"📩 Telegram comando recebido: {text}")
                utils.send_telegram_message(f"✅ Comando recebido: {text}")

                if cmd in ["/help", "/ajuda"]:
                    utils.send_telegram_message(
                        "<b>Comandos</b>\n"
                        "/status - Status dos ativos\n"
                        "/metrics - Métricas do dia\n"
                        "/sessionmetrics - Métricas por sessão\n"
                        "/dayreport [YYYY-MM-DD] - Relatório do dia\n"
                        "/trades [YYYY-MM-DD] - Exporta trades do dia\n"
                        "/trades_txt [YYYY-MM-DD] - Exporta trades em TXT\n"
                        "/snapshot - Snapshot imediato da carteira\n"
                        "/positions - Posições abertas\n"
                        "/pause SYMBOL MIN [motivo] - Pausa símbolo\n"
                        "/resume SYMBOL - Retoma símbolo\n"
                        "/pauseall MIN [motivo] - Pausa bot\n"
                        "/resumeall - Retoma bot\n"
                        "/closeall - Fecha posições do bot\n"
                        "/killswitch - Fecha tudo e encerra"
                    )
                    continue

                if cmd == "/status":
                    utils.send_telegram_message(_format_status_for_telegram())
                    continue

                if cmd == "/metrics":
                    m = utils.calculate_current_metrics()
                    if not m:
                        utils.send_telegram_message("Sem métricas disponíveis agora.")
                        continue
                    utils.send_telegram_message(
                        "<b>Métricas (Hoje)</b>\n"
                        f"Trades: {m.get('trades_today', 0)}\n"
                        f"WR: {m.get('win_rate', 0):.1f}%\n"
                        f"PF: {m.get('profit_factor_today', 0):.2f}\n"
                        f"PnL: ${m.get('pnl_today', 0):+.2f}\n"
                        f"PnL/trade: ${m.get('avg_trade', 0):+.2f}\n"
                        f"DD intraday: {m.get('max_dd_intraday', 0):.2f}%"
                    )
                    continue

                if cmd == "/sessionmetrics":
                    try:
                        utils.send_telegram_message(utils.get_session_metrics_summary())
                    except Exception:
                        utils.send_telegram_message("Falha ao gerar métricas por sessão.")
                    continue

                if cmd == "/dayreport":
                    date_arg = args[0] if args else None
                    try:
                        report = utils.generate_day_report(date_arg)
                        _send_telegram_long(report)
                    except Exception:
                        utils.send_telegram_message("Falha ao gerar dayreport.")
                    continue

                if cmd in ["/trades", "/trade"]:
                    date_arg = args[0] if args else None
                    path, summary = utils.export_bot_trades_csv(date_arg)
                    if not path:
                        utils.send_telegram_message(f"Falha ao exportar trades: {summary.get('error', 'erro desconhecido')}")
                        continue
                    caption = (
                        f"<b>Trades</b> ({summary.get('date')})\n"
                        f"Total: {summary.get('trades', 0)} | WR: {summary.get('win_rate', 0):.1f}% | "
                        f"PF: {summary.get('profit_factor', 0):.2f} | PnL: ${summary.get('pnl', 0):+.2f}"
                    )
                    if not utils.send_telegram_document(path, caption=caption):
                        utils.send_telegram_message(caption + f"\n\nArquivo: {path}")
                    continue

                if cmd in ["/trades_txt", "/tradestxt"]:
                    date_arg = args[0] if args else None
                    path, summary = utils.export_bot_trades_txt(date_arg)
                    if not path:
                        utils.send_telegram_message(f"Falha ao exportar trades: {summary.get('error', 'erro desconhecido')}")
                        continue
                    caption = (
                        f"<b>Trades (TXT)</b> ({summary.get('date')})\n"
                        f"Total: {summary.get('trades', 0)} | WR: {summary.get('win_rate', 0):.1f}% | "
                        f"PF: {summary.get('profit_factor', 0):.2f} | PnL: ${summary.get('pnl', 0):+.2f}"
                    )
                    if not utils.send_telegram_document(path, caption=caption):
                        utils.send_telegram_message(caption + f"\n\nArquivo: {path}")
                    continue

                if cmd == "/positions":
                    positions = mt5_exec(mt5.positions_get)
                    utils.send_telegram_message("<b>Posições</b>\n" + _format_positions_for_telegram(positions))
                    continue
                
                if cmd == "/balance":
                    acc = mt5_exec(mt5.account_info)
                    if not acc:
                        utils.send_telegram_message("Conta indisponível no momento.")
                        continue
                    utils.send_telegram_message(
                        "<b>Finanças</b>\n"
                        f"Saldo: ${acc.balance:,.2f}\n"
                        f"Equity: ${acc.equity:,.2f}\n"
                        f"Margem Livre: ${acc.margin_free:,.2f}\n"
                        f"PnL Atual: ${acc.profit:+,.2f}"
                    )
                    continue
                
                if cmd in ["/history", "/hist"]:
                    date_arg = args[0] if args else None
                    path, summary = utils.export_bot_trades_txt(date_arg)
                    if not path:
                        utils.send_telegram_message(f"Falha ao obter histórico: {summary.get('error', 'erro desconhecido')}")
                        continue
                    caption = (
                        f"<b>Histórico</b> ({summary.get('date')})\n"
                        f"Total: {summary.get('trades', 0)} | WR: {summary.get('win_rate', 0):.1f}% | "
                        f"PF: {summary.get('profit_factor', 0):.2f} | PnL: ${summary.get('pnl', 0):+.2f}"
                    )
                    if not utils.send_telegram_document(path, caption=caption):
                        utils.send_telegram_message(caption + f"\n\nArquivo: {path}")
                    continue

                if cmd in ["/snapshot", "/snap"]:
                    try:
                        result = utils.run_weekly_snapshot()
                        utils.send_telegram_message(f"Snapshot executado.\nJSON: {result.get('paths', {}).get('json')}")
                    except Exception:
                        utils.send_telegram_message("Falha ao executar snapshot.")
                    continue
                
                if cmd in ["/adaptive_backtest", "/abtest"]:
                    try:
                        from risk_engine import adaptive_manager
                        window = int(args[0]) if args else getattr(config, "ADAPTIVE_BACKTEST_DEFAULT_WINDOW", 100)
                        res = adaptive_manager.backtest(window=window)
                        utils.send_telegram_message(
                            "<b>Adaptive TP/SL Backtest</b>\n"
                            f"Janela: {int(window)} trades\n"
                            f"PnL estático: ${res.get('pnl_static', 0.0):+.2f}\n"
                            f"PnL dinâmico (escala TP): ${res.get('pnl_dynamic_scaled', 0.0):+.2f}"
                        )
                    except Exception as e:
                        utils.send_telegram_message(f"Falha no backtest adaptativo: {e}")
                    continue
                
                if cmd == "/riskstatus":
                    try:
                        import sqlite3, time
                        from risk_engine import DB_PATH as _DB
                        if not os.path.exists(_DB):
                            utils.send_telegram_message("Sem registros de bloqueio.")
                        else:
                            conn = sqlite3.connect(_DB)
                            cur = conn.cursor()
                            now_ts = int(time.time())
                            cur.execute("SELECT symbol, reason, end_ts FROM blocks WHERE end_ts > ? ORDER BY end_ts ASC LIMIT 50", (now_ts,))
                            rows = cur.fetchall()
                            conn.close()
                            if not rows:
                                utils.send_telegram_message("Sem bloqueios ativos.")
                            else:
                                lines = ["<b>Bloqueios Ativos</b>"]
                                for s, r, e in rows:
                                    until = datetime.fromtimestamp(e).strftime("%H:%M")
                                    lines.append(f"{s}: {r} (até {until})")
                                utils.send_telegram_message("\n".join(lines))
                    except Exception:
                        utils.send_telegram_message("Falha ao consultar bloqueios.")
                    continue
                
                if cmd == "/riskunblock" and len(args) >= 1:
                    try:
                        import sqlite3
                        from risk_engine import DB_PATH as _DB
                        sym = args[0].upper()
                        if os.path.exists(_DB):
                            conn = sqlite3.connect(_DB)
                            cur = conn.cursor()
                            cur.execute("DELETE FROM blocks WHERE symbol = ?", (sym,))
                            conn.commit()
                            conn.close()
                        if sym in PAUSED_SYMBOLS:
                            del PAUSED_SYMBOLS[sym]
                        utils.send_telegram_message(f"Desbloqueado {sym}.")
                    except Exception:
                        utils.send_telegram_message("Falha ao desbloquear símbolo.")
                    continue

                if cmd == "/pause" and len(args) >= 2:
                    symbol = args[0].upper()
                    minutes = int(float(args[1]))
                    reason = " ".join(args[2:]).strip() or "Telegram"
                    pause_bot(symbol, minutes, reason)
                    utils.send_telegram_message(f"⏸️ Pausado {symbol} por {minutes} min. Motivo: {reason}")
                    continue

                if cmd == "/resume" and len(args) >= 1:
                    symbol = args[0].upper()
                    if symbol in PAUSED_SYMBOLS:
                        del PAUSED_SYMBOLS[symbol]
                    bot_state.update_monitoring(symbol, "🔵 MONITORANDO", "Retomado via Telegram", 0.0)
                    utils.send_telegram_message(f"▶️ Retomado {symbol}.")
                    continue

                if cmd == "/pauseall" and len(args) >= 1:
                    minutes = int(float(args[0]))
                    reason = " ".join(args[1:]).strip() or "Telegram"
                    bot_state.pause_trading(f"{reason} ({minutes} min)")
                    utils.send_telegram_message(f"⏸️ Bot pausado por {minutes} min. Motivo: {reason}")
                    seconds = max(1, min(minutes * 60, 12 * 3600))
                    def _auto_resume():
                        time.sleep(seconds)
                        bot_state.resume_trading()
                        utils.send_telegram_message("▶️ Bot retomado automaticamente.")
                    threading.Thread(target=_auto_resume, daemon=True).start()
                    continue

                if cmd == "/resumeall":
                    bot_state.resume_trading()
                    utils.send_telegram_message("▶️ Bot retomado.")
                    continue

                if cmd == "/closeall":
                    utils.close_all_positions()
                    utils.send_telegram_message("✅ Fechamento solicitado para todas as posições do bot.")
                    continue

                if cmd in ["/killswitch", "/shutdown"]:
                    ks = Path(getattr(config, 'KILLSWITCH_FILE', 'killswitch.txt'))
                    try:
                        ks.touch()
                    except Exception:
                        pass
                    utils.close_all_positions()
                    shutdown_event.set()
                    utils.send_telegram_message("🚨 Kill switch acionado. Fechando posições e encerrando.")
                    break

                utils.send_telegram_message("Comando não reconhecido. Use /help.")

        except Exception:
            time.sleep(2)

# ===========================
# PANEL
# ===========================
def render_panel_enhanced():
    """Painel visual com Rich"""
    if not RICH_AVAILABLE or not getattr(config, 'ENABLE_DASHBOARD', False): # ✅ CORREÇÃO: getattr para ENABLE_DASHBOARD
        return

    def generate_display() -> Layout:
        layout = Layout()

        # Header
        market = utils.get_market_status()
        acc = mt5_exec(mt5.account_info)

        # ✅ REQUISITO: Proteção do Painel (Loading State)
        monitoring_data = bot_state.get_monitoring_status()
        if not acc or not monitoring_data:
            return Layout(Panel("⏳ CARREGANDO DADOS... (Aguardando indicadores)", title="XP3 PRO", border_style="yellow"))

        header_text = Text()
        header_text.append("🚀 XP3 PRO FOREX v4.2\n", style="bold cyan") # ✅ CORREÇÃO: Versão
        
        # ✅ NOVO: Exibição de PnL no Topo
        profit_color = "green" if acc.profit >= 0 else "red"
        header_text.append("💰 PnL Atual: ", style="bold white")
        header_text.append(f"${acc.profit:+,.2f}\n", style=f"bold {profit_color}")
        
        header_text.append(f"{market['emoji']} {market['message']}\n", style=market.get('color', 'green'))
        header_text.append(f"� Equity: ${acc.equity:,.2f} | Balance: ${acc.balance:,.2f}\n", style="white")
        header_text.append(f"📊 Elite Config: {len(ELITE_CONFIG)} símbolos otimizados\n", style="yellow")

        is_paused, pause_reason = bot_state.is_paused()
        if is_paused:
            header_text.append(f"⏸️ BOT PAUSADO: {pause_reason}\n", style="bold red")

        # --- NEWS & RISK STATUS v5.0 ---
        # News Status (Usando USD como proxy para o status geral ou o primeiro do top_pairs)
        top_pairs = bot_state.get_top_pairs()
        news_sym = top_pairs[0] if top_pairs else "EURUSD"
        is_blackout, news_msg = news_filter.is_news_blackout(news_sym)
        news_color = "red" if is_blackout else "green"
        header_text.append(f"📰 Status Notícias ({news_sym}): ", style="white")
        header_text.append(f"{news_msg}\n", style=news_color)

        # Risk Status
        daily_loss = acc.equity - acc.balance
        # ===========================
        # HEADER (Real-Time PnL & Finances)
        # ===========================
        acc_info = mt5_exec(mt5.account_info)
        current_profit = acc_info.profit if acc_info else 0.0
        
        # Cores Financeiras (Solicitação do Usuário)
        profit_color = "green" if current_profit >= 0 else "red"
        
        # Equity Color logic: Cyan if >= Balance, else Yellow/Red logic
        balance = acc_info.balance if acc_info else 0.0
        equity = acc_info.equity if acc_info else 0.0
        margin_free = acc_info.margin_free if acc_info else 0.0
        
        equity_style = "bold cyan" if equity >= balance else "bold yellow" if equity >= (balance * 0.95) else "bold red"
        
        header_text = Text() # Reinicia texto
        header_text.append("💰 PnL: ", style="bold white")
        header_text.append(f"${current_profit:+,.2f}", style=f"bold {profit_color}")
        header_text.append(" | 🏦 Saldo: ", style="bold white")
        header_text.append(f"${balance:,.2f}", style="bold green") 
        header_text.append(" | 📈 Equity: ", style="bold white")
        header_text.append(f"${equity:,.2f}", style=equity_style)
        header_text.append(" | 💸 Margem Livre: ", style="bold white")
        header_text.append(f"${margin_free:,.2f}", style="bold white")
        
        try:
            basis = str(getattr(config, 'MAX_TOTAL_EXPOSURE_BASIS', 'balance')).lower()
            limit_mult = float(getattr(config, 'MAX_TOTAL_EXPOSURE_MULTIPLIER', 2.0))
            base_val = balance if basis == "balance" else equity
            authorized = base_val * limit_mult
            positions = mt5_exec(mt5.positions_get)
            def _estimate(symbol: str, volume: float) -> float:
                info = utils.get_symbol_info(symbol)
                if not info or volume <= 0:
                    return 0.0
                contract = float(getattr(info, 'trade_contract_size', 100000) or 100000)
                tick = mt5_exec(mt5.symbol_info_tick, symbol)
                price = float(getattr(tick, 'bid', 0.0) or 0.0) if tick else 0.0
                s = symbol.upper()
                if len(s) >= 6 and s[3:6] == "USD":
                    return contract * volume * (price if price > 0 else 1.0)
                if len(s) >= 6 and s[0:3] == "USD":
                    return contract * volume
                if ("XAUUSD" in s) or ("XAGUSD" in s) or ("US30" in s) or ("US500" in s) or ("NAS100" in s) or ("USTEC" in s) or ("USA500" in s):
                    return contract * volume * (price if price > 0 else 1.0)
                return contract * volume
            exposure = 0.0
            for p in (positions or []):
                exposure += _estimate(p.symbol, float(getattr(p, 'volume', 0.0) or 0.0))
            usage = (exposure / authorized) if authorized > 0 else 0.0
            warn_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_WARNING_PCT', 0.80))
            alert_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_ALERT_PCT', 0.95))
            usage_style = "bold green"
            usage_label = f"{usage*100:.0f}%"
            if usage >= alert_pct:
                usage_style = "bold red"
                usage_label = f"🚨 {usage*100:.0f}%"
            elif usage >= warn_pct:
                usage_style = "bold yellow"
                usage_label = f"⚠️ {usage*100:.0f}%"
            header_text.append("\n🛡️ Margem Autorizada: ", style="bold white")
            header_text.append(f"${authorized:,.0f}", style="bold cyan")
            header_text.append(" | Exposição: ", style="bold white")
            header_text.append(f"${exposure:,.0f}", style=usage_style)
            header_text.append(" | Uso: ", style="bold white")
            header_text.append(usage_label, style=usage_style)
        except Exception:
            pass
        
        # Verifica status global
        total_open_positions = len(mt5_exec(mt5.positions_get) or [])
        global_max = getattr(config, 'MAX_GLOBAL_ALGO_ORDERS', 3)
        
        if total_open_positions >= global_max:
            system_status = "[bold red]🚨 LIMIT REACHED[/bold red]"
        elif bot_state.is_paused()[0]:
            system_status = f"[yellow]⏸️ PAUSADO: {bot_state.is_paused()[1]}[/yellow]"
        else:
            system_status = "[bold green]⚡ ATIVO[/bold green]"

        # ✅ SESSION STATUS
        session = utils.get_current_trading_session()
        s_display = session['display']
        s_emoji = session['emoji']
        s_color = "bold yellow" if session['name'] == "GOLDEN" else "bold red" if session['name'] == "PROTECTION" else f"bold {session['color']}"

        header_text.append(f"\n📊 Ordens: {total_open_positions}/{global_max} | {system_status}")
        header_text.append(f" | 🕒 SESSÃO: ", style="bold white")
        header_text.append(f"{s_display} {s_emoji}", style=s_color)
        
        header_panel = Panel(header_text, style="bold white", border_style="blue")

        # Posições
        # ===========================
        # POSITIONS (Limit 10)
        # ===========================
        pos_table = Table(show_header=True, header_style="bold green")
        pos_table.add_column("Symbol")
        pos_table.add_column("Side")
        pos_table.add_column("Profit")
        pos_table.add_column("Pips")
        pos_table.add_column("SL") # ✅ NOVO
        pos_table.add_column("TP") # ✅ NOVO

        positions = mt5_exec(mt5.positions_get)
            
        if positions:
            # ✅ MEMORY CLEAN: Limita a 10 posições visuais
            for pos in list(positions)[-10:]:
                if pos.magic != getattr(config, 'MAGIC_NUMBER', 123456): # ✅ CORREÇÃO: getattr para MAGIC_NUMBER
                    continue

                symbol_info = utils.get_symbol_info(pos.symbol)
                symbol_info_pos = mt5_exec(mt5.symbol_info, pos.symbol)
                digits = symbol_info_pos.digits if symbol_info_pos else 5
                
                # ✅ CORREÇÃO: Pip Size Robusto
                if symbol_info_pos:
                    if digits == 3 or digits == 5:
                        pip_size = symbol_info_pos.point * 10
                    else:
                        pip_size = symbol_info_pos.point
                else:
                    pip_size = 0.0001 # Fallback inseguro mas evita crash

                if pip_size == 0: # Evita divisão por zero
                    pips = 0.0
                elif pos.type == mt5.POSITION_TYPE_BUY:
                    pips = (pos.price_current - pos.price_open) / pip_size
                    side = "🟢 BUY"
                else:
                    pips = (pos.price_open - pos.price_current) / pip_size
                    side = "🔴 SELL"

                profit_color = "green" if pos.profit >= 0 else "red"
                pips_color = "green" if pips >= 0 else "red"

                pos_table.add_row(
                    pos.symbol,
                    side,
                    f"[{profit_color}]${pos.profit:+.2f}[/{profit_color}]",
                    f"[{pips_color}]{pips:+.1f}[/{pips_color}]",
                    f"{pos.sl:.{digits}f}", # ✅ NOVO: Mostra SL/TP
                    f"{pos.tp:.{digits}f}"
                )
        else:
            pos_table.add_row("-", "-", "-", "[dim](Nenhuma)[/dim]", "-", "-")

        positions_panel = Panel(pos_table, title="💼 POSIÇÕES (Last 10)", border_style="green")

        # Análises
        analysis_table = Table(show_header=True, header_style="bold yellow")
        analysis_table.add_column("Hora")
        analysis_table.add_column("Par")
        analysis_table.add_column("Sinal")
        analysis_table.add_column("Score")
        analysis_table.add_column("Status")
        analysis_table.add_column("Preço") # ✅ NOVO: Adiciona preço

        with signal_history_lock:
            recent = list(signal_history)[-10:]

            for analysis in reversed(recent):
                # Land Trading: Filtro de histórico
                if not getattr(config, 'SHOW_REJECTED_SIGNALS_HISTORY', True) and analysis.rejected:
                    pass

                time_str = analysis.timestamp.strftime("%H:%M:%S")

                if analysis.signal == "BUY":
                    signal_display = "[green]🟢BUY[/green]"
                elif analysis.signal == "SELL":
                    signal_display = "[red]🔴SELL[/red]"
                else:
                    signal_display = "[dim]--[/dim]"

                score_color = "green" if analysis.score >= 60 else "white" if analysis.score >= 40 else "dim"
                score_display = f"[{score_color}]{analysis.score:.0f}[/{score_color}]"

                reason = analysis.rejection_reason
                status_color = "bold green" if "EXECUTADA" in reason else "yellow" if "Spread Alto" in reason else "dim"

                # ✅ NOVO: Preço de fechamento da análise
                close_price_display = f"{analysis.indicators.get('close', 0):.5f}" if analysis.indicators.get('close', 0) > 0 else "[dim]N/A[/dim]"

                analysis_table.add_row(
                    f"[dim]{time_str}[/dim]",
                    analysis.symbol,
                    signal_display,
                    score_display,
                    f"[{status_color}]{reason[:50]}[/{status_color}]",
                    close_price_display
                )

        analysis_panel = Panel(analysis_table, title="🔍 ÚLTIMOS SINAIS", border_style="yellow")

        # MONITORAMENTO GLOBAL v5.1 (Land Trading)
        monitor_table = Table(box=box.MINIMAL, expand=True, border_style="bright_black")
        monitor_table.add_column("SYMBOL", style="cyan", width=12)
        monitor_table.add_column("STATUS", width=18)
        monitor_table.add_column("MOTIVO / TÉCNICA", style="white") # Renomeado para refletir conteúdo
        monitor_table.add_column("SPR", justify="right", width=5)   # ✅ NOVO COLUNA SPREAD
        monitor_table.add_column("ML", justify="right", width=6)

        # Usa o GLOBAL_MONITOR_CACHE solicitado pelo usuário
        for sym in sorted(GLOBAL_MONITOR_CACHE.keys()):
            m = GLOBAL_MONITOR_CACHE[sym]
            status = m['status']
            ml = m['ml_score']
            reason_text = m['reason']
            
            # 1. Cor pelo Status
            s_style = "white"
            if "SINAL" in status: s_style = "green"
            elif "OPERANDO" in status: s_style = "bright_blue"
            elif "BLOQUEADO" in status: s_style = "yellow"
            elif "MONITORANDO" in status: s_style = "blue"
            elif "ERRO" in status: s_style = "bold red"

            # 2. Cor pelo ML e Suffix de Baseline
            ml_style = "green" if ml > 60 else "bright_black" if ml < 40 else "white"
            ml_text = f"{ml:.0f}"
            if m.get("is_baseline", False):
                ml_text += " [dim](B)[/dim]" # (B) para Confiança Estatística (Backtest)
                ml_style = "yellow" if ml > 60 else ml_style # Amarelo indica "Estatística"
            
            # 3. Cor pelo Motivo
            reason_style = "yellow" if "Spread Alto" in reason_text else "white"
            if "❌" in reason_text: reason_style = "red" # Destaque para Vetos

            # 4. Spread (Busca do Cache ou Recalcula)
            spread_val = m.get('spread', 0)
            
            # Recalcula limite para validar cor (Lógica duplicada do fast_loop para display)
            symbol_upper = sym.upper()
            if any(idx in symbol_upper for idx in ["US30", "NAS100", "USTEC", "DE40", "GER40", "UK100", "US500", "USA500"]):
                max_spread = getattr(config, 'MAX_SPREAD_INDICES', 150)
            elif any(met in symbol_upper for met in ["XAU", "XAG", "GOLD", "SILVER"]):
                max_spread = getattr(config, 'MAX_SPREAD_METALS', 60)
            elif any(crypto in symbol_upper for crypto in ["BTC", "ETH", "SOL", "ADA", "BNB"]):
                max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
            else:
                max_spread = getattr(config, 'MAX_SPREAD_FOREX', 25)

            spread_style = "bold red" if spread_val > max_spread else "dim" # ✅ RED if above limit
            spread_display = str(spread_val)

            monitor_table.add_row(
                sym,
                Text(status, style=s_style),
                Text(reason_text[:75], style=reason_style), 
                Text(spread_display, style=spread_style), # ✅ Applied Style
                Text.from_markup(f"[{ml_style}]{ml_text}[/{ml_style}]")
            )
        
        monitor_panel = Panel(monitor_table, title="🛰️ LAND TRADING - MONITORAMENTO EM TEMPO REAL", border_style="blue")

        # Layout
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body", ratio=3)
        )

        layout["header"].update(header_panel)

        layout["body"].split_column(
            Layout(name="monitoring", ratio=2),
            Layout(name="analysis", ratio=1),
            Layout(name="positions", ratio=1)
        )

        layout["body"]["monitoring"].update(monitor_panel)
        layout["body"]["analysis"].update(analysis_panel)
        layout["body"]["positions"].update(positions_panel)

        return layout

    try:
        with Live(
            generate_display(),
            console=console,
            screen=True,
            refresh_per_second=1,
            auto_refresh=False
        ) as live:
            while not shutdown_event.is_set():
                try:
                    watchdog.heartbeat("Panel")
                    live.update(generate_display(), refresh=True)
                    panel_interval = max(getattr(config, 'DASHBOARD_REFRESH_RATE', 1), 3)
                    time.sleep(panel_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.critical(f"💀 ERRO FATAL NA THREAD Panel: {e}", exc_info=True)
                    time.sleep(5)
                
    except Exception as e:
        logger.error(f"❌ Painel falhou: {e}", exc_info=True)

# ===========================
# SIGNAL HANDLING (v5.0.7)
# ===========================
def handle_exit(sig, frame):
    """Garante fechamento limpo solicitado pelo sistema ou usuário."""
    logger.info(f"🛑 Fechamento limpo solicitado pelo sistema (Sinal {sig}).")
    shutdown_event.set()

# Configura sinais de encerramento
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ===========================
# MAIN
# ===========================
def main():
    print("="*80)
    print("🚀 XP3 PRO FOREX BOT v4.2") # ✅ CORREÇÃO: Versão
    print("="*80)

    try:
        Path("bot_heartbeat.timestamp").touch()
    except:
        pass

    if not mt5_exec(mt5.initialize, path=getattr(config, 'MT5_TERMINAL_PATH', None)):
        logger.critical(f"❌ Falha ao inicializar MT5 em: {getattr(config, 'MT5_TERMINAL_PATH', 'Caminho não especificado')}")
        return
    
    sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
    sector_map = getattr(config, "SECTOR_MAP", None)
    allowed = None
    if isinstance(sector_map, dict) and sector in sector_map:
        allowed = list(sector_map.get(sector) or [])
    else:
        allowed = list(getattr(config, "SYMBOL_MAP", []) or [])

    sync_result = utils.sync_market_watch(allowed)
    
    # ✅ REQUISITO: Log de Versão MT5 para Diagnóstico
    try:
        ver = mt5_exec(mt5.version)
        print(f"✅ MT5 Terminal Version: {ver}")
        logger.info(f"📊 MT5 Terminal Version: {ver}")
    except:
        pass

    account_info = mt5_exec(mt5.account_info)
    if not account_info:
        logger.critical("❌ Não foi possível obter informações da conta")
        mt5_exec(mt5.shutdown)
        return

    print(f"✅ Conta: {account_info.login} | Balance: ${account_info.balance:,.2f}")

    # ✅ NOVO: Inicializa ML Optimizer e o anexa ao utils para acesso global
    if getattr(config, 'ENABLE_ML_OPTIMIZER', False):
        try:
            from ml_optimizer import EnsembleOptimizer
            utils.ml_optimizer_instance = EnsembleOptimizer()
            logger.info("✅ ML Optimizer inicializado e anexado ao utils.")
        except ImportError as e:
            logger.error(f"❌ Não foi possível importar ML Optimizer: {e}. Desabilitando ML Optimizer.")
            config.ENABLE_ML_OPTIMIZER = False # Desabilita se não puder importar
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar ML Optimizer: {e}. Desabilitando ML Optimizer.", exc_info=True)
            config.ENABLE_ML_OPTIMIZER = False

    # ✅ REQUISITO: Sincronização e Filtragem de Ativos
    
    global GLOBAL_ACTIVE_SYMBOLS
    kept = (sync_result or {}).get("kept") if isinstance(sync_result, dict) else None
    GLOBAL_ACTIVE_SYMBOLS = kept if isinstance(kept, list) and kept else filter_and_validate_symbols()

    if getattr(config, "ENABLE_DYNAMIC_ASSET_SELECTION", True):
        try:
            sel_path = Path(getattr(config, "DATA_DIR", "data")) / "selected_assets.json"
            if sel_path.exists():
                payload = json.loads(sel_path.read_text(encoding="utf-8"))
                desired = list(payload.get("symbols", []))
                if desired:
                    desired_set = set(desired)
                    GLOBAL_ACTIVE_SYMBOLS = [s for s in GLOBAL_ACTIVE_SYMBOLS if s in desired_set]
        except Exception as e:
            logger.warning(f"⚠️ Falha ao aplicar seleção dinâmica de ativos: {e}")
    
    if not GLOBAL_ACTIVE_SYMBOLS:
        logger.critical("🚨 [CRITICAL] Nenhum símbolo da configuração foi encontrado no seu Terminal MT5. Verifique os nomes no Market Watch.")
        mt5_exec(mt5.shutdown)
        return

    logger.info(f"📊 Sessão iniciada com {len(GLOBAL_ACTIVE_SYMBOLS)} ativos operacionais.")

    # ✅ NOVO: Teste de Telegram no Boot
    if utils.get_telegram_credentials()[0]:
        utils.send_telegram_message("🚀 <b>XP3 PRO FOREX</b> Iniciado com Sucesso!\n\nPronto para operar sob regras da <i>Land Trading</i>.")

    # Registra threads
    watchdog.register_thread("FastLoop", fast_loop)
    watchdog.register_thread("SlowLoop", slow_loop)
    if utils.get_telegram_credentials()[0] and getattr(config, 'ENABLE_TELEGRAM_COMMANDS', True):
        watchdog.register_thread("Telegram", telegram_command_loop)

    # ✅ CORREÇÃO: Verifica ENABLE_DASHBOARD e lança via Streamlit (Headless Bot)
    if getattr(config, 'ENABLE_DASHBOARD', False):
        # watchdog.register_thread("Panel", render_panel_enhanced) # ❌ Disabled Render Panel
        try:
             # Lança o dashboard em processo separado na porta 8502
             cmd = ["streamlit", "run", "dashboard.py", "--server.port", "8502", "--server.headless", "true"]
             # subprocess.Popen(cmd, cwd=os.getcwd(), shell=True) # shell=True pode ser problemático com sinais, mas ok para Windows
             # Para evitar bloquear, usamos Popen.
             # No Windows, shell=True ajuda a resolver caminhos.
             subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), shell=True)
             logger.info("✅ Dashboard iniciado no navegador (Porta 8502)")
        except Exception as e:
             logger.error(f"❌ Falha ao iniciar dashboard: {e}")

    # ✅ REQUISITO: Sincronização de Nomes

    # Inicia threads
    threads = []
    for name, info in watchdog.threads.items():
        thread = threading.Thread(
            target=info["target"],
            args=info["args"],
            daemon=True,
            name=name
        )
        thread.start()
        threads.append(thread)
        print(f"✅ Thread '{name}' iniciada")
    hb = threading.Thread(target=heartbeat_writer, daemon=True, name="HeartbeatWriter")
    hb.start()
    threads.append(hb)

    print("="*80)
    print("📊 Sistema ativo")

    try:
        while not shutdown_event.is_set():
            watchdog.check_and_restart()
            time.sleep(getattr(config, 'WATCHDOG_CHECK_INTERVAL', 60))
    except KeyboardInterrupt:
        print("\n⏹️ Encerrando...")
    finally:
        shutdown_event.set()
        # Espera as threads terminarem
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=5) # Dá um tempo para a thread terminar
        mt5_exec(mt5.shutdown)
        print("👋 Shutdown completo")

def start():
    print("="*80)
    print("🚀 XP3 PRO FOREX BOT v4.2 (Linear Mode)")
    print("="*80)
    try:
        Path("bot_heartbeat.timestamp").touch()
    except:
        pass
    if not mt5_exec(mt5.initialize, path=getattr(config, 'MT5_TERMINAL_PATH', None)):
        logger.critical(f"❌ Falha ao inicializar MT5 em: {getattr(config, 'MT5_TERMINAL_PATH', 'Caminho não especificado')}")
        return
    try:
        ver = mt5_exec(mt5.version)
        logger.info(f"📊 MT5 Terminal Version: {ver}")
    except:
        pass
    account_info = mt5_exec(mt5.account_info)
    if not account_info:
        logger.critical("❌ Não foi possível obter informações da conta")
        mt5_exec(mt5.shutdown)
        return
    print(f"✅ Conta: {account_info.login} | Balance: ${account_info.balance:,.2f}")
    if getattr(config, 'ENABLE_ML_OPTIMIZER', False):
        try:
            from ml_optimizer import EnsembleOptimizer
            utils.ml_optimizer_instance = EnsembleOptimizer()
            logger.info("✅ ML Optimizer inicializado e anexado ao utils.")
        except Exception as e:
            logger.error(f"❌ Falha ao inicializar ML Optimizer: {e}")
            config.ENABLE_ML_OPTIMIZER = False
    sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
    sector_map = getattr(config, "SECTOR_MAP", None)
    allowed = list(sector_map.get(sector) or []) if isinstance(sector_map, dict) and sector in sector_map else list(getattr(config, "SYMBOL_MAP", []) or [])
    sync_result = utils.sync_market_watch(allowed)
    global GLOBAL_ACTIVE_SYMBOLS
    kept = (sync_result or {}).get("kept") if isinstance(sync_result, dict) else None
    GLOBAL_ACTIVE_SYMBOLS = kept if isinstance(kept, list) and kept else filter_and_validate_symbols()
    if not GLOBAL_ACTIVE_SYMBOLS:
        logger.critical("🚨 [CRITICAL] Nenhum símbolo válido encontrado no MT5.")
        mt5_exec(mt5.shutdown)
        return
    logger.info(f"📊 Sessão iniciada com {len(GLOBAL_ACTIVE_SYMBOLS)} ativos operacionais (Linear).")
    try:
        fast_loop()  # Executa loop único e linear
    except KeyboardInterrupt:
        logger.info("🛑 Encerrando Linear Mode...")
    finally:
        shutdown_event.set()
        try:
            mt5_exec(mt5.shutdown)
        except:
            pass

if __name__ == "__main__":
    try:
        start()
    except Exception as e:
        logger.critical(f"💀 Erro fatal na inicialização (Linear): {e}", exc_info=True)
        try:
            mt5_exec(mt5.shutdown)
        except:
            pass
        time.sleep(5)
        sys.exit(1)
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
