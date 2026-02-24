# config_forex.py - Configurações Institucionais XP3 PRO v4.2
"""
🚀 XP3 PRO FOREX - CONFIGURAÇÕES INSTITUCIONAIS
✅ Gestão de risco avançada
✅ Parâmetros otimizados por ativo
✅ Integração com otimizador Optuna
✅ Adição de constantes para evitar AttributeError
"""

import os
from pathlib import Path

# ===========================
# CAMINHOS E DIRETÓRIOS
# ===========================
MT5_TERMINAL_PATH = os.environ.get(
    "XP3_MT5_TERMINAL_PATH",
    r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe"
)
OPTIMIZER_OUTPUT = Path("optimizer_output")
LOGS_DIR = Path("logs")
DATA_DIR = Path("data")

os.makedirs(OPTIMIZER_OUTPUT, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===========================
# MODO DE OPERAÇÃO
# ===========================
SERVER_OFFSET = 0             # Diferença Horária: Servidor -> Brasília (Ex: Se Servidor é UTC+2 e Brasília UTC-3, Offset = -5)
# --- SESSÕES DE TRADING (Horário de Brasília) ---
GOLDEN_HOUR_START = "10:00"
GOLDEN_HOUR_END = "14:00"
NORMAL_SESSION_1_START = "05:15"
NORMAL_SESSION_1_END = "09:59"
NORMAL_SESSION_2_START = "14:01"
NORMAL_SESSION_2_END = "17:59"
PROTECTION_SESSION_START = "18:00"
PROTECTION_SESSION_END = "22:00"
ASIAN_SESSION_START = "22:00"
ASIAN_SESSION_END = "05:00"

# ===========================
# TOKENS DE CLASSES DE ATIVOS
# ===========================
TOKENS_CRYPTO = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]
TOKENS_INDICES = ["US30", "NAS100", "USTEC", "DE40", "GER40", "GER30", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"]
TOKENS_METALS = ["XAU", "XAG", "GOLD", "SILVER"]
TOKENS_EXOTICS = ["TRY", "ZAR", "MXN", "RUB", "CNH", "PLN", "HUF", "CZK", "DKK", "NOK", "SEK"]

# ===========================
# EXECUÇÃO: LIMITES DE LATÊNCIA/SLIPPAGE
# ===========================
MAX_EXECUTION_LATENCY_MS = int(os.environ.get("XP3_MAX_EXEC_LATENCY_MS", "2000"))
MAX_EXECUTION_SLIPPAGE_PIPS = float(os.environ.get("XP3_MAX_EXEC_SLIPPAGE_PIPS", "5.0"))
EXECUTION_PAUSE_MINUTES = int(os.environ.get("XP3_EXEC_PAUSE_MINUTES", "10"))
SLIPPAGE_BASE_PIPS = float(os.environ.get("XP3_SLIPPAGE_BASE_PIPS", "0.2"))
SLIPPAGE_ATR_FACTOR = float(os.environ.get("XP3_SLIPPAGE_ATR_FACTOR", "0.6"))
SLIPPAGE_VOLUME_FACTOR = float(os.environ.get("XP3_SLIPPAGE_VOLUME_FACTOR", "0.5"))
SLIPPAGE_ORDER_FACTOR = float(os.environ.get("XP3_SLIPPAGE_ORDER_FACTOR", "0.2"))
ASIAN_SLIPPAGE_MULT = float(os.environ.get("XP3_ASIAN_SLIPPAGE_MULT", "1.4"))
GOLDEN_SLIPPAGE_MULT = float(os.environ.get("XP3_GOLDEN_SLIPPAGE_MULT", "0.9"))
NORMAL_SLIPPAGE_MULT = float(os.environ.get("XP3_NORMAL_SLIPPAGE_MULT", "1.0"))
PROTECTION_SLIPPAGE_MULT = float(os.environ.get("XP3_PROTECTION_SLIPPAGE_MULT", "1.8"))
NEWS_IMPACT_SLIPPAGE_MULTIPLIER = float(os.environ.get("XP3_NEWS_SLIPPAGE_MULT", "1.8"))
NEWS_IMPACT_SPREAD_MULTIPLIER = float(os.environ.get("XP3_NEWS_SPREAD_MULT", "1.5"))
NEWS_IMPACT_SL_MULTIPLIER = float(os.environ.get("XP3_NEWS_SL_MULT", "1.2"))
NEWS_IMPACT_TP_MULTIPLIER = float(os.environ.get("XP3_NEWS_TP_MULT", "0.9"))

# ===========================
# PORTFÓLIO: LIMITES DE EXPOSIÇÃO POR MOEDA
# ===========================
MAX_CURRENCY_EXPOSURE_PCT = 0.03
ASSUMED_LEVERAGE = 30.0
MAX_CURRENCY_EXPOSURE_PCT_MAP = {
    "USD": 0.03,
    "EUR": 0.03,
    "JPY": 0.03,
    "GBP": 0.03,
    "AUD": 0.03,
    "CAD": 0.03,
    "CHF": 0.03,
    "NZD": 0.03
}
CORR_EXPOSURE_TIGHTEN_FACTOR = 0.5

# --- REGRAS DINÂMICAS POR SESSÃO ---
GOLDEN_ML_SCORE_THRESHOLD = 40
NORMAL_ML_SCORE_THRESHOLD = 50
ASIAN_ML_SCORE_THRESHOLD = 60
GOLDEN_VOLUME_REDUCTION_PCT = 0.30  # Reduz exigência de volume em 30%
GOLDEN_SPREAD_ALLOWANCE_PCT = 0.20 # Permite Spread 20% acima da média
PROTECTION_MAX_SPREAD_FOREX = 20   # Spread máximo reduzido (pips)
ASIAN_MAX_SPREAD_POINTS = 50       # Spread máximo na Ásia (5 pips / 50 points)
ASIAN_RSI_LOW = 25
ASIAN_RSI_HIGH = 75
ASIAN_PRIORITY_PAIRS = ["USDJPY", "AUDJPY", "GBPJPY", "AUDUSD"]
PAPER_TRADING_MODE = False
ENABLE_DASHBOARD = True
ENABLE_ML_OPTIMIZER = True
ENABLE_NEWS_FILTER = True
ENABLE_DYNAMIC_ASSET_SELECTION = True
ENABLE_TELEGRAM_COMMANDS = True
TELEGRAM_USE_WEBHOOK = False

MIN_DAILY_LIQUIDITY_BRL = 1000000.0
MIN_VOL_ANNUALIZED = 0.05
MAX_VOL_ANNUALIZED = 0.20
MIN_PRESENCE_RATIO = 0.80

TARGET_ASSETS_MIN = 15
TARGET_ASSETS_MAX = 25
MIN_ASSETS_PER_DAY = 5
AVAILABILITY_LOOKBACK_DAYS = 10
AVAILABILITY_DAY_MODE = "BUSINESS"
WEEKLY_MAX_TURNOVER_PCT = 0.30
REBALANCE_WEEKDAY = 0

CORR_LOOKBACK_DAYS = 30
MAX_PAIRWISE_CORR = 0.75
CORR_RELAX_STEP = 0.05

RANK_WEIGHTS = {"performance": 0.30, "liquidity": 0.25, "volatility": 0.20, "correlation": 0.15, "cost": 0.10}

USD_BRL = 5.0
LIQUIDITY_CURRENCY = "USD"
LIQUIDITY_USE_TICK_VOLUME = True
LIQUIDITY_UNITS_PER_TICK = 1000.0
LIQUIDITY_UNITS_PER_REAL_VOLUME = 1.0
HEARTBEAT_WRITE_INTERVAL = 3
WATCHDOG_CHECK_INTERVAL = 45
PRESENCE_EXPECTED_DAYS_MODE = "BUSINESS"
ENABLE_DAILY_MARKET_ANALYSIS = True
DAILY_ANALYSIS_FILE = 'daily_selected_pairs.json'
DAILY_ANALYSIS_MAX_AGE_HOURS = 24
DAILY_ANALYSIS_MIN_PAIRS = 3
ASSET_CLASS_OVERRIDES = {}
ASSET_CLASS_RULES_DEFAULT = {
    "min_daily_liquidity_brl": MIN_DAILY_LIQUIDITY_BRL,
    "min_vol_annualized": MIN_VOL_ANNUALIZED,
    "max_vol_annualized": MAX_VOL_ANNUALIZED,
    "min_presence_ratio": MIN_PRESENCE_RATIO,
    "presence_mode": PRESENCE_EXPECTED_DAYS_MODE,
}

ASSET_CLASS_RULES = {
    "FX": {
        "min_daily_liquidity_brl": 1_000_000.0,
        "min_vol_annualized": 0.05,
        "max_vol_annualized": 0.30,
        "min_presence_ratio": 0.80,
        "presence_mode": "BUSINESS",
    },
    "INDICES": {
        "min_daily_liquidity_brl": 1_000_000.0,
        "min_vol_annualized": 0.08,
        "max_vol_annualized": 0.60,
        "min_presence_ratio": 0.80,
        "presence_mode": "BUSINESS",
    },
    "METALS": {
        "min_daily_liquidity_brl": 1_000_000.0,
        "min_vol_annualized": 0.08,
        "max_vol_annualized": 0.70,
        "min_presence_ratio": 0.80,
        "presence_mode": "BUSINESS",
    },
    "CRYPTO": {
        "min_daily_liquidity_brl": 200_000.0,
        "min_vol_annualized": 0.20,
        "max_vol_annualized": 2.50,
        "min_presence_ratio": 0.80,
        "presence_mode": "BUSINESS",
    },
    "EQUITIES": {
        "min_daily_liquidity_brl": 1_000_000.0,
        "min_vol_annualized": 0.10,
        "max_vol_annualized": 0.60,
        "min_presence_ratio": 0.80,
        "presence_mode": "BUSINESS",
    },
}

ASSET_CLASS_TARGETS = {
    "FX": {"min": 8, "max": 15},
    "INDICES": {"min": 3, "max": 8},
    "METALS": {"min": 2, "max": 6},
    "CRYPTO": {"min": 2, "max": 6},
    "EQUITIES": {"min": 0, "max": 2},
}

# --- UX SETTINGS (NEGATIVE EDGE REPORTING) ---
SHOW_ALL_SYMBOLS_IN_DASHBOARD = True
UPDATE_STATUS_INTERVAL = 30     # Segundos para atualizar motivos de não compra
DASHBOARD_REFRESH_RATE = 1      # Segundos para atualizar a tela
SHOW_REJECTED_SIGNALS_HISTORY = True # Mostrar sinais negados no histórico

# --- NEWS FILTER SETTINGS ---
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_BLOCK_MINUTES_BEFORE = 60
NEWS_BLOCK_MINUTES_AFTER = 60
# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("XP3_TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("XP3_TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_CREDENTIALS_FILE = "data/telegram.json"
# --- EQUITY GUARD (RISK MANAGEMENT) ---
MAX_DAILY_LOSS_PCT = 0.02       # Para de operar se perder 2% no dia
REDUCE_RISK_ON_DD = True        # Se DD > 1%, reduz lote pela metade
EQUITY_PROTECTION_LEVEL = 0.90  # Hard stop se conta cair 10% do saldo inicial

# ===========================
# GESTÃO DE RISCO v5.3 - INSTITUCIONAL
# ===========================
MAX_DAILY_DRAWDOWN_PCT = 0.02       # ✅ v5.0: 2% de perda máxima diária (Circuit Breaker)
MAX_WEEKLY_DRAWDOWN_PCT = 0.05      # ✅ v5.3: 5% de perda máxima semanal (Hard Stop)
INSTITUTIONAL_KILL_SWITCH_PCT = 0.10 # ✅ v5.3: 10% Drawdown Total (Stop Global)
DAILY_DRAWDOWN_RESET_HOUR = 0       # ✅ v5.0: Hora UTC para reset do contador
RISK_MANAGER_FILE = "data/risk_manager.json" # Cache de risco semanal

# --- EXPOSIÇÃO INSTITUCIONAL ---
MAX_CURRENCY_EXPOSURE_PCT = 0.03    # Máximo 3% de exposição por moeda (USD, JPY, EUR)
MAX_ACCOUNT_LEVERAGE = 30           # Bloqueio de alavancagem implícita acima de 1:30

# ===========================
# RISK BLOCKS & PROFIT OPTIMIZER
# ===========================
ENABLE_RISK_BLOCKS = True
BLOCK_LOSS_PCT_THRESHOLD = 0.03
BLOCK_LOSS_ABS_THRESHOLD = 50.0
BLOCK_CONSECUTIVE_LOSSES = 3
BLOCK_DURATION_MINUTES = 120

PROFIT_OPTIMIZER_ENABLE = True
TRAILING_STOP_ACTIVATION_PIPS = 12
TRAILING_STOP_DISTANCE_PIPS = 9
TRAILING_STOP_STEP_PIPS = 4
TP_LEVELS_PIPS = [15, 30]
TP_PARTIAL_RATIOS = [0.40, 0.35]
BREAKEVEN_PIPS = 12
TP_FIXED_PIPS = 60
DYNAMIC_TP_SL_ENABLE = True
TP_ADJUST_STEP = 0.3
SL_ADJUST_STEP = 0.2
MIN_TP_ATR_MULT = 2.0
MAX_TP_ATR_MULT = 6.0
MIN_SL_ATR_MULT = 1.0
MAX_SL_ATR_MULT = 3.5
PERFORMANCE_WINDOW_TRADES = 20
WIN_RATE_THRESHOLD_UP = 0.55
PROFIT_FACTOR_THRESHOLD_UP = 1.3
WIN_RATE_THRESHOLD_DOWN = 0.45
PROFIT_FACTOR_THRESHOLD_DOWN = 1.0
ADAPTIVE_BACKTEST_DEFAULT_WINDOW = 100

# ===========================
# ROLLOVER PROTECTION v5.0
# ===========================
ROLLOVER_BLOCK_START = "16:55"      # ✅ v5.0: Horário NY (UTC-5) - Início do bloqueio
ROLLOVER_BLOCK_END = "18:05"        # ✅ v5.0: Horário NY (UTC-5) - Fim do bloqueio
ENABLE_ROLLOVER_BLOCK = True        # ✅ v5.0: Habilita bloqueio de rollover

# ===========================
# EMA 200 MACRO FILTER v5.0
# ===========================
EMA_200_PERIOD = 100
ENABLE_EMA_200_FILTER = True
EMA_200_PULLBACK_ATR_TOLERANCE = 2.0
EMA_200_ALLOW_PULLBACK = False
# ===========================
# ADX STRENGTH FILTER v5.0
# ===========================
ADX_MIN_STRENGTH = 28
ADX_THRESHOLD = 20
ADX_MIN_STRENGTH_BY_SYMBOL = {
    "XAUUSD": 28,
    "XAGUSD": 22,
    "EURUSD": 24,
    "GBPUSD": 26,
    "USDJPY": 25
}

# ===========================
# WIN RATE OPTIMIZATION v5.1 (Land Trading)
# ===========================
MIN_WIN_RATE_THRESHOLD = 0.60       # ✅ v5.1: Só opera se baseline >60%
CANDLE_CONFIRMATION_REQUIRED = True
VOLATILITY_FILTER_MIN_ATR = 0.0008

# ===========================
# TRAILING STOP AVANÇADO v5.1
# ===========================
TRAILING_STOP_ACTIVATION_PIPS = 12

# ===========================
# CORRELATION FILTER v5.1
# ===========================
CORRELATION_MAX = 0.80              # ✅ v5.1: Máximo correlação permitida
CORRELATION_MAX = 0.75

# ===========================
# SIMULATION MODE v5.1
# ===========================
SIMULATION_MODE = False             # ✅ v5.1: Trades virtuais sem MT5

# ===========================
# MT5 CALENDAR v5.1
# ===========================
MT5_CALENDAR_JSON_PATH = "data/mt5_calendar.json"  # ✅ v5.1: Caminho do JSON exportado pelo MQL5
MT5_CALENDAR_REFRESH_MINUTES = 30                  # ✅ v5.1: Refresh a cada 30min
USE_MT5_CALENDAR = True                            # ✅ v5.1: Usar calendário MT5 em vez de ForexFactory

# ===========================
# MULTI-TIMEFRAME v5.2
# ===========================
ENABLE_MULTI_TIMEFRAME = True
MULTI_TF_CONFIRMATION = "H4"
MULTI_TF_EMA_PERIOD = 200
MULTI_TF_EMA_STRICT = True          # sem tolerância de pullback

# ===========================
# KILL SWITCH v5.2
# ===========================
ENABLE_KILL_SWITCH = True           # ✅ v5.2: Ativa kill switch automático
KILL_SWITCH_WIN_RATE = 0.45         # ✅ v5.2: Pausa se WR < 45%
KILL_SWITCH_TRADES = 10             # ✅ v5.2: Mínimo de trades para avaliar
KILL_SWITCH_PAUSE_MINUTES = 60      # ✅ v5.2: Tempo de pausa em minutos
KILLSWITCH_TTL_SECONDS = 900

# ===========================
# DIAGNÓSTICO DE COMPRA (VISIBILIDADE)
# ===========================
ENABLE_BUY_DIAGNOSTIC = True
BUY_DIAGNOSTIC_INTERVAL_SECONDS = 90
BUY_DIAGNOSTIC_MAX_SYMBOLS = 30

# ===========================
# OTIMIZAÇÃO (OPTUNA / GENETIC) - v6.0
# ===========================
OPTUNA_N_TRIALS = 200               # ✅ v6.0: Mais exploração (era 60)
OPTUNA_TIMEOUT = 360                # ✅ v6.0: Timeout maior (segundos)
MIN_WIN_RATE_OPT = 0.50             # ✅ v6.0: Mínimo Win Rate aceitável na otimização
MAX_DD_OPT = 0.15                   # ✅ v6.0: Máximo Drawdown permitido (15%)
ENABLE_MULTI_OBJECTIVE = True       # ✅ v6.0: Otimização Multi-Objetivo (WinRate, DD, Profit)
SLIPPAGE_VARIABLE = True            # ✅ v6.0: Simula slippage baseado em volatilidade
OPTIMIZER_APPLY_MT5_MARKET_WATCH_FILTER = False
OPTIMIZER_ALLOW_MT5_FALLBACK = False

ML_CONFIDENCE_THRESHOLD = 0.82  # BUY/SELL baseline
ML_MIN_SCORE = 65               # score técnico mínimo
ML_GOLDEN_CONFIDENCE = 0.85     # Golden Hour
ML_OVERRIDE_MIN_CONFIDENCE = 0.90
ENABLE_ML_VETO_OVERRIDE = False
ML_CONFIDENCE_BY_SESSION = {
    "GOLDEN": 0.80,
    "NORMAL": 0.82,
    "ASIAN": 0.84,
    "PROTECTION": 0.86
}

# ===========================
# GESTÃO DE RISCO INSTITUCIONAL
# ===========================
RISK_PER_TRADE_PCT = 0.0035          # 0.35% por trade
MAX_DAILY_DRAWDOWN = 0.015
MAX_TOTAL_DRAWDOWN = 0.06
MAX_SYMBOLS = 6
ALLOWED_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NAS100"]
MAX_DAILY_TRADES_PER_SYMBOL = 3
MAX_TOTAL_EXPOSURE_MULTIPLIER = 2.0
MAX_TOTAL_EXPOSURE_BASIS = "balance"  # "balance" ou "equity"
MAX_TOTAL_EXPOSURE_WARNING_PCT = 0.80
MAX_TOTAL_EXPOSURE_ALERT_PCT = 0.95
SESSION_MAX_POSITIONS = {
    "ASIAN": 0,
    "GOLDEN": 8,
    "NORMAL": 6,
    "PROTECTION": 0
}

# --- OVERTRADING PROTECTION ---
MAX_ORDERS_PER_SYMBOL = 1            # ✅ NOVO: Máximo de posições simultâneas por símbolo
ORDER_COOLDOWN_SECONDS = 300         # ✅ NOVO: 5 minutos de cooldown após fechar uma posição
MAX_GLOBAL_ALGO_ORDERS = 3
REQUOTE_RETRY_MAX = 2

# ✅ NOVAS CONSTANTES PARA EVITAR AttributeError
DEVIATION = 5                        # Execução mais precisa
MAGIC_NUMBER = 123456                # Número mágico para identificar ordens do bot
DEFAULT_LOT = 0.01                   # ✅ NOVO: Lote padrão de segurança/fallback
MIN_VOLUME = 0.01                    # Volume mínimo por trade
MAX_VOLUME = 0.50                    # 🚨 EMERGENCY LOCK: Limite máximo institucional por trade
MAX_SPREAD_FOREX = 25
MAX_SPREAD_ACCEPTABLE = 25
MAX_SPREAD_INDICES = 3500
MAX_SPREAD_CRYPTO = 2500
MAX_SPREAD_METALS = 80
MAX_SPREAD_EXOTICS = 10000
SESSION_SPREAD_LIMITS = {
    "GOLDEN": {
        "FOREX": {"allowance_pct": 0.20},
        "INDICES": {"max_points": 3500},
        "METALS": {"max_points": 80},
        "CRYPTO": {"max_points": 2500}
    },
    "NORMAL": {
        "FOREX": {"max_points": MAX_SPREAD_FOREX},
        "INDICES": {"max_points": MAX_SPREAD_INDICES},
        "METALS": {"max_points": MAX_SPREAD_METALS},
        "CRYPTO": {"max_points": MAX_SPREAD_CRYPTO}
    },
    "ASIAN": {
        "FOREX": {"max_points": 50},
        "INDICES": {"max_points": 4000},
        "METALS": {"max_points": 80},
        "CRYPTO": {"max_points": 2500}
    },
    "PROTECTION": {
        "FOREX": {"max_pips": 20}
    }
}
DEFAULT_STOP_LOSS_ATR_MULTIPLIER = 1.5
DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER = 4.0
DEFAULT_RISK_REWARD_RATIO = 1.5      # ✅ OTIMIZADO: R:R 1.5 para melhor taxa de acerto

# ===========================
# PARÂMETROS DE EXECUÇÃO
# ===========================
MAX_MT5_LATENCY_MS = 500
FAST_LOOP_INTERVAL = 15      # segundos
SLOW_LOOP_INTERVAL = 300     # 5 minutos
PANEL_UPDATE_INTERVAL = 5

# ===========================
# BOLLINGER BANDS (OTIMIZADOR)
# ===========================
BB_SQUEEZE_THRESHOLD = 0.015
BB_PERIOD = 20                       # Período padrão para Bollinger Bands
BB_DEVIATION = 2.0                   # Desvio padrão para Bollinger Bands

# ===========================
# TRAILING STOP & BREAKEVEN
# ===========================
ENABLE_BREAKEVEN = True
BREAKEVEN_TRIGGER_PIPS = 12
BREAKEVEN_TRIGGER_R = 1.5            # (Gatilho alternativo por Risco)
BREAKEVEN_BUFFER_PIPS = 0
BREAKEVEN_OFFSET_PIPS = 0            # (Legacy)
BREAKEVEN_TP_THRESHOLD_PCT = 0.5     # (Legacy)

ENABLE_TRAILING_STOP = True
TRAILING_START_PIPS = 12
TRAILING_STEP_PIPS = 4
TRAILING_DISTANCE_PIPS = 9

# ✅ NOVAS CONSTANTES PARA CÁLCULO DE SL/TP
STOP_LOSS_ATR_MULTIPLIER = 2.0       # Multiplicador padrão para SL baseado no ATR
TAKE_PROFIT_ATR_MULTIPLIER = 3.0     # Multiplicador padrão para TP baseado no ATR
MIN_STOP_LOSS_PIPS = 5               # SL mínimo em pips

# SNAPSHOT SEMANAL
SNAPSHOT_OUTPUT_DIR = "data/portfolio_snapshots"
SNAPSHOT_ALERT_DEVIATION_PCT = 0.20
SNAPSHOT_RETRY_ATTEMPTS = 3
SNAPSHOT_BACKOFF_BASE = 1.0

# FECHAMENTO AUTOMÁTICO SEXTA (Horário de Brasília)
FRIDAY_AUTO_CLOSE_ENABLED = True
FRIDAY_AUTO_CLOSE_BRT = "16:30"
FRIDAY_AUTO_CLOSE_BUFFER_MINUTES = 0

# ===========================
# ATIVOS E PARÂMETROS OTIMIZADOS (v7 - 2026-01-14)
# ===========================
# Parâmetros Elite carregados de: elite_params_20260117.json
FOREX_PAIRS = {
  "ADAG.NAS": {
    "ema_short": 22,
    "ema_long": 143,
    "rsi_low": 40,
    "rsi_high": 64,
    "adx_threshold": 25,
    "sl_atr": 2.77,
    "tp_atr": 3.0,
    "ml_threshold": 0.8
  },
  "USDZAR": {
    "ema_short": 17,
    "ema_long": 128,
    "rsi_low": 38,
    "rsi_high": 72,
    "adx_threshold": 6,
    "sl_atr": 3.5,
    "tp_atr": 2.5,
    "ml_threshold": 0.725
  },
  "XAGAUD": {
    "ema_short": 19,
    "ema_long": 135,
    "rsi_low": 38,
    "rsi_high": 65,
    "adx_threshold": 18,
    "sl_atr": 2.82,
    "tp_atr": 3.33,
    "ml_threshold": 0.70
  },
  "XAUEUR": {
    "ema_short": 10,
    "ema_long": 182,
    "rsi_low": 30,
    "rsi_high": 59,
    "adx_threshold": 25,
    "sl_atr": 1.2,
    "tp_atr": 3.5,
    "ml_threshold": 0.675
  },
  "XAUUSD": {
    "ema_short": 12,
    "ema_long": 161,
    "rsi_low": 37,
    "rsi_high": 53,
    "adx_threshold": 6,
    "sl_atr": 2.1,
    "tp_atr": 2.5,
    "ml_threshold": 0.675
  },
  "USDMXN": {
    "ema_short": 20,
    "ema_long": 145,
    "rsi_low": 45,
    "rsi_high": 62,
    "adx_threshold": 19,
    "sl_atr": 2.72,
    "tp_atr": 2.33,
    "ml_threshold": 0.8
  },
  "GBPAUD": {
    "ema_short": 16,
    "ema_long": 123,
    "rsi_low": 39,
    "rsi_high": 64,
    "adx_threshold": 20,
    "sl_atr": 2.59,
    "tp_atr": 3.46,
    "ml_threshold": 0.8
  },
  "XAGUSD": {
    "ema_short": 22,
    "ema_long": 132,
    "rsi_low": 40,
    "rsi_high": 68,
    "adx_threshold": 21,
    "sl_atr": 2.67,
    "tp_atr": 3.12,
    "ml_threshold": 0.685
  },
  "EURAUD": {
    "ema_short": 12,
    "ema_long": 157,
    "rsi_low": 46,
    "rsi_high": 62,
    "adx_threshold": 8,
    "sl_atr": 3.5,
    "tp_atr": 3.5,
    "ml_threshold": 0.775
  },
  "GBPJPY": {
    "ema_short": 20,
    "ema_long": 138,
    "rsi_low": 37,
    "rsi_high": 64,
    "adx_threshold": 17,
    "sl_atr": 2.52,
    "tp_atr": 3.46,
    "ml_threshold": 0.73
  },
  "UK100": {
    "ema_short": 12,
    "ema_long": 172,
    "rsi_low": 43,
    "rsi_high": 77,
    "adx_threshold": 24,
    "sl_atr": 3.4,
    "tp_atr": 5.0,
    "ml_threshold": 0.6
  },
  "GBPUSD": {
    "ema_short": 12,
    "ema_long": 145,
    "rsi_low": 45,
    "rsi_high": 55,
    "adx_threshold": 11,
    "sl_atr": 3.2,
    "tp_atr": 2.0,
    "ml_threshold": 0.8
  }
}

ALL_AVAILABLE_SYMBOLS = list(FOREX_PAIRS.keys())

# ===========================
# MAPEAMENTO DE SÍMBOLOS (SETORES)
# ===========================
SYMBOL_MAP = {
    
        "AUDUSD",
        "EURUSD",
        "GBPUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "USDJPY",
        "BTCUSD",
        "ETHUSD",
        "SOLUSD",
        "AUDJPY",
        "CADJPY",
        "EURAUD",
        "EURGBP",
        "EURJPY",
        "GBPAUD",
        "GBPJPY",
    
        "XAGAUD",
        "XAGEUR",
        "XAGUSD",
        "XAUAUD",
        "XAUCHF",
        "XAUEUR",
        "XAUGBP",
        "XAUJPY",
        "XAUUSD",
    
        "ABNB.NAS",
        "ABTC.NAS",
        "ADAG.NAS",
        "ADAM.NAS",
        "ADAUSD",
        "BNBUSD",
        "BTCUSD",
        "ETHUSD",
        "NETH25",
        "SOLUSD",
    
        "GERN.NAS",
        "UK100",
        "US30",
   
        "CBRL.NAS",
        "EURTRY",
        "EURZAR",
        "GBPTRY",
        "USDMXN",
        "USDTRY",
        "USDZAR",
    
}

MT5_SECTOR_FILTER = "ALL"
SECTOR_MAP = {
    "ALL": list(SYMBOL_MAP),
}

OPTIMIZER_USE_SECTOR_MAP = True
OPTIMIZER_USE_DYNAMIC_SELECTED_ASSETS = False
OPTIMIZER_APPLY_MT5_MARKET_WATCH_FILTER = True

# ✅ NOVAS CONSTANTES PARA INDICADORES PADRÃO
EMA_SHORT_PERIOD = 20
EMA_LONG_PERIOD = 50
RSI_PERIOD = 14
RSI_LOW_LIMIT = 30
RSI_HIGH_LIMIT = 70
ADX_PERIOD = 14

# ===========================
# MACHINE LEARNING
# ===========================
ML_RETRAIN_INTERVAL_HOURS = 24
ML_MIN_SAMPLES = 100
ML_CONFIDENCE_THRESHOLD = 0.70
ML_GOLDEN_CONFIDENCE_THRESHOLD = 0.75 # ✅ GOLDEN HOUR
ML_LEARNING_RATE = 0.1               # ✅ NOVO: Taxa de aprendizado para Q-Learning
ML_DISCOUNT_FACTOR = 0.9             # ✅ NOVO: Fator de desconto para Q-Learning
ML_LIVE_EXPLORATION_RATE = 0.05      # ✅ NOVO: Taxa de exploração em modo live
RISK_REWARD_RATIO = 1.5              # ✅ OTIMIZADO: R:R 1.5
MIN_VOLUME_COEFFICIENT = 0.4         # ✅ OTIMIZADO: Volume reduzido para liquidez normal (0.4)
MIN_SIGNAL_SCORE = 30

# --- AJUSTES POR ATIVO (tendência baixa e score mínimo técnico) ---
# Ativos com tendência média mais baixa podem operar com ADX menor
LOW_TREND_ADX_THRESHOLDS = {
    "XAGUSD": 22,
    "XAUUSD": 23,
}
# Ajuste de min_score técnico por ativo quando não houver override de elite params
LOW_MIN_SIGNAL_SCORE = {
    "XAGUSD": 28,
}

# --- Parâmetros de Boost do ML (tornam boost mais acessível e forte) ---
ML_BOOST_MIN_TECH_SCORE = 28
ML_BOOST_MIN_ML_SCORE = 70
ML_BOOST_MAX_POINTS = 25

# ===========================
# ADAPTIVE ENGINE v6.0 - SISTEMA ADAPTATIVO 4 CAMADAS
# ===========================
ENABLE_ADAPTIVE_ENGINE = True                    # ✅ Ativa/Desativa o sistema adaptativo
ADAPTIVE_ENGINE_UPDATE_INTERVAL = 300            # ✅ Atualização a cada 5 minutos
ADAPTIVE_ENGINE_MAX_STRATEGY_CHANGES = 3         # ✅ Máximo de mudanças de estratégia por hora (previne loops)
ADAPTIVE_ENGINE_MIN_CONFIDENCE = 0.65            # ✅ Confiança mínima para aplicar ajustes
ADAPTIVE_ENGINE_PANIC_THRESHOLD = 0.85          # ✅ Threshold para ativação do Panic Mode (85% drawdown)
ADAPTIVE_ENGINE_VOLATILITY_WINDOW = 20           # ✅ Janela de análise de volatilidade (períodos)
ADAPTIVE_ENGINE_TREND_WINDOW = 50                # ✅ Janela de análise de tendência (períodos)
ADAPTIVE_ENGINE_VOLUME_WINDOW = 20               # ✅ Janela de análise de volume (períodos)
ADAPTIVE_ENGINE_LEARNING_RATE = 0.1              # ✅ Taxa de aprendizado do Evolution Layer
ADAPTIVE_ENGINE_MEMORY_SIZE = 1000               # ✅ Tamanho máximo da memória histórica
ADAPTIVE_ENGINE_MIN_SAMPLES_FOR_ADAPTATION = 50  # ✅ Mínimo de amostras para adaptação

# ===========================
# VALIDAÇÃO DE CONFIGURAÇÃO
# ===========================
def validate_config():
    """Valida configurações críticas"""
    errors = []

    if RISK_PER_TRADE_PCT <= 0 or RISK_PER_TRADE_PCT > 0.05:
        errors.append("RISK_PER_TRADE_PCT deve estar entre 0 e 0.05 (5%)")
    if MAX_SYMBOLS < 1 or MAX_SYMBOLS > 10:
        errors.append("MAX_SYMBOLS deve estar entre 1 e 10")
    if not MT5_TERMINAL_PATH or not Path(MT5_TERMINAL_PATH).exists():
        errors.append(f"MT5_TERMINAL_PATH inválido: {MT5_TERMINAL_PATH}")
    if len(ALL_AVAILABLE_SYMBOLS) == 0:
        errors.append("ALL_AVAILABLE_SYMBOLS está vazio")

    # ✅ VALIDAÇÃO ADICIONAL PARA NEWS FILTER
    if ENABLE_NEWS_FILTER and not NEWS_URL:
        errors.append("NEWS_URL deve ser preenchido se ENABLE_NEWS_FILTER for True")
    
    # ✅ VALIDAÇÃO PARA EQUITY GUARD
    if MAX_DAILY_LOSS_PCT <= 0 or MAX_DAILY_LOSS_PCT > 0.10:
        errors.append("MAX_DAILY_LOSS_PCT deve estar entre 0 e 0.10 (10%)")

    # ✅ VALIDAÇÃO ADICIONAL PARA NOVAS CONSTANTES
    if not isinstance(DEVIATION, int) or DEVIATION <= 0:
        errors.append("DEVIATION deve ser um inteiro positivo.")
    if not isinstance(MAGIC_NUMBER, int) or MAGIC_NUMBER <= 0:
        errors.append("MAGIC_NUMBER deve ser um inteiro positivo.")
    if not isinstance(MIN_VOLUME, (int, float)) or MIN_VOLUME <= 0:
        errors.append("MIN_VOLUME deve ser um número positivo.")
    if not isinstance(MAX_VOLUME, (int, float)) or MAX_VOLUME <= MIN_VOLUME:
        errors.append("MAX_VOLUME deve ser um número maior que MIN_VOLUME.")
    if not isinstance(MAX_SPREAD_FOREX, (int, float)) or MAX_SPREAD_FOREX <= 0:
        errors.append("MAX_SPREAD_FOREX deve ser um número positivo.")
    if not isinstance(MAX_SPREAD_INDICES, (int, float)) or MAX_SPREAD_INDICES <= 0:
        errors.append("MAX_SPREAD_INDICES deve ser um número positivo.")
    if not isinstance(MAX_SPREAD_CRYPTO, (int, float)) or MAX_SPREAD_CRYPTO <= 0:
        errors.append("MAX_SPREAD_CRYPTO deve ser um número positivo.")
    if not isinstance(MAX_SPREAD_METALS, (int, float)) or MAX_SPREAD_METALS <= 0:
        errors.append("MAX_SPREAD_METALS deve ser um número positivo.")
    if not isinstance(STOP_LOSS_ATR_MULTIPLIER, (int, float)) or STOP_LOSS_ATR_MULTIPLIER <= 0:
        errors.append("STOP_LOSS_ATR_MULTIPLIER deve ser um número positivo.")
    if not isinstance(TAKE_PROFIT_ATR_MULTIPLIER, (int, float)) or TAKE_PROFIT_ATR_MULTIPLIER <= 0:
        errors.append("TAKE_PROFIT_ATR_MULTIPLIER deve ser um número positivo.")
    if not isinstance(MIN_STOP_LOSS_PIPS, (int, float)) or MIN_STOP_LOSS_PIPS <= 0:
        errors.append("MIN_STOP_LOSS_PIPS deve ser um número positivo.")
    if not isinstance(ML_LEARNING_RATE, (int, float)) or not (0 < ML_LEARNING_RATE <= 1):
        errors.append("ML_LEARNING_RATE deve ser um número entre 0 e 1.")
    if not isinstance(ML_DISCOUNT_FACTOR, (int, float)) or not (0 <= ML_DISCOUNT_FACTOR <= 1):
        errors.append("ML_DISCOUNT_FACTOR deve ser um número entre 0 e 1.")
    if not isinstance(ML_LIVE_EXPLORATION_RATE, (int, float)) or not (0 <= ML_LIVE_EXPLORATION_RATE <= 1):
        errors.append("ML_LIVE_EXPLORATION_RATE deve ser um número entre 0 e 1.")
    if not isinstance(RISK_REWARD_RATIO, (int, float)) or RISK_REWARD_RATIO <= 0:
        errors.append("RISK_REWARD_RATIO deve ser um número positivo.")

    # ✅ VALIDAÇÃO OTIMIZAÇÃO v6.0
    if MIN_WIN_RATE_OPT < 0.5 or MIN_WIN_RATE_OPT > 1.0:
        errors.append("MIN_WIN_RATE_OPT deve estar entre 0.5 e 1.0")
    if MAX_DD_OPT <= 0 or MAX_DD_OPT > 0.5:
        errors.append("MAX_DD_OPT deve estar entre 0 e 0.5")

    if MIN_SIGNAL_SCORE < 0 or MIN_SIGNAL_SCORE > 100:
        errors.append("MIN_SIGNAL_SCORE deve estar entre 0 e 100")


    if errors:
        raise ValueError("❌ Erros de configuração:\n" + "\n".join(f"  • {e}" for e in errors))

    print("✅ Configuração validada com sucesso!")
