# XP3 PRO FOREX - Core Configuration Module
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
# CONFIGURAÇÕES DE RISCO
# ===========================
MAX_SPREAD_PIPS = 2.0
MAX_SLIPPAGE = 3
MAX_ORDERS_PER_SYMBOL = 1
MAX_CONCURRENT_ORDERS = 10
MAX_DAILY_LOSS_PERCENT = 5.0
MAX_WEEKLY_LOSS_PERCENT = 10.0
MAX_MONTHLY_LOSS_PERCENT = 15.0

# ===========================
# PARÂMETROS DE INDICADORES
# ===========================
ADX_PERIOD = 14
RSI_PERIOD = 14
ATR_PERIOD = 14
EMA_FAST_PERIOD = 20
EMA_SLOW_PERIOD = 50
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# ===========================
# CONFIGURAÇÕES DE OTIMIZAÇÃO
# ===========================
OPTIMIZATION_TRIALS = 100
OPTIMIZATION_TIMEOUT = 3600  # segundos
OPTIMIZATION_N_JOBS = -1   # usar todos os cores

# ===========================
# CONFIGURAÇÕES DE LOGGING
# ===========================
LOG_LEVEL = "INFO"
LOG_MAX_FILE_SIZE_MB = 50
LOG_BACKUP_COUNT = 3

# ===========================
# CONFIGURAÇÕES DE MACHINE LEARNING
# ===========================
ML_MODEL_UPDATE_INTERVAL = 24  # horas
ML_PREDICTION_THRESHOLD = 0.65
ML_FEATURES = ["adx", "rsi", "ema_diff", "atr", "volume_ratio"]

# ===========================
# ELITE CONFIG (PARÂMETROS OTIMIZADOS)
# ===========================
ELITE_CONFIG = {
    "EURUSD": {
        "adx_threshold": 25,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "atr_multiplier": 2.0,
        "risk_per_trade": 0.02,
        "lot_size_multiplier": 1.0
    },
    "GBPUSD": {
        "adx_threshold": 28,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "atr_multiplier": 2.5,
        "risk_per_trade": 0.015,
        "lot_size_multiplier": 0.8
    },
    "USDJPY": {
        "adx_threshold": 22,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "atr_multiplier": 1.8,
        "risk_per_trade": 0.025,
        "lot_size_multiplier": 1.2
    }
}

# ===========================
# SYMBOL MAP (PARA ML OPTIMIZER)
# ===========================
SYMBOL_MAP = {
    "EURUSD": {"type": "major", "volatility": "medium", "spread": "low"},
    "GBPUSD": {"type": "major", "volatility": "high", "spread": "low"},
    "USDJPY": {"type": "major", "volatility": "low", "spread": "low"},
    "AUDUSD": {"type": "major", "volatility": "medium", "spread": "medium"},
    "USDCAD": {"type": "major", "volatility": "medium", "spread": "medium"}
}

# ===========================
# CONFIGURAÇÕES DE BACKTESTING
# ===========================
BACKTEST_INITIAL_CAPITAL = 10000
BACKTEST_SPREAD = 1.5  # pips
BACKTEST_COMMISSION = 0.0  # percentual
BACKTEST_SLIPPAGE = 0.5  # pips

# ===========================
# CONFIGURAÇÕES DE DASHBOARD
# ===========================
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8080
DASHBOARD_UPDATE_INTERVAL = 5  # segundos

# ===========================
# CONFIGURAÇÕES DE NOTIFICAÇÕES
# ===========================
NOTIFICATIONS_ENABLED = True
NOTIFICATIONS_MIN_PROFIT = 10  # pips
NOTIFICATIONS_MIN_LOSS = -20   # pips