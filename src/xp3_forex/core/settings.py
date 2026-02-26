"""
🚀 XP3 PRO FOREX - CONFIGURAÇÕES INSTITUCIONAIS (Unified)
✅ Centralized Settings using Pydantic v2
✅ Environment Variable Support (.env)
✅ Clean Architecture Compliance
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ===========================
# CONSTANTES COMPLEXAS / ESTÁTICAS
# ===========================
# Tokens de Classes de Ativos
TOKENS_CRYPTO = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]
TOKENS_INDICES = ["US30", "NAS100", "USTEC", "DE40", "GER40", "GER30", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"]
TOKENS_METALS = ["XAU", "XAG", "GOLD", "SILVER"]
TOKENS_EXOTICS = ["TRY", "ZAR", "MXN", "RUB", "CNH", "PLN", "HUF", "CZK", "DKK", "NOK", "SEK"]

# Elite Config (Parâmetros Otimizados por Ativo)
# Pode ser movido para um arquivo JSON/YAML externo no futuro
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

SYMBOL_MAP = {
    "EURUSD": {"type": "major", "volatility": "medium", "spread": "low"},
    "GBPUSD": {"type": "major", "volatility": "high", "spread": "low"},
    "USDJPY": {"type": "major", "volatility": "low", "spread": "low"},
    "AUDUSD": {"type": "major", "volatility": "medium", "spread": "medium"},
    "USDCAD": {"type": "major", "volatility": "medium", "spread": "medium"}
}

class Settings(BaseSettings):
    """
    Configurações Centralizadas do XP3 PRO FOREX
    Carrega variáveis de ambiente do arquivo .env
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
    )

    # ===========================
    # 1. META TRADER 5 CONFIG
    # ===========================
    MT5_LOGIN: int = Field(default=0, description="Conta de Login MT5")
    MT5_PASSWORD: str = Field(default="", description="Senha MT5")
    MT5_SERVER: str = Field(default="", description="Servidor da Corretora")
    MT5_PATH: str = Field(
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        description="Caminho do executável terminal64.exe"
    )
    MT5_TIMEOUT: int = Field(default=60, description="Timeout para conexão MT5 (segundos)")
    
    # ===========================
    # 2. TRADING CONFIG
    # ===========================
    SYMBOLS: str = Field(default="EURUSD,GBPUSD,USDJPY,XAUUSD", description="Lista de símbolos (separados por vírgula)")
    TIMEFRAMES: str = Field(default="15,60,240", description="Lista de timeframes em minutos (ex: 15,60)")
    MAGIC_NUMBER: int = Field(default=123456, description="Magic Number para ordens")
    
    # Sessões de Trading (Horário de Brasília)
    SERVER_OFFSET: int = Field(default=0, description="Diferença Horária: Servidor -> Local")
    GOLDEN_HOUR_START: str = "10:00"
    GOLDEN_HOUR_END: str = "14:00"
    
    # ===========================
    # 3. RISK MANAGEMENT
    # ===========================
    RISK_PER_TRADE: float = Field(default=1.0, description="% do saldo por trade (ex: 1.0 para 1%)")
    MAX_POSITIONS: int = Field(default=5, description="Máximo de posições simultâneas")
    MAX_DAILY_LOSS_PERCENT: float = Field(default=5.0, description="Perda máxima diária (%)")
    MAX_WEEKLY_LOSS_PERCENT: float = Field(default=10.0, description="Perda máxima semanal (%)")
    MAX_MONTHLY_LOSS_PERCENT: float = Field(default=15.0, description="Perda máxima mensal (%)")
    MAX_SPREAD_PIPS: float = Field(default=3.0, description="Spread máximo permitido (pips)")
    MAX_SLIPPAGE: int = Field(default=3, description="Slippage máximo (points)")
    MAX_ORDERS_PER_SYMBOL: int = Field(default=1, description="Máximo de ordens abertas por símbolo")
    
    # ===========================
    # 4. INDICATORS & STRATEGY
    # ===========================
    ADX_PERIOD: int = 14
    RSI_PERIOD: int = 14
    ATR_PERIOD: int = 14
    EMA_FAST_PERIOD: int = 20
    EMA_SLOW_PERIOD: int = 50
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    
    # ===========================
    # 5. NEWS FILTER
    # ===========================
    ENABLE_NEWS_FILTER: bool = Field(default=True, description="Ativar filtro de notícias")
    NEWS_BLOCK_MINUTES_BEFORE: int = Field(default=30, description="Minutos antes da notícia para bloquear")
    NEWS_BLOCK_MINUTES_AFTER: int = Field(default=30, description="Minutos depois da notícia para bloquear")
    NEWS_IMPACT_LEVELS: List[str] = Field(default=["High", "Critical"], description="Níveis de impacto para filtrar")

    # ===========================
    # 6. SYSTEM & LOGGING
    # ===========================
    XP3_ENV: str = Field(default="Production", description="Ambiente (Production, Development)")
    LOG_LEVEL: str = Field(default="INFO", description="Nível de log (DEBUG, INFO, WARNING, ERROR)")
    LOGS_DIR: Path = Field(default=Path("logs"), description="Diretório de logs")
    DATA_DIR: Path = Field(default=Path("data"), description="Diretório de dados")
    OPTIMIZER_OUTPUT_DIR: Path = Field(default=Path("optimizer_output"), description="Diretório de saída do otimizador")
    
    # ===========================
    # 7. TELEGRAM NOTIFICATIONS
    # ===========================
    TELEGRAM_TOKEN: Optional[str] = Field(default=None, description="Token do Bot Telegram")
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, description="Chat ID do Telegram")
    NOTIFICATIONS_ENABLED: bool = Field(default=True, description="Ativar notificações")

    # ===========================
    # 8. OPTIMIZATION & ML
    # ===========================
    OPTIMIZATION_TRIALS: int = 100
    OPTIMIZATION_TIMEOUT: int = 3600
    OPTIMIZATION_N_JOBS: int = -1
    ML_MODEL_UPDATE_INTERVAL: int = 24
    ML_PREDICTION_THRESHOLD: float = 0.65

    @computed_field
    def symbols_list(self) -> List[str]:
        """Retorna lista de símbolos limpa"""
        if not self.SYMBOLS:
            return []
        return [s.strip().upper() for s in self.SYMBOLS.split(",") if s.strip()]

    @computed_field
    def timeframes_list(self) -> List[int]:
        """Retorna lista de timeframes como inteiros"""
        if not self.TIMEFRAMES:
            return []
        try:
            return [int(t.strip()) for t in self.TIMEFRAMES.split(",") if t.strip()]
        except ValueError:
            return [15, 60] # Fallback

    def get_log_file(self, filename: str = "xp3_forex.log") -> Path:
        """Retorna caminho completo do log"""
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return self.LOGS_DIR / filename
    
    def ensure_directories(self):
        """Cria diretórios necessários"""
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OPTIMIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Instância Global
settings = Settings()
settings.ensure_directories()
