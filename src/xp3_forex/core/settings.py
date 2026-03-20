"""
🚀 XP3 PRO FOREX - CONFIGURAÇÕES INSTITUCIONAIS (Unified)
✅ Centralized Settings using Pydantic v2
✅ Environment Variable Support (.env)
✅ Clean Architecture Compliance
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import Field, computed_field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# ===========================
# 0. MODEL DEFINITIONS & CONSTANTS
# ===========================

class SessionMetricsConfig(BaseModel):
    """Configuração de métricas alvo por sessão"""
    name: str
    start_time_utc: str  # Format "HH:MM"
    end_time_utc: str    # Format "HH:MM"
    min_profit_factor: float
    min_win_rate: float
    description: str

# Elite Config (Parâmetros Otimizados por Ativo)
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

# ===========================
# 1. CORE SETTINGS CLASS
# ===========================

class Settings(BaseSettings):
    """
    Configurações Centralizadas do XP3 PRO FOREX
    Carrega variáveis de ambiente do arquivo .env
    """
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent.parent / ".env"), 
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # ===========================
    # 0. ASSET CLASS TOKENS
    # ===========================
    TOKENS_CRYPTO: List[str] = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]
    TOKENS_INDICES: List[str] = ["US30", "NAS100", "USTEC", "DE40", "GER40", "GER30", "UK100", "US500", "USA500", "SPX500", "HK50", "JP225", "FRA40"]
    TOKENS_METALS: List[str] = ["XAU", "XAG", "GOLD", "SILVER"]
    TOKENS_EXOTICS: List[str] = ["TRY", "ZAR", "MXN", "RUB", "CNH", "PLN", "HUF", "CZK", "DKK", "NOK", "SEK", "THB", "BRL"]

    # ===========================
    # 1. META TRADER 5 CONFIG
    # ===========================
    MT5_LOGIN: int = Field(default=0, description="Conta de Login MT5")
    MT5_PASSWORD: str = Field(default="", description="Senha MT5")
    MT5_SERVER: str = Field(default="", description="Servidor da Corretora")
    MT5_PATH: str = Field(
        default=r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe",
        description="Caminho do executável terminal64.exe"
    )
    MT5_TIMEOUT: int = Field(default=60, description="Timeout para conexão MT5 (segundos)")
    
    # ===========================
    # 2. TRADING CONFIG
    # ===========================
    SYMBOLS: str = Field(default="EURUSD,GBPUSD,USDJPY,XAUUSD", description="Lista de símbolos (separados por vírgula)")
    SYMBOL_WHITELIST: str = Field(
        default="EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,NZDUSD,EURGBP,EURJPY,GBPJPY,USDCHF",
        description="Whitelist de símbolos permitidos"
    )
    TIMEFRAMES: str = Field(default="15,60,240", description="Lista de timeframes em minutos (ex: 15,60)")
    MAGIC_NUMBER: int = Field(default=123456, description="Magic Number para ordens")
    
    # Sessões de Trading (Horário de Brasília/UTC)
    SERVER_OFFSET: int = Field(default=0, description="Diferença Horária: Servidor -> Local")
    GOLDEN_HOUR_START: str = "10:00"
    GOLDEN_HOUR_END: str = "14:00"
    
    # Session Definitions (UTC)
    SESSION_ASIA: SessionMetricsConfig = SessionMetricsConfig(
        name="ASIA",
        start_time_utc="22:00",
        end_time_utc="08:00",
        min_profit_factor=1.2,
        min_win_rate=0.65,
        description="Baixa volatilidade, mercado de range (reversão à média)."
    )
    SESSION_LONDON: SessionMetricsConfig = SessionMetricsConfig(
        name="LONDON",
        start_time_utc="08:00",
        end_time_utc="16:00",
        min_profit_factor=1.8,
        min_win_rate=0.50,
        description="Alta liquidez, tendência e breakouts."
    )
    SESSION_NY: SessionMetricsConfig = SessionMetricsConfig(
        name="NY",
        start_time_utc="13:00",
        end_time_utc="22:00",
        min_profit_factor=2.0,
        min_win_rate=0.55,
        description="Altíssima volatilidade e notícias macro (Overlap Londres)."
    )
    
    # ===========================
    # 3. RISK MANAGEMENT
    # ===========================
    USE_VIRTUAL_BALANCE: bool = Field(default=False, description="Usar saldo virtual em vez do real do MT5")
    VIRTUAL_BALANCE: float = Field(default=100.0, description="Saldo virtual para cálculos de risco ($)")
    RISK_PER_TRADE: float = Field(default=1.0, description="% do saldo por trade (ex: 1.0 para 1.0%)")
    MAX_LOTS_PER_TRADE: float = Field(default=0.01, description="Volume máximo por trade (lotes)")
    FORCE_EXECUTION: bool = Field(default=False, description="Bypass filtros para testes (DEBUG ONLY)")
    RETRY_ATTEMPTS: int = Field(default=5, description="Tentativas de envio de ordem")
    RETRY_BACKOFF: float = Field(default=2.0, description="Backoff exponencial (segundos)")
    MAX_POSITIONS: int = Field(default=2, description="Máximo de posições simultâneas")
    MAX_DAILY_LOSS_PERCENT: float = Field(default=3.0, description="Perda máxima diária (%)")
    KILL_SWITCH_DD_PCT: float = Field(default=0.05, description="Kill Switch se DD Global > X% (0.05 = 5%)")
    LOSS_COOLDOWN_MINUTES: int = Field(default=720, description="Minutos de pausa após um Stop Loss no mesmo ativo")
    MIN_HOLDING_TIME: int = Field(default=60, description="Tempo mínimo de permanência (segundos) antes de um Dynamic Exit")
    
    # Spread Thresholds (Points)
    MAX_SPREAD_MAJORS: int = Field(default=30, description="Max spread for Majors/Minors (points)")
    MAX_SPREAD_EXOTICS: int = Field(default=500, description="Max spread for Exotics (points)")
    MAX_SPREAD_INDICES: int = Field(default=1000, description="Max spread for Indices (points)")
    MAX_SPREAD_CRYPTO: int = Field(default=3000, description="Max spread for Crypto (points)")
    MAX_SPREAD_METALS: int = Field(default=100, description="Max spread for Metals (points)")
    
    ALLOWED_ASSET_CATEGORIES: str = Field(default="major,metal,index,exotic,crypto", description="Categorias de ativos permitidas (separadas por vírgula)")

    # Novos Mecanismos de Defesa (P&L Base)
    PROFIT_ACTIVATION_THRESHOLD: float = Field(default=3.0, description="Lucro mínimo para ativar o Profit Trailing Shield ($)")
    PROFIT_TRAILING_PERCENT: float = Field(default=0.25, description="Percentual de devolução permitido do lucro máximo (ex: 0.25 = 25%)")
    MAX_LOSS_DOLLARS: float = Field(default=-2.0, description="Perda financeira máxima permitida por posição ($)")
    BREAK_EVEN_TRIGGER: float = Field(default=1.5, description="Lucro atingido para mover o SL ao Break-Even ($)")

    MAX_WEEKLY_LOSS_PERCENT: float = Field(default=10.0, description="Perda máxima semanal (%)")
    MAX_MONTHLY_LOSS_PERCENT: float = Field(default=15.0, description="Perda máxima mensal (%)")
    MAX_SPREAD_PIPS: float = Field(default=3.0, description="Spread máximo permitido (pips) - Deprecated, use Points")
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
    AUTO_CLOSE_ON_MT5: bool = Field(default=False, description="Fechar bot automaticamente quando o MT5 for aberto manualmente")

    # ===========================
    # 6. SYSTEM & LOGGING
    # ===========================
    XP3_ENV: str = Field(default="Production", description="Ambiente (Production, Development)")
    DEBUG_MODE: bool = Field(default=False, description="Ativar logs detalhados (Why Report) para todos os estados")
    LOG_LEVEL: str = Field(default="INFO", description="Nível de log (DEBUG, INFO, WARNING, ERROR)")
    
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    @computed_field
    def LOGS_DIR(self) -> Path:
        return self.project_root / "logs"

    @computed_field
    def DATA_DIR(self) -> Path:
        return self.project_root / "data"

    @computed_field
    def OPTIMIZER_OUTPUT_DIR(self) -> Path:
        return self.project_root / "optimizer_output"
    
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
    ML_PREDICTION_THRESHOLD: float = 0.60

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

    def get_quant_params(self, symbol: str) -> Dict[str, Any]:
        """
        Loads optimized quantitative parameters for a specific symbol.
        Returns default values if not found.
        """
        path = self.DATA_DIR / "quant_optimized_params.json"
        
        # Default fallback values
        defaults = {
            "hurst_lookback": 1000,
            "mmi_lookback": 300,
            "initial_r": 500.0,
            "min_q": 0.01,
            "max_q": 0.1
        }
        
        if not path.exists():
            return defaults
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data.get(symbol, defaults)
        except Exception:
            return defaults

# Instância Global
settings = Settings()
settings.ensure_directories()
