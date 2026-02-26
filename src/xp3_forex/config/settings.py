
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import Field, validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

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

    # --- META TRADER 5 CONFIG ---
    MT5_LOGIN: int = Field(default=0, description="Conta de Login MT5")
    MT5_PASSWORD: str = Field(default="", description="Senha MT5")
    MT5_SERVER: str = Field(default="", description="Servidor da Corretora")
    MT5_PATH: str = Field(
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        description="Caminho do executável terminal64.exe"
    )
    MT5_TIMEOUT: int = Field(default=60, description="Timeout para conexão MT5 (segundos)")
    
    # --- TRADING CONFIG ---
    SYMBOLS: str = Field(default="EURUSD,GBPUSD,USDJPY,XAUUSD", description="Lista de símbolos (separados por vírgula)")
    TIMEFRAMES: str = Field(default="15,60,240", description="Lista de timeframes em minutos (ex: 15,60)")
    MAGIC_NUMBER: int = Field(default=123456, description="Magic Number para ordens")
    
    # --- RISK MANAGEMENT ---
    RISK_PER_TRADE: float = Field(default=1.0, description="% do saldo por trade (ex: 1.0 para 1%)")
    MAX_POSITIONS: int = Field(default=5, description="Máximo de posições simultâneas")
    MAX_DAILY_LOSS_PERCENT: float = Field(default=5.0, description="Perda máxima diária (%)")
    MAX_SPREAD_PIPS: float = Field(default=3.0, description="Spread máximo permitido (pips)")
    
    # --- NEWS FILTER ---
    ENABLE_NEWS_FILTER: bool = Field(default=True, description="Ativar filtro de notícias")
    NEWS_BLOCK_MINUTES_BEFORE: int = Field(default=30, description="Minutos antes da notícia para bloquear")
    NEWS_BLOCK_MINUTES_AFTER: int = Field(default=30, description="Minutos depois da notícia para bloquear")
    NEWS_IMPACT_LEVELS: List[str] = Field(default=["High", "Critical"], description="Níveis de impacto para filtrar")

    # --- SYSTEM ---
    LOG_LEVEL: str = Field(default="INFO", description="Nível de log (DEBUG, INFO, WARNING, ERROR)")
    LOGS_DIR: Path = Field(default=Path("logs"), description="Diretório de logs")
    DATA_DIR: Path = Field(default=Path("data"), description="Diretório de dados")
    
    # --- TELEGRAM NOTIFICATIONS ---
    TELEGRAM_TOKEN: Optional[str] = Field(default=None, description="Token do Bot Telegram")
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, description="Chat ID do Telegram")

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

# Instância Global
settings = Settings()
