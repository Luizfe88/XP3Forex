from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    MT5_LOGIN: Optional[int] = None
    MT5_PASSWORD: Optional[str] = None
    MT5_SERVER: Optional[str] = None
    MT5_PATH: Optional[str] = None

    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    SYMBOLS: Optional[str] = "EURUSD,GBPUSD,USDJPY"
    TIMEFRAMES: Optional[str] = "15,60,240"
    RISK_PER_TRADE: Optional[float] = 0.02
    MAX_POSITIONS: Optional[int] = 5

    ENABLE_NEWS_FILTER: bool = True
    NEWS_BLOCK_MINUTES_BEFORE: int = 30
    NEWS_BLOCK_MINUTES_AFTER: int = 30
    MT5_CALENDAR_JSON_PATH: str = "data/mt5_calendar.json"
    MT5_CALENDAR_REFRESH_MINUTES: int = 30

    LOGS_DIR: str = "logs"

    @property
    def symbols_list(self) -> List[str]:
        if not self.SYMBOLS:
            return []
        return [s.strip().upper() for s in self.SYMBOLS.split(",") if s.strip()]

    @property
    def timeframes_list(self) -> List[int]:
        if not self.TIMEFRAMES:
            return []
        out: List[int] = []
        for part in self.TIMEFRAMES.split(","):
            part = part.strip()
            if part:
                try:
                    out.append(int(part))
                except ValueError:
                    continue
        return out


# Singleton
settings = Settings()

