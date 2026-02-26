from xp3_forex.core.settings import settings

# Aliases de configuração para compatibilidade legada
MT5_LOGIN = settings.MT5_LOGIN
MT5_PASSWORD = settings.MT5_PASSWORD
MT5_SERVER = settings.MT5_SERVER
MT5_PATH = settings.MT5_PATH

TELEGRAM_TOKEN = settings.TELEGRAM_TOKEN
TELEGRAM_CHAT_ID = settings.TELEGRAM_CHAT_ID

SYMBOLS = settings.SYMBOLS
TIMEFRAMES = settings.TIMEFRAMES
RISK_PER_TRADE = settings.RISK_PER_TRADE
MAX_POSITIONS = settings.MAX_POSITIONS

ENABLE_NEWS_FILTER = settings.ENABLE_NEWS_FILTER
NEWS_BLOCK_MINUTES_BEFORE = settings.NEWS_BLOCK_MINUTES_BEFORE
NEWS_BLOCK_MINUTES_AFTER = settings.NEWS_BLOCK_MINUTES_AFTER
MT5_CALENDAR_JSON_PATH = settings.MT5_CALENDAR_JSON_PATH
MT5_CALENDAR_REFRESH_MINUTES = settings.MT5_CALENDAR_REFRESH_MINUTES

LOGS_DIR = settings.LOGS_DIR

def get_symbols_list():
    return settings.symbols_list

def get_timeframes_list():
    return settings.timeframes_list
