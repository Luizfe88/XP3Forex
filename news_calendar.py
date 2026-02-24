<<<<<<< HEAD
import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime, timedelta
from threading import RLock
import config_forex as config
import pytz

logger = logging.getLogger("news_calendar")

# ✅ Lock para thread safety
mt5_lock = RLock()

class NewsCalendarMT5:
    def __init__(self):
        self.enabled = config.ENABLE_NEWS_FILTER
        # Define os fusos horários
        self.tz_br = pytz.timezone('America/Sao_Paulo')
        self.tz_utc = pytz.utc

    def apply_blackout(self, symbol: str) -> tuple:
        """
        Verifica se há notícias de alto impacto próximas.
        
        Returns:
            tuple: (bool, str) - (is_blackout, reason)
        """
        if not self.enabled:
            return False, ""

        try:
            # ✅ Verifica se a função existe no MT5
            if not hasattr(mt5, 'calendar_events_get'):
                logger.debug("News calendar não disponível nesta versão do MT5")
                return False, ""
            
            # 1. Pega o horário atual em Brasília
            now_br = datetime.now(self.tz_br)

            # 2. Busca eventos (MT5 trabalha com UTC no calendário)
            dt_from = datetime.now(self.tz_utc) - timedelta(hours=2)
            dt_to = datetime.now(self.tz_utc) + timedelta(hours=2)

            with mt5_lock:
                events = mt5.calendar_events_get(date_from=dt_from, date_to=dt_to)
            
            if events is None or len(events) == 0:
                return False, ""

            before = timedelta(minutes=config.NEWS_BLOCK_BEFORE_MIN)
            after = timedelta(minutes=config.NEWS_BLOCK_AFTER_MIN)

            for event in events:
                if event.importance == mt5.CALENDAR_IMPORTANCE_HIGH:
                    if event.currency in symbol or event.currency == "USD":
                        
                        # 3. CONVERSÃO: O MT5 dá o tempo em UTC. Convertemos para Brasília.
                        event_time_utc = datetime.fromtimestamp(event.time, self.tz_utc)
                        event_time_br = event_time_utc.astimezone(self.tz_br)
                        
                        # 4. COMPARAÇÃO: Tudo agora está no fuso de Brasília
                        if (event_time_br - before <= now_br <= event_time_br + after):
                            reason = f"Notícia {event.currency}: {event.name} às {event_time_br.strftime('%H:%M BRT')}"
                            logger.warning(f"⚠️ BLACKOUT (BRASÍLIA): {reason}")
                            return True, reason

            return False, ""
        
        except Exception as e:
            logger.error(f"Erro ao verificar news calendar: {e}")
            return False, ""


# ✅ Instância global para uso fácil
news_calendar = NewsCalendarMT5()


# ✅ Função wrapper para compatibilidade com imports antigos
def apply_blackout(symbol: str = "EURUSD") -> tuple:
    """
    Wrapper function para verificar blackout de notícias.
    
    Args:
        symbol: Par de moedas (ex: "EURUSD")
    
    Returns:
        tuple: (is_blackout: bool, reason: str)
    """
    return news_calendar.apply_blackout(symbol)


# ✅ Exportações
__all__ = [
    'NewsCalendarMT5',
    'news_calendar',
    'apply_blackout'
]

=======
import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime, timedelta
from threading import RLock
import config_forex as config
import pytz

logger = logging.getLogger("news_calendar")

# ✅ Lock para thread safety
mt5_lock = RLock()

class NewsCalendarMT5:
    def __init__(self):
        self.enabled = config.ENABLE_NEWS_FILTER
        # Define os fusos horários
        self.tz_br = pytz.timezone('America/Sao_Paulo')
        self.tz_utc = pytz.utc

    def apply_blackout(self, symbol: str) -> tuple:
        """
        Verifica se há notícias de alto impacto próximas.
        
        Returns:
            tuple: (bool, str) - (is_blackout, reason)
        """
        if not self.enabled:
            return False, ""

        try:
            # ✅ Verifica se a função existe no MT5
            if not hasattr(mt5, 'calendar_events_get'):
                logger.debug("News calendar não disponível nesta versão do MT5")
                return False, ""
            
            # 1. Pega o horário atual em Brasília
            now_br = datetime.now(self.tz_br)

            # 2. Busca eventos (MT5 trabalha com UTC no calendário)
            dt_from = datetime.now(self.tz_utc) - timedelta(hours=2)
            dt_to = datetime.now(self.tz_utc) + timedelta(hours=2)

            with mt5_lock:
                events = mt5.calendar_events_get(date_from=dt_from, date_to=dt_to)
            
            if events is None or len(events) == 0:
                return False, ""

            before = timedelta(minutes=config.NEWS_BLOCK_BEFORE_MIN)
            after = timedelta(minutes=config.NEWS_BLOCK_AFTER_MIN)

            for event in events:
                if event.importance == mt5.CALENDAR_IMPORTANCE_HIGH:
                    if event.currency in symbol or event.currency == "USD":
                        
                        # 3. CONVERSÃO: O MT5 dá o tempo em UTC. Convertemos para Brasília.
                        event_time_utc = datetime.fromtimestamp(event.time, self.tz_utc)
                        event_time_br = event_time_utc.astimezone(self.tz_br)
                        
                        # 4. COMPARAÇÃO: Tudo agora está no fuso de Brasília
                        if (event_time_br - before <= now_br <= event_time_br + after):
                            reason = f"Notícia {event.currency}: {event.name} às {event_time_br.strftime('%H:%M BRT')}"
                            logger.warning(f"⚠️ BLACKOUT (BRASÍLIA): {reason}")
                            return True, reason

            return False, ""
        
        except Exception as e:
            logger.error(f"Erro ao verificar news calendar: {e}")
            return False, ""


# ✅ Instância global para uso fácil
news_calendar = NewsCalendarMT5()


# ✅ Função wrapper para compatibilidade com imports antigos
def apply_blackout(symbol: str = "EURUSD") -> tuple:
    """
    Wrapper function para verificar blackout de notícias.
    
    Args:
        symbol: Par de moedas (ex: "EURUSD")
    
    Returns:
        tuple: (is_blackout: bool, reason: str)
    """
    return news_calendar.apply_blackout(symbol)


# ✅ Exportações
__all__ = [
    'NewsCalendarMT5',
    'news_calendar',
    'apply_blackout'
]

>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
logger.info("✅ news_calendar.py carregado com sucesso")