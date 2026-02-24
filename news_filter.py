<<<<<<< HEAD
# news_filter.py - Filtro de Notícias v5.0
"""
🛡️ XP3 PRO FOREX - NEWS FILTER
✅ Download de calendário JSON
✅ Cache robusto de 4 horas
✅ Filtro por importância e moeda
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from xp3_forex.core import config as config

logger = logging.getLogger("news_filter")

CACHE_FILE = Path("data/news_cache.json")
CACHE_DURATION_HOURS = 4

class NewsFilter:
    def __init__(self):
        self.enabled = getattr(config, 'ENABLE_NEWS_FILTER', True)
        self.news_url = getattr(config, 'NEWS_URL', "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
        self.block_before = getattr(config, 'NEWS_BLOCK_MINUTES_BEFORE', 30)
        self.block_after = getattr(config, 'NEWS_BLOCK_MINUTES_AFTER', 30)
        self.calendar = []
        self.use_mt5_calendar = getattr(config, 'USE_MT5_CALENDAR', True)
        self.mt5_calendar_path = Path(getattr(config, 'MT5_CALENDAR_JSON_PATH', 'data/mt5_calendar.json'))
        self._load_cache()

    def _load_cache(self):
        """Carrega o calendário do cache ou faz download se expirado."""
        # ✅ v5.1: Prioriza MT5 Calendar se habilitado
        if self.use_mt5_calendar and self.mt5_calendar_path.exists():
            try:
                mtime = datetime.fromtimestamp(self.mt5_calendar_path.stat().st_mtime)
                refresh_minutes = getattr(config, 'MT5_CALENDAR_REFRESH_MINUTES', 30)
                if datetime.now() - mtime < timedelta(minutes=refresh_minutes):
                    with open(self.mt5_calendar_path, 'r', encoding='utf-8') as f:
                        self.calendar = json.load(f)
                    logger.info(f"✅ Calendário MT5 carregado: {len(self.calendar)} eventos")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Erro ao ler calendário MT5: {e}")
        
        # Fallback: Cache tradicional ou ForexFactory
        if CACHE_FILE.exists():
            mtime = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=CACHE_DURATION_HOURS):
                try:
                    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                        self.calendar = json.load(f)
                    logger.debug("✅ Calendário de notícias carregado do cache.")
                    return
                except Exception as e:
                    logger.error(f"❌ Erro ao ler cache de notícias: {e}")

        self.update_calendar()

    def update_calendar(self):
        """Atualiza calendário: MT5 JSON ou ForexFactory como fallback."""
        if not self.enabled:
            return

        # ✅ v5.1: Tenta MT5 Calendar primeiro
        if self.use_mt5_calendar and self.mt5_calendar_path.exists():
            try:
                with open(self.mt5_calendar_path, 'r', encoding='utf-8') as f:
                    self.calendar = json.load(f)
                logger.info(f"✅ Calendário MT5 atualizado: {len(self.calendar)} eventos")
                return
            except Exception as e:
                logger.warning(f"⚠️ Falha ao ler MT5 Calendar, usando ForexFactory: {e}")

        # Fallback: ForexFactory
        try:
            logger.info(f"🌐 Baixando calendário ForexFactory (fallback): {self.news_url}")
            response = requests.get(self.news_url, timeout=10)
            response.raise_for_status()
            self.calendar = response.json()

            # Salva no cache
            CACHE_FILE.parent.mkdir(exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.calendar, f)
            
            logger.info(f"✅ Calendário ForexFactory atualizado com {len(self.calendar)} eventos.")
        except Exception as e:
            logger.error(f"❌ Falha ao atualizar calendário de notícias: {e}")

    def is_news_blackout(self, symbol: str) -> tuple:
        """
        Verifica se há notícias de 'High Impact' para as moedas do símbolo atual.
        
        Returns:
            tuple: (bool, str) - (bloqueado, mensagem)
        """
        if not self.enabled or not self.calendar:
            return False, "Livre"

        now_utc = datetime.utcnow()
        # Moedas envolvidas (ex: EUR, USD para EURUSD)
        currencies = [symbol[:3], symbol[3:6]]
        
        # News blocking window
        before_time = now_utc + timedelta(minutes=self.block_before)
        after_time = now_utc - timedelta(minutes=self.block_after)

        for event in self.calendar:
            # Filtra apenas impacto alto
            if event.get('impact') != 'High':
                continue
                
            event_currency = event.get('country') # O campo pode variar entre 'country' e 'currency' no JSON da ForexFactory
            if event_currency not in currencies:
                continue

            try:
                # Formato esperado: "2026-01-07T12:00:00-05:00"
                # Simplificando parse para o exemplo
                event_date_str = event.get('date')
                # Removendo fuso horário para simplificar comparação em UTC
                clean_date = event_date_str.split('-')[0] + '-' + event_date_str.split('-')[1] + '-' + event_date_str.split('-')[2]
                if 'T' in clean_date:
                    clean_date = clean_date.split('T')[0] + ' ' + clean_date.split('T')[1].split('-')[0].split('+')[0]
                
                from datetime import timezone
                event_time = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).astimezone(timezone.utc).replace(tzinfo=None)
                
                if after_time <= event_time <= before_time:
                    minutes_to = int((event_time - now_utc).total_seconds() / 60)
                    if minutes_to > 0:
                        msg = f"Bloqueado ({event_currency} {event.get('title')} em {minutes_to}min)"
                    else:
                        msg = f"Bloqueado ({event_currency} {event.get('title')} agora)"
                    return True, msg

            except Exception as e:
                logger.error(f"❌ Erro ao processar data da notícia: {e}")
                continue

        return False, "Livre"

    def should_close_for_news(self, symbol: str, position_profit: float) -> tuple:
        """
        ✅ v5.0: Verifica se deve fechar posição lucrativa antes de notícia de alto impacto.
        
        Args:
            symbol: Par de moedas
            position_profit: Lucro atual da posição
            
        Returns:
            tuple: (should_close: bool, reason: str, minutes_to_news: int)
        """
        if not self.enabled or not self.calendar:
            return False, "News filter desabilitado", 0
        
        # Só fecha se posição estiver lucrativa
        if position_profit <= 0:
            return False, "Posição não lucrativa", 0
        
        now_utc = datetime.utcnow()
        currencies = [symbol[:3], symbol[3:6]]
        
        # Janela de 15 minutos antes da notícia
        check_window = now_utc + timedelta(minutes=15)
        
        for event in self.calendar:
            if event.get('impact') != 'High':
                continue
                
            event_currency = event.get('country')
            if event_currency not in currencies:
                continue
            
            try:
                event_date_str = event.get('date')
                from datetime import timezone
                event_time = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).astimezone(timezone.utc).replace(tzinfo=None)
                
                # Verifica se a notícia está dentro de 15 minutos
                if now_utc < event_time <= check_window:
                    minutes_to = int((event_time - now_utc).total_seconds() / 60)
                    reason = f"Fechando antes de {event.get('title')} ({event_currency}) em {minutes_to}min"
                    logger.info(f"📰 {symbol}: {reason}")
                    return True, reason, minutes_to
                    
            except Exception as e:
                logger.error(f"❌ Erro ao processar data da notícia no should_close: {e}")
                continue
        
        return False, "Sem notícias iminentes", 0

# Instância global
news_filter = NewsFilter()
=======
# news_filter.py - Filtro de Notícias v5.0
"""
🛡️ XP3 PRO FOREX - NEWS FILTER
✅ Download de calendário JSON
✅ Cache robusto de 4 horas
✅ Filtro por importância e moeda
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import config_forex as config

logger = logging.getLogger("news_filter")

CACHE_FILE = Path("data/news_cache.json")
CACHE_DURATION_HOURS = 4

class NewsFilter:
    def __init__(self):
        self.enabled = getattr(config, 'ENABLE_NEWS_FILTER', True)
        self.news_url = getattr(config, 'NEWS_URL', "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
        self.block_before = getattr(config, 'NEWS_BLOCK_MINUTES_BEFORE', 30)
        self.block_after = getattr(config, 'NEWS_BLOCK_MINUTES_AFTER', 30)
        self.calendar = []
        self.use_mt5_calendar = getattr(config, 'USE_MT5_CALENDAR', True)
        self.mt5_calendar_path = Path(getattr(config, 'MT5_CALENDAR_JSON_PATH', 'data/mt5_calendar.json'))
        self._load_cache()

    def _load_cache(self):
        """Carrega o calendário do cache ou faz download se expirado."""
        # ✅ v5.1: Prioriza MT5 Calendar se habilitado
        if self.use_mt5_calendar and self.mt5_calendar_path.exists():
            try:
                mtime = datetime.fromtimestamp(self.mt5_calendar_path.stat().st_mtime)
                refresh_minutes = getattr(config, 'MT5_CALENDAR_REFRESH_MINUTES', 30)
                if datetime.now() - mtime < timedelta(minutes=refresh_minutes):
                    with open(self.mt5_calendar_path, 'r', encoding='utf-8') as f:
                        self.calendar = json.load(f)
                    logger.info(f"✅ Calendário MT5 carregado: {len(self.calendar)} eventos")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Erro ao ler calendário MT5: {e}")
        
        # Fallback: Cache tradicional ou ForexFactory
        if CACHE_FILE.exists():
            mtime = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=CACHE_DURATION_HOURS):
                try:
                    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                        self.calendar = json.load(f)
                    logger.debug("✅ Calendário de notícias carregado do cache.")
                    return
                except Exception as e:
                    logger.error(f"❌ Erro ao ler cache de notícias: {e}")

        self.update_calendar()

    def update_calendar(self):
        """Atualiza calendário: MT5 JSON ou ForexFactory como fallback."""
        if not self.enabled:
            return

        # ✅ v5.1: Tenta MT5 Calendar primeiro
        if self.use_mt5_calendar and self.mt5_calendar_path.exists():
            try:
                with open(self.mt5_calendar_path, 'r', encoding='utf-8') as f:
                    self.calendar = json.load(f)
                logger.info(f"✅ Calendário MT5 atualizado: {len(self.calendar)} eventos")
                return
            except Exception as e:
                logger.warning(f"⚠️ Falha ao ler MT5 Calendar, usando ForexFactory: {e}")

        # Fallback: ForexFactory
        try:
            logger.info(f"🌐 Baixando calendário ForexFactory (fallback): {self.news_url}")
            response = requests.get(self.news_url, timeout=10)
            response.raise_for_status()
            self.calendar = response.json()

            # Salva no cache
            CACHE_FILE.parent.mkdir(exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.calendar, f)
            
            logger.info(f"✅ Calendário ForexFactory atualizado com {len(self.calendar)} eventos.")
        except Exception as e:
            logger.error(f"❌ Falha ao atualizar calendário de notícias: {e}")

    def is_news_blackout(self, symbol: str) -> tuple:
        """
        Verifica se há notícias de 'High Impact' para as moedas do símbolo atual.
        
        Returns:
            tuple: (bool, str) - (bloqueado, mensagem)
        """
        if not self.enabled or not self.calendar:
            return False, "Livre"

        now_utc = datetime.utcnow()
        # Moedas envolvidas (ex: EUR, USD para EURUSD)
        currencies = [symbol[:3], symbol[3:6]]
        
        # News blocking window
        before_time = now_utc + timedelta(minutes=self.block_before)
        after_time = now_utc - timedelta(minutes=self.block_after)

        for event in self.calendar:
            # Filtra apenas impacto alto
            if event.get('impact') != 'High':
                continue
                
            event_currency = event.get('country') # O campo pode variar entre 'country' e 'currency' no JSON da ForexFactory
            if event_currency not in currencies:
                continue

            try:
                # Formato esperado: "2026-01-07T12:00:00-05:00"
                # Simplificando parse para o exemplo
                event_date_str = event.get('date')
                # Removendo fuso horário para simplificar comparação em UTC
                clean_date = event_date_str.split('-')[0] + '-' + event_date_str.split('-')[1] + '-' + event_date_str.split('-')[2]
                if 'T' in clean_date:
                    clean_date = clean_date.split('T')[0] + ' ' + clean_date.split('T')[1].split('-')[0].split('+')[0]
                
                from datetime import timezone
                event_time = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).astimezone(timezone.utc).replace(tzinfo=None)
                
                if after_time <= event_time <= before_time:
                    minutes_to = int((event_time - now_utc).total_seconds() / 60)
                    if minutes_to > 0:
                        msg = f"Bloqueado ({event_currency} {event.get('title')} em {minutes_to}min)"
                    else:
                        msg = f"Bloqueado ({event_currency} {event.get('title')} agora)"
                    return True, msg

            except Exception as e:
                logger.error(f"❌ Erro ao processar data da notícia: {e}")
                continue

        return False, "Livre"

    def should_close_for_news(self, symbol: str, position_profit: float) -> tuple:
        """
        ✅ v5.0: Verifica se deve fechar posição lucrativa antes de notícia de alto impacto.
        
        Args:
            symbol: Par de moedas
            position_profit: Lucro atual da posição
            
        Returns:
            tuple: (should_close: bool, reason: str, minutes_to_news: int)
        """
        if not self.enabled or not self.calendar:
            return False, "News filter desabilitado", 0
        
        # Só fecha se posição estiver lucrativa
        if position_profit <= 0:
            return False, "Posição não lucrativa", 0
        
        now_utc = datetime.utcnow()
        currencies = [symbol[:3], symbol[3:6]]
        
        # Janela de 15 minutos antes da notícia
        check_window = now_utc + timedelta(minutes=15)
        
        for event in self.calendar:
            if event.get('impact') != 'High':
                continue
                
            event_currency = event.get('country')
            if event_currency not in currencies:
                continue
            
            try:
                event_date_str = event.get('date')
                from datetime import timezone
                event_time = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).astimezone(timezone.utc).replace(tzinfo=None)
                
                # Verifica se a notícia está dentro de 15 minutos
                if now_utc < event_time <= check_window:
                    minutes_to = int((event_time - now_utc).total_seconds() / 60)
                    reason = f"Fechando antes de {event.get('title')} ({event_currency}) em {minutes_to}min"
                    logger.info(f"📰 {symbol}: {reason}")
                    return True, reason, minutes_to
                    
            except Exception as e:
                logger.error(f"❌ Erro ao processar data da notícia no should_close: {e}")
                continue
        
        return False, "Sem notícias iminentes", 0

# Instância global
news_filter = NewsFilter()
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
