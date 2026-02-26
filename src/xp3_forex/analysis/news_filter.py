"""
XP3 PRO FOREX - News Filter Module (Migrated to src-layout)

This module has been migrated to the new src-layout architecture.
Original: news_filter.py
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

from xp3_forex.core.settings import settings

logger = logging.getLogger("news_filter")

CACHE_FILE = Path("data/news_cache.json")
CACHE_DURATION_HOURS = 4

class NewsFilter:
    def __init__(self):
        self.enabled = settings.ENABLE_NEWS_FILTER
        self.news_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json" # Hardcoded default as fallback or move to settings
        self.block_before = settings.NEWS_BLOCK_MINUTES_BEFORE
        self.block_after = settings.NEWS_BLOCK_MINUTES_AFTER
        self.calendar = []
        self.use_mt5_calendar = True # Default or settings
        self.mt5_calendar_path = settings.DATA_DIR / 'mt5_calendar.json'
        self._load_cache()

    def _load_cache(self):
        """Carrega o calendário do cache ou faz download se expirado."""
        # ✅ v5.1: Prioriza MT5 Calendar se habilitado
        if self.use_mt5_calendar and self.mt5_calendar_path.exists():
            try:
                mtime = datetime.fromtimestamp(self.mt5_calendar_path.stat().st_mtime)
                refresh_minutes = 30 # Default or settings
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
                    logger.info(f"✅ Cache carregado: {len(self.calendar)} eventos")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao ler cache: {e}")
        
        # Download se necessário
        self._download_calendar()

    def _download_calendar(self):
        """Faz download do calendário econômico."""
        try:
            response = requests.get(self.news_url, timeout=30)
            response.raise_for_status()
            self.calendar = response.json()
            
            # Salva cache
            CACHE_FILE.parent.mkdir(exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.calendar, f, indent=2)
            
            logger.info(f"✅ Calendário baixado: {len(self.calendar)} eventos")
            
        except Exception as e:
            logger.error(f"❌ Erro ao baixar calendário: {e}")
            self.calendar = []

    def is_news_blocked(self, symbol: str, check_time: datetime = None) -> tuple[bool, str]:
        """
        Verifica se há notícias que bloqueiam trading para o símbolo.
        
        Args:
            symbol: Símbolo do par (ex: "EURUSD")
            check_time: Tempo para verificação (default: agora)
            
        Returns:
            tuple[bool, str]: (bloqueado, motivo)
        """
        if not self.enabled:
            return False, "News filter disabled"
        
        if check_time is None:
            check_time = datetime.now()
        
        # Extrai moedas do par
        base_currency = symbol[:3].upper()
        quote_currency = symbol[3:6].upper() if len(symbol) >= 6 else ""
        
        # Verifica eventos relevantes
        for event in self.calendar:
            try:
                # Parse do tempo do evento
                event_time = datetime.fromisoformat(event.get('date', '').replace('Z', '+00:00'))
                event_time = event_time.replace(tzinfo=None)  # Remove timezone
                
                # Verifica se está no período de bloqueio
                time_diff = abs((check_time - event_time).total_seconds() / 60)
                
                if time_diff <= self.block_before or time_diff <= self.block_after:
                    # Verifica se a moeda está envolvida
                    currency = event.get('currency', '').upper()
                    impact = event.get('impact', '').upper()
                    
                    if (currency == base_currency or currency == quote_currency) and impact in ['HIGH', 'MEDIUM']:
                        title = event.get('title', 'Unknown Event')
                        return True, f"News: {title} ({currency}) - Impact: {impact}"
                        
            except Exception as e:
                logger.debug(f"Erro ao processar evento: {e}")
                continue
        
        return False, "No blocking news"

# Global instance for backward compatibility
news_filter = NewsFilter()

# Backward compatibility functions
def is_news_blocked(symbol: str, check_time: datetime = None) -> tuple[bool, str]:
    """Backward compatibility function."""
    return news_filter.is_news_blocked(symbol, check_time)

print("✅ News filter migrated to src-layout")