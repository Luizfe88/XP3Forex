import threading
import time
import logging
import MetaTrader5 as mt5
from ..utils.mt5_utils import mt5_exec
from .settings import settings

logger = logging.getLogger(__name__)

class HealthMonitor(threading.Thread):
    """
    Monitor de saúde ativo que verifica conexão MT5 e gerencia estado do bot.
    """
    def __init__(self, bot_instance, interval=30):
        super().__init__(name="HealthMonitor", daemon=True)
        self.bot = bot_instance
        self.interval = interval
        self.running = True
        self._last_ping = 0
        
    def run(self):
        logger.info("Health Monitor iniciado")
        while self.running:
            try:
                self._check_health()
            except Exception as e:
                logger.error(f"Erro no Health Monitor: {e}")
            
            time.sleep(self.interval)

    def _check_health(self):
        # Ping MT5
        info = mt5_exec(mt5.terminal_info)
        
        if info is None:
            logger.critical("FALHA NO HEALTH CHECK: MT5 não responde!")
            self._handle_disconnection()
        elif not info.connected:
            logger.critical("FALHA NO HEALTH CHECK: MT5 desconectado do servidor!")
            self._handle_disconnection()
        else:
            # Conexão OK
            if not self.bot.is_connected:
                logger.info("Conexão MT5 restabelecida.")
                self.bot.is_connected = True
                self.bot.resume_trading()

    def _handle_disconnection(self):
        if self.bot.is_connected:
            self.bot.is_connected = False
            self.bot.pause_trading()
            
        # Tenta reconectar com Exponential Backoff
        logger.info("Tentando reconexão com Exponential Backoff...")
        
        max_retries = 5
        base_delay = 5
        
        from ..utils.mt5_utils import initialize_mt5
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Tentativa {attempt+1}/{max_retries} em {delay}s...")
                time.sleep(delay)
                
                # Usa configurações do settings global
                if initialize_mt5(
                    settings.MT5_LOGIN,
                    settings.MT5_PASSWORD,
                    settings.MT5_SERVER,
                    settings.MT5_PATH
                ):
                    logger.info("✅ Reconexão bem sucedida!")
                    self.bot.is_connected = True
                    self.bot.resume_trading()
                    return
                else:
                    logger.warning(f"Falha na reconexão (Tentativa {attempt+1})")
                    
            except Exception as e:
                logger.error(f"Erro ao tentar reconectar: {e}")
                
        logger.critical("❌ Falha definitiva na reconexão após várias tentativas.")

    def stop(self):
        self.running = False
