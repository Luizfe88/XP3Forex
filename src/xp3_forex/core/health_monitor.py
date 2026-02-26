import threading
import time
import logging
import MetaTrader5 as mt5
from ..utils.mt5_utils import mt5_exec

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
            
        # Tenta reconectar
        logger.info("Tentando reconexão...")
        # A lógica de reconexão pode envolver mt5.shutdown() e mt5.initialize()
        # Mas mt5_utils.initialize_mt5 já faz isso.
        # Vamos tentar re-inicializar usando a config do bot
        try:
            mt5_config = self.bot.config.get("mt5", {})
            from ..utils.mt5_utils import initialize_mt5
            if initialize_mt5(
                mt5_config.get("login"),
                mt5_config.get("password"),
                mt5_config.get("server"),
                mt5_config.get("path")
            ):
                logger.info("Reconexão bem sucedida!")
            else:
                logger.error("Falha na reconexão.")
        except Exception as e:
            logger.error(f"Erro ao tentar reconectar: {e}")

    def stop(self):
        self.running = False
