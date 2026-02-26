import threading
import time
import queue
import logging
import pandas as pd
from typing import List, Tuple, Dict
from ..utils.mt5_utils import get_rates
from .symbol_manager import SymbolManager

logger = logging.getLogger(__name__)

class DataFeeder(threading.Thread):
    """
    Data Feeder Dedicado (Produtor).
    Coleta dados de mercado e coloca numa fila para a estratégia consumir.
    """
    def __init__(self, 
                 data_queue: queue.Queue, 
                 symbols: List[str], 
                 timeframes: List[int],
                 bot_instance):
        super().__init__(name="DataFeeder", daemon=True)
        self.data_queue = data_queue
        self.symbols = symbols
        self.timeframes = timeframes
        self.bot = bot_instance
        self.running = True
        self.symbol_manager = SymbolManager()
        
    def run(self):
        logger.info("Data Feeder iniciado")
        while self.running:
            # Verifica se o bot está pausado (ex: por desconexão)
            if not self.bot.is_trading_active:
                time.sleep(1)
                continue
                
            for symbol in self.symbols:
                # Verifica Circuit Breaker
                if not self.symbol_manager.is_available(symbol):
                    continue
                
                for tf in self.timeframes:
                    try:
                        # Coleta dados
                        # Nota: get_rates já tem retry logic interno, mas aqui
                        # tratamos o sucesso/falha para o Circuit Breaker
                        df = get_rates(symbol, tf, 100)
                        
                        if df is not None and not df.empty:
                            self.symbol_manager.report_success(symbol)
                            
                            # Coloca na fila (non-blocking ou com timeout curto para não travar o feeder)
                            try:
                                self.data_queue.put((symbol, tf, df), timeout=1)
                            except queue.Full:
                                logger.debug("Fila de dados cheia, pulando atualização...")
                        else:
                            self.symbol_manager.report_failure(symbol)
                            
                    except Exception as e:
                        logger.error(f"Erro no Data Feeder para {symbol}: {e}")
                        self.symbol_manager.report_failure(symbol)
            
            # Pequena pausa para não saturar CPU se a lista for pequena
            time.sleep(0.1)

    def stop(self):
        self.running = False
