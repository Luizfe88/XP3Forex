"""Core bot functionality for XP3 PRO FOREX"""

import sys
import os
import time
import threading
import logging
import logging.handlers
import json
import signal
import traceback
import queue
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock, RLock, Event
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# Fix Windows encoding
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .settings import settings
from .models import TradeSignal, Position
from ..utils.mt5_utils import *
from ..utils.indicators import *
from ..utils.calculations import *
from ..utils.data_utils import *

# New Components
from ..mt5.symbol_manager import SymbolManager
from .health_monitor import HealthMonitor
from .data_feeder import DataFeeder
from .rate_cache import RateCache
from ..strategies.adaptive_ema_rsi import AdaptiveEmaRsiStrategy

logger = logging.getLogger("XP3_BOT")

SHUTDOWN_EVENT = Event()

class XP3Bot:
    """Main bot class"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.signals: deque = deque(maxlen=1000)
        self.performance_metrics = defaultdict(float)
        self.is_running = False
        self.lock = RLock()
        
        # State Flags
        self.is_connected = False
        self.is_trading_active = False
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Core Components
        self.symbol_manager = SymbolManager()
        self.data_queue = queue.Queue(maxsize=100)
        self.cache = RateCache()
        
        # Circuit Breaker & Risk Config
        self.circuit_breaker = defaultdict(int)
        self.CIRCUIT_BREAKER_THRESHOLD = 5
        self.MAX_SPREAD = getattr(settings, 'MAX_SPREAD', 50)  # Default 50 points
        
        # Get configured symbols and timeframes
        self.symbols = settings.symbols_list
        self.timeframes = settings.timeframes_list
        
        # Initialize Threads
        self.health_monitor = HealthMonitor(self)
        self.data_feeder = DataFeeder(self.data_queue, self.symbols, self.timeframes, self)
        
        # Strategy
        self.strategy = AdaptiveEmaRsiStrategy(self)

        # Rich Console for Why Report
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None
        self.last_report_time = 0
        
        logger.info(f"🚀 XP3 PRO FOREX BOT v5.0 INSTITUCIONAL Inicializado")
        logger.info(f"Symbols: {self.symbols}")
        
    def setup_logging(self):
        """Setup logging configuration with 3-hour rotation"""
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        
        # Create logs directory
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure logging with TimedRotatingFileHandler
        # Rotate every 3 hours ('H', interval=3)
        # Keep backup for 7 days (8 backups per day * 7 days = 56)
        
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s")
        
        # Root Logger config
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
            
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File Handler (Timed Rotation)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=settings.get_log_file(),
            when='H',
            interval=3,
            backupCount=56, # 7 days worth of logs
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            # Se já estiver conectado pelo HealthMonitor ou externo, retorna True
            if mt5.terminal_info() is not None:
                return True
                
            return initialize_mt5(
                settings.MT5_LOGIN,
                settings.MT5_PASSWORD,
                settings.MT5_SERVER,
                settings.MT5_PATH
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar MT5: {e}")
            return False
    
    def analyze_symbol(self, symbol: str, timeframe: int, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Analyze a symbol and generate trading signal using provided DataFrame"""
        try:
            # 1. Circuit Breaker Check
            if self.circuit_breaker[symbol] >= self.CIRCUIT_BREAKER_THRESHOLD:
                # Log esporádico para não floodar
                if np.random.random() < 0.05:
                    logger.warning(f"Circuit Breaker ATIVO para {symbol} (Falhas: {self.circuit_breaker[symbol]}). Trading pausado.")
                return None

            # 2. Spread Check (Silent & Smart)
            # Use SymbolManager's silent check
            if not self.symbol_manager.check_spread(symbol):
                # Log handled internally by check_spread (silent or summary)
                return None

            if df is None or len(df) < 50:
                logger.warning(f"Dados insuficientes para {symbol}")
                return None
            
            # Use Strategy to Analyze
            return self.strategy.analyze(symbol, timeframe, df)
            
        except Exception as e:
            logger.error(f"Erro ao analisar {symbol}: {e}")
            return None
    
    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trade based on signal"""
        try:
            with self.lock:
                # Check if we already have a position for this symbol
                if signal.symbol in self.positions:
                    logger.info(f"Posição já existe para {signal.symbol}")
                    return False
                
                # Check max positions
                if len(self.positions) >= settings.MAX_POSITIONS:
                    logger.info("Máximo de posições atingido")
                    return False
                
                # Execute trade logic here
                logger.info(f"Executando trade: {signal.symbol} {signal.order_type} @ {signal.entry_price}")
                
                initial_sl_dist = abs(signal.entry_price - signal.stop_loss)
                
                # Create position
                position = Position(
                    symbol=signal.symbol,
                    order_type=signal.order_type,
                    volume=signal.volume,
                    entry_price=signal.entry_price,
                    current_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    profit=0,
                    pips=0,
                    open_time=signal.timestamp,
                    magic_number=settings.MAGIC_NUMBER,
                    initial_sl_dist=initial_sl_dist,
                    partial_taken=False
                )
                
                self.positions[signal.symbol] = position
                self.signals.append(signal)
                
                # Save to database (Placeholder)
                trade_data = {
                    'symbol': signal.symbol,
                    'order_type': signal.order_type,
                    'volume': signal.volume,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'entry_time': signal.timestamp.isoformat(),
                    'status': 'open',
                    'magic_number': settings.MAGIC_NUMBER,
                    'comment': signal.reason
                }
                save_trade(trade_data)
                
                return True
                
        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}")
            return False
    
    def update_positions(self):
        """Update open positions using cached SymbolManager"""
        try:
            with self.lock:
                closed_positions = []
                
                for symbol, position in self.positions.items():
                    # Get current price (fresh tick)
                    tick = self.symbol_manager.get_tick(symbol)
                    if tick is None:
                        continue
                    
                    # Update position
                    current_price = tick.ask if position.order_type == "BUY" else tick.bid
                    position.current_price = current_price
                    
                    # Calculate profit
                    point = self.symbol_manager.get_point(symbol)
                    
                    if position.order_type == "BUY":
                        position.pips = (current_price - position.entry_price) / point
                        profit_r = (current_price - position.entry_price) / position.initial_sl_dist if position.initial_sl_dist > 0 else 0
                    else:
                        position.pips = (position.entry_price - current_price) / point
                        profit_r = (position.entry_price - current_price) / position.initial_sl_dist if position.initial_sl_dist > 0 else 0
                    
                    position.profit = position.pips * self.symbol_manager.get_tick_value(symbol)
                    
                    # --- MANAGEMENT ---
                    
                    # 1. Partial Take Profit (1.5R)
                    if not position.partial_taken and profit_r >= 1.5:
                        logger.info(f"Taking Partial Profit (1.5R) for {symbol}")
                        # Close 40%
                        # Implementation Note: In real MT5, we would order_close_partial. 
                        # Here we simulate by reducing volume.
                        position.volume *= 0.6 # Remaining 60%
                        position.partial_taken = True
                    
                    # 2. Trailing Stop (2xATR)
                    # We use initial_sl_dist as approximation for 2xATR
                    ts_dist = position.initial_sl_dist 
                    
                    if position.order_type == "BUY":
                        new_sl = current_price - ts_dist
                        if new_sl > position.stop_loss:
                            position.stop_loss = new_sl
                            logger.debug(f"Trailing SL moved to {new_sl} for {symbol}")
                    else:
                        new_sl = current_price + ts_dist
                        if new_sl < position.stop_loss:
                            position.stop_loss = new_sl
                            logger.debug(f"Trailing SL moved to {new_sl} for {symbol}")

                    # --- CHECK CLOSURE ---

                    # Check if position should be closed
                    if (position.order_type == "BUY" and current_price <= position.stop_loss) or \
                       (position.order_type == "SELL" and current_price >= position.stop_loss):
                        # Stop loss hit
                        logger.info(f"Stop loss atingido para {symbol}")
                        closed_positions.append(symbol)
                        # Update Daily Stats
                        if position.profit > 0:
                            self.strategy.daily_stats["wins"] += 1
                        else:
                            self.strategy.daily_stats["losses"] += 1
                        self.strategy.daily_stats["profit"] += position.profit
                        
                    elif (position.order_type == "BUY" and current_price >= position.take_profit) or \
                         (position.order_type == "SELL" and current_price <= position.take_profit):
                        # Take profit hit
                        logger.info(f"Take profit atingido para {symbol}")
                        closed_positions.append(symbol)
                        # Update Daily Stats
                        self.strategy.daily_stats["wins"] += 1
                        self.strategy.daily_stats["profit"] += position.profit
                
                # Remove closed positions
                for symbol in closed_positions:
                    if symbol in self.positions:
                        del self.positions[symbol]
                        
        except Exception as e:
            logger.error(f"Erro ao atualizar posições: {e}")

    def close_all_positions(self):
        """Close all open positions (Kill Switch)"""
        with self.lock:
            for symbol in list(self.positions.keys()):
                logger.warning(f"Closing {symbol} due to Kill Switch")
                del self.positions[symbol]
            logger.info("All positions closed.")
    
    def consumer_loop(self):
        """Consumes market data from queue and processes strategy"""
        logger.info("🔥 Consumer Loop iniciado")
        
        while not SHUTDOWN_EVENT.is_set():
            try:
                # Update positions
                self.update_positions()

                # --- Periodic Why Report Scan (Visual) ---
                if HAS_RICH and self.console and (time.time() - self.last_report_time > 60):
                    self.last_report_time = time.time()
                    if self.is_trading_active and self.symbols:
                        # logger.info("🔍 Executando Scan Visual de Mercado (Why Report)...")
                        
                        # Use a separate thread or limit symbols to avoid blocking consumer loop too much
                        # For now, let's just scan top 5 symbols or randomize
                        # But user wants to see it working.
                        
                        signals_found = 0
                        symbols_to_scan = self.symbols[:10] # Limit to first 10 for performance in loop
                        
                        # Only print header if we find something or once every few cycles
                        # self.console.print("[dim]🔍 Escaneando mercado para oportunidades...[/]")
                        
                        self.console.print(f"[bold cyan]🔍 Iniciando análise visual de {len(symbols_to_scan)} símbolos...[/]")

                        for symbol in symbols_to_scan:
                            try:
                                panel, conf = self.strategy.get_why_report(symbol)
                                if panel and conf > 60:
                                    self.console.print(panel)
                                    signals_found += 1
                            except Exception as e:
                                # logger.error(f"Erro no Why Report para {symbol}: {e}")
                                pass
                        
                        if signals_found == 0:
                             # Exibir mensagem de progresso a cada X ciclos ou quando solicitado
                             self.console.print(f"[dim]Scan concluído: Nenhum sinal >60% encontrado em {len(symbols_to_scan)} símbolos monitorados. Aguardando...[/]")
                
                # Process Data Queue
                try:
                    # Non-blocking get or short timeout
                    symbol, timeframe, df = self.data_queue.get(timeout=1)
                    
                    # Analyze
                    signal = self.analyze_symbol(symbol, timeframe, df)
                    if signal:
                        self.execute_trade(signal)
                        
                    self.data_queue.task_done()
                    
                except queue.Empty:
                    pass
                    
            except Exception as e:
                logger.error(f"Erro no consumer_loop: {e}")
                time.sleep(1)
                
        logger.info("🛑 Consumer Loop finalizado")

    def pause_trading(self):
        """Pausa o trading (consumer e feeder)"""
        logger.warning("⏸️ Trading PAUSADO")
        self.is_trading_active = False

    def resume_trading(self):
        """Retoma o trading"""
        logger.info("▶️ Trading RETOMADO")
        self.is_trading_active = True

    def initialize_market_data(self) -> List[str]:
        """Validate symbols and initialize market data"""
        logger.info("🔄 Inicializando Filtro de Símbolos Institucional...")
        
        # Use SymbolManager's smart filter
        valid_symbols = self.symbol_manager.get_tradable_symbols()
        
        if not valid_symbols:
            logger.error("❌ Nenhum símbolo válido encontrado após filtragem!")
            return []
            
        logger.info(f"✅ Filtro concluído: {len(valid_symbols)} símbolos aprovados para trading.")
        logger.debug(f"Símbolos ativos: {valid_symbols}")
        
        # Update bot symbols list
        self.symbols = valid_symbols
        
        # Also update DataFeeder if it was already initialized with old list
        if hasattr(self, 'data_feeder'):
            self.data_feeder.symbols = self.symbols
        
        return valid_symbols

    def start(self):
        """Start the bot"""
        try:
            logger.info("🚀 Iniciando XP3 PRO FOREX BOT")
            
            # Initialize MT5
            if not self.initialize_mt5():
                logger.error("Falha ao conectar ao MT5")
                return False
            
            # Initialize Market Data (Apply Filters)
            if not self.initialize_market_data():
                logger.error("Falha ao inicializar dados de mercado. Abortando.")
                return False

            self.is_connected = True
            self.is_trading_active = True
            
            # Start Threads
            self.health_monitor.start()
            
            # Start Feeder Thread
            feeder_thread = threading.Thread(target=self.data_feeder.run, daemon=True)
            feeder_thread.start()
            
            # Start Consumer Loop (Main Thread or Separate)
            # If running as daemon or service, this might block.
            # In CLI mode, we usually loop.
            self.consumer_loop()
            
        except KeyboardInterrupt:
            logger.info("🛑 Parando bot...")
            SHUTDOWN_EVENT.set()
        except Exception as e:
            logger.critical(f"Erro fatal: {e}")
            traceback.print_exc()
