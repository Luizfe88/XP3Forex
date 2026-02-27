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

from .settings import settings
from ..utils.mt5_utils import *
from ..utils.indicators import *
from ..utils.calculations import *
from ..utils.data_utils import *

# New Components
from ..mt5.symbol_manager import SymbolManager
from .health_monitor import HealthMonitor
from .data_feeder import DataFeeder

logger = logging.getLogger("XP3_BOT")

SHUTDOWN_EVENT = Event()

@dataclass
class TradeSignal:
    """Represents a trading signal"""
    symbol: str
    order_type: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    order_type: str
    volume: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    profit: float
    pips: float
    open_time: datetime
    magic_number: int

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
        
        logger.info(f"🚀 XP3 PRO FOREX BOT v5.0 INSTITUCIONAL Inicializado")
        logger.info(f"Symbols: {self.symbols}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        
        # Create logs directory
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.handlers.RotatingFileHandler(
                    settings.get_log_file(),
                    maxBytes=50 * 1024 * 1024,
                    backupCount=3,
                    encoding="utf-8"
                )
            ]
        )
    
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

            # 2. Spread Check
            symbol_info = get_symbol_info(symbol)
            if symbol_info:
                spread = symbol_info.get('spread', 0)
                if spread > self.MAX_SPREAD:
                    logger.info(f"Spread alto para {symbol}: {spread} > {self.MAX_SPREAD}. Ignorando.")
                    return None

            if df is None or len(df) < 50:
                logger.warning(f"Dados insuficientes para {symbol}")
                return None
            
            # Calculate indicators
            df['ema_fast'] = calculate_ema(df['close'], EMA_FAST_PERIOD)
            df['ema_slow'] = calculate_ema(df['close'], EMA_SLOW_PERIOD)
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['adx'] = calculate_adx(df['high'], df['low'], df['close'], ADX_PERIOD)
            df['atr'] = calculate_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Get symbol configuration
            symbol_config = ELITE_CONFIG.get(symbol, {})
            adx_threshold = symbol_config.get("adx_threshold", 25)
            rsi_oversold = symbol_config.get("rsi_oversold", 30)
            rsi_overbought = symbol_config.get("rsi_overbought", 70)
            
            # Generate signals
            signal = None
            
            # BUY signal
            if (df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] and
                df['adx'].iloc[-1] > adx_threshold and
                df['rsi'].iloc[-1] < rsi_overbought and
                df['close'].iloc[-1] > df['ema_fast'].iloc[-1]):
                
                sl, tp = calculate_sl_tp(symbol, current_price, "BUY", df['atr'].iloc[-1])
                volume = calculate_lot_size(symbol, 100, 20)  # Exemplo: risco $100, SL 20 pips
                
                signal = TradeSignal(
                    symbol=symbol,
                    order_type="BUY",
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    volume=volume,
                    confidence=0.75,
                    reason=f"XP3 v4.2 - ADX: {df['adx'].iloc[-1]:.1f}, RSI: {df['rsi'].iloc[-1]:.1f}"
                )
            
            # SELL signal
            elif (df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1] and
                  df['adx'].iloc[-1] > adx_threshold and
                  df['rsi'].iloc[-1] > rsi_oversold and
                  df['close'].iloc[-1] < df['ema_fast'].iloc[-1]):
                
                sl, tp = calculate_sl_tp(symbol, current_price, "SELL", df['atr'].iloc[-1])
                volume = calculate_lot_size(symbol, 100, 20)  # Exemplo: risco $100, SL 20 pips
                
                signal = TradeSignal(
                    symbol=symbol,
                    order_type="SELL",
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    volume=volume,
                    confidence=0.75,
                    reason=f"XP3 v4.2 - ADX: {df['adx'].iloc[-1]:.1f}, RSI: {df['rsi'].iloc[-1]:.1f}"
                )
            
            return signal
            
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
                if len(self.positions) >= self.config.get("trading", {}).get("max_positions", 5):
                    logger.info("Máximo de posições atingido")
                    return False
                
                # Execute trade logic here
                logger.info(f"Executando trade: {signal.symbol} {signal.order_type} @ {signal.entry_price}")
                
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
                    magic_number=12345
                )
                
                self.positions[signal.symbol] = position
                self.signals.append(signal)
                
                # Save to database
                trade_data = {
                    'symbol': signal.symbol,
                    'order_type': signal.order_type,
                    'volume': signal.volume,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'entry_time': signal.timestamp.isoformat(),
                    'status': 'open',
                    'magic_number': 12345,
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
                    else:
                        position.pips = (position.entry_price - current_price) / point
                    
                    position.profit = position.pips * self.symbol_manager.get_tick_value(symbol)
                    
                    # Check if position should be closed
                    if (position.order_type == "BUY" and current_price <= position.stop_loss) or \
                       (position.order_type == "SELL" and current_price >= position.stop_loss):
                        # Stop loss hit
                        logger.info(f"Stop loss atingido para {symbol}")
                        closed_positions.append(symbol)
                        
                    elif (position.order_type == "BUY" and current_price >= position.take_profit) or \
                         (position.order_type == "SELL" and current_price <= position.take_profit):
                        # Take profit hit
                        logger.info(f"Take profit atingido para {symbol}")
                        closed_positions.append(symbol)
                
                # Remove closed positions
                for symbol in closed_positions:
                    if symbol in self.positions:
                        del self.positions[symbol]
                        
        except Exception as e:
            logger.error(f"Erro ao atualizar posições: {e}")
    
    def consumer_loop(self):
        """Consumes market data from queue and processes strategy"""
        logger.info("🔥 Consumer Loop iniciado")
        
        while not SHUTDOWN_EVENT.is_set():
            try:
                # Update positions
                self.update_positions()
                
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

    def start(self):
        """Start the bot"""
        try:
            logger.info("🚀 Iniciando XP3 PRO FOREX BOT")
            
            # Initialize MT5
            if not self.initialize_mt5():
                logger.error("Falha ao conectar ao MT5")
                return False
            
            self.is_connected = True
            self.is_trading_active = True
            
            # Initialize database
            init_database()
            
            # Initialize Market Data (Validate symbols)
            validated_symbols = initialize_market_data(self.symbols)
            if not validated_symbols:
                logger.error("Nenhum símbolo válido encontrado! Abortando.")
                return False
                
            self.symbols = validated_symbols
            self.data_feeder.symbols = self.symbols
            logger.info(f"Símbolos validados para trading: {self.symbols}")
            
            # Start Threads
            self.health_monitor.start()
            self.data_feeder.start()
            
            # Start Consumer Loop (in a thread or main thread?)
            # Usually main thread can be the consumer or just wait.
            # Let's put consumer in a thread so main can handle signals/UI if needed
            self.consumer_thread = threading.Thread(target=self.consumer_loop, name="ConsumerLoop", daemon=True)
            self.consumer_thread.start()
            
            self.is_running = True
            logger.info("✅ Bot iniciado com sucesso")
            
            # Keep main thread alive
            try:
                while self.is_running and not SHUTDOWN_EVENT.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupção do teclado detectada")
                self.stop()
                
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            return False
    
    def stop(self):
        """Stop the bot"""
        try:
            logger.info("🛑 Parando XP3 PRO FOREX BOT")
            self.is_running = False
            SHUTDOWN_EVENT.set()
            
            # Stop feeder
            if hasattr(self, 'data_feeder'):
                self.data_feeder.stop()
            
            # Stop monitor
            if hasattr(self, 'health_monitor'):
                self.health_monitor.stop()
            
            # Shutdown MT5 worker
            mt5_shutdown_worker()
            
            # Shutdown MT5
            mt5.shutdown()
            
            logger.info("✅ Bot parado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao parar bot: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Sinal {signum} recebido, encerrando...")
    SHUTDOWN_EVENT.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main function"""
    try:
        # Create bot instance
        bot = XP3Bot()
        
        # Start the bot
        bot.start()
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
