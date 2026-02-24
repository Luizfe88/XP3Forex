"""Core bot functionality for XP3 PRO FOREX"""

import sys
import os
import time
import threading
import logging
import json
import signal
import traceback
import subprocess
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

from .config import *
from ..utils.mt5_utils import *
from ..utils.indicators import *
from ..utils.calculations import *
from ..utils.data_utils import *

logger = logging.getLogger("XP3_BOT")

# Global variables for bot state
FAST_LOOP_ACTIVE = False
WATCHDOG_ACTIVE = False
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
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.positions: Dict[str, Position] = {}
        self.signals: deque = deque(maxlen=1000)
        self.performance_metrics = defaultdict(float)
        self.is_running = False
        self.lock = RLock()
        
        # Setup logging
        self.setup_logging()
        
        logger.info("🚀 XP3 PRO FOREX BOT v4.2 INSTITUCIONAL Inicializado")
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Arquivo de configuração não encontrado: {self.config_path}")
                return self.create_default_config()
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return self.create_default_config()
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            "mt5": {
                "login": 12345678,
                "password": "password",
                "server": "Broker-Demo",
                "path": MT5_TERMINAL_PATH
            },
            "trading": {
                "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
                "timeframes": [15, 60, 240],
                "risk_per_trade": 0.02,
                "max_positions": 5
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get("logging", {}).get("level", "INFO"))
        
        # Create logs directory
        LOGS_DIR.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.handlers.RotatingFileHandler(
                    LOGS_DIR / "xp3_forex.log",
                    maxBytes=LOG_MAX_FILE_SIZE_MB * 1024 * 1024,
                    backupCount=LOG_BACKUP_COUNT,
                    encoding="utf-8"
                )
            ]
        )
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            mt5_config = self.config.get("mt5", {})
            return initialize_mt5(
                mt5_config.get("login"),
                mt5_config.get("password"),
                mt5_config.get("server"),
                mt5_config.get("path", MT5_TERMINAL_PATH)
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar MT5: {e}")
            return False
    
    def analyze_symbol(self, symbol: str, timeframe: int) -> Optional[TradeSignal]:
        """Analyze a symbol and generate trading signal"""
        try:
            # Get historical data
            df = get_rates(symbol, timeframe, 100)
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
                # This is a simplified version - you would implement the actual order sending
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
                    magic_number=12345  # Example magic number
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
        """Update open positions"""
        try:
            with self.lock:
                closed_positions = []
                
                for symbol, position in self.positions.items():
                    # Get current price
                    symbol_info = get_symbol_info(symbol)
                    if symbol_info is None:
                        continue
                    
                    # Update position
                    current_price = symbol_info.get('ask', position.entry_price) if position.order_type == "BUY" else symbol_info.get('bid', position.entry_price)
                    position.current_price = current_price
                    
                    # Calculate profit
                    if position.order_type == "BUY":
                        position.pips = (current_price - position.entry_price) / get_pip_size(symbol)
                    else:
                        position.pips = (position.entry_price - current_price) / get_pip_size(symbol)
                    
                    position.profit = position.pips * get_tick_value(symbol, position.volume)
                    
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
    
    def fast_loop(self):
        """Main trading loop"""
        global FAST_LOOP_ACTIVE
        
        try:
            FAST_LOOP_ACTIVE = True
            logger.info("🚀 Fast Loop iniciado")
            
            symbols = self.config.get("trading", {}).get("symbols", ["EURUSD", "GBPUSD", "USDJPY"])
            timeframes = self.config.get("trading", {}).get("timeframes", [15, 60, 240])
            
            while not SHUTDOWN_EVENT.is_set():
                try:
                    # Update positions
                    self.update_positions()
                    
                    # Analyze symbols
                    for symbol in symbols:
                        for timeframe in timeframes:
                            signal = self.analyze_symbol(symbol, timeframe)
                            if signal:
                                self.execute_trade(signal)
                    
                    # Sleep for a short period
                    time.sleep(1)  # 1 second between iterations
                    
                except Exception as e:
                    logger.error(f"Erro no fast_loop: {e}")
                    time.sleep(5)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Erro fatal no fast_loop: {e}")
        finally:
            FAST_LOOP_ACTIVE = False
            logger.info("🛑 Fast Loop finalizado")
    
    def watchdog(self):
        """Watchdog thread to monitor bot health"""
        global WATCHDOG_ACTIVE
        
        try:
            WATCHDOG_ACTIVE = True
            logger.info("🐕 Watchdog iniciado")
            
            while not SHUTDOWN_EVENT.is_set():
                try:
                    # Check MT5 connection
                    if not check_mt5_connection():
                        logger.warning("Conexão MT5 perdida, tentando reconectar...")
                        self.initialize_mt5()
                    
                    # Check fast loop
                    if not FAST_LOOP_ACTIVE:
                        logger.warning("Fast Loop inativo, reiniciando...")
                        threading.Thread(target=self.fast_loop, name="FastLoop", daemon=True).start()
                    
                    # Sleep for 30 seconds
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Erro no watchdog: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            logger.error(f"Erro fatal no watchdog: {e}")
        finally:
            WATCHDOG_ACTIVE = False
            logger.info("🛑 Watchdog finalizado")
    
    def start(self):
        """Start the bot"""
        try:
            logger.info("🚀 Iniciando XP3 PRO FOREX BOT")
            
            # Initialize MT5
            if not self.initialize_mt5():
                logger.error("Falha ao conectar ao MT5")
                return False
            
            # Initialize database
            init_database()
            
            # Start watchdog
            threading.Thread(target=self.watchdog, name="Watchdog", daemon=True).start()
            
            # Start fast loop
            threading.Thread(target=self.fast_loop, name="FastLoop", daemon=True).start()
            
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