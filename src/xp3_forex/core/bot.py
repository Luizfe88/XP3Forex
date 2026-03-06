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
from .trade_executor import trade_executor
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
from ..optimization.learner import DailyLearner
from ..reporting.daily_report import DailyReportGenerator
from ..utils.telegram_utils import send_telegram_message, send_telegram_document
from ..utils.process_watcher import ProcessWatcher

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
        self.last_cooldown_log = {}  # Controle de log por símbolo
        self._last_maintenance_date = None
        
        # Components for Learning
        self.learner = DailyLearner(self.symbols)
        self.report_gen = DailyReportGenerator()
        self.last_summary_table_time = 0  # Timer para o relatório de 5 min
        
        # Batching Logic (NEW)
        self.pending_signals: List[TradeSignal] = []
        
        logger.info(f"🚀 XP3 PRO FOREX BOT v5.0 INSTITUCIONAL Inicializado")
        logger.info(f"Symbols: {self.symbols}")
        
    def setup_logging(self):
        """Setup logging configuration with 3-hour rotation"""
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        
        # Create logs directory
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
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
        
        # File Handler (Timed Rotation 3h)
        # Naming convention: logger-info-dd-mm-yy.txt (Base)
        # However, TimedRotatingFileHandler appends suffix automatically (e.g., .2023-10-27_12-00)
        # To strictly follow "creates a new one every 3h", we use 'H', interval=3.
        # To control the filename format exactly, we need a custom namer or just accept the suffix.
        # User requested: "logger-info-dd-mm-aa.txt"
        
        current_date = datetime.now().strftime("%d-%m-%y")
        base_filename = settings.LOGS_DIR / f"logger-info-{current_date}.txt"
        
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=base_filename,
            when='H',
            interval=3,
            backupCount=56, # 7 days worth of logs (8 files per day)
            encoding='utf-8'
        )
        
        # Customize suffix to include time for clarity on rotation, though base name has date.
        # If we want strictly new files with date in name, we might need to rely on the rotation suffix.
        # Standard suffix is YYYY-MM-DD_HH-MM.
        file_handler.suffix = "%Y-%m-%d_%H-%M.log"
        
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            # Se já estiver conectado pelo HealthMonitor ou externo, retorna True
            if mt5.terminal_info() is not None:
                term_info = mt5.terminal_info()
                if not term_info.connected:
                    logger.error("❌ MT5 detectado mas SEM CONEXÃO com o servidor.")
                    return False
                if not term_info.trade_allowed:
                    logger.error("❌ MT5 detectado mas 'ALGO TRADING' está DESLIGADO no terminal.")
                    return False
                return True
            
            # Check if MT5 is already open manually
            if settings.AUTO_CLOSE_ON_MT5 and ProcessWatcher.is_mt5_running():
                logger.critical("🛑 Detectado que o MetaTrader 5 já está aberto manualmente. O robô não será iniciado para evitar conflitos.")
                return False

            success = initialize_mt5(
                settings.MT5_LOGIN,
                settings.MT5_PASSWORD,
                settings.MT5_SERVER,
                settings.MT5_PATH
            )

            if success:
                term_info = mt5.terminal_info()
                if term_info and not term_info.trade_allowed:
                    logger.critical("⚠️ MT5 Conectado, mas o botão 'ALGO TRADING' está DESATIVADO!")
                elif term_info:
                    logger.info("✅ MT5 Conectado e Algo Trading habilitado.")

            return success
        except Exception as e:
            logger.error(f"Erro ao inicializar MT5: {e}")
            return False
    
            return None

    def is_in_rollover_pause(self) -> bool:
        """
        🚫 Rollover Pause: Verifica se o robô deve pausar novas entradas.
        Bloqueio: Quarta 17:00 até Quinta 01:00 (BRT).
        """
        now = datetime.now()
        wd = now.weekday()
        hr = now.hour
        
        # Quarta das 17:00 em diante
        if wd == 2 and hr >= 17:
            return True
        # Quinta até as 01:00
        if wd == 3 and hr < 1:
            return True
            
        return False
        
    def is_friday_block_active(self) -> bool:
        """
        🚫 Friday Block: Impede o envio de NOVAS ordens na Sexta-feira
        após as 15:00 (Horário do Robô/BRT) para evitar risco de fim de semana (Gaps e spreads).
        """
        now = datetime.now()
        # Sexta-feira é 4 em Python weekday()
        if now.weekday() == 4 and now.hour >= 15:
            return True
        return False
    
    def check_friday_close_guard(self):
        """
        🛑 Friday Close Guard: Fecha TODAS as posições na Sexta-feira entre as 17:00 e 17:10 
        (Horário do Robô/BRT) para não passar o final de semana posicionado.
        """
        now = datetime.now() 
        
        if now.weekday() == 4: # Sexta-feira
            if now.hour == 17 and 0 <= now.minute < 10:
                if self.positions:
                    logger.warning("🛑 FRIDAY CLOSE GUARD: Fechando todas as posições antes do fechamento do mercado (Evitando Gaps de Fim de Semana)...")
                    
                    if not getattr(self, '_friday_close_notified_today', False):
                        send_telegram_message("🛑 *Friday Close Guard Ativado*\nFechando todas as posições abertas para não zerar a conta com gaps de segunda-feira.")
                        self._friday_close_notified_today = True
                        
                    self.close_all_positions()
            else:
                self._friday_close_notified_today = False
    
    def analyze_symbol(self, symbol: str, timeframe: int, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Analyze a symbol and generate trading signal using provided DataFrame"""
        try:
            # 0. Rollover Pause Check (Institutional)
            if self.is_in_rollover_pause():
                # logger.debug(f"Skipping {symbol}: Rollover Pause Active")
                return None
            
            if self.is_friday_block_active():
                # logger.debug(f"Skipping {symbol}: Friday Block Active")
                return None

            # 0. Cooldown / Sanity Check (FAIL FAST)
            if not trade_executor.can_trade(symbol, silent=True):
                # Log opcional limitado a 1x por minuto
                now = time.time()
                if now - self.last_cooldown_log.get(symbol, 0) > 60:
                    logger.debug(f"Skipping {symbol}: Active Cooldown/Account Issue")
                    self.last_cooldown_log[symbol] = now
                return None

            # 1. Circuit Breaker Check

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
        """Execute a trade based on signal using TradeExecutor"""
        try:
            with self.lock:
                # 1. Anti-Flood & Rate Limit (First line of defense)
                # Check if we already tried this symbol recently (handled by TradeExecutor but good to check here too)
                # Delegate entirely to TradeExecutor for consistency
                
                # Check if we already have a position for this symbol
                if signal.symbol in self.positions:
                    logger.debug(f"Posição já existe para {signal.symbol}. Ignorando duplicata.")
                    return False
                
                # Check max positions
                if len(self.positions) >= settings.MAX_POSITIONS:
                    logger.warning(f"🚫 Máximo de posições atingido ({len(self.positions)}/{settings.MAX_POSITIONS}). Ignorando sinal p/ {signal.symbol}")
                    return False
                
                # 2. Get Why Report for Logs (Institutional Requirement)
                try:
                    # Only fetch if we are actually going to trade
                    # This adds a small overhead but ensures logs are rich
                    _, _, why_log = self.strategy.get_why_report(signal.symbol)
                    if why_log:
                        logger.info(f"\n{why_log}")
                except: pass

                # 3. Delegate to TradeExecutor
                ticket = trade_executor.execute_order(signal)
                
                if not ticket:
                    return False
                
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
                    partial_taken=False,
                    ticket=ticket
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
                    'comment': signal.reason,
                    'ticket': ticket
                }
                save_trade(trade_data)
                
                return True
                
        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}")
            return False
    
    def update_positions(self):
        """Update open positions and synchronize with MT5 reality"""
        try:
            with self.lock:
                # 1. Sync with MT5 (Truth)
                # Obter todas as posições reais do MT5 para o nosso Magic Number
                mt5_positions = mt5_exec(mt5.positions_get, magic=settings.MAGIC_NUMBER)
                
                if mt5_positions is None:
                    # Se falhar o fetch (None), não limpamos a memória local para evitar pânico
                    # Mas se retornar [] (vazio), significa que nada está aberto.
                    logger.debug("Falha ao sincronizar posições com MT5 (None returned)")
                    return

                # Converter para dicionário por Ticket para busca rápida
                real_tickets = {p.ticket: p for p in mt5_positions}

                # a) Remover da memória o que não existe mais no MT5
                for symbol in list(self.positions.keys()):
                    pos = self.positions[symbol]
                    if pos.ticket not in real_tickets:
                        logger.info(f"🔄 Sincronismo: Removendo {symbol} da memória (fechado externamente ou no MT5)")
                        del self.positions[symbol]

                # b) Adicionar à memória o que está no MT5 mas o robô não sabia (ex: após restart)
                for p in mt5_positions:
                    if p.symbol not in self.positions:
                        logger.info(f"🔄 Sincronismo: Redescobrindo posição {p.symbol} (Ticket: {p.ticket})")
                        
                        # Tenta estimar SL dist ou usa default
                        sl_dist = abs(p.price_open - p.sl) if p.sl > 0 else 0
                        
                        self.positions[p.symbol] = Position(
                            symbol=p.symbol,
                            order_type="BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                            volume=p.volume,
                            entry_price=p.price_open,
                            current_price=p.price_current,
                            stop_loss=p.sl,
                            take_profit=p.tp,
                            profit=p.profit,
                            pips=0, # Calculado abaixo
                            open_time=datetime.fromtimestamp(p.time),
                            magic_number=p.magic,
                            initial_sl_dist=sl_dist,
                            partial_taken=False,
                            ticket=p.ticket
                        )

                # 2. Processar Lógica de Gerenciamento (Virtual SL/TP/Trailing)
                closed_positions = []
                
                # Get current DF for exit evaluation (efficiency: fetch only once per symbol)
                for symbol, position in self.positions.items():
                    # Get actual MT5 position data for this ticket
                    p_mt5 = real_tickets.get(position.ticket)
                    if not p_mt5:
                        continue 
                    
                    # Update position local state
                    current_price = p_mt5.price_current
                    position.current_price = current_price
                    position.profit = p_mt5.profit

                    # --- GESTÃO DINÂMICA (NOVO) ---
                    
                    # A. Evaluate Dynamic Exit (Exhaustion/Reversal)
                    # Fetch fresh data for evaluation
                    df_eval = self.cache.get_rates(symbol, 15, 50)
                    should_exit, exit_reason = self.strategy.evaluate_exit(position, df_eval)
                    
                    if should_exit:
                        logger.warning(f"🚀 [DYNAMIC EXIT] {exit_reason} em {symbol} (Ticket: {position.ticket})")
                        if trade_executor.close_position(position.ticket, symbol):
                            closed_positions.append(symbol)
                            self._update_stats_after_close(position)
                            send_telegram_message(f"🚀 *DYNAMIC EXIT EXECUTADO*\nAtivo: `{symbol}`\nMotivo: `{exit_reason}`\nLucro: `${position.profit:.2f}`")
                            continue

                    # B. Proactive Break-Even (BE) Management
                    # Determine Pips or R
                    point = self.symbol_manager.get_point(symbol)
                    if point > 0:
                        if position.order_type == "BUY":
                            profit_r = (current_price - position.entry_price) / position.initial_sl_dist if position.initial_sl_dist > 0 else 0
                        else:
                            profit_r = (position.entry_price - current_price) / position.initial_sl_dist if position.initial_sl_dist > 0 else 0
                    else:
                        profit_r = 0

                    if profit_r >= 1.5:
                        # Move SL to Entry + small buffer (Break-Even)
                        if position.order_type == "BUY" and position.stop_loss < position.entry_price:
                            position.stop_loss = position.entry_price
                            logger.info(f"🛡️ [BREAK-EVEN] SL movido para a entrada para {symbol} (1.5R alcançado)")
                        elif position.order_type == "SELL" and position.stop_loss > position.entry_price:
                            position.stop_loss = position.entry_price
                            logger.info(f"🛡️ [BREAK-EVEN] SL movido para a entrada para {symbol} (1.5R alcançado)")

                    # --- CHECK CLOSURE (VIRTUAL vs SERVER) ---

                    # Verifica se deve fechar (SL ou TP interno atingido)
                    sl_hit = (position.order_type == "BUY" and current_price <= position.stop_loss and position.stop_loss > 0) or \
                             (position.order_type == "SELL" and current_price >= position.stop_loss and position.stop_loss > 0)
                    
                    tp_hit = (position.order_type == "BUY" and current_price >= position.take_profit and position.take_profit > 0) or \
                             (position.order_type == "SELL" and current_price <= position.take_profit and position.take_profit > 0)

                    if sl_hit or tp_hit:
                        reason = "STOP LOSS VIRTUAL" if sl_hit else "TAKE PROFIT VIRTUAL"
                        logger.warning(f"🎯 [TRIGGER] {reason} atingido para {symbol} (Ticket: {position.ticket})")
                        
                        # EXECUÇÃO REAL DO FECHAMENTO
                        if trade_executor.close_position(position.ticket, symbol):
                            closed_positions.append(symbol)
                            # Update Daily Stats
                            if position.profit > 0:
                                self.strategy.daily_stats["wins"] += 1
                            else:
                                self.strategy.daily_stats["losses"] += 1
                                # Registra no Trade Executor para o cooldown pós-loss
                                trade_executor.register_loss(symbol)
                                
                            self.strategy.daily_stats["profit"] += position.profit
                            
                            # Notificação
                            send_telegram_message(f"✅ *{reason} EXECUTADO*\nAtivo: `{symbol}`\nLucro: `${position.profit:.2f}`")

                # Clear memory
                for symbol in closed_positions:
                    if symbol in self.positions:
                        del self.positions[symbol]
                        
        except Exception as e:
            logger.error(f"Erro ao atualizar/sincronizar posições: {e}", exc_info=True)

    def _update_stats_after_close(self, position: Position):
        """Helper para atualizar estatísticas após fechamento"""
        if position.profit > 0:
            self.strategy.daily_stats["wins"] += 1
        else:
            self.strategy.daily_stats["losses"] += 1
            trade_executor.register_loss(position.symbol)
        self.strategy.daily_stats["profit"] += position.profit

    def close_all_positions(self):
        """Close all open positions (Kill Switch)"""
        with self.lock:
            # Sync antes de fechar tudo para garantir que temos os tickets corretos
            mt5_positions = mt5_exec(mt5.positions_get, magic=settings.MAGIC_NUMBER)
            if mt5_positions:
                for p in mt5_positions:
                    logger.warning(f"Closing {p.symbol} (Ticket: {p.ticket}) due to Kill Switch")
                    trade_executor.close_position(p.ticket, p.symbol)
            
            # Limpa memória local
            self.positions.clear()
            logger.info("All positions closed via Kill Switch.")

    def load_open_trades(self):
        """Carrega trades abertas do banco de dados (Virtual SL/TP)"""
        try:
            open_trades = get_open_trades()
            if not open_trades:
                return

            logger.info(f"📂 Carregando {len(open_trades)} trades abertas do banco de dados...")
            
            with self.lock:
                for t in open_trades:
                    symbol = t['symbol']
                    ticket = t['ticket']
                    
                    # Criamos o objeto posição a partir do banco (preserva SL/TP virtual)
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        order_type=t['order_type'],
                        volume=t['volume'],
                        entry_price=t['entry_price'],
                        current_price=t['entry_price'], # Será atualizado pelo sync
                        stop_loss=t['stop_loss'] if t['stop_loss'] is not None else 0,
                        take_profit=t['take_profit'] if t['take_profit'] is not None else 0,
                        profit=0,
                        pips=0,
                        open_time=datetime.fromisoformat(t['entry_time']) if isinstance(t['entry_time'], str) else datetime.now(),
                        magic_number=t['magic_number'],
                        initial_sl_dist=abs(t['entry_price'] - t['stop_loss']) if t['stop_loss'] and t['stop_loss'] > 0 else 0,
                        partial_taken=False,
                        ticket=ticket
                    )
                    logger.debug(f"✅ Memória de SL/TP Virtual restaurada para {symbol} (Ticket: {ticket})")
        except Exception as e:
            logger.error(f"Erro ao carregar trades abertas: {e}")

    def display_summary_table(self):
        """Exibe uma tabela formatada com as posições abertas no console"""
        if not HAS_RICH or not self.console:
            # Fallback simples se Rich não estiver disponível
            if not self.positions:
                print("\n[INFO] Nenhuma posição aberta no momento.")
                return
            print("\n--- POSIÇÕES ABERTAS ---")
            for s, p in self.positions.items():
                print(f"{s}: {p.order_type} | Vol: {p.volume:.2f} | Entry: {p.entry_price:.5f} | Profit: ${p.profit:.2f}")
            return

        from rich.table import Table
        from rich.panel import Panel

        if not self.positions:
            self.console.print(Panel("[yellow]Nenhuma posição aberta no momento.[/]", title="XP3 STATUS", expand=False))
            return

        table = Table(title=f"XP3 PRO FOREX - POSIÇÕES ATIVAS ({datetime.now().strftime('%H:%M:%S')})", header_style="bold magenta")
        
        table.add_column("Símbolo", style="cyan", justify="left")
        table.add_column("Tipo", style="bold")
        table.add_column("Lote", justify="right")
        table.add_column("Entrada", justify="right")
        table.add_column("Atual", justify="right")
        table.add_column("SL (Virtual)", style="red", justify="right")
        table.add_column("TP (Virtual)", style="green", justify="right")
        table.add_column("P&L ($)", justify="right", style="bold")

        total_profit = 0
        for symbol, p in self.positions.items():
            type_style = "bold blue" if p.order_type == "BUY" else "bold red"
            profit_style = "green" if p.profit >= 0 else "red"
            
            table.add_row(
                symbol,
                f"[{type_style}]{p.order_type}[/]",
                f"{p.volume:.2f}",
                f"{p.entry_price:.5f}",
                f"{p.current_price:.5f}",
                f"{p.stop_loss:.5f}" if p.stop_loss > 0 else "-",
                f"{p.take_profit:.5f}" if p.take_profit > 0 else "-",
                f"[{profit_style}]${p.profit:.2f}[/]"
            )
            total_profit += p.profit

        # Footer com total
        table.add_section()
        total_style = "bold green" if total_profit >= 0 else "bold red"
        table.add_row("TOTAL", "", "", "", "", "", "", f"[{total_style}]${total_profit:.2f}[/]")

        self.console.print(table)
    
    def check_triple_swap_guard(self):
        """
        💸 Triple Swap Guard: Fecha posições na Quarta-feira às 17:45 (Horário do Robô/BRT)
        para evitar taxas triplas de rollover.
        """
        now = datetime.now() # Horário Local (BRT)
        
        # 2 = Wednesday em Python weekday()? No, 0=Mon, 1=Tue, 2=Wed
        if now.weekday() == 2: # Quarta-feira
            if now.hour == 17 and now.minute >= 45 and now.minute < 55:
                # Verifica se já fechamos hoje para não entrar em loop
                if self.positions:
                    logger.warning("💸 TRIPLE SWAP GUARD: Fechando todas as posições para evitar taxas de rollover...")
                    
                    # Notificar Telegram apenas uma vez por janela
                    if not getattr(self, '_triple_swap_notified_today', False):
                        send_telegram_message("💸 *Triple Swap Guard Ativado*\nFechando posições para evitar taxas triplas de rollover.")
                        self._triple_swap_notified_today = True
                        
                    self.close_all_positions()
            else:
                # Reset the flag outside the window
                self._triple_swap_notified_today = False
    
    def consumer_loop(self):
        """Consumes market data from queue and processes strategy"""
        logger.info("🔥 Consumer Loop iniciado")
        
        while not SHUTDOWN_EVENT.is_set():
            try:
                # Update positions
                self.update_positions()

                # --- Periodic Positions Report (5 min) ---
                if time.time() - self.last_summary_table_time > 300:
                    self.display_summary_table()
                    self.last_summary_table_time = time.time()
                
                # --- Auto Close Guard (MT5 Manual Activity) ---
                if settings.AUTO_CLOSE_ON_MT5 and ProcessWatcher.is_mt5_running():
                    logger.critical("🛑 MetaTrader 5 aberto manualmente detectado! Encerrando robô conforme configuração AUTO_CLOSE_ON_MT5.")
                    send_telegram_message("🛑 *Auto-shutdown Ativado*\nTerminal MT5 aberto manualmente detectado. Encerrando robô para segurança.")
                    SHUTDOWN_EVENT.set()
                    break

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
                                # FAIL FAST: Se o ativo estiver em cooldown, pula a geração do relatório pesado
                                if not trade_executor.can_trade(symbol, silent=True):
                                    continue

                                panel, conf, log_str = self.strategy.get_why_report(symbol)
                                
                                # Log visual (Console)
                                if panel and conf >= 80:
                                    self.console.print(panel)
                                    signals_found += 1
                                    
                                # Log textual (Arquivo)
                                # Apenas se for EXECUTE (conf >= 80) ou DEBUG_MODE
                                if log_str:
                                    if conf >= 80 or settings.DEBUG_MODE:
                                        logger.info(log_str)
                                    
                            except ValueError as e: # Handle unpack error if old version running
                                try:
                                    # Fallback for old signature
                                    panel, conf = self.strategy.get_why_report(symbol)
                                    if panel and conf > 60:
                                        self.console.print(panel)
                                        signals_found += 1
                                except: pass
                            except Exception as e:
                                # logger.error(f"Erro no Why Report para {symbol}: {e}")
                                pass
                        
                        if signals_found == 0:
                             # Exibir mensagem de progresso a cada X ciclos ou quando solicitado
                             self.console.print(f"[dim]Scan concluído: Nenhum sinal >60% encontrado em {len(symbols_to_scan)} símbolos monitorados. Aguardando...[/]")

                # --- Periodic Symbol Refresh (Every 30m) ---
                # Importante para redescobrir novos ativos que entraram no Market Watch se SYMBOLS=ALL
                if time.time() - getattr(self, '_last_symbol_refresh', 0) > 1800:
                    self._last_symbol_refresh = time.time()
                    logger.info("🔄 Atualizando lista de símbolos candidatos...")
                    self.initialize_market_data()
                
                # Process Data Queue
                try:
                    # Non-blocking get or short timeout
                    symbol, timeframe, df = self.data_queue.get(timeout=1)
                    
                    if symbol == "CYCLE_COMPLETE":
                        self.process_batch_signals()
                    else:
                        # Analyze
                        signal = self.analyze_symbol(symbol, timeframe, df)
                        if signal:
                            with self.lock:
                                # Avoid duplicate symbols in the same batch
                                self.pending_signals = [s for s in self.pending_signals if s.symbol != signal.symbol]
                                self.pending_signals.append(signal)
                                logger.debug(f"📝 Sinal acumulado para batch: {signal.symbol} (Conf: {signal.confidence:.2f})")
                        
                    self.data_queue.task_done()
                    
                except queue.Empty:
                    pass
                    
                # --- Daily Maintenance (23:55 UTC) ---
                now_utc = datetime.utcnow()
                if now_utc.hour == 23 and now_utc.minute >= 55:
                    if self._last_maintenance_date != now_utc.date():
                        self.run_daily_maintenance()
                        self._last_maintenance_date = now_utc.date()
                
                # --- Triple Swap Guard (Wed 17:45 BRT) ---
                self.check_triple_swap_guard()
                
                # --- Friday Close Guard (Fri 17:00 BRT) ---
                self.check_friday_close_guard()

            except Exception as e:
                logger.error(f"Erro no consumer_loop: {e}")
                time.sleep(1)
            
            # CPU Respiro (Obrigatório)
            time.sleep(1.0)
                
        logger.info("🛑 Consumer Loop finalizado")

    def process_batch_signals(self):
        """
        Processes accumulated signals: sorts by confidence and executes the best ones
        up to the maximum positions limit.
        """
        if not self.pending_signals:
            return

        with self.lock:
            # 1. Sort by confidence (descending)
            self.pending_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"⚖️ Processando BATCH de {len(self.pending_signals)} sinais encontrados no ciclo.")
            
            # 2. Iterate and execute
            executed_count = 0
            for signal in self.pending_signals:
                # Execution method will check MAX_POSITIONS internally, 
                # but we can do a pre-check to log better.
                if len(self.positions) >= settings.MAX_POSITIONS:
                    logger.warning(f"🚫 Limite de posições ({settings.MAX_POSITIONS}) atingido no meio do batch. Descartando sinais restantes.")
                    break
                
                logger.info(f"🎯 Selecionado: {signal.symbol} com confiança {signal.confidence:.2f}")
                if self.execute_trade(signal):
                    executed_count += 1
            
            # 3. Clear batch
            self.pending_signals.clear()
            
            if executed_count > 0:
                logger.info(f"✅ Batch finalizado: {executed_count} ordens enviadas.")

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
        
        # Use SymbolManager's smart filter - INICIALIZA COM TODOS (Spread check agora é no Feeder)
        valid_symbols = self.symbol_manager.get_tradable_symbols(ignore_spread=True)
        
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

    def run_daily_maintenance(self):
        """Executa ciclo de aprendizado e relatório diário"""
        logger.info("🛠️ Iniciando MANUTENÇÃO DIÁRIA (Aprendizado + Relatórios)...")
        try:
            # 1. Rodar Learner
            learnings = self.learner.run_full_learning()
            
            # 2. Gerar Relatório
            if learnings:
                report_path = self.report_gen.generate_learning_report(learnings)
                logger.info(f"📚 Lições do dia arquivadas em: {report_path}")
                
                # 3. Enviar para Telegram
                send_telegram_message("📊 *Relatório Diário de Aprendizado XP3 Disponível*")
                send_telegram_document(report_path, caption=f"Learnings {datetime.now().strftime('%Y-%m-%d')}")
            else:
                logger.warning("⚠️ O Ciclo de Aprendizado não gerou novos parâmetros (dados insuficientes?)")
                
            logger.info("✅ Manutenção Diária concluída com sucesso.")
        except Exception as e:
            logger.error(f"Erro durante a manutenção diária: {e}")

    def start(self):
        """Start the bot"""
        try:
            logger.info("🚀 Iniciando XP3 PRO FOREX BOT")
            
            # Initialize MT5
            if not self.initialize_mt5():
                logger.error("Falha ao conectar ao MT5")
                return False
            
            # Initialize Database and persistence
            init_database()
            self.load_open_trades()
            
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
