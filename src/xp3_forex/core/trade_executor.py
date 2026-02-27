"""
✅ TRADING EXECUTION CORE (XP3 PRO FOREX)
Módulo responsável pela execução segura e auditável de ordens no MetaTrader 5.
"""

import MetaTrader5 as mt5
import logging
import time
from typing import Optional, Dict, Any, Union, List
from datetime import datetime

from xp3_forex.core.settings import settings
from xp3_forex.mt5.symbol_manager import symbol_manager
from xp3_forex.core.models import TradeSignal, Position
from xp3_forex.utils.mt5_utils import mt5_exec, get_symbol_info

try:
    from rich.panel import Panel
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger("XP3.TradeExecutor")

class TradeExecutor:
    """
    Executor de ordens com validações institucionais.
    Garante que:
    1. O símbolo é negociável (filtro de spread).
    2. O modo (Live/Paper) é respeitado.
    3. Retorno detalhado da operação.
    4. Rate Limiting e Position Sizing robustos.
    """
    
    def __init__(self):
        # Determina modo de operação
        self.mode = "paper"
        if hasattr(settings, "XP3_ENV"):
             if settings.XP3_ENV.lower() == "production":
                 self.mode = "live"
             elif settings.XP3_ENV.lower() == "development":
                 self.mode = "paper"
        
        # Override se existir flag TRADING_MODE
        if hasattr(settings, "TRADING_MODE"):
             self.mode = settings.TRADING_MODE.lower()
             
        # Rate Limiting State
        self.last_trade_attempt: Dict[str, float] = {}
        self.COOLDOWN_SECONDS = 15.0
        self.MAX_CONCURRENT_TRADES = 6
        
        # Circuit Breaker State (Error Tracking)
        self.consecutive_failures = 0
        self.CIRCUIT_BREAKER_LIMIT = 5
        self.circuit_breaker_until = 0.0
        
        logger.info(f"TradeExecutor inicializado em modo: {self.mode.upper()}")

    def can_trade(self, symbol: str) -> bool:
        """Verifica se pode operar (Rate Limit, Circuit Breaker, Max Trades)"""
        now = time.time()
        
        # 1. Global Circuit Breaker
        if now < self.circuit_breaker_until:
            logger.warning(f"🚫 Circuit Breaker ATIVO. Trading pausado por mais {self.circuit_breaker_until - now:.1f}s")
            return False
            
        # 2. Symbol Cooldown
        if now - self.last_trade_attempt.get(symbol, 0) < self.COOLDOWN_SECONDS:
            logger.warning(f"⏳ Cooldown ativo para {symbol}. Aguarde...")
            return False
            
        # 3. Max Trades (Check MT5 positions)
        if self.mode in ["live", "demo"]:
            positions = mt5_exec(mt5.positions_total)
            if positions is not None and positions >= self.MAX_CONCURRENT_TRADES:
                logger.warning(f"🚫 Limite de posições atingido ({positions}/{self.MAX_CONCURRENT_TRADES})")
                return False
                
        return True

    def calculate_lot_size(self, symbol: str, sl_price: float, entry_price: float, risk_percent: float = 0.3) -> float:
        """
        Calcula o tamanho do lote baseado no risco financeiro.
        Risco padrão: 0.3% do saldo.
        """
        try:
            account_info = mt5_exec(mt5.account_info)
            if not account_info:
                return 0.01
                
            balance = account_info.balance
            risk_amount = balance * (risk_percent / 100.0)
            
            symbol_info = get_symbol_info(symbol)
            if not symbol_info:
                return 0.01
                
            tick_value = symbol_info.get('trade_tick_value', 0)
            tick_size = symbol_info.get('trade_tick_size', 0)
            contract_size = symbol_info.get('trade_contract_size', 100000)
            
            # Cálculo de pips/points
            # Distância em preço
            dist_price = abs(entry_price - sl_price)
            
            if dist_price == 0:
                return 0.01
                
            # Fórmula padrão Forex: Lot = Risk / (SL_pips * Pip_Value_per_Lot)
            # Mas tick_value no MT5 é por tick/point para 1 lote padrão?
            # Geralmente: Profit = (Close - Open) * Contract_Size * Volume (para Forex direto)
            # Ou Profit = (Close - Open) / Tick_Size * Tick_Value * Volume
            
            # Vamos usar a fórmula baseada em tick_value (mais universal para indices/commodities)
            # Loss = (Dist / Tick_Size) * Tick_Value * Volume
            # Volume = Risk / ((Dist / Tick_Size) * Tick_Value)
            
            if tick_size == 0 or tick_value == 0:
                # Fallback para Forex Standard
                # Assume contract size 100k
                loss_per_lot = dist_price * contract_size
            else:
                loss_per_lot = (dist_price / tick_size) * tick_value
                
            if loss_per_lot == 0:
                return 0.01
                
            volume = risk_amount / loss_per_lot
            
            # Normalizar volume (step, min, max)
            step = symbol_info.get('volume_step', 0.01)
            min_vol = symbol_info.get('volume_min', 0.01)
            max_vol = symbol_info.get('volume_max', 100.0) # Cap at 100 or symbol max
            
            # Hard cap institutional
            max_vol = min(max_vol, 5.0) 
            
            # Round to step
            if step > 0:
                volume = round(volume / step) * step
                
            volume = max(min_vol, min(volume, max_vol))
            
            return round(volume, 2)
            
        except Exception as e:
            logger.error(f"Erro ao calcular lote para {symbol}: {e}")
            return 0.01

    def execute_order(self, signal: TradeSignal) -> Optional[int]:
        """
        Executa uma ordem de trading baseada no sinal.
        Retorna Ticket (int) se sucesso, ou None se falha.
        """
        symbol = signal.symbol
        
        # 0. Rate Limiting & Checks
        if not self.can_trade(symbol):
            return None
            
        self.last_trade_attempt[symbol] = time.time()

        # 1. Validação de Filtro de Spread (CRÍTICO)
        resolved_symbol = symbol_manager.resolve_name(symbol)
        if not resolved_symbol:
            logger.error(f"❌ TRADE BLOQUEADO: Símbolo {symbol} não resolvido no MT5.")
            return None
            
        tradable_symbols = symbol_manager.get_tradable_symbols()
        if resolved_symbol not in tradable_symbols:
            # Double check spread just in case list is stale
            if not symbol_manager.check_spread(resolved_symbol):
                logger.error(f"❌ TRADE BLOQUEADO: {resolved_symbol} rejeitado pelo filtro de spread.")
                return None
        
        # Check if position exists
        if mt5_exec(mt5.positions_get, symbol=resolved_symbol):
            logger.warning(f"⚠️ Posição já existe para {resolved_symbol}. Ignorando sinal.")
            return None

        # 2. Recalcular Lote (Position Sizing)
        # Ignora volume do sinal original se for absurdo
        calculated_volume = self.calculate_lot_size(resolved_symbol, signal.stop_loss, signal.entry_price, risk_percent=0.3)
        signal.volume = calculated_volume # Atualiza sinal
        
        order_type = signal.order_type
        price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        comment = f"XP3v5:{signal.reason[:20]}" 

        # 3. Log de Intenção
        logger.info(f"🎯 INTENÇÃO: {resolved_symbol} {order_type} @ {price:.5f} (Vol: {calculated_volume:.2f} | Risco: 0.3%)")

        # 4. Preparação da Ordem
        action = mt5.TRADE_ACTION_DEAL
        type_op = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": action,
            "symbol": resolved_symbol,
            "volume": float(calculated_volume),
            "type": type_op,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": settings.MAX_SLIPPAGE,
            "magic": settings.MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 5. Execução (Live ou Paper)
        if self.mode in ["live", "demo"]:
            logger.info(f"📤 Enviando ordem para MT5 | {resolved_symbol} | Vol: {calculated_volume}...")
            
            # Check connection
            if not mt5.terminal_info().connected:
                 logger.error("❌ Falha crítica: MT5 desconectado.")
                 return None

            # Retry Logic
            for attempt in range(2):
                result = mt5_exec(mt5.order_send, request)
                
                if result is None:
                    logger.error("❌ Falha crítica: mt5.order_send timeout/erro.")
                    break
                    
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ ORDEM EXECUTADA com sucesso | Ticket: {result.order} | Preço: {result.price}")
                    signal.entry_price = result.price # Atualiza com preço real
                    self.consecutive_failures = 0 # Reset failures
                    return result.order
                    
                # Erros recuperáveis (Timeout, Requote, Invalid Price)
                # Adicionado 10015 (Invalid Price) que acontece em fast moves
                # Adicionado 10027 (AutoTrading Disabled by Client) para logar melhor
                # Adicionado 10009 (Done) para garantir
                elif result.retcode in [mt5.TRADE_RETCODE_TIMEOUT, mt5.TRADE_RETCODE_REQUOTE, 10004, 10015, 10021, 10016]:
                    logger.warning(f"⚠️ Erro temporário ({result.retcode} - {result.comment}). Retentando {attempt+1}/2...")
                    time.sleep(1.0 + (attempt * 0.5))
                    
                    # Refresh prices before retry
                    tick = symbol_manager.get_tick(resolved_symbol)
                    if tick:
                        request['price'] = tick.ask if order_type == "BUY" else tick.bid
                        request['sl'] = float(signal.stop_loss) # Mantém SL original
                        request['tp'] = float(signal.take_profit) # Mantém TP original
                    continue
                    
                else:
                    logger.error(f"❌ order_send FALHOU | Retcode: {result.retcode} ({result.comment}) | Symbol: {resolved_symbol}")
                    
                    # Se for erro de volume (Invalid Volume), logar detalhes
                    if result.retcode == 10014:
                         logger.error(f"Volume inválido: {request['volume']}")
                         
                    self.handle_failure()
                    return None
            
            return None
                
        else:
            # Paper Trading
            logger.info("🧪 PAPER TRADE SIMULADO (não enviado ao MT5)")
            logger.info(f"✅ [SIMULAÇÃO] ORDEM EXECUTADA | Ticket: PAPER-{int(time.time())}")
            return int(time.time())

    def handle_failure(self):
        """Gerencia falhas consecutivas para ativar Circuit Breaker"""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.CIRCUIT_BREAKER_LIMIT:
            self.circuit_breaker_until = time.time() + 600 # 10 minutos
            logger.critical(f"🚨 CIRCUIT BREAKER ATIVADO: 5 falhas consecutivas. Trading pausado por 10min.")

# Instância Global
trade_executor = TradeExecutor()
