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

    def check_account_health(self) -> Dict[str, Any]:
        """
        Verifica saúde da conta (margin, equity, balance).
        Retorna dict com status detalhado.
        """
        account = mt5_exec(mt5.account_info)
        if not account:
            return {"healthy": False, "reason": "Falha ao obter informações da conta"}

        # Calculate margin_level correctly
        # When margin == 0 (no positions), margin_level is undefined (set to inf)
        margin_level = account.margin_level if account.margin > 0 else float("inf")

        info = {
            "healthy": True,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "margin_free": account.margin_free,
            "margin_level": margin_level,
            "leverage": account.leverage,
        }

        # Check critical conditions
        if account.margin_free <= 0:
            info["healthy"] = False
            info["reason"] = (
                f"❌ SEM MARGIN DISPONÍVEL (margin_free={account.margin_free:.2f})"
            )
        # Only check margin_level if there are positions (margin > 0)
        elif account.margin > 0 and margin_level < 100:
            info["healthy"] = False
            info["reason"] = f"❌ MARGIN CALL RISCO (level={margin_level:.1f}% < 100%)"
        elif account.equity <= 0:
            info["healthy"] = False
            info["reason"] = f"❌ CONTA NEGATIVA (equity={account.equity:.2f})"
        elif account.margin_free < account.balance * 0.05:  # Less than 5% free margin
            info["healthy"] = False
            info["reason"] = (
                f"⚠️ MARGIN CRÍTICA (free {account.margin_free:.2f} < 5% do balance)"
            )

        return info

    def can_trade(self, symbol: str) -> bool:
        """Verifica se pode operar (Rate Limit, Circuit Breaker, Max Trades, Margin)"""
        now = time.time()

        # 1. Global Circuit Breaker
        if now < self.circuit_breaker_until:
            logger.warning(
                f"🚫 Circuit Breaker ATIVO. Trading pausado por mais {self.circuit_breaker_until - now:.1f}s"
            )
            return False

        # 2. Account Health (NEW - CRITICAL)
        if self.mode in ["live", "demo"]:
            health = self.check_account_health()
            if not health["healthy"]:
                logger.error(health["reason"])
                return False

        # 3. Symbol Cooldown
        if now - self.last_trade_attempt.get(symbol, 0) < self.COOLDOWN_SECONDS:
            logger.warning(f"⏳ Cooldown ativo para {symbol}. Aguarde...")
            return False

        # 4. Max Trades (Check MT5 positions)
        if self.mode in ["live", "demo"]:
            positions = mt5_exec(mt5.positions_total)
            if positions is not None and positions >= self.MAX_CONCURRENT_TRADES:
                logger.warning(
                    f"🚫 Limite de posições atingido ({positions}/{self.MAX_CONCURRENT_TRADES})"
                )
                return False

        return True

    def calculate_lot_size(
        self,
        symbol: str,
        sl_price: float,
        entry_price: float,
        risk_percent: float = 0.3,
    ) -> float:
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

            tick_value = symbol_info.get("trade_tick_value", 0)
            tick_size = symbol_info.get("trade_tick_size", 0)
            contract_size = symbol_info.get("trade_contract_size", 100000)

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
            step = symbol_info.get("volume_step", 0.01)
            min_vol = symbol_info.get("volume_min", 0.01)
            max_vol = symbol_info.get("volume_max", 100.0)  # Cap at 100 or symbol max

            # Hard cap institutional
            max_vol = min(max_vol, settings.MAX_LOTS_PER_TRADE)

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
                logger.error(
                    f"❌ TRADE BLOQUEADO: {resolved_symbol} rejeitado pelo filtro de spread."
                )
                return None

        # Check if position exists
        if mt5_exec(mt5.positions_get, symbol=resolved_symbol):
            logger.warning(
                f"⚠️ Posição já existe para {resolved_symbol}. Ignorando sinal."
            )
            return None

        # 2. Recalcular Lote (Position Sizing)
        # Ignora volume do sinal original se for absurdo
        calculated_volume = self.calculate_lot_size(
            resolved_symbol, signal.stop_loss, signal.entry_price, risk_percent=0.3
        )
        signal.volume = calculated_volume  # Atualiza sinal

        order_type = signal.order_type
        price = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        comment = f"XP3v5:{signal.reason[:20]}"

        # 3. Log de Intenção
        logger.info(
            f"🎯 INTENÇÃO: {resolved_symbol} {order_type} @ {price:.5f} (Vol: {calculated_volume:.2f} | Risco: 0.3%)"
        )

        # 4. Preparação da Ordem
        action = mt5.TRADE_ACTION_DEAL
        type_op = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL

        # Auto-detect filling mode (corretoras diferem: IOC, RETURN, FOK)
        filling_mode = mt5.ORDER_FILLING_IOC  # default
        sym_info = mt5_exec(mt5.symbol_info, resolved_symbol)
        if sym_info is not None:
            fm = sym_info.filling_mode
            # filling_mode é bitmask: 1=FOK, 2=IOC, 4=RETURN
            # Prioridade: IOC (mais universal) > FOK > RETURN
            if fm & 2:  # IOC disponível (mais universal)
                filling_mode = mt5.ORDER_FILLING_IOC
            elif fm & 4:  # RETURN disponível
                filling_mode = mt5.ORDER_FILLING_RETURN
            elif fm & 1:  # FOK disponível
                filling_mode = mt5.ORDER_FILLING_FOK
            else:
                logger.warning(
                    f"⚠️ Nenhum filling mode padrão disponível para {resolved_symbol}"
                )
            logger.info(
                f"Filling mode para {resolved_symbol}: {filling_mode} (bitmask={fm})"
            )

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
            "type_filling": filling_mode,
        }

        # Validate symbol is enabled for trading
        if sym_info is None:
            logger.error(
                f"❌ Symbol {resolved_symbol} não possui informações de trading"
            )
            return None

        # Check if symbol is in close-only mode or disabled
        # trade_mode: 0=disabled, 1=buy-only, 2=sell-only, 3=buy+sell, 4=close-only
        if sym_info.trade_mode == 0:
            logger.error(f"❌ Symbol {resolved_symbol} está DESABILITADO para trading")
            return None

        if sym_info.trade_mode == 4:
            logger.error(
                f"❌ Symbol {resolved_symbol} está em modo CLOSE-ONLY (possível alta volatilidade/notícias)"
            )
            return None

        # Validate trade direction is supported
        if order_type == "BUY" and sym_info.trade_mode == 2:
            logger.error(
                f"❌ Symbol {resolved_symbol} permite apenas SELL (trade_mode=2)"
            )
            return None

        if order_type == "SELL" and sym_info.trade_mode == 1:
            logger.error(
                f"❌ Symbol {resolved_symbol} permite apenas BUY (trade_mode=1)"
            )
            return None

        # 5. Execução (Live ou Paper)
        if self.mode in ["live", "demo"]:
            logger.info(
                f"📤 Enviando ordem para MT5 | {resolved_symbol} | Vol: {calculated_volume}..."
            )

            # Check connection
            if not mt5.terminal_info().connected:
                logger.error("❌ Falha crítica: MT5 desconectado.")
                return None

            # CRITICAL: Refresh prices before order_check to avoid stale price rejections
            tick = symbol_manager.get_tick(resolved_symbol)
            if tick:
                old_price = request["price"]
                request["price"] = tick.ask if order_type == "BUY" else tick.bid
                price_diff = abs(request["price"] - old_price)
                if price_diff > sym_info.point * 5:  # More than 5 points difference
                    logger.warning(
                        f"⚠️ STALE PRICES DETECTED | Old: {old_price:.5f} → New: {request['price']:.5f} "
                        f"(Diff: {price_diff:.5f})"
                    )
            else:
                logger.warning(
                    f"⚠️ Não foi possível obter preço atual para {resolved_symbol}"
                )
                return None

            # Retry Logic
            for attempt in range(3):
                # Diagnóstico antes do envio (order_check revela o erro exato)
                if attempt == 0:
                    check = mt5_exec(mt5.order_check, request=request, timeout=10)

                    # Always log the request details
                    logger.info(
                        f"📋 Order Request: {request['symbol']} {'BUY' if request['type'] == mt5.ORDER_TYPE_BUY else 'SELL'} "
                        f"Vol={request['volume']:.2f} Price={request['price']:.5f} "
                        f"SL={request['sl']:.5f} TP={request['tp']:.5f}"
                    )

                    if check is not None:
                        account_info = self.check_account_health()
                        margin_level_str = (
                            f"{account_info['margin_level']:.1f}%"
                            if account_info["margin_level"] != float("inf")
                            else "∞ (no positions)"
                        )

                        if check.retcode != 0:
                            # Log account info junto com erro (importante para diagnosticar margin issues)
                            logger.error(
                                f"❌ order_check FALHOU | Retcode: {check.retcode} ({check.comment}) "
                                f"| margin={check.margin:.2f} | margin_free={check.margin_free:.2f} "
                                f"| ACCOUNT: Balance={account_info['balance']:.2f} Equity={account_info['equity']:.2f} "
                                f"FL={margin_level_str} | Symbol trade_mode={sym_info.trade_mode if sym_info else 'N/A'}"
                            )
                            # CRITICAL: Do not attempt order_send if order_check failed!
                            self.handle_failure()
                            return None
                        else:
                            logger.info(
                                f"✅ order_check OK | Required Margin: ${check.margin:.2f} | "
                                f"Available: ${account_info['margin_free']:.2f} | "
                                f"Fee: ${check.commission:.2f} | Profit: ${check.profit:.2f}"
                            )
                    else:
                        logger.error(
                            f"❌ order_check retornou None para {resolved_symbol}"
                        )
                        self.handle_failure()
                        return None

                # Timeout maior para order_send (operação crítica)
                result = mt5_exec(mt5.order_send, request=request, timeout=60)

                if result is None:
                    last_err = mt5.last_error()
                    logger.error(
                        f"❌ mt5.order_send retornou None | Tentativa {attempt + 1}/3 | last_error={last_err}"
                    )
                    # Não desiste imediatamente: aguarda e tenta de novo
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                        # Atualizar preço antes de retentar
                        tick = symbol_manager.get_tick(resolved_symbol)
                        if tick:
                            request["price"] = (
                                tick.ask if order_type == "BUY" else tick.bid
                            )
                        continue
                    else:
                        self.handle_failure()
                        return None

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"✅ ORDEM EXECUTADA com sucesso | Ticket: {result.order} | Preço: {result.price}"
                    )
                    signal.entry_price = result.price  # Atualiza com preço real
                    self.consecutive_failures = 0  # Reset failures
                    return result.order

                # Erros recuperáveis (Timeout, Requote, Invalid Price)
                elif result.retcode in [
                    mt5.TRADE_RETCODE_TIMEOUT,
                    mt5.TRADE_RETCODE_REQUOTE,
                    10004,
                    10015,
                    10021,
                    10016,
                ]:
                    logger.warning(
                        f"⚠️ Erro temporário ({result.retcode} - {result.comment}). Retentando {attempt + 1}/3..."
                    )
                    time.sleep(1.0 + (attempt * 0.5))

                    # Refresh prices before retry
                    tick = symbol_manager.get_tick(resolved_symbol)
                    if tick:
                        request["price"] = tick.ask if order_type == "BUY" else tick.bid
                        request["sl"] = float(signal.stop_loss)  # Mantém SL original
                        request["tp"] = float(signal.take_profit)  # Mantém TP original
                    continue

                else:
                    # Log detalhado para diagnóstico
                    error_msg = (
                        f"❌ order_send FALHOU | Retcode: {result.retcode} ({result.comment}) | Symbol: {resolved_symbol} "
                        f"| Vol={request['volume']} Price={request['price']} SL={request['sl']} TP={request['tp']} Filling={request['type_filling']}"
                    )

                    # Se for erro 10013 (Invalid Request), pode ser margin
                    if result.retcode == 10013:
                        account_info = self.check_account_health()
                        margin_level_str = (
                            f"{account_info['margin_level']:.1f}%"
                            if account_info["margin_level"] != float("inf")
                            else "∞ (no positions)"
                        )
                        error_msg += (
                            f"\n  💰 Account Info: Balance={account_info['balance']:.2f} | "
                            f"Equity={account_info['equity']:.2f} | Free Margin={account_info['margin_free']:.2f} | "
                            f"Margin Level={margin_level_str}"
                        )
                        if not account_info["healthy"]:
                            error_msg += f"\n  ⚠️ {account_info['reason']}"

                    # Se for erro de volume (Invalid Volume)
                    elif result.retcode == 10014:
                        error_msg += f"\n  📊 Volume inválido: {request['volume']}"

                    logger.error(error_msg)
                    self.handle_failure()
                    return None

            return None

        else:
            # Paper Trading
            logger.info("🧪 PAPER TRADE SIMULADO (não enviado ao MT5)")
            logger.info(
                f"✅ [SIMULAÇÃO] ORDEM EXECUTADA | Ticket: PAPER-{int(time.time())}"
            )
            return int(time.time())

    def handle_failure(self):
        """Gerencia falhas consecutivas para ativar Circuit Breaker"""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.CIRCUIT_BREAKER_LIMIT:
            self.circuit_breaker_until = time.time() + 600  # 10 minutos
            logger.critical(
                f"🚨 CIRCUIT BREAKER ATIVADO: 5 falhas consecutivas. Trading pausado por 10min."
            )


# Instância Global
trade_executor = TradeExecutor()
