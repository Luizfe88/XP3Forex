# validation_forex.py - XP3 PRO FOREX VALIDATION v4.2 INSTITUTIONAL
"""
🚀 XP3 PRO FOREX VALIDATION - VERSÃO INSTITUCIONAL v4.2
✅ Validação de ordens antes da execução
✅ Gerenciamento de risco e volume
✅ Conexão MT5
✅ Melhorias de logging e arredondamento
✅ CORREÇÃO: Tratamento de AttributeError para mt5.symbol_info_tick
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import MetaTrader5 as mt5
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from xp3_forex.core import config as config
from xp3_forex.utils import mt5_utils, indicators, calculations, data_utils as utils
import os
import time
from daily_analysis_logger import daily_logger

logger = logging.getLogger("XP3_VALIDATION")

# ===========================
# ENUMS
# ===========================
class OrderSide(Enum):
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL

# ===========================
# DATACLASSES
# ===========================
@dataclass
class OrderParams:
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    sl: float
    tp: float
    comment: str
    magic: int

# ===========================
# ORDER VALIDATION & EXECUTION
# ===========================
def validate_and_create_order_forex(params: OrderParams) -> Tuple[bool, str, int]:
    """
    Valida e cria uma ordem no MT5.
    Retorna (sucesso, mensagem, ticket_da_ordem)
    """
    # 1. Verifica conexão MT5
    if not utils.check_mt5_connection():
        logger.error("❌ MT5 não conectado. Falha na validação da ordem.")
        return False, "MT5 não conectado", 0

    # 2. Verifica se o símbolo está disponível
    symbol_info = utils.get_symbol_info(params.symbol)
    if not symbol_info:
        logger.error(f"❌ Símbolo {params.symbol} indisponível. Falha na validação da ordem.")
        return False, f"Símbolo {params.symbol} indisponível", 0

    # 3. Valida volume
    if params.volume < symbol_info.volume_min:
        logger.error(f"❌ Volume {params.volume} menor que o mínimo {symbol_info.volume_min} para {params.symbol}.")
        return False, f"Volume {params.volume} menor que o mínimo {symbol_info.volume_min}", 0
    if params.volume > symbol_info.volume_max:
        logger.error(f"❌ Volume {params.volume} maior que o máximo {symbol_info.volume_max} para {params.symbol}.")
        return False, f"Volume {params.volume} maior que o máximo {symbol_info.volume_max}", 0
    step = float(symbol_info.volume_step)
    ratio = params.volume / step
    nearest = round(ratio)
    if abs(ratio - nearest) <= 1e-8:
        normalized = nearest * step
        params.volume = float(round(normalized, 8))
    else:
        logger.error(f"❌ Volume {params.volume} não é múltiplo do step {symbol_info.volume_step} para {params.symbol}.")
        return False, f"Volume {params.volume} não é múltiplo do step {symbol_info.volume_step}", 0

    # 4. Valida SL/TP (preço deve ser diferente de 0 e válido)
    if params.sl <= 0 or params.tp <= 0:
        logger.error(f"❌ SL ({params.sl}) ou TP ({params.tp}) inválidos (<= 0) para {params.symbol}.")
        return False, "SL ou TP inválidos (<= 0)", 0

    # Garante que SL/TP estejam no lado correto
    if params.side == OrderSide.BUY:
        if params.sl >= params.entry_price:
            logger.error(f"❌ SL de compra ({params.sl}) deve ser menor que o preço de entrada ({params.entry_price}) para {params.symbol}.")
            return False, "SL de compra deve ser menor que o preço de entrada", 0
        if params.tp <= params.entry_price:
            logger.error(f"❌ TP de compra ({params.tp}) deve ser maior que o preço de entrada ({params.entry_price}) para {params.symbol}.")
            return False, "TP de compra deve ser maior que o preço de entrada", 0
    else: # SELL
        if params.sl <= params.entry_price:
            logger.error(f"❌ SL de venda ({params.sl}) deve ser maior que o preço de entrada ({params.entry_price}) para {params.symbol}.")
            return False, "SL de venda deve ser maior que o preço de entrada", 0
        if params.tp >= params.entry_price:
            logger.error(f"❌ TP de venda ({params.tp}) deve ser menor que o preço de entrada ({params.entry_price}) para {params.symbol}.")
            return False, "TP de venda deve ser menor que o preço de entrada", 0

    # 5. Verifica margem disponível (melhoria de logging)
    account_info = mt5.account_info()
    if not account_info:
        logger.error("❌ Não foi possível obter informações da conta para validação de margem.")
        return False, "Não foi possível obter informações da conta", 0

    # Estimativa de margem necessária (simplificada)
    # Para uma estimativa mais precisa, precisaríamos do SymbolInfo.margin_initial
    # e SymbolInfo.margin_maintenance, mas isso é complexo e o MT5 faz o cálculo exato.
    # Por enquanto, apenas logamos a margem livre.
    # if account_info.margin_free < (params.volume * params.entry_price * symbol_info.margin_initial / account_info.leverage): # Estimativa muito grosseira
    #     logger.warning(f"⚠️ Margem livre ({account_info.margin_free:.2f}) pode ser insuficiente para a ordem de {params.symbol} (volume: {params.volume}).")
    #     # return False, "Margem insuficiente", 0 # Descomente para forçar a validação de margem

    # 6. Prepara a requisição da ordem
    current_tick = mt5.symbol_info_tick(params.symbol)
    if current_tick is None: # ✅ CORREÇÃO: Verifica se current_tick é None
        logger.error(f"❌ Não foi possível obter tick atual para {params.symbol}. Falha na validação da ordem.")
        return False, f"Não foi possível obter tick atual para {params.symbol}", 0

    # ✅ REQUISITO: Detecção Automática de Filling Mode (Compatibilidade com Corretoras)
    # FOK -> IOC -> RETURN (Utilizando Bitwise para evitar AttributeError)
    filling_mode = mt5.ORDER_FILLING_FOK # Default
    fill_mode_broker = symbol_info.filling_mode
    
    if fill_mode_broker & 1: # SYMBOL_FILLING_FOK
        filling_mode = mt5.ORDER_FILLING_FOK
    elif fill_mode_broker & 2: # SYMBOL_FILLING_IOC
        filling_mode = mt5.ORDER_FILLING_IOC
    else:
        filling_mode = mt5.ORDER_FILLING_RETURN

    logger.debug(f"🔍 Símbolo {params.symbol} suporta modo de preenchimento: {filling_mode} (Flags: {fill_mode_broker})")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": params.symbol,
        "volume": params.volume,
        "type": params.side.value,
        "price": current_tick.ask if params.side == OrderSide.BUY else current_tick.bid, # ✅ CORREÇÃO: Usa current_tick.ask/bid
        "sl": params.sl,
        "tp": params.tp,
        "deviation": config.DEVIATION,
        "magic": params.magic,
        "comment": params.comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    max_retries = int(getattr(config, "REQUOTE_RETRY_MAX", 2))
    attempt = 0
    last_result = None
    start_ts = time.time()
    while attempt <= max_retries:
        with utils.mt5_lock:
            last_result = mt5.order_send(request)
        if not last_result:
            logger.error(f"❌ Falha crítica: Sem resposta do MT5 para {params.symbol}")
            return False, "Sem resposta do MT5", 0
        if last_result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            break
        err_code = mt5.last_error()
        msg = last_result.comment if last_result.comment else f"Erro Retcode {last_result.retcode}"
        if last_result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF):
            logger.warning(f"⚠️ Requote/PriceOff para {params.symbol}: {msg} | tentativa {attempt+1}/{max_retries} | preço {request['price']}")
            try:
                daily_logger.log_analysis(
                    symbol=params.symbol,
                    signal=params.side.name,
                    strategy="EXECUTION",
                    score=0,
                    rejected=True,
                    reason=f"Requote/PriceOff: {msg} (tentativa {attempt+1}/{max_retries})",
                    indicators={},
                    ml_score=0.0,
                    is_baseline=False,
                    user=os.getenv("XP3_OPERATOR", "XP3_BOT"),
                    context={"action": "ORDER_SEND_RETRY"}
                )
            except Exception:
                pass
            time.sleep(0.5)
            tick = mt5.symbol_info_tick(params.symbol)
            if not tick:
                logger.error(f"❌ Tick indisponível em retry para {params.symbol}")
                return False, "Tick indisponível em retry", 0
            request["price"] = tick.ask if params.side == OrderSide.BUY else tick.bid
            attempt += 1
            continue
        elif last_result.retcode == mt5.TRADE_RETCODE_CONNECTION:
            logger.error(f"❌ Erro de Conexão MT5 ao enviar ordem para {params.symbol}")
            logger.critical(f"🔍 DIAGNÓSTICO MT5: Cod: {err_code[0]} | Desc: {err_code[1]}")
            return False, msg, 0
        else:
            logger.error(f"❌ Erro ao enviar ordem para {params.symbol}: {msg} (retcode: {last_result.retcode})")
            logger.critical(f"🔍 DIAGNÓSTICO MT5: Cod: {err_code[0]} | Desc: {err_code[1]}")
            return False, msg, 0

    if last_result and last_result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        return False, f"Requote limite atingido ({max_retries}x)", 0

    # Verificação de Execução Parcial (Trade Policy)
    if last_result.retcode == mt5.TRADE_RETCODE_DONE:
        if last_result.volume < params.volume:
            logger.warning(f"⚠️ Execução PARCIAL para {params.symbol}: {last_result.volume:.2f}/{params.volume:.2f}")
            # Em execuções parciais, o MT5 retorna DONE, mas o volume pode ser menor.
            # O bot processa o que entrou.

    exec_latency_ms = int((time.time() - start_ts) * 1000)
    executed_price = float(getattr(last_result, "price", 0.0) or 0.0)
    req_price = float(request["price"])
    pip = utils.get_pip_size(params.symbol)
    slippage_pips = abs(executed_price - req_price) / pip if pip > 0 else 0.0
    try:
        daily_logger.log_analysis(
            symbol=params.symbol,
            signal=params.side.name,
            strategy="EXECUTION_QUALITY",
            score=0,
            rejected=False,
            reason=f"latency_ms={exec_latency_ms} slippage_pips={slippage_pips:.2f}",
            indicators={},
            ml_score=0.0,
            is_baseline=False,
            user=os.getenv("XP3_OPERATOR", "XP3_BOT"),
            context={"action": "ORDER_SEND_DONE"}
        )
    except Exception:
        pass
    try:
        max_lat_ms = int(getattr(config, "MAX_EXECUTION_LATENCY_MS", 2000))
        max_slip = float(getattr(config, "MAX_EXECUTION_SLIPPAGE_PIPS", 5.0))
        if exec_latency_ms > max_lat_ms or slippage_pips > max_slip:
            pause_min = int(getattr(config, "EXECUTION_PAUSE_MINUTES", 10))
            path = os.path.join("data", "pause_requests.json")
            os.makedirs("data", exist_ok=True)
            payload = {}
            if os.path.exists(path):
                try:
                    import json
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    payload = {}
            until_ts = int(time.time() + pause_min * 60)
            payload[params.symbol] = {
                "until": until_ts,
                "reason": f"exec_latency={exec_latency_ms}ms slippage={slippage_pips:.2f} pips",
                "minutes": pause_min
            }
            try:
                import json
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                pass
    except Exception:
        pass
    logger.info(f"✅ Ordem {params.symbol} ({params.side.name}) confirmada! Ticket: {last_result.order} | Retcode: {last_result.retcode} | Vol: {last_result.volume:.2f}/{params.volume:.2f}")
    return True, "Ordem executada com sucesso", last_result.order
