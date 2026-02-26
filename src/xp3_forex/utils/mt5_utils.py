"""MT5 connection and data utilities"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from concurrent.futures import Future
from typing import Optional, Dict, Any, List, Union
import logging
import time
import math

logger = logging.getLogger(__name__)

# Lock para operações MT5 (evita race conditions)
mt5_lock = threading.RLock()
_mt5_queue = queue.Queue()
_mt5_worker_started = False
_mt5_worker_thread = None

# Mapeamento de Timeframes (Inteiro Config -> Constante MT5)
TIMEFRAME_MAP = {
    1: mt5.TIMEFRAME_M1,
    2: mt5.TIMEFRAME_M2,
    3: mt5.TIMEFRAME_M3,
    4: mt5.TIMEFRAME_M4,
    5: mt5.TIMEFRAME_M5,
    6: mt5.TIMEFRAME_M6,
    10: mt5.TIMEFRAME_M10,
    12: mt5.TIMEFRAME_M12,
    15: mt5.TIMEFRAME_M15,
    20: mt5.TIMEFRAME_M20,
    30: mt5.TIMEFRAME_M30,
    60: mt5.TIMEFRAME_H1,      # 16385
    120: mt5.TIMEFRAME_H2,     # 16386
    180: mt5.TIMEFRAME_H3,     # 16387
    240: mt5.TIMEFRAME_H4,     # 16388
    360: mt5.TIMEFRAME_H6,     # 16389
    480: mt5.TIMEFRAME_H8,     # 16390
    720: mt5.TIMEFRAME_H12,    # 16391
    1440: mt5.TIMEFRAME_D1,    # 16408
    10080: mt5.TIMEFRAME_W1,   # 32769
    43200: mt5.TIMEFRAME_MN1   # 49153
}

def get_mt5_timeframe(tf: int) -> int:
    """Converte timeframe em minutos para constante MT5"""
    if tf in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tf]
    # Se já for um valor alto (provável constante), retorna ele mesmo
    if tf > 16000:
        return tf
    logger.warning(f"Timeframe {tf} não mapeado para constante MT5. Usando valor original.")
    return tf

def _mt5_worker():
    """Worker thread para operações MT5 seguras"""
    while True:
        item = _mt5_queue.get()
        if item is None:
            break
        func, args, kwargs, fut = item
        try:
            res = func(*args, **kwargs)
            fut.set_result(res)
        except Exception as e:
            fut.set_exception(e)
        finally:
            _mt5_queue.task_done()

def ensure_mt5_worker():
    """Garante que o worker MT5 está rodando"""
    global _mt5_worker_started, _mt5_worker_thread
    if not _mt5_worker_started:
        _mt5_worker_thread = threading.Thread(target=_mt5_worker, name="MT5Worker", daemon=True)
        _mt5_worker_thread.start()
        _mt5_worker_started = True

def mt5_exec(func, *args, **kwargs):
    """Executa função MT5 de forma thread-safe com timeout"""
    timeout = kwargs.pop('timeout', 30)
    ensure_mt5_worker()
    fut = Future()
    _mt5_queue.put((func, args, kwargs, fut))
    try:
        return fut.result(timeout=timeout)
    except Exception as e:
        logger.error(f"Erro na execução MT5 ({func.__name__}): {e}")
        return None

def mt5_shutdown_worker():
    """Encerra o worker MT5"""
    if _mt5_worker_started:
        _mt5_queue.put(None)

def resolve_symbol_name(base_symbol: str) -> Optional[str]:
    """
    Tenta encontrar o nome correto do símbolo (lidando com sufixos).
    Ex: EURUSD -> EURUSD.a, EURUSD.pro, etc.
    """
    # 1. Tenta exato
    info = mt5_exec(mt5.symbol_info, base_symbol)
    if info is not None:
        return base_symbol

    # 2. Procura na lista de todos os símbolos
    # Nota: symbols_get pode ser pesado, ideal cachear
    all_symbols = mt5_exec(mt5.symbols_get)
    if all_symbols:
        for s in all_symbols:
            if s.name.startswith(base_symbol) and len(s.name) <= len(base_symbol) + 4:
                # Verifica se é apenas sufixo (ex: EURUSD.a)
                # Evita falsos positivos como EURUSD -> EURUSDCAD (não acontece, mas segurança)
                return s.name
    
    return None

def ensure_symbol_ready(symbol: str, timeout: int = 15) -> bool:
    """
    Garante que o símbolo está selecionado no Market Watch e pronto para uso.
    """
    start_time = time.time()
    
    # 1. Seleciona no Market Watch
    if not mt5_exec(mt5.symbol_select, symbol, True):
        logger.warning(f"Falha ao selecionar {symbol} no Market Watch.")
        # Tenta resolver sufixo se falhar
        resolved = resolve_symbol_name(symbol)
        if resolved and resolved != symbol:
            logger.info(f"Símbolo resolvido: {symbol} -> {resolved}")
            if not mt5_exec(mt5.symbol_select, resolved, True):
                return False
            symbol = resolved # Atualiza para o nome correto
        else:
            return False

    # 2. Aguarda dados (tick)
    while time.time() - start_time < timeout:
        tick = mt5_exec(mt5.symbol_info_tick, symbol)
        if tick is not None:
            return True
        time.sleep(0.5)
        
    logger.warning(f"Timeout aguardando dados para {symbol}")
    return False

def get_rates(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """
    Obtém dados históricos de forma robusta com retry logic e mapeamento de timeframe.
    """
    attempts = 6 # Aumentado conforme solicitado
    base_delay = 0.5
    
    # Mapeia timeframe (CRÍTICO: Corrige erro -2 Invalid Params)
    mt5_tf = get_mt5_timeframe(timeframe)
    
    # Validação básica
    if not isinstance(count, int) or count <= 0:
        logger.error(f"Count inválido para {symbol}: {count}")
        return None
        
    for attempt in range(attempts):
        try:
            # Backoff exponencial com jitter
            if attempt > 0:
                sleep_time = base_delay * (2 ** attempt) + (np.random.random() * 0.5)
                time.sleep(sleep_time)

            # 1. Verifica/Seleciona Símbolo
            if attempt == 0: # Faz apenas na primeira ou se falhar muito
                if not mt5_exec(mt5.symbol_select, symbol, True):
                     # Tenta resolver nome se não encontrado
                     pass

            # 2. Check info (rápido)
            info = mt5_exec(mt5.symbol_info, symbol)
            if info is None:
                logger.warning(f"[{attempt+1}/{attempts}] Info não disponível para {symbol}")
                continue

            # 3. Request Data
            start_t = time.time()
            rates = mt5_exec(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, count)
            duration = time.time() - start_t

            if rates is None:
                err = mt5.last_error()
                logger.warning(f"[{attempt+1}/{attempts}] Falha copy_rates {symbol} TF:{timeframe}({mt5_tf}). Erro: {err}")
                
                # Se erro for Invalid Params (-2), aborta logo pois não vai resolver com retry
                if err[0] == -2:
                    logger.error(f"ERRO CRÍTICO (-2) PARAMETROS INVÁLIDOS: Symbol={symbol}, TF={mt5_tf}, Count={count}")
                    return None
                continue
            
            if len(rates) == 0:
                logger.warning(f"[{attempt+1}/{attempts}] Zero rates retornados para {symbol}")
                continue

            # Sucesso
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Logging rico para performance (apenas debug/info se lento)
            if duration > 1.0:
                logger.info(f"Slow rates fetch: {symbol} took {duration:.2f}s")
                
            return df
            
        except Exception as e:
            logger.error(f"Exception em get_rates {symbol}: {e}")
            
    logger.error(f"Falha definitiva get_rates {symbol} após {attempts} tentativas.")
    return None

def get_symbol_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Obtém informações detalhadas do símbolo"""
    try:
        info = mt5_exec(mt5.symbol_info, symbol)
        if info is None:
            return None
            
        return {
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'trade_tick_size': info.trade_tick_size,
            'trade_tick_value': info.trade_tick_value,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'path': info.path # Útil para verificar grupo/sufixo
        }
    except Exception as e:
        logger.error(f"Erro ao obter info do símbolo {symbol}: {e}")
        return None

def check_mt5_connection() -> bool:
    """Verifica se o MT5 está conectado"""
    try:
        return mt5_exec(mt5.terminal_info) is not None
    except:
        return False

def initialize_mt5(login: int = None, password: str = None, server: str = None, path: str = None) -> bool:
    """Inicializa conexão MT5 (assumindo logado se params não fornecidos)"""
    try:
        # Se path fornecido, usa. Senão tenta default.
        if path:
            init_ok = mt5.initialize(path=path)
        else:
            init_ok = mt5.initialize()
            
        if init_ok:
            logger.info(f"MT5 inicializado com sucesso.")
            # Opcional: Verificar login se credenciais fornecidas (mas removido conforme pedido anterior)
            return True
        else:
            logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
            return False
    except Exception as e:
        logger.error(f"Erro ao inicializar MT5: {e}")
        return False

def initialize_market_data(symbols: List[str]) -> List[str]:
    """
    Inicializa todos os símbolos no startup.
    Retorna lista de símbolos validados (com sufixo correto se necessário).
    """
    validated = []
    logger.info(f"Inicializando Market Data para {len(symbols)} símbolos...")
    
    for s in symbols:
        if ensure_symbol_ready(s, timeout=10):
            validated.append(s)
            logger.info(f"✅ Símbolo pronto: {s}")
        else:
            # Tenta resolver sufixo
            resolved = resolve_symbol_name(s)
            if resolved and ensure_symbol_ready(resolved, timeout=10):
                validated.append(resolved)
                logger.info(f"✅ Símbolo resolvido e pronto: {resolved}")
            else:
                logger.error(f"❌ Falha ao inicializar símbolo: {s}")
                
    return validated
