"""MT5 connection and data utilities"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from concurrent.futures import Future
from typing import Optional, Dict, Any, List
import logging
import time

logger = logging.getLogger(__name__)

# Lock para operações MT5 (evita race conditions)
mt5_lock = threading.RLock()
_mt5_queue = queue.Queue()
_mt5_worker_started = False
_mt5_worker_thread = None

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
    return fut.result(timeout=timeout)

def mt5_shutdown_worker():
    """Encerra o worker MT5"""
    if _mt5_worker_started:
        _mt5_queue.put(None)

def get_rates(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """Obtém dados históricos de forma robusta com retry logic"""
    attempts = 3
    timeout_seconds = 10
    
    # Garantir que os parâmetros são do tipo correto
    if not isinstance(timeframe, int):
        logger.error(f"Timeframe inválido para {symbol}: {timeframe} ({type(timeframe)})")
        return None
        
    if not isinstance(count, int):
        logger.error(f"Count inválido para {symbol}: {count} ({type(count)})")
        return None
    
    for attempt in range(attempts):
        try:
            # Tenta selecionar o símbolo no Market Watch
            if not mt5_exec(mt5.symbol_select, symbol, True, timeout=timeout_seconds):
                 # Apenas loga aviso, pode ser que já esteja selecionado ou erro temporário
                 pass

            symbol_info = mt5_exec(mt5.symbol_info, symbol, timeout=timeout_seconds)
            if symbol_info is None:
                err = mt5.last_error()
                logger.warning(f"Símbolo {symbol} não encontrado (tentativa {attempt + 1}). Erro: {err}")
                if attempt < attempts - 1:
                    time.sleep(1)
                continue
                
            rates = mt5_exec(mt5.copy_rates_from_pos, symbol, timeframe, 0, count, timeout=timeout_seconds)
            if rates is None:
                err = mt5.last_error()
                logger.warning(f"Falha ao obter rates para {symbol} (tentativa {attempt + 1}). Erro: {err}")
                if attempt < attempts - 1:
                    time.sleep(1)
                continue
            
            if len(rates) == 0:
                logger.warning(f"Rates vazios para {symbol} (tentativa {attempt + 1})")
                if attempt < attempts - 1:
                    time.sleep(1)
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter rates para {symbol} (tentativa {attempt + 1}): {e}")
            if attempt < attempts - 1:
                time.sleep(1)
            else:
                return None
    
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

def initialize_mt5(login: int, password: str, server: str, path: str) -> bool:
    """Inicializa conexão MT5"""
    try:
        if mt5.initialize(path=path):
            if True:
                logger.info(f"MT5 conectado: {server}")
                return True
            else:
                logger.error(f"Falha no login MT5: {mt5.last_error()}")
                mt5.shutdown()
                return False
        else:
            logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
            return False
    except Exception as e:
        logger.error(f"Erro ao inicializar MT5: {e}")
        return False