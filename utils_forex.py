<<<<<<< HEAD
# mt5_utils.py - XP3 PRO FOREX UTILS v4.2 INSTITUCIONAL
"""
🚀 XP3 PRO FOREX UTILS - VERSÃO INSTITUCIONAL v4.2
✅ Funções auxiliares para o bot
✅ Cálculo de indicadores técnicos
✅ Conexão MT5 e dados de mercado
✅ Cálculo de volume e SL/TP
✅ CORREÇÃO: get_tick_value e get_pip_size mais robustos
✅ Suporte a SYMBOL_MAP (para ML Optimizer)
✅ Backtesting seguro
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import time
import logging
import threading
import queue
from concurrent.futures import Future
import json
import os
from pathlib import Path
import csv
import sqlite3
from typing import Optional, Tuple, Dict, Any, List

from xp3_forex.core import config as config
from decimal import Decimal, ROUND_HALF_UP
from numba import njit

logger = logging.getLogger("XP3_UTILS")

# Lock para operações MT5 (evita race conditions)
mt5_lock = threading.RLock()
_mt5_queue = queue.Queue()
_mt5_worker_started = False
_mt5_worker_thread = None
def _mt5_worker():
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
    global _mt5_worker_started, _mt5_worker_thread
    if not _mt5_worker_started:
        _mt5_worker_thread = threading.Thread(target=_mt5_worker, name="MT5Worker", daemon=True)
        _mt5_worker_thread.start()
        _mt5_worker_started = True
def mt5_exec(func, *args, **kwargs):
    ensure_mt5_worker()
    fut = Future()
    _mt5_queue.put((func, args, kwargs, fut))
    return fut.result(timeout=30)
def mt5_shutdown_worker():
    if _mt5_worker_started:
        _mt5_queue.put(None)

# Cache para silenciamento de erros repetitivos (ex: 10016)
ERROR_CACHE = {} # {key: {"time": float, "price": float}}
INDICATOR_CACHE = {}
INDICATOR_CACHE_TTL = 60

SESSION_METRICS_LOCK = threading.Lock()
DAYREPORT_CACHE_LOCK = threading.Lock()
DAYREPORT_CACHE = {}

@njit
def ema_numba(x, period):
    alpha = 2.0 / (period + 1.0)
    result = np.empty_like(x)
    if len(x) == 0:
        return result
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1.0 - alpha) * result[i - 1]
    return result

@njit
def calculate_rsi_numba(close, period=14):
    rsi = np.zeros_like(close)
    gains = np.zeros_like(close)
    losses = np.zeros_like(close)
    for i in range(1, len(close)):
        change = close[i] - close[i - 1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = abs(change)
    avg_gain = 0.0
    avg_loss = 0.0
    end = min(len(close) - 1, period)
    for i in range(1, end + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    if end > 0:
        avg_gain /= end
        avg_loss /= end
    for i in range(period, len(close)):
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    return rsi

@njit
def calculate_atr_numba(high, low, close, period):
    tr = np.zeros_like(close)
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    if len(close) > period:
        s = 0.0
        for i in range(1, period + 1):
            s += tr[i]
        atr[period] = s / period
        for i in range(period + 1, len(close)):
            atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period
    return atr

@njit
def calculate_adx_numba(high, low, close, period=14):
    adx = np.zeros_like(close)
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)
    tr = np.zeros_like(close)
    for i in range(1, len(close)):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    smooth_plus_dm = 0.0
    smooth_minus_dm = 0.0
    smooth_tr = 0.0
    end = min(len(close) - 1, period)
    for i in range(1, end + 1):
        smooth_plus_dm += plus_dm[i]
        smooth_minus_dm += minus_dm[i]
        smooth_tr += tr[i]
    for i in range(period, len(close)):
        smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period) + plus_dm[i]
        smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period) + minus_dm[i]
        smooth_tr = smooth_tr - (smooth_tr / period) + tr[i]
        if smooth_tr == 0.0:
            adx[i] = 0.0
            continue
        plus_di = 100.0 * (smooth_plus_dm / smooth_tr)
        minus_di = 100.0 * (smooth_minus_dm / smooth_tr)
        denom = plus_di + minus_di
        dx = 0.0
        if denom > 0.0:
            dx = 100.0 * abs(plus_di - minus_di) / denom
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i - 1] * (period - 1)) + dx) / period
    return adx

def _get_session_metrics_path(date_str: str) -> str:
    return str(Path("analysis_logs") / f"session_metrics_{date_str}.json")

def _load_session_metrics(date_str: str) -> Dict[str, Any]:
    path = _get_session_metrics_path(date_str)
    if not os.path.exists(path):
        return {
            "date": date_str,
            "sessions": {},
            "last_update": None
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid json")
        data.setdefault("date", date_str)
        data.setdefault("sessions", {})
        return data
    except Exception:
        return {
            "date": date_str,
            "sessions": {},
            "last_update": None
        }

def _save_session_metrics(date_str: str, data: Dict[str, Any]) -> None:
    path = _get_session_metrics_path(date_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_session_metrics(session_name: str, executed: bool, rejected: bool, reason: str) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    sess = (session_name or "UNKNOWN").upper()
    reason_key = (reason or "N/A").strip()
    if len(reason_key) > 120:
        reason_key = reason_key[:120] + "..."

    with SESSION_METRICS_LOCK:
        data = _load_session_metrics(date_str)
        sessions = data.setdefault("sessions", {})
        block = sessions.setdefault(sess, {"analyzed": 0, "executed": 0, "rejected": 0, "reasons": {}})
        block["analyzed"] = int(block.get("analyzed", 0)) + 1
        if executed and not rejected:
            block["executed"] = int(block.get("executed", 0)) + 1
        if rejected:
            block["rejected"] = int(block.get("rejected", 0)) + 1
            reasons = block.setdefault("reasons", {})
            reasons[reason_key] = int(reasons.get(reason_key, 0)) + 1
        data["last_update"] = datetime.now().isoformat(timespec="seconds")
        _save_session_metrics(date_str, data)

def get_session_metrics_summary() -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    with SESSION_METRICS_LOCK:
        data = _load_session_metrics(date_str)
    sessions = data.get("sessions") or {}
    if not sessions:
        return "Sem métricas por sessão ainda (aguardando análises)."

    lines = [f"<b>Métricas por Sessão</b> ({date_str})"]
    for sess in sorted(sessions.keys()):
        s = sessions[sess] or {}
        analyzed = int(s.get("analyzed", 0))
        executed = int(s.get("executed", 0))
        rejected = int(s.get("rejected", 0))
        exec_rate = (executed / analyzed * 100) if analyzed > 0 else 0.0
        lines.append(f"\n<b>{sess}</b> | Análises: {analyzed} | Exec: {executed} | Rej: {rejected} | Exec%: {exec_rate:.1f}%")
        reasons = s.get("reasons") or {}
        if reasons:
            top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append("Top motivos:")
            for r, c in top:
                lines.append(f"- {c}x {r}")
    text = "\n".join(lines)
    if len(text) > 3900:
        text = text[:3900] + "\n..."
    return text

def get_session_metrics_summary_by_date(date_str: str) -> str:
    with SESSION_METRICS_LOCK:
        data = _load_session_metrics(date_str)
    sessions = data.get("sessions") or {}
    if not sessions:
        return f"Sem métricas por sessão em {date_str}."

    lines = [f"<b>Métricas por Sessão</b> ({date_str})"]
    for sess in sorted(sessions.keys()):
        s = sessions[sess] or {}
        analyzed = int(s.get("analyzed", 0))
        executed = int(s.get("executed", 0))
        rejected = int(s.get("rejected", 0))
        exec_rate = (executed / analyzed * 100) if analyzed > 0 else 0.0
        lines.append(f"\n<b>{sess}</b> | Análises: {analyzed} | Exec: {executed} | Rej: {rejected} | Exec%: {exec_rate:.1f}%")
        reasons = s.get("reasons") or {}
        if reasons:
            top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append("Top motivos:")
            for r, c in top:
                lines.append(f"- {c}x {r}")
    text = "\n".join(lines)
    if len(text) > 3900:
        text = text[:3900] + "\n..."
    return text

def _escape_html(text: str) -> str:
    try:
        import html
        return html.escape(text or "")
    except Exception:
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def generate_day_report(date_str: Optional[str] = None) -> str:
    import re
    import collections

    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    analysis_path = str(Path("analysis_logs") / f"analysis_log_{date_str}.txt")
    runtime_log_path = str(Path("logs") / "xp3_forex.log")

    cache_key = date_str
    analysis_mtime = os.path.getmtime(analysis_path) if os.path.exists(analysis_path) else 0
    runtime_mtime = os.path.getmtime(runtime_log_path) if os.path.exists(runtime_log_path) else 0

    with DAYREPORT_CACHE_LOCK:
        cached = DAYREPORT_CACHE.get(cache_key)
        if cached and cached.get("analysis_mtime") == analysis_mtime and cached.get("runtime_mtime") == runtime_mtime:
            return cached.get("text", "")

    lines_out = [f"📅 <b>Day Report</b> ({date_str})"]

    if date_str == datetime.now().strftime("%Y-%m-%d"):
        try:
            m = calculate_current_metrics()
            if m:
                lines_out.append(
                    "\n<b>Métricas (Hoje)</b>\n"
                    f"Trades: {m.get('trades_today', 0)} | WR: {m.get('win_rate', 0):.1f}% | "
                    f"PF: {m.get('profit_factor_today', 0):.2f} | PnL: ${m.get('pnl_today', 0):+.2f} | "
                    f"DD: {m.get('max_dd_intraday', 0):.2f}%"
                )
        except Exception:
            pass

    reasons = collections.Counter()
    symbols = collections.Counter()
    strategies = collections.Counter()
    signals = collections.Counter()
    ml_vals = []
    first_ts = None
    last_ts = None

    re_ts = re.compile(r'(\d\d:\d\d:\d\d)\s*\|\s*([^|]+?)\s*\|')

    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "📊" in line and "Sinal:" in line and "Estratégia:" in line and "ML:" in line:
                        try:
                            parts = line.split("|")
                            sig = parts[0].split("Sinal:")[1].strip()
                            strat = parts[1].split("Estratégia:")[1].strip()
                            signals[sig] += 1
                            strategies[strat] += 1
                            ml_part = parts[3].split("ML:")[1]
                            ml_num = re.sub(r"[^0-9.]", "", ml_part)
                            if ml_num:
                                ml_vals.append(float(ml_num))
                        except Exception:
                            pass
                        continue

                    if "💬" in line and "Motivo:" in line:
                        try:
                            reason = line.split("Motivo:", 1)[1].strip()
                            reasons[reason] += 1
                        except Exception:
                            pass
                        continue

                    if " | " in line and ":" in line:
                        m = re_ts.search(line)
                        if m:
                            t = m.group(1)
                            sym = m.group(2).strip()
                            symbols[sym] += 1
                            if first_ts is None:
                                first_ts = t
                            last_ts = t
        except Exception:
            pass

    total_entries = int(sum(symbols.values()))
    if total_entries:
        lines_out.append(
            "\n<b>Atividade (analysis log)</b>\n"
            f"Intervalo: {first_ts} → {last_ts}\n"
            f"Entradas: {total_entries} | Ativos: {len(symbols)}"
        )

        if ml_vals:
            ml_vals.sort()
            ml_min = ml_vals[0]
            ml_med = ml_vals[len(ml_vals) // 2]
            ml_max = ml_vals[-1]
            lines_out.append(f"ML (min/med/max): {ml_min:.0f}/{ml_med:.0f}/{ml_max:.0f}")

        top_reasons = reasons.most_common(10)
        if top_reasons:
            lines_out.append("\n<b>Top motivos (analysis)</b>")
            for r, c in top_reasons:
                lines_out.append(f"- {c}x {_escape_html(r)}")

        top_symbols = symbols.most_common(10)
        if top_symbols:
            lines_out.append("\n<b>Top ativos (activity)</b>")
            for s, c in top_symbols:
                lines_out.append(f"- {c}x {_escape_html(s)}")

        top_signals = signals.most_common(6)
        if top_signals:
            lines_out.append("\n<b>Sinais</b>")
            for s, c in top_signals:
                lines_out.append(f"- {c}x {_escape_html(s)}")

        top_strats = strategies.most_common(6)
        if top_strats:
            lines_out.append("\n<b>Estratégias</b>")
            for s, c in top_strats:
                lines_out.append(f"- {c}x {_escape_html(s)}")
    else:
        lines_out.append("\n<b>Atividade (analysis log)</b>\nSem dados (arquivo ausente ou vazio).")

    session_metrics_path = _get_session_metrics_path(date_str)
    if os.path.exists(session_metrics_path):
        try:
            lines_out.append("\n" + get_session_metrics_summary_by_date(date_str))
        except Exception:
            pass

    runtime_counts = collections.Counter()
    if os.path.exists(runtime_log_path):
        try:
            with open(runtime_log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith(date_str):
                        continue
                    if "Thread FastLoop MORTA" in line:
                        runtime_counts["fastloop_dead"] += 1
                    if "Tentativa de iniciar FastLoop duplicado" in line:
                        runtime_counts["fastloop_dup"] += 1
                    if "BLOQUEIO INSTITUCIONAL" in line:
                        runtime_counts["risk_block"] += 1
                    if "Spread Alto" in line:
                        runtime_counts["spread_high"] += 1
                    if "ORDEM EXECUTADA" in line:
                        runtime_counts["orders_executed"] += 1
        except Exception:
            pass

    if runtime_counts:
        lines_out.append("\n<b>Estabilidade (runtime)</b>")
        for k in ["orders_executed", "spread_high", "risk_block", "fastloop_dead", "fastloop_dup"]:
            if k in runtime_counts:
                lines_out.append(f"- {k}: {runtime_counts[k]}")

    text = "\n".join(lines_out)
    if len(text) > 3900:
        text = text[:3900] + "\n..."

    with DAYREPORT_CACHE_LOCK:
        DAYREPORT_CACHE[cache_key] = {"analysis_mtime": analysis_mtime, "runtime_mtime": runtime_mtime, "text": text}

    return text

# ✅ Land Trading: Mapeamento de Aliases (Nomenclatura Corretora)
SYMBOL_ALIASES = {
    "NAS100": ["USTEC", "US100", "NAS100.cash", "NAS100.m", "NAS100.raw"],
    "US30": ["US30.cash", "US30.m", "US30.raw", "WS30"],
    "GER40": ["GER40.cash", "DE40", "DAX40", "GER40.m", "DE40.cash"],
    "UK100": ["UK100.cash", "FTSE100", "UK100.m"],
    "US500": ["SPX500", "USA500", "US500.cash"],
    "XAUUSD": ["GOLD", "XAUUSD.raw", "XAUUSD.m"],
    "XAGUSD": ["SILVER", "XAGUSD.raw"],
    "BTCUSD": ["BTCUSD.spot", "BTCUSD.m"],
}

# ✅ Land Trading: Normalização de Símbolos
def normalize_symbol(symbol: str) -> str:
    """
    Tenta encontrar a variação correta do símbolo no Market Watch.
    Utiliza aliases e sufixos comuns.
    """
    if not check_mt5_connection():
        return symbol
        
    # 1. Verifica se o símbolo já é válido
    with mt5_lock:
        if mt5.symbol_select(symbol, True):
            return symbol
            
    # 2. Tenta Aliases
    potential_names = SYMBOL_ALIASES.get(symbol, [])
    
    # 3. Adiciona variações comuns de sufixo
    variants = [symbol + ".raw", symbol + ".m", symbol + ".cash", symbol + "#", symbol + ".spot"]
    for p in potential_names:
        variants.append(p)
        variants.append(p + ".raw")
        variants.append(p + ".m")
    
    with mt5_lock:
        for v in variants:
            if mt5.symbol_select(v, True):
                return v

        # 4. Busca Fuzzy (Inteligente - Land Trading)
        # Se falhar tudo, varre todos os símbolos e tenta achar substring
        all_symbols = mt5.symbols_get()
        if all_symbols:
            for s in all_symbols:
                # Se o nome base (ex: EURUSD) estiver contido no nome do MT5 (ex: EURUSD.a)
                if symbol in s.name:
                    if mt5.symbol_select(s.name, True):
                        logger.info(f"🔍 Auto-Correction: '{symbol}' -> '{s.name}'")
                        return s.name
    
    return symbol

# ✅ Land Trading: Check de Sincronização de Nomes
def check_symbol_sync(symbols: List[str]):
    """
    Verifica se os símbolos do config existem no Market Watch e avisa se houver erro.
    Também popula uma lista de exclusão para símbolos não encontrados.
    """
    if not check_mt5_connection():
        return
        
    # ✅ REQUISITO: Imprime no log e no TERMINAL todos os símbolos que o MT5 está vendo
    with mt5_lock:
        all_mt5_symbols = [s.name for s in mt5.symbols_get()]
    
    logger.info(f"📊 Diagnóstico MT5: Mercado oferece {len(all_mt5_symbols)} símbolos.")
    print("\n" + "="*50)
    print(f"🔍 DISPONÍVEIS NO MT5 ({len(all_mt5_symbols)} ativos):")
    print(all_mt5_symbols[:200]) # Mostra os primeiros 200 para não estourar o buffer
    print("="*50 + "\n")
    
    # Adiciona ao config uma lista de exclusão dinâmica se não existir
    if not hasattr(config, 'BLACKLISTED_SYMBOLS'):
        config.BLACKLISTED_SYMBOLS = set()

    for sym in symbols:
        if not mt5.symbol_select(sym, True):
            best_v = normalize_symbol(sym)
            if best_v != sym:
                logger.warning(f"⚠️ Símbolo '{sym}' não encontrado! Sugestão: Use '{best_v}' no config_forex.py")
            else:
                logger.error(f"❌ Símbolo '{sym}' totalmente inacessível. Adicionado à lista de exclusão.")
                config.BLACKLISTED_SYMBOLS.add(sym)
        else:
            # ✅ WARMUP: Força download de histórico
            print(f"🔥 Aquecendo dados para {sym}...")
            with mt5_lock:
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 1000)
            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️ Falha no aquecimento de dados para {sym}")
            else:
                logger.info(f"✅ {sym}: {len(rates)} candles carregados.")

# ===========================
# MT5 CONNECTION
# ===========================
def ensure_mt5_connection() -> bool:
    term_path = getattr(config, 'MT5_TERMINAL_PATH', None)
    ok_init = mt5_exec(mt5.initialize, path=term_path)
    if not ok_init:
        logger.error(f"❌ Falha ao inicializar MT5 em {term_path}")
        return False
    login = str(getattr(config, 'MT5_LOGIN', '')).strip()
    passwd = str(getattr(config, 'MT5_PASSWORD', '')).strip()
    server = str(getattr(config, 'MT5_SERVER', '')).strip()
    if login and passwd and server:
        ok_login = mt5_exec(mt5.login, login, passwd, server)
        if not ok_login:
            logger.error(f"❌ Falha ao logar no MT5: {mt5.last_error()}")
            mt5_exec(mt5.shutdown)
            return False
    ti = mt5_exec(mt5.terminal_info)
    if ti is None:
        logger.error("❌ Terminal MT5 não disponível após init.")
        return False
    return True

def check_mt5_connection() -> bool:
    ti = mt5_exec(mt5.terminal_info)
    ai = mt5_exec(mt5.account_info)
    return ti is not None and ai is not None

# ===========================
# MARKET STATUS
# ===========================
def get_brasilia_time() -> datetime:
    """Retorna o horário de Brasília considerando o SERVER_OFFSET se necessário"""
    # Horário de Brasília é UTC-3
    tz_br = pytz.timezone('America/Sao_Paulo')
    now_br = datetime.now(tz_br)
    
    # Se o sistema precisar de um offset específico em relação ao servidor MT5,
    # o usuário pode configurar no config.py. 
    # Por padrão, estamos usando o horário da máquina local (UTC-3).
    if getattr(config, 'SERVER_OFFSET', 0) != 0:
        now_br += timedelta(hours=config.SERVER_OFFSET)
        
    return now_br

def get_current_trading_session() -> Dict[str, str]:
    """Retorna a sessão de trading atual baseada no horário de Brasília"""
    br_time = get_brasilia_time()
    current_hm = br_time.strftime("%H:%M")
    
    def is_between(start, end, now):
        if start <= end:
            return start <= now <= end
        else: # Crosses midnight
            return now >= start or now <= end

    # Asian Session: 22:00 às 05:00
    if is_between(config.ASIAN_SESSION_START, config.ASIAN_SESSION_END, current_hm):
        return {
            "name": "ASIAN",
            "display": "ASIÁTICA (SNIPER MODE)",
            "emoji": "🎌",
            "color": "magenta"
        }
    
    # Golden Hour: 10:00 às 14:00
    if is_between(config.GOLDEN_HOUR_START, config.GOLDEN_HOUR_END, current_hm):
        return {
            "name": "GOLDEN",
            "display": "HORÁRIO DE OURO",
            "emoji": "🌟",
            "color": "gold"
        }
    
    # Protection: 18:00 às 22:00
    if is_between(config.PROTECTION_SESSION_START, config.PROTECTION_SESSION_END, current_hm):
        return {
            "name": "PROTECTION",
            "display": "PROTEÇÃO (RISCO ALTO)",
            "emoji": "🛡️",
            "color": "red"
        }
    
    # Normal: 05:00 às 09:59 e 14:01 às 17:59
    if is_between(config.NORMAL_SESSION_1_START, config.NORMAL_SESSION_1_END, current_hm) or \
       is_between(config.NORMAL_SESSION_2_START, config.NORMAL_SESSION_2_END, current_hm):
        return {
            "name": "NORMAL",
            "display": "NORMAL",
            "emoji": "⚖️",
            "color": "blue"
        }

    return {
        "name": "OFF_HOURS",
        "display": "FORA DE HORÁRIO",
        "emoji": "💤",
        "color": "gray"
    }

def is_market_open() -> bool:
    """Verifica se o mercado está aberto."""
    market_status = get_market_status()
    return market_status['status'] == 'open'

def get_market_status() -> Dict[str, str]:
    """Retorna o status atual do mercado."""
    utc_now = datetime.now(pytz.utc)
    weekday = utc_now.weekday()  # Monday=0, Sunday=6
    hour = utc_now.hour

    # Fechamento: Sexta 21:00 UTC até Domingo 21:00 UTC
    is_closed = False
    
    if weekday == 5:  # Sábado
        is_closed = True
    elif weekday == 4 and hour >= 21:  # Sexta após 21:00 UTC
        is_closed = True
    elif weekday == 6 and hour < 21:   # Domingo antes das 21:00 UTC
        is_closed = True

    if is_closed:
        return {'status': 'closed', 'message': 'Mercado Fechado (Fim de Semana)', 'emoji': '💤', 'color': 'red'}

    return {'status': 'open', 'message': 'Mercado Aberto', 'emoji': '🟢', 'color': 'green'}

def _parse_hhmm_to_minutes(hhmm: str) -> int:
    parts = str(hhmm).strip().split(":")
    if len(parts) != 2:
        return 0
    h = int(parts[0])
    m = int(parts[1])
    return (h * 60) + m

def get_weekend_protection_state() -> Dict[str, Any]:
    utc_now = datetime.now(pytz.utc)
    weekday = utc_now.weekday()
    now_min = utc_now.hour * 60 + utc_now.minute

    close_min = _parse_hhmm_to_minutes(getattr(config, 'FRIDAY_MARKET_CLOSE_UTC', "21:00"))
    entry_cutoff_min = _parse_hhmm_to_minutes(getattr(config, 'FRIDAY_ENTRY_CUTOFF_UTC', "20:00"))
    force_close_min = _parse_hhmm_to_minutes(getattr(config, 'FRIDAY_FORCE_CLOSE_UTC', "20:55"))

    state = {
        "block_entries": False,
        "force_close": False,
        "reason": ""
    }

    if weekday == 4:
        if now_min >= entry_cutoff_min and now_min < close_min:
            state["block_entries"] = True
            state["reason"] = "Fim de pregão (Sexta): novas entradas bloqueadas"
        if now_min >= force_close_min and now_min < close_min:
            state["block_entries"] = True
            state["force_close"] = True
            state["reason"] = "Fim de pregão (Sexta): fechando posições para evitar gap"

    if weekday == 6:
        open_min = _parse_hhmm_to_minutes(getattr(config, 'SUNDAY_MARKET_OPEN_UTC', "21:00"))
        buffer_min = int(getattr(config, 'SUNDAY_OPEN_BUFFER_MINUTES', 30))
        if now_min >= open_min and now_min < (open_min + buffer_min):
            state["block_entries"] = True
            state["reason"] = "Abertura (Domingo): buffer de spread/liquidez (sem entradas)"

    return state

# ===========================
# SYMBOL INFO
# ===========================
def get_symbol_info(symbol: str) -> Optional[mt5.SymbolInfo]:
    info = mt5_exec(mt5.symbol_info, symbol)
    if info is None:
        logger.warning(f"⚠️ Símbolo {symbol} não encontrado ou indisponível.")
        return None
    if not info.visible:
        ok_sel = mt5_exec(mt5.symbol_select, symbol, True)
        if not ok_sel:
            logger.warning(f"⚠️ Símbolo {symbol} não visível e não pôde ser selecionado.")
            return None
        info = mt5_exec(mt5.symbol_info, symbol)
        if info is None:
            logger.warning(f"⚠️ Símbolo {symbol} ainda indisponível após seleção.")
            return None
    return info

def get_pip_size(symbol: str) -> float:
    """
    Retorna o tamanho do pip para um símbolo.
    ✅ CORRIGIDO: Usa info.point e ajusta para JPY.
    """
    info = get_symbol_info(symbol)
    if info:
        # info.point é o menor incremento de preço (tick size)
        # Para a maioria dos pares Forex, 1 pip = 10 pontos (ex: 0.0001 para EURUSD, onde point é 0.00001)
        # Para JPY, 1 pip = 1 ponto (ex: 0.01 para USDJPY, onde point é 0.001)
        # A definição de "pip" pode variar, mas "point" é o menor incremento de preço.
        if "JPY" in symbol.upper(): # Verifica se é um par JPY
            return info.point * 10 # Ex: 0.001 * 10 = 0.01 (1 pip para JPY)
        return info.point * 10 # Ex: 0.00001 * 10 = 0.0001 (1 pip para EURUSD)
    logger.warning(f"⚠️ Não foi possível obter info para {symbol} para calcular pip size. Usando 0.0001.")
    return 0.0001 # Default para a maioria dos pares

def get_tick_value(symbol: str) -> float:
    """
    Retorna o valor de 1 pip em moeda da conta para 1 lote padrão (100.000 unidades).
    ✅ CORRIGIDO: Usa trade_tick_value_profit e info.point para calcular o valor de 1 pip.
    """
    info = get_symbol_info(symbol)
    if not info:
        logger.warning(f"⚠️ Não foi possível obter info para {symbol}. Usando tick_value padrão de 1.0.")
        return 1.0 # Default seguro

    # info.trade_tick_value_profit é o valor de 1 tick em moeda da conta para 1 lote.
    # info.point é o tamanho de 1 tick.
    # get_pip_size(symbol) retorna o tamanho de 1 pip (ex: 0.0001 para EURUSD).

    # Valor de 1 pip em moeda da conta para 1 lote = (tamanho de 1 pip / tamanho de 1 tick) * valor de 1 tick
    if info.point == 0:
        logger.warning(f"⚠️ info.point é zero para {symbol}. Não é possível calcular pip value. Usando 1.0.")
        return 1.0

    pip_value_per_lot = (get_pip_size(symbol) / info.point) * info.trade_tick_value_profit

    if pip_value_per_lot <= 0:
        logger.warning(f"⚠️ Pip value calculado é zero ou negativo para {symbol}. Usando 1.0.")
        return 1.0

    return pip_value_per_lot

# ===========================
# DATA RETRIEVAL
# ===========================
def get_rates(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    try:
        symbol_info = mt5_exec(mt5.symbol_info, symbol)
        if symbol_info is None:
            if symbol != "PANEL_UPDATE":
                logger.error(f"❌ Símbolo {symbol} não existe na plataforma MT5")
            return None
        ok_sel = mt5_exec(mt5.symbol_select, symbol, True)
        if not ok_sel:
            if symbol != "PANEL_UPDATE":
                logger.error(f"❌ Falha ao selecionar {symbol} no Market Watch")
            return None
        logger.debug(f"🔍 Buscando {count} velas de {symbol} no timeframe {timeframe}")
        rates = mt5_exec(mt5.copy_rates_from_pos, symbol, timeframe, 1, count)
        if rates is None or len(rates) == 0:
            logger.debug(f"⚠️ Tentativa com start_pos=1 falhou, tentando start_pos=0")
            rates = mt5_exec(mt5.copy_rates_from_pos, symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            if err[0] != 1:
                logger.warning(f"⚠️ Falha ao obter rates para {symbol} (TF:{timeframe}, Count:{count}): Código={err[0]}, Msg={err[1]}")
            else:
                logger.warning(f"⚠️ {symbol}: Nenhum dado histórico disponível (rates vazias)")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logger.error(f"❌ Erro crítico em get_rates ({symbol}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def safe_copy_rates(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """
    Função segura para obter rates, usada no backtest do ML Optimizer.
    Garante conexão e tratamento de erros.
    """
    if not ensure_mt5_connection():
        logger.error("❌ MT5 não conectado para safe_copy_rates.")
        return None
    return get_rates(symbol, timeframe, count)

# ===========================
# INDICATORS (v4.2 - PATCH 1)
# ===========================
def get_indicators_forex(
    symbol: str,
    ema_short: int = None,
    ema_long: int = None,
    rsi_period: int = None, # ✅ NOVO: Período RSI otimizado
    rsi_low: int = None,
    rsi_high: int = None,
    adx_period: int = None, # ✅ NOVO: Período ADX otimizado
    bb_period: int = None,  # ✅ NOVO: Período BB otimizado
    bb_dev: float = None    # ✅ NOVO: Desvio BB otimizado
) -> dict:
    """
    ✅ v4.2: Calcula indicadores técnicos para um símbolo.
    Aceita parâmetros otimizados para EMA, RSI, ADX e Bollinger Bands.
    """
    # Usa parâmetros otimizados se fornecidos, senão usa config padrão
    ema_short_period = ema_short if ema_short is not None else config.EMA_SHORT_PERIOD
    ema_long_period = ema_long if ema_long is not None else config.EMA_LONG_PERIOD
    rsi_calc_period = rsi_period if rsi_period is not None else config.RSI_PERIOD
    rsi_low_limit = rsi_low if rsi_low is not None else config.RSI_LOW_LIMIT
    rsi_high_limit = rsi_high if rsi_high is not None else config.RSI_HIGH_LIMIT
    adx_calc_period = adx_period if adx_period is not None else config.ADX_PERIOD
    bb_calc_period = bb_period if bb_period is not None else config.BB_PERIOD
    bb_calc_dev = bb_dev if bb_dev is not None else config.BB_DEVIATION

    # ✅ REQUISITO: Validação de símbolo no início
    real_symbol = normalize_symbol(symbol)
    ok_sel = mt5_exec(mt5.symbol_select, real_symbol, True)
    if not ok_sel:
            logger.error(f"❌ Símbolo {symbol}/{real_symbol} não encontrado no Market Watch")
            return {"error": True, "message": f"Symbol {symbol} not found in MT5"}

    timeframe = mt5.TIMEFRAME_M15
    last_ts = None
    try:
        r = mt5_exec(mt5.copy_rates_from_pos, real_symbol, timeframe, 0, 1)
        if r is not None and len(r) > 0:
            last_ts = int(r[0]['time'])
    except Exception:
        last_ts = None
    if last_ts is not None:
        cache_key = f"{real_symbol}:{last_ts}:{ema_short_period}:{ema_long_period}:{rsi_calc_period}:{adx_calc_period}:{bb_calc_period}:{bb_calc_dev}"
        c = INDICATOR_CACHE.get(cache_key)
        if c:
            return c["data"]
    # Precisamos de dados suficientes para todos os indicadores
    max_period = max(ema_long_period, rsi_calc_period, adx_calc_period, bb_calc_period)
    df = get_rates(real_symbol, timeframe, max_period + 50)
    
    # ✅ REQUISITO: Nunca retornar None, retornar dicionário de erro
    if df is None or df.empty or len(df) < max_period:
        return {"error": True, "message": f"Dados insuficientes para {real_symbol}"}

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema_short_arr = ema_numba(close, ema_short_period)
    ema_long_arr = ema_numba(close, ema_long_period)
    df['ema_short'] = pd.Series(ema_short_arr)
    df['ema_long'] = pd.Series(ema_long_arr)
    ema_trend = "UP" if df['ema_short'].iloc[-1] > df['ema_long'].iloc[-1] else "DOWN"

    rsi_arr = calculate_rsi_numba(close, rsi_calc_period)
    df['rsi'] = pd.Series(rsi_arr)
    rsi_now = df['rsi'].iloc[-1]

    adx_arr = calculate_adx_numba(high, low, close, adx_calc_period)
    df['adx'] = pd.Series(adx_arr)
    adx_now = df['adx'].iloc[-1]

    atr_arr = calculate_atr_numba(high, low, close, adx_calc_period)
    df['atr'] = pd.Series(atr_arr)
    atr_now = df['atr'].iloc[-1]
    pip_size = get_pip_size(symbol)
    atr_pips = atr_now / pip_size if pip_size > 0 else 0
    try:
        atr20_arr = calculate_atr_numba(high, low, close, 20)
        atr20 = float(atr20_arr[-1]) if len(atr20_arr) else 0.0
    except Exception:
        atr20 = 0.0

    # Bollinger Bands
    df['sma_bb'] = df['close'].rolling(window=bb_calc_period).mean()
    df['std_bb'] = df['close'].rolling(window=bb_calc_period).std()
    df['bb_upper'] = df['sma_bb'] + (df['std_bb'] * bb_calc_dev)
    df['bb_lower'] = df['sma_bb'] - (df['std_bb'] * bb_calc_dev)
    bb_upper = df['bb_upper'].iloc[-1]
    bb_lower = df['bb_lower'].iloc[-1]
    bb_width = (bb_upper - bb_lower) / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0
    # MTF EMA200 H4 (estrita)
    ema200_h4 = None
    try:
        if getattr(config, 'ENABLE_MULTI_TIMEFRAME', True):
            rates_h4 = get_rates(real_symbol, mt5.TIMEFRAME_H4, getattr(config, 'MULTI_TF_EMA_PERIOD', 200) + 5)
            if rates_h4 is not None and not rates_h4.empty:
                ema200_h4 = float(rates_h4['close'].ewm(span=getattr(config, 'MULTI_TF_EMA_PERIOD', 200), adjust=False).mean().iloc[-1])
    except Exception:
        ema200_h4 = None

    # MACD (12, 26, 9) - Implementação Manual para evitar dependência de TA-Lib
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    macd_val = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    macd_hist = df['macd_hist'].iloc[-1]

    # Spread e Volume
    info = get_symbol_info(symbol)
    spread_points = info.spread if info else 0
    spread_pips = (spread_points * info.point) / pip_size if info and pip_size > 0 else 0
    # Verifica se 'tick_volume' existe antes de usar
    volume_ratio = df['tick_volume'].iloc[-1] / df['tick_volume'].mean() if 'tick_volume' in df.columns and df['tick_volume'].mean() > 0 else 0
    
    # ✅ REQUISITO: Flexibilização Inteligente de Spreads por Categoria
    symbol_upper = symbol.upper()
    if any(idx in symbol_upper for idx in ["US30", "NAS100", "USTEC", "DE40", "GER40", "UK100", "US500", "USA500"]):
        max_spread = getattr(config, 'MAX_SPREAD_INDICES', 150)
        spread_ok = spread_points <= max_spread
    elif any(met in symbol_upper for met in ["XAU", "XAG", "GOLD", "SILVER"]):
        max_spread = getattr(config, 'MAX_SPREAD_METALS', 60)
        spread_ok = spread_points <= max_spread
    elif any(crypto in symbol_upper for crypto in ["BTC", "ETH", "SOL", "ADA", "BNB"]):
        max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
        spread_ok = spread_points <= max_spread
    else:
        max_spread = getattr(config, 'MAX_SPREAD_FOREX', 25)
        spread_ok = spread_points <= max_spread

    out = {
        "time": df['time'].iloc[-1],
        "open": df['open'].iloc[-1],
        "high": df['high'].iloc[-1],
        "low": df['low'].iloc[-1],
        "close": df['close'].iloc[-1],
        "df": df, # ✅ NOVO: DataFrame completo para ML Optimizer
        "ema_short": df['ema_short'].iloc[-1],
        "ema_long": df['ema_long'].iloc[-1],
        "ema_trend": ema_trend,
        "rsi": rsi_now,
        "adx": adx_now,
        "atr": atr_now,
        "atr_pips": atr_pips,
        "atr20": atr20,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "ema200_h4": ema200_h4,
        "spread_points": spread_points,
        "spread_pips": spread_pips,
        "volume_ratio": volume_ratio,
        "spread_ok": spread_ok,
        "rsi_low_limit": rsi_low_limit,
        "rsi_high_limit": rsi_high_limit,
        "rsi_high_limit": rsi_high_limit,
        "current_price": df['close'].iloc[-1], # Adiciona para facilitar o cálculo de SL/TP
        "macd": macd_val,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist
    }
    try:
        last_ts2 = int(df['time'].iloc[-1].timestamp())
        cache_key = f"{real_symbol}:{last_ts2}:{ema_short_period}:{ema_long_period}:{rsi_calc_period}:{adx_calc_period}:{bb_calc_period}:{bb_calc_dev}"
        INDICATOR_CACHE[cache_key] = {"data": out}
    except Exception:
        pass
    return out
def get_volatility_regime(symbol: str, df: Optional[pd.DataFrame] = None) -> str:
    try:
        real_symbol = normalize_symbol(symbol)
        if df is None:
            df = get_rates(real_symbol, mt5.TIMEFRAME_M15, 120)
        if df is None or df.empty or len(df) < 100:
            return "NORMAL"
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        atr_short = calculate_atr_numba(high, low, close, 14)
        atr_long = calculate_atr_numba(high, low, close, 100)
        a_s = float(atr_short[-1]) if len(atr_short) else 0.0
        a_l = float(atr_long[-1]) if len(atr_long) else 0.0
        if a_s <= 0 or a_l <= 0:
            return "NORMAL"
        if a_s > 1.5 * a_l:
            return "HIGH"
        if a_s < 0.7 * a_l:
            return "LOW"
        return "NORMAL"
    except Exception:
        return "NORMAL"
def sync_market_watch(required_symbols):
    """
    Remove todos os ativos desnecessários e adiciona apenas os que o bot usa.
    """
    try:
        # 1. Pega todos os símbolos que estão ATUALMENTE no Market Watch
        current_symbols = mt5_exec(mt5.symbols_get)
        current_names = [s.name for s in current_symbols] if current_symbols else []

        required_raw = list(required_symbols or [])
        required_resolved = []
        missing = []
        for s in required_raw:
            if not s:
                continue
            resolved = normalize_symbol(str(s))
            if resolved and resolved != s:
                logger.info(f"🔁 Market Watch: '{s}' -> '{resolved}'")
            ok_select = mt5_exec(mt5.symbol_select, resolved, True)
            if not ok_select:
                missing.append(str(s))
                continue
            required_resolved.append(resolved)

        required_set = set(required_resolved)
        logger.info(f"🔄 Sincronizando Market Watch para {len(required_set)} ativos...")

        # 2. Adiciona os ativos necessários primeiro
        kept = []
        for symbol in required_resolved:
            ok_select = mt5_exec(mt5.symbol_select, symbol, True)
            if ok_select:
                kept.append(symbol)

        # 3. Remove os que NÃO estão na lista (opcional, mas limpa o terminal)
        removed = []
        for s_name in current_names:
            if s_name not in required_set:
                # Tenta remover (só funciona se não houver gráfico aberto do ativo)
                ok = mt5_exec(mt5.symbol_select, s_name, False)
                if ok:
                    removed.append(s_name)
        
        logger.info(f"✅ Market Watch sincronizado | kept={len(kept)} removed={len(removed)} missing={len(missing)}")
        return {"ok": True, "kept": kept, "removed": removed, "missing": missing}
    except Exception as e:
        logger.error(f"❌ Erro na sincronização do Market Watch: {e}")
        return {"ok": False, "error": str(e), "kept": [], "removed": [], "missing": []}
# ===========================
# POSITION SIZING
# ===========================
def calculate_position_size_atr_forex(symbol: str, price: float, atr_pips: float, sl_atr_mult: float = 2.0, risk_multiplier: float = 1.0) -> float:
    """
    Calcula o volume da posição baseado no ATR e no risco por trade.
    ✅ v5.1: Gestão de Money Management Cirúrgica (Risco Exato com base no SL)
    """
    account_info = mt5_exec(mt5.account_info)
    if not account_info:
        logger.error("❌ Não foi possível obter informações da conta para calcular o volume.")
        return getattr(config, 'DEFAULT_LOT', 0.01)

    equity = account_info.equity
    balance = account_info.balance
    
    # --- GESTÃO DE RISCO DINÂMICA v5.0 ---
    # 1. Verifica perda diária acumulada
    daily_profit = equity - balance 
    daily_loss_pct = abs(min(0, daily_profit)) / balance if balance > 0 else 0
    
    # Bloqueio Total (Equity Guard)
    max_loss = getattr(config, 'MAX_DAILY_LOSS_PCT', 0.02)
    if daily_loss_pct >= max_loss:
        logger.warning(f"🚨 EQUITY GUARD: Perda diária ({daily_loss_pct:.2%}) atingiu limite ({max_loss:.2%})")
        return 0.0

    # Redução de Risco em Drawdown (Soft Guard)
    risk_pct = config.RISK_PER_TRADE_PCT
    if getattr(config, 'REDUCE_RISK_ON_DD', True) and daily_loss_pct > 0.012:
        risk_pct *= 0.5
        logger.info(f"🛡️ RISCO REDUZIDO (-50%): Drawdown atual de {daily_loss_pct:.2%}")
    
    try:
        session_name = (get_current_trading_session() or {}).get("name")
        mult_map = getattr(config, "SESSION_RISK_MULTIPLIERS", {})
        if isinstance(mult_map, dict) and session_name in mult_map:
            risk_pct *= float(mult_map.get(session_name, 1.0))
    except Exception:
        pass

    try:
        rm = float(risk_multiplier)
        if rm > 0:
            risk_pct *= rm
    except Exception:
        pass

    # -------------------------------------

    risk_per_trade_usd = equity * risk_pct

    pip_value_per_lot = get_tick_value(symbol)

    if pip_value_per_lot <= 0:
        logger.warning(f"⚠️ Valor do pip por lote é zero ou negativo para {symbol}. Usando volume p/ fallback.")
        return getattr(config, 'DEFAULT_LOT', 0.01)

    # ✅ CORREÇÃO: Cálculo baseado na distância REAL do SL (ATR * Multiplicador)
    sl_pips = atr_pips * sl_atr_mult
    if sl_pips <= 0:
        logger.warning(f"⚠️ SL Pips calculado é zero ({atr_pips} * {sl_atr_mult}). Usando volume p/ fallback.")
        return getattr(config, 'DEFAULT_LOT', 0.01)

    # Volume = (Risco em $) / (Distância SL em pips * Valor do Pip por Lote)
    # Ex: $1000 / (20 pips * $10/pip) = 5 lotes
    volume = risk_per_trade_usd / (sl_pips * pip_value_per_lot)

    # Limita volume ao mínimo/máximo permitido
    volume = max(config.MIN_VOLUME, min(config.MAX_VOLUME, volume))

    # ✅ SAFETY CAP: Proteção absoluta contra lotes gigantes
    if volume > 0.50:
        logger.critical(f"🚨 VOLUME SAFETY CAP: Lote calculado ({volume}) excedeu 0.50. Forçando 0.01.")
        return 0.01

    # Arredonda para o step do volume
    symbol_info = get_symbol_info(symbol)
    step = symbol_info.volume_step if symbol_info else 0.01

    try:
        vol_dec = Decimal(str(volume))
        step_dec = Decimal(str(step))
        units = (vol_dec / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        volume = float((units * step_dec))
    except Exception:
        volume = round(volume / step) * step

    return volume

# ===========================
# DYNAMIC LOT CALCULATION v5.0
# ===========================
def calculate_dynamic_lot(symbol: str, risk_percent: float, stop_loss_pips: float) -> float:
    """
    ✅ v5.0: Calcula tamanho do lote baseado estritamente no % de risco do Equity.
    Fórmula: (Equity * Risco%) / (Distância_SL_Pips * Valor_do_Pip_por_Lote)
    
    Args:
        symbol: Par de moedas
        risk_percent: Porcentagem de risco (ex: 0.01 = 1%)
        stop_loss_pips: Distância do Stop Loss em pips
        
    Returns:
        float: Tamanho do lote calculado
    """
    try:
        with mt5_lock:
            account_info = mt5.account_info()
        
        if not account_info:
            logger.error("❌ calculate_dynamic_lot: Não foi possível obter informações da conta.")
            return getattr(config, 'MIN_VOLUME', 0.01)
        
        equity = account_info.equity
        
        if equity <= 0 or stop_loss_pips <= 0 or risk_percent <= 0:
            logger.warning(f"⚠️ calculate_dynamic_lot: Parâmetros inválidos (Equity={equity}, SL_Pips={stop_loss_pips}, Risk={risk_percent})")
            return getattr(config, 'MIN_VOLUME', 0.01)
        
        # Calcula valor de risco em $
        risk_amount = equity * risk_percent
        
        # Obtém valor do pip por lote padrão
        pip_value_per_lot = get_tick_value(symbol)
        
        if pip_value_per_lot <= 0:
            logger.warning(f"⚠️ calculate_dynamic_lot: Pip value inválido para {symbol}")
            return getattr(config, 'MIN_VOLUME', 0.01)
        
        # Fórmula: Lote = Risco$ / (SL_Pips * Valor_Pip_por_Lote)
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Aplica limites de segurança
        min_vol = getattr(config, 'MIN_VOLUME', 0.01)
        max_vol = getattr(config, 'MAX_VOLUME', 0.50)
        lot_size = max(min_vol, min(max_vol, lot_size))
        
        # Arredonda para o step do volume
        symbol_info = get_symbol_info(symbol)
        if symbol_info:
            step = symbol_info.volume_step
            try:
                lot_dec = Decimal(str(lot_size))
                step_dec = Decimal(str(step))
                units = (lot_dec / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                lot_size = float((units * step_dec))
            except Exception:
                lot_size = round(lot_size / step) * step
        
        logger.debug(f"📊 calculate_dynamic_lot: {symbol} | Equity={equity:.2f} | Risk={risk_percent:.2%} | SL_Pips={stop_loss_pips:.1f} | Lot={lot_size:.2f}")
        
        return lot_size
        
    except Exception as e:
        logger.error(f"❌ calculate_dynamic_lot: Erro crítico - {e}")
        return getattr(config, 'MIN_VOLUME', 0.01)

# ===========================
# EMA 200 MACRO FILTER v5.0
# ===========================
def get_ema_200(symbol: str, timeframe: int = None) -> dict:
    """
    ✅ v5.0: Calcula EMA 200 para filtro de tendência macro.
    
    Args:
        symbol: Par de moedas
        timeframe: Timeframe MT5 (default: H1)
        
    Returns:
        dict: {ema_200, current_price, is_above_ema, trend_direction, error}
    """
    try:
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
        
        ema_period = getattr(config, 'EMA_200_PERIOD', 200)
        
        # Normaliza símbolo
        real_symbol = normalize_symbol(symbol)
        
        # Obtém dados históricos
        df = get_rates(real_symbol, timeframe, ema_period + 50)
        
        if df is None or df.empty or len(df) < ema_period:
            return {"error": True, "message": f"Dados insuficientes para EMA 200 de {symbol}"}
        
        # Calcula EMA 200
        df['ema_200'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        ema_200_value = df['ema_200'].iloc[-1]
        current_price = df['close'].iloc[-1]
        is_above_ema = current_price > ema_200_value
        
        # Determina direção da tendência
        if is_above_ema:
            trend_direction = "BULLISH"
        else:
            trend_direction = "BEARISH"
        
        return {
            "error": False,
            "ema_200": ema_200_value,
            "current_price": current_price,
            "is_above_ema": is_above_ema,
            "trend_direction": trend_direction,
            "distance_pips": abs(current_price - ema_200_value) / get_pip_size(symbol)
        }
        
    except Exception as e:
        logger.error(f"❌ get_ema_200: Erro ao calcular para {symbol} - {e}")
        return {"error": True, "message": str(e)}

# ===========================
# ROLLOVER PROTECTION v5.0
# ===========================
def is_rollover_period() -> tuple:
    """
    ✅ v5.0: Verifica se está no período de rollover bancário (16:55-18:05 NY).
    
    Returns:
        tuple: (is_blocked: bool, reason: str)
    """
    try:
        if not getattr(config, 'ENABLE_ROLLOVER_BLOCK', True):
            return False, "Rollover block disabled"
        
        # Obtém horário NY (UTC-5 / UTC-4 DST)
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        current_time_str = now_ny.strftime("%H:%M")
        
        rollover_start = getattr(config, 'ROLLOVER_BLOCK_START', "16:55")
        rollover_end = getattr(config, 'ROLLOVER_BLOCK_END', "18:05")
        
        # Verifica se está no período
        if rollover_start <= current_time_str <= rollover_end:
            return True, f"Período de Rollover Bancário ({rollover_start}-{rollover_end} NY)"
        
        return False, "Fora do período de rollover"
        
    except Exception as e:
        logger.error(f"❌ is_rollover_period: Erro - {e}")
        return False, f"Erro: {e}"

# ===========================
# TELEGRAM INTEGRATION
# ===========================
def get_telegram_credentials() -> tuple:
    try:
        bot_token = str(getattr(config, 'TELEGRAM_BOT_TOKEN', '')).strip()
        chat_id = str(getattr(config, 'TELEGRAM_CHAT_ID', '')).strip()
        if not bot_token or not chat_id:
            bot_token = str(getattr(config, 'TELEGRAM_BOT_TOKEN_OVERRIDE', bot_token)).strip()
            chat_id = str(getattr(config, 'TELEGRAM_CHAT_ID_OVERRIDE', chat_id)).strip()
        if not bot_token or not chat_id:
            try:
                creds_path = Path(getattr(config, 'TELEGRAM_CREDENTIALS_FILE', 'data/telegram.json'))
                if creds_path.exists():
                    data = json.loads(creds_path.read_text(encoding='utf-8'))
                    bot_token = str(data.get('bot_token', bot_token)).strip()
                    chat_id = str(data.get('chat_id', chat_id)).strip()
            except Exception:
                pass
        return (bot_token if bot_token else None, chat_id if chat_id else None)
    except Exception:
        return (None, None)
def send_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    """
    Envia mensagem via Telegram Bot.
    
    Args:
        message: Mensagem a ser enviada (suporta HTML)
        parse_mode: Modo de parse (HTML ou Markdown)
        
    Returns:
        bool: True se enviado com sucesso
    """
    try:
        import requests
        
        bot_token, chat_id = get_telegram_credentials()
        
        if not bot_token or not chat_id:
            logger.warning("⚠️ Telegram não configurado (TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID ausentes no config)")
            return False
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.debug(f"✅ Mensagem Telegram enviada com sucesso")
            return True
        else:
            logger.error(f"❌ Erro ao enviar Telegram: {response.status_code} - {response.text}")
            return False
            
    except ImportError:
        logger.error("❌ Biblioteca 'requests' não instalada. Execute: pip install requests")
        return False
    except Exception as e:
        logger.error(f"❌ send_telegram_message: Erro - {e}")
        return False

def send_telegram_alert(message: str, level: str = "INFO") -> bool:
    """
    Envia alerta via Telegram com formatação especial.
    
    Args:
        message: Mensagem de alerta
        level: Nível do alerta (INFO, WARNING, ERROR)
        
    Returns:
        bool: True se enviado com sucesso
    """
    try:
        # Emojis por nível
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "SUCCESS": "✅"
        }
        
        emoji = emoji_map.get(level.upper(), "📢")
        
        # Formata mensagem com header
        formatted_message = f"{emoji} <b>{level.upper()}</b>\n\n{message}"
        
        return send_telegram_message(formatted_message)
        
    except Exception as e:
        logger.error(f"❌ send_telegram_alert: Erro - {e}")
        return False

def send_telegram_document(file_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
    try:
        import requests

        bot_token, chat_id = get_telegram_credentials()

        if not bot_token or not chat_id:
            return False

        if not file_path or not os.path.exists(file_path):
            return False

        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            response = requests.post(url, data=data, files=files, timeout=30)

        return response.status_code == 200
    except Exception:
        return False

def set_telegram_chat_id(chat_id: str) -> None:
    try:
        creds_path = Path(getattr(config, 'TELEGRAM_CREDENTIALS_FILE', 'data/telegram.json'))
        creds_path.parent.mkdir(exist_ok=True)
        payload = {}
        if creds_path.exists():
            try:
                payload = json.loads(creds_path.read_text(encoding='utf-8'))
            except Exception:
                payload = {}
        payload['chat_id'] = str(chat_id)
        # Mantém token existente
        bot_token, _ = get_telegram_credentials()
        if bot_token:
            payload['bot_token'] = bot_token
        creds_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info(f"✅ Telegram chat_id salvo em {creds_path}")
    except Exception as e:
        logger.error(f"❌ Falha ao salvar chat_id do Telegram: {e}")

def send_telegram_message_to(chat_id: str, message: str, parse_mode: str = "HTML") -> bool:
    try:
        import requests
        bot_token, _ = get_telegram_credentials()
        if not bot_token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": str(chat_id), "text": message, "parse_mode": parse_mode, "disable_web_page_preview": True}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def export_bot_trades_csv(date_str: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        day = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None, {"error": "data inválida"}

    if not check_mt5_connection():
        if not ensure_mt5_connection():
            return None, {"error": "MT5 desconectado"}

    from_date = day.replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = from_date + timedelta(days=1)
    magic = getattr(config, 'MAGIC_NUMBER', 123456)

    with mt5_lock:
        deals = mt5.history_deals_get(from_date, to_date)

    rows = []
    total_profit = 0.0
    total_loss = 0.0
    wins = 0
    losses = 0

    if deals:
        for d in deals:
            try:
                if getattr(d, "magic", None) != magic:
                    continue
                if getattr(d, "type", None) not in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
                    continue
                entry = getattr(d, "entry", None)
                if entry is not None and entry != mt5.DEAL_ENTRY_OUT:
                    continue
                profit = float(getattr(d, "profit", 0.0))
                if profit > 0:
                    wins += 1
                    total_profit += profit
                elif profit < 0:
                    losses += 1
                    total_loss += profit
                ts = datetime.fromtimestamp(getattr(d, "time", 0))
                side = "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL"
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": getattr(d, "symbol", ""),
                    "side": side,
                    "volume": float(getattr(d, "volume", 0.0)),
                    "price": float(getattr(d, "price", 0.0)),
                    "profit": profit,
                    "commission": float(getattr(d, "commission", 0.0)),
                    "swap": float(getattr(d, "swap", 0.0)),
                    "comment": str(getattr(d, "comment", "")),
                    "order": str(getattr(d, "order", "")),
                    "position_id": str(getattr(d, "position_id", "")),
                })
            except Exception:
                continue

    out_dir = Path("analysis_logs")
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir / f"trades_{date_str}.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "symbol", "side", "volume", "price",
            "profit", "commission", "swap", "comment", "order", "position_id"
        ])
        writer.writeheader()
        writer.writerows(rows)

    total = wins + losses
    pnl = total_profit + total_loss
    pf = (total_profit / abs(total_loss)) if total_loss != 0 else 0.0
    wr = (wins / total * 100) if total > 0 else 0.0

    summary = {
        "date": date_str,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "profit_factor": pf,
        "pnl": pnl,
        "file": out_path
    }
    return out_path, summary
def export_bot_trades_txt(date_str: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None, {"error": "data inválida"}
    if not check_mt5_connection():
        if not ensure_mt5_connection():
            return None, {"error": "MT5 desconectado"}
    from_date = day.replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = from_date + timedelta(days=1)
    magic = getattr(config, 'MAGIC_NUMBER', 123456)
    with mt5_lock:
        deals = mt5.history_deals_get(from_date, to_date)
    lines = []
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    if deals:
        for d in deals:
            try:
                if getattr(d, "magic", None) != magic:
                    continue
                if getattr(d, "type", None) not in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
                    continue
                entry = getattr(d, "entry", None)
                if entry is not None and entry != mt5.DEAL_ENTRY_OUT:
                    continue
                ts = datetime.fromtimestamp(getattr(d, "time", 0)).strftime("%Y-%m-%d %H:%M:%S")
                sym = str(getattr(d, "symbol", ""))
                side = "BUY" if getattr(d, "type", None) == mt5.DEAL_TYPE_BUY else "SELL"
                vol = float(getattr(d, "volume", 0.0))
                price = float(getattr(d, "price", 0.0))
                profit = float(getattr(d, "profit", 0.0))
                commission = float(getattr(d, "commission", 0.0))
                swap = float(getattr(d, "swap", 0.0))
                order_id = str(getattr(d, "order", ""))
                position_id = str(getattr(d, "position_id", ""))
                comment = str(getattr(d, "comment", ""))
                if profit > 0:
                    wins += 1
                    total_profit += profit
                elif profit < 0:
                    losses += 1
                    total_loss += profit
                line = (
                    f"{ts} | {sym} | {side} {vol:.2f} | "
                    f"price={price:.5f} | profit={profit:+.2f} | "
                    f"commission={commission:.2f} | swap={swap:.2f} | "
                    f"order={order_id} | position_id={position_id} | comment={comment}"
                )
                lines.append(line)
            except Exception:
                continue
    out_dir = Path("analysis_logs")
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir / f"trades_{date_str}.txt")
    header = [
        f"Trades do Dia ({date_str})",
        "-" * 64
    ]
    summary_lines = []
    total = wins + losses
    pnl = total_profit + total_loss
    pf = (total_profit / abs(total_loss)) if total_loss != 0 else 0.0
    wr = (wins / total * 100) if total > 0 else 0.0
    summary_lines.append(f"Total: {total} | Wins: {wins} | Losses: {losses}")
    summary_lines.append(f"WR: {wr:.1f}% | PF: {pf:.2f} | PnL: ${pnl:+.2f}")
    summary_lines.append("-" * 64)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for l in header:
                f.write(l + "\n")
            for l in summary_lines:
                f.write(l + "\n")
            for l in lines:
                f.write(l + "\n")
    except Exception:
        return None, {"error": "falha ao salvar arquivo"}
    summary = {
        "date": date_str,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "profit_factor": pf,
        "pnl": pnl,
        "file": out_path
    }
    return out_path, summary
def _get_db_path() -> str:
    base = Path(getattr(config, "DATA_DIR", "data"))
    base.mkdir(exist_ok=True)
    return str(base / "trades.db")

def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_trade_db() -> None:
    conn = _get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS trades ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "order_id INTEGER,"
        "ticket INTEGER,"
        "symbol TEXT,"
        "side TEXT,"
        "volume REAL,"
        "open_time TEXT,"
        "close_time TEXT,"
        "open_price REAL,"
        "close_price REAL,"
        "sl REAL,"
        "tp REAL,"
        "commission REAL,"
        "swap REAL,"
        "profit REAL,"
        "magic INTEGER,"
        "comment TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS open_positions ("
        "order_id INTEGER,"
        "ticket INTEGER,"
        "symbol TEXT,"
        "side TEXT,"
        "volume REAL,"
        "open_time TEXT,"
        "open_price REAL,"
        "sl REAL,"
        "tp REAL,"
        "magic INTEGER,"
        "comment TEXT)"
    )
    conn.commit()
    conn.close()

def record_order_open(symbol: str, side: str, volume: float, entry_price: float, sl: float, tp: float, order_id: int, comment: str = "") -> None:
    try:
        init_trade_db()
        conn = _get_db_conn()
        cur = conn.cursor()
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO open_positions (order_id, ticket, symbol, side, volume, open_time, open_price, sl, tp, magic, comment) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(order_id) if order_id else None,
                None,
                str(symbol or ""),
                str(side or ""),
                float(volume or 0.0),
                now_ts,
                float(entry_price or 0.0),
                float(sl or 0.0),
                float(tp or 0.0),
                int(getattr(config, "MAGIC_NUMBER", 123456)),
                str(comment or "")
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"erro record_order_open: {e}")

def record_trade_close(ticket: int, symbol: str, side: str, volume: float, open_time: datetime, close_time: datetime, open_price: float, close_price: float, sl: float, tp: float, profit: float, commission: float, swap: float, magic: int, comment: str = "") -> None:
    try:
        init_trade_db()
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO trades (order_id, ticket, symbol, side, volume, open_time, close_time, open_price, close_price, sl, tp, commission, swap, profit, magic, comment) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                None,
                int(ticket) if ticket else None,
                str(symbol or ""),
                str(side or ""),
                float(volume or 0.0),
                (open_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(open_time, datetime) else str(open_time or "")),
                (close_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(close_time, datetime) else str(close_time or "")),
                float(open_price or 0.0),
                float(close_price or 0.0),
                float(sl or 0.0),
                float(tp or 0.0),
                float(commission or 0.0),
                float(swap or 0.0),
                float(profit or 0.0),
                int(magic or getattr(config, "MAGIC_NUMBER", 123456)),
                str(comment or "")
            )
        )
        try:
            cur.execute("DELETE FROM open_positions WHERE ticket = ? OR (symbol = ? AND ABS(open_price - ?) < 1e-6)", (int(ticket or 0), str(symbol or ""), float(open_price or 0.0)))
        except Exception:
            pass
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"erro record_trade_close: {e}")

def sync_mt5_trades_to_db() -> Dict[str, Any]:
    summary = {"inserted": 0, "checked": 0}
    try:
        init_trade_db()
        conn = _get_db_conn()
        cur = conn.cursor()
        try:
            if not check_mt5_connection():
                ensure_mt5_connection()
        except Exception:
            pass
        day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        with mt5_lock:
            deals = mt5.history_deals_get(day, datetime.now())
        magic = getattr(config, "MAGIC_NUMBER", 123456)
        if deals:
            for d in deals:
                summary["checked"] += 1
                try:
                    if getattr(d, "magic", None) != magic:
                        continue
                    if getattr(d, "entry", None) != mt5.DEAL_ENTRY_OUT:
                        continue
                    ts = datetime.fromtimestamp(getattr(d, "time", 0))
                    symbol = str(getattr(d, "symbol", ""))
                    side = "BUY" if getattr(d, "type", None) == mt5.DEAL_TYPE_BUY else "SELL"
                    volume = float(getattr(d, "volume", 0.0))
                    price = float(getattr(d, "price", 0.0))
                    profit = float(getattr(d, "profit", 0.0))
                    commission = float(getattr(d, "commission", 0.0))
                    swap = float(getattr(d, "swap", 0.0))
                    order_id = int(getattr(d, "order", 0))
                    position_id = int(getattr(d, "position_id", 0))
                    cur.execute("SELECT COUNT(1) FROM trades WHERE ticket = ? AND close_time = ?", (position_id, ts.strftime("%Y-%m-%d %H:%M:%S")))
                    exists = cur.fetchone()[0] > 0
                    if exists:
                        continue
                    cur.execute(
                        "INSERT INTO trades (order_id, ticket, symbol, side, volume, open_time, close_time, open_price, close_price, sl, tp, commission, swap, profit, magic, comment) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            order_id,
                            position_id,
                            symbol,
                            side,
                            volume,
                            "",
                            ts.strftime("%Y-%m-%d %H:%M:%S"),
                            0.0,
                            price,
                            0.0,
                            0.0,
                            commission,
                            swap,
                            profit,
                            magic,
                            str(getattr(d, "comment", ""))
                        )
                    )
                    summary["inserted"] += 1
                except Exception:
                    continue
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"erro sync_mt5_trades_to_db: {e}")
    return summary

# ===========================
# RSI REVERSAL DETECTION v5.0
# ===========================
def detect_rsi_reversal(symbol: str, lookback: int = 5) -> dict:
    """
    ✅ v5.0: Detecta se RSI está "virando" (reversão de sobrecompra/sobrevenda).
    
    Regras:
    - BUY Reversal: RSI estava < 30 e agora está > 30 (saindo de sobrevenda)
    - SELL Reversal: RSI estava > 70 e agora está < 70 (saindo de sobrecompra)
    
    Args:
        symbol: Par de moedas
        lookback: Número de candles para verificar o histórico do RSI
        
    Returns:
        dict: {is_buy_reversal, is_sell_reversal, rsi_current, rsi_previous, error}
    """
    try:
        real_symbol = normalize_symbol(symbol)
        rsi_period = getattr(config, 'RSI_PERIOD', 14)
        
        # Precisa de dados extras para calcular RSI corretamente
        df = get_rates(real_symbol, mt5.TIMEFRAME_M15, rsi_period + lookback + 10)
        
        if df is None or df.empty or len(df) < rsi_period + lookback:
            return {"error": True, "message": f"Dados insuficientes para RSI reversal de {symbol}"}
        
        # Calcula RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        rsi_current = df['rsi'].iloc[-1]
        rsi_previous = df['rsi'].iloc[-2]
        
        # Busca nos últimos 'lookback' candles se estava em zona extrema
        rsi_lookback = df['rsi'].iloc[-(lookback+1):-1].values
        
        was_oversold = any(r < 30 for r in rsi_lookback)
        was_overbought = any(r > 70 for r in rsi_lookback)
        
        # Detecta reversões
        is_buy_reversal = was_oversold and rsi_current > 30 and rsi_previous <= 30
        is_sell_reversal = was_overbought and rsi_current < 70 and rsi_previous >= 70
        
        return {
            "error": False,
            "is_buy_reversal": is_buy_reversal,
            "is_sell_reversal": is_sell_reversal,
            "rsi_current": rsi_current,
            "rsi_previous": rsi_previous,
            "was_oversold": was_oversold,
            "was_overbought": was_overbought
        }
        
    except Exception as e:
        logger.error(f"❌ detect_rsi_reversal: Erro para {symbol} - {e}")
        return {"error": True, "message": str(e)}

# ===========================
# CORRELATION FILTER v5.1 (Land Trading)
# ===========================
def check_correlation(symbol: str) -> tuple:
    """
    ✅ v5.1: Verifica se há trades abertos em pares correlacionados.
    
    Args:
        symbol: Par a ser verificado
        
    Returns:
        tuple: (is_blocked: bool, reason: str, correlated_symbol: str)
    """
    try:
        correlations = getattr(config, 'SYMBOL_CORRELATIONS', {})
        max_correlation = getattr(config, 'CORRELATION_MAX', 0.80)
        
        if symbol not in correlations:
            return False, "Sem correlações configuradas", ""
        
        symbol_corrs = correlations[symbol]
        
        # Obtém posições abertas
        with mt5_lock:
            positions = mt5.positions_get()
        
        if not positions:
            return False, "Sem posições abertas", ""
        
        magic_number = getattr(config, 'MAGIC_NUMBER', 123456)
        
        for pos in positions:
            if pos.magic != magic_number:
                continue
            
            pos_symbol = pos.symbol
            
            # Verifica se o símbolo da posição está correlacionado
            if pos_symbol in symbol_corrs:
                corr_value = abs(symbol_corrs[pos_symbol])
                if corr_value >= max_correlation:
                    reason = f"Correlação alta com {pos_symbol} ({corr_value:.0%})"
                    logger.info(f"🔗 {symbol}: Bloqueado - {reason}")
                    return True, reason, pos_symbol
        
        return False, "Sem conflitos de correlação", ""
        
    except Exception as e:
        logger.error(f"❌ check_correlation: Erro para {symbol} - {e}")
        return False, f"Erro: {e}", ""

# ===========================
# VOLATILITY FILTER v5.1 (Land Trading)
# ===========================
def is_volatility_ok(symbol: str, indicators: dict = None) -> tuple:
    """
    ✅ v5.1: Verifica se ATR está acima do mínimo para operar.
    
    Args:
        symbol: Par de moedas
        indicators: Dict de indicadores (opcional, se não fornecido calcula)
        
    Returns:
        tuple: (is_ok: bool, reason: str, atr_value: float)
    """
    try:
        min_atr = getattr(config, 'VOLATILITY_FILTER_MIN_ATR', 0.0005)
        
        # Usa indicadores fornecidos ou calcula
        if indicators and 'atr' in indicators:
            atr_value = indicators.get('atr', 0)
        else:
            ind = get_indicators_forex(symbol)
            if ind.get('error'):
                return True, "Não foi possível calcular ATR", 0
            atr_value = ind.get('atr', 0)
        
        if atr_value < min_atr:
            reason = f"ATR muito baixo ({atr_value:.6f} < {min_atr})"
            return False, reason, atr_value
        
        return True, f"ATR OK ({atr_value:.6f})", atr_value
        
    except Exception as e:
        logger.error(f"❌ is_volatility_ok: Erro para {symbol} - {e}")
        return True, f"Erro: {e}", 0

# ===========================
# CANDLE CONFIRMATION v5.1 (Land Trading)
# ===========================
def is_candle_confirmed(symbol: str, signal: str) -> tuple:
    """
    ✅ v5.1: Verifica se o último candle confirma a direção do sinal.
    
    Args:
        symbol: Par de moedas
        signal: "BUY" ou "SELL"
        
    Returns:
        tuple: (is_confirmed: bool, reason: str)
    """
    try:
        if not getattr(config, 'CANDLE_CONFIRMATION_REQUIRED', True):
            return True, "Candle confirmation disabled"
        
        real_symbol = normalize_symbol(symbol)
        df = get_rates(real_symbol, mt5.TIMEFRAME_M15, 3)
        
        if df is None or len(df) < 2:
            return True, "Dados insuficientes para confirmação"
        
        # ✅ v5.3: Com start_pos=1 em get_rates, iloc[-1] é o candle fechado mais recente
        last_candle = df.iloc[-1]
        candle_time = last_candle['time']
        
        # ✅ v5.3: Invalidação por Timestamp (Máximo 2 timeframes de atraso)
        now = datetime.now() 
        # Candle time do MT5 geralmente é ingênuo (naive).
        # Se for naive, comparamos com datetime.now() (também naive local).
        if (now - candle_time).total_seconds() > 1800: # 30 min (2 candles de M15)
            return False, f"Sinal obsoleto: Candle de {candle_time.strftime('%H:%M')}"

        candle_close = last_candle['close']
        candle_open = last_candle['open']
        
        is_bullish = candle_close > candle_open
        is_bearish = candle_close < candle_open
        
        if signal == "BUY" and not is_bullish:
            return False, f"Candle não confirmou BUY (O:{candle_open:.5f} C:{candle_close:.5f})"
        
        if signal == "SELL" and not is_bearish:
            return False, f"Candle não confirmou SELL (O:{candle_open:.5f} C:{candle_close:.5f})"
        
        return True, f"Candle confirmado ({signal})"
        
    except Exception as e:
        logger.error(f"❌ is_candle_confirmed: Erro para {symbol} - {e}")
        return True, f"Erro: {e}"

# ===========================
# MULTI-TIMEFRAME TREND v5.2
# ===========================
def get_multi_timeframe_trend(symbol: str, signal: str) -> tuple:
    """
    ✅ v5.3: Verifica se macro timeframe (H4) confirma a direção do sinal.
    MODIFICADO: Retorna penalidade de score em vez de veto absoluto.
    
    Args:
        symbol: Par de moedas
        signal: "BUY" ou "SELL"
        
    Returns:
        tuple: (score_penalty: int, reason: str, h4_trend: str)
              score_penalty = 0 se alinhado, -15 se divergente
    """
    try:
        if not getattr(config, 'ENABLE_MULTI_TIMEFRAME', True):
            return 0, "Multi-timeframe disabled", "N/A"
        
        # Determina timeframe de confirmação
        tf_str = getattr(config, 'MULTI_TF_CONFIRMATION', "H4")
        tf_map = {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
        timeframe = tf_map.get(tf_str, mt5.TIMEFRAME_H4)
        
        ema_period = getattr(config, 'MULTI_TF_EMA_PERIOD', 200)
        
        # Obtém dados H4
        real_symbol = normalize_symbol(symbol)
        df = get_rates(real_symbol, timeframe, ema_period + 50)
        
        if df is None or df.empty or len(df) < ema_period:
            return 0, f"Dados {tf_str} insuficientes", "UNKNOWN"
        
        # Calcula EMA no timeframe alto
        df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        current_price = df['close'].iloc[-1]
        ema_value = df['ema'].iloc[-1]
        
        # Determina tendência macro
        if current_price > ema_value:
            h4_trend = "BULLISH"
        else:
            h4_trend = "BEARISH"
        
        # ✅ v5.3: SCORE PENALTY em vez de veto
        if signal == "BUY" and h4_trend != "BULLISH":
            return -8, f"{tf_str} divergente: BEARISH (Penalidade -8)", h4_trend
        
        if signal == "SELL" and h4_trend != "BEARISH":
            return -8, f"{tf_str} divergente: BULLISH (Penalidade -8)", h4_trend
        
        return 0, f"{tf_str} alinhado ({h4_trend})", h4_trend
        
    except Exception as e:
        logger.error(f"❌ get_multi_timeframe_trend: Erro para {symbol} - {e}")
        return 0, f"Erro: {e}", "ERROR"

# ===========================
# ROLLING WIN RATE v5.2
# ===========================
def calculate_rolling_win_rate(symbol: str = None, window: int = None) -> tuple:
    """
    ✅ v5.2: Calcula win rate dos últimos N trades via MT5 history.
    
    Args:
        symbol: Par específico (None = todos os símbolos do bot)
        window: Número de trades a considerar (default: config.KILL_SWITCH_TRADES)
        
    Returns:
        tuple: (win_rate: float, total_trades: int, wins: int)
    """
    try:
        if window is None:
            window = getattr(config, 'KILL_SWITCH_TRADES', 10)
        
        magic_number = getattr(config, 'MAGIC_NUMBER', 123456)
        
        # Obtém histórico de deals do MT5
        from datetime import timedelta
        from_date = datetime.now() - timedelta(days=30)
        to_date = datetime.now() + timedelta(days=1)
        
        with mt5_lock:
            deals = mt5.history_deals_get(from_date, to_date)
        
        if not deals or len(deals) == 0:
            return 0.5, 0, 0  # Default 50% se sem histórico
        
        # Filtra deals do bot
        bot_deals = []
        for deal in deals:
            if deal.magic != magic_number:
                continue
            if deal.type not in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
                continue
            if deal.profit == 0:  # Ignora entradas (só fechamentos)
                continue
            if symbol and deal.symbol != symbol:
                continue
            bot_deals.append(deal)
        
        # Ordena por tempo (mais recente primeiro) e pega os últimos N
        bot_deals.sort(key=lambda x: x.time, reverse=True)
        recent_deals = bot_deals[:window]
        
        if len(recent_deals) == 0:
            return 0.5, 0, 0
        
        # Conta wins
        wins = sum(1 for d in recent_deals if d.profit > 0)
        total = len(recent_deals)
        win_rate = wins / total if total > 0 else 0.5
        
        logger.debug(f"📊 Rolling WR: {win_rate:.1%} ({wins}/{total})")
        
        return win_rate, total, wins
        
    except Exception as e:
        logger.error(f"❌ calculate_rolling_win_rate: Erro - {e}")
        return 0.5, 0, 0

# ===========================
# SL/TP CALCULATION (v4.2 - PATCH 1)
# ===========================
def calculate_dynamic_levels(
    symbol: str,
    current_price: float,
    indicators: dict,
    sl_atr_mult: float = None,   # ✅ NOVO: Multiplicador SL otimizado
    tp_atr_mult: float = None,    # ✅ NOVO: Multiplicador TP otimizado
    signal: Optional[str] = None
) -> Tuple[float, float]:
    """
    ✅ v4.2: Calcula SL e TP dinamicamente usando ATR e multiplicadores otimizados.
    """
    atr_value = indicators.get("atr", 0)
    if atr_value <= 0:
        logger.warning(f"⚠️ ATR é zero para {symbol}. Retornando SL/TP 0.0.")
        return 0.0, 0.0

    # Usa multiplicadores otimizados se fornecidos, senão usa config padrão
    sl_multiplier = sl_atr_mult if sl_atr_mult is not None else getattr(config, 'DEFAULT_STOP_LOSS_ATR_MULTIPLIER', 2.0) # ✅ CORREÇÃO: STOP_LOSS_ATR_MULTIPLIER deve estar em config
    tp_multiplier = tp_atr_mult if tp_atr_mult is not None else getattr(config, 'DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER', 3.0) # ✅ CORREÇÃO: TAKE_PROFIT_ATR_MULTIPLIER deve estar em config

    sl_distance = atr_value * sl_multiplier
    regime = check_market_regime(indicators)
    tp_mult_adj = tp_multiplier
    if regime == "RANGING":
        tp_mult_adj = tp_multiplier * 0.8
    tp_distance = atr_value * tp_mult_adj

    # Garante que SL/TP não sejam muito pequenos
    min_sl_pips = config.MIN_STOP_LOSS_PIPS # ✅ CORREÇÃO: MIN_STOP_LOSS_PIPS deve estar em config
    pip_size = get_pip_size(symbol)
    min_sl_distance = min_sl_pips * pip_size

    if sl_distance < min_sl_distance:
        sl_distance = min_sl_distance
        logger.debug(f"Ajustando SL para {symbol} para o mínimo: {min_sl_pips} pips")

    side = signal
    if side is None:
        ema_trend = indicators.get("ema_trend")
        if ema_trend == "UP":
            side = "BUY"
        elif ema_trend == "DOWN":
            side = "SELL"

    if side == "BUY":
        sl = current_price - sl_distance
        tp = current_price + tp_distance
    elif side == "SELL":
        sl = current_price + sl_distance
        tp = current_price - tp_distance
    else:
        logger.warning(f"⚠️ Tendência EMA desconhecida para {symbol}. Não foi possível calcular SL/TP.")
        return 0.0, 0.0

    # Arredonda ao número correto de dígitos
    info = get_symbol_info(symbol)
    if info:
        digits = info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)
    else:
        logger.warning(f"⚠️ Não foi possível obter info para {symbol} para arredondar SL/TP. Usando 5 casas decimais.")
        sl = round(sl, 5) # Fallback
        tp = round(tp, 5) # Fallback

    return sl, tp

def check_market_regime(indicators: dict) -> str:
    try:
        adx = float(indicators.get("adx", 0) or 0.0)
        bb_width = float(indicators.get("bb_width", 0) or 0.0)
        adx_min = float(getattr(config, 'ADX_MIN_STRENGTH', 15))
        bb_thresh = float(getattr(config, 'BB_SQUEEZE_THRESHOLD', 0.015))
        if adx >= adx_min and bb_width >= bb_thresh:
            return "TRENDING"
        else:
            return "RANGING"
    except Exception:
        return "RANGING"

def calculate_signal_score(
    indicators: dict,
    ema_short: int = 20,
    ema_long: int = 50,
    rsi_low: int = 30,
    rsi_high: int = 70,
    adx_threshold: int = 25,
    bb_squeeze_threshold: float = 0.015
) -> Tuple[float, dict]:
    score = 50.0
    details = {}
    rsi = indicators.get("rsi", 50)
    adx = indicators.get("adx", 0)
    ema_trend = indicators.get("ema_trend", "N/A")
    bb_width = indicators.get("bb_width", 0)
    close = indicators.get("close", 0)
    open_price = indicators.get("open", 0)
    rsi_bull_zone = rsi_low + 10
    rsi_bear_zone = rsi_high - 10
    if ema_trend in ["UP", "DOWN"]:
        score += 15
        details["trend_bonus"] = 15
    if ema_trend == "UP":
        if rsi_bull_zone <= rsi <= rsi_high:
            score += 20
            details["rsi_bonus"] = 20
        elif rsi_low <= rsi < rsi_bull_zone:
            score += 10
            details["rsi_bonus"] = 10
    elif ema_trend == "DOWN":
        if rsi_low <= rsi <= rsi_bear_zone:
            score += 20
            details["rsi_bonus"] = 20
        elif rsi_bear_zone < rsi <= rsi_high:
            score += 10
            details["rsi_bonus"] = 10
    else:
        if rsi < rsi_low or rsi > rsi_high:
            score += 20
            details["rsi_reversal_bonus"] = 20
    if adx > adx_threshold:
        score += 15
        details["adx_bonus"] = 15
    elif adx > (adx_threshold * 0.7):
        score += 8
        details["adx_bonus"] = 8
    if bb_width > bb_squeeze_threshold:
        score += 10
        details["volatility_bonus"] = 10
    if close > open_price and ema_trend == "UP":
        score += 10
        details["candle_bonus"] = 10
    elif close < open_price and ema_trend == "DOWN":
        score += 10
        details["candle_bonus"] = 10
    total_penalty = 0.0
    ema_penalty = indicators.get("ema_penalty", 0)
    if ema_penalty > 0:
        penalty = min(float(ema_penalty), 12.0)
        score -= penalty
        total_penalty += penalty
        details["ema_penalty"] = -penalty
    mtf_penalty = indicators.get("mtf_penalty", 0)
    if mtf_penalty > 0:
        penalty = min(float(mtf_penalty), 8.0)
        score -= penalty
        total_penalty += penalty
        details["mtf_penalty"] = -penalty
    macd_hist = indicators.get("macd_hist", 0)
    if ema_trend == "UP" and macd_hist < 0:
        score -= 5
        total_penalty += 5
        details["macd_penalty"] = -5
    elif ema_trend == "DOWN" and macd_hist > 0:
        score -= 5
        total_penalty += 5
        details["macd_penalty"] = -5
    if bb_width < bb_squeeze_threshold:
        score -= 8
        total_penalty += 8
        details["squeeze_penalty"] = -8
    if total_penalty > 30:
        excess = total_penalty - 30
        score += excess
        details["penalty_cap_applied"] = excess
    score = max(20.0, min(100.0, score))
    try:
        bonus_sum = sum(v for k, v in details.items() if "bonus" in k)
    except Exception:
        bonus_sum = 0
    logger.debug(f"Score Breakdown: Base=50 | Bônus={bonus_sum} | Penalidades=-{total_penalty} | Final={score:.1f}")
    return score, details

# ===========================
# TRADE EXECUTION HELPERS
# ===========================

def modify_position_sl_tp(ticket, sl, tp):
    """
    ✅ v4.3: Modifica o Stop Loss e o Take Profit de uma posição existente.
    Inclui verificação de Stop Levels, Normalização e Silenciamento de Erros.
    """
    try:
        # 1. Obter informações da posição
        with mt5_lock:
            positions = mt5.positions_get(ticket=ticket)
        
        if not positions or len(positions) == 0:
            logger.error(f"❌ Não foi possível encontrar a posição {ticket} para modificação.")
            return False
            
        pos = positions[0]
        symbol = pos.symbol
        
        # 2. Obter informações do símbolo e tick atual
        info = get_symbol_info(symbol)
        if not info:
            return False
            
        with mt5_lock:
            tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
            
        digits = info.digits
        point = info.point
        stops_level = info.trade_stops_level * point
        freeze_level = info.trade_freeze_level * point
        # A distância mínima deve considerar stops_level e freeze_level
        min_dist = max(stops_level, freeze_level)
        
        # Garante uma margem mínima se a corretora retornar 0 (comum em algumas mas perigoso)
        if min_dist == 0:
            min_dist = 2 * point
            
        bid = tick.bid
        ask = tick.ask

        # 3. Normalização de Preços e Margem de Segurança (Buffer)
        # ✅ NOVO: Buffer adicional para Índices para evitar erro 10016 por milissegundos
        symbol_upper = symbol.upper()
        if any(idx in symbol_upper for idx in ["US30", "GER40", "DE40", "UK100", "US500", "NAS100", "USTEC"]):
            min_dist += 2 * point # Buffer cirúrgico de 2 pontos extras

        sl = round(sl, digits)
        tp = round(tp, digits)
        
        # 4. Trava Lógica de Direção e Distância (Evita erro 10016)
        is_buy = (pos.type == mt5.POSITION_TYPE_BUY)
        current_sl = pos.sl
        invalid_reason = None
        
        if is_buy:
            # Regras para COMPRA:
            # - Novo SL deve ser maior que o SL atual (Trava de Direção Trailing)
            # - Novo SL deve ser menor que (Bid - stops_level)
            if sl != 0:
                if sl <= current_sl and current_sl != 0:
                    return False # Silencioso: ignoramos movimentos para trás ou iguais
                if sl > (bid - min_dist):
                    invalid_reason = f"SL muito próximo do Bid ({sl} > {bid - min_dist:.{digits}f})"
        else:
            # Regras para VENDA:
            # - Novo SL deve ser menor que o SL atual (Trava de Direção Trailing)
            # - Novo SL deve ser maior que (Ask + stops_level)
            if sl != 0:
                if sl >= current_sl and current_sl != 0:
                    return False # Silencioso: ignoramos movimentos para trás ou iguais
                if sl < (ask + min_dist):
                    invalid_reason = f"SL muito próximo do Ask ({sl} < {ask + min_dist:.{digits}f})"

        # Se houver violação de distância, logamos com throttling (1x por minuto)
        if invalid_reason:
            current_time = time.time()
            cache_key = f"invalid_{symbol}_{ticket}"
            last_log = ERROR_CACHE.get(cache_key, {"time": 0})["time"]
            
            if current_time - last_log > 60:
                logger.warning(f"⚠️ {symbol}: Modificação ignorada | {invalid_reason}")
                ERROR_CACHE[cache_key] = {"time": current_time}
            return False

        # 5. Envio da Ordem
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        
        with mt5_lock:
            result = mt5.order_send(request)
            
        if result is None:
            logger.error(f"❌ Falha crítica ao enviar modificação para {ticket}: result is None")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Silenciamento para Erro 10016 (Invalid Stops)
            if result.retcode == 10016:
                current_time = time.time()
                cache_key = f"err10016_{symbol}"
                last_info = ERROR_CACHE.get(cache_key, {"time": 0, "price": 0})
                
                ref_price = bid if is_buy else ask
                price_changed = abs(ref_price - last_info["price"]) > (min_dist * 2)
                
                if current_time - last_info["time"] > 60 or price_changed:
                    logger.error(f"❌ Erro 10016 ({symbol}): {result.comment} | SL Tentado: {sl} | MinDist: {min_dist:.{digits}f}")
                    ERROR_CACHE[cache_key] = {"time": current_time, "price": ref_price}
            else:
                logger.error(f"❌ Falha ao modificar {symbol} #{ticket}: {result.comment} ({result.retcode})")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Erro crítico em modify_position_sl_tp: {e}", exc_info=True)
        return False

def close_all_positions():
    """
    ✅ v4.2: Fecha todas as posições abertas pelo bot (mesmo Magic Number).
    Utilizado principalmente pelo KILL SWITCH ou shutdown.
    """
    try:
        positions = mt5.positions_get()
        if not positions:
            logger.info("ℹ️ Nenhuma posição aberta para fechar.")
            return

        magic_number = getattr(config, 'MAGIC_NUMBER', 123456)
        closed_count = 0
        
        for pos in positions:
            if pos.magic == magic_number:
                symbol_info = mt5.symbol_info(pos.symbol)
                if not symbol_info:
                    continue
                    
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue

                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": getattr(config, 'DEVIATION', 20),
                    "magic": magic_number,
                    "comment": "Bot Shutdown - Close All",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result is None:
                    logger.error(f"❌ Erro ao fechar {pos.ticket}: result is None")
                    continue
                    
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"❌ Falha ao fechar posição {pos.ticket}: {result.comment} (retcode: {result.retcode})")
                else:
                    logger.info(f"✅ Posição {pos.ticket} fechada com sucesso.")
                    closed_count += 1
        
        if closed_count > 0:
            logger.info(f"🛑 Fechamento em massa concluído: {closed_count} posições encerradas.")
            
    except Exception as e:
        logger.error(f"❌ Erro crítico em close_all_positions: {e}", exc_info=True)

# ===========================
# INSTITUTIONAL RISK v5.3
# ===========================

def get_risk_manager_state() -> dict:
    """Carrega o estado do gerenciador de risco do arquivo JSON."""
    file_path = getattr(config, 'RISK_MANAGER_FILE', 'data/risk_manager.json')
    default_state = {
        "weekly_start_balance": 0.0,
        "weekly_drawdown_max": 0.0,
        "last_update_week": 0
    }
    
    if not os.path.exists(file_path):
        return default_state
        
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Erro ao ler {file_path}: {e}")
        return default_state

def save_risk_manager_state(state: dict):
    """Salva o estado do gerenciador de risco no arquivo JSON."""
    file_path = getattr(config, 'RISK_MANAGER_FILE', 'data/risk_manager.json')
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logger.error(f"❌ Erro ao salvar {file_path}: {e}")

def check_institutional_risk() -> Tuple[bool, str]:
    """
    ✅ v5.3: Validação de risco institucional (Hard Stop).
    Retorna (is_ok, reason)
    """
    try:
        with mt5_lock:
            acc = mt5.account_info()
        if not acc:
            return False, "Erro ao obter info da conta"

        # 1. Kill Switch por Equity (Drawdown Total)
        total_dd = (acc.balance - acc.equity) / acc.balance if acc.balance > 0 else 0
        if total_dd > getattr(config, 'INSTITUTIONAL_KILL_SWITCH_PCT', 0.10):
            return False, f"🔴 KILL SWITCH: Drawdown Total {total_dd:.1%} > {config.INSTITUTIONAL_KILL_SWITCH_PCT:.1%}"

        # 2. Drawdown Diário (Circuit Breaker)
        # Nota: O bot já deve ter o balance inicial do dia salvo em algum lugar.
        # Se não tiver, usamos o balance atual vs equity como aproximação de DD flutuante.
        # Mas para Hard Stop diário, é melhor ter o balance do início do dia.
        daily_dd = (acc.balance - acc.equity) / acc.balance if acc.balance > 0 else 0
        if daily_dd > getattr(config, 'MAX_DAILY_DRAWDOWN_PCT', 0.02):
            return False, f"🔴 DAILY STOP: Drawdown {daily_dd:.1%} > {config.MAX_DAILY_DRAWDOWN_PCT:.1%}"

        # 3. Drawdown Semanal (Persistente)
        state = get_risk_manager_state()
        curr_week = datetime.now().isocalendar()[1]
        
        if state['last_update_week'] != curr_week:
            # Reset semanal
            state['weekly_start_balance'] = acc.balance
            state['last_update_week'] = curr_week
            save_risk_manager_state(state)
            
        weekly_start = state['weekly_start_balance']
        if weekly_start > 0:
            weekly_dd = (weekly_start - acc.equity) / weekly_start
            if weekly_dd > getattr(config, 'MAX_WEEKLY_DRAWDOWN_PCT', 0.05):
                return False, f"🔴 WEEKLY STOP: Drawdown {weekly_dd:.1%} > {config.MAX_WEEKLY_DRAWDOWN_PCT:.1%}"

        # 4. Bloqueio de Alavancagem Implícita
        # Alavancagem real = Exposição Total / Equity
        # Para simplificar, verificamos a alavancagem da conta configurada
        if acc.leverage > getattr(config, 'MAX_ACCOUNT_LEVERAGE', 30):
             # Apenas loga aviso, não bloqueia se já estiver operando, 
             # mas podemos bloquear novas entradas
             pass

        return True, "Risk OK"

    except Exception as e:
        logger.error(f"❌ Erro em check_institutional_risk: {e}")
        return True, "Erro na validação (Ignorado)"

def check_currency_exposure(positions) -> Tuple[bool, str]:
    """
    ✅ v5.3: Controle básico de exposição por moeda (USD, JPY, EUR).
    """
    if not positions:
        return True, ""
        
    exposure = {"USD": 0.0, "JPY": 0.0, "EUR": 0.0}
    total_equity = mt5.account_info().equity if mt5.account_info() else 1.0
    
    for pos in positions:
        symbol = pos.symbol.upper()
        # Estimativa simplificada de valor nocional em USD
        # Volume * Preço 
        # Para Forex, volume é em unidades da moeda base.
        nocional = pos.volume * 100000 # Lote padrão
        
        for curr in exposure.keys():
            if curr in symbol:
                exposure[curr] += nocional
                
    max_exp_pct = getattr(config, 'MAX_CURRENCY_EXPOSURE_PCT', 0.03)
    max_exp_val = total_equity * max_exp_pct * 30 # Considerando alavancagem 1:30
    
    # Se a exposição nocional ultrapassar 3% do equity * alavancagem (ou similar)
    # Na verdade, o user pediu "Controle básico de exposição por moeda (USD, JPY, EUR)"
    # Vamos apenas logar ou vetar se ultrapassar um limite configurado.
    
    for curr, val in exposure.items():
        if val > max_exp_val:
            return False, f"🟠 EXPOSIÇÃO {curr}: ${val:,.2f} ultrapassa limite institucional"
            
    return True, ""

def check_total_exposure_limit(pending_symbol: Optional[str] = None, pending_volume: float = 0.0, pending_side: Optional[str] = None) -> Tuple[bool, str]:
    try:
        acc = mt5.account_info()
        basis = str(getattr(config, 'MAX_TOTAL_EXPOSURE_BASIS', 'balance')).lower()
        base_val = 0.0
        if acc:
            base_val = acc.balance if basis == "balance" else acc.equity
        limit_mult = float(getattr(config, 'MAX_TOTAL_EXPOSURE_MULTIPLIER', 2.0))
        if base_val <= 0:
            return True, ""
        total_limit_usd = base_val * limit_mult
        with mt5_lock:
            positions = mt5.positions_get()
        def estimate_exposure_usd(symbol: str, volume: float) -> float:
            info = get_symbol_info(symbol)
            if not info or volume <= 0:
                return 0.0
            contract = float(getattr(info, 'trade_contract_size', 100000) or 100000)
            tick = mt5.symbol_info_tick(symbol)
            price = 0.0
            if tick:
                if pending_side == "BUY":
                    price = float(getattr(tick, 'ask', 0.0) or 0.0)
                elif pending_side == "SELL":
                    price = float(getattr(tick, 'bid', 0.0) or 0.0)
                else:
                    price = float(getattr(tick, 'bid', 0.0) or 0.0)
            s = symbol.upper()
            if len(s) >= 6 and s[3:6] == "USD":
                return contract * volume * (price if price > 0 else 1.0)
            if len(s) >= 6 and s[0:3] == "USD":
                return contract * volume
            if ("XAUUSD" in s) or ("XAGUSD" in s) or ("US30" in s) or ("US500" in s) or ("NAS100" in s) or ("USTEC" in s) or ("USA500" in s):
                return contract * volume * (price if price > 0 else 1.0)
            return contract * volume
        current_exposure = 0.0
        if positions:
            for p in positions:
                current_exposure += estimate_exposure_usd(p.symbol, float(getattr(p, 'volume', 0.0) or 0.0))
        if pending_symbol and pending_volume and pending_volume > 0:
            current_exposure += estimate_exposure_usd(pending_symbol, pending_volume)
        if current_exposure > total_limit_usd:
            return False, f"Exposição total {current_exposure:,.2f} USD > limite {total_limit_usd:,.2f} USD"
        warn_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_WARNING_PCT', 0.80))
        alert_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_ALERT_PCT', 0.95))
        usage = (current_exposure / total_limit_usd) if total_limit_usd > 0 else 0.0
        if usage >= alert_pct:
            return True, f"🚨 Uso de margem {usage*100:.0f}% (Limite={total_limit_usd:,.0f} USD)"
        if usage >= warn_pct:
            return True, f"⚠️ Uso de margem {usage*100:.0f}% (Limite={total_limit_usd:,.0f} USD)"
        return True, ""
    except Exception as e:
        logger.error(f"check_total_exposure_limit erro: {e}")
        return True, ""

def _fx_base_quote(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        s = str(symbol).upper()
        if len(s) >= 6 and s[:3].isalpha() and s[3:6].isalpha():
            return s[:3], s[3:6]
    except Exception:
        pass
    return None, None

def _collect_net_currency_exposure(positions: List[Any]) -> Dict[str, float]:
    exposure: Dict[str, float] = {}
    try:
        for p in positions or []:
            base, quote = _fx_base_quote(p.symbol)
            vol_units = float(getattr(p, "volume", 0.0) or 0.0) * 100000.0
            if base and quote and vol_units > 0:
                if int(getattr(p, "type", 0)) == mt5.POSITION_TYPE_BUY:
                    exposure[base] = exposure.get(base, 0.0) + vol_units
                    exposure[quote] = exposure.get(quote, 0.0) - vol_units
                else:
                    exposure[base] = exposure.get(base, 0.0) - vol_units
                    exposure[quote] = exposure.get(quote, 0.0) + vol_units
            else:
                exposure["USD"] = exposure.get("USD", 0.0) + vol_units
        return exposure
    except Exception as e:
        logger.error(f"_collect_net_currency_exposure erro: {e}")
        return {}

def check_portfolio_exposure(pending_symbol: Optional[str], pending_volume: float, pending_side: Optional[str]) -> Tuple[bool, str]:
    try:
        acc = mt5.account_info()
        eq = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0
        with mt5_lock:
            positions = mt5.positions_get()
        net = _collect_net_currency_exposure([p for p in (positions or []) if int(getattr(p, "magic", 0)) == int(getattr(config, "MAGIC_NUMBER", 123456))])
        base, quote = _fx_base_quote(pending_symbol or "")
        vol_units = float(pending_volume or 0.0) * 100000.0
        if base and quote and vol_units > 0 and pending_side in ("BUY", "SELL"):
            if pending_side == "BUY":
                net[base] = net.get(base, 0.0) + vol_units
                net[quote] = net.get(quote, 0.0) - vol_units
            else:
                net[base] = net.get(base, 0.0) - vol_units
                net[quote] = net.get(quote, 0.0) + vol_units
        else:
            net["USD"] = net.get("USD", 0.0) + vol_units
        limits_map = getattr(config, "MAX_CURRENCY_EXPOSURE_PCT_MAP", {}) or {}
        default_pct = float(getattr(config, "MAX_CURRENCY_EXPOSURE_PCT", 0.03))
        lev = float(getattr(config, "ASSUMED_LEVERAGE", 30.0))
        corr_tighten = float(getattr(config, "CORR_EXPOSURE_TIGHTEN_FACTOR", 0.5))
        corr_limit = float(getattr(config, "CORRELATION_MAX", 0.75))
        tightened_currencies = set()
        correlations = getattr(config, "SYMBOL_CORRELATIONS", {})
        try:
            if pending_symbol and pending_symbol in correlations:
                sym_corrs = correlations[pending_symbol]
                for p in positions or []:
                    ps = getattr(p, "symbol", "")
                    if ps in sym_corrs and abs(float(sym_corrs[ps])) >= corr_limit:
                        if base:
                            tightened_currencies.add(base)
                        if quote:
                            tightened_currencies.add(quote)
        except Exception:
            pass
        for curr, val in net.items():
            pct = float(limits_map.get(curr, default_pct))
            if curr in tightened_currencies:
                pct *= corr_tighten
            max_val = eq * pct * lev
            if max_val <= 0:
                continue
            if abs(val) > max_val:
                return False, f"Exposição net {curr} {val:,.0f} > limite {max_val:,.0f}"
        return True, ""
    except Exception as e:
        logger.error(f"check_portfolio_exposure erro: {e}")
        return True, ""
# ===========================
# ESSENTIAL METRICS v5.3
# ===========================

def get_daily_metrics() -> dict:
    """
    ✅ v5.3: Carrega ou cria métricas do dia.
    Retorna dict com trades_today, wins, losses, max_dd, durations, exposure.
    """
    file_path = "data/metrics.json"
    today = datetime.now().strftime("%Y-%m-%d")
    
    default_metrics = {
        "date": today,
        "trades_today": 0,
        "wins": 0,
        "losses": 0,
        "max_dd_intraday": 0.0,
        "trade_durations": [],
        "currency_exposure": {"USD": 0.0, "JPY": 0.0, "EUR": 0.0}
    }
    
    if not os.path.exists(file_path):
        return default_metrics
        
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
            # Reset se mudou de dia
            if metrics.get("date") != today:
                return default_metrics
            return metrics
    except Exception as e:
        logger.error(f"❌ Erro ao ler metrics.json: {e}")
        return default_metrics

def save_daily_metrics(metrics: dict):
    """✅ v5.3: Salva métricas do dia."""
    file_path = "data/metrics.json"
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logger.error(f"❌ Erro ao salvar metrics.json: {e}")

def update_trade_metric(won: bool, duration_minutes: float = 0):
    """
    ✅ v5.3: Atualiza métricas após fechamento de trade.
    Args:
        won: True se trade vencedor
        duration_minutes: Tempo em minutos que a posição ficou aberta
    """
    metrics = get_daily_metrics()
    metrics["trades_today"] += 1
    
    if won:
        metrics["wins"] += 1
    else:
        metrics["losses"] += 1
    
    if duration_minutes > 0:
        metrics["trade_durations"].append(duration_minutes)
    
    save_daily_metrics(metrics)
    logger.info(f"📊 Métricas atualizadas: {metrics['trades_today']} trades, WR: {metrics['wins']}/{metrics['trades_today']}")

def calculate_current_metrics() -> dict:
    """
    ✅ v5.3: Calcula métricas em tempo real do MT5.
    Retorna: {win_rate, max_dd_today, avg_duration, exposure_by_currency}
    """
    try:
        with mt5_lock:
            acc = mt5.account_info()
            positions = mt5.positions_get()
        
        if not acc:
            return {}
        
        # 1. Win Rate (baseado em deals de hoje)
        from_date = datetime.now().replace(hour=0, minute=0, second=0)
        to_date = datetime.now() + timedelta(days=1)
        
        with mt5_lock:
            deals = mt5.history_deals_get(from_date, to_date)
        
        wins = 0
        total = 0
        total_profit = 0.0
        total_loss = 0.0
        
        if deals:
            magic = getattr(config, 'MAGIC_NUMBER', 123456)
            for deal in deals:
                if deal.magic != magic or deal.profit == 0:
                    continue
                total += 1
                if deal.profit > 0:
                    wins += 1
                    total_profit += float(deal.profit)
                else:
                    total_loss += float(deal.profit)
        
        win_rate = (wins / total * 100) if total > 0 else 0.0
        profit_factor = (total_profit / abs(total_loss)) if total_loss != 0 else 0.0
        pnl_today = total_profit + total_loss
        avg_trade = (pnl_today / total) if total > 0 else 0.0
        
        # 2. Max DD Intraday (simplificado: equity drawdown atual)
        # Para DD real do dia, precisaríamos do equity máximo do dia (não disponível facilmente)
        # Usamos aproximação: (balance - equity) / balance
        dd_current = ((acc.balance - acc.equity) / acc.balance * 100) if acc.balance > 0 else 0.0
        
        # 3. Tempo médio em trade (requer histórico completo - skip por enquanto)
        avg_duration = 0.0
        
        # 4. Exposição por moeda
        exposure = {"USD": 0.0, "JPY": 0.0, "EUR": 0.0}
        if positions:
            for pos in positions:
                symbol = pos.symbol.upper()
                nocional = pos.volume * 100000  # Lote padrão
                for curr in exposure.keys():
                    if curr in symbol:
                        exposure[curr] += nocional
        
        return {
            "win_rate": win_rate,
            "trades_today": total,
            "max_dd_intraday": dd_current,
            "avg_duration_minutes": avg_duration,
            "currency_exposure": exposure,
            "profit_factor_today": profit_factor,
            "pnl_today": pnl_today,
            "avg_trade": avg_trade
        }
        
    except Exception as e:
        logger.error(f"❌ Erro em calculate_current_metrics: {e}")
        return {}

def log_metrics_summary():
    """✅ v5.3: Loga resumo de métricas essenciais."""
    try:
        metrics = calculate_current_metrics()
        if not metrics:
            return
        
        logger.info("="*60)
        logger.info("📊 MÉTRICAS ESSENCIAIS")
        logger.info(f"  Trades Hoje: {metrics.get('trades_today', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        logger.info(f"  Profit Factor (Hoje): {metrics.get('profit_factor_today', 0):.2f}")
        logger.info(f"  PnL (Hoje): ${metrics.get('pnl_today', 0):+.2f}")
        logger.info(f"  PnL Médio/Trade: ${metrics.get('avg_trade', 0):+.2f}")
        logger.info(f"  Max DD Intraday: {metrics.get('max_dd_intraday', 0):.2f}%")
        logger.info(f"  Tempo Médio: {metrics.get('avg_duration_minutes', 0):.0f}min")
        
        exp = metrics.get('currency_exposure', {})
        logger.info(f"  Exposição USD: ${exp.get('USD', 0):,.0f}")
        logger.info(f"  Exposição JPY: ${exp.get('JPY', 0):,.0f}")
        logger.info(f"  Exposição EUR: ${exp.get('EUR', 0):,.0f}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Erro ao logar métricas: {e}")

def _retry_call(fn, max_attempts: int = 3, base_sleep: float = 1.0, *args, **kwargs):
    attempts = 0
    last_exc = None
    while attempts < max_attempts:
        try:
            return True, fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(base_sleep * (2 ** attempts))
            attempts += 1
    return False, last_exc

def _classify_asset(symbol: str) -> Tuple[str, str]:
    s = symbol.upper()
    if any(k in s for k in ["US30", "NAS100", "US500", "USA500", "USTEC", "DE40", "GER40", "GER30", "UK100", "HK50", "JP225", "FRA40"]):
        if any(k in s for k in ["US30", "NAS100", "US500", "USA500", "USTEC"]):
            return "INDICES", "NA"
        if any(k in s for k in ["DE40", "GER40", "GER30", "FRA40"]):
            return "INDICES", "EU"
        if "UK100" in s:
            return "INDICES", "EU"
        if "HK50" in s or "JP225" in s:
            return "INDICES", "AS"
        return "INDICES", "GL"
    if any(k in s for k in ["XAU", "XAG", "GOLD", "SILVER"]):
        return "METALS", "GL"
    if any(k in s for k in ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]):
        return "CRYPTO", "GL"
    if len(s) >= 6:
        base = s[0:3]
        quote = s[3:6]
        region_map = {
            "USD": "NA", "EUR": "EU", "JPY": "AS", "GBP": "EU", "AUD": "OC",
            "NZD": "OC", "CAD": "NA", "CHF": "EU", "BRL": "SA", "CNH": "AS", "CNY": "AS"
        }
        return "FX", region_map.get(base, "GL")
    return "UNKNOWN", "GL"

def _estimate_notional_usd(symbol: str, volume: float) -> float:
    info = get_symbol_info(symbol)
    if not info or volume <= 0:
        return 0.0
    contract = float(getattr(info, 'trade_contract_size', 100000) or 100000)
    tick = mt5.symbol_info_tick(symbol)
    price = 0.0
    if tick:
        price = float(getattr(tick, 'bid', 0.0) or 0.0)
    s = symbol.upper()
    if len(s) >= 6 and s[3:6] == "USD":
        return contract * volume * (price if price > 0 else 1.0)
    if len(s) >= 6 and s[0:3] == "USD":
        return contract * volume
    if any(k in s for k in ["XAUUSD", "XAGUSD", "US30", "US500", "NAS100", "USTEC", "USA500"]):
        return contract * volume * (price if price > 0 else 1.0)
    return contract * volume

def collect_portfolio_snapshot() -> Dict[str, Any]:
    br_now = get_brasilia_time()
    ts = datetime.now(pytz.utc)
    ok_conn = check_mt5_connection() or ensure_mt5_connection()
    issues = []
    meta = {
        "timestamp_utc": ts.isoformat(timespec="seconds"),
        "timestamp_brt": br_now.isoformat(timespec="seconds"),
        "system_version": "XP3_PRO_FOREX_v4.2",
    }
    positions = []
    orders_pending = []
    deals_today = []
    acc = None
    t0 = time.time()
    try:
        with mt5_lock:
            acc = mt5.account_info()
    except Exception as e:
        issues.append(f"account_info: {e}")
    ok_pos, res_pos = _retry_call(lambda: mt5.positions_get(), getattr(config, "SNAPSHOT_RETRY_ATTEMPTS", 3), getattr(config, "SNAPSHOT_BACKOFF_BASE", 1.0))
    if ok_pos and res_pos:
        for p in res_pos:
            cls, region = _classify_asset(p.symbol)
            positions.append({
                "ticket": int(getattr(p, "ticket", 0)),
                "symbol": str(getattr(p, "symbol", "")),
                "type": ("BUY" if getattr(p, "type", 0) == mt5.POSITION_TYPE_BUY else "SELL"),
                "volume": float(getattr(p, "volume", 0.0)),
                "price_open": float(getattr(p, "price_open", 0.0)),
                "price_current": float(getattr(p, "price_current", 0.0)),
                "profit": float(getattr(p, "profit", 0.0)),
                "sl": float(getattr(p, "sl", 0.0)),
                "tp": float(getattr(p, "tp", 0.0)),
                "magic": int(getattr(p, "magic", 0)),
                "asset_class": cls,
                "region": region,
                "notional_usd": _estimate_notional_usd(getattr(p, "symbol", ""), float(getattr(p, "volume", 0.0)))
            })
    else:
        issues.append(f"positions_get_fail: {res_pos}")
    ok_ord, res_ord = _retry_call(lambda: mt5.orders_get(), getattr(config, "SNAPSHOT_RETRY_ATTEMPTS", 3), getattr(config, "SNAPSHOT_BACKOFF_BASE", 1.0))
    if ok_ord and res_ord:
        for o in res_ord:
            orders_pending.append({
                "ticket": int(getattr(o, "ticket", 0)),
                "symbol": str(getattr(o, "symbol", "")),
                "type": int(getattr(o, "type", 0)),
                "volume": float(getattr(o, "volume_initial", 0.0)),
                "price": float(getattr(o, "price_open", 0.0)),
                "sl": float(getattr(o, "sl", 0.0)),
                "tp": float(getattr(o, "tp", 0.0)),
                "state": int(getattr(o, "state", 0)),
                "time": int(getattr(o, "time_setup", 0))
            })
    else:
        issues.append(f"orders_get_fail: {res_ord}")
    from_date = br_now.replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = from_date + timedelta(days=1)
    ok_deals, res_deals = _retry_call(lambda: mt5.history_deals_get(from_date, to_date), getattr(config, "SNAPSHOT_RETRY_ATTEMPTS", 3), getattr(config, "SNAPSHOT_BACKOFF_BASE", 1.0))
    if ok_deals and res_deals:
        magic = getattr(config, "MAGIC_NUMBER", 123456)
        for d in res_deals or []:
            if getattr(d, "magic", None) != magic:
                continue
            deals_today.append({
                "time": datetime.fromtimestamp(getattr(d, "time", 0)).isoformat(timespec="seconds"),
                "symbol": str(getattr(d, "symbol", "")),
                "type": int(getattr(d, "type", 0)),
                "entry": int(getattr(d, "entry", 0)),
                "volume": float(getattr(d, "volume", 0.0)),
                "price": float(getattr(d, "price", 0.0)),
                "profit": float(getattr(d, "profit", 0.0)),
                "commission": float(getattr(d, "commission", 0.0)),
                "swap": float(getattr(d, "swap", 0.0)),
                "order": int(getattr(d, "order", 0)),
                "position_id": int(getattr(d, "position_id", 0)),
            })
    else:
        issues.append(f"history_deals_get_fail: {res_deals}")
    total_value = sum(p.get("notional_usd", 0.0) for p in positions)
    total_pnl = sum(p.get("profit", 0.0) for p in positions)
    by_class = {}
    by_currency = {}
    by_region = {}
    for p in positions:
        cls = p.get("asset_class", "UNKNOWN")
        by_class[cls] = by_class.get(cls, 0.0) + p.get("notional_usd", 0.0)
        s = str(p.get("symbol", "")).upper()
        if len(s) >= 6:
            base = s[0:3]
            quote = s[3:6]
            for curr in [base, quote]:
                by_currency[curr] = by_currency.get(curr, 0.0) + p.get("notional_usd", 0.0) * 0.5
        reg = p.get("region", "GL")
        by_region[reg] = by_region.get(reg, 0.0) + p.get("notional_usd", 0.0)
    complete = ok_conn and len(positions) >= 0
    t_elapsed = time.time() - t0
    snapshot = {
        "meta": meta,
        "account": {
            "login": int(getattr(acc, "login", 0)) if acc else 0,
            "balance": float(getattr(acc, "balance", 0.0)) if acc else 0.0,
            "equity": float(getattr(acc, "equity", 0.0)) if acc else 0.0,
            "margin": float(getattr(acc, "margin", 0.0)) if acc else 0.0,
            "leverage": int(getattr(acc, "leverage", 0)) if acc else 0,
        },
        "positions": positions,
        "orders_pending": orders_pending,
        "deals_today": deals_today,
        "metrics": {
            "positions_count": len(positions),
            "orders_pending_count": len(orders_pending),
            "deals_today_count": len(deals_today),
            "total_value_usd": total_value,
            "total_pnl_usd": total_pnl,
            "allocation_by_class_usd": by_class,
            "exposure_by_currency_usd": by_currency,
            "exposure_by_region_usd": by_region,
            "processing_seconds": t_elapsed
        },
        "complete": complete,
        "issues": issues
    }
    return snapshot

def save_portfolio_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(getattr(config, "SNAPSHOT_OUTPUT_DIR", "data/portfolio_snapshots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    br_ts = snapshot.get("meta", {}).get("timestamp_brt", "")
    date_str = br_ts[:10] if br_ts else datetime.now().strftime("%Y-%m-%d")
    time_str = br_ts[11:16].replace(":", "-") if br_ts else datetime.now().strftime("%H-%M")
    base = f"snapshot_{date_str}_{time_str}"
    json_path = out_dir / f"{base}.json"
    csv_pos_path = out_dir / f"{base}_positions.csv"
    csv_ord_path = out_dir / f"{base}_orders.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    with open(csv_pos_path, "w", encoding="utf-8", newline="") as f:
        fields = ["ticket", "symbol", "type", "volume", "price_open", "price_current", "profit", "sl", "tp", "magic", "asset_class", "region", "notional_usd"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in snapshot.get("positions", []):
            w.writerow({k: p.get(k) for k in fields})
    with open(csv_ord_path, "w", encoding="utf-8", newline="") as f:
        fields = ["ticket", "symbol", "type", "volume", "price", "sl", "tp", "state", "time"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in snapshot.get("orders_pending", []):
            w.writerow({k: o.get(k) for k in fields})
    latest_path = out_dir / "latest.json"
    try:
        if latest_path.exists():
            backup_dir = out_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"latest_{ts}.json"
            latest_path.replace(backup_path)
        latest_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"erro ao versionar latest.json: {e}")
    return {"json": str(json_path), "positions_csv": str(csv_pos_path), "orders_csv": str(csv_ord_path)}

def compare_and_alert_snapshot(prev: Dict[str, Any], curr: Dict[str, Any]) -> None:
    try:
        th = float(getattr(config, "SNAPSHOT_ALERT_DEVIATION_PCT", 0.2))
        pv = float(((prev or {}).get("metrics") or {}).get("total_value_usd", 0.0))
        cv = float(((curr or {}).get("metrics") or {}).get("total_value_usd", 0.0))
        if pv > 0:
            delta = (cv - pv) / pv
            if abs(delta) >= th:
                utils_msg = (
                    f"📦 Snapshot Sexta 19h\n"
                    f"Valor total mudou {delta*100:.1f}%\n"
                    f"Anterior: ${pv:,.2f} | Atual: ${cv:,.2f}\n"
                    f"Posições: {((curr or {}).get('metrics') or {}).get('positions_count', 0)}"
                )
                send_telegram_message(utils_msg)
    except Exception:
        pass

def run_weekly_snapshot() -> Dict[str, Any]:
    snap = collect_portfolio_snapshot()
    paths = save_portfolio_snapshot(snap)
    prev = None
    try:
        out_dir = Path(getattr(config, "SNAPSHOT_OUTPUT_DIR", "data/portfolio_snapshots"))
        latest = out_dir / "latest.json"
        if latest.exists():
            prev = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        prev = None
    try:
        compare_and_alert_snapshot(prev, snap)
    except Exception:
        pass
    if not snap.get("complete", True) or snap.get("issues"):
        send_telegram_message("⚠️ Snapshot Sexta 19h com dados parciais ou issues. Verifique logs.")
    else:
        send_telegram_message(f"✅ Snapshot Sexta 19h salvo.\nJSON: {paths.get('json')}\nCSV: {paths.get('positions_csv')}")
=======
# utils_forex.py - XP3 PRO FOREX UTILS v4.2 INSTITUCIONAL
"""
🚀 XP3 PRO FOREX UTILS - VERSÃO INSTITUCIONAL v4.2
✅ Funções auxiliares para o bot
✅ Cálculo de indicadores técnicos
✅ Conexão MT5 e dados de mercado
✅ Cálculo de volume e SL/TP
✅ CORREÇÃO: get_tick_value e get_pip_size mais robustos
✅ Suporte a SYMBOL_MAP (para ML Optimizer)
✅ Backtesting seguro
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import time
import logging
import threading
import queue
from concurrent.futures import Future
import json
import os
from pathlib import Path
import csv
import sqlite3
from typing import Optional, Tuple, Dict, Any, List

import config_forex as config
from decimal import Decimal, ROUND_HALF_UP
from numba import njit

logger = logging.getLogger("XP3_UTILS")

# Lock para operações MT5 (evita race conditions)
mt5_lock = threading.RLock()
_mt5_queue = queue.Queue()
_mt5_worker_started = False
_mt5_worker_thread = None
def _mt5_worker():
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
    global _mt5_worker_started, _mt5_worker_thread
    if not _mt5_worker_started:
        _mt5_worker_thread = threading.Thread(target=_mt5_worker, name="MT5Worker", daemon=True)
        _mt5_worker_thread.start()
        _mt5_worker_started = True
def mt5_exec(func, *args, **kwargs):
    ensure_mt5_worker()
    fut = Future()
    _mt5_queue.put((func, args, kwargs, fut))
    return fut.result(timeout=30)
def mt5_shutdown_worker():
    if _mt5_worker_started:
        _mt5_queue.put(None)

# Cache para silenciamento de erros repetitivos (ex: 10016)
ERROR_CACHE = {} # {key: {"time": float, "price": float}}
INDICATOR_CACHE = {}
INDICATOR_CACHE_TTL = 60

SESSION_METRICS_LOCK = threading.Lock()
DAYREPORT_CACHE_LOCK = threading.Lock()
DAYREPORT_CACHE = {}

@njit
def ema_numba(x, period):
    alpha = 2.0 / (period + 1.0)
    result = np.empty_like(x)
    if len(x) == 0:
        return result
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1.0 - alpha) * result[i - 1]
    return result

@njit
def calculate_rsi_numba(close, period=14):
    rsi = np.zeros_like(close)
    gains = np.zeros_like(close)
    losses = np.zeros_like(close)
    for i in range(1, len(close)):
        change = close[i] - close[i - 1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = abs(change)
    avg_gain = 0.0
    avg_loss = 0.0
    end = min(len(close) - 1, period)
    for i in range(1, end + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    if end > 0:
        avg_gain /= end
        avg_loss /= end
    for i in range(period, len(close)):
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    return rsi

@njit
def calculate_atr_numba(high, low, close, period):
    tr = np.zeros_like(close)
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    if len(close) > period:
        s = 0.0
        for i in range(1, period + 1):
            s += tr[i]
        atr[period] = s / period
        for i in range(period + 1, len(close)):
            atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period
    return atr

@njit
def calculate_adx_numba(high, low, close, period=14):
    adx = np.zeros_like(close)
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)
    tr = np.zeros_like(close)
    for i in range(1, len(close)):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    smooth_plus_dm = 0.0
    smooth_minus_dm = 0.0
    smooth_tr = 0.0
    end = min(len(close) - 1, period)
    for i in range(1, end + 1):
        smooth_plus_dm += plus_dm[i]
        smooth_minus_dm += minus_dm[i]
        smooth_tr += tr[i]
    for i in range(period, len(close)):
        smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period) + plus_dm[i]
        smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period) + minus_dm[i]
        smooth_tr = smooth_tr - (smooth_tr / period) + tr[i]
        if smooth_tr == 0.0:
            adx[i] = 0.0
            continue
        plus_di = 100.0 * (smooth_plus_dm / smooth_tr)
        minus_di = 100.0 * (smooth_minus_dm / smooth_tr)
        denom = plus_di + minus_di
        dx = 0.0
        if denom > 0.0:
            dx = 100.0 * abs(plus_di - minus_di) / denom
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i - 1] * (period - 1)) + dx) / period
    return adx

def _get_session_metrics_path(date_str: str) -> str:
    return str(Path("analysis_logs") / f"session_metrics_{date_str}.json")

def _load_session_metrics(date_str: str) -> Dict[str, Any]:
    path = _get_session_metrics_path(date_str)
    if not os.path.exists(path):
        return {
            "date": date_str,
            "sessions": {},
            "last_update": None
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid json")
        data.setdefault("date", date_str)
        data.setdefault("sessions", {})
        return data
    except Exception:
        return {
            "date": date_str,
            "sessions": {},
            "last_update": None
        }

def _save_session_metrics(date_str: str, data: Dict[str, Any]) -> None:
    path = _get_session_metrics_path(date_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_session_metrics(session_name: str, executed: bool, rejected: bool, reason: str) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    sess = (session_name or "UNKNOWN").upper()
    reason_key = (reason or "N/A").strip()
    if len(reason_key) > 120:
        reason_key = reason_key[:120] + "..."

    with SESSION_METRICS_LOCK:
        data = _load_session_metrics(date_str)
        sessions = data.setdefault("sessions", {})
        block = sessions.setdefault(sess, {"analyzed": 0, "executed": 0, "rejected": 0, "reasons": {}})
        block["analyzed"] = int(block.get("analyzed", 0)) + 1
        if executed and not rejected:
            block["executed"] = int(block.get("executed", 0)) + 1
        if rejected:
            block["rejected"] = int(block.get("rejected", 0)) + 1
            reasons = block.setdefault("reasons", {})
            reasons[reason_key] = int(reasons.get(reason_key, 0)) + 1
        data["last_update"] = datetime.now().isoformat(timespec="seconds")
        _save_session_metrics(date_str, data)

def get_session_metrics_summary() -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    with SESSION_METRICS_LOCK:
        data = _load_session_metrics(date_str)
    sessions = data.get("sessions") or {}
    if not sessions:
        return "Sem métricas por sessão ainda (aguardando análises)."

    lines = [f"<b>Métricas por Sessão</b> ({date_str})"]
    for sess in sorted(sessions.keys()):
        s = sessions[sess] or {}
        analyzed = int(s.get("analyzed", 0))
        executed = int(s.get("executed", 0))
        rejected = int(s.get("rejected", 0))
        exec_rate = (executed / analyzed * 100) if analyzed > 0 else 0.0
        lines.append(f"\n<b>{sess}</b> | Análises: {analyzed} | Exec: {executed} | Rej: {rejected} | Exec%: {exec_rate:.1f}%")
        reasons = s.get("reasons") or {}
        if reasons:
            top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append("Top motivos:")
            for r, c in top:
                lines.append(f"- {c}x {r}")
    text = "\n".join(lines)
    if len(text) > 3900:
        text = text[:3900] + "\n..."
    return text

def get_session_metrics_summary_by_date(date_str: str) -> str:
    with SESSION_METRICS_LOCK:
        data = _load_session_metrics(date_str)
    sessions = data.get("sessions") or {}
    if not sessions:
        return f"Sem métricas por sessão em {date_str}."

    lines = [f"<b>Métricas por Sessão</b> ({date_str})"]
    for sess in sorted(sessions.keys()):
        s = sessions[sess] or {}
        analyzed = int(s.get("analyzed", 0))
        executed = int(s.get("executed", 0))
        rejected = int(s.get("rejected", 0))
        exec_rate = (executed / analyzed * 100) if analyzed > 0 else 0.0
        lines.append(f"\n<b>{sess}</b> | Análises: {analyzed} | Exec: {executed} | Rej: {rejected} | Exec%: {exec_rate:.1f}%")
        reasons = s.get("reasons") or {}
        if reasons:
            top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append("Top motivos:")
            for r, c in top:
                lines.append(f"- {c}x {r}")
    text = "\n".join(lines)
    if len(text) > 3900:
        text = text[:3900] + "\n..."
    return text

def _escape_html(text: str) -> str:
    try:
        import html
        return html.escape(text or "")
    except Exception:
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def generate_day_report(date_str: Optional[str] = None) -> str:
    import re
    import collections

    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    analysis_path = str(Path("analysis_logs") / f"analysis_log_{date_str}.txt")
    runtime_log_path = str(Path("logs") / "xp3_forex.log")

    cache_key = date_str
    analysis_mtime = os.path.getmtime(analysis_path) if os.path.exists(analysis_path) else 0
    runtime_mtime = os.path.getmtime(runtime_log_path) if os.path.exists(runtime_log_path) else 0

    with DAYREPORT_CACHE_LOCK:
        cached = DAYREPORT_CACHE.get(cache_key)
        if cached and cached.get("analysis_mtime") == analysis_mtime and cached.get("runtime_mtime") == runtime_mtime:
            return cached.get("text", "")

    lines_out = [f"📅 <b>Day Report</b> ({date_str})"]

    if date_str == datetime.now().strftime("%Y-%m-%d"):
        try:
            m = calculate_current_metrics()
            if m:
                lines_out.append(
                    "\n<b>Métricas (Hoje)</b>\n"
                    f"Trades: {m.get('trades_today', 0)} | WR: {m.get('win_rate', 0):.1f}% | "
                    f"PF: {m.get('profit_factor_today', 0):.2f} | PnL: ${m.get('pnl_today', 0):+.2f} | "
                    f"DD: {m.get('max_dd_intraday', 0):.2f}%"
                )
        except Exception:
            pass

    reasons = collections.Counter()
    symbols = collections.Counter()
    strategies = collections.Counter()
    signals = collections.Counter()
    ml_vals = []
    first_ts = None
    last_ts = None

    re_ts = re.compile(r'(\d\d:\d\d:\d\d)\s*\|\s*([^|]+?)\s*\|')

    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "📊" in line and "Sinal:" in line and "Estratégia:" in line and "ML:" in line:
                        try:
                            parts = line.split("|")
                            sig = parts[0].split("Sinal:")[1].strip()
                            strat = parts[1].split("Estratégia:")[1].strip()
                            signals[sig] += 1
                            strategies[strat] += 1
                            ml_part = parts[3].split("ML:")[1]
                            ml_num = re.sub(r"[^0-9.]", "", ml_part)
                            if ml_num:
                                ml_vals.append(float(ml_num))
                        except Exception:
                            pass
                        continue

                    if "💬" in line and "Motivo:" in line:
                        try:
                            reason = line.split("Motivo:", 1)[1].strip()
                            reasons[reason] += 1
                        except Exception:
                            pass
                        continue

                    if " | " in line and ":" in line:
                        m = re_ts.search(line)
                        if m:
                            t = m.group(1)
                            sym = m.group(2).strip()
                            symbols[sym] += 1
                            if first_ts is None:
                                first_ts = t
                            last_ts = t
        except Exception:
            pass

    total_entries = int(sum(symbols.values()))
    if total_entries:
        lines_out.append(
            "\n<b>Atividade (analysis log)</b>\n"
            f"Intervalo: {first_ts} → {last_ts}\n"
            f"Entradas: {total_entries} | Ativos: {len(symbols)}"
        )

        if ml_vals:
            ml_vals.sort()
            ml_min = ml_vals[0]
            ml_med = ml_vals[len(ml_vals) // 2]
            ml_max = ml_vals[-1]
            lines_out.append(f"ML (min/med/max): {ml_min:.0f}/{ml_med:.0f}/{ml_max:.0f}")

        top_reasons = reasons.most_common(10)
        if top_reasons:
            lines_out.append("\n<b>Top motivos (analysis)</b>")
            for r, c in top_reasons:
                lines_out.append(f"- {c}x {_escape_html(r)}")

        top_symbols = symbols.most_common(10)
        if top_symbols:
            lines_out.append("\n<b>Top ativos (activity)</b>")
            for s, c in top_symbols:
                lines_out.append(f"- {c}x {_escape_html(s)}")

        top_signals = signals.most_common(6)
        if top_signals:
            lines_out.append("\n<b>Sinais</b>")
            for s, c in top_signals:
                lines_out.append(f"- {c}x {_escape_html(s)}")

        top_strats = strategies.most_common(6)
        if top_strats:
            lines_out.append("\n<b>Estratégias</b>")
            for s, c in top_strats:
                lines_out.append(f"- {c}x {_escape_html(s)}")
    else:
        lines_out.append("\n<b>Atividade (analysis log)</b>\nSem dados (arquivo ausente ou vazio).")

    session_metrics_path = _get_session_metrics_path(date_str)
    if os.path.exists(session_metrics_path):
        try:
            lines_out.append("\n" + get_session_metrics_summary_by_date(date_str))
        except Exception:
            pass

    runtime_counts = collections.Counter()
    if os.path.exists(runtime_log_path):
        try:
            with open(runtime_log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith(date_str):
                        continue
                    if "Thread FastLoop MORTA" in line:
                        runtime_counts["fastloop_dead"] += 1
                    if "Tentativa de iniciar FastLoop duplicado" in line:
                        runtime_counts["fastloop_dup"] += 1
                    if "BLOQUEIO INSTITUCIONAL" in line:
                        runtime_counts["risk_block"] += 1
                    if "Spread Alto" in line:
                        runtime_counts["spread_high"] += 1
                    if "ORDEM EXECUTADA" in line:
                        runtime_counts["orders_executed"] += 1
        except Exception:
            pass

    if runtime_counts:
        lines_out.append("\n<b>Estabilidade (runtime)</b>")
        for k in ["orders_executed", "spread_high", "risk_block", "fastloop_dead", "fastloop_dup"]:
            if k in runtime_counts:
                lines_out.append(f"- {k}: {runtime_counts[k]}")

    text = "\n".join(lines_out)
    if len(text) > 3900:
        text = text[:3900] + "\n..."

    with DAYREPORT_CACHE_LOCK:
        DAYREPORT_CACHE[cache_key] = {"analysis_mtime": analysis_mtime, "runtime_mtime": runtime_mtime, "text": text}

    return text

# ✅ Land Trading: Mapeamento de Aliases (Nomenclatura Corretora)
SYMBOL_ALIASES = {
    "NAS100": ["USTEC", "US100", "NAS100.cash", "NAS100.m", "NAS100.raw"],
    "US30": ["US30.cash", "US30.m", "US30.raw", "WS30"],
    "GER40": ["GER40.cash", "DE40", "DAX40", "GER40.m", "DE40.cash"],
    "UK100": ["UK100.cash", "FTSE100", "UK100.m"],
    "US500": ["SPX500", "USA500", "US500.cash"],
    "XAUUSD": ["GOLD", "XAUUSD.raw", "XAUUSD.m"],
    "XAGUSD": ["SILVER", "XAGUSD.raw"],
    "BTCUSD": ["BTCUSD.spot", "BTCUSD.m"],
}

# ✅ Land Trading: Normalização de Símbolos
def normalize_symbol(symbol: str) -> str:
    """
    Tenta encontrar a variação correta do símbolo no Market Watch.
    Utiliza aliases e sufixos comuns.
    """
    if not check_mt5_connection():
        return symbol
        
    # 1. Verifica se o símbolo já é válido
    with mt5_lock:
        if mt5.symbol_select(symbol, True):
            return symbol
            
    # 2. Tenta Aliases
    potential_names = SYMBOL_ALIASES.get(symbol, [])
    
    # 3. Adiciona variações comuns de sufixo
    variants = [symbol + ".raw", symbol + ".m", symbol + ".cash", symbol + "#", symbol + ".spot"]
    for p in potential_names:
        variants.append(p)
        variants.append(p + ".raw")
        variants.append(p + ".m")
    
    with mt5_lock:
        for v in variants:
            if mt5.symbol_select(v, True):
                return v

        # 4. Busca Fuzzy (Inteligente - Land Trading)
        # Se falhar tudo, varre todos os símbolos e tenta achar substring
        all_symbols = mt5.symbols_get()
        if all_symbols:
            for s in all_symbols:
                # Se o nome base (ex: EURUSD) estiver contido no nome do MT5 (ex: EURUSD.a)
                if symbol in s.name:
                    if mt5.symbol_select(s.name, True):
                        logger.info(f"🔍 Auto-Correction: '{symbol}' -> '{s.name}'")
                        return s.name
    
    return symbol

# ✅ Land Trading: Check de Sincronização de Nomes
def check_symbol_sync(symbols: List[str]):
    """
    Verifica se os símbolos do config existem no Market Watch e avisa se houver erro.
    Também popula uma lista de exclusão para símbolos não encontrados.
    """
    if not check_mt5_connection():
        return
        
    # ✅ REQUISITO: Imprime no log e no TERMINAL todos os símbolos que o MT5 está vendo
    with mt5_lock:
        all_mt5_symbols = [s.name for s in mt5.symbols_get()]
    
    logger.info(f"📊 Diagnóstico MT5: Mercado oferece {len(all_mt5_symbols)} símbolos.")
    print("\n" + "="*50)
    print(f"🔍 DISPONÍVEIS NO MT5 ({len(all_mt5_symbols)} ativos):")
    print(all_mt5_symbols[:200]) # Mostra os primeiros 200 para não estourar o buffer
    print("="*50 + "\n")
    
    # Adiciona ao config uma lista de exclusão dinâmica se não existir
    if not hasattr(config, 'BLACKLISTED_SYMBOLS'):
        config.BLACKLISTED_SYMBOLS = set()

    for sym in symbols:
        if not mt5.symbol_select(sym, True):
            best_v = normalize_symbol(sym)
            if best_v != sym:
                logger.warning(f"⚠️ Símbolo '{sym}' não encontrado! Sugestão: Use '{best_v}' no config_forex.py")
            else:
                logger.error(f"❌ Símbolo '{sym}' totalmente inacessível. Adicionado à lista de exclusão.")
                config.BLACKLISTED_SYMBOLS.add(sym)
        else:
            # ✅ WARMUP: Força download de histórico
            print(f"🔥 Aquecendo dados para {sym}...")
            with mt5_lock:
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 1000)
            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️ Falha no aquecimento de dados para {sym}")
            else:
                logger.info(f"✅ {sym}: {len(rates)} candles carregados.")

# ===========================
# MT5 CONNECTION
# ===========================
def ensure_mt5_connection() -> bool:
    term_path = getattr(config, 'MT5_TERMINAL_PATH', None)
    ok_init = mt5_exec(mt5.initialize, path=term_path)
    if not ok_init:
        logger.error(f"❌ Falha ao inicializar MT5 em {term_path}")
        return False
    login = str(getattr(config, 'MT5_LOGIN', '')).strip()
    passwd = str(getattr(config, 'MT5_PASSWORD', '')).strip()
    server = str(getattr(config, 'MT5_SERVER', '')).strip()
    if login and passwd and server:
        ok_login = mt5_exec(mt5.login, login, passwd, server)
        if not ok_login:
            logger.error(f"❌ Falha ao logar no MT5: {mt5.last_error()}")
            mt5_exec(mt5.shutdown)
            return False
    ti = mt5_exec(mt5.terminal_info)
    if ti is None:
        logger.error("❌ Terminal MT5 não disponível após init.")
        return False
    return True

def check_mt5_connection() -> bool:
    ti = mt5_exec(mt5.terminal_info)
    ai = mt5_exec(mt5.account_info)
    return ti is not None and ai is not None

# ===========================
# MARKET STATUS
# ===========================
def get_brasilia_time() -> datetime:
    """Retorna o horário de Brasília considerando o SERVER_OFFSET se necessário"""
    # Horário de Brasília é UTC-3
    tz_br = pytz.timezone('America/Sao_Paulo')
    now_br = datetime.now(tz_br)
    
    # Se o sistema precisar de um offset específico em relação ao servidor MT5,
    # o usuário pode configurar no config.py. 
    # Por padrão, estamos usando o horário da máquina local (UTC-3).
    if getattr(config, 'SERVER_OFFSET', 0) != 0:
        now_br += timedelta(hours=config.SERVER_OFFSET)
        
    return now_br

def get_current_trading_session() -> Dict[str, str]:
    """Retorna a sessão de trading atual baseada no horário de Brasília"""
    br_time = get_brasilia_time()
    current_hm = br_time.strftime("%H:%M")
    
    def is_between(start, end, now):
        if start <= end:
            return start <= now <= end
        else: # Crosses midnight
            return now >= start or now <= end

    # Asian Session: 22:00 às 05:00
    if is_between(config.ASIAN_SESSION_START, config.ASIAN_SESSION_END, current_hm):
        return {
            "name": "ASIAN",
            "display": "ASIÁTICA (SNIPER MODE)",
            "emoji": "🎌",
            "color": "magenta"
        }
    
    # Golden Hour: 10:00 às 14:00
    if is_between(config.GOLDEN_HOUR_START, config.GOLDEN_HOUR_END, current_hm):
        return {
            "name": "GOLDEN",
            "display": "HORÁRIO DE OURO",
            "emoji": "🌟",
            "color": "gold"
        }
    
    # Protection: 18:00 às 22:00
    if is_between(config.PROTECTION_SESSION_START, config.PROTECTION_SESSION_END, current_hm):
        return {
            "name": "PROTECTION",
            "display": "PROTEÇÃO (RISCO ALTO)",
            "emoji": "🛡️",
            "color": "red"
        }
    
    # Normal: 05:00 às 09:59 e 14:01 às 17:59
    if is_between(config.NORMAL_SESSION_1_START, config.NORMAL_SESSION_1_END, current_hm) or \
       is_between(config.NORMAL_SESSION_2_START, config.NORMAL_SESSION_2_END, current_hm):
        return {
            "name": "NORMAL",
            "display": "NORMAL",
            "emoji": "⚖️",
            "color": "blue"
        }

    return {
        "name": "OFF_HOURS",
        "display": "FORA DE HORÁRIO",
        "emoji": "💤",
        "color": "gray"
    }

def is_market_open() -> bool:
    """Verifica se o mercado está aberto."""
    market_status = get_market_status()
    return market_status['status'] == 'open'

def get_market_status() -> Dict[str, str]:
    """Retorna o status atual do mercado."""
    utc_now = datetime.now(pytz.utc)
    weekday = utc_now.weekday()  # Monday=0, Sunday=6
    hour = utc_now.hour

    # Fechamento: Sexta 21:00 UTC até Domingo 21:00 UTC
    is_closed = False
    
    if weekday == 5:  # Sábado
        is_closed = True
    elif weekday == 4 and hour >= 21:  # Sexta após 21:00 UTC
        is_closed = True
    elif weekday == 6 and hour < 21:   # Domingo antes das 21:00 UTC
        is_closed = True

    if is_closed:
        return {'status': 'closed', 'message': 'Mercado Fechado (Fim de Semana)', 'emoji': '💤', 'color': 'red'}

    return {'status': 'open', 'message': 'Mercado Aberto', 'emoji': '🟢', 'color': 'green'}

def _parse_hhmm_to_minutes(hhmm: str) -> int:
    parts = str(hhmm).strip().split(":")
    if len(parts) != 2:
        return 0
    h = int(parts[0])
    m = int(parts[1])
    return (h * 60) + m

def get_weekend_protection_state() -> Dict[str, Any]:
    utc_now = datetime.now(pytz.utc)
    weekday = utc_now.weekday()
    now_min = utc_now.hour * 60 + utc_now.minute

    close_min = _parse_hhmm_to_minutes(getattr(config, 'FRIDAY_MARKET_CLOSE_UTC', "21:00"))
    entry_cutoff_min = _parse_hhmm_to_minutes(getattr(config, 'FRIDAY_ENTRY_CUTOFF_UTC', "20:00"))
    force_close_min = _parse_hhmm_to_minutes(getattr(config, 'FRIDAY_FORCE_CLOSE_UTC', "20:55"))

    state = {
        "block_entries": False,
        "force_close": False,
        "reason": ""
    }

    if weekday == 4:
        if now_min >= entry_cutoff_min and now_min < close_min:
            state["block_entries"] = True
            state["reason"] = "Fim de pregão (Sexta): novas entradas bloqueadas"
        if now_min >= force_close_min and now_min < close_min:
            state["block_entries"] = True
            state["force_close"] = True
            state["reason"] = "Fim de pregão (Sexta): fechando posições para evitar gap"

    if weekday == 6:
        open_min = _parse_hhmm_to_minutes(getattr(config, 'SUNDAY_MARKET_OPEN_UTC', "21:00"))
        buffer_min = int(getattr(config, 'SUNDAY_OPEN_BUFFER_MINUTES', 30))
        if now_min >= open_min and now_min < (open_min + buffer_min):
            state["block_entries"] = True
            state["reason"] = "Abertura (Domingo): buffer de spread/liquidez (sem entradas)"

    return state

# ===========================
# SYMBOL INFO
# ===========================
def get_symbol_info(symbol: str) -> Optional[mt5.SymbolInfo]:
    info = mt5_exec(mt5.symbol_info, symbol)
    if info is None:
        logger.warning(f"⚠️ Símbolo {symbol} não encontrado ou indisponível.")
        return None
    if not info.visible:
        ok_sel = mt5_exec(mt5.symbol_select, symbol, True)
        if not ok_sel:
            logger.warning(f"⚠️ Símbolo {symbol} não visível e não pôde ser selecionado.")
            return None
        info = mt5_exec(mt5.symbol_info, symbol)
        if info is None:
            logger.warning(f"⚠️ Símbolo {symbol} ainda indisponível após seleção.")
            return None
    return info

def get_pip_size(symbol: str) -> float:
    """
    Retorna o tamanho do pip para um símbolo.
    ✅ CORRIGIDO: Usa info.point e ajusta para JPY.
    """
    info = get_symbol_info(symbol)
    if info:
        # info.point é o menor incremento de preço (tick size)
        # Para a maioria dos pares Forex, 1 pip = 10 pontos (ex: 0.0001 para EURUSD, onde point é 0.00001)
        # Para JPY, 1 pip = 1 ponto (ex: 0.01 para USDJPY, onde point é 0.001)
        # A definição de "pip" pode variar, mas "point" é o menor incremento de preço.
        if "JPY" in symbol.upper(): # Verifica se é um par JPY
            return info.point * 10 # Ex: 0.001 * 10 = 0.01 (1 pip para JPY)
        return info.point * 10 # Ex: 0.00001 * 10 = 0.0001 (1 pip para EURUSD)
    logger.warning(f"⚠️ Não foi possível obter info para {symbol} para calcular pip size. Usando 0.0001.")
    return 0.0001 # Default para a maioria dos pares

def get_tick_value(symbol: str) -> float:
    """
    Retorna o valor de 1 pip em moeda da conta para 1 lote padrão (100.000 unidades).
    ✅ CORRIGIDO: Usa trade_tick_value_profit e info.point para calcular o valor de 1 pip.
    """
    info = get_symbol_info(symbol)
    if not info:
        logger.warning(f"⚠️ Não foi possível obter info para {symbol}. Usando tick_value padrão de 1.0.")
        return 1.0 # Default seguro

    # info.trade_tick_value_profit é o valor de 1 tick em moeda da conta para 1 lote.
    # info.point é o tamanho de 1 tick.
    # get_pip_size(symbol) retorna o tamanho de 1 pip (ex: 0.0001 para EURUSD).

    # Valor de 1 pip em moeda da conta para 1 lote = (tamanho de 1 pip / tamanho de 1 tick) * valor de 1 tick
    if info.point == 0:
        logger.warning(f"⚠️ info.point é zero para {symbol}. Não é possível calcular pip value. Usando 1.0.")
        return 1.0

    pip_value_per_lot = (get_pip_size(symbol) / info.point) * info.trade_tick_value_profit

    if pip_value_per_lot <= 0:
        logger.warning(f"⚠️ Pip value calculado é zero ou negativo para {symbol}. Usando 1.0.")
        return 1.0

    return pip_value_per_lot

# ===========================
# DATA RETRIEVAL
# ===========================
def get_rates(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    try:
        symbol_info = mt5_exec(mt5.symbol_info, symbol)
        if symbol_info is None:
            if symbol != "PANEL_UPDATE":
                logger.error(f"❌ Símbolo {symbol} não existe na plataforma MT5")
            return None
        ok_sel = mt5_exec(mt5.symbol_select, symbol, True)
        if not ok_sel:
            if symbol != "PANEL_UPDATE":
                logger.error(f"❌ Falha ao selecionar {symbol} no Market Watch")
            return None
        logger.debug(f"🔍 Buscando {count} velas de {symbol} no timeframe {timeframe}")
        rates = mt5_exec(mt5.copy_rates_from_pos, symbol, timeframe, 1, count)
        if rates is None or len(rates) == 0:
            logger.debug(f"⚠️ Tentativa com start_pos=1 falhou, tentando start_pos=0")
            rates = mt5_exec(mt5.copy_rates_from_pos, symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            if err[0] != 1:
                logger.warning(f"⚠️ Falha ao obter rates para {symbol} (TF:{timeframe}, Count:{count}): Código={err[0]}, Msg={err[1]}")
            else:
                logger.warning(f"⚠️ {symbol}: Nenhum dado histórico disponível (rates vazias)")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logger.error(f"❌ Erro crítico em get_rates ({symbol}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def safe_copy_rates(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """
    Função segura para obter rates, usada no backtest do ML Optimizer.
    Garante conexão e tratamento de erros.
    """
    if not ensure_mt5_connection():
        logger.error("❌ MT5 não conectado para safe_copy_rates.")
        return None
    return get_rates(symbol, timeframe, count)

# ===========================
# INDICATORS (v4.2 - PATCH 1)
# ===========================
def get_indicators_forex(
    symbol: str,
    ema_short: int = None,
    ema_long: int = None,
    rsi_period: int = None, # ✅ NOVO: Período RSI otimizado
    rsi_low: int = None,
    rsi_high: int = None,
    adx_period: int = None, # ✅ NOVO: Período ADX otimizado
    bb_period: int = None,  # ✅ NOVO: Período BB otimizado
    bb_dev: float = None    # ✅ NOVO: Desvio BB otimizado
) -> dict:
    """
    ✅ v4.2: Calcula indicadores técnicos para um símbolo.
    Aceita parâmetros otimizados para EMA, RSI, ADX e Bollinger Bands.
    """
    # Usa parâmetros otimizados se fornecidos, senão usa config padrão
    ema_short_period = ema_short if ema_short is not None else config.EMA_SHORT_PERIOD
    ema_long_period = ema_long if ema_long is not None else config.EMA_LONG_PERIOD
    rsi_calc_period = rsi_period if rsi_period is not None else config.RSI_PERIOD
    rsi_low_limit = rsi_low if rsi_low is not None else config.RSI_LOW_LIMIT
    rsi_high_limit = rsi_high if rsi_high is not None else config.RSI_HIGH_LIMIT
    adx_calc_period = adx_period if adx_period is not None else config.ADX_PERIOD
    bb_calc_period = bb_period if bb_period is not None else config.BB_PERIOD
    bb_calc_dev = bb_dev if bb_dev is not None else config.BB_DEVIATION

    # ✅ REQUISITO: Validação de símbolo no início
    real_symbol = normalize_symbol(symbol)
    ok_sel = mt5_exec(mt5.symbol_select, real_symbol, True)
    if not ok_sel:
            logger.error(f"❌ Símbolo {symbol}/{real_symbol} não encontrado no Market Watch")
            return {"error": True, "message": f"Symbol {symbol} not found in MT5"}

    timeframe = mt5.TIMEFRAME_M15
    last_ts = None
    try:
        r = mt5_exec(mt5.copy_rates_from_pos, real_symbol, timeframe, 0, 1)
        if r is not None and len(r) > 0:
            last_ts = int(r[0]['time'])
    except Exception:
        last_ts = None
    if last_ts is not None:
        cache_key = f"{real_symbol}:{last_ts}:{ema_short_period}:{ema_long_period}:{rsi_calc_period}:{adx_calc_period}:{bb_calc_period}:{bb_calc_dev}"
        c = INDICATOR_CACHE.get(cache_key)
        if c:
            return c["data"]
    # Precisamos de dados suficientes para todos os indicadores
    max_period = max(ema_long_period, rsi_calc_period, adx_calc_period, bb_calc_period)
    df = get_rates(real_symbol, timeframe, max_period + 50)
    
    # ✅ REQUISITO: Nunca retornar None, retornar dicionário de erro
    if df is None or df.empty or len(df) < max_period:
        return {"error": True, "message": f"Dados insuficientes para {real_symbol}"}

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema_short_arr = ema_numba(close, ema_short_period)
    ema_long_arr = ema_numba(close, ema_long_period)
    df['ema_short'] = pd.Series(ema_short_arr)
    df['ema_long'] = pd.Series(ema_long_arr)
    ema_trend = "UP" if df['ema_short'].iloc[-1] > df['ema_long'].iloc[-1] else "DOWN"

    rsi_arr = calculate_rsi_numba(close, rsi_calc_period)
    df['rsi'] = pd.Series(rsi_arr)
    rsi_now = df['rsi'].iloc[-1]

    adx_arr = calculate_adx_numba(high, low, close, adx_calc_period)
    df['adx'] = pd.Series(adx_arr)
    adx_now = df['adx'].iloc[-1]

    atr_arr = calculate_atr_numba(high, low, close, adx_calc_period)
    df['atr'] = pd.Series(atr_arr)
    atr_now = df['atr'].iloc[-1]
    pip_size = get_pip_size(symbol)
    atr_pips = atr_now / pip_size if pip_size > 0 else 0
    try:
        atr20_arr = calculate_atr_numba(high, low, close, 20)
        atr20 = float(atr20_arr[-1]) if len(atr20_arr) else 0.0
    except Exception:
        atr20 = 0.0

    # Bollinger Bands
    df['sma_bb'] = df['close'].rolling(window=bb_calc_period).mean()
    df['std_bb'] = df['close'].rolling(window=bb_calc_period).std()
    df['bb_upper'] = df['sma_bb'] + (df['std_bb'] * bb_calc_dev)
    df['bb_lower'] = df['sma_bb'] - (df['std_bb'] * bb_calc_dev)
    bb_upper = df['bb_upper'].iloc[-1]
    bb_lower = df['bb_lower'].iloc[-1]
    bb_width = (bb_upper - bb_lower) / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0
    # MTF EMA200 H4 (estrita)
    ema200_h4 = None
    try:
        if getattr(config, 'ENABLE_MULTI_TIMEFRAME', True):
            rates_h4 = get_rates(real_symbol, mt5.TIMEFRAME_H4, getattr(config, 'MULTI_TF_EMA_PERIOD', 200) + 5)
            if rates_h4 is not None and not rates_h4.empty:
                ema200_h4 = float(rates_h4['close'].ewm(span=getattr(config, 'MULTI_TF_EMA_PERIOD', 200), adjust=False).mean().iloc[-1])
    except Exception:
        ema200_h4 = None

    # MACD (12, 26, 9) - Implementação Manual para evitar dependência de TA-Lib
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    macd_val = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    macd_hist = df['macd_hist'].iloc[-1]

    # Spread e Volume
    info = get_symbol_info(symbol)
    spread_points = info.spread if info else 0
    spread_pips = (spread_points * info.point) / pip_size if info and pip_size > 0 else 0
    # Verifica se 'tick_volume' existe antes de usar
    volume_ratio = df['tick_volume'].iloc[-1] / df['tick_volume'].mean() if 'tick_volume' in df.columns and df['tick_volume'].mean() > 0 else 0
    
    # ✅ REQUISITO: Flexibilização Inteligente de Spreads por Categoria
    symbol_upper = symbol.upper()
    if any(idx in symbol_upper for idx in ["US30", "NAS100", "USTEC", "DE40", "GER40", "UK100", "US500", "USA500"]):
        max_spread = getattr(config, 'MAX_SPREAD_INDICES', 150)
        spread_ok = spread_points <= max_spread
    elif any(met in symbol_upper for met in ["XAU", "XAG", "GOLD", "SILVER"]):
        max_spread = getattr(config, 'MAX_SPREAD_METALS', 60)
        spread_ok = spread_points <= max_spread
    elif any(crypto in symbol_upper for crypto in ["BTC", "ETH", "SOL", "ADA", "BNB"]):
        max_spread = getattr(config, 'MAX_SPREAD_CRYPTO', 2500)
        spread_ok = spread_points <= max_spread
    else:
        max_spread = getattr(config, 'MAX_SPREAD_FOREX', 25)
        spread_ok = spread_points <= max_spread

    out = {
        "time": df['time'].iloc[-1],
        "open": df['open'].iloc[-1],
        "high": df['high'].iloc[-1],
        "low": df['low'].iloc[-1],
        "close": df['close'].iloc[-1],
        "df": df, # ✅ NOVO: DataFrame completo para ML Optimizer
        "ema_short": df['ema_short'].iloc[-1],
        "ema_long": df['ema_long'].iloc[-1],
        "ema_trend": ema_trend,
        "rsi": rsi_now,
        "adx": adx_now,
        "atr": atr_now,
        "atr_pips": atr_pips,
        "atr20": atr20,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "ema200_h4": ema200_h4,
        "spread_points": spread_points,
        "spread_pips": spread_pips,
        "volume_ratio": volume_ratio,
        "spread_ok": spread_ok,
        "rsi_low_limit": rsi_low_limit,
        "rsi_high_limit": rsi_high_limit,
        "rsi_high_limit": rsi_high_limit,
        "current_price": df['close'].iloc[-1], # Adiciona para facilitar o cálculo de SL/TP
        "macd": macd_val,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist
    }
    try:
        last_ts2 = int(df['time'].iloc[-1].timestamp())
        cache_key = f"{real_symbol}:{last_ts2}:{ema_short_period}:{ema_long_period}:{rsi_calc_period}:{adx_calc_period}:{bb_calc_period}:{bb_calc_dev}"
        INDICATOR_CACHE[cache_key] = {"data": out}
    except Exception:
        pass
    return out
def get_volatility_regime(symbol: str, df: Optional[pd.DataFrame] = None) -> str:
    try:
        real_symbol = normalize_symbol(symbol)
        if df is None:
            df = get_rates(real_symbol, mt5.TIMEFRAME_M15, 120)
        if df is None or df.empty or len(df) < 100:
            return "NORMAL"
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        atr_short = calculate_atr_numba(high, low, close, 14)
        atr_long = calculate_atr_numba(high, low, close, 100)
        a_s = float(atr_short[-1]) if len(atr_short) else 0.0
        a_l = float(atr_long[-1]) if len(atr_long) else 0.0
        if a_s <= 0 or a_l <= 0:
            return "NORMAL"
        if a_s > 1.5 * a_l:
            return "HIGH"
        if a_s < 0.7 * a_l:
            return "LOW"
        return "NORMAL"
    except Exception:
        return "NORMAL"
def sync_market_watch(required_symbols):
    """
    Remove todos os ativos desnecessários e adiciona apenas os que o bot usa.
    """
    try:
        # 1. Pega todos os símbolos que estão ATUALMENTE no Market Watch
        current_symbols = mt5_exec(mt5.symbols_get)
        current_names = [s.name for s in current_symbols] if current_symbols else []

        required_raw = list(required_symbols or [])
        required_resolved = []
        missing = []
        for s in required_raw:
            if not s:
                continue
            resolved = normalize_symbol(str(s))
            if resolved and resolved != s:
                logger.info(f"🔁 Market Watch: '{s}' -> '{resolved}'")
            ok_select = mt5_exec(mt5.symbol_select, resolved, True)
            if not ok_select:
                missing.append(str(s))
                continue
            required_resolved.append(resolved)

        required_set = set(required_resolved)
        logger.info(f"🔄 Sincronizando Market Watch para {len(required_set)} ativos...")

        # 2. Adiciona os ativos necessários primeiro
        kept = []
        for symbol in required_resolved:
            ok_select = mt5_exec(mt5.symbol_select, symbol, True)
            if ok_select:
                kept.append(symbol)

        # 3. Remove os que NÃO estão na lista (opcional, mas limpa o terminal)
        removed = []
        for s_name in current_names:
            if s_name not in required_set:
                # Tenta remover (só funciona se não houver gráfico aberto do ativo)
                ok = mt5_exec(mt5.symbol_select, s_name, False)
                if ok:
                    removed.append(s_name)
        
        logger.info(f"✅ Market Watch sincronizado | kept={len(kept)} removed={len(removed)} missing={len(missing)}")
        return {"ok": True, "kept": kept, "removed": removed, "missing": missing}
    except Exception as e:
        logger.error(f"❌ Erro na sincronização do Market Watch: {e}")
        return {"ok": False, "error": str(e), "kept": [], "removed": [], "missing": []}
# ===========================
# POSITION SIZING
# ===========================
def calculate_position_size_atr_forex(symbol: str, price: float, atr_pips: float, sl_atr_mult: float = 2.0, risk_multiplier: float = 1.0) -> float:
    """
    Calcula o volume da posição baseado no ATR e no risco por trade.
    ✅ v5.1: Gestão de Money Management Cirúrgica (Risco Exato com base no SL)
    """
    account_info = mt5_exec(mt5.account_info)
    if not account_info:
        logger.error("❌ Não foi possível obter informações da conta para calcular o volume.")
        return getattr(config, 'DEFAULT_LOT', 0.01)

    equity = account_info.equity
    balance = account_info.balance
    
    # --- GESTÃO DE RISCO DINÂMICA v5.0 ---
    # 1. Verifica perda diária acumulada
    daily_profit = equity - balance 
    daily_loss_pct = abs(min(0, daily_profit)) / balance if balance > 0 else 0
    
    # Bloqueio Total (Equity Guard)
    max_loss = getattr(config, 'MAX_DAILY_LOSS_PCT', 0.02)
    if daily_loss_pct >= max_loss:
        logger.warning(f"🚨 EQUITY GUARD: Perda diária ({daily_loss_pct:.2%}) atingiu limite ({max_loss:.2%})")
        return 0.0

    # Redução de Risco em Drawdown (Soft Guard)
    risk_pct = config.RISK_PER_TRADE_PCT
    if getattr(config, 'REDUCE_RISK_ON_DD', True) and daily_loss_pct > 0.012:
        risk_pct *= 0.5
        logger.info(f"🛡️ RISCO REDUZIDO (-50%): Drawdown atual de {daily_loss_pct:.2%}")
    
    try:
        session_name = (get_current_trading_session() or {}).get("name")
        mult_map = getattr(config, "SESSION_RISK_MULTIPLIERS", {})
        if isinstance(mult_map, dict) and session_name in mult_map:
            risk_pct *= float(mult_map.get(session_name, 1.0))
    except Exception:
        pass

    try:
        rm = float(risk_multiplier)
        if rm > 0:
            risk_pct *= rm
    except Exception:
        pass

    # -------------------------------------

    risk_per_trade_usd = equity * risk_pct

    pip_value_per_lot = get_tick_value(symbol)

    if pip_value_per_lot <= 0:
        logger.warning(f"⚠️ Valor do pip por lote é zero ou negativo para {symbol}. Usando volume p/ fallback.")
        return getattr(config, 'DEFAULT_LOT', 0.01)

    # ✅ CORREÇÃO: Cálculo baseado na distância REAL do SL (ATR * Multiplicador)
    sl_pips = atr_pips * sl_atr_mult
    if sl_pips <= 0:
        logger.warning(f"⚠️ SL Pips calculado é zero ({atr_pips} * {sl_atr_mult}). Usando volume p/ fallback.")
        return getattr(config, 'DEFAULT_LOT', 0.01)

    # Volume = (Risco em $) / (Distância SL em pips * Valor do Pip por Lote)
    # Ex: $1000 / (20 pips * $10/pip) = 5 lotes
    volume = risk_per_trade_usd / (sl_pips * pip_value_per_lot)

    # Limita volume ao mínimo/máximo permitido
    volume = max(config.MIN_VOLUME, min(config.MAX_VOLUME, volume))

    # ✅ SAFETY CAP: Proteção absoluta contra lotes gigantes
    if volume > 0.50:
        logger.critical(f"🚨 VOLUME SAFETY CAP: Lote calculado ({volume}) excedeu 0.50. Forçando 0.01.")
        return 0.01

    # Arredonda para o step do volume
    symbol_info = get_symbol_info(symbol)
    step = symbol_info.volume_step if symbol_info else 0.01

    try:
        vol_dec = Decimal(str(volume))
        step_dec = Decimal(str(step))
        units = (vol_dec / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        volume = float((units * step_dec))
    except Exception:
        volume = round(volume / step) * step

    return volume

# ===========================
# DYNAMIC LOT CALCULATION v5.0
# ===========================
def calculate_dynamic_lot(symbol: str, risk_percent: float, stop_loss_pips: float) -> float:
    """
    ✅ v5.0: Calcula tamanho do lote baseado estritamente no % de risco do Equity.
    Fórmula: (Equity * Risco%) / (Distância_SL_Pips * Valor_do_Pip_por_Lote)
    
    Args:
        symbol: Par de moedas
        risk_percent: Porcentagem de risco (ex: 0.01 = 1%)
        stop_loss_pips: Distância do Stop Loss em pips
        
    Returns:
        float: Tamanho do lote calculado
    """
    try:
        with mt5_lock:
            account_info = mt5.account_info()
        
        if not account_info:
            logger.error("❌ calculate_dynamic_lot: Não foi possível obter informações da conta.")
            return getattr(config, 'MIN_VOLUME', 0.01)
        
        equity = account_info.equity
        
        if equity <= 0 or stop_loss_pips <= 0 or risk_percent <= 0:
            logger.warning(f"⚠️ calculate_dynamic_lot: Parâmetros inválidos (Equity={equity}, SL_Pips={stop_loss_pips}, Risk={risk_percent})")
            return getattr(config, 'MIN_VOLUME', 0.01)
        
        # Calcula valor de risco em $
        risk_amount = equity * risk_percent
        
        # Obtém valor do pip por lote padrão
        pip_value_per_lot = get_tick_value(symbol)
        
        if pip_value_per_lot <= 0:
            logger.warning(f"⚠️ calculate_dynamic_lot: Pip value inválido para {symbol}")
            return getattr(config, 'MIN_VOLUME', 0.01)
        
        # Fórmula: Lote = Risco$ / (SL_Pips * Valor_Pip_por_Lote)
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Aplica limites de segurança
        min_vol = getattr(config, 'MIN_VOLUME', 0.01)
        max_vol = getattr(config, 'MAX_VOLUME', 0.50)
        lot_size = max(min_vol, min(max_vol, lot_size))
        
        # Arredonda para o step do volume
        symbol_info = get_symbol_info(symbol)
        if symbol_info:
            step = symbol_info.volume_step
            try:
                lot_dec = Decimal(str(lot_size))
                step_dec = Decimal(str(step))
                units = (lot_dec / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                lot_size = float((units * step_dec))
            except Exception:
                lot_size = round(lot_size / step) * step
        
        logger.debug(f"📊 calculate_dynamic_lot: {symbol} | Equity={equity:.2f} | Risk={risk_percent:.2%} | SL_Pips={stop_loss_pips:.1f} | Lot={lot_size:.2f}")
        
        return lot_size
        
    except Exception as e:
        logger.error(f"❌ calculate_dynamic_lot: Erro crítico - {e}")
        return getattr(config, 'MIN_VOLUME', 0.01)

# ===========================
# EMA 200 MACRO FILTER v5.0
# ===========================
def get_ema_200(symbol: str, timeframe: int = None) -> dict:
    """
    ✅ v5.0: Calcula EMA 200 para filtro de tendência macro.
    
    Args:
        symbol: Par de moedas
        timeframe: Timeframe MT5 (default: H1)
        
    Returns:
        dict: {ema_200, current_price, is_above_ema, trend_direction, error}
    """
    try:
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
        
        ema_period = getattr(config, 'EMA_200_PERIOD', 200)
        
        # Normaliza símbolo
        real_symbol = normalize_symbol(symbol)
        
        # Obtém dados históricos
        df = get_rates(real_symbol, timeframe, ema_period + 50)
        
        if df is None or df.empty or len(df) < ema_period:
            return {"error": True, "message": f"Dados insuficientes para EMA 200 de {symbol}"}
        
        # Calcula EMA 200
        df['ema_200'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        ema_200_value = df['ema_200'].iloc[-1]
        current_price = df['close'].iloc[-1]
        is_above_ema = current_price > ema_200_value
        
        # Determina direção da tendência
        if is_above_ema:
            trend_direction = "BULLISH"
        else:
            trend_direction = "BEARISH"
        
        return {
            "error": False,
            "ema_200": ema_200_value,
            "current_price": current_price,
            "is_above_ema": is_above_ema,
            "trend_direction": trend_direction,
            "distance_pips": abs(current_price - ema_200_value) / get_pip_size(symbol)
        }
        
    except Exception as e:
        logger.error(f"❌ get_ema_200: Erro ao calcular para {symbol} - {e}")
        return {"error": True, "message": str(e)}

# ===========================
# ROLLOVER PROTECTION v5.0
# ===========================
def is_rollover_period() -> tuple:
    """
    ✅ v5.0: Verifica se está no período de rollover bancário (16:55-18:05 NY).
    
    Returns:
        tuple: (is_blocked: bool, reason: str)
    """
    try:
        if not getattr(config, 'ENABLE_ROLLOVER_BLOCK', True):
            return False, "Rollover block disabled"
        
        # Obtém horário NY (UTC-5 / UTC-4 DST)
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        current_time_str = now_ny.strftime("%H:%M")
        
        rollover_start = getattr(config, 'ROLLOVER_BLOCK_START', "16:55")
        rollover_end = getattr(config, 'ROLLOVER_BLOCK_END', "18:05")
        
        # Verifica se está no período
        if rollover_start <= current_time_str <= rollover_end:
            return True, f"Período de Rollover Bancário ({rollover_start}-{rollover_end} NY)"
        
        return False, "Fora do período de rollover"
        
    except Exception as e:
        logger.error(f"❌ is_rollover_period: Erro - {e}")
        return False, f"Erro: {e}"

# ===========================
# TELEGRAM INTEGRATION
# ===========================
def get_telegram_credentials() -> tuple:
    try:
        bot_token = str(getattr(config, 'TELEGRAM_BOT_TOKEN', '')).strip()
        chat_id = str(getattr(config, 'TELEGRAM_CHAT_ID', '')).strip()
        if not bot_token or not chat_id:
            bot_token = str(getattr(config, 'TELEGRAM_BOT_TOKEN_OVERRIDE', bot_token)).strip()
            chat_id = str(getattr(config, 'TELEGRAM_CHAT_ID_OVERRIDE', chat_id)).strip()
        if not bot_token or not chat_id:
            try:
                creds_path = Path(getattr(config, 'TELEGRAM_CREDENTIALS_FILE', 'data/telegram.json'))
                if creds_path.exists():
                    data = json.loads(creds_path.read_text(encoding='utf-8'))
                    bot_token = str(data.get('bot_token', bot_token)).strip()
                    chat_id = str(data.get('chat_id', chat_id)).strip()
            except Exception:
                pass
        return (bot_token if bot_token else None, chat_id if chat_id else None)
    except Exception:
        return (None, None)
def send_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    """
    Envia mensagem via Telegram Bot.
    
    Args:
        message: Mensagem a ser enviada (suporta HTML)
        parse_mode: Modo de parse (HTML ou Markdown)
        
    Returns:
        bool: True se enviado com sucesso
    """
    try:
        import requests
        
        bot_token, chat_id = get_telegram_credentials()
        
        if not bot_token or not chat_id:
            logger.warning("⚠️ Telegram não configurado (TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID ausentes no config)")
            return False
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.debug(f"✅ Mensagem Telegram enviada com sucesso")
            return True
        else:
            logger.error(f"❌ Erro ao enviar Telegram: {response.status_code} - {response.text}")
            return False
            
    except ImportError:
        logger.error("❌ Biblioteca 'requests' não instalada. Execute: pip install requests")
        return False
    except Exception as e:
        logger.error(f"❌ send_telegram_message: Erro - {e}")
        return False

def send_telegram_alert(message: str, level: str = "INFO") -> bool:
    """
    Envia alerta via Telegram com formatação especial.
    
    Args:
        message: Mensagem de alerta
        level: Nível do alerta (INFO, WARNING, ERROR)
        
    Returns:
        bool: True se enviado com sucesso
    """
    try:
        # Emojis por nível
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "SUCCESS": "✅"
        }
        
        emoji = emoji_map.get(level.upper(), "📢")
        
        # Formata mensagem com header
        formatted_message = f"{emoji} <b>{level.upper()}</b>\n\n{message}"
        
        return send_telegram_message(formatted_message)
        
    except Exception as e:
        logger.error(f"❌ send_telegram_alert: Erro - {e}")
        return False

def send_telegram_document(file_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
    try:
        import requests

        bot_token, chat_id = get_telegram_credentials()

        if not bot_token or not chat_id:
            return False

        if not file_path or not os.path.exists(file_path):
            return False

        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            response = requests.post(url, data=data, files=files, timeout=30)

        return response.status_code == 200
    except Exception:
        return False

def set_telegram_chat_id(chat_id: str) -> None:
    try:
        creds_path = Path(getattr(config, 'TELEGRAM_CREDENTIALS_FILE', 'data/telegram.json'))
        creds_path.parent.mkdir(exist_ok=True)
        payload = {}
        if creds_path.exists():
            try:
                payload = json.loads(creds_path.read_text(encoding='utf-8'))
            except Exception:
                payload = {}
        payload['chat_id'] = str(chat_id)
        # Mantém token existente
        bot_token, _ = get_telegram_credentials()
        if bot_token:
            payload['bot_token'] = bot_token
        creds_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info(f"✅ Telegram chat_id salvo em {creds_path}")
    except Exception as e:
        logger.error(f"❌ Falha ao salvar chat_id do Telegram: {e}")

def send_telegram_message_to(chat_id: str, message: str, parse_mode: str = "HTML") -> bool:
    try:
        import requests
        bot_token, _ = get_telegram_credentials()
        if not bot_token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": str(chat_id), "text": message, "parse_mode": parse_mode, "disable_web_page_preview": True}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def export_bot_trades_csv(date_str: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        day = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None, {"error": "data inválida"}

    if not check_mt5_connection():
        if not ensure_mt5_connection():
            return None, {"error": "MT5 desconectado"}

    from_date = day.replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = from_date + timedelta(days=1)
    magic = getattr(config, 'MAGIC_NUMBER', 123456)

    with mt5_lock:
        deals = mt5.history_deals_get(from_date, to_date)

    rows = []
    total_profit = 0.0
    total_loss = 0.0
    wins = 0
    losses = 0

    if deals:
        for d in deals:
            try:
                if getattr(d, "magic", None) != magic:
                    continue
                if getattr(d, "type", None) not in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
                    continue
                entry = getattr(d, "entry", None)
                if entry is not None and entry != mt5.DEAL_ENTRY_OUT:
                    continue
                profit = float(getattr(d, "profit", 0.0))
                if profit > 0:
                    wins += 1
                    total_profit += profit
                elif profit < 0:
                    losses += 1
                    total_loss += profit
                ts = datetime.fromtimestamp(getattr(d, "time", 0))
                side = "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL"
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": getattr(d, "symbol", ""),
                    "side": side,
                    "volume": float(getattr(d, "volume", 0.0)),
                    "price": float(getattr(d, "price", 0.0)),
                    "profit": profit,
                    "commission": float(getattr(d, "commission", 0.0)),
                    "swap": float(getattr(d, "swap", 0.0)),
                    "comment": str(getattr(d, "comment", "")),
                    "order": str(getattr(d, "order", "")),
                    "position_id": str(getattr(d, "position_id", "")),
                })
            except Exception:
                continue

    out_dir = Path("analysis_logs")
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir / f"trades_{date_str}.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "symbol", "side", "volume", "price",
            "profit", "commission", "swap", "comment", "order", "position_id"
        ])
        writer.writeheader()
        writer.writerows(rows)

    total = wins + losses
    pnl = total_profit + total_loss
    pf = (total_profit / abs(total_loss)) if total_loss != 0 else 0.0
    wr = (wins / total * 100) if total > 0 else 0.0

    summary = {
        "date": date_str,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "profit_factor": pf,
        "pnl": pnl,
        "file": out_path
    }
    return out_path, summary
def export_bot_trades_txt(date_str: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None, {"error": "data inválida"}
    if not check_mt5_connection():
        if not ensure_mt5_connection():
            return None, {"error": "MT5 desconectado"}
    from_date = day.replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = from_date + timedelta(days=1)
    magic = getattr(config, 'MAGIC_NUMBER', 123456)
    with mt5_lock:
        deals = mt5.history_deals_get(from_date, to_date)
    lines = []
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    if deals:
        for d in deals:
            try:
                if getattr(d, "magic", None) != magic:
                    continue
                if getattr(d, "type", None) not in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
                    continue
                entry = getattr(d, "entry", None)
                if entry is not None and entry != mt5.DEAL_ENTRY_OUT:
                    continue
                ts = datetime.fromtimestamp(getattr(d, "time", 0)).strftime("%Y-%m-%d %H:%M:%S")
                sym = str(getattr(d, "symbol", ""))
                side = "BUY" if getattr(d, "type", None) == mt5.DEAL_TYPE_BUY else "SELL"
                vol = float(getattr(d, "volume", 0.0))
                price = float(getattr(d, "price", 0.0))
                profit = float(getattr(d, "profit", 0.0))
                commission = float(getattr(d, "commission", 0.0))
                swap = float(getattr(d, "swap", 0.0))
                order_id = str(getattr(d, "order", ""))
                position_id = str(getattr(d, "position_id", ""))
                comment = str(getattr(d, "comment", ""))
                if profit > 0:
                    wins += 1
                    total_profit += profit
                elif profit < 0:
                    losses += 1
                    total_loss += profit
                line = (
                    f"{ts} | {sym} | {side} {vol:.2f} | "
                    f"price={price:.5f} | profit={profit:+.2f} | "
                    f"commission={commission:.2f} | swap={swap:.2f} | "
                    f"order={order_id} | position_id={position_id} | comment={comment}"
                )
                lines.append(line)
            except Exception:
                continue
    out_dir = Path("analysis_logs")
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir / f"trades_{date_str}.txt")
    header = [
        f"Trades do Dia ({date_str})",
        "-" * 64
    ]
    summary_lines = []
    total = wins + losses
    pnl = total_profit + total_loss
    pf = (total_profit / abs(total_loss)) if total_loss != 0 else 0.0
    wr = (wins / total * 100) if total > 0 else 0.0
    summary_lines.append(f"Total: {total} | Wins: {wins} | Losses: {losses}")
    summary_lines.append(f"WR: {wr:.1f}% | PF: {pf:.2f} | PnL: ${pnl:+.2f}")
    summary_lines.append("-" * 64)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for l in header:
                f.write(l + "\n")
            for l in summary_lines:
                f.write(l + "\n")
            for l in lines:
                f.write(l + "\n")
    except Exception:
        return None, {"error": "falha ao salvar arquivo"}
    summary = {
        "date": date_str,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "profit_factor": pf,
        "pnl": pnl,
        "file": out_path
    }
    return out_path, summary
def _get_db_path() -> str:
    base = Path(getattr(config, "DATA_DIR", "data"))
    base.mkdir(exist_ok=True)
    return str(base / "trades.db")

def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_trade_db() -> None:
    conn = _get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS trades ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "order_id INTEGER,"
        "ticket INTEGER,"
        "symbol TEXT,"
        "side TEXT,"
        "volume REAL,"
        "open_time TEXT,"
        "close_time TEXT,"
        "open_price REAL,"
        "close_price REAL,"
        "sl REAL,"
        "tp REAL,"
        "commission REAL,"
        "swap REAL,"
        "profit REAL,"
        "magic INTEGER,"
        "comment TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS open_positions ("
        "order_id INTEGER,"
        "ticket INTEGER,"
        "symbol TEXT,"
        "side TEXT,"
        "volume REAL,"
        "open_time TEXT,"
        "open_price REAL,"
        "sl REAL,"
        "tp REAL,"
        "magic INTEGER,"
        "comment TEXT)"
    )
    conn.commit()
    conn.close()

def record_order_open(symbol: str, side: str, volume: float, entry_price: float, sl: float, tp: float, order_id: int, comment: str = "") -> None:
    try:
        init_trade_db()
        conn = _get_db_conn()
        cur = conn.cursor()
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO open_positions (order_id, ticket, symbol, side, volume, open_time, open_price, sl, tp, magic, comment) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(order_id) if order_id else None,
                None,
                str(symbol or ""),
                str(side or ""),
                float(volume or 0.0),
                now_ts,
                float(entry_price or 0.0),
                float(sl or 0.0),
                float(tp or 0.0),
                int(getattr(config, "MAGIC_NUMBER", 123456)),
                str(comment or "")
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"erro record_order_open: {e}")

def record_trade_close(ticket: int, symbol: str, side: str, volume: float, open_time: datetime, close_time: datetime, open_price: float, close_price: float, sl: float, tp: float, profit: float, commission: float, swap: float, magic: int, comment: str = "") -> None:
    try:
        init_trade_db()
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO trades (order_id, ticket, symbol, side, volume, open_time, close_time, open_price, close_price, sl, tp, commission, swap, profit, magic, comment) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                None,
                int(ticket) if ticket else None,
                str(symbol or ""),
                str(side or ""),
                float(volume or 0.0),
                (open_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(open_time, datetime) else str(open_time or "")),
                (close_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(close_time, datetime) else str(close_time or "")),
                float(open_price or 0.0),
                float(close_price or 0.0),
                float(sl or 0.0),
                float(tp or 0.0),
                float(commission or 0.0),
                float(swap or 0.0),
                float(profit or 0.0),
                int(magic or getattr(config, "MAGIC_NUMBER", 123456)),
                str(comment or "")
            )
        )
        try:
            cur.execute("DELETE FROM open_positions WHERE ticket = ? OR (symbol = ? AND ABS(open_price - ?) < 1e-6)", (int(ticket or 0), str(symbol or ""), float(open_price or 0.0)))
        except Exception:
            pass
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"erro record_trade_close: {e}")

def sync_mt5_trades_to_db() -> Dict[str, Any]:
    summary = {"inserted": 0, "checked": 0}
    try:
        init_trade_db()
        conn = _get_db_conn()
        cur = conn.cursor()
        try:
            if not check_mt5_connection():
                ensure_mt5_connection()
        except Exception:
            pass
        day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        with mt5_lock:
            deals = mt5.history_deals_get(day, datetime.now())
        magic = getattr(config, "MAGIC_NUMBER", 123456)
        if deals:
            for d in deals:
                summary["checked"] += 1
                try:
                    if getattr(d, "magic", None) != magic:
                        continue
                    if getattr(d, "entry", None) != mt5.DEAL_ENTRY_OUT:
                        continue
                    ts = datetime.fromtimestamp(getattr(d, "time", 0))
                    symbol = str(getattr(d, "symbol", ""))
                    side = "BUY" if getattr(d, "type", None) == mt5.DEAL_TYPE_BUY else "SELL"
                    volume = float(getattr(d, "volume", 0.0))
                    price = float(getattr(d, "price", 0.0))
                    profit = float(getattr(d, "profit", 0.0))
                    commission = float(getattr(d, "commission", 0.0))
                    swap = float(getattr(d, "swap", 0.0))
                    order_id = int(getattr(d, "order", 0))
                    position_id = int(getattr(d, "position_id", 0))
                    cur.execute("SELECT COUNT(1) FROM trades WHERE ticket = ? AND close_time = ?", (position_id, ts.strftime("%Y-%m-%d %H:%M:%S")))
                    exists = cur.fetchone()[0] > 0
                    if exists:
                        continue
                    cur.execute(
                        "INSERT INTO trades (order_id, ticket, symbol, side, volume, open_time, close_time, open_price, close_price, sl, tp, commission, swap, profit, magic, comment) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            order_id,
                            position_id,
                            symbol,
                            side,
                            volume,
                            "",
                            ts.strftime("%Y-%m-%d %H:%M:%S"),
                            0.0,
                            price,
                            0.0,
                            0.0,
                            commission,
                            swap,
                            profit,
                            magic,
                            str(getattr(d, "comment", ""))
                        )
                    )
                    summary["inserted"] += 1
                except Exception:
                    continue
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"erro sync_mt5_trades_to_db: {e}")
    return summary

# ===========================
# RSI REVERSAL DETECTION v5.0
# ===========================
def detect_rsi_reversal(symbol: str, lookback: int = 5) -> dict:
    """
    ✅ v5.0: Detecta se RSI está "virando" (reversão de sobrecompra/sobrevenda).
    
    Regras:
    - BUY Reversal: RSI estava < 30 e agora está > 30 (saindo de sobrevenda)
    - SELL Reversal: RSI estava > 70 e agora está < 70 (saindo de sobrecompra)
    
    Args:
        symbol: Par de moedas
        lookback: Número de candles para verificar o histórico do RSI
        
    Returns:
        dict: {is_buy_reversal, is_sell_reversal, rsi_current, rsi_previous, error}
    """
    try:
        real_symbol = normalize_symbol(symbol)
        rsi_period = getattr(config, 'RSI_PERIOD', 14)
        
        # Precisa de dados extras para calcular RSI corretamente
        df = get_rates(real_symbol, mt5.TIMEFRAME_M15, rsi_period + lookback + 10)
        
        if df is None or df.empty or len(df) < rsi_period + lookback:
            return {"error": True, "message": f"Dados insuficientes para RSI reversal de {symbol}"}
        
        # Calcula RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        rsi_current = df['rsi'].iloc[-1]
        rsi_previous = df['rsi'].iloc[-2]
        
        # Busca nos últimos 'lookback' candles se estava em zona extrema
        rsi_lookback = df['rsi'].iloc[-(lookback+1):-1].values
        
        was_oversold = any(r < 30 for r in rsi_lookback)
        was_overbought = any(r > 70 for r in rsi_lookback)
        
        # Detecta reversões
        is_buy_reversal = was_oversold and rsi_current > 30 and rsi_previous <= 30
        is_sell_reversal = was_overbought and rsi_current < 70 and rsi_previous >= 70
        
        return {
            "error": False,
            "is_buy_reversal": is_buy_reversal,
            "is_sell_reversal": is_sell_reversal,
            "rsi_current": rsi_current,
            "rsi_previous": rsi_previous,
            "was_oversold": was_oversold,
            "was_overbought": was_overbought
        }
        
    except Exception as e:
        logger.error(f"❌ detect_rsi_reversal: Erro para {symbol} - {e}")
        return {"error": True, "message": str(e)}

# ===========================
# CORRELATION FILTER v5.1 (Land Trading)
# ===========================
def check_correlation(symbol: str) -> tuple:
    """
    ✅ v5.1: Verifica se há trades abertos em pares correlacionados.
    
    Args:
        symbol: Par a ser verificado
        
    Returns:
        tuple: (is_blocked: bool, reason: str, correlated_symbol: str)
    """
    try:
        correlations = getattr(config, 'SYMBOL_CORRELATIONS', {})
        max_correlation = getattr(config, 'CORRELATION_MAX', 0.80)
        
        if symbol not in correlations:
            return False, "Sem correlações configuradas", ""
        
        symbol_corrs = correlations[symbol]
        
        # Obtém posições abertas
        with mt5_lock:
            positions = mt5.positions_get()
        
        if not positions:
            return False, "Sem posições abertas", ""
        
        magic_number = getattr(config, 'MAGIC_NUMBER', 123456)
        
        for pos in positions:
            if pos.magic != magic_number:
                continue
            
            pos_symbol = pos.symbol
            
            # Verifica se o símbolo da posição está correlacionado
            if pos_symbol in symbol_corrs:
                corr_value = abs(symbol_corrs[pos_symbol])
                if corr_value >= max_correlation:
                    reason = f"Correlação alta com {pos_symbol} ({corr_value:.0%})"
                    logger.info(f"🔗 {symbol}: Bloqueado - {reason}")
                    return True, reason, pos_symbol
        
        return False, "Sem conflitos de correlação", ""
        
    except Exception as e:
        logger.error(f"❌ check_correlation: Erro para {symbol} - {e}")
        return False, f"Erro: {e}", ""

# ===========================
# VOLATILITY FILTER v5.1 (Land Trading)
# ===========================
def is_volatility_ok(symbol: str, indicators: dict = None) -> tuple:
    """
    ✅ v5.1: Verifica se ATR está acima do mínimo para operar.
    
    Args:
        symbol: Par de moedas
        indicators: Dict de indicadores (opcional, se não fornecido calcula)
        
    Returns:
        tuple: (is_ok: bool, reason: str, atr_value: float)
    """
    try:
        min_atr = getattr(config, 'VOLATILITY_FILTER_MIN_ATR', 0.0005)
        
        # Usa indicadores fornecidos ou calcula
        if indicators and 'atr' in indicators:
            atr_value = indicators.get('atr', 0)
        else:
            ind = get_indicators_forex(symbol)
            if ind.get('error'):
                return True, "Não foi possível calcular ATR", 0
            atr_value = ind.get('atr', 0)
        
        if atr_value < min_atr:
            reason = f"ATR muito baixo ({atr_value:.6f} < {min_atr})"
            return False, reason, atr_value
        
        return True, f"ATR OK ({atr_value:.6f})", atr_value
        
    except Exception as e:
        logger.error(f"❌ is_volatility_ok: Erro para {symbol} - {e}")
        return True, f"Erro: {e}", 0

# ===========================
# CANDLE CONFIRMATION v5.1 (Land Trading)
# ===========================
def is_candle_confirmed(symbol: str, signal: str) -> tuple:
    """
    ✅ v5.1: Verifica se o último candle confirma a direção do sinal.
    
    Args:
        symbol: Par de moedas
        signal: "BUY" ou "SELL"
        
    Returns:
        tuple: (is_confirmed: bool, reason: str)
    """
    try:
        if not getattr(config, 'CANDLE_CONFIRMATION_REQUIRED', True):
            return True, "Candle confirmation disabled"
        
        real_symbol = normalize_symbol(symbol)
        df = get_rates(real_symbol, mt5.TIMEFRAME_M15, 3)
        
        if df is None or len(df) < 2:
            return True, "Dados insuficientes para confirmação"
        
        # ✅ v5.3: Com start_pos=1 em get_rates, iloc[-1] é o candle fechado mais recente
        last_candle = df.iloc[-1]
        candle_time = last_candle['time']
        
        # ✅ v5.3: Invalidação por Timestamp (Máximo 2 timeframes de atraso)
        now = datetime.now() 
        # Candle time do MT5 geralmente é ingênuo (naive).
        # Se for naive, comparamos com datetime.now() (também naive local).
        if (now - candle_time).total_seconds() > 1800: # 30 min (2 candles de M15)
            return False, f"Sinal obsoleto: Candle de {candle_time.strftime('%H:%M')}"

        candle_close = last_candle['close']
        candle_open = last_candle['open']
        
        is_bullish = candle_close > candle_open
        is_bearish = candle_close < candle_open
        
        if signal == "BUY" and not is_bullish:
            return False, f"Candle não confirmou BUY (O:{candle_open:.5f} C:{candle_close:.5f})"
        
        if signal == "SELL" and not is_bearish:
            return False, f"Candle não confirmou SELL (O:{candle_open:.5f} C:{candle_close:.5f})"
        
        return True, f"Candle confirmado ({signal})"
        
    except Exception as e:
        logger.error(f"❌ is_candle_confirmed: Erro para {symbol} - {e}")
        return True, f"Erro: {e}"

# ===========================
# MULTI-TIMEFRAME TREND v5.2
# ===========================
def get_multi_timeframe_trend(symbol: str, signal: str) -> tuple:
    """
    ✅ v5.3: Verifica se macro timeframe (H4) confirma a direção do sinal.
    MODIFICADO: Retorna penalidade de score em vez de veto absoluto.
    
    Args:
        symbol: Par de moedas
        signal: "BUY" ou "SELL"
        
    Returns:
        tuple: (score_penalty: int, reason: str, h4_trend: str)
              score_penalty = 0 se alinhado, -15 se divergente
    """
    try:
        if not getattr(config, 'ENABLE_MULTI_TIMEFRAME', True):
            return 0, "Multi-timeframe disabled", "N/A"
        
        # Determina timeframe de confirmação
        tf_str = getattr(config, 'MULTI_TF_CONFIRMATION', "H4")
        tf_map = {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
        timeframe = tf_map.get(tf_str, mt5.TIMEFRAME_H4)
        
        ema_period = getattr(config, 'MULTI_TF_EMA_PERIOD', 200)
        
        # Obtém dados H4
        real_symbol = normalize_symbol(symbol)
        df = get_rates(real_symbol, timeframe, ema_period + 50)
        
        if df is None or df.empty or len(df) < ema_period:
            return 0, f"Dados {tf_str} insuficientes", "UNKNOWN"
        
        # Calcula EMA no timeframe alto
        df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        current_price = df['close'].iloc[-1]
        ema_value = df['ema'].iloc[-1]
        
        # Determina tendência macro
        if current_price > ema_value:
            h4_trend = "BULLISH"
        else:
            h4_trend = "BEARISH"
        
        # ✅ v5.3: SCORE PENALTY em vez de veto
        if signal == "BUY" and h4_trend != "BULLISH":
            return -8, f"{tf_str} divergente: BEARISH (Penalidade -8)", h4_trend
        
        if signal == "SELL" and h4_trend != "BEARISH":
            return -8, f"{tf_str} divergente: BULLISH (Penalidade -8)", h4_trend
        
        return 0, f"{tf_str} alinhado ({h4_trend})", h4_trend
        
    except Exception as e:
        logger.error(f"❌ get_multi_timeframe_trend: Erro para {symbol} - {e}")
        return 0, f"Erro: {e}", "ERROR"

# ===========================
# ROLLING WIN RATE v5.2
# ===========================
def calculate_rolling_win_rate(symbol: str = None, window: int = None) -> tuple:
    """
    ✅ v5.2: Calcula win rate dos últimos N trades via MT5 history.
    
    Args:
        symbol: Par específico (None = todos os símbolos do bot)
        window: Número de trades a considerar (default: config.KILL_SWITCH_TRADES)
        
    Returns:
        tuple: (win_rate: float, total_trades: int, wins: int)
    """
    try:
        if window is None:
            window = getattr(config, 'KILL_SWITCH_TRADES', 10)
        
        magic_number = getattr(config, 'MAGIC_NUMBER', 123456)
        
        # Obtém histórico de deals do MT5
        from datetime import timedelta
        from_date = datetime.now() - timedelta(days=30)
        to_date = datetime.now() + timedelta(days=1)
        
        with mt5_lock:
            deals = mt5.history_deals_get(from_date, to_date)
        
        if not deals or len(deals) == 0:
            return 0.5, 0, 0  # Default 50% se sem histórico
        
        # Filtra deals do bot
        bot_deals = []
        for deal in deals:
            if deal.magic != magic_number:
                continue
            if deal.type not in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
                continue
            if deal.profit == 0:  # Ignora entradas (só fechamentos)
                continue
            if symbol and deal.symbol != symbol:
                continue
            bot_deals.append(deal)
        
        # Ordena por tempo (mais recente primeiro) e pega os últimos N
        bot_deals.sort(key=lambda x: x.time, reverse=True)
        recent_deals = bot_deals[:window]
        
        if len(recent_deals) == 0:
            return 0.5, 0, 0
        
        # Conta wins
        wins = sum(1 for d in recent_deals if d.profit > 0)
        total = len(recent_deals)
        win_rate = wins / total if total > 0 else 0.5
        
        logger.debug(f"📊 Rolling WR: {win_rate:.1%} ({wins}/{total})")
        
        return win_rate, total, wins
        
    except Exception as e:
        logger.error(f"❌ calculate_rolling_win_rate: Erro - {e}")
        return 0.5, 0, 0

# ===========================
# SL/TP CALCULATION (v4.2 - PATCH 1)
# ===========================
def calculate_dynamic_levels(
    symbol: str,
    current_price: float,
    indicators: dict,
    sl_atr_mult: float = None,   # ✅ NOVO: Multiplicador SL otimizado
    tp_atr_mult: float = None,    # ✅ NOVO: Multiplicador TP otimizado
    signal: Optional[str] = None
) -> Tuple[float, float]:
    """
    ✅ v4.2: Calcula SL e TP dinamicamente usando ATR e multiplicadores otimizados.
    """
    atr_value = indicators.get("atr", 0)
    if atr_value <= 0:
        logger.warning(f"⚠️ ATR é zero para {symbol}. Retornando SL/TP 0.0.")
        return 0.0, 0.0

    # Usa multiplicadores otimizados se fornecidos, senão usa config padrão
    sl_multiplier = sl_atr_mult if sl_atr_mult is not None else getattr(config, 'DEFAULT_STOP_LOSS_ATR_MULTIPLIER', 2.0) # ✅ CORREÇÃO: STOP_LOSS_ATR_MULTIPLIER deve estar em config
    tp_multiplier = tp_atr_mult if tp_atr_mult is not None else getattr(config, 'DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER', 3.0) # ✅ CORREÇÃO: TAKE_PROFIT_ATR_MULTIPLIER deve estar em config

    sl_distance = atr_value * sl_multiplier
    regime = check_market_regime(indicators)
    tp_mult_adj = tp_multiplier
    if regime == "RANGING":
        tp_mult_adj = tp_multiplier * 0.8
    tp_distance = atr_value * tp_mult_adj

    # Garante que SL/TP não sejam muito pequenos
    min_sl_pips = config.MIN_STOP_LOSS_PIPS # ✅ CORREÇÃO: MIN_STOP_LOSS_PIPS deve estar em config
    pip_size = get_pip_size(symbol)
    min_sl_distance = min_sl_pips * pip_size

    if sl_distance < min_sl_distance:
        sl_distance = min_sl_distance
        logger.debug(f"Ajustando SL para {symbol} para o mínimo: {min_sl_pips} pips")

    side = signal
    if side is None:
        ema_trend = indicators.get("ema_trend")
        if ema_trend == "UP":
            side = "BUY"
        elif ema_trend == "DOWN":
            side = "SELL"

    if side == "BUY":
        sl = current_price - sl_distance
        tp = current_price + tp_distance
    elif side == "SELL":
        sl = current_price + sl_distance
        tp = current_price - tp_distance
    else:
        logger.warning(f"⚠️ Tendência EMA desconhecida para {symbol}. Não foi possível calcular SL/TP.")
        return 0.0, 0.0

    # Arredonda ao número correto de dígitos
    info = get_symbol_info(symbol)
    if info:
        digits = info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)
    else:
        logger.warning(f"⚠️ Não foi possível obter info para {symbol} para arredondar SL/TP. Usando 5 casas decimais.")
        sl = round(sl, 5) # Fallback
        tp = round(tp, 5) # Fallback

    return sl, tp

def check_market_regime(indicators: dict) -> str:
    try:
        adx = float(indicators.get("adx", 0) or 0.0)
        bb_width = float(indicators.get("bb_width", 0) or 0.0)
        adx_min = float(getattr(config, 'ADX_MIN_STRENGTH', 15))
        bb_thresh = float(getattr(config, 'BB_SQUEEZE_THRESHOLD', 0.015))
        if adx >= adx_min and bb_width >= bb_thresh:
            return "TRENDING"
        else:
            return "RANGING"
    except Exception:
        return "RANGING"

def calculate_signal_score(
    indicators: dict,
    ema_short: int = 20,
    ema_long: int = 50,
    rsi_low: int = 30,
    rsi_high: int = 70,
    adx_threshold: int = 25,
    bb_squeeze_threshold: float = 0.015
) -> Tuple[float, dict]:
    score = 50.0
    details = {}
    rsi = indicators.get("rsi", 50)
    adx = indicators.get("adx", 0)
    ema_trend = indicators.get("ema_trend", "N/A")
    bb_width = indicators.get("bb_width", 0)
    close = indicators.get("close", 0)
    open_price = indicators.get("open", 0)
    rsi_bull_zone = rsi_low + 10
    rsi_bear_zone = rsi_high - 10
    if ema_trend in ["UP", "DOWN"]:
        score += 15
        details["trend_bonus"] = 15
    if ema_trend == "UP":
        if rsi_bull_zone <= rsi <= rsi_high:
            score += 20
            details["rsi_bonus"] = 20
        elif rsi_low <= rsi < rsi_bull_zone:
            score += 10
            details["rsi_bonus"] = 10
    elif ema_trend == "DOWN":
        if rsi_low <= rsi <= rsi_bear_zone:
            score += 20
            details["rsi_bonus"] = 20
        elif rsi_bear_zone < rsi <= rsi_high:
            score += 10
            details["rsi_bonus"] = 10
    else:
        if rsi < rsi_low or rsi > rsi_high:
            score += 20
            details["rsi_reversal_bonus"] = 20
    if adx > adx_threshold:
        score += 15
        details["adx_bonus"] = 15
    elif adx > (adx_threshold * 0.7):
        score += 8
        details["adx_bonus"] = 8
    if bb_width > bb_squeeze_threshold:
        score += 10
        details["volatility_bonus"] = 10
    if close > open_price and ema_trend == "UP":
        score += 10
        details["candle_bonus"] = 10
    elif close < open_price and ema_trend == "DOWN":
        score += 10
        details["candle_bonus"] = 10
    total_penalty = 0.0
    ema_penalty = indicators.get("ema_penalty", 0)
    if ema_penalty > 0:
        penalty = min(float(ema_penalty), 12.0)
        score -= penalty
        total_penalty += penalty
        details["ema_penalty"] = -penalty
    mtf_penalty = indicators.get("mtf_penalty", 0)
    if mtf_penalty > 0:
        penalty = min(float(mtf_penalty), 8.0)
        score -= penalty
        total_penalty += penalty
        details["mtf_penalty"] = -penalty
    macd_hist = indicators.get("macd_hist", 0)
    if ema_trend == "UP" and macd_hist < 0:
        score -= 5
        total_penalty += 5
        details["macd_penalty"] = -5
    elif ema_trend == "DOWN" and macd_hist > 0:
        score -= 5
        total_penalty += 5
        details["macd_penalty"] = -5
    if bb_width < bb_squeeze_threshold:
        score -= 8
        total_penalty += 8
        details["squeeze_penalty"] = -8
    if total_penalty > 30:
        excess = total_penalty - 30
        score += excess
        details["penalty_cap_applied"] = excess
    score = max(20.0, min(100.0, score))
    try:
        bonus_sum = sum(v for k, v in details.items() if "bonus" in k)
    except Exception:
        bonus_sum = 0
    logger.debug(f"Score Breakdown: Base=50 | Bônus={bonus_sum} | Penalidades=-{total_penalty} | Final={score:.1f}")
    return score, details

# ===========================
# TRADE EXECUTION HELPERS
# ===========================

def modify_position_sl_tp(ticket, sl, tp):
    """
    ✅ v4.3: Modifica o Stop Loss e o Take Profit de uma posição existente.
    Inclui verificação de Stop Levels, Normalização e Silenciamento de Erros.
    """
    try:
        # 1. Obter informações da posição
        with mt5_lock:
            positions = mt5.positions_get(ticket=ticket)
        
        if not positions or len(positions) == 0:
            logger.error(f"❌ Não foi possível encontrar a posição {ticket} para modificação.")
            return False
            
        pos = positions[0]
        symbol = pos.symbol
        
        # 2. Obter informações do símbolo e tick atual
        info = get_symbol_info(symbol)
        if not info:
            return False
            
        with mt5_lock:
            tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
            
        digits = info.digits
        point = info.point
        stops_level = info.trade_stops_level * point
        freeze_level = info.trade_freeze_level * point
        # A distância mínima deve considerar stops_level e freeze_level
        min_dist = max(stops_level, freeze_level)
        
        # Garante uma margem mínima se a corretora retornar 0 (comum em algumas mas perigoso)
        if min_dist == 0:
            min_dist = 2 * point
            
        bid = tick.bid
        ask = tick.ask

        # 3. Normalização de Preços e Margem de Segurança (Buffer)
        # ✅ NOVO: Buffer adicional para Índices para evitar erro 10016 por milissegundos
        symbol_upper = symbol.upper()
        if any(idx in symbol_upper for idx in ["US30", "GER40", "DE40", "UK100", "US500", "NAS100", "USTEC"]):
            min_dist += 2 * point # Buffer cirúrgico de 2 pontos extras

        sl = round(sl, digits)
        tp = round(tp, digits)
        
        # 4. Trava Lógica de Direção e Distância (Evita erro 10016)
        is_buy = (pos.type == mt5.POSITION_TYPE_BUY)
        current_sl = pos.sl
        invalid_reason = None
        
        if is_buy:
            # Regras para COMPRA:
            # - Novo SL deve ser maior que o SL atual (Trava de Direção Trailing)
            # - Novo SL deve ser menor que (Bid - stops_level)
            if sl != 0:
                if sl <= current_sl and current_sl != 0:
                    return False # Silencioso: ignoramos movimentos para trás ou iguais
                if sl > (bid - min_dist):
                    invalid_reason = f"SL muito próximo do Bid ({sl} > {bid - min_dist:.{digits}f})"
        else:
            # Regras para VENDA:
            # - Novo SL deve ser menor que o SL atual (Trava de Direção Trailing)
            # - Novo SL deve ser maior que (Ask + stops_level)
            if sl != 0:
                if sl >= current_sl and current_sl != 0:
                    return False # Silencioso: ignoramos movimentos para trás ou iguais
                if sl < (ask + min_dist):
                    invalid_reason = f"SL muito próximo do Ask ({sl} < {ask + min_dist:.{digits}f})"

        # Se houver violação de distância, logamos com throttling (1x por minuto)
        if invalid_reason:
            current_time = time.time()
            cache_key = f"invalid_{symbol}_{ticket}"
            last_log = ERROR_CACHE.get(cache_key, {"time": 0})["time"]
            
            if current_time - last_log > 60:
                logger.warning(f"⚠️ {symbol}: Modificação ignorada | {invalid_reason}")
                ERROR_CACHE[cache_key] = {"time": current_time}
            return False

        # 5. Envio da Ordem
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        
        with mt5_lock:
            result = mt5.order_send(request)
            
        if result is None:
            logger.error(f"❌ Falha crítica ao enviar modificação para {ticket}: result is None")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Silenciamento para Erro 10016 (Invalid Stops)
            if result.retcode == 10016:
                current_time = time.time()
                cache_key = f"err10016_{symbol}"
                last_info = ERROR_CACHE.get(cache_key, {"time": 0, "price": 0})
                
                ref_price = bid if is_buy else ask
                price_changed = abs(ref_price - last_info["price"]) > (min_dist * 2)
                
                if current_time - last_info["time"] > 60 or price_changed:
                    logger.error(f"❌ Erro 10016 ({symbol}): {result.comment} | SL Tentado: {sl} | MinDist: {min_dist:.{digits}f}")
                    ERROR_CACHE[cache_key] = {"time": current_time, "price": ref_price}
            else:
                logger.error(f"❌ Falha ao modificar {symbol} #{ticket}: {result.comment} ({result.retcode})")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Erro crítico em modify_position_sl_tp: {e}", exc_info=True)
        return False

def close_all_positions():
    """
    ✅ v4.2: Fecha todas as posições abertas pelo bot (mesmo Magic Number).
    Utilizado principalmente pelo KILL SWITCH ou shutdown.
    """
    try:
        positions = mt5.positions_get()
        if not positions:
            logger.info("ℹ️ Nenhuma posição aberta para fechar.")
            return

        magic_number = getattr(config, 'MAGIC_NUMBER', 123456)
        closed_count = 0
        
        for pos in positions:
            if pos.magic == magic_number:
                symbol_info = mt5.symbol_info(pos.symbol)
                if not symbol_info:
                    continue
                    
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue

                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": getattr(config, 'DEVIATION', 20),
                    "magic": magic_number,
                    "comment": "Bot Shutdown - Close All",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result is None:
                    logger.error(f"❌ Erro ao fechar {pos.ticket}: result is None")
                    continue
                    
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"❌ Falha ao fechar posição {pos.ticket}: {result.comment} (retcode: {result.retcode})")
                else:
                    logger.info(f"✅ Posição {pos.ticket} fechada com sucesso.")
                    closed_count += 1
        
        if closed_count > 0:
            logger.info(f"🛑 Fechamento em massa concluído: {closed_count} posições encerradas.")
            
    except Exception as e:
        logger.error(f"❌ Erro crítico em close_all_positions: {e}", exc_info=True)

# ===========================
# INSTITUTIONAL RISK v5.3
# ===========================

def get_risk_manager_state() -> dict:
    """Carrega o estado do gerenciador de risco do arquivo JSON."""
    file_path = getattr(config, 'RISK_MANAGER_FILE', 'data/risk_manager.json')
    default_state = {
        "weekly_start_balance": 0.0,
        "weekly_drawdown_max": 0.0,
        "last_update_week": 0
    }
    
    if not os.path.exists(file_path):
        return default_state
        
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Erro ao ler {file_path}: {e}")
        return default_state

def save_risk_manager_state(state: dict):
    """Salva o estado do gerenciador de risco no arquivo JSON."""
    file_path = getattr(config, 'RISK_MANAGER_FILE', 'data/risk_manager.json')
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logger.error(f"❌ Erro ao salvar {file_path}: {e}")

def check_institutional_risk() -> Tuple[bool, str]:
    """
    ✅ v5.3: Validação de risco institucional (Hard Stop).
    Retorna (is_ok, reason)
    """
    try:
        with mt5_lock:
            acc = mt5.account_info()
        if not acc:
            return False, "Erro ao obter info da conta"

        # 1. Kill Switch por Equity (Drawdown Total)
        total_dd = (acc.balance - acc.equity) / acc.balance if acc.balance > 0 else 0
        if total_dd > getattr(config, 'INSTITUTIONAL_KILL_SWITCH_PCT', 0.10):
            return False, f"🔴 KILL SWITCH: Drawdown Total {total_dd:.1%} > {config.INSTITUTIONAL_KILL_SWITCH_PCT:.1%}"

        # 2. Drawdown Diário (Circuit Breaker)
        # Nota: O bot já deve ter o balance inicial do dia salvo em algum lugar.
        # Se não tiver, usamos o balance atual vs equity como aproximação de DD flutuante.
        # Mas para Hard Stop diário, é melhor ter o balance do início do dia.
        daily_dd = (acc.balance - acc.equity) / acc.balance if acc.balance > 0 else 0
        if daily_dd > getattr(config, 'MAX_DAILY_DRAWDOWN_PCT', 0.02):
            return False, f"🔴 DAILY STOP: Drawdown {daily_dd:.1%} > {config.MAX_DAILY_DRAWDOWN_PCT:.1%}"

        # 3. Drawdown Semanal (Persistente)
        state = get_risk_manager_state()
        curr_week = datetime.now().isocalendar()[1]
        
        if state['last_update_week'] != curr_week:
            # Reset semanal
            state['weekly_start_balance'] = acc.balance
            state['last_update_week'] = curr_week
            save_risk_manager_state(state)
            
        weekly_start = state['weekly_start_balance']
        if weekly_start > 0:
            weekly_dd = (weekly_start - acc.equity) / weekly_start
            if weekly_dd > getattr(config, 'MAX_WEEKLY_DRAWDOWN_PCT', 0.05):
                return False, f"🔴 WEEKLY STOP: Drawdown {weekly_dd:.1%} > {config.MAX_WEEKLY_DRAWDOWN_PCT:.1%}"

        # 4. Bloqueio de Alavancagem Implícita
        # Alavancagem real = Exposição Total / Equity
        # Para simplificar, verificamos a alavancagem da conta configurada
        if acc.leverage > getattr(config, 'MAX_ACCOUNT_LEVERAGE', 30):
             # Apenas loga aviso, não bloqueia se já estiver operando, 
             # mas podemos bloquear novas entradas
             pass

        return True, "Risk OK"

    except Exception as e:
        logger.error(f"❌ Erro em check_institutional_risk: {e}")
        return True, "Erro na validação (Ignorado)"

def check_currency_exposure(positions) -> Tuple[bool, str]:
    """
    ✅ v5.3: Controle básico de exposição por moeda (USD, JPY, EUR).
    """
    if not positions:
        return True, ""
        
    exposure = {"USD": 0.0, "JPY": 0.0, "EUR": 0.0}
    total_equity = mt5.account_info().equity if mt5.account_info() else 1.0
    
    for pos in positions:
        symbol = pos.symbol.upper()
        # Estimativa simplificada de valor nocional em USD
        # Volume * Preço 
        # Para Forex, volume é em unidades da moeda base.
        nocional = pos.volume * 100000 # Lote padrão
        
        for curr in exposure.keys():
            if curr in symbol:
                exposure[curr] += nocional
                
    max_exp_pct = getattr(config, 'MAX_CURRENCY_EXPOSURE_PCT', 0.03)
    max_exp_val = total_equity * max_exp_pct * 30 # Considerando alavancagem 1:30
    
    # Se a exposição nocional ultrapassar 3% do equity * alavancagem (ou similar)
    # Na verdade, o user pediu "Controle básico de exposição por moeda (USD, JPY, EUR)"
    # Vamos apenas logar ou vetar se ultrapassar um limite configurado.
    
    for curr, val in exposure.items():
        if val > max_exp_val:
            return False, f"🟠 EXPOSIÇÃO {curr}: ${val:,.2f} ultrapassa limite institucional"
            
    return True, ""

def check_total_exposure_limit(pending_symbol: Optional[str] = None, pending_volume: float = 0.0, pending_side: Optional[str] = None) -> Tuple[bool, str]:
    try:
        acc = mt5.account_info()
        basis = str(getattr(config, 'MAX_TOTAL_EXPOSURE_BASIS', 'balance')).lower()
        base_val = 0.0
        if acc:
            base_val = acc.balance if basis == "balance" else acc.equity
        limit_mult = float(getattr(config, 'MAX_TOTAL_EXPOSURE_MULTIPLIER', 2.0))
        if base_val <= 0:
            return True, ""
        total_limit_usd = base_val * limit_mult
        with mt5_lock:
            positions = mt5.positions_get()
        def estimate_exposure_usd(symbol: str, volume: float) -> float:
            info = get_symbol_info(symbol)
            if not info or volume <= 0:
                return 0.0
            contract = float(getattr(info, 'trade_contract_size', 100000) or 100000)
            tick = mt5.symbol_info_tick(symbol)
            price = 0.0
            if tick:
                if pending_side == "BUY":
                    price = float(getattr(tick, 'ask', 0.0) or 0.0)
                elif pending_side == "SELL":
                    price = float(getattr(tick, 'bid', 0.0) or 0.0)
                else:
                    price = float(getattr(tick, 'bid', 0.0) or 0.0)
            s = symbol.upper()
            if len(s) >= 6 and s[3:6] == "USD":
                return contract * volume * (price if price > 0 else 1.0)
            if len(s) >= 6 and s[0:3] == "USD":
                return contract * volume
            if ("XAUUSD" in s) or ("XAGUSD" in s) or ("US30" in s) or ("US500" in s) or ("NAS100" in s) or ("USTEC" in s) or ("USA500" in s):
                return contract * volume * (price if price > 0 else 1.0)
            return contract * volume
        current_exposure = 0.0
        if positions:
            for p in positions:
                current_exposure += estimate_exposure_usd(p.symbol, float(getattr(p, 'volume', 0.0) or 0.0))
        if pending_symbol and pending_volume and pending_volume > 0:
            current_exposure += estimate_exposure_usd(pending_symbol, pending_volume)
        if current_exposure > total_limit_usd:
            return False, f"Exposição total {current_exposure:,.2f} USD > limite {total_limit_usd:,.2f} USD"
        warn_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_WARNING_PCT', 0.80))
        alert_pct = float(getattr(config, 'MAX_TOTAL_EXPOSURE_ALERT_PCT', 0.95))
        usage = (current_exposure / total_limit_usd) if total_limit_usd > 0 else 0.0
        if usage >= alert_pct:
            return True, f"🚨 Uso de margem {usage*100:.0f}% (Limite={total_limit_usd:,.0f} USD)"
        if usage >= warn_pct:
            return True, f"⚠️ Uso de margem {usage*100:.0f}% (Limite={total_limit_usd:,.0f} USD)"
        return True, ""
    except Exception as e:
        logger.error(f"check_total_exposure_limit erro: {e}")
        return True, ""

def _fx_base_quote(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        s = str(symbol).upper()
        if len(s) >= 6 and s[:3].isalpha() and s[3:6].isalpha():
            return s[:3], s[3:6]
    except Exception:
        pass
    return None, None

def _collect_net_currency_exposure(positions: List[Any]) -> Dict[str, float]:
    exposure: Dict[str, float] = {}
    try:
        for p in positions or []:
            base, quote = _fx_base_quote(p.symbol)
            vol_units = float(getattr(p, "volume", 0.0) or 0.0) * 100000.0
            if base and quote and vol_units > 0:
                if int(getattr(p, "type", 0)) == mt5.POSITION_TYPE_BUY:
                    exposure[base] = exposure.get(base, 0.0) + vol_units
                    exposure[quote] = exposure.get(quote, 0.0) - vol_units
                else:
                    exposure[base] = exposure.get(base, 0.0) - vol_units
                    exposure[quote] = exposure.get(quote, 0.0) + vol_units
            else:
                exposure["USD"] = exposure.get("USD", 0.0) + vol_units
        return exposure
    except Exception as e:
        logger.error(f"_collect_net_currency_exposure erro: {e}")
        return {}

def check_portfolio_exposure(pending_symbol: Optional[str], pending_volume: float, pending_side: Optional[str]) -> Tuple[bool, str]:
    try:
        acc = mt5.account_info()
        eq = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0
        with mt5_lock:
            positions = mt5.positions_get()
        net = _collect_net_currency_exposure([p for p in (positions or []) if int(getattr(p, "magic", 0)) == int(getattr(config, "MAGIC_NUMBER", 123456))])
        base, quote = _fx_base_quote(pending_symbol or "")
        vol_units = float(pending_volume or 0.0) * 100000.0
        if base and quote and vol_units > 0 and pending_side in ("BUY", "SELL"):
            if pending_side == "BUY":
                net[base] = net.get(base, 0.0) + vol_units
                net[quote] = net.get(quote, 0.0) - vol_units
            else:
                net[base] = net.get(base, 0.0) - vol_units
                net[quote] = net.get(quote, 0.0) + vol_units
        else:
            net["USD"] = net.get("USD", 0.0) + vol_units
        limits_map = getattr(config, "MAX_CURRENCY_EXPOSURE_PCT_MAP", {}) or {}
        default_pct = float(getattr(config, "MAX_CURRENCY_EXPOSURE_PCT", 0.03))
        lev = float(getattr(config, "ASSUMED_LEVERAGE", 30.0))
        corr_tighten = float(getattr(config, "CORR_EXPOSURE_TIGHTEN_FACTOR", 0.5))
        corr_limit = float(getattr(config, "CORRELATION_MAX", 0.75))
        tightened_currencies = set()
        correlations = getattr(config, "SYMBOL_CORRELATIONS", {})
        try:
            if pending_symbol and pending_symbol in correlations:
                sym_corrs = correlations[pending_symbol]
                for p in positions or []:
                    ps = getattr(p, "symbol", "")
                    if ps in sym_corrs and abs(float(sym_corrs[ps])) >= corr_limit:
                        if base:
                            tightened_currencies.add(base)
                        if quote:
                            tightened_currencies.add(quote)
        except Exception:
            pass
        for curr, val in net.items():
            pct = float(limits_map.get(curr, default_pct))
            if curr in tightened_currencies:
                pct *= corr_tighten
            max_val = eq * pct * lev
            if max_val <= 0:
                continue
            if abs(val) > max_val:
                return False, f"Exposição net {curr} {val:,.0f} > limite {max_val:,.0f}"
        return True, ""
    except Exception as e:
        logger.error(f"check_portfolio_exposure erro: {e}")
        return True, ""
# ===========================
# ESSENTIAL METRICS v5.3
# ===========================

def get_daily_metrics() -> dict:
    """
    ✅ v5.3: Carrega ou cria métricas do dia.
    Retorna dict com trades_today, wins, losses, max_dd, durations, exposure.
    """
    file_path = "data/metrics.json"
    today = datetime.now().strftime("%Y-%m-%d")
    
    default_metrics = {
        "date": today,
        "trades_today": 0,
        "wins": 0,
        "losses": 0,
        "max_dd_intraday": 0.0,
        "trade_durations": [],
        "currency_exposure": {"USD": 0.0, "JPY": 0.0, "EUR": 0.0}
    }
    
    if not os.path.exists(file_path):
        return default_metrics
        
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
            # Reset se mudou de dia
            if metrics.get("date") != today:
                return default_metrics
            return metrics
    except Exception as e:
        logger.error(f"❌ Erro ao ler metrics.json: {e}")
        return default_metrics

def save_daily_metrics(metrics: dict):
    """✅ v5.3: Salva métricas do dia."""
    file_path = "data/metrics.json"
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logger.error(f"❌ Erro ao salvar metrics.json: {e}")

def update_trade_metric(won: bool, duration_minutes: float = 0):
    """
    ✅ v5.3: Atualiza métricas após fechamento de trade.
    Args:
        won: True se trade vencedor
        duration_minutes: Tempo em minutos que a posição ficou aberta
    """
    metrics = get_daily_metrics()
    metrics["trades_today"] += 1
    
    if won:
        metrics["wins"] += 1
    else:
        metrics["losses"] += 1
    
    if duration_minutes > 0:
        metrics["trade_durations"].append(duration_minutes)
    
    save_daily_metrics(metrics)
    logger.info(f"📊 Métricas atualizadas: {metrics['trades_today']} trades, WR: {metrics['wins']}/{metrics['trades_today']}")

def calculate_current_metrics() -> dict:
    """
    ✅ v5.3: Calcula métricas em tempo real do MT5.
    Retorna: {win_rate, max_dd_today, avg_duration, exposure_by_currency}
    """
    try:
        with mt5_lock:
            acc = mt5.account_info()
            positions = mt5.positions_get()
        
        if not acc:
            return {}
        
        # 1. Win Rate (baseado em deals de hoje)
        from_date = datetime.now().replace(hour=0, minute=0, second=0)
        to_date = datetime.now() + timedelta(days=1)
        
        with mt5_lock:
            deals = mt5.history_deals_get(from_date, to_date)
        
        wins = 0
        total = 0
        total_profit = 0.0
        total_loss = 0.0
        
        if deals:
            magic = getattr(config, 'MAGIC_NUMBER', 123456)
            for deal in deals:
                if deal.magic != magic or deal.profit == 0:
                    continue
                total += 1
                if deal.profit > 0:
                    wins += 1
                    total_profit += float(deal.profit)
                else:
                    total_loss += float(deal.profit)
        
        win_rate = (wins / total * 100) if total > 0 else 0.0
        profit_factor = (total_profit / abs(total_loss)) if total_loss != 0 else 0.0
        pnl_today = total_profit + total_loss
        avg_trade = (pnl_today / total) if total > 0 else 0.0
        
        # 2. Max DD Intraday (simplificado: equity drawdown atual)
        # Para DD real do dia, precisaríamos do equity máximo do dia (não disponível facilmente)
        # Usamos aproximação: (balance - equity) / balance
        dd_current = ((acc.balance - acc.equity) / acc.balance * 100) if acc.balance > 0 else 0.0
        
        # 3. Tempo médio em trade (requer histórico completo - skip por enquanto)
        avg_duration = 0.0
        
        # 4. Exposição por moeda
        exposure = {"USD": 0.0, "JPY": 0.0, "EUR": 0.0}
        if positions:
            for pos in positions:
                symbol = pos.symbol.upper()
                nocional = pos.volume * 100000  # Lote padrão
                for curr in exposure.keys():
                    if curr in symbol:
                        exposure[curr] += nocional
        
        return {
            "win_rate": win_rate,
            "trades_today": total,
            "max_dd_intraday": dd_current,
            "avg_duration_minutes": avg_duration,
            "currency_exposure": exposure,
            "profit_factor_today": profit_factor,
            "pnl_today": pnl_today,
            "avg_trade": avg_trade
        }
        
    except Exception as e:
        logger.error(f"❌ Erro em calculate_current_metrics: {e}")
        return {}

def log_metrics_summary():
    """✅ v5.3: Loga resumo de métricas essenciais."""
    try:
        metrics = calculate_current_metrics()
        if not metrics:
            return
        
        logger.info("="*60)
        logger.info("📊 MÉTRICAS ESSENCIAIS")
        logger.info(f"  Trades Hoje: {metrics.get('trades_today', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        logger.info(f"  Profit Factor (Hoje): {metrics.get('profit_factor_today', 0):.2f}")
        logger.info(f"  PnL (Hoje): ${metrics.get('pnl_today', 0):+.2f}")
        logger.info(f"  PnL Médio/Trade: ${metrics.get('avg_trade', 0):+.2f}")
        logger.info(f"  Max DD Intraday: {metrics.get('max_dd_intraday', 0):.2f}%")
        logger.info(f"  Tempo Médio: {metrics.get('avg_duration_minutes', 0):.0f}min")
        
        exp = metrics.get('currency_exposure', {})
        logger.info(f"  Exposição USD: ${exp.get('USD', 0):,.0f}")
        logger.info(f"  Exposição JPY: ${exp.get('JPY', 0):,.0f}")
        logger.info(f"  Exposição EUR: ${exp.get('EUR', 0):,.0f}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Erro ao logar métricas: {e}")

def _retry_call(fn, max_attempts: int = 3, base_sleep: float = 1.0, *args, **kwargs):
    attempts = 0
    last_exc = None
    while attempts < max_attempts:
        try:
            return True, fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(base_sleep * (2 ** attempts))
            attempts += 1
    return False, last_exc

def _classify_asset(symbol: str) -> Tuple[str, str]:
    s = symbol.upper()
    if any(k in s for k in ["US30", "NAS100", "US500", "USA500", "USTEC", "DE40", "GER40", "GER30", "UK100", "HK50", "JP225", "FRA40"]):
        if any(k in s for k in ["US30", "NAS100", "US500", "USA500", "USTEC"]):
            return "INDICES", "NA"
        if any(k in s for k in ["DE40", "GER40", "GER30", "FRA40"]):
            return "INDICES", "EU"
        if "UK100" in s:
            return "INDICES", "EU"
        if "HK50" in s or "JP225" in s:
            return "INDICES", "AS"
        return "INDICES", "GL"
    if any(k in s for k in ["XAU", "XAG", "GOLD", "SILVER"]):
        return "METALS", "GL"
    if any(k in s for k in ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "LTC", "DOGE"]):
        return "CRYPTO", "GL"
    if len(s) >= 6:
        base = s[0:3]
        quote = s[3:6]
        region_map = {
            "USD": "NA", "EUR": "EU", "JPY": "AS", "GBP": "EU", "AUD": "OC",
            "NZD": "OC", "CAD": "NA", "CHF": "EU", "BRL": "SA", "CNH": "AS", "CNY": "AS"
        }
        return "FX", region_map.get(base, "GL")
    return "UNKNOWN", "GL"

def _estimate_notional_usd(symbol: str, volume: float) -> float:
    info = get_symbol_info(symbol)
    if not info or volume <= 0:
        return 0.0
    contract = float(getattr(info, 'trade_contract_size', 100000) or 100000)
    tick = mt5.symbol_info_tick(symbol)
    price = 0.0
    if tick:
        price = float(getattr(tick, 'bid', 0.0) or 0.0)
    s = symbol.upper()
    if len(s) >= 6 and s[3:6] == "USD":
        return contract * volume * (price if price > 0 else 1.0)
    if len(s) >= 6 and s[0:3] == "USD":
        return contract * volume
    if any(k in s for k in ["XAUUSD", "XAGUSD", "US30", "US500", "NAS100", "USTEC", "USA500"]):
        return contract * volume * (price if price > 0 else 1.0)
    return contract * volume

def collect_portfolio_snapshot() -> Dict[str, Any]:
    br_now = get_brasilia_time()
    ts = datetime.now(pytz.utc)
    ok_conn = check_mt5_connection() or ensure_mt5_connection()
    issues = []
    meta = {
        "timestamp_utc": ts.isoformat(timespec="seconds"),
        "timestamp_brt": br_now.isoformat(timespec="seconds"),
        "system_version": "XP3_PRO_FOREX_v4.2",
    }
    positions = []
    orders_pending = []
    deals_today = []
    acc = None
    t0 = time.time()
    try:
        with mt5_lock:
            acc = mt5.account_info()
    except Exception as e:
        issues.append(f"account_info: {e}")
    ok_pos, res_pos = _retry_call(lambda: mt5.positions_get(), getattr(config, "SNAPSHOT_RETRY_ATTEMPTS", 3), getattr(config, "SNAPSHOT_BACKOFF_BASE", 1.0))
    if ok_pos and res_pos:
        for p in res_pos:
            cls, region = _classify_asset(p.symbol)
            positions.append({
                "ticket": int(getattr(p, "ticket", 0)),
                "symbol": str(getattr(p, "symbol", "")),
                "type": ("BUY" if getattr(p, "type", 0) == mt5.POSITION_TYPE_BUY else "SELL"),
                "volume": float(getattr(p, "volume", 0.0)),
                "price_open": float(getattr(p, "price_open", 0.0)),
                "price_current": float(getattr(p, "price_current", 0.0)),
                "profit": float(getattr(p, "profit", 0.0)),
                "sl": float(getattr(p, "sl", 0.0)),
                "tp": float(getattr(p, "tp", 0.0)),
                "magic": int(getattr(p, "magic", 0)),
                "asset_class": cls,
                "region": region,
                "notional_usd": _estimate_notional_usd(getattr(p, "symbol", ""), float(getattr(p, "volume", 0.0)))
            })
    else:
        issues.append(f"positions_get_fail: {res_pos}")
    ok_ord, res_ord = _retry_call(lambda: mt5.orders_get(), getattr(config, "SNAPSHOT_RETRY_ATTEMPTS", 3), getattr(config, "SNAPSHOT_BACKOFF_BASE", 1.0))
    if ok_ord and res_ord:
        for o in res_ord:
            orders_pending.append({
                "ticket": int(getattr(o, "ticket", 0)),
                "symbol": str(getattr(o, "symbol", "")),
                "type": int(getattr(o, "type", 0)),
                "volume": float(getattr(o, "volume_initial", 0.0)),
                "price": float(getattr(o, "price_open", 0.0)),
                "sl": float(getattr(o, "sl", 0.0)),
                "tp": float(getattr(o, "tp", 0.0)),
                "state": int(getattr(o, "state", 0)),
                "time": int(getattr(o, "time_setup", 0))
            })
    else:
        issues.append(f"orders_get_fail: {res_ord}")
    from_date = br_now.replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = from_date + timedelta(days=1)
    ok_deals, res_deals = _retry_call(lambda: mt5.history_deals_get(from_date, to_date), getattr(config, "SNAPSHOT_RETRY_ATTEMPTS", 3), getattr(config, "SNAPSHOT_BACKOFF_BASE", 1.0))
    if ok_deals and res_deals:
        magic = getattr(config, "MAGIC_NUMBER", 123456)
        for d in res_deals or []:
            if getattr(d, "magic", None) != magic:
                continue
            deals_today.append({
                "time": datetime.fromtimestamp(getattr(d, "time", 0)).isoformat(timespec="seconds"),
                "symbol": str(getattr(d, "symbol", "")),
                "type": int(getattr(d, "type", 0)),
                "entry": int(getattr(d, "entry", 0)),
                "volume": float(getattr(d, "volume", 0.0)),
                "price": float(getattr(d, "price", 0.0)),
                "profit": float(getattr(d, "profit", 0.0)),
                "commission": float(getattr(d, "commission", 0.0)),
                "swap": float(getattr(d, "swap", 0.0)),
                "order": int(getattr(d, "order", 0)),
                "position_id": int(getattr(d, "position_id", 0)),
            })
    else:
        issues.append(f"history_deals_get_fail: {res_deals}")
    total_value = sum(p.get("notional_usd", 0.0) for p in positions)
    total_pnl = sum(p.get("profit", 0.0) for p in positions)
    by_class = {}
    by_currency = {}
    by_region = {}
    for p in positions:
        cls = p.get("asset_class", "UNKNOWN")
        by_class[cls] = by_class.get(cls, 0.0) + p.get("notional_usd", 0.0)
        s = str(p.get("symbol", "")).upper()
        if len(s) >= 6:
            base = s[0:3]
            quote = s[3:6]
            for curr in [base, quote]:
                by_currency[curr] = by_currency.get(curr, 0.0) + p.get("notional_usd", 0.0) * 0.5
        reg = p.get("region", "GL")
        by_region[reg] = by_region.get(reg, 0.0) + p.get("notional_usd", 0.0)
    complete = ok_conn and len(positions) >= 0
    t_elapsed = time.time() - t0
    snapshot = {
        "meta": meta,
        "account": {
            "login": int(getattr(acc, "login", 0)) if acc else 0,
            "balance": float(getattr(acc, "balance", 0.0)) if acc else 0.0,
            "equity": float(getattr(acc, "equity", 0.0)) if acc else 0.0,
            "margin": float(getattr(acc, "margin", 0.0)) if acc else 0.0,
            "leverage": int(getattr(acc, "leverage", 0)) if acc else 0,
        },
        "positions": positions,
        "orders_pending": orders_pending,
        "deals_today": deals_today,
        "metrics": {
            "positions_count": len(positions),
            "orders_pending_count": len(orders_pending),
            "deals_today_count": len(deals_today),
            "total_value_usd": total_value,
            "total_pnl_usd": total_pnl,
            "allocation_by_class_usd": by_class,
            "exposure_by_currency_usd": by_currency,
            "exposure_by_region_usd": by_region,
            "processing_seconds": t_elapsed
        },
        "complete": complete,
        "issues": issues
    }
    return snapshot

def save_portfolio_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(getattr(config, "SNAPSHOT_OUTPUT_DIR", "data/portfolio_snapshots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    br_ts = snapshot.get("meta", {}).get("timestamp_brt", "")
    date_str = br_ts[:10] if br_ts else datetime.now().strftime("%Y-%m-%d")
    time_str = br_ts[11:16].replace(":", "-") if br_ts else datetime.now().strftime("%H-%M")
    base = f"snapshot_{date_str}_{time_str}"
    json_path = out_dir / f"{base}.json"
    csv_pos_path = out_dir / f"{base}_positions.csv"
    csv_ord_path = out_dir / f"{base}_orders.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    with open(csv_pos_path, "w", encoding="utf-8", newline="") as f:
        fields = ["ticket", "symbol", "type", "volume", "price_open", "price_current", "profit", "sl", "tp", "magic", "asset_class", "region", "notional_usd"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in snapshot.get("positions", []):
            w.writerow({k: p.get(k) for k in fields})
    with open(csv_ord_path, "w", encoding="utf-8", newline="") as f:
        fields = ["ticket", "symbol", "type", "volume", "price", "sl", "tp", "state", "time"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in snapshot.get("orders_pending", []):
            w.writerow({k: o.get(k) for k in fields})
    latest_path = out_dir / "latest.json"
    try:
        if latest_path.exists():
            backup_dir = out_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"latest_{ts}.json"
            latest_path.replace(backup_path)
        latest_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"erro ao versionar latest.json: {e}")
    return {"json": str(json_path), "positions_csv": str(csv_pos_path), "orders_csv": str(csv_ord_path)}

def compare_and_alert_snapshot(prev: Dict[str, Any], curr: Dict[str, Any]) -> None:
    try:
        th = float(getattr(config, "SNAPSHOT_ALERT_DEVIATION_PCT", 0.2))
        pv = float(((prev or {}).get("metrics") or {}).get("total_value_usd", 0.0))
        cv = float(((curr or {}).get("metrics") or {}).get("total_value_usd", 0.0))
        if pv > 0:
            delta = (cv - pv) / pv
            if abs(delta) >= th:
                utils_msg = (
                    f"📦 Snapshot Sexta 19h\n"
                    f"Valor total mudou {delta*100:.1f}%\n"
                    f"Anterior: ${pv:,.2f} | Atual: ${cv:,.2f}\n"
                    f"Posições: {((curr or {}).get('metrics') or {}).get('positions_count', 0)}"
                )
                send_telegram_message(utils_msg)
    except Exception:
        pass

def run_weekly_snapshot() -> Dict[str, Any]:
    snap = collect_portfolio_snapshot()
    paths = save_portfolio_snapshot(snap)
    prev = None
    try:
        out_dir = Path(getattr(config, "SNAPSHOT_OUTPUT_DIR", "data/portfolio_snapshots"))
        latest = out_dir / "latest.json"
        if latest.exists():
            prev = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        prev = None
    try:
        compare_and_alert_snapshot(prev, snap)
    except Exception:
        pass
    if not snap.get("complete", True) or snap.get("issues"):
        send_telegram_message("⚠️ Snapshot Sexta 19h com dados parciais ou issues. Verifique logs.")
    else:
        send_telegram_message(f"✅ Snapshot Sexta 19h salvo.\nJSON: {paths.get('json')}\nCSV: {paths.get('positions_csv')}")
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
    return {"snapshot": snap, "paths": paths}

# ✅ ADAPTIVE ENGINE SUPPORT FUNCTIONS
# Funções auxiliares para integração com Adaptive Engine (4-Layer System)

def get_price_data(symbol: str, timeframe: int = mt5.TIMEFRAME_M15, bars: int = 100) -> dict:
    """
    Coleta dados de preço para o Adaptive Engine
    Retorna: dict com preço atual, médias móveis, suporte/resistência
    """
    try:
        real_symbol = normalize_symbol(symbol)
        df = get_rates(real_symbol, timeframe, bars)
        if df is None or df.empty:
            return {"error": True, "message": "Dados de preço indisponíveis"}
        
        current_price = df['close'].iloc[-1]
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        
        # Calcula médias móveis simples
        sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
        sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else current_price
        
        # Identifica suporte/resistência básico (últimos 20 períodos)
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        return {
            "current_price": float(current_price),
            "high": float(high),
            "low": float(low),
            "sma_20": float(sma_20),
            "sma_50": float(sma_50),
            "resistance": float(recent_high),
            "support": float(recent_low),
            "price_change_24h": float((current_price - df['close'].iloc[-96]) / df['close'].iloc[-96] * 100) if len(df) >= 96 else 0.0
        }
    except Exception as e:
        logger.error(f"Erro ao obter dados de preço para {symbol}: {e}")
        return {"error": True, "message": str(e)}

def get_volatility(symbol: str, timeframe: int = mt5.TIMEFRAME_M15, bars: int = 100) -> dict:
    """
    Coleta dados de volatilidade para o Adaptive Engine
    Retorna: dict com ATR, volatilidade histórica, regime de volatilidade
    """
    try:
        real_symbol = normalize_symbol(symbol)
        df = get_rates(real_symbol, timeframe, bars)
        if df is None or df.empty or len(df) < 14:
            return {"error": True, "message": "Dados de volatilidade insuficientes"}
        
        # Calcula ATR
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        atr_values = calculate_atr_numba(high, low, close, 14)
        current_atr = float(atr_values[-1]) if len(atr_values) else 0.0
        
        # Calcula volatilidade histórica (desvio padrão dos retornos)
        returns = df['close'].pct_change().dropna()
        volatility_20 = float(returns.tail(20).std() * np.sqrt(96)) if len(returns) >= 20 else 0.0  # Anualizada
        volatility_50 = float(returns.tail(50).std() * np.sqrt(96)) if len(returns) >= 50 else 0.0
        
        # Determina regime de volatilidade
        regime = get_volatility_regime(symbol, df)
        
        return {
            "atr": current_atr,
            "volatility_20d": volatility_20,
            "volatility_50d": volatility_50,
            "regime": regime,
            "atr_percent": float(current_atr / df['close'].iloc[-1] * 100)
        }
    except Exception as e:
        logger.error(f"Erro ao obter volatilidade para {symbol}: {e}")
        return {"error": True, "message": str(e)}

def get_volume_data(symbol: str, timeframe: int = mt5.TIMEFRAME_M15, bars: int = 100) -> dict:
    """
    Coleta dados de volume para o Adaptive Engine
    Retorna: dict com volume atual, médias de volume, ratio de volume
    """
    try:
        real_symbol = normalize_symbol(symbol)
        df = get_rates(real_symbol, timeframe, bars)
        if df is None or df.empty:
            return {"error": True, "message": "Dados de volume indisponíveis"}
        
        current_volume = float(df['tick_volume'].iloc[-1]) if 'tick_volume' in df.columns else 0.0
        avg_volume_20 = float(df['tick_volume'].tail(20).mean()) if len(df) >= 20 else current_volume
        avg_volume_50 = float(df['tick_volume'].tail(50).mean()) if len(df) >= 50 else current_volume
        
        # Calcula ratio de volume
        volume_ratio = float(current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0
        
        # Volume acumulado nas últimas horas (últimos 16 períodos de 15min = 4 horas)
        recent_volume = float(df['tick_volume'].tail(16).sum()) if len(df) >= 16 else current_volume
        
        return {
            "current_volume": current_volume,
            "avg_volume_20": avg_volume_20,
            "avg_volume_50": avg_volume_50,
            "volume_ratio": volume_ratio,
            "recent_volume_4h": recent_volume,
            "volume_trend": "INCREASING" if volume_ratio > 1.2 else "DECREASING" if volume_ratio < 0.8 else "STABLE"
        }
    except Exception as e:
        logger.error(f"Erro ao obter dados de volume para {symbol}: {e}")
        return {"error": True, "message": str(e)}

<<<<<<< HEAD
# ✅ FIM DAS FUNÇÕES ADAPTIVE ENGINE

=======
# ✅ FIM DAS FUNÇÕES ADAPTIVE ENGINE

>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
