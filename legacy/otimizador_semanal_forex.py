# otimizador_semanal_forex.py - Institutional Grade Forex Optimizer
"""
🚀 XP3 PRO - OTIMIZADOR LAND TRADING v7.0
✅ Ray Parallelism | ✅ Bayesian Optuna | ✅ Rolling WFO
✅ Monte Carlo Verification | ✅ ML-Enhanced Trades
✅ Multi-Asset (Majors & High-Calmar Minors)
"""
print("Iniciando imports...", flush=True)

print("--> Carregando OS/Sys/IO...", flush=True)
import os
import sys
import time
import logging
import json
import warnings
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import threading
import concurrent.futures

print("--> Carregando NumPy...", flush=True)
import numpy as np

print("--> Carregando Pandas...", flush=True)
import pandas as pd

print("--> Carregando Optuna...", flush=True)
try:
    import optuna
    print("    ✅ Optuna carregado.", flush=True)
except Exception as e:
    print(f"    ❌ Falha ao carregar Optuna: {e}", flush=True)
    raise

print("--> Carregando Numba...", flush=True)
# Import de Numba movido para dentro de main() com fallback controlado

print("--> Carregando TQDM...", flush=True)
from tqdm import tqdm

print("--> Carregando Scikit-Learn...", flush=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("--> Carregando Configurações (config_forex)...", flush=True)
import config_forex as config

print("--> Carregando Requests/YFinance...", flush=True)
try:
    import requests
    import yfinance as yf
    print("    ✅ Requests/YFinance carregados.", flush=True)
except Exception as e:
    print(f"    ❌ Falha ao carregar Requests/YFinance: {e}", flush=True)

print("--> Configurando warnings/Optuna logging...", flush=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("--> Carregando Otimizador Core (optimizer_optuna_forex)...", flush=True)
# Coloque isso num try/except para garantir que vejamos se falhar aqui
try:
    import optimizer_optuna_forex as optimizer
    print("    ✅ Otimizador carregado.", flush=True)
except Exception as e:
    print(f"    ❌ Falha ao carregar Otimizador: {e}", flush=True)
    raise

# ===========================
# CONFIGURAÇÕES GLOBAIS
# ===========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("XP3_V7")

OPT_OUTPUT_DIR = Path(config.OPTIMIZER_OUTPUT) / f"v7_{datetime.now().strftime('%Y%m%d')}"
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)

DUKASCOPY_DIR = Path("dukascopy_data")
MIN_WIN_RATE = 0.55
MAX_DRAWDOWN = 0.15
RISK_PER_TRADE = 0.01
SLIPPAGE_PIPS = 0.5

def ema_numpy(x, period):
    alpha = 2.0 / (period + 1.0)
    result = np.empty_like(x)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
    return result

def calculate_rsi_numpy(close, period):
    n = len(close)
    rsi = np.zeros(n)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, n):
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        if i < n - 1:
            gain = gains[i] if i < len(gains) else 0.0
            loss = losses[i] if i < len(losses) else 0.0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
    return rsi

def calculate_atr_numpy(high, low, close, period):
    n = len(close)
    atr = np.zeros(n)
    if n < 2:
        return atr
    tr = np.zeros(n - 1)
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i - 1] = max(h_l, h_pc, l_pc)
    if len(tr) >= period:
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr

# Numba JIT Compilation (Global Scope)
try:
    from numba import njit
    ema_numba           = njit(cache=True)(ema_numpy)
    calculate_rsi_numba = njit(cache=True)(calculate_rsi_numpy)
    calculate_atr_numba = njit(cache=True)(calculate_atr_numpy)
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    ema_numba = ema_numpy
    calculate_rsi_numba = calculate_rsi_numpy
    calculate_atr_numba = calculate_atr_numpy

MT5_LOCK = threading.Lock()
MT5_GLOBAL = None

def _mt5_open_session() -> bool:
    global MT5_GLOBAL
    try:
        import MetaTrader5 as mt5
        with MT5_LOCK:
            path = getattr(config, "MT5_TERMINAL_PATH", None)
            ok = mt5.initialize(path=path) if path else mt5.initialize()
        if not ok:
            logger.warning(f"⚠️ MT5 init falhou: {mt5.last_error()}")
            return False
        MT5_GLOBAL = mt5
        return True
    except Exception as e:
        logger.warning(f"⚠️ MT5 init exception: {e}")
        return False

def _mt5_close_session() -> None:
    global MT5_GLOBAL
    if MT5_GLOBAL is None:
        return
    try:
        with MT5_LOCK:
            MT5_GLOBAL.shutdown()
    except Exception:
        pass
    MT5_GLOBAL = None

def run_preflight_checks() -> bool:
    print("\n" + "=" * 80)
    print("🔍 PRE-FLIGHT CHECKS - XP3 PRO v7.0")
    print("=" * 80 + "\n")
    all_ok = True

    py_ver = sys.version_info
    if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 8):
        print(f"❌ Python {py_ver.major}.{py_ver.minor} detectado - Requer 3.8+")
        all_ok = False
    else:
        print(f"✅ Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    required_modules = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'optuna': 'Optuna',
        'numba': 'Numba',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'TQDM',
        'MetaTrader5': 'MetaTrader5',
        'psutil': 'PSUtil',
        'requests': 'Requests',
        'yfinance': 'Yahoo Finance API',
    }
    print("\n📦 Verificando Dependências:")
    for module, desc in required_modules.items():
        try:
            __import__(module)
            print(f"   ✅ {module:20s} - {desc}")
        except ImportError:
            print(f"   ❌ {module:20s} - AUSENTE ({desc})")
            print(f"      → pip install {module}")
            all_ok = False

    required_files = [
        'config_forex.py',
        'optimizer_optuna_forex.py',
        'utils_forex.py',
    ]
    print("\n📄 Verificando Arquivos Necessários:")
    for filepath in required_files:
        if Path(filepath).exists():
            print(f"   ✅ {filepath}")
        else:
            print(f"   ❌ {filepath} - NÃO ENCONTRADO")
            all_ok = False

    if Path('optimizer_optuna_forex.py').exists():
        try:
            import optimizer_optuna_forex as opt_test
            required_funcs = [
                'train_ml_model',
                'predict_ml_model',
                'optimize_with_optuna',
                'run_backtest_with_params',
                'run_monte_carlo',
            ]
            print("\n🔧 Verificando Funções do Optimizer:")
            for func_name in required_funcs:
                if hasattr(opt_test, func_name):
                    print(f"   ✅ {func_name}()")
                else:
                    print(f"   ❌ {func_name}() - AUSENTE")
                    all_ok = False
        except Exception as e:
            print(f"   ❌ Erro ao importar optimizer_optuna_forex: {e}")
            all_ok = False

    print("\n⚙️ Verificando Configurações:")
    try:
        import config_forex as cfg
        critical_attrs = [
            'FOREX_PAIRS',
            'OPTIMIZER_OUTPUT',
            'OPTUNA_N_TRIALS',
            'OPTUNA_TIMEOUT',
        ]
        for attr in critical_attrs:
            if hasattr(cfg, attr):
                value = getattr(cfg, attr)
                print(f"   ✅ {attr:25s} = {value}")
            else:
                print(f"   ❌ {attr:25s} - NÃO DEFINIDO")
                all_ok = False
    except Exception as e:
        print(f"   ❌ Erro ao importar config_forex: {e}")
        all_ok = False

    return all_ok

def _log_startup_metrics(stage: str) -> None:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        rss_mb = p.memory_info().rss / (1024 * 1024)
        logger.info(f"🧠 Startup[{stage}] | RSS={rss_mb:.1f}MB | Threads={p.num_threads()}")
    except Exception:
        pass

def _get_optimizer_symbols() -> List[str]:
    base = list(config.SYMBOL_MAP) if hasattr(config, "SYMBOL_MAP") else list(config.FOREX_PAIRS.keys())

    use_sector = bool(getattr(config, "OPTIMIZER_USE_SECTOR_MAP", True))
    if use_sector:
        sector = str(getattr(config, "MT5_SECTOR_FILTER", "ALL")).upper().strip() or "ALL"
        sector_map = getattr(config, "SECTOR_MAP", None)
        if isinstance(sector_map, dict) and sector in sector_map:
            base = list(sector_map.get(sector) or base)

    use_dynamic = bool(getattr(config, "OPTIMIZER_USE_DYNAMIC_SELECTED_ASSETS", False))
    if use_dynamic:
        try:
            sel_path = Path(getattr(config, "DATA_DIR", "data")) / "selected_assets.json"
            if sel_path.exists():
                payload = json.loads(sel_path.read_text(encoding="utf-8"))
                desired = set(payload.get("symbols", []) or [])
                if desired:
                    base = [s for s in base if s in desired]
        except Exception:
            pass

    return [str(s).strip() for s in base if str(s).strip()]

def _apply_mt5_market_watch_filter(symbols: List[str]) -> None:
    if not bool(getattr(config, "OPTIMIZER_APPLY_MT5_MARKET_WATCH_FILTER", True)):
        return
    try:
        import MetaTrader5 as mt5
        import utils_forex as utils
    except Exception:
        return

    try:
        path = getattr(config, "MT5_TERMINAL_PATH", None)
        if path:
            ok = mt5.initialize(path=path)
        else:
            ok = mt5.initialize()
        if not ok:
            logger.warning(f"⚠️ MT5 init falhou no otimizador: {mt5.last_error()}")
            return
        result = utils.sync_market_watch(symbols)
        if isinstance(result, dict):
            logger.info(
                "🧩 Sector Filter (otimizador) | kept=%d removed=%d missing=%d",
                len(result.get("kept", []) or []),
                len(result.get("removed", []) or []),
                len(result.get("missing", []) or []),
            )
    except Exception as e:
        logger.warning(f"⚠️ Falha ao aplicar filtro Market Watch no otimizador: {e}")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

# ===========================
# DATA LOADING (v7)
# ===========================

YAHOO_TICKER_MAP: Dict[str, str] = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X",
    "AUDJPY": "AUDJPY=X", "EURJPY": "EURJPY=X", "EURGBP": "EURGBP=X",
    "USDTRY": "TRY=X",   "USDZAR": "ZAR=X",    "USDMXN": "MXN=X",
    "XAUUSD": "GC=F",    "XAGUSD": "SI=F",      # futuros são mais confiáveis
    "US30":   "^DJI",    "UK100":  "^FTSE",
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
}


def _aggressive_mt5_symbol_search(symbol: str, mt5) -> Optional[str]:
    base = symbol.upper()
    suffixes = [".m", ".i", ".raw", "Cash", ".pro", ".ecn", ".a", ".b", ".c", ".d", ".e", ".f", ".g", ".h", ".j", ".k", ".l", ".n", ".o", ".p", ".q", ".r", ".s", ".t", ".u", ".v", ".x"]
    prefixes = ["", "FX_", "INDEX_", "CRYPTO_", "COMMODITY_", "STOCK_", "ETF_", "FUTURE_", "FOREX_", "CURRENCY_", "CFD_", "SPOT_"]
    known_remove = [".NAS", ".NYSE", "Cash", ".m", ".i", ".raw", ".pro", ".ecn"]
    candidates = [base]
    for s in known_remove:
        if s in base:
            candidates.append(base.replace(s, ""))
    for p in prefixes:
        for s in suffixes:
            candidates.append(f"{p}{base}{s}")
    candidates.extend([base.replace(".", ""), base.replace("_", ""), base.replace("-", "")])
    reversed_pairs = {"EURUSD": "USDEUR", "GBPUSD": "USDGBP", "USDJPY": "JPYUSD", "USDCAD": "CADUSD", "USDCHF": "CHFUSD", "AUDUSD": "USDAUD", "AUDJPY": "JPYAUD", "EURJPY": "JPYEUR", "EURGBP": "GBPEUR", "GBPJPY": "JPYGBP", "NZDUSD": "USDNZD"}
    if base in reversed_pairs:
        candidates.append(reversed_pairs[base])
    candidates = list(set(candidates))
    logger.info(f"🔍 Tentando {len(candidates)} variações para {symbol}")
    try:
        with MT5_LOCK:
            all_syms = mt5.symbols_get()
    except Exception:
        all_syms = []
    norm_base = base.replace(".", "").replace("_", "").replace("-", "")
    for s in all_syms or []:
        sym_name = str(getattr(s, "name", "")).upper()
        norm_sym = sym_name.replace(".", "").replace("_", "").replace("-", "")
        if norm_base in norm_sym or norm_sym.startswith(norm_base) or norm_sym.endswith(norm_base):
            candidates.append(sym_name)
    candidates = list(set(candidates))
    for candidate in candidates:
        try:
            with MT5_LOCK:
                if mt5.symbol_select(candidate, True):
                    rates = mt5.copy_rates_from_pos(candidate, getattr(mt5, "TIMEFRAME_M15", None), 0, 100)
            if rates is not None and len(rates) > 0:
                logger.info(f"✅ {symbol} resolvido para: {candidate}")
                return candidate
        except Exception:
            continue
    logger.warning(f"❌ Nenhuma variação válida encontrada para {symbol}")
    return None

def _enhanced_yahoo_finance_loader(symbol: str) -> Optional[Dict]:
    try:
        s = symbol.upper()
        t = YAHOO_TICKER_MAP.get(s)
        if not t:
            return None
        for cfg in configs:
            try:
                df = yf.download(t, period=cfg["period"], interval=cfg["interval"], progress=False, threads=True, timeout=15)
            except Exception:
                df = None
            if df is None or len(df) < 1000:
                continue
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "tick_volume"})
            df = df[["open", "high", "low", "close", "tick_volume"]].dropna()
            pip_size = 0.01 if ("JPY" in s or "XAU" in s) else 0.0001
            tick_value = 100.0 if "XAU" in s else (9.0 if "JPY" in s else 10.0)
            return {"df": df, "pip_size": pip_size, "tick_value": tick_value, "source": f"YAHOO_{cfg['interval']}", "spread": 1.5, "news_mask": np.zeros(len(df), dtype=np.bool_)}
        return None
    except Exception:
        return None

def load_data_v7_enhanced(symbol: str) -> Optional[Dict]:
    duka_file = DUKASCOPY_DIR / f"{symbol}_M15.csv"
    if duka_file.exists():
        try:
            df = pd.read_csv(duka_file)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            if len(df) >= 5000:
                pip_size = 0.01 if ("JPY" in symbol or "XAU" in symbol) else 0.0001
                tick_value = 100.0 if "XAU" in symbol else (9.0 if "JPY" in symbol else 10.0)
                df_clean = validate_data_quality(df[["open", "high", "low", "close", "tick_volume"]].dropna())
                return {"df": df_clean, "pip_size": pip_size, "tick_value": tick_value, "source": "DUKASCOPY", "spread": 0.8, "news_mask": np.zeros(len(df_clean), dtype=np.bool_)}
        except Exception:
            pass
    mt5 = MT5_GLOBAL
    if mt5 is not None:
        resolved = _aggressive_mt5_symbol_search(symbol, mt5)
        if resolved:
            attempts = [(getattr(mt5, "TIMEFRAME_M15", None), 50000), (getattr(mt5, "TIMEFRAME_M15", None), 40000), (getattr(mt5, "TIMEFRAME_M30", None), 30000), (getattr(mt5, "TIMEFRAME_M15", None), 20000), (getattr(mt5, "TIMEFRAME_H1", None), 15000)]
            for tf, cnt in attempts:
                if tf is None:
                    continue
                try:
                    with MT5_LOCK:
                        rates = mt5.copy_rates_from_pos(resolved, tf, 0, cnt)
                except Exception:
                    rates = None
                if rates is not None and len(rates) >= 5000:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df.set_index("time", inplace=True)
                    pip_size = 0.01 if ("JPY" in symbol or "XAU" in symbol) else 0.0001
                    tick_value = 100.0 if "XAU" in symbol else (9.0 if "JPY" in symbol else 10.0)
                    df_clean = validate_data_quality(df[["open", "high", "low", "close", "tick_volume"]].dropna())
                    return {"df": df_clean, "pip_size": pip_size, "tick_value": tick_value, "source": "MT5", "spread": 1.2, "news_mask": np.zeros(len(df_clean), dtype=np.bool_)}
    yd = _enhanced_yahoo_finance_loader(symbol)
    if yd:
        yd["df"] = validate_data_quality(yd["df"])
        return yd
    api_key = os.getenv("XP3_MASSIVE_API_KEY") or getattr(config, "MASSIVE_API_KEY", None)
    if api_key:
        try:
            url = f"https://api.massive.com/v1/marketdata/{symbol}/history"
            params = {"interval": "15m", "limit": "50000"}
            headers = {"Authorization": f"Bearer {api_key}"}
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code != 200:
                logger.warning(f"⚠️ Massive API erro {r.status_code} para {symbol}")
            else:
                j = r.json()
                records = j.get("data") or j
                df = pd.DataFrame.from_records(records)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                elif "timestamp" in df.columns:
                    df["time"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
                df.set_index("time", inplace=True)
                if len(df) >= 5000:
                    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "tick_volume"})
                    df = df[["open", "high", "low", "close", "tick_volume"]].dropna()
                    pip_size = 0.01 if ("JPY" in symbol or "XAU" in symbol) else 0.0001
                    tick_value = 100.0 if "XAU" in symbol else (9.0 if "JPY" in symbol else 10.0)
                    df_clean = validate_data_quality(df)
                    return {"df": df_clean, "pip_size": pip_size, "tick_value": tick_value, "source": "MASSIVE", "spread": 1.5, "news_mask": np.zeros(len(df_clean), dtype=np.bool_)}
        except Exception:
            logger.warning(f"⚠️ Falha ao carregar via Massive API para {symbol}")
    logger.error(f"❌ {symbol}: TODAS as fontes de dados falharam")
    return None

def validate_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicados e ordena
    df2 = df.copy()
    df2 = df2[~df2.index.duplicated(keep='last')]
    df2 = df2.sort_index()
    # Preenche pequenos gaps (até 2 candles) com forward-fill
    if isinstance(df2.index, pd.DatetimeIndex) and len(df2) > 0:
        freq = pd.Timedelta(minutes=15)
        full_idx = pd.date_range(start=df2.index[0], end=df2.index[-1], freq=freq)
        df2 = df2.reindex(full_idx)
        df2[['open','high','low','close','tick_volume']] = df2[['open','high','low','close','tick_volume']].ffill(limit=2)
        df2 = df2.dropna()
    # Detecção de outliers: spikes com range > 5x ATR rolling
    try:
        high = df2['high'].values.astype(np.float64)
        low = df2['low'].values.astype(np.float64)
        close = df2['close'].values.astype(np.float64)
        atr = calculate_atr_numpy(high, low, close, 14)
        rng = (df2['high'] - df2['low']).values
        atr_safe = np.where(atr <= 0, np.nan, atr)
        spike_mask = rng > (5.0 * np.nanmedian(atr_safe))
        if np.any(spike_mask):
            window = 500  # ~5 dias de dados M15
            roll_p99 = df2['high'].rolling(window, min_periods=50).quantile(0.99)
            roll_p01 = df2['low'].rolling(window, min_periods=50).quantile(0.01)

            df2.loc[spike_mask, 'high'] = np.minimum(
                df2.loc[spike_mask, 'high'],
                roll_p99.loc[spike_mask].fillna(df2['high'].quantile(0.99))
            )
            df2.loc[spike_mask, 'low'] = np.maximum(
                df2.loc[spike_mask, 'low'],
                roll_p01.loc[spike_mask].fillna(df2['low'].quantile(0.01))
            )
            df2.loc[spike_mask, 'close'] = np.clip(
                df2.loc[spike_mask, 'close'],
                df2.loc[spike_mask, 'low'],
                df2.loc[spike_mask, 'high']
            )
    except Exception:
        pass
    return df2

def validate_minimum_trades(data: Dict, symbol: str, min_trades: int = 10) -> bool:
    df = data["df"]

    # Busca os parâmetros reais do WFO para calcular o mínimo necessário
    train_len = int(os.getenv("XP3_WFO_TRAIN_LEN", "12000"))
    test_len  = int(os.getenv("XP3_WFO_TEST_LEN",  "2000"))
    min_candles = train_len + test_len  # mínimo para 1 fold

    if len(df) < min_candles:
        logger.warning(
            f"⚠️ {symbol}: Insuficiente ({len(df)} candles < {min_candles} necessários para WFO)"
        )
        return False
    nan_ratio = df.isna().sum().sum() / max(1, (len(df) * len(df.columns)))
    if nan_ratio > 0.05:
        logger.warning(f"⚠️ {symbol}: Muitos NaN ({nan_ratio:.2%})")
        return False
    atr = calculate_atr_numpy(df["high"].values, df["low"].values, df["close"].values, 14)
    arr = atr[atr > 0]
    avg_atr = float(np.mean(arr)) if arr.size > 0 else 0.0
    if avg_atr == 0 or np.isnan(avg_atr):
        logger.warning(f"⚠️ {symbol}: ATR = 0")
        return False
    return True

# ===========================
# WORKER PROCESS (v7 - Multiprocessing)
# ===========================

def worker_process_asset(symbol, data):
    """
    Worker que processa um único ativo: WFO + Optuna + MC + ML.
    Recebe os dados já carregados para evitar race conditions no MT5.
    """
    import optimizer_optuna_forex as optimizer
    import config_forex as config
    # Se dados não foram carregados, retorna erro
    if data is None or len(data['df']) < 5000:
        return {"symbol": symbol, "status": "INSUFFICIENT_DATA"}
    
    try:
        df = data['df']
        len_df = len(df)
        # Rolling WFO: múltiplos folds
        train_len = 12000
        test_len = 2000
        try:
            tl_env = int(os.getenv("XP3_WFO_TRAIN_LEN", str(train_len)))
            te_env = int(os.getenv("XP3_WFO_TEST_LEN", str(test_len)))
            if tl_env > 0 and te_env > 0:
                train_len = tl_env
                test_len = te_env
        except Exception:
            pass
        # número máximo de folds baseado no tamanho disponível (cap configurável)
        max_folds_possible = (len_df - (train_len + test_len)) // test_len if len_df >= (train_len + test_len) else 0
        cap = 12
        try:
            cap_env = int(os.getenv("XP3_WFO_MAX_FOLDS", str(cap)))
            if cap_env > 0:
                cap = cap_env
        except Exception:
            pass
        n_folds = int(max(1, min(cap, max_folds_possible)))
        # início da janela deslizante nos últimos meses
        last_start = max(0, len_df - (train_len + test_len * n_folds))
        fold_metrics = []
        fold_params = []
        last_fold_data = None
        for j in range(n_folds):
            tr_start = last_start + j * test_len
            tr_end = tr_start + train_len
            te_start = tr_end
            te_end = te_start + test_len
            if te_end > len_df:
                break
            df_train = df.iloc[tr_start:tr_end]
            df_test = df.iloc[te_start:te_end]
            data_train = data.copy()
            data_train['df'] = df_train
            data_train['news_mask'] = data['news_mask'][tr_start:tr_end]
            ml_model, ml_confidence_train = optimizer.train_ml_model(df_train)
            data_train['ml_confidence'] = ml_confidence_train
            opt_results = optimizer.optimize_with_optuna(data_train, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT)
            best_p = opt_results.get("best_params", {})
            if not best_p:
                continue
            data_oos = data.copy()
            data_oos['df'] = df_test
            data_oos['news_mask'] = data['news_mask'][te_start:te_end]
            data_oos['ml_confidence'] = optimizer.predict_ml_model(ml_model, df_test)
            metrics_oos, t_res_oos, _ = optimizer.run_backtest_with_params(data_oos, best_p)
            if isinstance(metrics_oos, dict):
                m = dict(metrics_oos)
                m['total_trades'] = int(getattr(t_res_oos, 'size', 0))
                fold_metrics.append(m)
                fold_params.append(best_p)
                last_fold_data = (data_oos, best_p)
        if not fold_metrics:
            return {"symbol": symbol, "status": "NO_TRIALS", "message": "Nenhum trial bem sucedido nos folds"}
        # agregação de métricas
        wr = float(np.mean([m.get('win_rate', 0.0) for m in fold_metrics]))
        pf = float(np.mean([m.get('profit_factor', 0.0) for m in fold_metrics]))
        dd = float(np.mean([m.get('drawdown', 0.0) for m in fold_metrics]))
        sharpe = float(np.mean([m.get('sharpe', 0.0) for m in fold_metrics]))
        total_trades = int(np.sum([m.get('total_trades', 0) for m in fold_metrics]))
        # estabilidade (desvio padrão)
        wr_std = float(np.std([m.get('win_rate', 0.0) for m in fold_metrics]))
        pf_std = float(np.std([m.get('profit_factor', 0.0) for m in fold_metrics]))
        dd_std = float(np.std([m.get('drawdown', 0.0) for m in fold_metrics]))
        # parâmetros médios (inteiros arredondados)
        agg_params = {}
        keys = ['ema_short','ema_long','rsi_low','rsi_high','adx_threshold','sl_atr','tp_atr','ml_threshold']
        for k in keys:
            vals = [p[k] for p in fold_params if k in p]
            if not vals:
                continue
            mean_v = float(np.mean(vals))
            if k in ['ema_short','ema_long','rsi_low','rsi_high','adx_threshold']:
                agg_params[k] = int(round(mean_v))
            else:
                agg_params[k] = float(mean_v)
        metrics_oos = {
            "win_rate": wr,
            "profit_factor": pf,
            "drawdown": dd,
            "sharpe": sharpe,
            "total_trades": total_trades,
            "wr_std": wr_std,
            "pf_std": pf_std,
            "dd_std": dd_std,
        }
        # Monte Carlo na última dobra para risco
        mc_dd_95 = None
        try:
            if last_fold_data is not None:
                lf_data, lf_params = last_fold_data
                _, t_res_lf, _ = optimizer.run_backtest_with_params(lf_data, lf_params)
                mc_dd_95 = optimizer.run_monte_carlo(t_res_lf)
        except Exception:
            mc_dd_95 = None
        # métricas de erro na última dobra
        def _compute_error_metrics(df_eval, probs_eval, horizon=optimizer.ML_HORIZON):
            try:
                close = df_eval["close"].values.astype(np.float64)
                future_close = np.roll(close, -horizon)
                future_ret = (future_close - close) / close
                min_ret = float(getattr(config, "ML_MIN_RETURN", 0.0005))
                valid_len = len(close) - horizon
                if valid_len <= 0:
                    return {"mae": None, "rmse": None, "mape": None, "accuracy": None}
                y_bin = np.full(valid_len, -1, dtype=np.int8)
                y_bin[future_ret[:valid_len] >= min_ret] = 1
                y_bin[future_ret[:valid_len] <= -min_ret] = 0
                probs_valid = np.asarray(probs_eval[:valid_len], dtype=np.float64)
                mask = y_bin >= 0
                if not np.any(mask):
                    return {"mae": None, "rmse": None, "mape": None, "accuracy": None}
                y_true = y_bin[mask].astype(np.float64)
                y_pred = np.clip(probs_valid[mask], 0.0, 1.0)
                err = np.abs(y_pred - y_true)
                mae = float(np.mean(err))
                rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                eps = 1e-6
                mape = float(np.mean(err / (np.maximum(eps, y_true))))
                acc = float(np.mean((y_pred >= 0.5).astype(int) == y_true.astype(int)))
                return {"mae": mae, "rmse": rmse, "mape": mape, "accuracy": acc}
            except Exception:
                return {"mae": None, "rmse": None, "mape": None, "accuracy": None}
        err_metrics = {}
        try:
            if last_fold_data is not None:
                lf_data, _lf_params = last_fold_data
                df_eval = lf_data['df']
                probs_eval = lf_data.get('ml_confidence', np.zeros(len(df_eval)))
                err_metrics = _compute_error_metrics(df_eval, probs_eval)
        except Exception:
            err_metrics = {}
        # Testes A/B na última dobra
        tuned_p = dict(agg_params)
        try:
            tuned_p["ema_long"] = int(min(200, int(tuned_p.get("ema_long", 100)) * 1.10))
            tuned_p["adx_threshold"] = int(min(30, int(tuned_p.get("adx_threshold", 15)) + 3))
            tuned_p["ml_threshold"] = float(min(0.80, float(tuned_p.get("ml_threshold", 0.60)) + 0.05))
        except Exception:
            pass
        # usar a última dobra para A/B
        if last_fold_data is not None:
            lf_data, lf_params = last_fold_data
            ab_metrics_A, ab_tres_A, _ = optimizer.run_backtest_with_params(lf_data, lf_params)
            ab_metrics_B, ab_tres_B, _ = optimizer.run_backtest_with_params(lf_data, tuned_p)
        else:
            ab_metrics_A, ab_tres_A, ab_metrics_B, ab_tres_B = {}, np.array([]), {}, np.array([])
        ab_test = {
            "A": {"params": fold_params[-1] if fold_params else {}, "metrics": ab_metrics_A},
            "B": {"params": tuned_p, "metrics": ab_metrics_B},
        }
        return {
            "symbol": symbol,
            "best_params": agg_params,
            "metrics_oos": metrics_oos,
            "mc_drawdown_95": mc_dd_95,
            "error_metrics": err_metrics,
            "ab_test": ab_test,
            "wfo_folds": {
                "count": len(fold_metrics),
                "per_fold": fold_metrics,
            },
            "n_folds": int(len(fold_metrics)),
            "status": "SUCCESS",
            "source": data['source']
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        return {"symbol": symbol, "status": "ERROR", "message": str(e), "traceback": error_trace}

# ===========================
# MAIN EXECUTION (v7)
# ===========================

def main():
    print("="*80, flush=True)
    print("🚀 XP3 PRO - OTIMIZADOR LAND TRADING v7.0 (Multiprocessing)", flush=True)
    print("="*80, flush=True)

    print("Importando optimizer_optuna_forex...", flush=True)
    t0 = time.time()
    try:
        global optimizer
        import optimizer_optuna_forex as optimizer
        took = time.time() - t0
        print(f"✅ optimizer_optuna_forex importado em {took:.2f}s", flush=True)
        required_funcs = [
            'train_ml_model',
            'predict_ml_model',
            'optimize_with_optuna',
            'run_backtest_with_params',
            'run_monte_carlo',
        ]
        for _fn in required_funcs:
            if not hasattr(optimizer, _fn):
                print(f"❌ Função ausente no optimizer: {_fn}", flush=True)
                return
    except Exception as e:
        print(f"❌ Falha ao importar optimizer_optuna_forex: {e}", flush=True)
        return

    if not run_preflight_checks():
        print("❌ Pre-flight falhou. Corrija os itens acima e execute novamente.", flush=True)
        return

    use_numba = _NUMBA_AVAILABLE and os.getenv("XP3_DISABLE_NUMBA", "").strip().lower() not in ("1", "true", "yes")
    if use_numba:
        print("--> Numba já compilado em nível de módulo.", flush=True)
        print("Aquecendo motor Numba (JIT)... aguarde.", flush=True)
    else:
        print("⚠️ Numba desativado ou indisponível; usando fallback NumPy.", flush=True)

    x = np.array([1.0, 1.1, 1.2, 1.15, 1.18, 1.22], dtype=np.float64)
    _ = ema_numba(x, 3)
    _ = calculate_rsi_numba(x, 3)
    _ = calculate_atr_numba(x, x, x, 3)

    start_ts = time.time()
    _log_startup_metrics("boot")

    all_symbols = _get_optimizer_symbols()
    logger.info(f"📌 Universo do otimizador: {len(all_symbols)} símbolos")

    _apply_mt5_market_watch_filter(all_symbols)
    _log_startup_metrics("after_mt5_filter")
    
    # 1. Carregamento Sequencial (Safety First)
    print(f"📥 Carregando dados para {len(all_symbols)} ativos...", flush=True)
    tasks = []

    # Abrir sessão MT5 uma única vez para acelerar carregamento
    mt5_ready = _mt5_open_session()
    if not mt5_ready:
        logger.warning("⚠️ MT5 indisponível; apenas Dukascopy será usado onde houver CSV.")
    else:
        if os.getenv("XP3_ALIAS_REPORT", "").strip() in ("1", "true", "TRUE", "yes", "YES"):
            try:
                with MT5_LOCK:
                    mt5 = MT5_GLOBAL
                    syms = mt5.symbols_get()
                print("🔎 Alias Report (MT5):", flush=True)
                for s in all_symbols:
                    base = s.upper().replace(".", "").replace("_", "")
                    matches = []
                    for info in syms or []:
                        name = str(getattr(info, "name", "")).upper()
                        norm = name.replace(".", "").replace("_", "")
                        if base in norm or norm.startswith(base):
                            matches.append(info.name)
                        if len(matches) >= 5:
                            break
                    print(f"  {s} -> {matches[:5]}", flush=True)
            except Exception as e:
                print(f"⚠️ Alias Report falhou: {e}", flush=True)
    
    for s in all_symbols:
        try:
            d = load_data_v7_enhanced(s)
            if d and len(d['df']) >= 5000 and validate_minimum_trades(d, s):
                tasks.append((s, d))
                print(f"   ✅ {s} carregado ({len(d['df'])} candles)", flush=True)
            elif d:
                print(f"   ⚠️ {s} insuficiente (<5000 candles)", flush=True)
            else:
                print(f"   ❌ {s} falhou ao carregar", flush=True)
        except Exception as e:
            print(f"   ❌ {s} erro: {e}", flush=True)
            logger.exception(f"Erro ao carregar {s}")

    if not tasks:
        print("Nenhum dado carregado. Encerrando.", flush=True)
        _mt5_close_session()
        return

    logger.info(f"⏱️ Startup: carregamento de dados concluído em {time.time() - start_ts:.1f}s | ativos_ok={len(tasks)}")
    _log_startup_metrics("after_data_load")

    if os.getenv("XP3_OPTIMIZER_STARTUP_ONLY", "").strip() in ("1", "true", "TRUE", "yes", "YES"):
        logger.info("🧪 XP3_OPTIMIZER_STARTUP_ONLY ativo: encerrando após startup.")
        _mt5_close_session()
        return

    # 2. Processamento Paralelo (CPU Bound)
    print(f"\n⚙️ Iniciando otimização paralela ({os.cpu_count()} núcleos)...", flush=True)
    results = []
    
    max_workers = min(os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1, len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(worker_process_asset, s, d): s for s, d in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Otimizando"):
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                print(f"Task gerou exceção: {exc}")

    valid_results = [r for r in results if r['status'] == 'SUCCESS']
    significant_results = [r for r in valid_results if int(r['metrics_oos'].get('total_trades', 0)) >= 15]
    insufficient_results = [r for r in valid_results if int(r['metrics_oos'].get('total_trades', 0)) < 15]
    
    for r in significant_results:
        m = r['metrics_oos']
        safe_dd = max(float(m.get('drawdown', 0.0)), 0.01)
        trades = int(m.get('total_trades', 0))
        r['rank_score'] = (float(m.get('win_rate', 0.0)) * float(m.get('profit_factor', 0.0)) * float(np.log1p(trades))) / safe_dd
    
    ranked = sorted(significant_results, key=lambda x: x['rank_score'], reverse=True)

    elite = ranked[:12]
    rejected = ranked[12:] + insufficient_results + [r for r in results if r['status'] != 'SUCCESS']

    # Exportação Reports
    timestamp_str = datetime.now().strftime('%Y%m%d')  # <--- Manter só uma.
    full_report_file = OPT_OUTPUT_DIR / f"full_report_{timestamp_str}.txt"  # Remover report_file anterior se não usado.
    json_file = OPT_OUTPUT_DIR / f"elite_params_{timestamp_str}.json"
    
    with open(full_report_file, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write(f"📊 RELATÓRIO COMPLETO DE OTIMIZAÇÃO XP3 PRO v7.0 - {timestamp_str}\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"🔹 TOP 12 ELITE (SELECIONADOS):\n")
        for i, asset in enumerate(elite, 1):
            m = asset['metrics_oos']
            f.write(f"{i:02d}. [ELITE] {asset['symbol']} | Rank: {asset['rank_score']:.2f}\n")
            f.write(f"    WR: {m['win_rate']:6.2%} | PF: {m['profit_factor']:5.2f} | DD: {m['drawdown']:6.2%} | Sharpe: {m['sharpe']:5.2f} | Trades: {int(m.get('total_trades', 0))}\n")
            f.write(f"    Best Params: {asset['best_params']}\n")
            f.write("-" * 80 + "\n")
            
        f.write("\n" + "="*100 + "\n")
        f.write(f"🔸 REJEITADOS / NÃO SELECIONADOS:\n")
        
        # Process Rejected (Ranked but not Elite)
        rejected_ranked = ranked[12:]
        for asset in rejected_ranked:
            m = asset['metrics_oos']
            f.write(f"[REJEITADO] {asset['symbol']} | Rank: {asset.get('rank_score', 0):.2f}\n")
            f.write(f"    WR: {m['win_rate']:6.2%} | PF: {m['profit_factor']:5.2f} | DD: {m['drawdown']:6.2%} | Sharpe: {m['sharpe']:5.2f} | Trades: {int(m.get('total_trades', 0))}\n")
            f.write(f"    Best Params: {asset['best_params']}\n")
            f.write("-" * 80 + "\n")
            
        # Process Failures
        failures = [r for r in results if r['status'] != 'SUCCESS']
        if failures:
            f.write("\n" + "="*100 + "\n")
            f.write(f"❌ FALHAS DE EXECUÇÃO:\n")
            for asset in failures:
                f.write(f"❌ {asset['symbol']}: {asset.get('message', 'Erro Desconhecido')}\n")

    # JSON Export (Only Elite)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({a['symbol']: a['best_params'] for a in elite}, f, indent=4)

    # ✅ Python Format Export (Direct Copy-Paste to config_forex.py)
    python_export_file = OPT_OUTPUT_DIR / f"elite_params_config_{timestamp_str}.py"
    with open(python_export_file, "w", encoding="utf-8") as f:
        f.write("# 📋 COPIE E COLE ESTE BLOCO NO ARQUIVO config_forex.py\n")
        f.write("# Substitua a variável FOREX_PAIRS atual por esta:\n\n")
        f.write("FOREX_PAIRS = {\n")
        for asset in elite:
            symbol = asset['symbol']
            params = asset['best_params']
            f.write(f'    "{symbol}": {{\n')
            for k, v in params.items():
                if isinstance(v, str):
                    f.write(f'        "{k}": "{v}",\n')
                else:
                    f.write(f'        "{k}": {v},\n')
            f.write("    },\n")
        f.write("}\n")
    
    print(f"\n✅ Relatório Completo salvo em: {full_report_file}", flush=True)
    print(f"✅ Arquivo JSON (Elite) salvo em: {json_file}", flush=True)
    print(f"✅ Arquivo Config Python salvo em: {python_export_file}", flush=True)
    # ===========================
    # ✅ Relatórios individuais por ativo (formato solicitado)
    # ===========================
    for r in valid_results:
        symbol = str(r.get("symbol", "UNKNOWN"))
        metrics = r.get("metrics_oos", {}) or {}
        params = r.get("best_params", {}) or {}
        status = str(r.get("status", "ERROR"))
        report_file = OPT_OUTPUT_DIR / f"report_{symbol}_{timestamp_str}.txt"
        try:
            with open(report_file, "w", encoding="utf-8") as rf:
                rf.write(" ================================================== \n")
                rf.write(f" RELATÓRIO FINAL ({symbol}) \n")
                rf.write(" ================================================== \n")
                rf.write(f" Status: {status} \n")
                rf.write(f" Win Rate:      {float(metrics.get('win_rate', 0.0)):.2%} \n")
                rf.write(f" Profit Factor: {float(metrics.get('profit_factor', 0.0)):.2f} \n")
                rf.write(f" Total Trades:  {int(metrics.get('total_trades', 0))} \n")
                rf.write(f" Drawdown:      {float(metrics.get('drawdown', 0.0)):.2%} \n")
                rf.write(f" Sharpe V7:     {float(metrics.get('sharpe', 0.0)):.2f} \n")
                rf.write(" ------------------------------ \n")
                rf.write(" 🏆 Melhores Parâmetros Encontrados: \n")
                rf.write(" {\n")
                for k, v in params.items():
                    if isinstance(v, str):
                        rf.write(f'   "{k}": "{v}", \n')
                    else:
                        rf.write(f'   "{k}": {v} \n')
                rf.write(" } \n\n")
                pf = float(metrics.get('profit_factor', 0.0))
                if status == "SUCCESS" and pf >= 1.0:
                    rf.write(" ✅ SUCESSO: O sistema encontrou convergência lucrativa.\n")
                else:
                    rf.write(" ❌ FALHA NA OTIMIZAÇÃO OU PF < 1.0\n")
            print(f"✅ Relatório individual salvo: {report_file}", flush=True)
        except Exception as e:
            print(f"⚠️ Falha ao salvar relatório individual {symbol}: {e}", flush=True)
    # ===========================
    # ✅ Persistência adicional: resumo semanal para dashboard
    # ===========================
    try:
        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_candidates": len(valid_results),
            "elite_count": len(elite),
            "items": [
                {
                    "symbol": r["symbol"],
                    "rank": float(r.get("rank_score", 0.0)),
                    "win_rate": float(r["metrics_oos"].get("win_rate", 0.0)),
                    "profit_factor": float(r["metrics_oos"].get("profit_factor", 0.0)),
                    "drawdown": float(r["metrics_oos"].get("drawdown", 0.0)),
                    "total_trades": int(r["metrics_oos"].get("total_trades", 0)),
                    "error_metrics": r.get("error_metrics", {}),
                    "ab_test": r.get("ab_test", {})
                }
                for r in ranked
            ]
        }
        os.makedirs("analysis_logs", exist_ok=True)
        with open("analysis_logs/optimizer_weekly.json", "w", encoding="utf-8") as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)
        print("✅ Resumo semanal salvo em analysis_logs/optimizer_weekly.json", flush=True)
    except Exception as e:
        logger.warning(f"Falha ao salvar resumo semanal para dashboard: {e}")


if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
