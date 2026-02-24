# otimizador_semanal_forex.py - VERSÃO v6.0 (DUKASCOPY INTEGRADO)
"""
🚀 XP3 PRO - OTIMIZADOR INSTITUCIONAL v6.0
✅ Integração com Dukascopy (dados históricos perfeitos)
✅ Fallback automático: MT5 → Dukascopy
✅ Validação de qualidade
✅ Pronto para produzir resultados reais
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
import warnings

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️  MetaTrader5 não instalado - usando apenas Dukascopy")
    MT5_AVAILABLE = False
    mt5 = None

import config_forex as config

# ===========================
# CONFIGURAÇÕES E LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("XP3_PRO_OPT")

OPT_OUTPUT_DIR = config.OPTIMIZER_OUTPUT
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)

DUKASCOPY_DIR = Path("dukascopy_data")

SPREAD_ESTIMADO = 1.2
COMISSAO_LOTE = 7.0

MIN_CANDLES = 10000  # Pelo menos 10.000 velas (Land Trading Standard)
MAX_CANDLES = 35040  # 12 meses M15
MIN_VARIATION = 0.00001
MAX_SPREAD_ACCEPTABLE = 5.0

warnings.filterwarnings("ignore", category=UserWarning)
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# ===========================
# AUXILIARES (NUMBA)
# ===========================
@njit
def calculate_atr_numba(high, low, close, period):
    tr = np.zeros_like(close)
    atr = np.zeros_like(close)

    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))

    current_sum = 0.0
    for i in range(1, len(close)):
        current_sum += tr[i]
        if i >= period:
            current_sum -= tr[i-period]
            atr[i] = current_sum / period
        elif i > 0:
            atr[i] = current_sum / i

    return atr

@njit
def calculate_rsi_numba(close, period=14):
    rsi = np.zeros_like(close)
    gains = np.zeros_like(close)
    losses = np.zeros_like(close)

    for i in range(1, len(close)):
        change = close[i] - close[i-1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = abs(change)

    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]

    avg_gain /= period
    avg_loss /= period

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
def calculate_adx_numba(high, low, close, period=14):
    adx = np.zeros_like(close)
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)
    tr = np.zeros_like(close)

    for i in range(1, len(close)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))

    smooth_plus_dm = 0.0
    smooth_minus_dm = 0.0
    smooth_tr = 0.0

    for i in range(1, period + 1):
        smooth_plus_dm += plus_dm[i]
        smooth_minus_dm += minus_dm[i]
        smooth_tr += tr[i]

    for i in range(period, len(close)):
        smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period) + plus_dm[i]
        smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period) + minus_dm[i]
        smooth_tr = smooth_tr - (smooth_tr / period) + tr[i]

        if smooth_tr == 0:
            adx[i] = 0
            continue

        plus_di = 100 * (smooth_plus_dm / smooth_tr)
        minus_di = 100 * (smooth_minus_dm / smooth_tr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0

        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period

    return adx

@njit
def fast_backtest_forex(close, ema_s, ema_l, rsi, adx, atr,
                        bb_upper, bb_lower,
                        rsi_l, rsi_h, adx_t, bb_squeeze_threshold,
                        sl_atr_mult, tp_atr_mult,
                        pip_size, tick_value, spread_pips, commission):
    equity = 1000000.0
    balance_history = [equity]
    win_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    in_position = 0
    entry_price = 0.0
    trades = 0
    spread_cost = (spread_pips * pip_size) / pip_size * tick_value # Custo fixo por trade

    for i in range(1, len(close)):
        if atr[i] <= 0:
            continue

        bb_width = (bb_upper[i] - bb_lower[i]) / close[i]

        if in_position == 0:
            if (ema_s[i] > ema_l[i]) and (rsi[i] < rsi_l) and (adx[i] > adx_t):
                if close[i] <= bb_lower[i] * 1.002:
                    in_position = 1
                    entry_price = close[i] + (spread_pips * pip_size)

            elif (ema_s[i] < ema_l[i]) and (rsi[i] > rsi_h) and (adx[i] > adx_t):
                if close[i] >= bb_upper[i] * 0.998:
                    in_position = -1
                    entry_price = close[i]

        else:
            sl_dist = atr[i] * sl_atr_mult
            tp_dist = atr[i] * tp_atr_mult

            pnl_gross = 0.0
            close_signal = False

            if in_position == 1:
                if close[i] <= (entry_price - sl_dist):
                    pnl_gross = (close[i] - entry_price) / pip_size * tick_value
                    close_signal = True
                elif close[i] >= (entry_price + tp_dist):
                    pnl_gross = (close[i] - entry_price) / pip_size * tick_value
                    close_signal = True

            elif in_position == -1:
                exit_price_w_spread = close[i] + (spread_pips * pip_size)
                if exit_price_w_spread >= (entry_price + sl_dist):
                    pnl_gross = (entry_price - exit_price_w_spread) / pip_size * tick_value
                    close_signal = True
                elif exit_price_w_spread <= (entry_price - tp_dist):
                    pnl_gross = (entry_price - exit_price_w_spread) / pip_size * tick_value
                    close_signal = True

            if close_signal:
                net_profit = pnl_gross - commission - spread_cost
                equity += net_profit
                balance_history.append(equity)
                trades += 1
                
                if net_profit > 0:
                    win_trades += 1
                    gross_profit += net_profit
                else:
                    gross_loss += abs(net_profit)
                    
                in_position = 0

    return balance_history, trades, win_trades, gross_profit, gross_loss
# ===========================
# CARREGAMENTO DUKASCOPY
# ===========================
def load_from_dukascopy(symbol: str) -> Optional[pd.DataFrame]:
    """
    Carrega dados do Dukascopy (M15) diretamente de CSV
    """
    duka_file = DUKASCOPY_DIR / f"{symbol}_M15.csv"

    if not duka_file.exists():
        return None

    try:
        df = pd.read_csv(duka_file)
        df['time'] = pd.to_datetime(df['time'])

        # valida colunas
        required = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
        if not all(col in df.columns for col in required):
            return None

        return df

    except Exception as e:
        logger.warning(f"Erro ao carregar {symbol} do Dukascopy: {e}")
        return None



# ===========================
# VALIDAÇÃO DE DADOS
# ===========================
def validate_data_quality(df: pd.DataFrame, symbol: str, pip_size: float, spread: float) -> Tuple[bool, str]:
    """
    Retorna True/False + razão do problema
    """
    # 1) Tamanho mínimo do histórico
    if len(df) < MIN_CANDLES:
        return False, f"insufficient_candles: {len(df)} < {MIN_CANDLES}"

    # 2) Variação mínima real (evita preço "congelado")
    cdiff = df['close'].diff().abs()
    min_var = cdiff[cdiff > 0].min() if (cdiff > 0).any() else 0.0

    if min_var < MIN_VARIATION:
        return False, f"low_variation: {min_var:.8f}"

    # 3) Spread médio aceitável
    if spread > MAX_SPREAD_ACCEPTABLE:
        return False, f"high_spread: {spread:.2f} pips"

    # 4) Duplicados de candle
    dup = df.duplicated(subset=['time']).sum()
    if dup > len(df) * 0.01:
        return False, f"too_many_duplicates: {dup}"

    # 5) Zeros
    zeros = (df['close'] == 0).sum()
    if zeros > 0:
        return False, f"zero_prices: {zeros}"

    # 6) Gaps extremos
    pct = df['close'].pct_change().abs()
    big_gaps = (pct > 0.05).sum()
    if big_gaps > len(df) * 0.01:
        return False, f"extreme_gaps: {big_gaps}"

    return True, "valid"



# ===========================
# DOWNLOAD (MT5 → fallback Dukascopy)
# ===========================
def download_all_data(symbols: List[str]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Baixa dados usando MT5 → fallback Dukascopy.
    Retorna (data_dict, quality_report)
    """

    print("\n" + "="*80)
    print("📊 FASE 1: Baixando dados (MT5 + Dukascopy Fallback)")
    print("="*80 + "\n")

    data_dict = {}
    quality_report = {}

    for symbol in tqdm(symbols, desc="Baixando dados"):

        df = None
        src = None
        pip_size = 0.0001  # padrão
        tick_value = 10.0  # padrão

        # ===========================
        # 1) TENTATIVA MT5
        # ===========================
        if MT5_AVAILABLE:
            try:
                variants = [
                    symbol,
                    symbol + ".raw",
                    symbol + ".m",
                    symbol + ".cash",
                ]
                valid_name = None

                for v in variants:
                    if mt5.symbol_select(v, True):
                        valid_name = v
                        break

                if valid_name:
                    rates = None
                    import time
                    for attempt in range(3): # ✅ Land Trading: Retry loop
                        rates = mt5.copy_rates_from_pos(valid_name, mt5.TIMEFRAME_M15, 0, MAX_CANDLES)
                        if rates is not None and len(rates) >= MIN_CANDLES:
                            break
                        time.sleep(1)

                    info = mt5.symbol_info(valid_name)
                    if rates is not None and len(rates) >= MIN_CANDLES and info:
                        df = pd.DataFrame(rates)
                        # ✅ Limpeza rigorosa solicitada
                        df = df.dropna()
                        df = df.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'tick_volume': 'float64'})
                        
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        pip_size = float(info.point)
                        tick_value = float(info.trade_tick_value)
                        src = "MT5"

            except Exception as e:
                logger.error(f"❌ {symbol}: erro CRÍTICO MT5: {e}. Cliente Land Trading exige dados do broker.")

        # ===========================
        # 2) FALLBACK → DUKASCOPY (Apenas se MT5 falhar e for aceitável)
        # ===========================
        if df is None:
            logger.warning(f"⚠️ {symbol}: MT5 falhou. Tentando Dukascopy...")
            df = load_from_dukascopy(symbol)
            if df is not None:
                src = "Dukascopy"

                # pip_size padrão por símbolo
                if "JPY" in symbol:
                    pip_size = 0.01
                    tick_value = 9.0
                elif "XAU" in symbol or "GOLD" in symbol:
                    pip_size = 0.1
                    tick_value = 1.0
                else:
                    pip_size = 0.0001
                    tick_value = 10.0

        # ===========================
        # 3) Se nada deu certo
        # ===========================
        if df is None:
            quality_report[symbol] = "no_data_available"
            continue

        # Informações de spread
        avg_spread = df['spread'].mean() if 'spread' in df.columns else 0

        # ===========================
        # 4) Validação automática
        # ===========================
        ok, reason = validate_data_quality(df, symbol, pip_size, avg_spread)

        if not ok:
            quality_report[symbol] = f"{reason} (src={src})"
            continue

        # ===========================
        # Se passou nas validações
        # ===========================
        data_dict[symbol] = {
            "df": df,
            "tick_value": tick_value,
            "point": pip_size,
            "digits": 5 if pip_size == 0.00001 else (3 if pip_size == 0.01 else 4),
            "source": src,
            "spread_avg": avg_spread,
            "candles": len(df),
        }

        quality_report[symbol] = f"valid ({len(df)} candles, spread={avg_spread:.1f}, src={src})"

    print(f"\n✅ Dados válidos: {len(data_dict)} símbolos")
    print(f"❌ Inválidos: {len(symbols) - len(data_dict)} símbolos")

    return data_dict, quality_report
# ===========================
# OTIMIZAÇÃO (WORKER)
# ===========================
def run_optimization(symbol: str, df_train: pd.DataFrame, pip_size: float, tick_value: float) -> Tuple[Dict, float]:
    """
    Otimiza parâmetros usando Optuna
    Retorna (best_params, best_calmar)
    """
    # ✅ Limpeza pré-otimização (Numba Safe)
    df_train = df_train.dropna()
    df_train = df_train.astype({'close': 'float64', 'high': 'float64', 'low': 'float64'})
    
    close = df_train['close'].values.astype(np.float64)
    high = df_train['high'].values.astype(np.float64)
    low = df_train['low'].values.astype(np.float64)

    def objective(trial):
        ema_s = trial.suggest_int("ema_short", 5, 50)
        ema_l = trial.suggest_int("ema_long", 50, 200)
        rsi_l = trial.suggest_int("rsi_low", 20, 40)
        rsi_h = trial.suggest_int("rsi_high", 60, 80)
        adx_t = trial.suggest_int("adx_threshold", 15, 35)
        sl_atr = trial.suggest_float("sl_atr", 1.0, 3.0, step=0.5)
        tp_atr = trial.suggest_float("tp_atr", 1.5, 4.0, step=0.5)
        bb_sq = trial.suggest_float("bb_squeeze_threshold", 0.005, 0.10, step=0.005)

        # Calcula indicadores
        ema_s_vals = df_train['close'].ewm(span=ema_s, adjust=False).mean().values
        ema_l_vals = df_train['close'].ewm(span=ema_l, adjust=False).mean().values
        rsi = calculate_rsi_numba(close, 14)
        adx = calculate_adx_numba(high, low, close, 14)
        atr = calculate_atr_numba(high, low, close, 14)

        sma = df_train['close'].rolling(20).mean()
        std = df_train['close'].rolling(20).std()
        bb_up = (sma + 2 * std).values
        bb_lo = (sma - 2 * std).values

        # Backtest
        eq, trades, win_trades, gross_profit, gross_loss = fast_backtest_forex(
            close, ema_s_vals, ema_l_vals, rsi, adx, atr,
            bb_up, bb_lo,
            rsi_l, rsi_h, adx_t, bb_sq,
            sl_atr, tp_atr,
            pip_size, tick_value,
            SPREAD_ESTIMADO, COMISSAO_LOTE
        )

        if trades < 10:
            return -1.0

        ret = (eq[-1] / eq[0]) - 1
        max_dd = 0.0
        peak = eq[0]

        for val in eq:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        if max_dd == 0 or max_dd > 0.50:
            return -1.0

        calmar = ret / max_dd
        return calmar

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    return study.best_params, study.best_value


# ===========================
# WORKER WFO (WALK FORWARD)
# ===========================
def worker_wfo_forex(sym: str, data_dict: Dict) -> Dict[str, Any]:
    """
    Worker: otimiza em treino, testa em OOS
    """
    try:
        if sym not in data_dict:
            return {"symbol": sym, "error": "data_not_available"}

        data_info = data_dict[sym]
        df = data_info["df"]

        pip_size = data_info["point"]
        tick_value = data_info["tick_value"]

        if pip_size <= 0 or tick_value <= 0:
            return {"symbol": sym, "error": "invalid_pip_or_tick"}

        # Split 70/30
        split_idx = int(len(df) * 0.7)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()

        if len(df_train) < 5000 or len(df_test) < 2000:
            return {"symbol": sym, "error": "insufficient_data_after_split"}

        # Otimiza no treino
        best_params, train_calmar = run_optimization(sym, df_train, pip_size, tick_value)

        if train_calmar < 0:
            return {"symbol": sym, "error": "optimization_failed"}

        # Testa em OOS
        close_test = df_test['close'].values.astype(np.float64)
        high_test = df_test['high'].values.astype(np.float64)
        low_test = df_test['low'].values.astype(np.float64)

        atr_test = calculate_atr_numba(high_test, low_test, close_test, 14)

        sma_test = df_test['close'].rolling(20).mean()
        std_test = df_test['close'].rolling(20).std()
        bb_up_test = (sma_test + 2 * std_test).values
        bb_lo_test = (sma_test - 2 * std_test).values

        ema_s_test = df_test['close'].ewm(span=best_params["ema_short"], adjust=False).mean().values
        ema_l_test = df_test['close'].ewm(span=best_params["ema_long"], adjust=False).mean().values
        rsi_test = calculate_rsi_numba(close_test, 14)
        adx_test = calculate_adx_numba(high_test, low_test, close_test, 14)

        eq_test, trades_oos, win_oos, profit_oos, loss_oos = fast_backtest_forex(
            close_test, ema_s_test, ema_l_test, rsi_test, adx_test, atr_test,
            bb_up_test, bb_lo_test,
            best_params["rsi_low"], best_params["rsi_high"],
            best_params["adx_threshold"], best_params["bb_squeeze_threshold"],
            best_params["sl_atr"], best_params["tp_atr"],
            pip_size, tick_value,
            SPREAD_ESTIMADO, COMISSAO_LOTE
        )

        if trades_oos < 5:
            return {"symbol": sym, "error": "too_few_trades_oos"}

        # Métricas OOS
        ret_oos = (eq_test[-1] / eq_test[0]) - 1

        max_dd_oos = 0.0
        peak = eq_test[0]
        for val in eq_test:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd_oos:
                max_dd_oos = dd

        if max_dd_oos == 0 or max_dd_oos < 0.01:
            return {"symbol": sym, "error": "invalid_drawdown"}

        if max_dd_oos > 0.50:
            return {"symbol": sym, "error": "excessive_drawdown"}

        calmar_oos = ret_oos / max_dd_oos

        days_oos = len(df_test) / (24 * 4)
        cagr_oos = ((1 + ret_oos) ** (365 / days_oos)) - 1 if days_oos > 0 else 0

        win_rate = win_oos / trades_oos if trades_oos > 0 else 0
        profit_factor = profit_oos / loss_oos if loss_oos > 0 else (profit_oos if profit_oos > 0 else 0)

        return {
            "status": "SUCCESS",
            "symbol": sym,
            "selected_params": best_params,
            "train_calmar": train_calmar,
            "test_metrics": {
                "calmar": calmar_oos,
                "cagr": cagr_oos,
                "return": ret_oos,
                "drawdown": max_dd_oos,
                "win_rate": win_rate,
                "profit_factor": profit_factor
            },
            "trades_oos": trades_oos,
            "source": data_info.get("source", "unknown")
        }

    except Exception as e:
        logger.error(f"❌ Erro no worker para {sym}: {e}")
        return {"symbol": sym, "error": str(e)}


# ===========================
# MAIN
# ===========================
def main():
    global MT5_AVAILABLE
    print("="*80)
    print("🚀 XP3 PRO - OTIMIZADOR INSTITUCIONAL v6.0")
    print("="*80)
    print("📌 MT5 + Dukascopy | Validação automática | Dados perfeitos")
    print("="*80)

    # Inicializa MT5 (se disponível)
    if MT5_AVAILABLE:
        print(f"\n🔍 Tentando inicializar MT5...")
        try:
            if mt5.initialize(path=config.MT5_TERMINAL_PATH):
                print("✅ MT5 inicializado")
            else:
                print("⚠️  MT5 não disponível - usando apenas Dukascopy")
                MT5_AVAILABLE = False
        except Exception as e:
            print(f"⚠️  MT5 erro: {e}")
            MT5_AVAILABLE = False

    # Coleta símbolos
    all_symbols = []
    symbol_to_sector = {}

    for sector, syms in config.SYMBOL_MAP.items():
        for s in syms:
            all_symbols.append(s)
            symbol_to_sector[s] = sector

    print(f"\n✅ {len(all_symbols)} símbolos do SYMBOL_MAP\n")

    # Download com fallback
    data_dict, quality_report = download_all_data(all_symbols)

    # Fecha MT5
    if MT5_AVAILABLE:
        mt5.shutdown()
        print("✅ MT5 fechado\n")

    if not data_dict:
        print("❌ Nenhum dado disponível!")
        return

    # Salva relatório de qualidade
    quality_file = OPT_OUTPUT_DIR / f"data_quality_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(quality_file, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("📊 RELATÓRIO DE QUALIDADE DE DADOS (v6.0)\n")
        f.write("="*100 + "\n\n")

        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mínimo de candles: {MIN_CANDLES} (6 meses M15)\n")
        f.write(f"Spread máximo: {MAX_SPREAD_ACCEPTABLE} pips\n\n")

        valid = sum(1 for v in quality_report.values() if v.startswith("valid"))
        invalid = len(quality_report) - valid

        f.write(f"✅ Válidos: {valid}\n")
        f.write(f"❌ Inválidos: {invalid}\n\n")

        f.write("="*100 + "\n")
        f.write("DETALHES\n")
        f.write("="*100 + "\n\n")

        for sym in sorted(quality_report.keys()):
            status = quality_report[sym]
            f.write(f"{sym:15s}: {status}\n")

    print(f"✅ Relatório de qualidade: {quality_file}\n")

    # Otimização paralela
    print("\n" + "="*80)
    print("📊 FASE 2: Otimização paralela (sem MT5)")
    print("="*80 + "\n")

    final_results = []
    all_attempts = [] # ✅ NOVO: Captura tudo para o dashboard
    errors_by_type = defaultdict(int)

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(worker_wfo_forex, sym, data_dict): sym
            for sym in data_dict.keys()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Otimizando"):
            res = future.result()
            all_attempts.append(res) # Guarda tudo

            if "status" in res:
                final_results.append(res)
            elif "error" in res:
                error_type = res["error"]
                errors_by_type[error_type] += 1

    # Resumo de erros
    print("\n" + "="*80)
    print("📊 RESUMO DE ERROS:")
    print("="*80)
    for error_type, count in sorted(errors_by_type.items()):
        print(f"  • {error_type}: {count} símbolos")
    print("="*80)

    # Classificação
    elite_portfolio = {}
    candidate_portfolio = {}

    for asset in final_results:
        calmar_oos = asset["test_metrics"]["calmar"]

        if calmar_oos > 0.3:
            elite_portfolio[asset["symbol"]] = asset
        elif 0.15 < calmar_oos <= 0.3:
            candidate_portfolio[asset["symbol"]] = asset

    # Relatório por setor
    print("\n" + "="*80)
    print("--- RESULTADOS AUDITADOS POR SETOR ---")
    print("="*80)

    for sector in sorted(set(symbol_to_sector.values())):
        print(f"\n📊 SETOR: {sector}")
        print("-"*80)

        sector_assets = [a for a in final_results if symbol_to_sector.get(a["symbol"]) == sector]

        if sector_assets:
            for asset in sorted(sector_assets, key=lambda x: x["test_metrics"]["calmar"], reverse=True):
                m = asset["test_metrics"]
                src = asset.get("source", "?")
                print(f"  {asset['symbol']:12s}: Calmar={m['calmar']:6.2f} | CAGR={m['cagr']:8.1%} | DD={m['drawdown']:6.1%} | Trades={asset['trades_oos']:3d} | Src={src}")
        else:
            print("  (Nenhum ativo aprovado)")

    # Salva relatório final
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = OPT_OUTPUT_DIR / f"elite_settings_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("🏆 ELITE PORTFOLIO (v6.0 - DUKASCOPY)\n")
        f.write("="*100 + "\n\n")

        for sector in sorted(set(symbol_to_sector.values())):
            sector_elite = {k: v for k, v in elite_portfolio.items() if symbol_to_sector.get(k) == sector}
            if not sector_elite:
                continue

            f.write(f"\n--- {sector} ---\n")
            for sym, asset in sorted(sector_elite.items(), key=lambda x: x[1]['test_metrics']['calmar'], reverse=True):
                m = asset['test_metrics']
                p = asset['selected_params']
                src = asset.get('source', 'unknown')

                f.write(f"\n{sym}:\n")
                f.write(f"  Fonte: {src}\n")
                f.write(f"  Métricas Out-of-Sample:\n")
                f.write(f"    • Calmar Ratio:    {m['calmar']:8.2f}\n")
                f.write(f"    • CAGR:            {m['cagr']:8.1%}\n")
                f.write(f"    • Retorno Total:   {m['return']:8.1%}\n")
                f.write(f"    • Max Drawdown:    {m['drawdown']:8.1%}\n")
                f.write(f"    • Trades OOS:      {asset['trades_oos']:8d}\n")
                f.write(f"    • Calmar Train:    {asset['train_calmar']:8.2f}\n")
                f.write(f"  Parâmetros:\n")
                f.write(f"    • EMA Short:       {p['ema_short']:8d}\n")
                f.write(f"    • EMA Long:        {p['ema_long']:8d}\n")
                f.write(f"    • RSI Low:         {p['rsi_low']:8d}\n")
                f.write(f"    • RSI High:        {p['rsi_high']:8d}\n")
                f.write(f"    • ADX Threshold:   {p['adx_threshold']:8d}\n")
                f.write(f"    • SL ATR Multi:    {p['sl_atr']:8.1f}\n")
                f.write(f"    • TP ATR Multi:    {p['tp_atr']:8.1f}\n")
                f.write(f"    • BB Squeeze:      {p['bb_squeeze_threshold']:8.3f}\n")
                f.write(f"  {'-'*80}\n")

        f.write("\n\n" + "="*100 + "\n")
        f.write("📝 CONFIGURAÇÃO PYTHON (ELITE)\n")
        f.write("="*100 + "\n\n")
        f.write("ELITE_CONFIG = {\n")
        for sym in sorted(elite_portfolio.keys()):
            asset = elite_portfolio[sym]
            m = asset['test_metrics']
            f.write(f"    '{sym}': {asset['selected_params']},  # Calmar: {m['calmar']:.2f} | CAGR: {m['cagr']:.1%} | DD: {m['drawdown']:.1%}\n")
        f.write("}\n")

    # ✅ NOVO: Salva backtest_results.json para o ML Optimizer (Full Report)
    backtest_results_file = OPT_OUTPUT_DIR / "backtest_results.json"
    backtest_data = {}
    
    for res in all_attempts:
        sym = res["symbol"]
        if "status" in res:
            win_rate = res["test_metrics"]["win_rate"]
            backtest_data[sym] = {
                "win_rate": win_rate,
                "profit_factor": res["test_metrics"]["profit_factor"],
                "score": int(win_rate * 100),
                "status": "valid"
            }
        else:
            backtest_data[sym] = {
                "win_rate": 0,
                "profit_factor": 0,
                "score": 0,
                "status": f"error: {res.get('error', 'unknown')}"
            }
    
    with open(backtest_results_file, "w", encoding="utf-8") as f:
        json.dump(backtest_data, f, indent=4)
    
    print(f"✅ Resultados de backtest salvos para ML: {backtest_results_file}")

    print("\n" + "="*80)
    print(f"✨ Relatório completo salvo em: {output_file}")
    print(f"   • Elite: {len(elite_portfolio)} ativos")
    print(f"   • Candidatos: {len(candidate_portfolio)} ativos")
    print(f"   • Total processado: {len(final_results)} ativos")
    print("="*80)

if __name__ == "__main__":
    main()
