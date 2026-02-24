# optimizer_optuna_forex.py - VERSÃO CORRIGIDA
import optuna
import logging
import numpy as np
import pandas as pd
import config_forex as config
import utils_forex as utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
ML_HORIZON = 4

# Custos Fixos para Simulação
# ===========================
# INDICADORES (NUMBA - ULTRA FAST)
# ===========================

from numba import njit
import numpy as np
import pandas as pd

@njit
def sma_numba(data, period):
    res = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        res[i] = np.mean(data[i - period + 1 : i + 1])
    return res

@njit
def ema_numba(data, period):
    res = np.zeros_like(data)
    alpha = 2.0 / (period + 1.0)
    res[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        res[i] = (data[i] - res[i - 1]) * alpha + res[i - 1]
    return res

@njit
def calculate_atr_numba(high, low, close, period):
    tr = np.zeros_like(close)
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    
    alpha = 1.0 / period
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(close)):
        atr[i] = (tr[i] - atr[i-1]) * alpha + atr[i-1]
    return atr

@njit
def calculate_rsi_numba(close, period=14):
    rsi = np.zeros_like(close)
    gains = np.zeros_like(close)
    losses = np.zeros_like(close)
    for i in range(1, len(close)):
        change = close[i] - close[i-1]
        if change > 0: gains[i] = change
        else: losses[i] = abs(change)
    
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0: rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
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
        if high_diff > low_diff and high_diff > 0: plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0: minus_dm[i] = low_diff
        tr[i] = max(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    # Correção: Adicionar min_length check antes de ema
    if len(plus_dm) < period * 2:  # <--- Adicionado: Skip se data muito curta.
        return np.zeros_like(close)  # Retornar zeros para ADX.
    smooth_pdm = ema_numba(plus_dm, period)
    smooth_mdm = ema_numba(minus_dm, period)
    smooth_tr = ema_numba(tr, period)

    dx = np.zeros_like(close)
    for i in range(period, len(close)):
        if smooth_tr[i] == 0: continue
        pdi = 100 * smooth_pdm[i] / smooth_tr[i]
        mdi = 100 * smooth_mdm[i] / smooth_tr[i]
        if (pdi + mdi) == 0: dx[i] = 0
        else: dx[i] = 100 * abs(pdi - mdi) / (pdi + mdi)
    
    adx = ema_numba(dx, period)
    return adx

@njit
def calculate_macd_numba(close, fast=12, slow=26, signal=9):
    ema_f = ema_numba(close, fast)
    ema_s = ema_numba(close, slow)
    macd_line = ema_f - ema_s
    signal_line = ema_numba(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

@njit
def check_rsi_divergence_numba(close, rsi, window=5):
    if len(close) < window * 2: return 0
    if close[-1] < np.min(close[-window-1:-1]) and rsi[-1] > np.min(rsi[-window-1:-1]):
        return 1
    if close[-1] > np.max(close[-window-1:-1]) and rsi[-1] < np.max(rsi[-window-1:-1]):
        return -1
    return 0

# ===========================
# MOTOR DE BACKTEST (v7)
# ===========================

@njit
def fast_backtest_v7(
    time_indices, open_arr, close, high, low,
    ema_s, ema_l, rsi, adx, atr, macd_hist,
    vol_rel, session_code,
    rsi_l, rsi_h, adx_t,
    sl_atr_mult, tp_base_mult,
    pip_size, tick_value, spread_pips_base, commission,
    news_mask,
    ml_confidence, ml_threshold,
    time_exit_bars,
    slippage_base_pips,
    slippage_atr_factor,
    slippage_volume_factor,
    slippage_order_factor,
    news_slippage_mult,
    news_spread_mult,
    news_sl_mult,
    news_tp_mult,
    asian_slip_mult,
    golden_slip_mult,
    normal_slip_mult,
    protection_slip_mult
):
    equity = 100000.0
    balance_history = [equity]
    trade_results = [] 
    
    in_position = 0
    entry_price = 0.0
    entry_atr = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_lot = 0.0
    tp_mult = tp_base_mult
    
    trades_count = 0
    win_trades = 0
    entry_index = -1
    entry_risk_pips = 0.0
    durations = []
    rr_list = []
    
    def _session_mult(code):
        if code == 0:
            return normal_slip_mult
        elif code == 1:
            return golden_slip_mult
        elif code == 2:
            return asian_slip_mult
        else:
            return protection_slip_mult
    
    def _dynamic_slippage_pips(i, lot_hint):
        atr_pips = atr[i] / pip_size
        start = 0 if i < 100 else i - 100
        end = i if i > 0 else 1
        base = 0.0
        cnt = 0
        for k in range(start, end):
            ap = atr[k] / pip_size
            if ap > 0:
                base += ap
                cnt += 1
        base = (base / cnt) if cnt > 0 else max(atr_pips, 1.0)
        atr_norm = atr_pips / base if base > 0 else 1.0
        vol_ratio = vol_rel[i] if vol_rel[i] > 0 else 1.0
        liq_penalty = (1.0 - vol_ratio) if vol_ratio < 1.0 else 0.0
        lot_component = 0.0
        if lot_hint > 0:
            lot_component = np.log10(1.0 + lot_hint)
        slip = slippage_base_pips
        slip += slippage_atr_factor * atr_norm
        slip += slippage_volume_factor * liq_penalty
        slip += slippage_order_factor * lot_component
        slip *= _session_mult(session_code[i])
        if news_mask[i]:
            slip *= news_slippage_mult
        if slip < 0.0:
            slip = 0.0
        return slip
    
    start_i = max(100, int(len(close) * 0.05))
    for i in range(start_i, len(close)):
        atr_avg = 0.0
        if i > 100:
            atr_avg = np.mean(atr[i-100:i])
        
        current_time_hour = time_indices[i]
        
        if in_position == 0:
            # Buy
            if (ema_s[i] > ema_l[i]) and (rsi[i] < rsi_l) and (adx[i] > adx_t) and (ml_confidence[i] >= ml_threshold) and (adx[i] >= 10.0 if ml_threshold > 0.0 else True):
                in_position = 1
                eff_sl_mult = sl_atr_mult * (news_sl_mult if news_mask[i] else 1.0)
                tp_mult = (tp_base_mult if adx[i] <= 30 else 3.0) * (news_tp_mult if news_mask[i] else 1.0)
                spread_pips_eff = spread_pips_base * (news_spread_mult if news_mask[i] else 1.0)
                risk_pips_hint = (atr[i] * eff_sl_mult) / pip_size
                lot_hint = (equity * 0.01) / (risk_pips_hint * tick_value) if risk_pips_hint > 0 else 0.01
                slip_pips = _dynamic_slippage_pips(i, lot_hint)
                entry_price = close[i] + (spread_pips_eff + slip_pips) * pip_size
                entry_atr = atr[i]
                sl_price = entry_price - (entry_atr * eff_sl_mult)
                tp_price = entry_price + (entry_atr * tp_mult)
                risk_pips = (entry_atr * eff_sl_mult) / pip_size
                entry_lot = (equity * 0.01) / (risk_pips * tick_value) if risk_pips > 0 else 0.01
                entry_index = i
                entry_risk_pips = risk_pips

            # Sell
            elif (ema_s[i] < ema_l[i]) and (rsi[i] > rsi_h) and (adx[i] > adx_t) and (ml_confidence[i] <= (1.0 - ml_threshold)) and (adx[i] >= 10.0 if ml_threshold > 0.0 else True):
                in_position = -1
                eff_sl_mult = sl_atr_mult * (news_sl_mult if news_mask[i] else 1.0)
                tp_mult = (tp_base_mult if adx[i] <= 30 else 3.0) * (news_tp_mult if news_mask[i] else 1.0)
                spread_pips_eff = spread_pips_base * (news_spread_mult if news_mask[i] else 1.0)
                risk_pips_hint = (atr[i] * eff_sl_mult) / pip_size
                lot_hint = (equity * 0.01) / (risk_pips_hint * tick_value) if risk_pips_hint > 0 else 0.01
                slip_pips = _dynamic_slippage_pips(i, lot_hint)
                entry_price = close[i] - (spread_pips_eff + slip_pips) * pip_size
                entry_atr = atr[i]
                sl_price = entry_price + (entry_atr * eff_sl_mult)
                tp_price = entry_price - (entry_atr * tp_mult)
                risk_pips = (entry_atr * eff_sl_mult) / pip_size
                entry_lot = (equity * 0.01) / (risk_pips * tick_value) if risk_pips > 0 else 0.01
                entry_index = i
                entry_risk_pips = risk_pips

        else:
            close_signal = False
            pnl_pips = 0.0
            exit_price = 0.0
            
            if in_position == 1:
                # Breakeven
                if high[i] >= entry_price + (entry_atr * tp_mult * 0.5):
                    new_sl = entry_price + (entry_atr * 0.5)
                    if new_sl > sl_price: sl_price = new_sl
                if open_arr[i] <= sl_price:
                    slip_exit = _dynamic_slippage_pips(i, entry_lot)
                    exit_price = open_arr[i] - slip_exit * pip_size
                    pnl_pips = (exit_price - entry_price) / pip_size
                    close_signal = True
                elif open_arr[i] >= tp_price:
                    exit_price = tp_price
                    pnl_pips = (exit_price - entry_price) / pip_size
                    close_signal = True
                
                if low[i] <= sl_price:
                    slip_exit = _dynamic_slippage_pips(i, entry_lot)
                    exit_price = sl_price - slip_exit * pip_size
                    pnl_pips = (exit_price - entry_price) / pip_size
                    close_signal = True
                elif high[i] >= tp_price:
                    exit_price = tp_price
                    pnl_pips = (exit_price - entry_price) / pip_size
                    close_signal = True
                if (low[i] <= sl_price) and (high[i] >= tp_price):
                    dist_sl = abs(open_arr[i] - sl_price)
                    dist_tp = abs(tp_price - open_arr[i])
                    if dist_tp < dist_sl:
                        exit_price = tp_price
                        pnl_pips = (exit_price - entry_price) / pip_size
                    else:
                        slip_exit = _dynamic_slippage_pips(i, entry_lot)
                        exit_price = sl_price - slip_exit * pip_size
                        pnl_pips = (exit_price - entry_price) / pip_size
                    close_signal = True
            
            elif in_position == -1:
                # Breakeven
                if low[i] <= entry_price - (entry_atr * tp_mult * 0.5):
                    new_sl = entry_price - (entry_atr * 0.5)
                    if new_sl < sl_price: sl_price = new_sl
                if open_arr[i] >= sl_price:
                    slip_exit = _dynamic_slippage_pips(i, entry_lot)
                    exit_price = open_arr[i] + slip_exit * pip_size
                    pnl_pips = (entry_price - exit_price) / pip_size
                    close_signal = True
                elif open_arr[i] <= tp_price:
                    exit_price = tp_price
                    pnl_pips = (entry_price - exit_price) / pip_size
                    close_signal = True

                if high[i] >= sl_price:
                    slip_exit = _dynamic_slippage_pips(i, entry_lot)
                    exit_price = sl_price + slip_exit * pip_size
                    pnl_pips = (entry_price - exit_price) / pip_size
                    close_signal = True
                elif low[i] <= tp_price:
                    exit_price = tp_price
                    pnl_pips = (entry_price - exit_price) / pip_size
                    close_signal = True
                if (high[i] >= sl_price) and (low[i] <= tp_price):
                    dist_sl = abs(open_arr[i] - sl_price)
                    dist_tp = abs(open_arr[i] - tp_price)
                    if dist_tp < dist_sl:
                        exit_price = tp_price
                        pnl_pips = (entry_price - exit_price) / pip_size
                    else:
                        slip_exit = _dynamic_slippage_pips(i, entry_lot)
                        exit_price = sl_price + slip_exit * pip_size
                        pnl_pips = (entry_price - exit_price) / pip_size
                    close_signal = True

            if not close_signal and time_exit_bars > 0 and entry_index >= 0:
                if (i - entry_index) >= time_exit_bars:
                    slip_exit = _dynamic_slippage_pips(i, entry_lot)
                    if in_position == 1:
                        exit_price = open_arr[i] - slip_exit * pip_size
                        pnl_pips = (exit_price - entry_price) / pip_size
                    else:
                        exit_price = open_arr[i] + slip_exit * pip_size
                        pnl_pips = (entry_price - exit_price) / pip_size
                    close_signal = True

            if close_signal:
                lot_size = entry_lot if entry_lot > 0 else 0.01
                net_pnl = (pnl_pips * tick_value * lot_size) - (commission * lot_size)
                equity += net_pnl
                balance_history.append(equity)
                trade_results.append(net_pnl)
                trades_count += 1
                if net_pnl > 0: win_trades += 1
                if entry_index >= 0:
                    durations.append(i - entry_index)
                if entry_risk_pips > 0:
                    rr_list.append(abs(pnl_pips) / entry_risk_pips)
                in_position = 0
                entry_lot = 0.0
                entry_index = -1
                entry_risk_pips = 0.0

    return (
        np.array(balance_history),
        trades_count,
        win_trades,
        np.array(trade_results),
        np.array(durations),
        np.array(rr_list),
        len(close) - start_i
    )

def calculate_metrics_v7(trade_results: np.array, balance_history: np.array, durations: np.array = None, rr_list: np.array = None, total_bars: int = None):
    # --- 1. Cálculos Básicos ---
    if len(balance_history) < 2:
        return {"sharpe": 0.0, "profit_factor": 0.0, "score_v7": -100.0, "total_trades": 0}

    ret_total = (balance_history[-1] / balance_history[0]) - 1
    
    # Drawdown (Cálculo Otimizado)
    peak = np.maximum.accumulate(balance_history)
    drawdowns = (peak - balance_history) / peak
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # --- 2. Correção Crítica do Sharpe ---
    # Calculamos o Sharpe sobre os RETORNOS DOS TRADES, não sobre o tempo.
    # Isso evita o erro de anualização incorreta.
    trades_np = np.array(trade_results, dtype=np.float64)
    
    if len(trades_np) > 1:
        # Sharpe Simples por Trade (Média / Desvio)
        mean_trade = np.mean(trades_np)
        std_trade = np.std(trades_np)
        
        # Proteção contra divisão por zero e números minúsculos
        if std_trade > 1e-9:
            # Sharpe "Bruto" (sem anualizar, foco na consistência do setup)
            sharpe = float(mean_trade / std_trade)
            
            # Se quiser anualizar (assumindo N trades por ano), seria: sharpe * sqrt(252 * trades_per_day)
            # Mas para otimização, o Sharpe bruto é mais honesto.
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # --- 3. Métricas de Performance ---
    wins = trades_np[trades_np > 0]
    losses = np.abs(trades_np[trades_np < 0])
    
    win_rate = (len(wins) / len(trades_np)) if len(trades_np) > 0 else 0.0
    
    # Profit Factor (com proteção de Cap)
    loss_sum = np.sum(losses)
    p_factor = (np.sum(wins) / loss_sum) if loss_sum > 0 else 5.0
    p_factor = min(p_factor, 10.0) # Cap para evitar distorções no score

    # Calmar Ratio
    calmar = (ret_total / max_dd) if max_dd > 0.01 else 0.0

    # --- 4. Score V7 (Simplificado e Robusto) ---
    # Prioriza: PF > 1.2, DD < 10%, WR > 40%
    score = (p_factor * 10) + (sharpe * 5)
    
    # Penalidades
    if max_dd > 0.15: score *= 0.5    # Penalidade Severa por DD alto
    if win_rate < 0.35: score *= 0.7  # Penalidade por WR muito baixo
    if len(trades_np) < 10: score *= 0.4 # Penalidade por falta de amostragem

    # Retorno do Dicionário
    return {
        "retorno_total": float(ret_total),
        "drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "sortino": 0.0, # Desabilitado por redundância
        "win_rate": float(win_rate),
        "profit_factor": float(p_factor),
        "calmar": float(calmar),
        "score_v7": float(score),
        "total_trades": int(len(trades_np)),
        "trades_per_day": float(len(trades_np) / (total_bars / 288)) if total_bars else 0.0 # Ajustado para M5 (288 barras/dia)
    }

def build_ml_features(df: pd.DataFrame):
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        numerator = np.asarray(numerator, dtype=np.float64)
        denominator = np.asarray(denominator, dtype=np.float64)
        out = np.zeros_like(numerator, dtype=np.float64)
        np.divide(numerator, denominator, out=out, where=denominator != 0)
        return out

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    if "tick_volume" in df.columns:
        volume = df["tick_volume"].values.astype(np.float64)
    else:
        volume = np.ones_like(close)

    rsi = calculate_rsi_numba(close, 14)
    adx = calculate_adx_numba(high, low, close, 14)
    atr = calculate_atr_numba(high, low, close, 14)

    macd_line, macd_signal, macd_hist = calculate_macd_numba(close)

    ema200 = ema_numba(close, 200)
    ema200_dist = _safe_divide(close - ema200, ema200)

    sma50 = sma_numba(close, 50)
    sma50_dist = _safe_divide(close - sma50, close)

    candle_range = _safe_divide(high - low, close)

    vol_mean_50 = pd.Series(volume).rolling(50, min_periods=1).mean().values
    vol_rel_50 = _safe_divide(volume, vol_mean_50)

    momentum1 = np.diff(close, prepend=close[0])
    momentum3 = close - np.roll(close, 3)
    momentum10 = close - np.roll(close, 10)

    rsi_window = 5
    rsi_div = np.zeros_like(close)
    for i in range(rsi_window, len(close)):
        price_delta = close[i] - close[i - rsi_window]
        rsi_delta = rsi[i] - rsi[i - rsi_window]
        if price_delta > 0 and rsi_delta < 0:
            rsi_div[i] = -1.0
        elif price_delta < 0 and rsi_delta > 0:
            rsi_div[i] = 1.0

    adx_slope = adx - np.roll(adx, 5)

    close_series = pd.Series(close)
    bb_period = 20
    sma_bb = close_series.rolling(bb_period, min_periods=1).mean().values
    std_bb = close_series.rolling(bb_period, min_periods=1).std(ddof=0).values
    bb_upper = sma_bb + 2.0 * std_bb
    bb_lower = sma_bb - 2.0 * std_bb
    bb_width = _safe_divide(bb_upper - bb_lower, sma_bb)
    band_range = bb_upper - bb_lower
    bb_position = _safe_divide(close - bb_lower, band_range)

    start_idx = 200

    features = np.column_stack(
        (
            rsi,
            adx,
            atr,
            momentum1,
            momentum3,
            momentum10,
            sma50_dist,
            ema200_dist,
            candle_range,
            vol_rel_50,
            macd_line,
            macd_signal,
            macd_hist,
            rsi_div,
            adx_slope,
            bb_width,
            bb_position,
        )
    )
    return features, start_idx

def _compute_volume_ratio(df: pd.DataFrame) -> np.ndarray:
    vol = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.ones(len(df), dtype=np.float64)
    vol_mean = np.zeros_like(vol)
    acc = 0.0
    for i in range(len(vol)):
        acc += vol[i]
        if i >= 50:
            acc -= vol[i - 50]
            vol_mean[i] = acc / 50.0
        else:
            vol_mean[i] = acc / float(i + 1)
    vol_rel = np.zeros_like(vol)
    for i in range(len(vol)):
        if vol_mean[i] != 0:
            vol_rel[i] = vol[i] / vol_mean[i]
        else:
            vol_rel[i] = 1.0
    return vol_rel

def _compute_session_code(hours: np.ndarray) -> np.ndarray:
    # 0=NORMAL, 1=GOLDEN, 2=ASIAN, 3=PROTECTION
    code = np.zeros(len(hours), dtype=np.int32)
    for i in range(len(hours)):
        h = int(hours[i])
        if (h >= 10 and h < 14):
            code[i] = 1
        elif (h >= 22 or h < 5):
            code[i] = 2
        elif (h >= 18 and h < 22):
            code[i] = 3
        else:
            code[i] = 0
    return code

def objective(trial, data_bundle):
    # Retrieve params
    ema_s_period = trial.suggest_int("ema_short", 8, 30)
    ema_l_period = trial.suggest_int("ema_long", 40, 200)
    rsi_low = trial.suggest_int("rsi_low", 20, 50)
    rsi_high = trial.suggest_int("rsi_high", 50, 80)
    try:
        import os as _os
        _adx_min = int(_os.getenv("XP3_OPT_RANGE_ADX_MIN", "5"))
    except Exception:
        _adx_min = 5
    adx_thresh = trial.suggest_int("adx_threshold", max(5, _adx_min), 25)
    sl_atr = trial.suggest_float("sl_atr", 1.0, 3.5, step=0.1)
    tp_atr = trial.suggest_float("tp_atr", 1.5, 5.0, step=0.5)
    try:
        import os as _os2
        _ml_min = float(_os2.getenv("XP3_OPT_RANGE_ML_MIN", "0.5"))
    except Exception:
        _ml_min = 0.5
    ml_threshold = trial.suggest_float("ml_threshold", max(0.5, _ml_min), 0.85, step=0.025)
    
    # In-Sample Training Logic (Using pre-loaded arrays from data_bundle)
    if 'df' not in data_bundle: return -1.0 # Error check
    
    df = data_bundle['df']
    
    # Calculate Indicators on-the-fly (Fast with Numba)
    open_arr = df['open'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    time_indices = df.index.hour.values
    vol_rel = _compute_volume_ratio(df)
    session_code = _compute_session_code(time_indices)
    
    # ML Confidence Access
    ml_confidence = data_bundle.get('ml_confidence')
    if ml_confidence is None:
        ml_confidence = np.zeros(len(close))
    
    ema_s = ema_numba(close, ema_s_period)
    ema_l = ema_numba(close, ema_l_period)
    rsi = calculate_rsi_numba(close, 14) 
    adx = calculate_adx_numba(high, low, close, 14)
    atr = calculate_atr_numba(high, low, close, 14)
    _, _, macd_hist = calculate_macd_numba(close)
    
    # Fallback de gating: se previsões forem praticamente constantes, relaxar threshold
    try:
        if (np.std(ml_confidence) < 1e-6) or (np.count_nonzero(ml_confidence > 0.55) < 10):
            ml_threshold = 0.0
    except Exception:
        pass
    bal, t_cnt, win_cnt, t_res, durations, rr_vals, total_bars = fast_backtest_v7(
        time_indices, open_arr, close, high, low,
        ema_s, ema_l, rsi, adx, atr, macd_hist,
        vol_rel, session_code,
        rsi_low, rsi_high, adx_thresh,
        sl_atr, tp_atr,
        data_bundle['pip_size'], data_bundle['tick_value'],
        data_bundle.get('spread', 1.2), 7.0,
        data_bundle['news_mask'],
        ml_confidence, ml_threshold,
        int(getattr(config, "TIME_EXIT_BARS", 64)),
        # dynamic params from config with safe defaults
        float(getattr(config, "SLIPPAGE_BASE_PIPS", 0.2)),
        float(getattr(config, "SLIPPAGE_ATR_FACTOR", 0.6)),
        float(getattr(config, "SLIPPAGE_VOLUME_FACTOR", 0.5)),
        float(getattr(config, "SLIPPAGE_ORDER_FACTOR", 0.2)),
        float(getattr(config, "NEWS_IMPACT_SLIPPAGE_MULTIPLIER", 1.8)),
        float(getattr(config, "NEWS_IMPACT_SPREAD_MULTIPLIER", 1.5)),
        float(getattr(config, "NEWS_IMPACT_SL_MULTIPLIER", 1.2)),
        float(getattr(config, "NEWS_IMPACT_TP_MULTIPLIER", 0.9)),
        float(getattr(config, "ASIAN_SLIPPAGE_MULT", 1.4)),
        float(getattr(config, "GOLDEN_SLIPPAGE_MULT", 0.9)),
        float(getattr(config, "NORMAL_SLIPPAGE_MULT", 1.0)),
        float(getattr(config, "PROTECTION_SLIPPAGE_MULT", 1.8)),
    )
    
    metrics = calculate_metrics_v7(t_res, bal, durations, rr_vals, total_bars)
    
    if config.ENABLE_MULTI_OBJECTIVE:
        trades_utilization = 0.0
        tt = float(metrics.get('total_trades', 0) or 0.0)
        tb = float(total_bars or 1.0)
        scaled_trades = tt * (20000.0 / max(tb, 1.0))
        if 40.0 <= scaled_trades <= 60.0:
            trades_utilization = 1.0
        elif scaled_trades < 40.0:
            trades_utilization = max(0.0, scaled_trades / 40.0)
        else:
            trades_utilization = max(0.0, 60.0 / scaled_trades)
        if scaled_trades < 30.0:
            trades_utilization *= (0.5 + 0.5 * (scaled_trades / 30.0))
        wr = float(metrics.get('win_rate', 0.0))
        dd = float(metrics.get('drawdown', 0.0))
        pf = float(metrics.get('profit_factor', 0.0))
        ret_vol = float(metrics.get('ret_volatility', 0.0))
        rr_mean = float(metrics.get('rr_mean', 0.0))
        return (
            wr,
            -dd,
            pf,
            -ret_vol,
            rr_mean,
            trades_utilization
        )
        
    return metrics['score_v7']

def optimize_with_optuna(data_bundle, n_trials: int = None, timeout: int = None):
    """
    Executa a otimização com Optuna usando o engine Numba.
    Expected data_bundle keys: df, pip_size, tick_value, spread, news_mask
    """
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    timeout = timeout or config.OPTUNA_TIMEOUT
    
    if config.ENABLE_MULTI_OBJECTIVE:
        # Novo multi-objetivo com métricas institucionais:
        # 0: WinRate (maximize)
        # 1: -Drawdown (maximize)
        # 2: ProfitFactor (maximize)
        # 3: -Return Volatility (maximize)
        # 4: RR Mean (maximize)
        # 5: Trades Utilization Score (maximize)
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize", "maximize", "maximize", "maximize"],
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
    study.optimize(lambda t: objective(t, data_bundle), n_trials=n_trials, timeout=timeout)
    
    # Best Params Extraction
    if config.ENABLE_MULTI_OBJECTIVE:
        # Pega a solução do Pareto Front com maior WinRate que satisfaça DD < 15%
        best_trials = study.best_trials
        # Filtra trials via trade-off logic
        valid_trials = [t for t in best_trials if t.values[1] > -config.MAX_DD_OPT]
        
        if valid_trials:
            # Sort by ProfitFactor
            # Combine PF and RR mean for ranking
            best_trial = sorted(valid_trials, key=lambda t: (t.values[2] * 0.7 + (t.values[4] or 0.0) * 0.3), reverse=True)[0]
            return {"best_params": best_trial.params}
        elif best_trials:
             return {"best_params": best_trials[0].params}
        else:
             return {"best_params": {}}
             
    else:
        return {"best_params": study.best_params}

def run_backtest_with_params(data_bundle, params):
    """
    Executa um backtest único com parâmetros definidos.
    Útil para validação OOS (Out-of-Sample).
    """
    if 'df' not in data_bundle: return None

    df = data_bundle['df']
    open_arr = df['open'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    time_indices = df.index.hour.values
    vol_rel = _compute_volume_ratio(df)
    session_code = _compute_session_code(time_indices)

    # Extract params
    ema_s_period = int(params['ema_short'])
    ema_l_period = int(params['ema_long'])
    rsi_low = int(params['rsi_low'])
    rsi_high = int(params['rsi_high'])
    adx_thresh = int(params['adx_threshold'])
    sl_atr = float(params['sl_atr'])
    tp_atr = float(params['tp_atr'])
    # Default threshold if not optimized (e.g. legacy params)
    ml_threshold = float(params.get('ml_threshold', 0.6)) 

    # ML Confidence Access
    ml_confidence = data_bundle.get('ml_confidence')
    if ml_confidence is None:
        ml_confidence = np.zeros(len(close))
    # Fallback: se o vetor de confiança for constante/zero, desativar gating de ML
    try:
        if (np.std(ml_confidence) < 1e-6) or (np.count_nonzero(ml_confidence > 0.55) < 10):
            ml_threshold = 0.0
    except Exception:
        pass

    # Calc indicators
    ema_s = ema_numba(close, ema_s_period)
    ema_l = ema_numba(close, ema_l_period)
    rsi = calculate_rsi_numba(close, 14)
    adx = calculate_adx_numba(high, low, close, 14)
    atr = calculate_atr_numba(high, low, close, 14)
    _, _, macd_hist = calculate_macd_numba(close)

    bal, t_cnt, win_cnt, t_res, durations, rr_vals, total_bars = fast_backtest_v7(
        time_indices, open_arr, close, high, low,
        ema_s, ema_l, rsi, adx, atr, macd_hist,
        vol_rel, session_code,
        rsi_low, rsi_high, adx_thresh,
        sl_atr, tp_atr,
        data_bundle['pip_size'], data_bundle['tick_value'], 
        data_bundle.get('spread', 1.2), 7.0, 
        data_bundle['news_mask'],
        ml_confidence, ml_threshold,
        int(getattr(config, "TIME_EXIT_BARS", 64)),
        float(getattr(config, "SLIPPAGE_BASE_PIPS", 0.2)),
        float(getattr(config, "SLIPPAGE_ATR_FACTOR", 0.6)),
        float(getattr(config, "SLIPPAGE_VOLUME_FACTOR", 0.5)),
        float(getattr(config, "SLIPPAGE_ORDER_FACTOR", 0.2)),
        float(getattr(config, "NEWS_IMPACT_SLIPPAGE_MULTIPLIER", 1.8)),
        float(getattr(config, "NEWS_IMPACT_SPREAD_MULTIPLIER", 1.5)),
        float(getattr(config, "NEWS_IMPACT_SL_MULTIPLIER", 1.2)),
        float(getattr(config, "NEWS_IMPACT_TP_MULTIPLIER", 0.9)),
        float(getattr(config, "ASIAN_SLIPPAGE_MULT", 1.4)),
        float(getattr(config, "GOLDEN_SLIPPAGE_MULT", 0.9)),
        float(getattr(config, "NORMAL_SLIPPAGE_MULT", 1.0)),
        float(getattr(config, "PROTECTION_SLIPPAGE_MULT", 1.8)),
    )

    metrics = calculate_metrics_v7(t_res, bal, durations, rr_vals, total_bars)
    return metrics, t_res, bal

def run_monte_carlo(trade_results: np.array, n_sims=1000):
    if len(trade_results) < 10: return 1.0 # 100% risk if no data
    
    # Gaussian Noise parameters (2026 volatility simulation)
    std_dev_noise = 50.0 # Ex: $50 variance
    
    sim_dds = []
    for _ in range(n_sims):
        if len(trade_results) == 0:  # <--- Adicionado: Handle empty.
            sim_dds.append(1.0)  # Assume full DD.
            continue
        # Bootstrap with noise
        sim_trades = np.random.choice(trade_results, size=len(trade_results), replace=True)  # <--- Agora seguro.        sim_trades = sim_trades + noise
        
        sim_balance = [100000.0]
        for t in sim_trades:
            sim_balance.append(sim_balance[-1] + t)
        
        # Calculate DD for sim
        peak = sim_balance[0]
        max_dd = 0.0
        for val in sim_balance:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
        sim_dds.append(max_dd)
    
    return np.percentile(sim_dds, 95) # 95th percentile DD

def train_ml_model(df):
    """
    Treina um modelo RandomForest simples para prever direção do mercado.
    Retorna o array de probabilidades para todo o dataset.
    """
    try:
        close = df["close"].values.astype(np.float64)

        features, start_idx = build_ml_features(df)

        min_return = getattr(config, "ML_MIN_RETURN", 0.0005)

        future_close = np.roll(close, -ML_HORIZON)
        future_ret = (future_close - close) / close

        y_full = np.full(len(close), -1, dtype=np.int8)
        y_full[future_ret >= min_return] = 1
        y_full[future_ret <= -min_return] = 0

        valid_idx = np.arange(len(close) - ML_HORIZON)
        mask = (valid_idx >= start_idx) & (y_full[valid_idx] >= 0)

        if not np.any(mask):
            logger.warning("Dataset ML insuficiente após filtro de retorno mínimo.")
            # Fallback: probabilidade neutra para não bloquear entradas
            return None, np.full(len(close), 0.5, dtype=np.float64)

        X = features[valid_idx][mask]
        y = y_full[valid_idx][mask]

        X = np.nan_to_num(X)

        if len(X) < 100:
            logger.warning("Dataset ML com menos de 100 amostras. Abortando treino.")
            # Fallback: probabilidade neutra para não bloquear entradas
            return None, np.full(len(close), 0.5, dtype=np.float64)

        model = RandomForestClassifier(
            n_estimators=getattr(config, "ML_RF_N_ESTIMATORS", 200),
            max_depth=getattr(config, "ML_RF_MAX_DEPTH", 10),
            min_samples_split=getattr(config, "ML_RF_MIN_SAMPLES_SPLIT", 50),
            class_weight=getattr(config, "ML_RF_CLASS_WEIGHT", "balanced"),
            bootstrap=True,
            max_features=getattr(config, "ML_RF_MAX_FEATURES", "sqrt"),
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        train_probs = model.predict_proba(X)[:, 1]
        train_pred = (train_probs >= 0.5).astype(int)
        train_acc = float(np.mean(train_pred == y))
        logger.info(f"ML RF Train Accuracy: {train_acc:.3f}")

        if hasattr(model, "oob_score_"):
            logger.info(f"ML RF OOB Score: {model.oob_score_:.3f}")

        if train_probs.size > 0:
            logger.info(
                "ML RF Train prob dist: mean=%.3f min=%.3f max=%.3f std=%.3f",
                float(np.mean(train_probs)),
                float(np.min(train_probs)),
                float(np.max(train_probs)),
                float(np.std(train_probs)),
            )

        all_probs = np.zeros(len(close))

        valid_features_full = features[start_idx : len(close) - ML_HORIZON]
        valid_features_full = np.nan_to_num(valid_features_full)

        if valid_features_full.shape[0] > 0:
            probs_full = model.predict_proba(valid_features_full)[:, 1]
            all_probs[start_idx : start_idx + len(probs_full)] = probs_full

            logger.info(
                "ML RF Full prob dist: mean=%.3f min=%.3f max=%.3f std=%.3f",
                float(np.mean(probs_full)),
                float(np.min(probs_full)),
                float(np.max(probs_full)),
                float(np.std(probs_full)),
            )

        return model, all_probs

    except Exception as e:
        logger.error(f"Erro no treino ML: {e}")
        # Fallback: probabilidade neutra em caso de erro
        return None, np.full(len(df), 0.5, dtype=np.float64)

def predict_ml_model(model, df):
    """
    Gera previsões para um dataframe novo usando modelo treinado.
    """
    if model is None:
        return np.zeros(len(df))

    try:
        close = df["close"].values.astype(np.float64)

        features, start_idx = build_ml_features(df)

        all_probs = np.zeros(len(close))

        valid_features = features[start_idx : len(close) - ML_HORIZON]
        valid_features = np.nan_to_num(valid_features)

        if valid_features.shape[0] == 0:
            return all_probs

        probs = model.predict_proba(valid_features)[:, 1]
        all_probs[start_idx : start_idx + len(probs)] = probs

        logger.info(
            "ML RF Predict prob dist: mean=%.3f min=%.3f max=%.3f std=%.3f",
            float(np.mean(probs)),
            float(np.min(probs)),
            float(np.max(probs)),
            float(np.std(probs)),
        )

        return all_probs
    except Exception as e:
        logger.error(f"Erro na predicao ML: {e}")
        return np.zeros(len(df))
