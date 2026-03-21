# engine.py - Optimized Engine for XP3 PRO FOREX
"""
🚀 XP3 PRO - MOTOR DE OTIMIZAÇÃO v5.0
Baseado no legacy/optimizer_optuna_forex.py v7.0
"""

import optuna
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from numba import njit
import os

from xp3_forex.core.settings import settings
from xp3_forex import utils

logger = logging.getLogger(__name__)

# Configurações de Compatibilidade (Fallbacks do config_forex legado)
ML_HORIZON = 4
TIME_EXIT_BARS = 64
ENABLE_MULTI_OBJECTIVE = True
MAX_DD_OPT = 0.15
OPTUNA_N_TRIALS = settings.OPTIMIZATION_TRIALS
OPTUNA_TIMEOUT = settings.OPTIMIZATION_TIMEOUT

# ===========================
# INDICADORES (NUMBA - ULTRA FAST)
# ===========================

@njit
def sma_numba(data, period):
    res = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        res[i] = np.mean(data[i - period + 1 : i + 1])
    return res

@njit
def ema_numba(data, period):
    res = np.zeros_like(data)
    if len(data) < period: return res
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
    if len(close) > period:
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(close)):
            atr[i] = (tr[i] - atr[i-1]) * alpha + atr[i-1]
    return atr

@njit
def calculate_rsi_numba(close, period=14):
    rsi = np.zeros_like(close)
    if len(close) <= period: return rsi
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
    
    if len(plus_dm) < period * 2:
        return np.zeros_like(close)
        
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

# ===========================
# MOTOR DE BACKTEST
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
        if code == 1: return golden_slip_mult
        elif code == 2: return asian_slip_mult
        elif code == 3: return protection_slip_mult
        return normal_slip_mult
    
    def _dynamic_slippage_pips(i, lot_hint):
        atr_pips = atr[i] / pip_size
        start = max(0, i-100)
        end = max(1, i)
        base = np.mean(atr[start:end]) / pip_size
        atr_norm = atr_pips / base if base > 0 else 1.0
        vol_ratio = vol_rel[i] if vol_rel[i] > 0 else 1.0
        liq_penalty = (1.0 - vol_ratio) if vol_ratio < 1.0 else 0.0
        lot_component = np.log10(1.0 + lot_hint) if lot_hint > 0 else 0.0
        
        slip = slippage_base_pips
        slip += slippage_atr_factor * atr_norm
        slip += slippage_volume_factor * liq_penalty
        slip += slippage_order_factor * lot_component
        slip *= _session_mult(session_code[i])
        if news_mask[i]: slip *= news_slippage_mult
        return max(0.0, slip)
    
    start_i = max(200, int(len(close) * 0.05))
    for i in range(start_i, len(close)):
        if in_position == 0:
            # Buy Logic
            if (ema_s[i] > ema_l[i]) and (rsi[i] < rsi_l) and (adx[i] > adx_t) and (ml_confidence[i] >= ml_threshold):
                in_position = 1
                eff_sl_mult = sl_atr_mult * (news_sl_mult if news_mask[i] else 1.0)
                tp_mult = (tp_base_mult if adx[i] <= 30 else tp_base_mult * 1.5) * (news_tp_mult if news_mask[i] else 1.0)
                spread_pips_eff = spread_pips_base * (news_spread_mult if news_mask[i] else 1.0)
                
                risk_pips_hint = (atr[i] * eff_sl_mult) / pip_size
                lot_hint = (equity * 0.01) / (max(0.1, risk_pips_hint) * tick_value)
                slip_pips = _dynamic_slippage_pips(i, lot_hint)
                
                entry_price = close[i] + (spread_pips_eff + slip_pips) * pip_size
                entry_atr = atr[i]
                sl_price = entry_price - (entry_atr * eff_sl_mult)
                tp_price = entry_price + (entry_atr * tp_mult)
                entry_lot = lot_hint
                entry_index = i
                entry_risk_pips = risk_pips_hint

            # Sell Logic
            elif (ema_s[i] < ema_l[i]) and (rsi[i] > rsi_h) and (adx[i] > adx_t) and (ml_confidence[i] <= (1.0 - ml_threshold)):
                in_position = -1
                eff_sl_mult = sl_atr_mult * (news_sl_mult if news_mask[i] else 1.0)
                tp_mult = (tp_base_mult if adx[i] <= 30 else tp_base_mult * 1.5) * (news_tp_mult if news_mask[i] else 1.0)
                spread_pips_eff = spread_pips_base * (news_spread_mult if news_mask[i] else 1.0)
                
                risk_pips_hint = (atr[i] * eff_sl_mult) / pip_size
                lot_hint = (equity * 0.01) / (max(0.1, risk_pips_hint) * tick_value)
                slip_pips = _dynamic_slippage_pips(i, lot_hint)
                
                entry_price = close[i] - (spread_pips_eff + slip_pips) * pip_size
                entry_atr = atr[i]
                sl_price = entry_price + (entry_atr * eff_sl_mult)
                tp_price = entry_price - (entry_atr * tp_mult)
                entry_lot = lot_hint
                entry_index = i
                entry_risk_pips = risk_pips_hint

        else:
            close_signal = False
            pnl_pips = 0.0
            
            if in_position == 1:
                if high[i] >= tp_price:
                    pnl_pips = (tp_price - entry_price) / pip_size
                    close_signal = True
                elif low[i] <= sl_price:
                    pnl_pips = (sl_price - _dynamic_slippage_pips(i, entry_lot)*pip_size - entry_price) / pip_size
                    close_signal = True
                elif (i - entry_index) >= time_exit_bars:
                    pnl_pips = (open_arr[i] - entry_price) / pip_size
                    close_signal = True
            
            elif in_position == -1:
                if low[i] <= tp_price:
                    pnl_pips = (entry_price - tp_price) / pip_size
                    close_signal = True
                elif high[i] >= sl_price:
                    pnl_pips = (entry_price - (sl_price + _dynamic_slippage_pips(i, entry_lot)*pip_size)) / pip_size
                    close_signal = True
                elif (i - entry_index) >= time_exit_bars:
                    pnl_pips = (entry_price - open_arr[i]) / pip_size
                    close_signal = True

            if close_signal:
                net_pnl = (pnl_pips * tick_value * entry_lot) - (commission * entry_lot)
                equity += net_pnl
                balance_history.append(equity)
                trade_results.append(net_pnl)
                trades_count += 1
                if net_pnl > 0: win_trades += 1
                durations.append(i - entry_index)
                if entry_risk_pips > 0: rr_list.append(abs(pnl_pips) / entry_risk_pips)
                in_position = 0

    return (
        np.array(balance_history),
        trades_count,
        win_trades,
        np.array(trade_results),
        np.array(durations),
        np.array(rr_list),
        len(close) - start_i
    )

def calculate_metrics_v7(trade_results, balance_history, total_bars):
    if len(balance_history) < 2:
        return {"score_v7": -100.0, "total_trades": 0}

    ret_total = (balance_history[-1] / balance_history[0]) - 1
    peak = np.maximum.accumulate(balance_history)
    drawdowns = (peak - balance_history) / peak
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    trades_np = np.array(trade_results, dtype=np.float64)
    if len(trades_np) > 1:
        mean_trade = np.mean(trades_np)
        std_trade = np.std(trades_np)
        sharpe = float(mean_trade / std_trade) if std_trade > 1e-9 else 0.0
    else:
        sharpe = 0.0

    wins = trades_np[trades_np > 0]
    loss_sum = np.sum(np.abs(trades_np[trades_np < 0]))
    win_rate = (len(wins) / len(trades_np)) if len(trades_np) > 0 else 0.0
    p_factor = min(np.sum(wins) / loss_sum if loss_sum > 0 else 5.0, 10.0)

    score = (p_factor * 10) + (sharpe * 5)
    if max_dd > 0.15: score *= 0.5
    if win_rate < 0.35: score *= 0.7
    if len(trades_np) < 10: score *= 0.4

    return {
        "retorno_total": float(ret_total),
        "drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "profit_factor": float(p_factor),
        "score_v7": float(score),
        "total_trades": int(len(trades_np))
    }

def run_backtest_with_params(data_bundle, params):
    df = data_bundle['df']
    open_arr = df['open'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    time_indices = df.index.hour.values
    
    # Compute basic volume ratio
    vol = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.ones(len(df))
    vol_mean = pd.Series(vol).rolling(50, min_periods=1).mean().values
    vol_rel = np.where(vol_mean > 0, vol / vol_mean, 1.0)
    
    # Session code
    session_code = np.zeros(len(time_indices), dtype=np.int32)
    for i, h in enumerate(time_indices):
        if 10 <= h < 14: session_code[i] = 1 # Golden
        elif h >= 22 or h < 5: session_code[i] = 2 # Asian
        elif 18 <= h < 22: session_code[i] = 3 # Protection

    ml_confidence = data_bundle.get('ml_confidence', np.zeros(len(close)))
    
    ema_s = ema_numba(close, int(params.get('ema_short', 20)))
    ema_l = ema_numba(close, int(params.get('ema_long', 50)))
    rsi = calculate_rsi_numba(close, 14)
    adx = calculate_adx_numba(high, low, close, 14)
    atr = calculate_atr_numba(high, low, close, 14)
    _, _, macd_hist = calculate_macd_numba(close)

    bal, t_cnt, win_cnt, t_res, durations, rr_vals, total_bars = fast_backtest_v7(
        time_indices, open_arr, close, high, low,
        ema_s, ema_l, rsi, adx, atr, macd_hist,
        vol_rel, session_code,
        int(params.get('rsi_low', 30)), int(params.get('rsi_high', 70)), int(params.get('adx_threshold', 20)),
        float(params.get('sl_atr', 2.0)), float(params.get('tp_atr', 3.0)),
        data_bundle.get('pip_size', 0.0001), data_bundle.get('tick_value', 10.0), 
        data_bundle.get('spread', 1.2), 7.0, 
        data_bundle.get('news_mask', np.zeros(len(close), dtype=bool)),
        ml_confidence, float(params.get('ml_threshold', 0.6)),
        TIME_EXIT_BARS,
        0.2, 0.6, 0.5, 0.2, 1.8, 1.5, 1.2, 0.9, 1.4, 0.9, 1.0, 1.8
    )

    metrics = calculate_metrics_v7(t_res, bal, total_bars)
    return metrics, t_res, bal

def optimize_with_optuna(data_bundle, n_trials=50, timeout=360):
    def objective(trial):
        ema_short = trial.suggest_int("ema_short", 5, 50)
        params = {
            "ema_short": ema_short,
            "ema_long": trial.suggest_int("ema_long", ema_short + 5, 200),
            "rsi_low": trial.suggest_int("rsi_low", 25, 45),
            "rsi_high": trial.suggest_int("rsi_high", 55, 75),
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 35),
            "sl_atr": trial.suggest_float("sl_atr", 1.0, 3.5),
            "tp_atr": trial.suggest_float("tp_atr", 1.5, 5.0),
            "ml_threshold": trial.suggest_float("ml_threshold", 0.50, 0.85)
        }
        metrics, _, _ = run_backtest_with_params(data_bundle, params)
        return metrics.get("score_v7", -100.0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return {"best_params": study.best_params}

def train_ml_model(df):
    # Simplified ML trainer for engine integration
    return None, np.full(len(df), 0.5)

def predict_ml_model(model, df):
    return np.full(len(df), 0.5)

def run_monte_carlo(trade_results, n_sims=1000):
    if len(trade_results) < 5: return 1.0
    sim_dds = []
    for _ in range(n_sims):
        sim_trades = np.random.choice(trade_results, size=len(trade_results), replace=True)
        bal = [100000.0]
        for t in sim_trades: bal.append(bal[-1] + t)
        peak = np.maximum.accumulate(bal)
        dd = (peak - bal) / peak
        sim_dds.append(np.max(dd))
    return np.percentile(sim_dds, 95)
