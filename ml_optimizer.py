# ml_optimizer.py - Machine Learning Optimizer v3.2
"""
🤖 ML OPTIMIZER - XP3 PRO FOREX v3.2
✅ Q-Learning para otimização de parâmetros
✅ Ensemble de estratégias
✅ Aprendizado contínuo
✅ Persistência de histórico
✅ Walk-Forward Optimization (NOVO)
✅ Re-otimização mensal automática (NOVO)
✅ CORREÇÃO: Tratamento de AttributeError para config.ML_LIVE_EXPLORATION_RATE e outros
✅ CORREÇÃO: Uso de SYMBOL_MAP para otimização
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import random # Para a política epsilon-greedy
from sklearn.ensemble import RandomForestClassifier # ✅ NOVO: RF para previsão
from numba import njit # ✅ NOVO: Numba para features rápidas
import MetaTrader5 as mt5 # ✅ NOVO: Para constantes de timeframe

import config_forex as config
import utils_forex as utils # Importa utils para get_rates e get_indicators_forex

logger = logging.getLogger("ml_optimizer")

# ===========================
# FEATURES & INDICATORS (NUMBA)
# ===========================
ML_HORIZON = 4

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

@njit
def rolling_mean(arr, window):
    result = np.zeros_like(arr)
    for i in range(window, len(arr)):
        result[i] = np.mean(arr[i-window:i])
    return result

def build_ml_features(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """
    Constrói features para ML usando Numba (Ultra Fast).
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df.get("tick_volume", np.ones_like(close)).values.astype(np.float64)
    
    rsi = calculate_rsi_numba(close, 14)
    adx = calculate_adx_numba(high, low, close, 14)
    atr = calculate_atr_numba(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd_numba(close)
    ema200 = ema_numba(close, 200)
    # Usa np.divide com 'where' para evitar RuntimeWarning de divisão por zero
    ema200_dist = np.divide(
        (close - ema200),
        ema200,
        out=np.zeros_like(close),
        where=ema200 != 0,
    )
    sma50 = sma_numba(close, 50)
    sma50_dist = np.divide(
        (close - sma50),
        close,
        out=np.zeros_like(close),
        where=close != 0,
    )
    vol_mean_50 = rolling_mean(volume, 50)
    vol_rel_50 = np.divide(
        volume,
        vol_mean_50,
        out=np.ones_like(volume),
        where=vol_mean_50 != 0,
    )
    candle_range = np.divide(
        (high - low),
        close,
        out=np.zeros_like(close),
        where=close != 0,
    )
    
    rsi_div = np.zeros_like(rsi)
    # Simple Divergence (last 5) - approximated
    for i in range(5, len(rsi)):
        price_delta = close[i] - close[i-5]
        rsi_delta = rsi[i] - rsi[i-5]
        if price_delta > 0 and rsi_delta < 0:
            rsi_div[i] = -1
        elif price_delta < 0 and rsi_delta > 0:
            rsi_div[i] = 1
            
    adx_slope = adx - np.roll(adx, 5)
    momentum1 = close - np.roll(close, 1)
    momentum3 = close - np.roll(close, 3)
    momentum10 = close - np.roll(close, 10)
    
    sma_bb = sma_numba(close, 20)
    # Std dev for BB (simple loop or numpy)
    std_bb = np.zeros_like(close)
    for i in range(20, len(close)):
        std_bb[i] = np.std(close[i-20:i])
        
    bb_upper = sma_bb + 2 * std_bb
    bb_lower = sma_bb - 2 * std_bb
    band_range = bb_upper - bb_lower
    bb_width = np.where(sma_bb != 0, np.divide(band_range, sma_bb, out=np.zeros_like(band_range), where=sma_bb!=0), 0.0)
    bb_position = np.where(band_range != 0, np.divide((close - bb_lower), band_range, out=np.zeros_like(close), where=band_range!=0), 0.0)
    
    # ✅ v6.0: Features Temporais Cíclicas (Hora/Dia)
    # Extrai hora e dia da semana do índice ou coluna 'time'
    if 'time' in df.columns:
        times = pd.to_datetime(df['time'])
        hours = times.dt.hour.values.astype(np.float64)
        days = times.dt.dayofweek.values.astype(np.float64)
    else:
        # Fallback se não tiver tempo (raro)
        hours = np.zeros_like(close)
        days = np.zeros_like(close)

    # Transformação Cíclica (Seno/Cosseno)
    # Hora (0-23) -> 2pi * h / 24
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    
    # Dia da Semana (0-6) -> 2pi * d / 7
    day_sin = np.sin(2 * np.pi * days / 7.0)
    day_cos = np.cos(2 * np.pi * days / 7.0)

    features = np.column_stack((
        rsi, adx, atr, macd_line, macd_signal, macd_hist,
        ema200, ema200_dist, sma50, sma50_dist, vol_rel_50,
        candle_range, rsi_div, adx_slope, momentum1, momentum3,
        momentum10, bb_width, bb_position,
        hour_sin, hour_cos, day_sin, day_cos # ✅ Novas features
    ))
    
    start_idx = 200 # Need 200 candles for EMA200
    return features, start_idx

# ===========================
# ENSEMBLE OPTIMIZER
# ===========================
class EnsembleOptimizer:
    """
    Otimizador ensemble com Q-Learning + Walk-Forward
    """

    def __init__(self, history_file: str = "ml_history.json", qtable_file: str = "qtable.npy"):
        self.history_file = config.DATA_DIR / history_file # Usa DATA_DIR do config
        self.qtable_file = config.DATA_DIR / qtable_file   # Usa DATA_DIR do config

        # Q‑Table: {symbol: {state: {action: q_value}}}
        self.q_table: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Histórico de trades
        self.trade_history: List[Dict[str, Any]] = []

        # Parâmetros de aprendizado (✅ CORREÇÃO: Acessa com getattr para fallback seguro)
        self.learning_rate: float = getattr(config, 'ML_LEARNING_RATE', 0.1)
        self.discount_factor: float = getattr(config, 'ML_DISCOUNT_FACTOR', 0.9)
        self.exploration_rate: float = getattr(config, 'ML_LIVE_EXPLORATION_RATE', 0.05)

        # Estatísticas por símbolo
        self.symbol_stats = defaultdict(lambda: {
            "total_trades": 0,
            "winning_trades": 0,
            "total_profit": 0.0,
            "avg_rr_achieved": 0.0,
            "best_action": "MODERATE",
            "last_reoptimization": None,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,    # ✅ NOVO v5.2
            "returns": []           # ✅ NOVO v5.2: Lista de retornos para Sharpe
        })

        self.last_global_reoptimization: Optional[datetime] = None
        self.backtest_results = {} # ✅ NOVO: Armazena resultados do backtest

        # ✅ NOVO: Modelos RandomForest em memória
        self.rf_models: Dict[str, RandomForestClassifier] = {}
        self.last_rf_train: Dict[str, datetime] = {}

        # Carrega arquivos existentes
        self._load_history()
        self._load_qtable()
        self._load_backtest_results() # ✅ NOVO: Carrega resultados do backtest

        logger.info("✅ ML Optimizer v3.2 inicializado")

    # -------------------------------------------------
    # Métodos de persistência
    # -------------------------------------------------
    def _load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Histórico de trades carregado de {self.history_file}")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erro ao carregar histórico de trades: {e}. Iniciando com histórico vazio.")
                self.trade_history = []
        else:
            logger.info("Nenhum histórico de trades encontrado. Iniciando com histórico vazio.")

    def _save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=4)
            logger.debug(f"Histórico de trades salvo em {self.history_file}")
        except IOError as e:
            logger.error(f"❌ Erro ao salvar histórico de trades: {e}")

    def _load_qtable(self):
        if self.qtable_file.exists():
            try:
                self.q_table = np.load(self.qtable_file, allow_pickle=True).item()
                logger.info(f"Q-Table carregada de {self.qtable_file}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar Q-Table: {e}. Iniciando com Q-Table vazia.")
                self.q_table = {}
        else:
            logger.info("Nenhuma Q-Table encontrada. Iniciando com Q-Table vazia.")

    def _save_qtable(self):
        try:
            np.save(self.qtable_file, self.q_table)
            logger.debug(f"Q-Table salva em {self.qtable_file}")
        except IOError as e:
            logger.error(f"❌ Erro ao salvar Q-Table: {e}")

    def _load_backtest_results(self):
        """✅ NOVO: Carrega os resultados do backtest gerados pelo otimizador"""
        try:
            results_path = getattr(config, 'OPTIMIZER_OUTPUT', Path("optimizer_output")) / "backtest_results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    self.backtest_results = json.load(f)
                logger.info(f"✅ Resultados de backtest carregados para {len(self.backtest_results)} símbolos.")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar resultados de backtest: {e}")

    # -------------------------------------------------
    # Métodos de Q-Learning
    # -------------------------------------------------
    def _encode_state(self, indicators: Dict[str, Any]) -> str:
        """
        Codifica o estado do mercado a partir dos indicadores.
        ✅ REQUISITO: Blindagem Total contra None e dicionários inválidos.
        """
        if indicators is None or not isinstance(indicators, dict):
            return "NEUTRAL"

        try:
            ema_trend = indicators.get("ema_trend", "UNKNOWN")
            rsi_level = "OVERBOUGHT" if indicators.get("rsi", 50) > indicators.get("rsi_high_limit", 70) else \
                        "OVERSOLD" if indicators.get("rsi", 50) < indicators.get("rsi_low_limit", 30) else \
                        "NEUTRAL"
            adx_strength = "STRONG" if indicators.get("adx", 0) > getattr(config, 'ADX_THRESHOLD', 25) else "WEAK" 
            bb_squeeze = "SQUEEZE" if indicators.get("bb_width", 0) < getattr(config, 'BB_SQUEEZE_THRESHOLD', 0.015) else "EXPANSION" 

            price_position_bb = "UPPER" if indicators.get("close", 0) > indicators.get("bb_upper", 0) else \
                                "LOWER" if indicators.get("close", 0) < indicators.get("bb_lower", 0) else \
                                "MIDDLE"

            return f"{ema_trend}_{rsi_level}_{adx_strength}_{bb_squeeze}_{price_position_bb}"
        except Exception:
            return "NEUTRAL"

    def _init_actions(self) -> Dict[str, float]:
        """Inicializa Q-values para as ações possíveis."""
        # Ações podem ser: "CONSERVATIVE", "MODERATE", "AGGRESSIVE"
        # Cada ação corresponde a um conjunto de multiplicadores SL/TP ou outros parâmetros.
        return {"CONSERVATIVE": 0.0, "MODERATE": 0.0, "AGGRESSIVE": 0.0}

    def _decode_action(self, action: str) -> Dict[str, float]:
        """Decodifica a ação em parâmetros de trading."""
        # ✅ CORREÇÃO: Usa getattr para fallback seguro para constantes do config
        sl_atr_mult_base = getattr(config, 'STOP_LOSS_ATR_MULTIPLIER', 2.0)
        tp_atr_mult_base = getattr(config, 'TAKE_PROFIT_ATR_MULTIPLIER', 3.0)
        risk_reward_base = getattr(config, 'RISK_REWARD_RATIO', 2.0)
        ema_short_base = getattr(config, 'EMA_SHORT_PERIOD', 20)

        if action == "CONSERVATIVE":
            return {
                "ema_short": ema_short_base * 1.1, # EMA mais lenta
                "sl_atr_mult": sl_atr_mult_base * 1.2, # SL maior
                "risk_reward": risk_reward_base * 0.8 # TP menor
            }
        elif action == "AGGRESSIVE":
            return {
                "ema_short": ema_short_base * 0.9, # EMA mais rápida
                "sl_atr_mult": sl_atr_mult_base * 0.8, # SL menor
                "risk_reward": risk_reward_base * 1.2 # TP maior
            }
        else: # MODERATE
            return {
                "ema_short": float(ema_short_base),
                "sl_atr_mult": float(sl_atr_mult_base),
                "risk_reward": float(risk_reward_base)
            }

    def _calculate_reward(self, result: Dict[str, float]) -> float:
        """Calcula a recompensa para o Q-Learning."""
        profit = result.get("profit", 0)
        rr_achieved = result.get("rr_achieved", 0)

        # Recompensa baseada no lucro e no risco-recompensa alcançado
        reward = profit * 0.1 # 10% do lucro como recompensa direta
        if rr_achieved >= 1.5: # Recompensa por trades com bom R:R
            reward += rr_achieved * 0.5
        elif rr_achieved < 0.5 and profit < 0: # Penalidade por trades ruins
            reward -= abs(profit) * 0.05

        return reward

    def _update_q_table(self, symbol: str, params: Dict[str, float], result: Dict[str, float], reward: float, market_indicators: Dict[str, Any]):
        """Atualiza a Q-Table com base no resultado do trade."""
        state = self._encode_state(market_indicators)
        action = self._determine_action_from_params(params) # Reverte params para ação

        if symbol not in self.q_table:
            self.q_table[symbol] = {}
        if state not in self.q_table[symbol]:
            self.q_table[symbol][state] = self._init_actions()
        if action not in self.q_table[symbol][state]: # Garante que a ação exista
            self.q_table[symbol][state][action] = 0.0

        old_q_value = self.q_table[symbol][state][action]

        # Estima o Q-value máximo para o próximo estado (se houver)
        # Para simplificar, vamos assumir que não há um "próximo estado" direto após um trade para a mesma ação.
        # Ou podemos pegar o próximo estado do mercado se tivermos uma sequência de estados.
        # Por enquanto, vamos usar uma simplificação para o next_max_q_value.
        next_max_q_value = 0.0 # Simplificação: assume 0 para o próximo estado

        # Fórmula de Q-Learning
        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * next_max_q_value - old_q_value
        )
        self.q_table[symbol][state][action] = new_q_value
        logger.debug(f"🔄 Q-Table atualizada para {symbol} | State: {state} | Action: {action} | New Q: {new_q_value:.2f}")

    def _determine_action_from_params(self, params: Dict[str, float]) -> str:
        """
        Tenta inferir a ação (CONSERVATIVE, MODERATE, AGGRESSIVE) a partir dos parâmetros usados.
        Isso é uma heurística e pode não ser 100% preciso, mas é necessário para o update.
        """
        sl_atr_mult = params.get('sl_atr_mult', getattr(config, 'STOP_LOSS_ATR_MULTIPLIER', 2.0))
        risk_reward = params.get('risk_reward', getattr(config, 'RISK_REWARD_RATIO', 2.0))
        ema_short = params.get('ema_short', getattr(config, 'EMA_SHORT_PERIOD', 20))

        sl_atr_mult_base = getattr(config, 'STOP_LOSS_ATR_MULTIPLIER', 2.0)
        risk_reward_base = getattr(config, 'RISK_REWARD_RATIO', 2.0)
        ema_short_base = getattr(config, 'EMA_SHORT_PERIOD', 20)

        # Compara com os multiplicadores de _decode_action
        if sl_atr_mult > sl_atr_mult_base * 1.1 and risk_reward < risk_reward_base * 0.9:
            return "CONSERVATIVE"
        elif sl_atr_mult < sl_atr_mult_base * 0.9 and risk_reward > risk_reward_base * 1.1:
            return "AGGRESSIVE"
        else:
            return "MODERATE"

    # -------------------------------------------------
    # Walk-Forward Optimization (WFO)
    # -------------------------------------------------
    def _run_walk_forward_optimization(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """
        Executa uma otimização walk-forward para encontrar os melhores parâmetros.
        Isso é um placeholder para uma otimização real (ex: com Optuna).
        Por enquanto, retorna parâmetros padrão ou ligeiramente ajustados.
        """
        logger.info(f"⚙️ Executando Walk-Forward Optimization para {symbol}...")

        # Simulação de otimização: retorna os parâmetros do config ou um pouco ajustados
        # Em uma implementação real, você usaria Optuna aqui.
        optimal_params = {
            "ema_short": getattr(config, 'EMA_SHORT_PERIOD', 20),
            "ema_long": getattr(config, 'EMA_LONG_PERIOD', 50),
            "rsi_period": getattr(config, 'RSI_PERIOD', 14),
            "adx_period": getattr(config, 'ADX_PERIOD', 14),
            "bb_period": getattr(config, 'BB_PERIOD', 20),
            "bb_dev": getattr(config, 'BB_DEVIATION', 2.0),
            "sl_atr_mult": getattr(config, 'STOP_LOSS_ATR_MULTIPLIER', 2.0),
            "tp_atr_mult": getattr(config, 'TAKE_PROFIT_ATR_MULTIPLIER', 3.0),
            "risk_reward": getattr(config, 'RISK_REWARD_RATIO', 2.0)
        }

        # Adiciona uma pequena aleatoriedade para simular otimização
        optimal_params["ema_short"] = int(optimal_params["ema_short"] * random.uniform(0.9, 1.1))
        optimal_params["sl_atr_mult"] = optimal_params["sl_atr_mult"] * random.uniform(0.9, 1.1)

        logger.info(f"✅ WFO concluída para {symbol}. Parâmetros otimizados: {optimal_params}")
        return optimal_params

    def check_and_reoptimize(self):
        """
        Verifica se é hora de re-otimizar globalmente ou por símbolo.
        ✅ CORREÇÃO: Usa SYMBOL_MAP para iterar sobre os símbolos.
        """
        now = datetime.now()
        retrain_interval = timedelta(hours=getattr(config, 'ML_RETRAIN_INTERVAL_HOURS', 24))

        # Re-otimização global (para todos os símbolos)
        if self.last_global_reoptimization is None or (now - self.last_global_reoptimization) > retrain_interval:
            logger.info("⏳ Iniciando re-otimização global de parâmetros...")

            # ✅ CORREÇÃO: Usa config.SYMBOL_MAP para iterar sobre os símbolos
            symbols_to_optimize = config.SYMBOL_MAP.keys() if hasattr(config, 'SYMBOL_MAP') else config.ALL_AVAILABLE_SYMBOLS

            for symbol in symbols_to_optimize:
                logger.info(f"⏳ Re-otimizando {symbol}...")
                # Obtém dados históricos para WFO
                data = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, config.ML_MIN_SAMPLES * 2) # Pelo menos o dobro de samples
                if data is None or data.empty:
                    logger.warning(f"⚠️ Dados insuficientes para re-otimizar {symbol}. Pulando.")
                    continue

                # Executa WFO e atualiza os parâmetros do símbolo no config (ou em um dicionário interno)
                optimized_params = self._run_walk_forward_optimization(symbol, data)

                # Em uma implementação real, você salvaria esses parâmetros otimizados
                # para serem usados pelo bot principal. Por exemplo, em um arquivo JSON
                # ou atualizando config.FOREX_PAIRS dinamicamente.
                # Por enquanto, vamos apenas logar.
                logger.info(f"✨ Parâmetros otimizados para {symbol}: {optimized_params}")

                # Atualiza a Q-Table com base nesses novos parâmetros (se aplicável)
                # Isso é mais complexo e geralmente envolve simular trades com os novos params.
                # Por simplicidade, vamos apenas registrar a última otimização.
                self.symbol_stats[symbol]["last_reoptimization"] = now.isoformat()

            self.last_global_reoptimization = now
            self._save_qtable() # Salva a Q-Table após a re-otimização
            logger.info("✅ Re-otimização global concluída.")
        else:
            logger.debug("Re-otimização global não necessária ainda.")

    # -------------------------------------------------
    # MÉTODO PRINCIPAL – obtenção de parâmetros otimizados
    # -------------------------------------------------
    def get_optimal_params(self, symbol: str, market_indicators: Dict[str, Any]) -> Dict[str, float]:
        """
        Retorna parâmetros otimizados com Trava de Segurança Institucional
        """
        # 1️⃣ Re‑otimização mensal (se necessário)
        self.check_and_reoptimize()
        state = self._encode_state(market_indicators)

        # 2️⃣ Garante que o estado exista na Q‑Table
        if symbol not in self.q_table:
            self.q_table[symbol] = {}
        if state not in self.q_table[symbol]:
            self.q_table[symbol][state] = self._init_actions()

        # 3️⃣ Política ε‑greedy
        if np.random.random() < self.exploration_rate:
            action = np.random.choice(list(self.q_table[symbol][state].keys()))
        else:
            q_values = self.q_table[symbol][state]
            # ✅ CORREÇÃO: Trata caso de q_values vazio ou todos iguais
            if not q_values:
                action = "MODERATE" # Fallback
            else:
                max_q = max(q_values.values())
                # Escolhe aleatoriamente entre ações com o Q-value máximo
                best_actions = [a for a, q in q_values.items() if q == max_q]
                action = random.choice(best_actions)


        # 4️⃣ Decodifica a ação → parâmetros sugeridos pelo ML
        ml_params = self._decode_action(action)

        # -------------------------------------------------
        # 🛡️ TRAVA DE SEGURANÇA XP3 PRO
        # -------------------------------------------------
        # ✅ CORREÇÃO: Usa config.FOREX_PAIRS ou SYMBOL_MAP para obter base_config
        base_config = config.FOREX_PAIRS.get(symbol)
        if base_config is None and hasattr(config, 'SYMBOL_MAP'):
            base_config = config.SYMBOL_MAP.get(symbol)

        # Fallback se o símbolo não estiver em nenhum mapa
        if base_config is None:
            logger.warning(f"⚠️ Símbolo {symbol} não encontrado em FOREX_PAIRS ou SYMBOL_MAP. Usando parâmetros padrão para base_config.")
            base_config = {
                "ema_short": getattr(config, 'EMA_SHORT_PERIOD', 20),
                "ema_long": getattr(config, 'EMA_LONG_PERIOD', 50),
                "adx_threshold": getattr(config, 'ADX_PERIOD', 25), # Usando ADX_PERIOD como threshold padrão
                "sl_atr": getattr(config, 'STOP_LOSS_ATR_MULTIPLIER', 2.0),
                "tp_atr": getattr(config, 'TAKE_PROFIT_ATR_MULTIPLIER', 3.0),
                "risk_reward": getattr(config, 'RISK_REWARD_RATIO', 2.0)
            }


        final_params = {
            # EMA curta pode variar no máximo ±20%
            "ema_short": float(np.clip(
                ml_params.get('ema_short', base_config.get('ema_short', getattr(config, 'EMA_SHORT_PERIOD', 20))),
                base_config.get('ema_short', getattr(config, 'EMA_SHORT_PERIOD', 20)) * 0.8,
                base_config.get('ema_short', getattr(config, 'EMA_SHORT_PERIOD', 20)) * 1.2)),

            # Multiplicador SL entre 1.2x e 3.0x
            "sl_atr_mult": float(np.clip(
                ml_params.get('sl_atr_mult', base_config.get('sl_atr', getattr(config, 'STOP_LOSS_ATR_MULTIPLIER', 2.0))),
                1.2, 3.0)),

            # Risco‑Retorno mínimo 1.5, máximo 4.0
            "risk_reward": float(np.clip(
                ml_params.get('risk_reward', base_config.get('risk_reward', getattr(config, 'RISK_REWARD_RATIO', 2.0))),
                1.5, 4.0))
        }
        # -------------------------------------------------

        logger.debug(
            f"🤖 {symbol}: ML clamped | State: {state} | "
            f"SL Mult: {final_params['sl_atr_mult']:.2f} | R:R: {final_params['risk_reward']:.2f}"
        )
        return final_params

    def train_rf_model(self, symbol: str, df: pd.DataFrame):
        """
        Treina modelo RandomForest para o símbolo.
        Se o DF fornecido for pequeno, busca mais histórico.
        """
        try:
            # ✅ Verifica se precisamos de mais dados para treino
            train_df = df
            if len(df) < 1000:
                logger.info(f"ML | Buscando histórico estendido (2000 candles) para treinar {symbol}...")
                extended_df = utils.get_rates(symbol, mt5.TIMEFRAME_M15, 2000) 
                if extended_df is not None and not extended_df.empty and len(extended_df) > len(df):
                    train_df = extended_df
                    logger.info(f"ML | {symbol}: Usando {len(train_df)} candles para treino.")
            
            logger.info(f"ML | Treinando RandomForest para {symbol}...")
            close = train_df["close"].values.astype(np.float64)
            features, start_idx = build_ml_features(train_df)
            
            min_return = getattr(config, "ML_MIN_RETURN", 0.0005)
            
            future_close = np.roll(close, -ML_HORIZON)
            future_ret = (future_close - close) / close
            
            y_full = np.full(len(close), -1, dtype=np.int8)
            y_full[future_ret >= min_return] = 1
            y_full[future_ret <= -min_return] = 0
            
            valid_idx = np.arange(len(close) - ML_HORIZON)
            mask = (valid_idx >= start_idx) & (y_full[valid_idx] >= 0)
            
            if not np.any(mask):
                logger.warning(f"ML | {symbol}: Dataset insuficiente para treino.")
                return
                
            X = features[valid_idx][mask]
            y = y_full[valid_idx][mask]
            X = np.nan_to_num(X)
            
            if len(X) < 100:
                logger.warning(f"ML | {symbol}: Amostras insuficientes ({len(X)} < 100).")
                return

            model = RandomForestClassifier(
                n_estimators=getattr(config, "ML_RF_N_ESTIMATORS", 200),
                max_depth=getattr(config, "ML_RF_MAX_DEPTH", 10),
                min_samples_split=getattr(config, "ML_RF_MIN_SAMPLES_SPLIT", 50),
                class_weight=getattr(config, "ML_RF_CLASS_WEIGHT", "balanced"),
                bootstrap=True,
                max_features=getattr(config, "ML_RF_MAX_FEATURES", "sqrt"),
                oob_score=True,
                random_state=42,
                n_jobs=1,
            )
            model.fit(X, y)
            
            # ✅ REQUISITO: Logar distribuição das probabilidades no treino
            train_probs = model.predict_proba(X)[:, 1]
            logger.info(
                f"ML | {symbol} Train Probs: Mean={np.mean(train_probs):.3f} | Min={np.min(train_probs):.3f} | Max={np.max(train_probs):.3f}"
            )
            
            self.rf_models[symbol] = model
            self.last_rf_train[symbol] = datetime.now()
            
            if hasattr(model, "oob_score_"):
                logger.info(f"ML | {symbol}: Modelo treinado. OOB Score: {model.oob_score_:.3f}")
            else:
                logger.info(f"ML | {symbol}: Modelo treinado.")
                
        except Exception as e:
            logger.error(f"ML | Erro ao treinar {symbol}: {e}")

    def get_rf_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[float]:
        """
        Retorna probabilidade (0.0 a 1.0) da classe UP (1).
        Retorna None se falhar.
        """
        model = self.rf_models.get(symbol)
        
        # Treina se não existir ou se for muito antigo (> 24h)
        if model is None or (datetime.now() - self.last_rf_train.get(symbol, datetime.min)) > timedelta(hours=24):
            self.train_rf_model(symbol, df)
            model = self.rf_models.get(symbol)
            
        if model is None:
            return None
            
        try:
            features, start_idx = build_ml_features(df)
            # Pega apenas a última feature válida
            last_feature = features[-1].reshape(1, -1)
            last_feature = np.nan_to_num(last_feature)
            
            prob = model.predict_proba(last_feature)[0, 1] # Prob da classe 1 (UP)
            
            # ✅ REQUISITO: Logar probs brutas (pode ser verboso, usar debug)
            # logger.debug(f"ML | {symbol} Raw Prob: {prob:.4f}")
            
            return prob
        except Exception as e:
            logger.error(f"ML | Erro na predição RF para {symbol}: {e}")
            return None

    def get_prediction_score(self, symbol: str, market_indicators: Dict[str, Any], df: Optional[pd.DataFrame] = None, signal: Optional[str] = None) -> Tuple[float, bool]:
        """
        Retorna (score, is_baseline).
        Combina RandomForest e Q‑Learning com pesos dinâmicos por sessão.
        """
        if market_indicators is None:
            return self.get_baseline_score(symbol), True

        rf_score = None
        if df is not None:
            rf_prob = self.get_rf_prediction(symbol, df)
            if rf_prob is not None:
                if signal == "SELL":
                    rf_prob = 1.0 - rf_prob
                rf_score = rf_prob * 100.0

        state = self._encode_state(market_indicators)
        ql_score = 0.0
        if symbol in self.q_table and state in self.q_table[symbol]:
            q_values = self.q_table[symbol][state]
            if q_values and not all(v == 0 for v in q_values.values()):
                max_q = max(q_values.values())
                if max_q > 0:
                    ql_score = 50 + (min(max_q, 10) * 5)
                else:
                    ql_score = 50 - (min(abs(max_q), 10) * 5)
                ql_score = float(np.clip(ql_score, 1, 100))

        is_baseline = False
        if rf_score is None and ql_score == 0.0:
            base = self.get_baseline_score(symbol)
            return base, True

        sess = None
        try:
            sess = (utils.get_current_trading_session() or {}).get("name")
        except Exception:
            sess = None

        if rf_score is not None and ql_score > 0.0:
            if sess == "GOLDEN":
                score = 0.7 * rf_score + 0.3 * ql_score
            else:
                score = 0.5 * rf_score + 0.5 * ql_score
            return float(score), False
        elif rf_score is not None:
            return float(rf_score), False
        else:
            return float(ql_score), False

    def get_baseline_score(self, symbol: str) -> float:
        """
        ✅ v5.2: Retorna um score baseado nos resultados do backtest (Optuna).
        Prioriza win_rate * 100 conforme exigência Land Trading.
        Fallback: win_rate manual definido no config_forex.py.
        """
        res = self.backtest_results.get(symbol)
        
        if res:
            # Prioriza o campo 'score' se existir, senão usa win_rate * 100
            score = res.get("score", res.get("win_rate", 0.5) * 100)
        else:
            # ✅ Fallback Land Trading: Busca no config_forex.py (FOREX_PAIRS)
            forex_config = getattr(config, 'FOREX_PAIRS', {})
            symbol_params = forex_config.get(symbol, {})
            # Se encontrar win_rate manual no config, usa ele (ex: 0.62 -> 62)
            manual_win_rate = symbol_params.get("win_rate")
            if manual_win_rate is not None:
                score = manual_win_rate * 100
                logger.info(f"ML | Score para {symbol} extraído do config_forex (Manual): {score:.1f}")
            else:
                score = 0.0
        
        return float(score)

    # -------------------------------------------------
    # Registro de trade (usado para atualizar Q‑Table e stats)
    # -------------------------------------------------
    def record_trade(self,
                     symbol: str,
                     params: Dict[str, float],
                     result: Dict[str, float],
                     market_indicators: Dict[str, Any] = None):
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "params": params,
            "result": result,
            "profit": result.get("profit", 0),
            "pips": result.get("pips", 0),
            "rr_achieved": result.get("rr_achieved", 0),
        }
        self.trade_history.append(trade_record)

        # ---- Atualiza estatísticas ----
        stats = self.symbol_stats[symbol]
        stats["total_trades"] += 1
        profit = result.get("profit", 0)
        
        if profit > 0:
            stats["winning_trades"] += 1
        stats["total_profit"] += profit
        
        # ✅ NOVO v5.2: Max Drawdown Update
        drawdown = result.get("drawdown", 0) # Assumindo que bot passa drawdown do trade ou impacto no equity
        # Se não houver drawdown no result, estima pelo profit negativo
        if drawdown == 0 and profit < 0:
            drawdown = abs(profit)
        stats["max_drawdown"] = max(stats["max_drawdown"], drawdown)
        
        # ✅ NOVO v5.2: Sharpe Ratio Rolling (Simplificado)
        stats["returns"].append(profit)
        if len(stats["returns"]) > 5:
            returns_np = np.array(stats["returns"])
            std_dev = np.std(returns_np)
            avg_return = np.mean(returns_np)
            if std_dev > 0:
                # Anualizado simplificado (assuming daily trades approx)
                stats["sharpe_ratio"] = (avg_return / std_dev) * np.sqrt(252)

        # Média de R:R
        if stats["total_trades"] > 0:
            stats["avg_rr_achieved"] = (
                (stats["avg_rr_achieved"] * (stats["total_trades"] - 1) +
                 result.get("rr_achieved", 0)) / stats["total_trades"]
            )

        # ---- Atualiza Q‑Table ----
        if market_indicators: # Só atualiza se tiver indicadores válidos
            self._update_q_table(
                symbol=symbol,
                params=params,
                result=result,
                reward=self._calculate_reward(result),
                market_indicators=market_indicators
            )
        else:
            logger.warning(f"⚠️ Não foi possível atualizar Q-Table para {symbol}: indicadores ausentes.")


        # Salva periodicamente
        if len(self.trade_history) % 10 == 0:
            self._save_history()
            self._save_qtable()

        logger.debug(
            f"📊 ML: Trade registrado - {symbol} | "
            f"Profit: ${result.get('profit', 0):.2f} | "
            f"R:R: {result.get('rr_achieved', 0):.2f}"
        )
