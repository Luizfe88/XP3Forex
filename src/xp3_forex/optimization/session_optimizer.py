"""
🎯 SESSION OPTIMIZER - XP3 PRO FOREX
Otimização de parâmetros baseada em sessões (Asia, London, NY).
"""

import json
import optuna
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from xp3_forex.core.settings import settings

logger = logging.getLogger("XP3.SessionOptimizer")

def filter_data_by_session(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    """
    Filtra as barras do DataFrame baseadas no horário da sessão (UTC).
    """
    if session_name == "ASIA":
        # 22:00 - 08:00
        return df[(df.index.hour >= 22) | (df.index.hour < 8)]
    elif session_name == "LONDON":
        # 08:00 - 16:00
        return df[(df.index.hour >= 8) & (df.index.hour < 16)]
    elif session_name == "NY":
        # 13:00 - 21:00
        return df[(df.index.hour >= 13) & (df.index.hour < 21)]
    return df

class SessionOptimizer:
    """
    Gerencia a otimização de parâmetros por sessão usando Optuna.
    """
    
    def __init__(self, symbol: str, session_name: str, historical_data: pd.DataFrame):
        self.symbol = symbol
        self.session_name = session_name
        self.raw_data = historical_data
        self.filtered_data = filter_data_by_session(historical_data, session_name)
        
        if len(self.filtered_data) < 100:
            logger.warning(f"Dados insuficientes para a sessão {session_name} do ativo {symbol}. (Barras: {len(self.filtered_data)})")

    def objective(self, trial: optuna.Trial) -> float:
        import optimizer_optuna_forex as optimizer

        ema_short     = trial.suggest_int("ema_short", 5, 50)
        ema_long      = trial.suggest_int("ema_long", 20, 200)
        rsi_low       = trial.suggest_int("rsi_low", 25, 45)
        rsi_high      = trial.suggest_int("rsi_high", 55, 75)
        adx_threshold = trial.suggest_int("adx_threshold", 15, 35)
        sl_atr        = trial.suggest_float("sl_atr", 1.0, 3.0)
        tp_atr        = trial.suggest_float("tp_atr", 1.5, 4.0)
        ml_threshold  = trial.suggest_float("ml_threshold", 0.50, 0.80)

        params = {
            "ema_short": ema_short, "ema_long": ema_long,
            "rsi_low": rsi_low, "rsi_high": rsi_high,
            "adx_threshold": adx_threshold, "sl_atr": sl_atr,
            "tp_atr": tp_atr, "ml_threshold": ml_threshold,
        }

        # Dados já filtrados por sessão (self.filtered_data)
        data_payload = {
            "df": self.filtered_data,
            "pip_size": 0.0001,
            "tick_value": 10.0,
            "spread": 1.2,
            "news_mask": np.zeros(len(self.filtered_data), dtype=np.bool_),
            "ml_confidence": np.full(len(self.filtered_data), 0.6),
        }

        if len(data_payload["df"]) < 500:
            return -10.0

        try:
            metrics, _, _ = optimizer.run_backtest_with_params(data_payload, params)
            if not isinstance(metrics, dict):
                return -10.0
            pf  = float(metrics.get("profit_factor", 0.0))
            wr  = float(metrics.get("win_rate", 0.0))
            dd  = max(float(metrics.get("drawdown", 1.0)), 0.01)
            return (wr * pf) / dd
        except Exception:
            return -10.0

    def run_optimization(self, n_trials: int = 50):
        """Executa a otimização para a sessão específica"""
        if len(self.filtered_data) < 50:
            return None
            
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"Otimização finalizada para {self.symbol} na sessão {self.session_name}.")
        logger.info(f"Melhor Score: {study.best_value}")
        
        return study.best_params

def update_session_params_json(symbol: str, session_name: str, params: Dict[str, Any]):
    """
    Atualiza o arquivo data/session_optimized_params.json com os novos resultados.
    """
    json_path = settings.DATA_DIR / "session_optimized_params.json"
    
    try:
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                all_params = json.load(f)
        else:
            all_params = {}

        if symbol not in all_params:
            all_params[symbol] = {}
            
        all_params[symbol][session_name] = params
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_params, f, indent=4)
            
        logger.info(f"Parâmetros de {symbol} na sessão {session_name} atualizados no JSON.")

    except Exception as e:
        logger.error(f"Erro ao atualizar JSON de parâmetros: {e}")

if __name__ == "__main__":
    # Exemplo de uso para treinamento local
    from xp3_forex.utils.mt5_utils import get_rates, initialize_mt5
    
    if initialize_mt5():
        df_h1 = get_rates("EURUSD", 60, 2000)
        if df_h1 is not None:
            df_h1.set_index('time', inplace=True)
            
            for sess in ["ASIA", "LONDON", "NY"]:
                optimizer = SessionOptimizer("EURUSD", sess, df_h1)
                best = optimizer.run_optimization(n_trials=20)
                if best:
                    update_session_params_json("EURUSD", sess, best)
