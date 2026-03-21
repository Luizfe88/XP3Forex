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
            self.ml_confidence = np.full(len(self.filtered_data), 0.5)
            self.optimal_ml_threshold = 0.55
        else:
            # Treinar ML real uma única vez por sessão (performance!)
            from xp3_forex.ml import train_ml_model
            self._ml_model, self.ml_confidence = train_ml_model(self.filtered_data)
            self.optimal_ml_threshold = self._get_optimal_ml_threshold()

    def objective(self, trial: optuna.Trial) -> float:
        from xp3_forex.optimization import engine as optimizer
        
        # Revertendo para o espaço 7D (ciclo anterior) para melhor convergência em pares com dados.
        # Parâmetros sugeridos pelo Optuna (7 dimensões):
        ema_short     = trial.suggest_int("ema_short", 5, 50)
        ema_long      = trial.suggest_int("ema_long", ema_short + 10, 200)
        rsi_low       = trial.suggest_int("rsi_low", 25, 45)
        rsi_high      = trial.suggest_int("rsi_high", 55, 75)
        adx_threshold = trial.suggest_int("adx_threshold", 15, 35)
        sl_atr        = trial.suggest_float("sl_atr", 1.0, 3.0)
        
        # Mantendo R:R mínimo de 1.5 via ratio
        rr_ratio      = trial.suggest_float("rr_ratio", 1.5, 4.0)
        tp_atr        = sl_atr * rr_ratio
        
        # ml_threshold continua fixo por precisão real (implementação superior)
        ml_threshold  = self.optimal_ml_threshold

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
            "ml_confidence": self.ml_confidence, # Vindo do treino no __init__
        }

        if len(data_payload["df"]) < 200:  # 200 barras filtradas = ~2 meses de uma sessão
            return -10.0

        try:
            metrics, _, _ = optimizer.run_backtest_with_params(data_payload, params)
            if not isinstance(metrics, dict):
                return -10.0
            
            total_trades = metrics.get("total_trades", 0)
            win_rate = metrics.get("win_rate", 0.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            # Floor no drawdown para evitar scores infinitos em séries curtas
            drawdown = max(metrics.get("drawdown", 0.0), 0.02)
            sharpe = metrics.get("sharpe", 0.0)
            
            # Mínimo absoluto de trades para considerar válido
            if total_trades < 10:
                return (total_trades / 10.0) * 0.5  # Penalidade severa, nunca chega a 1.0
            
            # Score composto sem cap artificial:
            # - Premia win_rate e profit_factor (retorno)
            # - Premia volume de trades (significância estatística)  
            # - Penaliza drawdown
            # - Premia consistência via Sharpe
            score = (win_rate * profit_factor * (1 + np.log1p(total_trades)) * (1 + sharpe * 0.1)) / drawdown
            
            # Sanity check: penalizar fortemente se tiver menos de 30 trades
            if total_trades < 30:
                score *= np.sqrt(total_trades / 30.0)
            
            return float(score)
        except Exception:
            return -10.0

    def run_optimization(self, n_trials: int = 50):
        """Executa a otimização para a sessão específica"""
        if len(self.filtered_data) < 50:
            return None, 0.0
            
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"Otimização finalizada para {self.symbol} na sessão {self.session_name}.")
        logger.info(f"Melhor Score: {study.best_value}")
        
        best_params = study.best_params.copy()
        # sl_atr já está no best_params
        if "rr_ratio" in best_params:
            best_params["tp_atr"] = best_params["sl_atr"] * best_params["rr_ratio"]
            del best_params["rr_ratio"]
            
        best_params["ml_threshold"] = self.optimal_ml_threshold
        
        return best_params, study.best_value

    def _get_optimal_ml_threshold(self) -> float:
        """Encontra o threshold que maximiza precision no conjunto de validação."""
        if not hasattr(self, '_ml_model') or self._ml_model is None:
            return 0.55  # fallback conservador
        
        from xp3_forex.ml import predict_ml_model
        probs = predict_ml_model(self._ml_model, self.filtered_data)
        best_threshold, best_precision = 0.55, 0.0
        
        for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            selected = probs >= t
            if selected.sum() < 20:
                continue
            # precision proxy: taxa de retornos positivos nos candles selecionados (4 candles à frente = 1h em M15)
            close = self.filtered_data['close'].values
            future_ret = (np.roll(close, -4) - close) / close
            precision = float(np.mean(future_ret[selected] > 0))
            if precision > best_precision:
                best_precision, best_threshold = precision, t
        
        logger.info(f"[{self.symbol}-{self.session_name}] Optimal ML Threshold: {best_threshold} (Precision: {best_precision:.4f})")
        return best_threshold

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
