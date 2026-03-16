"""
🧠 LEARNER - XP3 PRO FOREX
Automatiza a recalibração de parâmetros usando SessionOptimizer.
"""

import logging
import json
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from xp3_forex.core.settings import settings
from xp3_forex.optimization.session_optimizer import SessionOptimizer, update_session_params_json
from xp3_forex.utils.mt5_utils import get_rates, initialize_mt5

logger = logging.getLogger("XP3.Learner")

class DailyLearner:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.history_days = 30 # Analisar os últimos 30 dias para aprender
        
    def run_full_learning(self):
        """Executa o ciclo de aprendizado para todos os símbolos ativos"""
        logger.info(f"🔮 Iniciando Ciclo de Aprendizado para {len(self.symbols)} ativos...")
        
        if not initialize_mt5():
            logger.error("MT5 não inicializado para o Learner.")
            return False
            
        results = {}
        
        for symbol in self.symbols:
            try:
                # Pegar dados de H1 para otimização de sessão (mais estável)
                # 24 barras por dia * 30 dias = 720 barras. Pegamos 1000 para ter margem.
                df = get_rates(symbol, 60, 1000)
                if df is None or len(df) < 100:
                    logger.warning(f"Dados insuficientes para aprender com {symbol}")
                    continue
                
                df.set_index('time', inplace=True)
                symbol_learnings = {}
                
                for sess in ["ASIA", "LONDON", "NY"]:
                    logger.info(f"🧠 Aprendendo com {symbol} na sessão {sess}...")
                    optimizer = SessionOptimizer(symbol, sess, df)
                    best_params = optimizer.run_optimization(n_trials=30) # 30 trials por sessão
                    
                    if best_params:
                        symbol_learnings[sess] = best_params
                        update_session_params_json(symbol, sess, best_params)
                
                if symbol_learnings:
                    results[symbol] = symbol_learnings
                    
            except Exception as e:
                logger.error(f"Erro no aprendizado de {symbol}: {e}")
                
        logger.info(f"✅ Ciclo de aprendizado concluído. {len(results)} ativos atualizados.")
        return results

if __name__ == "__main__":
    # Teste isolado
    logging.basicConfig(level=logging.INFO)
    learner = DailyLearner(["EURUSD", "USDJPY", "GBPUSD"])
    learner.run_full_learning()
