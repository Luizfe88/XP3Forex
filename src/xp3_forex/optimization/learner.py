"""
🧠 LEARNER - XP3 PRO FOREX
Automatiza a recalibração de parâmetros usando SessionOptimizer.
"""

import logging
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
                # Quantidade necessária considerando o filtro de sessão:
                # ASIA = 10h/24h = 42% dos candles M15
                # Mínimo 1000 barras pós-filtro → precisamos de 1000 / 0.42 ≈ 2.400 barras M15 brutas
                # Usar 3.000 barras brutas garante margem para todas as sessões
                df = get_rates(symbol, 15, 3000)
                if df is None or len(df) < 500:
                    logger.warning(f"Dados insuficientes para aprender com {symbol}")
                    continue
                
                df.set_index('time', inplace=True)
                symbol_learnings = {}
                
                for sess in ["ASIA", "LONDON", "NY"]:
                    logger.info(f"🧠 Aprendendo com {symbol} na sessão {sess}...")
                    optimizer = SessionOptimizer(symbol, sess, df)
                    best_params, best_value = optimizer.run_optimization(n_trials=30)
                    
                    if best_params and best_value > 0.0:
                        symbol_learnings[sess] = best_params
                        update_session_params_json(symbol, sess, best_params)
                    else:
                        logger.warning(f"⚠️ {symbol}/{sess}: Score {best_value} — sem trades suficientes ou sem lucro. Ignorando atualização.")
                
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
