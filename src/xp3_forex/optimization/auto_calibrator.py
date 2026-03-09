"""
XP3 PRO FOREX - Total Automation Calibrator
Discovers all symbols and runs quantitative optimization in batch.
"""

import logging
from typing import List
from xp3_forex.core.settings import settings
from xp3_forex.utils.mt5_utils import get_rates, initialize_mt5, initialize_market_data
from xp3_forex.optimization.quant_optimizer import QuantOptimizer, save_optimized_quant_params

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XP3.AutoCalibrator")

def run_total_automation(n_trials: int = 30):
    """
    Runs the calibration process for all active symbols in the configuration.
    """
    logger.info("🚀 Iniciando automação total de calibração quantitativa...")
    
    if not initialize_mt5():
        logger.error("❌ Falha ao inicializar MT5. Abortando.")
        return

    # 1. Discover Symbols
    symbols = settings.symbols_list
    logger.info(f"🔍 Símbolos detectados: {symbols}")
    
    # 2. Ensure data availability
    validated_symbols = initialize_market_data(symbols)
    
    # 3. Iterate and Optimize
    results_count = 0
    for sym in validated_symbols:
        try:
            logger.info(f"--- Otimizando {sym} ---")
            
            # Fetch H1 data (default lookback for calibration)
            df = get_rates(sym, 60, 3000)
            if df is None or len(df) < 1500:
                logger.warning(f"⚠️ Dados insuficientes para {sym}. Pulando.")
                continue
                
            optimizer = QuantOptimizer(sym, df)
            best_params = optimizer.run_optimization(n_trials=n_trials)
            
            if best_params:
                save_optimized_quant_params(sym, best_params)
                results_count += 1
                logger.info(f"✅ {sym} calibrado com sucesso.")
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar {sym}: {e}")
            
    logger.info(f"🏁 Automação concluída. {results_count}/{len(symbols)} ativos calibrados.")

if __name__ == "__main__":
    # We can pass an argument to control the depth of optimization
    run_total_automation(n_trials=20) # Conservative trials for batch processing
