"""
XP3 PRO FOREX - Total Automation Calibrator
Discovers all symbols and runs quantitative optimization in batch.
"""

import logging
from typing import List, Optional
from xp3_forex.core.settings import settings
from xp3_forex.utils.mt5_utils import get_rates, initialize_mt5, initialize_market_data
from xp3_forex.optimization.quant_optimizer import QuantOptimizer, save_optimized_quant_params

import argparse
import MetaTrader5 as mt5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XP3.AutoCalibrator")

def run_total_automation(target_symbols: Optional[List[str]] = None, n_trials: int = 30):
    """
    Runs the calibration process for all active symbols in the configuration.
    """
    logger.info("🚀 Iniciando automação total de calibração quantitativa...")
    
    if not initialize_mt5(
        login=settings.MT5_LOGIN,
        password=settings.MT5_PASSWORD,
        server=settings.MT5_SERVER,
        path=settings.MT5_PATH
    ):
        logger.error("❌ Falha ao inicializar MT5. Abortando.")
        return

    # 1. Discover Symbols
    if target_symbols:
        symbols = target_symbols
    else:
        symbols = settings.symbols_list
        
    if "ALL" in [s.upper() for s in symbols]:
        logger.info("🌍 'ALL' detectado. Buscando todos os símbolos do Market Watch...")
        all_mt5_symbols = mt5.symbols_get()
        if all_mt5_symbols:
            # Filtra apenas o que está no Market Watch (visible=True)
            symbols = [s.name for s in all_mt5_symbols if s.visible]
            if not symbols:
                 # Fallback: Se o Market Watch estiver vazio, pega os principais
                 logger.warning("⚠️ Market Watch vazio. Buscando primeiros 20 ativos do terminal...")
                 symbols = [s.name for s in all_mt5_symbols][:20]
        else:
            logger.warning("⚠️ Nenhum símbolo encontrado no terminal.")
            
    logger.info(f"🔍 Símbolos para calibração: {symbols}")
    
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
    parser = argparse.ArgumentParser(description="XP3 Quantitative Auto-Calibrator")
    parser.add_argument("--symbols", type=str, help="Lista de símbolos separados por vírgula ou 'ALL'")
    parser.add_argument("--trials", type=int, default=20, help="Número de tentativas Optuna por ativo")
    
    args = parser.parse_args()
    
    target_list = None
    if args.symbols:
        target_list = [s.strip().upper() for s in args.symbols.split(",")]
        
    run_total_automation(target_symbols=target_list, n_trials=args.trials)
