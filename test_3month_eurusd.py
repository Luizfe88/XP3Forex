import os
import sys
import pandas as pd
import numpy as np
import logging

# --- Configuração de Ambiente ---
# 3 meses em M15 (~6000 candles)
os.environ["XP3_WFO_TEST_LEN"] = "6000" 
os.environ["XP3_WFO_TRAIN_LEN"] = "15000" # Aumentei o treino para o ML ter mais dados

# Ajuste de Logging para ver o progresso real
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

try:
    import otimizador_semanal_forex as app
except ImportError:
    sys.path.append(os.getcwd())
    import otimizador_semanal_forex as app

# --- MONKEYPATCH CRÍTICO ---
import config_forex as config

# AUMENTO DE TENTATIVAS: De 20 para 100
# Menos que 50 é sorte. Entre 50 e 100 é ciência.
config.OPTUNA_N_TRIALS = 100 

print(f"🔧 Configuração Land Trading: Trials={config.OPTUNA_N_TRIALS} | TrainLen={os.environ['XP3_WFO_TRAIN_LEN']}")

def run_test():
    print("🚀 Inicializando Sistema Land Trading...")
    
    if not app._mt5_open_session():
        print("⚠️ Falha no MT5. Tentando modo offline/CSV...")

    symbol = "EURUSD"
    print(f"📥 Carregando dados para {symbol}...")
    
    data = app.load_data_v7_enhanced(symbol)

    if data:
        print(f"✅ Dados carregados: {len(data['df'])} candles")
        
        # --- INJEÇÃO DE ESPAÇO DE BUSCA (Se suportado pelo otimizador) ---
        # Isso força o Optuna a não perder tempo com parâmetros inúteis
        # Se o seu 'otimizador_semanal_forex.py' usar 'search_space_injection', isso ajuda.
        # Caso contrário, o aumento de trials já resolverá.
        
        print("\n🔎 Iniciando Otimização (Aguarde o processamento de 100 gerações)...")
        
        # Executa o Worker
        try:
            result = app.worker_process_asset(symbol, data)
        except KeyboardInterrupt:
            print("\n🛑 Interrompido pelo usuário.")
            return

        print("\n" + "="*50)
        print(f"RELATÓRIO FINAL ({symbol})")
        print("="*50)
        print(f"Status: {result.get('status')}")
        
        if result.get('status') == 'SUCCESS':
            metrics = result.get('metrics_oos', {})
            
            # Cores para o terminal
            GREEN = '\033[92m'
            RED = '\033[91m'
            RESET = '\033[0m'
            
            pf = metrics.get('profit_factor', 0)
            color = GREEN if pf > 1.1 else RED
            
            print(f"Win Rate:      {metrics.get('win_rate', 0):.2%}")
            print(f"Profit Factor: {color}{pf:.2f}{RESET}")
            print(f"Total Trades:  {metrics.get('total_trades', 0)}")
            print(f"Drawdown:      {metrics.get('drawdown', 0):.2%}")
            print(f"Sharpe V7:     {metrics.get('sharpe', 0):.2f}")
            print("-" * 30)
            print("🏆 Melhores Parâmetros Encontrados:")
            import json
            print(json.dumps(result.get('best_params'), indent=2))
            
            if pf < 1.0:
                print("\n⚠️ ANÁLISE DO ENG. CHEFE: O sistema ainda não é lucrativo.")
                print("Sugestão: Aumente OPTUNA_N_TRIALS para 300 ou reduza o 'sl_atr' no search space.")
            else:
                print("\n✅ SUCESSO: O sistema encontrou convergência lucrativa.")
                
        else:
            print("❌ FALHA NA OTIMIZAÇÃO")
            print("Erro:", result.get('message'))

        app._mt5_close_session()
    else:
        print("❌ Erro ao carregar dados. Verifique o MT5 ou arquivo CSV.")

if __name__ == "__main__":
    run_test()