"""
🚀 MANUAL LEARNING RUNNER
Executa o ciclo de aprendizado manualmente para teste.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from xp3_forex.optimization.learner import DailyLearner
from xp3_forex.reporting.daily_report import DailyReportGenerator
from xp3_forex.core.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
)

def run_test():
    print("--- 🚀 Iniciando Teste do XP3 Learner ---")
    
    # Símbolos para teste rápido
    test_symbols = ["EURUSD", "USDJPY"]
    
    learner = DailyLearner(test_symbols)
    report_gen = DailyReportGenerator()
    
    print(f"1. Rodando Otimização para {test_symbols} (isso pode levar alguns minutos)...")
    learnings = learner.run_full_learning()
    
    if learnings:
        print("2. Gerando Relatório de Aprendizado...")
        report_path = report_gen.generate_learning_report(learnings)
        print(f"✅ SUCESSO! Relatório gerado em: {report_path}")
        
        # Mostrar o conteúdo do relatório
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n--- CONTEÚDO DO RELATÓRIO ---\n")
            print(f.read())
    else:
        print("❌ FALHA: O learner não retornou resultados.")

if __name__ == "__main__":
    run_test()
