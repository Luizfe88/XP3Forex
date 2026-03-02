import sys
import os

# Adicionar o diretório src ao path para importar o pacote
sys.path.append(os.path.abspath("src"))

from xp3_forex.core.settings import settings
from xp3_forex.core.trade_executor import trade_executor
from xp3_forex.utils.calculations import calculate_lot_size

def test_lot_size_limit():
    print(f"Configuração MAX_LOTS_PER_TRADE: {settings.MAX_LOTS_PER_TRADE}")
    
    # Mock de informações de símbolo para o executor
    # No executor real, ele pega do MT5, mas podemos testar a lógica de cálculo
    
    # Testando a função utilitária diretamente
    risk_amount = 1000  # Risco alto para forçar lote alto
    stop_loss_pips = 10
    
    # EURUSD: pip_size = 0.0001
    # Profit = lot * pips * pip_value
    # Lot = risk / (pips * pip_size * tick_value)
    
    calculated = calculate_lot_size("EURUSD", risk_amount, stop_loss_pips)
    print(f"Lote calculado para EURUSD (Risco ${risk_amount}, SL 10 pips): {calculated}")
    
    if calculated <= settings.MAX_LOTS_PER_TRADE:
        print("✅ Verificação de calculate_lot_size: SUCESSO")
    else:
        print("❌ Verificação de calculate_lot_size: FALHA")

if __name__ == "__main__":
    test_lot_size_limit()
