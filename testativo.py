import MetaTrader5 as mt5
import pandas as pd

def testar_ativo(symbol):
    if not mt5.initialize():
        print("❌ Falha ao iniciar MT5")
        return

    # Tenta selecionar o ativo no Market Watch
    selected = mt5.symbol_select(symbol, True)
    if not selected:
        print(f"❌ O ativo {symbol} não foi encontrado ou não pode ser selecionado.")
        return

    # Tenta pegar o último tick
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"✅ Conexão OK para {symbol}!")
        print(f"BID: {tick.bid} | ASK: {tick.ask} | Last: {tick.last}")
    else:
        print(f"⚠️ O ativo {symbol} existe, mas não está recebendo ticks (Mercado fechado?)")

    mt5.shutdown()

# TESTE AQUI (Exemplo com Par de Forex)
testar_ativo("EURUSD")
