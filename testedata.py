import MetaTrader5 as mt5
import pandas as pd

symbol = "EURUSD"

mt5.initialize("C:\\Program Files\\MetaTrader 5 IC Markets Global\\terminal64.exe")

rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1000)

df = pd.DataFrame(rates)

print(df.head(20))
print(df.tail(20))

print("\nVariação mínima:", df['close'].diff().abs().min())
print("Variação média:", df['close'].diff().abs().mean())
print("Número de candles duplicados:", df.duplicated().sum())
print("Número de candles zerados:", (df['close'] == 0).sum())
print("Número de gaps > 1%:", (df['close'].pct_change().abs() > 0.01).sum())

mt5.shutdown()
