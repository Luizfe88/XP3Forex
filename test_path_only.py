
import MetaTrader5 as mt5
path = r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe"
print(f"Testing initialization with path only: {path}")
success = mt5.initialize(path=path)
if success:
    print("✅ SUCCESS!")
    term_info = mt5.terminal_info()
    acc_info = mt5.account_info()
    if term_info: print(f"Path: {term_info.path}")
    if acc_info: print(f"Account: {acc_info.login}, Server: {acc_info.server}")
    mt5.shutdown()
else:
    print(f"❌ FAILED: {mt5.last_error()}")
