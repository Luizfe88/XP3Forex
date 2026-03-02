#!/usr/bin/env python3
"""
🔍 XP3 Order Diagnostic Tool
Testa um ciclo completo de order_check para diagnosticar rejeições 10013
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import MetaTrader5 as mt5
from xp3_forex.utils.mt5_utils import initialize_mt5, mt5_exec
from xp3_forex.core.settings import settings
from xp3_forex.mt5.symbol_manager import symbol_manager
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)

logger = logging.getLogger(__name__)


def test_order_check():
    """Test order_check on GBPJPY to diagnose error 10013"""

    print("\n" + "=" * 70)
    print("🔍 XP3 ORDER CHECK DIAGNOSTIC TOOL")
    print("=" * 70 + "\n")

    # Initialize MT5
    print("📡 Conectando ao MetaTrader 5...")
    if not initialize_mt5(
        settings.MT5_LOGIN,
        settings.MT5_PASSWORD,
        settings.MT5_SERVER,
        settings.MT5_PATH,
    ):
        print("❌ Falha ao conectar ao MT5")
        return False

    print("✅ Conectado ao MT5\n")

    # Get account info
    account = mt5_exec(mt5.account_info)
    print("💰 ACCOUNT STATUS")
    print("-" * 70)
    print(f"Balance: ${account.balance:,.2f}")
    print(f"Equity: ${account.equity:,.2f}")
    print(f"Margin Free: ${account.margin_free:,.2f}")
    print(
        f"Margin Level: {account.margin_level if account.margin > 0 else '∞ (no positions)'}%"
    )
    print()

    # Test GBPJPY
    symbol = "GBPJPY"
    print(f"🧪 TESTING SYMBOL: {symbol}")
    print("-" * 70)

    # Get symbol info
    sym_info = mt5_exec(mt5.symbol_info, symbol)
    if not sym_info:
        print(f"❌ Symbol {symbol} not found")
        mt5.shutdown()
        return False

    print(f"Trade Mode: {sym_info.trade_mode}")
    print(
        f"Volume Min: {sym_info.volume_min}, Max: {sym_info.volume_max}, Step: {sym_info.volume_step}"
    )
    print(f"Bid: {sym_info.bid:.5f}, Ask: {sym_info.ask:.5f}")
    print(f"Point: {sym_info.point}")
    print()

    # Create a test order
    print("📋 CREATING TEST BUY ORDER")
    print("-" * 70)

    entry_price = sym_info.ask  # Fresh price
    volume = 0.01
    sl = entry_price - 100 * sym_info.point  # 100 points below
    tp = entry_price + 200 * sym_info.point  # 200 points above

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY,
        "price": float(entry_price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 50,
        "magic": settings.MAGIC_NUMBER,
        "comment": "DIAGNOSTIC",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    print(f"Symbol: {request['symbol']}")
    print(f"Type: BUY")
    print(f"Volume: {request['volume']:.2f}")
    print(f"Price: {request['price']:.5f}")
    print(f"SL: {request['sl']:.5f}")
    print(f"TP: {request['tp']:.5f}")
    print()

    # Test order_check
    print("🔍 RUNNING ORDER_CHECK")
    print("-" * 70)

    check = mt5_exec(mt5.order_check, request=request, timeout=10)

    if check is None:
        print("❌ order_check returned None!")
        mt5.shutdown()
        return False

    print(f"Retcode: {check.retcode} ({check.comment})")
    print(f"Check Margin: ${check.margin:.2f}")
    print(f"Check Margin Free: ${check.margin_free:.2f}")

    # Check for optional attributes
    if hasattr(check, "commission"):
        print(f"Check Commission: ${check.commission:.2f}")
    if hasattr(check, "profit"):
        print(f"Check Profit: ${check.profit:.2f}")
    print()

    if check.retcode == 0:
        print("✅ ORDER_CHECK OK - Order should be valid")
    else:
        print(f"❌ ORDER_CHECK FAILED with retcode {check.retcode}")
        print(f"   Comment: {check.comment}")
        if check.retcode == 10013:
            print("   This usually means:")
            print("   1. Invalid order parameters")
            print("   2. Symbol not properly enabled")
            print("   3. Prices outside valid range")
            print("   4. Stop loss too close to entry")

    print()
    print("=" * 70)
    mt5.shutdown()

    return check.retcode == 0


if __name__ == "__main__":
    success = test_order_check()
    sys.exit(0 if success else 1)
