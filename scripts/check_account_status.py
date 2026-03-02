#!/usr/bin/env python3
"""
🔍 XP3 Account Diagnostic Tool
Verifica status da conta, margin, equity e identifica problemas potenciais.
"""

import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import MetaTrader5 as mt5
from xp3_forex.utils.mt5_utils import initialize_mt5, mt5_exec
from xp3_forex.core.settings import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)

logger = logging.getLogger(__name__)


def check_account_status():
    """Verifica status completo da conta e reporta issues"""

    print("\n" + "=" * 70)
    print("🔍 XP3 ACCOUNT DIAGNOSTIC TOOL")
    print("=" * 70 + "\n")

    # 1. Initialize MT5
    print("📡 Conectando ao MetaTrader 5...")
    if not initialize_mt5(
        settings.MT5_LOGIN,
        settings.MT5_PASSWORD,
        settings.MT5_SERVER,
        settings.MT5_PATH,
    ):
        print("❌ Falha ao conectar ao MT5. Verifique credenciais.")
        return False

    print("✅ Conectado ao MT5\n")

    # 2. Get Account Info
    account = mt5_exec(mt5.account_info)
    if not account:
        print("❌ Falha ao obter informações da conta.")
        return False

    # 3. Display Account Info
    print("💰 ACCOUNT INFORMATION")
    print("-" * 70)
    print(f"Login:                 {account.login}")
    print(f"Server:                {account.server}")
    print(f"Company:               {account.company}")
    print(f"Leverage:              {account.leverage}:1")
    print(f"Trading Mode:          {account.trade_mode}")
    print()

    # 4. Display Balance Info
    print("💵 BALANCE & EQUITY")
    print("-" * 70)
    print(f"Balance:               ${account.balance:,.2f}")
    print(f"Equity:                ${account.equity:,.2f}")
    print(f"Credit:                ${account.credit:,.2f}")
    print()

    # 5. Display Margin Info (CRITICAL)
    print("📊 MARGIN INFORMATION")
    print("-" * 70)
    print(f"Margin Required:       ${account.margin:,.2f}")
    print(f"Margin Free:           ${account.margin_free:,.2f}")
    if account.margin > 0:
        print(f"Margin Level:          {account.margin_level:.1f}%")
    else:
        print("Margin Level:          ∞ (no positions - all margin available)")
    print()

    # 6. Check Positions
    print("📈 OPEN POSITIONS")
    print("-" * 70)
    positions = mt5_exec(mt5.positions_total)
    if positions is None:
        print("❌ Falha ao contar posições")
    elif positions == 0:
        print("✅ Nenhuma posição aberta")
    else:
        print(f"⚠️  {positions} posição(ões) aberta(s)")
        pos_list = mt5_exec(mt5.positions_get)
        if pos_list:
            for pos in pos_list:
                print(
                    f"   - {pos.symbol}: {pos.type} {pos.volume}@{pos.price_open} (Profit=${pos.profit:,.2f})"
                )
    print()

    # 7. Health Checks (DIAGNOSIS)
    print("🏥 HEALTH CHECK")
    print("-" * 70)

    issues = []
    warnings = []

    # Calculate margin_level correctly
    margin_level = account.margin_level if account.margin > 0 else float('inf')

    # Check 1: Margin
    if account.margin_free <= 0:
        issues.append("❌ MARGIN CRÍTICO: Sem margin disponível! Trading IMPOSSÍVEL.")
    elif account.margin > 0 and margin_level < 100:  # Only check if positions exist
        issues.append(
            f"❌ MARGIN CALL: Margin Level {margin_level:.1f}% < 100%"
        )
    elif account.margin > 0 and margin_level < 200:  # Warning level
        warnings.append(f"⚠️  Margin Level baixo: {margin_level:.1f}% (risco de margin call)")
            f"⚠️  Margin Level baixo: {account.margin_level:.1f}% (risco de margin call)"
        )

    # Check 2: Equity
    if account.equity <= 0:
        issues.append(f"❌ CONTA NEGATIVA: Equity ${account.equity:,.2f}")
    elif account.equity < account.balance * 0.8:
        pct = (account.equity / account.balance) * 100
        warnings.append(f"⚠️  Grande drawdown: Equity é {pct:.1f}% do balance")

    # Check 3: Profit/Loss
    profit = account.equity - account.balance - account.credit
    if profit < -account.balance * 0.5:
        warnings.append(f"⚠️  Perda acumulada > 50% do balance: ${profit:,.2f}")

    # Display Results
    if not issues and not warnings:
        print("✅ CONTA SAUDÁVEL - Sem problemas detectados!")
    else:
        if issues:
            print("\n🚨 ISSUES CRÍTICAS:")
            for issue in issues:
                print(f"  {issue}")

        if warnings:
            print("\n⚠️  AVISOS:")
            for warning in warnings:
                print(f"  {warning}")

    print()

    # 8. Additional Diagnostics
    print("📋 TRADEABLE SYMBOLS CHECK")
    print("-" * 70)

    from xp3_forex.mt5.symbol_manager import symbol_manager

    tradable = symbol_manager.get_tradable_symbols()
    print(f"Símbolos disponíveis: {len(tradable)}")
    if tradable:
        print(f"Exemplos: {', '.join(tradable[:5])}")
    else:
        print("⚠️  Nenhum símbolo negociável encontrado!")

    print()
    print("=" * 70)

    # Cleanup
    mt5.shutdown()

    return len(issues) == 0


if __name__ == "__main__":
    success = check_account_status()
    sys.exit(0 if success else 1)
