#!/usr/bin/env python3
"""
🔍 XP3 Execution Verification Tool
Tests connection, AutoTrading status, and performs a dry-run order_check.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import MetaTrader5 as mt5
from xp3_forex.core.settings import settings
from xp3_forex.utils.mt5_utils import initialize_mt5, mt5_exec
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("XP3.Verifier")

def verify():
    print("\n" + "=" * 70)
    print("🔍 XP3 EXECUTION VERIFIER")
    print("=" * 70 + "\n")

    # 1. Connection
    logger.info("📡 Conectando ao MetaTrader 5...")
    if not initialize_mt5(
        settings.MT5_LOGIN,
        settings.MT5_PASSWORD,
        settings.MT5_SERVER,
        settings.MT5_PATH
    ):
        logger.error("❌ Falha na inicialização do MT5.")
        return False

    term_info = mt5.terminal_info()
    if not term_info:
        logger.error("❌ Não foi possível obter terminal_info.")
        return False

    print("\n📊 TERMINAL STATUS")
    print("-" * 70)
    print(f"Connected:     {term_info.connected}")
    print(f"Trade Allowed: {term_info.trade_allowed} (BOTÃO ALGO TRADING)")
    print(f"DLL Allowed:   {term_info.dlls_allowed}")
    print(f"Server:        {term_info.trade_server}")
    print(f"Company:       {term_info.company}")
    print("-" * 70)

    if not term_info.connected:
        logger.error("❌ O terminal não está conectado à internet/corretora.")
    if not term_info.trade_allowed:
        logger.critical("🚨 O BOTÃO 'ALGO TRADING' ESTÁ DESLIGADO NO MT5!")
        logger.info("💡 Por favor, clique no botão 'Algo Trading' no topo do MetaTrader 5.")

    # 2. Account Health
    acc = mt5.account_info()
    if acc:
        print("\n💰 ACCOUNT INFO")
        print("-" * 70)
        print(f"Login:      {acc.login}")
        print(f"Balance:    ${acc.balance:,.2f}")
        print(f"Equity:     ${acc.equity:,.2f}")
        print(f"Margin Free:${acc.margin_free:,.2f}")
        print(f"Trade Mode: {acc.trade_mode} (0=Demo, 1=Contest, 2=Real)")
        print("-" * 70)
    
    # 3. Dry Run Check (EURUSD)
    symbol = "EURUSD"
    logger.info(f"🧪 Executando teste de order_check em {symbol}...")
    
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.warning(f"⚠️ Não foi possível obter tick para {symbol}. Tentando selecionar...")
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
    
    if tick:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "sl": tick.ask - 200 * mt5.symbol_info(symbol).point,
            "tp": tick.ask + 400 * mt5.symbol_info(symbol).point,
            "deviation": 20,
            "magic": settings.MAGIC_NUMBER,
            "comment": "VERIFY",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        check = mt5.order_check(request)
        if check:
            if check.retcode == 0:
                logger.info(f"✅ order_check OK! Margem necessária: ${check.margin:.2f}")
            else:
                logger.error(f"❌ order_check REJEITADO: {check.retcode} ({check.comment})")
        else:
            logger.error("❌ order_check retornou None.")
    else:
        logger.error(f"❌ Falha ao obter dados de {symbol}.")

    mt5.shutdown()
    print("\n" + "=" * 70)
    return True

if __name__ == "__main__":
    verify()
