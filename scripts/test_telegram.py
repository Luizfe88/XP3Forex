"""
🧪 TELEGRAM TESTER - XP3 PRO FOREX
Testa se a conexão com o Telegram está funcionando.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from xp3_forex.utils.telegram_utils import send_telegram_message
from xp3_forex.core.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
)

def run_test():
    print("--- 🚀 Testando Conexão Telegram ---")
    
    token = settings.TELEGRAM_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID
    
    if not token or token == "your_bot_token":
        print("❌ ERRO: TELEGRAM_TOKEN não configurado no .env")
        return
    
    if not chat_id or chat_id == "your_chat_id":
        print("❌ ERRO: TELEGRAM_CHAT_ID não configurado no .env")
        return

    print(f"Token: {token[:5]}...{token[-5:]}")
    print(f"Chat ID: {chat_id}")
    
    msg = "🔔 *XP3 PRO FOREX*\nTeste de conexão bem-sucedido! Seu robô agora está pronto para enviar notificações."
    
    print("Enviando mensagem de teste...")
    success = send_telegram_message(msg)
    
    if success:
        print("✅ SUCESSO! Verifique seu Telegram.")
    else:
        print("❌ FALHA: Verifique o console para erros do log.")

if __name__ == "__main__":
    run_test()
