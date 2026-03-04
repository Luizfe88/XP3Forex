"""
📱 TELEGRAM UTILS - XP3 PRO FOREX
Envio de notificações de trades e relatórios para o Telegram.
"""

import requests
import logging
import os
from pathlib import Path
from typing import Optional, Union

from xp3_forex.core.settings import settings

logger = logging.getLogger("XP3.Telegram")

def send_telegram_message(message: str) -> bool:
    """Envia uma mensagem de texto simples ou Markdown para o chat configurado."""
    if not settings.NOTIFICATIONS_ENABLED or not settings.TELEGRAM_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Erro ao enviar mensagem para o Telegram: {e}")
        return False

def send_telegram_document(file_path: Union[str, Path], caption: Optional[str] = None) -> bool:
    """Envia um arquivo (documento) para o chat configurado."""
    if not settings.NOTIFICATIONS_ENABLED or not settings.TELEGRAM_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return False

    path = Path(file_path)
    if not path.exists():
        logger.error(f"Arquivo não encontrado para envio Telegram: {path}")
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/sendDocument"
    params = {"chat_id": settings.TELEGRAM_CHAT_ID}
    if caption:
        params["caption"] = caption

    try:
        with open(path, 'rb') as f:
            files = {'document': f}
            response = requests.post(url, data=params, files=files, timeout=30)
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Erro ao enviar documento para o Telegram: {e}")
        return False

if __name__ == "__main__":
    # Teste rápido
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        send_telegram_message(sys.argv[1])
    else:
        print("Uso: python telegram_utils.py 'Minha mensagem'")
