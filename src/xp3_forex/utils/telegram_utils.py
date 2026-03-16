"""
📱 TELEGRAM UTILS - XP3 PRO FOREX
Envio de notificações de trades e relatórios para o Telegram.
"""

import requests
import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datetime import datetime

from xp3_forex.core.settings import settings

logger = logging.getLogger("XP3.Telegram")

def send_telegram_message(message: str, parse_mode: str = "Markdown") -> bool:
    """Envia uma mensagem de texto para o Telegram."""
    if not settings.NOTIFICATIONS_ENABLED or not settings.TELEGRAM_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode
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

class TelegramControl:
    """
    Listener de comandos para o Telegram via Long Polling.
    Lida com comandos de controle do robô.
    """
    def __init__(self, bot):
        self.bot = bot
        self.token = settings.TELEGRAM_TOKEN
        self.authorized_chat_id = str(settings.TELEGRAM_CHAT_ID)
        self.last_update_id = 0
        self.is_running = False
        self._thread = None

    def start(self):
        """Inicia o listener em uma thread separada"""
        if not self.token or not self.authorized_chat_id:
            logger.warning("TelegramControl: Token ou Chat ID não configurados. Listener desativado.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._thread.start()
        logger.info("📱 Telegram Control Listener iniciado.")

    def stop(self):
        """Para o listener"""
        self.is_running = False

    def _polling_loop(self):
        """Loop de polling para buscar novos comandos"""
        while self.is_running:
            try:
                url = f"https://api.telegram.org/bot{self.token}/getUpdates"
                params = {"offset": self.last_update_id + 1, "timeout": 30}
                
                response = requests.get(url, params=params, timeout=35)
                if response.status_code == 200:
                    updates = response.json().get("result", [])
                    for update in updates:
                        self.last_update_id = update["update_id"]
                        self._process_update(update)
                elif response.status_code == 401:
                    logger.error("Telegram: Token inválido.")
                    self.is_running = False
                    break
            except Exception as e:
                logger.error(f"Erro no loop de polling do Telegram: {e}")
                time.sleep(5)
            
            time.sleep(1)

    def _process_update(self, update: Dict[str, Any]):
        """Valida e processa uma atualização recebida"""
        message = update.get("message")
        if not message:
            return

        chat_id = str(message.get("chat", {}).get("id"))
        text = message.get("text", "")

        # Segurança: Apenas chat_id autorizado
        if chat_id != self.authorized_chat_id:
            logger.warning(f"Recebido comando de chat não autorizado: {chat_id}")
            return

        if text.startswith("/"):
            self._handle_command(text.lower())

    def _handle_command(self, command_text: str):
        """Distribui os comandos para as funções do Bot"""
        parts = command_text.split()
        if not parts:
            return
            
        cmd = parts[0]
        logger.info(f"Comando Telegram recebido: {cmd}")

        if cmd == "/status":
            self._cmd_status()
        elif cmd == "/profit":
            self._cmd_profit()
        elif cmd == "/pause":
            self._cmd_pause()
        elif cmd == "/start":
            self._cmd_start()
        elif cmd in ["/close_all", "/closeall"]:
            self._cmd_close_all()
        elif cmd in ["/learner_run", "/learnerrun"]:
            self._cmd_learner_run()
        elif cmd == "/report":
            self._cmd_report()
        elif cmd == "/help":
            self._cmd_help()
        else:
            send_telegram_message(f"❓ Comando desconhecido: {cmd}\nUse /help para ver a lista.")

    def _cmd_status(self):
        """Retorna o status atual do robô e posições"""
        status_msg = "📊 *Status do XP3 Pro*\n"
        status_msg += f"Status: {'🟢 Operando' if self.bot.is_trading_active else '⏸️ Pausado'}\n"
        status_msg += f"Ativos Monitorados: {len(self.bot.symbols)}\n"
        status_msg += f"Posições Abertas: {len(self.bot.positions)}\n"
        
        # Último aprendizado
        params_path = settings.DATA_DIR / "session_optimized_params.json"
        if params_path.exists():
            mtime = params_path.stat().st_mtime
            update_dt = datetime.fromtimestamp(mtime).strftime('%d/%m %H:%M')
            status_msg += f"🧠 Último Aprendizado: {update_dt}\n"
        
        if self.bot.positions:
            status_msg += "\n*Posições Ativas:*\n"
            for s, p in self.bot.positions.items():
                emoji = "🔵" if p.order_type == "BUY" else "🔴"
                status_msg += f"{emoji} {s}: {p.order_type} | Vol: {p.volume:.2f} | P&L: `${p.profit:.2f}`\n"
        
        send_telegram_message(status_msg)

    def _cmd_profit(self):
        """Retorna o lucro do dia"""
        stats = self.bot.strategy.daily_stats
        msg = "💰 *Relatório de Ganhos* (Hoje)\n"
        msg += f"📅 Data: {stats['date']}\n"
        msg += f"✅ Wins: {stats['wins']}\n"
        msg += f"❌ Losses: {stats['losses']}\n"
        msg += f"💵 Lucro Total: `${stats['profit']:.2f}`\n"
        
        # P&L das abertas
        open_pl = sum(p.profit for p in self.bot.positions.values())
        msg += f"⏳ P&L Flutuante: `${open_pl:.2f}`"
        
        send_telegram_message(msg)

    def _cmd_pause(self):
        self.bot.pause_trading()
        send_telegram_message("⏸️ *Trading Pausado*. Novas ordens não serão abertas.")

    def _cmd_start(self):
        self.bot.resume_trading()
        send_telegram_message("▶️ *Trading Retomado*. Escaneando mercado...")

    def _cmd_close_all(self):
        count = len(self.bot.positions)
        if count == 0:
            send_telegram_message("ℹ️ Nenhuma posição aberta para fechar.")
            return
        
        send_telegram_message(f"🛑 Fechando {count} posições IMEDIATAMENTE...")
        self.bot.close_all_positions()
        send_telegram_message("✅ Todas as posições foram encerradas.")

    def _cmd_learner_run(self):
        send_telegram_message("🧠 Iniciando ciclo de aprendizado manual... Isso pode levar alguns minutos.")
        # Executa em thread para não travar o polling
        threading.Thread(target=self.bot.run_daily_maintenance, daemon=True).start()

    def _cmd_report(self):
        """Envia o relatório mais recente"""
        reports_dir = Path("data/reports")
        if not reports_dir.exists():
            send_telegram_message("❌ Pasta de relatórios não encontrada.")
            return

        files = sorted(reports_dir.glob("learning_report_*.md"), key=os.path.getmtime, reverse=True)
        if files:
            send_telegram_message(f"📄 Enviando relatório mais recente: `{files[0].name}`")
            send_telegram_document(files[0])
        else:
            send_telegram_message("❌ Nenhum relatório encontrado.")

    def _cmd_help(self):
        help_text = "📖 *Comandos Disponíveis:*\n\n"
        help_text += "/status - Status geral e posições\n"
        help_text += "/profit - Lucro do dia\n"
        help_text += "/pause - Pausa novas entradas\n"
        help_text += "/start - Retoma operações\n"
        help_text += "/closeall - Fecha tudo agora\n"
        help_text += "/learnerrun - Rodar aprendizado agora\n"
        help_text += "/report - Receber último relatório\n"
        help_text += "/help - Esta lista"
        send_telegram_message(help_text)

if __name__ == "__main__":
    # Teste rápido
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        send_telegram_message(sys.argv[1])
    else:
        print("Uso: python telegram_utils.py 'Minha mensagem'")
