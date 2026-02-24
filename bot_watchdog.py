#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WATCHDOG - Monitor Externo do XP3 Forex Bot
Reinicia o bot se ele morrer ou travar
✅ VERSÃO COM PAUSE NO FINAL PARA VER ERROS
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# ✅ FIX WINDOWS ENCODING
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | WATCHDOG | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("watchdog.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("watchdog")

# Configurações
HEARTBEAT_FILE = "bot_heartbeat.timestamp"
BOT_SCRIPT = "bot_forex.py"
CHECK_INTERVAL = 45
MAX_HEARTBEAT_AGE = 300

bot_process = None

def is_bot_alive() -> bool:
    """Verifica se o bot está vivo via heartbeat"""
    heartbeat_path = Path(HEARTBEAT_FILE)
    
    if not heartbeat_path.exists():
        logger.warning("⚠️ Heartbeat file não existe")
        return False
    
    last_heartbeat = heartbeat_path.stat().st_mtime
    age = time.time() - last_heartbeat
    
    if age > MAX_HEARTBEAT_AGE:
        logger.error(f"❌ Bot travado! Último heartbeat: {age:.0f}s atrás")
        return False
    
    return True

def start_bot():
    """Inicia o bot em nova janela (Windows)"""
    global bot_process
    
    logger.info("🚀 Iniciando bot em nova janela...")
    
    try:
        if sys.platform == "win32":
            bot_process = subprocess.Popen(
                [sys.executable, BOT_SCRIPT],
                creationflags=subprocess.CREATE_NEW_CONSOLE  # Nova janela com painel visível!
            )
            logger.info(f"✅ Bot iniciado em nova janela (PID: {bot_process.pid})")
            logger.info("   ➜ O painel colorido do XP3 PRO aparece na janela nova!")
        else:
            bot_process = subprocess.Popen([sys.executable, BOT_SCRIPT])
            logger.info(f"✅ Bot iniciado (PID: {bot_process.pid})")
        
        return True
    except FileNotFoundError:
        logger.critical(f"❌ Arquivo não encontrado: {BOT_SCRIPT}")
        logger.critical("   Verifique se o bot_forex.py está na mesma pasta!")
        return False
    except Exception as e:
        logger.critical(f"❌ Erro ao iniciar bot: {e}")
        return False

def stop_bot():
    """Para o bot graciosamente"""
    global bot_process
    
    if bot_process and bot_process.poll() is None:
        logger.info("🛑 Parando bot...")
        bot_process.terminate()
        time.sleep(5)
        
        if bot_process.poll() is None:
            bot_process.kill()
            logger.warning("⚠️ Bot forçado a parar")
    
    bot_process = None

def main():
    logger.info("=" * 70)
    logger.info("🐕 WATCHDOG INICIADO - Monitorando XP3 Forex Bot")
    logger.info("=" * 70)
    logger.info(f"   Bot: {BOT_SCRIPT}")
    logger.info(f"   Intervalo: {CHECK_INTERVAL}s | Timeout heartbeat: {MAX_HEARTBEAT_AGE}s")
    logger.info("=" * 70)
    
    consecutive_failures = 0
    MAX_FAILURES = 5

    if not start_bot():
        input("\n❌ Pressione ENTER para fechar...")  # Pause se falhar logo de cara
        return

    # ✅ REQUISITO: Registro do início para Grace Period de 60s
    start_time = time.time()

    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            
            elapsed_since_start = time.time() - start_time
            
            if bot_process and bot_process.poll() is not None:
                returncode = bot_process.returncode
                logger.error(f"💀 Bot morreu inesperadamente! Código: {returncode}")
                consecutive_failures += 1
            elif elapsed_since_start < 60:
                # ✅ REQUISITO: No Heartbeat check nos primeiros 60s
                logger.info("⏳ Aguardando inicialização (Grace Period 60s)...")
                continue
            elif not is_bot_alive():
                logger.error(f"💔 Heartbeat parado! Falha {consecutive_failures + 1}/{MAX_FAILURES}")
                consecutive_failures += 1
            else:
                if consecutive_failures > 0:
                    logger.info("💚 Bot recuperado!")
                consecutive_failures = 0
                logger.debug("✓ Bot OK")
                continue  # Pula reinício

            # Só chega aqui se deu ruim
            stop_bot()
            time.sleep(5)
            
            if start_bot():
                logger.info("✅ Bot reiniciado com sucesso")
                consecutive_failures = 0
            else:
                logger.error(f"❌ Falha ao reiniciar ({consecutive_failures}/{MAX_FAILURES})")
            
            if consecutive_failures >= MAX_FAILURES:
                logger.critical("💀 Muitas falhas seguidas - Watchdog desistindo")
                break

    except KeyboardInterrupt:
        logger.info("\n⚠️ Watchdog interrompido pelo usuário (CTRL+C)")
    except Exception as e:
        logger.critical(f"💀 Erro crítico no watchdog: {e}", exc_info=True)
    finally:
        stop_bot()
        logger.info("🛑 Watchdog encerrado")

    # ✅ PAUSE FINAL PARA VOCÊ VER O QUE ACONTECEU
    print("\n" + "="*70)
    print("Watchdog finalizado.")
    print("Se houve erro, veja acima ou no arquivo 'watchdog.log'")
    print("="*70)
    input("Pressione ENTER para fechar esta janela...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Captura qualquer erro inesperado no início
        print(f"\n💀 ERRO FATAL AO INICIAR WATCHDOG:\n{e}")
        print("\nDetalhes completos no arquivo 'watchdog.log'")
        input("\nPressione ENTER para fechar...")
