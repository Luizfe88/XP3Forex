#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WATCHDOG - Monitor Externo do XP3 Forex Bot
Reinicia o bot se ele morrer ou travar.
Adaptado para a nova estrutura src/run_bot.py.
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | WATCHDOG | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("watchdog.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("watchdog")

# Caminho para o novo bot
BOT_SCRIPT = "src/run_bot.py"
# Usar o mesmo python que está rodando o watchdog
PYTHON_CMD = sys.executable

def start_bot_process():
    """Inicia o processo do bot e retorna o objeto Popen"""
    cmd = [PYTHON_CMD, BOT_SCRIPT]
    
    # Adiciona argumentos de linha de comando passados para o watchdog
    if len(sys.argv) > 1:
        # Repassa argumentos do watchdog para o bot (exceto o próprio nome do script)
        cmd.extend(sys.argv[1:])
    else:
        # Se nenhum argumento foi passado, tenta usar config padrão se existir
        config_path = Path("config/config.json")
        if config_path.exists():
            cmd.extend(["--config", str(config_path)])
            
    logger.info(f"🚀 Iniciando bot: {' '.join(cmd)}")
    
    # Inicia o processo
    process = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),  # Executa na raiz do projeto
        env=os.environ.copy()  # Mantém variáveis de ambiente
    )
    return process

def main():
    logger.info("🛡️ XP3 Forex Watchdog Iniciado")
    
    # Verifica se o script do bot existe
    if not Path(BOT_SCRIPT).exists():
        logger.error(f"❌ Arquivo do bot não encontrado: {BOT_SCRIPT}")
        logger.info("Certifique-se de estar na raiz do projeto e que src/run_bot.py existe.")
        sys.exit(1)
        
    while True:
        try:
            process = start_bot_process()
            logger.info(f"✅ Bot iniciado com PID: {process.pid}")
            
            # Aguarda o processo terminar
            return_code = process.wait()
            
            if return_code == 0:
                logger.info("🛑 Bot parou normalmente (exit code 0). Reiniciando em 5 segundos...")
                time.sleep(5)
            else:
                logger.error(f"❌ Bot caiu com erro (exit code {return_code}). Reiniciando em 10 segundos...")
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.warning("🛑 Watchdog interrompido pelo usuário.")
            if 'process' in locals() and process:
                process.terminate()
            break
        except Exception as e:
            logger.error(f"❌ Erro no Watchdog: {e}")
            time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
