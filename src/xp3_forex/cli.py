
import argparse
import sys
import os
import signal
import time
from pathlib import Path
from typing import Optional

# Adiciona o diretório atual ao path se necessário
sys.path.insert(0, os.getcwd())

from xp3_forex.config.settings import settings
from xp3_forex.core.bot import XP3Bot
from xp3_forex.mt5.symbol_manager import symbol_manager
import logging

logger = logging.getLogger("XP3_CLI")

def setup_signal_handlers(bot_instance):
    def signal_handler(sig, frame):
        print("\n🛑 Recebido sinal de parada. Encerrando graciosamente...")
        if bot_instance:
            bot_instance.stop() # Assumindo que existe um método stop
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def run_bot(args):
    """Inicia o Bot"""
    print(f"🚀 Iniciando XP3 PRO FOREX v5.0")
    print(f"🌍 Ambiente: {os.getenv('XP3_ENV', 'Production')}")
    print(f"📈 Símbolos: {settings.symbols_list}")
    
    # Override settings from CLI args if provided (basic ones)
    if args.symbols:
        os.environ["SYMBOLS"] = args.symbols
        # Re-instantiate settings to pick up env var changes or modify directly?
        # Pydantic settings are immutable by default usually, but we can try hack or just warn.
        # Better: Since settings is already imported, changing env var now might be too late if it's already instantiated.
        # But we can modify the singleton if needed, though not recommended.
        print("⚠️ Nota: Argumentos de CLI para símbolos podem não sobrescrever .env se já carregado. Use .env preferencialmente.")

    try:
        bot = XP3Bot()
        setup_signal_handlers(bot)
        
        if args.mode == "live":
            print("🔴 MODO LIVE TRADING ATIVADO - Cuidado!")
        else:
            print("🟢 Modo Demo/Paper Trading")

        # Start Bot
        bot.start() 
        
    except KeyboardInterrupt:
        print("\n👋 Encerrado pelo usuário.")
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")
        sys.exit(1)

def run_monitor(args):
    """Inicia o Monitor (Dashboard/Logs)"""
    print("📊 Iniciando Monitor...")
    from xp3_forex.monitor import start_monitor
    start_monitor()

def init_project(args):
    """Cria arquivos iniciais (.env)"""
    env_path = Path(".env")
    if env_path.exists():
        print("⚠️ Arquivo .env já existe.")
        return
    
    example = Path(".env.example")
    if example.exists():
        import shutil
        shutil.copy(example, env_path)
        print("✅ Arquivo .env criado a partir de .env.example")
    else:
        print("❌ .env.example não encontrado.")

def main():
    parser = argparse.ArgumentParser(
        description="XP3 PRO FOREX - CLI de Gerenciamento",
        prog="xp3-forex"
    )
    parser.add_argument("--version", action="version", version="XP3 PRO FOREX v5.0.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Command: run
    run_parser = subparsers.add_parser("run", help="Inicia o robô de trading")
    run_parser.add_argument("--mode", choices=["live", "demo"], default="demo", help="Modo de operação")
    run_parser.add_argument("--symbols", type=str, help="Override lista de símbolos (ex: EURUSD,GBPUSD)")
    run_parser.add_argument("--account", type=int, help="Override conta MT5")
    
    # Command: monitor
    monitor_parser = subparsers.add_parser("monitor", help="Inicia o monitor de logs/saúde")
    
    # Command: init
    init_parser = subparsers.add_parser("init", help="Inicializa configuração do projeto")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_bot(args)
    elif args.command == "monitor":
        run_monitor(args)
    elif args.command == "init":
        init_project(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
