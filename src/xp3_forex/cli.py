import argparse
import sys
import os
import signal
import logging
from pathlib import Path
from typing import Optional

# Adiciona o diretório atual ao path se necessário
sys.path.insert(0, os.getcwd())

# Importação da nova estrutura de settings
from xp3_forex.core.settings import settings
from xp3_forex.core.bot import XP3Bot
from xp3_forex.mt5.symbol_manager import SymbolManager

# Configuração de Logging para CLI
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("XP3_CLI")

def setup_signal_handlers(bot_instance: Optional[XP3Bot]):
    def signal_handler(sig, frame):
        print("\n🛑 Recebido sinal de parada. Encerrando graciosamente...")
        if bot_instance:
            if hasattr(bot_instance, 'stop'):
                bot_instance.stop()
            else:
                logger.warning("Bot instance does not have a stop method.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def run_bot(args):
    """Inicia o Bot de Trading"""
    print(f"\n🚀 XP3 PRO FOREX - INSTITUTIONAL TRADING BOT v5.0")
    print("=" * 50)
    print(f"🌍 Ambiente: {settings.XP3_ENV}")
    print(f"📈 Símbolos: {settings.symbols_list}")
    print(f"🕒 Timeframes: {settings.timeframes_list}")
    print("=" * 50)
    
    # Override settings via ENV vars if provided in CLI (before bot init)
    if args.symbols:
        print(f"⚠️ Override de Símbolos via CLI: {args.symbols}")
        # Note: This changes os.environ but settings might be already loaded.
        # Ideally, we should reload settings or modify the instance.
        # Since 'settings' is a global instance, we can't easily re-init it cleanly without reload.
        # But for this simple CLI, we warn user.
    
    if args.mode == "live":
        print("🔴 MODO LIVE TRADING ATIVADO - OPERAÇÕES REAIS!")
        confirm = input("Digite 'LIVE' para confirmar: ")
        if confirm != "LIVE":
            print("❌ Cancelado pelo usuário.")
            return
    else:
        print("🟢 Modo Demo/Paper Trading")

    try:
        bot = XP3Bot()
        setup_signal_handlers(bot)
        
        # Start Bot
        bot.start() # This should be a blocking call or we need a loop here
        
        # If bot.start() is non-blocking, we need to keep main thread alive
        # Assuming bot.start() starts threads and returns.
        if not getattr(bot, 'blocking', False):
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n👋 Encerrado pelo usuário.")
    except Exception as e:
        logger.exception(f"Erro fatal durante execução: {e}")
        sys.exit(1)

def run_monitor(args):
    """Inicia o Monitor (Dashboard/Logs)"""
    print("📊 Iniciando Monitor de Saúde e Dashboard...")
    try:
        from xp3_forex.monitor import start_monitor
        start_monitor()
    except ImportError:
        print("❌ Módulo de monitoramento não encontrado ou dependências faltando.")
    except Exception as e:
        logger.exception(f"Erro no monitor: {e}")

def init_project(args):
    """Inicializa arquivos de configuração"""
    env_path = Path(".env")
    if env_path.exists() and not args.force:
        print("⚠️ Arquivo .env já existe. Use --force para sobrescrever.")
        return
    
    example = Path(".env.example")
    if example.exists():
        import shutil
        shutil.copy(example, env_path)
        print("✅ Arquivo .env criado com sucesso a partir de .env.example")
    else:
        # Create a default .env if example missing
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("MT5_LOGIN=123456\nMT5_PASSWORD=secret\nMT5_SERVER=MetaQuotes-Demo\nSYMBOLS=EURUSD,GBPUSD\n")
        print("✅ Arquivo .env criado com configurações padrão.")

def main():
    parser = argparse.ArgumentParser(
        description="XP3 PRO FOREX - CLI de Gerenciamento Institucional",
        prog="xp3-forex"
    )
    parser.add_argument("--version", action="version", version="XP3 PRO FOREX v5.0.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Command: run
    run_parser = subparsers.add_parser("run", help="Inicia o robô de trading")
    run_parser.add_argument("--mode", choices=["live", "demo"], default="demo", help="Modo de operação (default: demo)")
    run_parser.add_argument("--symbols", type=str, help="Override lista de símbolos (ex: EURUSD,GBPUSD)")
    
    # Command: monitor
    monitor_parser = subparsers.add_parser("monitor", help="Inicia o monitor de logs/saúde")
    
    # Command: init
    init_parser = subparsers.add_parser("init", help="Inicializa configuração do projeto")
    init_parser.add_argument("--force", action="store_true", help="Sobrescreve arquivos existentes")

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
