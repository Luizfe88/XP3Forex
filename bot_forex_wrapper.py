#!/usr/bin/env python3
"""
XP3 PRO FOREX - Bot Wrapper (Legacy Compatibility)

This wrapper provides backward compatibility for the old bot_forex.py interface
while using the new src-layout architecture.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from xp3_forex.core.bot import XP3Bot
    from xp3_forex.core.config import LOGS_DIR
except ImportError as e:
    print(f"❌ Erro ao importar módulos do novo sistema: {e}")
    print("Certifique-se de que o pacote xp3_forex está instalado.")
    sys.exit(1)

def main():
    """Main entry point for the legacy bot wrapper."""
    try:
        # Initialize the new bot
        bot = XP3Bot()
        
        print("🚀 XP3 PRO FOREX BOT v4.2 INSTITUCIONAL (Wrapper Mode)")
        print("✅ Usando nova arquitetura src-layout")
        print(f"📁 Logs: {LOGS_DIR}")
        print("=" * 60)
        
        # Start the bot
        bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot interrompido pelo usuário")
        return 0
    except Exception as e:
        print(f"❌ Erro fatal no bot: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())