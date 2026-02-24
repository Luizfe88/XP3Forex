#!/usr/bin/env python3
"""
XP3 PRO FOREX - Main execution script

Usage:
    python src/run_bot.py [config_path]

Arguments:
    config_path: Path to configuration file (default: config/config.json)
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xp3_forex.core.bot import XP3Bot

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="XP3 PRO FOREX Trading Bot")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"XP3 PRO FOREX v4.2.0"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print(f"   Please create it or use the template: config/config_template.json")
        return 1
    
    print("🚀 XP3 PRO FOREX BOT v4.2.0")
    print("=" * 50)
    
    try:
        # Create bot instance
        bot = XP3Bot(config_path=args.config)
        
        # Start the bot
        bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())