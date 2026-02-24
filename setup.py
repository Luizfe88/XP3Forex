"""
XP3 PRO FOREX - Setup Script

Installation and configuration script for the trading bot.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "config",
        "backups",
        "reports",
        "screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_config_template():
    """Create configuration template"""
    config_template = {
        "mt5": {
            "login": 12345678,
            "password": "your_password_here",
            "server": "YourBroker-Demo",
            "path": "C:/Program Files/MetaTrader 5/terminal64.exe"
        },
        "trading": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"],
            "timeframes": [15, 60, 240],
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "max_daily_loss": 5.0,
            "max_weekly_loss": 10.0,
            "max_monthly_loss": 15.0
        },
        "telegram": {
            "enabled": False,
            "bot_token": "your_bot_token_here",
            "chat_id": "your_chat_id_here"
        },
        "logging": {
            "level": "INFO",
            "max_file_size_mb": 50,
            "backup_count": 3
        },
        "optimization": {
            "enabled": True,
            "auto_optimize": False,
            "optimization_frequency": "weekly"
        }
    }
    
    config_path = Path("config/config_template.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"✅ Created configuration template: {config_path}")

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import MetaTrader5
        import pandas
        import numpy
        import sklearn
        import numba
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def create_run_scripts():
    """Create run scripts for different platforms"""
    # Windows batch file
    batch_content = """@echo off
echo Starting XP3 PRO FOREX Bot...
cd /d "%~dp0"
python src/run_bot.py --config config/config.json
pause
"""
    
    with open("run_bot.bat", 'w', encoding='utf-8') as f:
        f.write(batch_content)
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "Starting XP3 PRO FOREX Bot..."
cd "$(dirname "$0")"
python3 src/run_bot.py --config config/config.json
"""
    
    with open("run_bot.sh", 'w', encoding='utf-8') as f:
        f.write(shell_content)
    
    # Make shell script executable on Unix
    if os.name != 'nt':
        os.chmod("run_bot.sh", 0o755)
    
    print("✅ Created run scripts")

def main():
    """Main setup function"""
    print("🚀 XP3 PRO FOREX - Setup Script")
    print("=" * 50)
    
    try:
        # Create directories
        create_directories()
        
        # Create config template
        create_config_template()
        
        # Check dependencies
        if not check_dependencies():
            print("\n⚠️  Please install missing dependencies before running the bot")
        
        # Create run scripts
        create_run_scripts()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy config/config_template.json to config/config.json")
        print("2. Edit config/config.json with your MT5 credentials")
        print("3. Run: python src/run_bot.py")
        print("4. Monitor: python src/monitor.py")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())